"""P5b bulk generation: one image per frozen downstream ID, atomic idempotent shards, and a
generated manifest that the DA3 G/M modes (data/rgb_modes.py) accept.

  run          render missing/invalid images (generator venv, GPU), shard by shard
  manifest     build generated_manifest.jsonl from the shard status files and prove that
               RGBModeDataset accepts it in G and M modes (any venv with the source data)
  verify       re-check every image (checksum + full generation fingerprint) without rendering
  fingerprint  print source/geometry fingerprints for the first N IDs (cross-environment check)

The render path is the project's own (p5_matrix.render_images -> model.generate); nothing
here retrains or filters. Images are retained whatever they look like or score; only technical
failures (CUDA OOM, non-finite output) are retried, and persistent ones are recorded, never
silently dropped. A recipe is one JSON file (schema in `load_recipe`); see
docs/generator-integration.md, "Bulk generation (P5b)".
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

from PIL import Image
import torch

from generative_sim2real.data import rgb_modes as rm

from . import lowrank_controller as lrc

SHARD_SIZE = 50
MAX_ATTEMPTS = 3
SIZE = 512
FINGERPRINT_VERSION = 2
PRODUCTION_BATCH = 1
TOP_KEYS = {'schema_version', 'purpose', 'name', 'label', 'notes', 'seed', 'generator_config', 'base',
            'prompt', 'precision', 'size', 'scheduler', 'steps', 'guidance', 'source_lora',
            'target_lora', 'controller', 'batch'}
REQUIRED_KEYS = TOP_KEYS - {'name', 'label', 'notes'}
SUB_KEYS = {'base': {'model_id', 'revision'}, 'source_lora': {'run_dir', 'step'},
            'target_lora': {'run_dir', 'step', 'scale'}, 'controller': {'run_dir', 'step', 'scale'}}
# Optional extras (low-rank controller, generation/lowrank_controller.py); absent = full controller.
OPTIONAL_SUB_KEYS = {'controller': {'kind', 'block_rank', 'before_rank', 'after_rank'}}
# Files whose bytes can change a pixel; hashed into the environment fingerprint (no git commits exist).
CODE_FILES = ('src/generative_sim2real/generation/bulk.py', 'src/generative_sim2real/generation/p5_matrix.py',
              'src/generative_sim2real/generation/model.py', 'src/generative_sim2real/generation/inputs.py',
              'src/generative_sim2real/generation/gate.py', 'src/generative_sim2real/generation/controlnet.py',
              'configs/generator.yaml', 'src/generative_sim2real/data/rgb_modes.py')
PACKAGES = ('torch', 'diffusers', 'peft', 'transformers', 'numpy')


class BulkError(RuntimeError):
    """A protocol violation or an incomplete result (never a retried technical failure)."""


class TechnicalFailure(RuntimeError):
    """Retryable: non-finite or mis-shaped output (CUDA OOM is recognised separately)."""


def is_technical(error):
    """Only CUDA out-of-memory and non-finite output are retried; everything else is a bug and propagates."""
    if isinstance(error, TechnicalFailure):
        return True
    oom = getattr(torch.cuda, 'OutOfMemoryError', ())
    if oom and isinstance(error, oom):
        return True
    return isinstance(error, RuntimeError) and 'out of memory' in str(error).lower()


# ---------------------------------------------------------------- recipe ----

def file_sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def checkpoint_file(spec, filename):
    return Path(spec['run_dir']) / 'checkpoints' / f'step_{int(spec["step"]):06d}' / filename


def repo_root():
    return Path(__file__).resolve().parents[3]


def environment_fingerprint():
    """Content hash of the pixel-affecting code plus library versions and GPU model."""
    from importlib import metadata

    root = repo_root()
    code = hashlib.sha256()
    for name in CODE_FILES:
        code.update(name.encode() + b'\0' + (root / name).read_bytes())
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {'code_sha256': code.hexdigest(), 'code_files': list(CODE_FILES), 'versions': versions,
            'gpu': gpu}


def cross_check_controller(recipe, files, root):
    """The controller run's own record must say it was trained with this source LoRA and base."""
    path = Path(root) / recipe['controller']['run_dir'] / 'resolved.json'
    if not path.is_file():
        raise BulkError(f'Controller run record missing: {path}')
    compat = json.loads(path.read_text()).get('compatibility', {})
    source = files['source_adapter']
    recorded = [v for k, v in compat.get('files_sha256', {}).items() if k == source['path']]
    if recorded != [source['sha256']]:
        raise BulkError('recipe source_lora does not match the source LoRA recorded by the controller run')
    if compat.get('revision') != recipe['base']['revision'] or compat.get('precision') != recipe['precision']:
        raise BulkError('recipe base revision/precision differ from the controller run record')


def load_recipe(path, root='.'):
    """Read, strictly validate and resolve the recipe. Unknown keys are rejected. Schema:

      schema_version: 1
      purpose: production | smoke          (smoke recipes only render into a path containing 'smoke')
      name, label, notes                   optional free text; not part of the hash
      seed: int                            per-ID seed = f(seed, sample_id)
      generator_config: configs/generator.yaml   (LoRA rank/alpha, controller depth, prompt-cache path)
      base: {model_id, revision}
      prompt, precision (bfloat16|float32), size (512)
      scheduler (DPMSolverMultistepScheduler), steps, guidance
      source_lora: {run_dir, step}         provenance only: the adapter the controller was trained under
      target_lora: {run_dir, step, scale}  ACTIVE at render; scale must be 1.0 (the only supported value)
      controller:  {run_dir, step, scale}  control scale = conditioning strength
                   optional low-rank controller: + {kind: lowrank, block_rank, before_rank, after_rank};
                   file controller_lowrank.safetensors, ranks hashed (absent/full: hash unchanged)
      batch: int                           images per forward pass; production recipes require 1

    The recipe hash covers seed, guidance, batch, base model/revision, prompt, precision, scheduler, steps,
    both scales, size and the sha256 of the three weight files. It does NOT cover the generator config
    contents, code or library versions; those are guarded by `environment_fingerprint`."""
    root = Path(root)
    recipe = json.loads(Path(path).read_text())
    if recipe.get('schema_version') != 1 or recipe.get('purpose') not in {'production', 'smoke'}:
        raise BulkError('Recipe needs schema_version 1 and purpose production|smoke')
    if set(recipe) - TOP_KEYS:
        raise BulkError(f'Unknown recipe keys: {sorted(set(recipe) - TOP_KEYS)}')
    for key in REQUIRED_KEYS:
        if recipe.get(key) is None:
            raise BulkError(f'Recipe is missing {key!r}')
    for key, allowed in SUB_KEYS.items():
        optional = OPTIONAL_SUB_KEYS.get(key, set())
        if (not isinstance(recipe[key], dict) or not allowed <= set(recipe[key])
                or set(recipe[key]) - allowed - optional):
            raise BulkError(f'Recipe {key!r} must have exactly the keys {sorted(allowed)}'
                            + (f' (plus optional {sorted(optional)})' if optional else ''))
    if recipe['size'] != SIZE:
        raise BulkError('Only 512-square generation is supported')
    if recipe['scheduler'] != 'DPMSolverMultistepScheduler':
        raise BulkError('Only DPMSolverMultistepScheduler is supported')
    if float(recipe['target_lora']['scale']) != 1.0:
        raise BulkError('target_lora.scale other than 1.0 is not implemented in the render path')
    if type(recipe['seed']) is not int or type(recipe['steps']) is not int or recipe['steps'] < 2:
        raise BulkError('seed must be an int and steps an int >= 2')
    if type(recipe['batch']) is not int or recipe['batch'] < 1:
        raise BulkError('batch must be a positive int')
    if recipe['purpose'] == 'production' and recipe['batch'] != PRODUCTION_BATCH:
        raise BulkError(f'Production recipes require batch: {PRODUCTION_BATCH}')
    try:
        controller_spec = lrc.recipe_spec(recipe['controller'])
    except ValueError as error:
        raise BulkError(f'Recipe controller: {error}') from None
    files = {'source_adapter': checkpoint_file(recipe['source_lora'], 'lora_source.safetensors'),
             'target_adapter': checkpoint_file(recipe['target_lora'], 'lora_target.safetensors'),
             'controller': checkpoint_file(recipe['controller'], lrc.file_name(controller_spec))}
    resolved_files = {}
    for key, rel in files.items():
        if not (root / rel).is_file():
            raise BulkError(f'Recipe file missing: {rel}')
        resolved_files[key] = {'path': str(rel), 'sha256': file_sha256(root / rel)}
    cross_check_controller(recipe, resolved_files, root)
    if controller_spec:
        try:
            lrc.check_file_matches_spec(root / files['controller'], controller_spec)
        except ValueError as error:
            raise BulkError(str(error)) from None
    label = {k: f'{v["path"]}#sha256={v["sha256"]}' for k, v in resolved_files.items()}
    provenance = {  # exactly rgb_modes.GENERATED_PROVENANCE minus the per-ID seed
        'base_model': recipe['base']['model_id'], 'base_revision': recipe['base']['revision'],
        'source_adapter': label['source_adapter'], 'target_adapter': label['target_adapter'],
        'adapter_weight': float(recipe['target_lora']['scale']),
        'controller': label['controller'], 'controller_scale': float(recipe['controller']['scale']),
        'prompt': recipe['prompt'], 'scheduler': recipe['scheduler'],
        'inference_steps': int(recipe['steps']), 'precision': recipe['precision'],
        'size_hw': [SIZE, SIZE]}
    if set(provenance) | {'seed'} != set(rm.GENERATED_PROVENANCE):
        raise BulkError('Provenance fields drifted from rgb_modes.GENERATED_PROVENANCE')
    defining = {'seed': recipe['seed'], 'guidance': float(recipe['guidance']), 'batch': recipe['batch'],
                'provenance': provenance, 'fingerprint_version': FINGERPRINT_VERSION}
    if controller_spec:  # hashed only for a non-full controller: full-controller hashes do not change
        defining['controller_spec'] = controller_spec
    return {'recipe': recipe, 'files': resolved_files, 'provenance': provenance, 'defining': defining,
            'guidance': float(recipe['guidance']), 'seed': recipe['seed'], 'batch': recipe['batch'],
            'recipe_hash': rm.digest(defining)[:16], 'purpose': recipe['purpose']}


def per_id_seed(recipe_seed, sample_id):
    return int(rm.digest({'recipe_seed': recipe_seed, 'sample_id': sample_id})[:15], 16)


# ----------------------------------------------------------- bookkeeping ----

def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.tmp-{os.getpid()}-{path.name}')
    with open(tmp, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_json(path, value):
    atomic_write(path, (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n').encode())


def atomic_png(path, tensor):
    """8-bit RGB PNG, same quantization as the P4/P5 gate renders; returns the file sha256."""
    array = (tensor.detach().cpu().float().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.tmp-{os.getpid()}-{path.name}')
    Image.fromarray(array).save(tmp, format='PNG')
    with open(tmp, 'rb') as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return file_sha256(path)


def clean_temporaries(directory):
    for tmp in Path(directory).rglob('.tmp-*'):
        tmp.unlink(missing_ok=True)


class Layout:
    """<out_root>/generated/<recipe_hash>/{recipe.json, shards/shard_NNNN/{status.json,NNNNN.png},
    generated_manifest.jsonl}. Manifest image paths are relative to out_root."""

    def __init__(self, out_root, recipe_hash):
        self.out_root = Path(out_root)
        self.directory = self.out_root / 'generated' / recipe_hash

    def shard_dir(self, shard):
        return self.directory / 'shards' / f'shard_{shard:04d}'

    def image(self, index):
        return self.shard_dir(index // SHARD_SIZE) / f'{index:05d}.png'

    def status(self, shard):
        return self.shard_dir(shard) / 'status.json'

    @property
    def manifest(self):
        return self.directory / 'generated_manifest.jsonl'


def source_lookup(source):
    return {row['sample_id']: i for i, row in enumerate(source.rows)}


def expected_fingerprints(resolved, source, lookup, sample_id):
    """Per-ID identity of everything that determines the image: recipe provenance, seed, source
    transform policy, exact 512-stage depth/mask/K bytes and the control encoding constants."""
    index = lookup[sample_id]
    row = source.rows[index]
    sample = source[index]
    seed = per_id_seed(resolved['seed'], sample_id)
    pre = rm.source_fingerprint(row, source.sidecar)
    geometry = rm.geometry_sha256(sample)
    fingerprint = rm.digest({
        'version': FINGERPRINT_VERSION, 'recipe_hash': resolved['recipe_hash'],
        'sample_id': sample_id, 'seed': seed, 'preprocessing_sha256': pre,
        'source_geometry_sha256': geometry, 'guidance': resolved['guidance'],
        'batch': resolved['batch'], 'provenance': resolved['provenance'],
        'control_encoding': source.sidecar['depth_control_encoding']['constants']})
    return {'sample': sample, 'seed': seed, 'preprocessing_sha256': pre,
            'source_geometry_sha256': geometry, 'fingerprint': fingerprint,
            'source_depth_path': row['depth_path']}


def load_status(layout, shard, recipe_hash):
    path = layout.status(shard)
    if path.is_file():
        try:
            status = json.loads(path.read_text())
            if status.get('recipe_hash') == recipe_hash and isinstance(status.get('images'), dict):
                return status
        except (ValueError, OSError):
            pass  # torn/foreign status: treated as empty; images are re-validated against nothing
    return {'schema_version': 1, 'recipe_hash': recipe_hash, 'shard': shard, 'images': {},
            'failures': {}}


def image_is_valid(layout, index, entry, expected):
    """Accept an existing image only if its recorded path is the canonical one for this index, the file
    checksum matches, it fully decodes as 512x512 RGB, and the full generation fingerprint matches."""
    if not entry or entry.get('fingerprint') != expected['fingerprint'] or entry.get('seed') != expected['seed']:
        return False
    if entry.get('file') != str(layout.image(index).relative_to(layout.out_root)):
        return False
    path = layout.out_root / entry['file']
    if not path.is_file() or file_sha256(path) != entry.get('sha256'):
        return False
    try:
        with Image.open(path) as image:
            image.load()  # full decode, not just the header
            return image.mode == 'RGB' and image.size == (SIZE, SIZE)
    except Exception:
        return False


@contextlib.contextmanager
def locked(layout):
    """Exclusive non-blocking lock on the recipe directory; released automatically if the process dies."""
    layout.directory.mkdir(parents=True, exist_ok=True)
    with open(layout.directory / '.lock', 'w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise BulkError(f'Another process holds the lock on {layout.directory}') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def has_progress(layout):
    return any((layout.directory / 'shards').glob('shard_*/status.json'))


def prepare_directory(resolved, layout, environment, override):
    """Create recipe.json once; afterwards never rewrite it. Refuse to resume under a different recipe,
    code or environment unless `override` is set (the override is then recorded in run_history.jsonl).
    Returns True when this run uses an overridden (changed) environment."""
    path = layout.directory / 'recipe.json'
    record = {'recipe_hash': resolved['recipe_hash'], 'purpose': resolved['purpose'],
              'defining': resolved['defining'], 'recipe': resolved['recipe'],
              'files': resolved['files'], 'provenance': resolved['provenance'],
              'environment': environment, 'created_utc': utc()}
    if not path.is_file():
        atomic_json(path, record)
        return False
    old = json.loads(path.read_text())
    if old.get('recipe_hash') != resolved['recipe_hash'] or old.get('defining') != resolved['defining']:
        raise BulkError('Existing recipe.json differs from this recipe in a hashed field')
    if old.get('environment') == environment:
        return False
    if not has_progress(layout):
        atomic_json(path, {**old, 'environment': environment})  # nothing generated yet: adopt it
        return False
    if not override:
        raise BulkError(
            'Refusing to resume a partially generated recipe directory under a different code/library/GPU '
            f'environment (recorded {old.get("environment")}, now {environment}). Pass '
            '--allow-environment-change to continue; the override is recorded in run_history.jsonl.')
    return True


def append_history(layout, entry):
    """Append-only run history; never rewritten."""
    layout.directory.mkdir(parents=True, exist_ok=True)
    with open(layout.directory / 'run_history.jsonl', 'a') as handle:
        handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


# ------------------------------------------------------------ generation ----

def render_items(render, items, max_attempts, log):
    """Render `items` as one batch; retry technical failures, then fall back to singles.
    Returns ({sample_id: (image, attempts)}, {sample_id: failure}). Non-technical errors propagate."""
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            images = render([i['sample'] for i in items], [i['seed'] for i in items])
            if images.shape != (len(items), 3, SIZE, SIZE) or not torch.isfinite(images).all():
                raise TechnicalFailure('Non-finite or mis-shaped generator output')
            return {i['id']: (images[k], attempt) for k, i in enumerate(items)}, {}
        except Exception as error:
            if not is_technical(error):
                raise
            last = error
            log(f'technical failure (attempt {attempt}/{max_attempts}): {str(error)[:200]}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if len(items) > 1:
        done, failed = {}, {}
        for item in items:
            more, less = render_items(render, [item], max_attempts, log)
            done.update(more)
            failed.update(less)
        return done, failed
    item = items[0]
    return {}, {item['id']: {'index': item['index'], 'attempts': max_attempts,
                             'error': f'{type(last).__name__}: {str(last)[:300]}', 'utc': utc()}}


def generate(resolved, source, ids, layout, render, *, max_attempts=MAX_ATTEMPTS, log=print,
             environment=None, allow_environment_change=False, argv=None):
    """Resumable generation of one image per ID. Safe to kill at any point and rerun.
    The batch size is the recipe's (hashed and fingerprinted); every run appends to run_history.jsonl."""
    lookup = source_lookup(source)
    if len(ids) != len(set(ids)) or any(i not in lookup for i in ids):
        raise BulkError('IDs must be unique source_train IDs')
    environment = environment or environment_fingerprint()
    batch = resolved['batch']
    with locked(layout):
        overridden = prepare_directory(resolved, layout, environment, allow_environment_change)
        clean_temporaries(layout.directory)
        summary = {'skipped': 0, 'generated': 0, 'regenerated': 0, 'failed': 0, 'batches': 0}
        failures = {}
        history = {'recipe_hash': resolved['recipe_hash'], 'purpose': resolved['purpose'],
                   'start_utc': utc(), 'host': socket.gethostname(), 'gpu': environment['gpu'],
                   'code_sha256': environment['code_sha256'], 'versions': environment['versions'],
                   'environment_override': overridden, 'batch': batch, 'n_ids': len(ids),
                   'argv': list(sys.argv if argv is None else argv), 'outcome': 'started'}
        started = time.time()
        render_seconds = 0.0
        try:
            consecutive_total_failures = 0
            for shard in sorted({i // SHARD_SIZE for i in range(len(ids))}):
                indices = list(range(shard * SHARD_SIZE, min(len(ids), (shard + 1) * SHARD_SIZE)))
                status = load_status(layout, shard, resolved['recipe_hash'])
                status['failures'] = {}  # failures are retried on every run
                todo = []
                for index in indices:
                    sid = ids[index]
                    expected = expected_fingerprints(resolved, source, lookup, sid)
                    entry = status['images'].get(sid)
                    if image_is_valid(layout, index, entry, expected):
                        summary['skipped'] += 1
                        continue
                    if entry is not None or layout.image(index).exists():
                        summary['regenerated'] += 1  # stale, corrupt or foreign: replaced below
                    status['images'].pop(sid, None)
                    todo.append({'id': sid, 'index': index, **expected})
                status['expected_ids'] = [ids[i] for i in indices]

                def save():
                    status['complete'] = len(status['images']) == len(indices) and not status['failures']
                    status['updated_utc'] = utc()
                    atomic_json(layout.status(shard), status)
                save()
                for start in range(0, len(todo), batch):
                    items = todo[start:start + batch]
                    tick = time.time()
                    done, failed = render_items(render, items, max_attempts, log)
                    render_seconds += time.time() - tick
                    for item in items:
                        if item['id'] in done:
                            image, attempts = done[item['id']]
                            path = layout.image(item['index'])
                            sha = atomic_png(path, image)
                            status['images'][item['id']] = {
                                'index': item['index'], 'file': str(path.relative_to(layout.out_root)),
                                'sha256': sha, 'fingerprint': item['fingerprint'], 'seed': item['seed'],
                                'attempts': attempts, 'batch_size': len(items), 'utc': utc()}
                            summary['generated'] += 1
                        else:
                            status['failures'][item['id']] = failed[item['id']]
                            failures[item['id']] = failed[item['id']]
                            summary['failed'] += 1
                    summary['batches'] += 1
                    save()  # image files first, then the status that vouches for them
                    # Every image of several batches in a row failing means a systematic fault.
                    consecutive_total_failures = consecutive_total_failures + 1 if not done else 0
                    if consecutive_total_failures >= 3:
                        raise BulkError('Three consecutive batches failed completely; aborting (status kept)')
                    log(f'shard {shard}: {len(status["images"])}/{len(indices)} images, '
                        f'{len(status["failures"])} failures')
                save()
            history['outcome'] = 'completed'
        except BaseException as error:
            history['outcome'] = f'{type(error).__name__}: {str(error)[:300]}'
            raise
        finally:
            summary.update(images=len(ids), wall_seconds=time.time() - started,
                           render_seconds=render_seconds, utc=utc(),
                           images_per_second_render=(summary['generated'] / render_seconds
                                                     if render_seconds else None))
            append_history(layout, {**history, 'end_utc': utc(), 'counts': dict(summary),
                                    'failures': failures,
                                    'images_per_second_render': summary['images_per_second_render']})
    return summary


# -------------------------------------------------------------- manifest ----

def collect_records(resolved, source, ids, layout):
    """Records for every ID, each re-validated (checksum + fingerprint); raises with a precise list."""
    lookup = source_lookup(source)
    records, problems = [], {}
    statuses = {}
    for index, sid in enumerate(ids):
        shard = index // SHARD_SIZE
        status = statuses.setdefault(shard, load_status(layout, shard, resolved['recipe_hash']))
        if sid in status.get('failures', {}):
            problems[sid] = 'recorded technical failure: ' + status['failures'][sid]['error']
            continue
        expected = expected_fingerprints(resolved, source, lookup, sid)
        entry = status['images'].get(sid)
        if entry is None:
            problems[sid] = 'not generated'
        elif not image_is_valid(layout, index, entry, expected):
            problems[sid] = 'image missing, corrupt, or fingerprint mismatch'
        else:
            records.append({
                'sample_id': sid, 'source_sample_id': sid,
                'source_depth_path': expected['source_depth_path'],
                'source_transform_id': 'source_generation512',
                'preprocessing_sha256': expected['preprocessing_sha256'],
                'source_geometry_sha256': expected['source_geometry_sha256'],
                'generated_rgb_path': entry['file'], 'generated_rgb_sha256': entry['sha256'],
                **resolved['provenance'], 'seed': expected['seed'],
                'recipe_hash': resolved['recipe_hash'], 'generation_fingerprint': entry['fingerprint'],
                'guidance_scale': resolved['guidance'], 'batch_size': resolved['batch']})
    if problems:
        shown = dict(list(problems.items())[:10])
        raise BulkError(f'{len(problems)} of {len(ids)} IDs have no valid generated image '
                        f'(excluded IDs need a documented technical reason): {shown}')
    return records


def validate_with_rgb_modes(source, ids, manifest, out_root, *, seed=0):
    """The consumers' own loader: G and M must accept the manifest, and every G item must pass
    the geometry/checksum guards inside get_occurrence."""
    report = {}
    for mode in ('G', 'M'):
        dataset = rm.RGBModeDataset(source, ids, mode=mode, seed=seed,
                                    generated_manifest=manifest, outputs_root=out_root)
        report[f'{mode}_len'] = len(dataset)
        if mode == 'G':
            for i in range(len(dataset)):
                sample = dataset[i]
                if sample['rgb'].shape != (3, 504, 504):
                    raise BulkError('G sample has unexpected shape')
            report['G_all_items_loaded'] = len(dataset)
    return report


def build_manifest(resolved, source, ids, layout, *, validate=True):
    with locked(layout):
        return _build_manifest(resolved, source, ids, layout, validate)


def _build_manifest(resolved, source, ids, layout, validate):
    records = collect_records(resolved, source, ids, layout)
    text = ''.join(json.dumps(r, allow_nan=False) + '\n' for r in records)
    atomic_write(layout.manifest, text.encode())
    result = {'manifest': str(layout.manifest), 'records': len(records),
              'manifest_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'recipe_hash': resolved['recipe_hash'], 'purpose': resolved['purpose']}
    if validate:
        result['rgb_modes_validation'] = validate_with_rgb_modes(
            source, ids, layout.manifest, layout.out_root)
    atomic_json(layout.directory / 'manifest_info.json', result)
    return result


def verify(resolved, source, ids, layout):
    """Re-check every image without rendering. Returns a count summary."""
    lookup = source_lookup(source)
    bad = []
    for index, sid in enumerate(ids):
        status = load_status(layout, index // SHARD_SIZE, resolved['recipe_hash'])
        if not image_is_valid(layout, index, status['images'].get(sid),
                              expected_fingerprints(resolved, source, lookup, sid)):
            bad.append(sid)
    return {'checked': len(ids), 'valid': len(ids) - len(bad), 'invalid': bad[:20],
            'invalid_count': len(bad)}


# --------------------------------------------------------------- real GPU ----

def effective_generator_config(resolved):
    from . import gate

    recipe = resolved['recipe']
    config = gate.load_config(recipe['generator_config'])
    config['model_id'], config['revision'] = recipe['base']['model_id'], recipe['base']['revision']
    config['prompt'], config['precision'] = recipe['prompt'], recipe['precision']
    config['resolution'] = recipe['size']
    config['inference'] = {'scheduler': recipe['scheduler'], 'steps': int(recipe['steps']),
                           'guidance_scale': resolved['guidance'], 'seed': recipe['seed']}
    config['controller']['conditioning_scale'] = float(recipe['controller']['scale'])
    gate.validate_config(config)
    return config


def make_renderer(ctx, source, resolved):
    """Closure (samples, seeds) -> [B,3,512,512]: target adapter ON, source camera-z depth ->
    inverse-depth control -> deterministic VAE mode latent -> p5_matrix.render_images."""
    from . import p5_matrix
    from .inputs import control_image, encode_image

    scale = float(resolved['recipe']['controller']['scale'])

    def render(samples, seeds):
        images = torch.stack([control_image(s, source.sidecar) for s in samples])
        latents = encode_image(ctx.vae, images, control=True)
        return p5_matrix.render_images(ctx, 'target', list(seeds), scale=scale,
                                       batch=len(seeds), control=latents)
    return render


def load_context(resolved):
    from . import gate
    from .p4_eval import load_weights

    config = effective_generator_config(resolved)
    ctx = gate.start(config, Path.cwd())
    root = Path('.')
    files = resolved['files']
    loaded = load_weights(ctx, None, config, target_lora=root / files['target_adapter']['path'],
                          controller=root / files['controller']['path'])
    for key, name in (('target_adapter', 'lora_target'), ('controller', 'controller')):
        if loaded[name]['sha256'] != files[key]['sha256']:
            raise BulkError(f'{key} weights changed since the recipe was resolved')
    return ctx, gate.snapshot_hashes(ctx.model, ctx.controller)


def open_source(path, data_root):
    from .inputs import generation_dataset

    source = generation_dataset(path, data_root, stage='generation')
    if source.split != 'source_train' or source.geometry != 'source_generation512':
        raise BulkError('Bulk generation requires source_train at source_generation512 geometry')
    return source


def select_ids(ids_path, source, limit, purpose):
    ids = rm.load_matched_ids(ids_path, source, smoke=False)  # frozen production list only
    if limit is not None:
        if purpose != 'smoke':
            raise BulkError('--limit is only allowed with a smoke recipe')
        ids = ids[:limit]
    return ids


def check_destination(purpose, out_root):
    parts = Path(out_root).resolve().parts
    if purpose == 'smoke' and not any('smoke' in p for p in parts):
        raise BulkError('Smoke recipes may only write below a path containing "smoke"')
    if purpose == 'production' and any('smoke' in p for p in parts):
        raise BulkError('Production recipes must not write below a smoke path')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('command', choices=['run', 'manifest', 'verify', 'fingerprint'])
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--ids', default='outputs/manifests/milestone_v1/downstream_ids.json')
    parser.add_argument('--source', default='outputs/manifests/milestone_v1/source_train.jsonl')
    parser.add_argument('--out-root', default='outputs')
    parser.add_argument('--limit', type=int, default=None, help='first N IDs; smoke recipes only')
    parser.add_argument('--batch', type=int, default=None, help='must equal the recipe batch if given')
    parser.add_argument('--allow-environment-change', action='store_true',
                        help='resume despite changed code/library/GPU; recorded in run_history.jsonl')
    parser.add_argument('--no-manifest', action='store_true', help='run: skip the manifest step')
    args = parser.parse_args()
    resolved = load_recipe(args.recipe)
    if args.batch is not None and args.batch != resolved['batch']:
        raise BulkError(f'--batch {args.batch} differs from the recipe batch {resolved["batch"]}')
    check_destination(resolved['purpose'], args.out_root)
    source = open_source(args.source, os.environ['DATA_ROOT'])
    ids = select_ids(args.ids, source, args.limit, resolved['purpose'])
    layout = Layout(args.out_root, resolved['recipe_hash'])
    if args.command == 'fingerprint':
        lookup = source_lookup(source)
        print(json.dumps({sid: {k: v for k, v in expected_fingerprints(
            resolved, source, lookup, sid).items() if k != 'sample'} for sid in ids}, indent=1))
        return
    if args.command == 'run':
        ctx, before = load_context(resolved)
        render = make_renderer(ctx, source, resolved)
        summary = generate(resolved, source, ids, layout, render,
                           allow_environment_change=args.allow_environment_change)
        from . import gate
        after = gate.snapshot_hashes(ctx.model, ctx.controller)
        summary.update(weights_unchanged=before == after, recipe_hash=resolved['recipe_hash'],
                       purpose=resolved['purpose'], batch=resolved['batch'], vram=gate.vram(ctx.device))
        atomic_json(layout.directory / 'run_summary.json', summary)
        print(json.dumps(summary, indent=1))
        if not summary['weights_unchanged']:
            raise BulkError('Model weights changed during rendering')
        if summary['failed'] or args.no_manifest:
            raise SystemExit(2 if summary['failed'] else 0)
    if args.command in ('run', 'manifest'):
        print(json.dumps(build_manifest(resolved, source, ids, layout), indent=1))
    if args.command == 'verify':
        result = verify(resolved, source, ids, layout)
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if not result['invalid_count'] else 1)


if __name__ == '__main__':
    main()
