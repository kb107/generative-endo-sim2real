"""Frozen C2 evaluation registry: build, freeze and verify (no model inference, no target pixels).

The registry is rebuilt from the files it names and compared with its stored
content, so any change to a head, run record, manifest, sidecar, pose binding,
matched-ID list, generated manifest or evaluation-relevant source file is
detected. Only a frozen registry whose recorded hash still matches its content
can drive the locked C2 campaign (``evaluation/c2.py``).

Commands: ``python -m generative_sim2real.evaluation.registry {build,freeze,verify}``.
"""
from __future__ import annotations
import argparse
import datetime
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re

EVALUATOR_VERSION = 'c2_locked_eval_v1'
METHOD_ORDER = ('Z', 'R', 'A', 'G', 'M')          # the original campaign's names (legacy 'declared absent' listing)
METHOD_NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_+.-]{0,39}$')
HEAD_CHOICES = ('best', 'last')
# Optional per-entry inference-time modifiers for DIAGNOSTIC arms of an already-registered head (never for selection).
MODIFIER_KEYS = ('process_hw', 'output_scale')
MODES = ('R', 'A', 'G', 'M')
SELECTION = 'source_validation_sequence_macro_raw_abs_rel'
FINAL_STEP_SELECTION = 'final_step_of_completed_run (no selection; source-validation AbsRel reported for reference)'
TARGET_GEOMETRY = 'target_rectified_full'
TARGET_HW = [504, 630]
# Only these fields may vary between the trained methods' configs (see depth/train.py LOCKED_EXPERIMENT). audit_run
# separately pins every run's model block to exactly the head-only or the bbLoRA variant, so those two keys may differ.
VARYING_CONFIG_FIELDS = (('data', 'mode'), ('data', 'generated_manifest'),
                         ('training', 'microbatch_size'), ('training', 'checkpoint_every'), ('training', 'seed'),
                         ('model', 'trainable'), ('model', 'backbone_lora'))
CODE_GLOBS = ('src/generative_sim2real/__init__.py', 'src/generative_sim2real/data/*.py',
              'src/generative_sim2real/depth/*.py', 'src/generative_sim2real/evaluation/*.py')
SPEC_KEYS = {'schema_version', 'name', 'output', 'methods', 'data'}
# Optional spec key: files (relative to src/generative_sim2real) that may differ from the hashes recorded in a run's
# resolved.json. Each affected run records the old/new hashes in the registry. Absent = any change is refused.
SPEC_OPTIONAL_KEYS = {'code_changes_since_training'}
SPEC_DATA_KEYS = {'source_train', 'source_validation', 'downstream_ids', 'target_test'}

def modifier_protocol() -> dict:
    """Disclosure block that is added to the protocol only when a spec uses diagnostic modifiers."""
    return {'scope': 'DIAGNOSTIC arms only: per-entry modifiers of an already-registered head; training records, '
                     'selection and the metric definitions are unchanged',
            'process_hw': 'the network runs at this size (multiples of 14) instead of the native grid; RGB is resized '
                          'bicubically by DA3MetricDepth.prepare_inputs, K is scaled per axis (fx*w/native_w, fy*h/native_h, '
                          'cx, cy consistently), the metric factor uses the scaled mean focal, and the prediction is '
                          'bilinearly resized (align_corners=False) onto the native GT grid before any metric',
            'output_scale': 'the metric prediction is multiplied by a fixed positive value (pinned with the sha256 of '
                            'its source file) before every metric, the oracle median scale and median_pred_m'}


def protocol_section(with_modifiers=False) -> dict:
    """Evaluation protocol pinned in the registry (target resolution is read at call time)."""
    protocol = _protocol_section()
    if with_modifiers:
        protocol['diagnostic_modifiers'] = modifier_protocol()
    return protocol


def _protocol_section() -> dict:
    h, w = TARGET_HW
    return {
        'target_geometry': {'transform_id': TARGET_GEOMETRY, 'hw': list(TARGET_HW),
                            'description': 'full rectified C3VDv2 C2 view, no crop, adapter-native resolution'},
        'model_input': {
            'process_hw': [h, w],
            'description': 'NATIVE processing: DA3MetricDepth.process_hw is set to the target grid (504x630, 630=45*14); '
                           'the frame is not resized and not squashed to a square; RGB float32 [0,1], ImageNet '
                           'mean/std, batch size 1, float32. Heads were trained at 504x504 (isotropic) source '
                           'crops; this is a disclosed resolution shift between training and C2 evaluation.',
            'intrinsics': 'rectified K unchanged (fx=fy=300 px, cx=314.5, cy=251.5); the wrapper rescales K by '
                          'process/native size, which is exactly 1.0 here'},
        'prediction': {'quantity': 'axial camera-z metric depth', 'units': 'meters',
                       'canonical_to_metric': 'canonical depth * mean(fx,fy)/300 with the unchanged focal '
                                              '(300/300 = 1.0 at the rectified geometry)',
                       'resize_to_ground_truth': 'none: the network output is already on the adapter-native GT grid; '
                                                 'a shape mismatch aborts the run (no anisotropic resize)'},
        'valid_mask': 'adapter depth_valid (raw_gt_zero_not_65535_and_remap_inside); never modified by predictions; '
                      'frames with zero valid GT pixels are recorded as no_valid_gt and excluded for every method',
        'metrics': {'abs_rel': 'mean(|p-g|/g) over valid pixels (dimensionless)',
                    'rmse_mm': '1000*sqrt(mean((p-g)^2)), p,g in meters',
                    'delta_1_1': 'fraction with max(p/g,g/p) < 1.1', 'delta_1_25': 'fraction with max(p/g,g/p) < 1.25',
                    'oracle_median_rmse_mm': 'ORACLE DIAGNOSTIC: RMSE after per-frame scalar GT_median/prediction_median; '
                                             'uses ground truth, never a headline and never used for selection'},
        'alignment': {'headline': 'none (raw metric prediction)',
                      'oracle_diagnostic': 'per-frame GT-median scalar, lower median in double precision'},
        'aggregation': {'headline': 'sequence_macro: per-frame metrics averaged within each C2 sequence, then equal '
                                    'weight per sequence',
                        'secondary': ['frame_macro (equal weight per frame)', 'pixel_global (pooled valid pixels)',
                                      'matched_subset (rows that are ok for every method; sequence-macro)']},
        'failures': 'nonfinite or nonpositive prediction at a valid pixel is recorded as a failed frame '
                    '(status=failed, metrics empty), excluded from aggregates, counted per sequence, and marks the '
                    'method incomplete; the ground-truth mask is never edited',
    }


DISCLOSURES = (
    'C2 has prior lab benchmark use; it is held out from training in this experiment but is not untouched.',
    'Single phantom family, single training seed per method; no variance estimate.',
    'Selection used only source-validation sequence-macro raw AbsRel; no target label, image or score was used.',
    'Image contents of C2 are not hashed; manifests, sidecars and pose files bound to the target_test sidecar are.',
)


class RegistryError(ValueError):
    pass


def sha256_file(path) -> str:
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def ids_digest(ids) -> str:
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def diff(stored, rebuilt, prefix='', limit=12):
    """Human-readable differences between two JSON-like values."""
    out = []
    if isinstance(stored, dict) and isinstance(rebuilt, dict):
        for key in sorted(set(stored) | set(rebuilt)):
            if key not in stored or key not in rebuilt:
                out.append(f'{prefix}/{key}: {"missing in registry" if key not in stored else "missing on disk"}')
            else:
                out.extend(diff(stored[key], rebuilt[key], f'{prefix}/{key}'))
    elif isinstance(stored, list) and isinstance(rebuilt, list) and len(stored) == len(rebuilt):
        for i, (a, b) in enumerate(zip(stored, rebuilt)):
            out.extend(diff(a, b, f'{prefix}/{i}'))
    elif stored != rebuilt:
        out.append(f'{prefix}: registry={str(stored)[:60]} current={str(rebuilt)[:60]}')
    return out[:limit]


# ---------------------------------------------------------------- code identity

def code_manifest(code_root) -> dict:
    """Hash over evaluation-relevant source (git has no commits, so this is the code revision)."""
    code_root = Path(code_root)
    files = sorted({p for pattern in CODE_GLOBS for p in code_root.glob(pattern)})
    if not files:
        raise RegistryError(f'No evaluation-relevant source files under {code_root}')
    hashes = {str(p.relative_to(code_root)): sha256_file(p) for p in files}
    return {'scope': list(CODE_GLOBS), 'files_sha256': hashes, 'sha256': digest(hashes),
            'note': 'git has no commits; revision = hash of evaluation-relevant source contents'}


def environment_info() -> dict:
    from importlib import metadata
    result = {'python': platform.python_version()}
    for name in ('torch', 'torchvision', 'numpy', 'opencv-python-headless', 'pillow', 'depth-anything-3', 'xformers'):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        result['torch_cuda'] = torch.version.cuda
    except ImportError:
        pass
    return result


RUNTIME_KEYS = ('python', 'torch', 'numpy', 'xformers', 'torch_cuda')
COMPARED_RUNTIME_KEYS = ('torch', 'numpy', 'xformers', 'torch_cuda')


def runtime_requirements() -> dict:
    info = environment_info()
    return {k: info.get(k) for k in RUNTIME_KEYS}


def check_runtime(registry, info=None) -> list:
    """Version mismatches between the registry's recorded build runtime and this process (empty = match)."""
    info = info or runtime_requirements()
    recorded = registry.get('runtime') or {}
    return [f'{k}: registry {recorded.get(k)!r} vs runtime {info.get(k)!r}'
            for k in COMPARED_RUNTIME_KEYS if recorded.get(k) != info.get(k)]


# ---------------------------------------------------------------- model identity

def official_digests(model) -> dict:
    """Content digests of the official head and frozen backbone of a loaded DA3MetricDepth."""
    from ..depth.checkpoints import head_digest
    backbone = {k: v for k, v in model.network.state_dict().items() if not k.startswith('head.')}
    return {'official_head_sha256': head_digest(model.head_state_dict()), 'backbone_sha256': head_digest(backbone)}


def official_digests_from_files() -> dict:
    """Recompute official head/backbone digests from the cached pinned safetensors file (no model is built)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from ..depth.checkpoints import head_digest
    from ..depth.model import CHECKPOINT, CHECKPOINT_REVISION
    try:
        path = hf_hub_download(CHECKPOINT, 'model.safetensors', revision=CHECKPOINT_REVISION, local_files_only=True)
    except Exception as error:
        raise RegistryError(f'Pinned official checkpoint is not in the local cache: {error}') from error
    state = {k[len('model.'):]: v for k, v in load_file(path).items() if k.startswith('model.')}
    head = {k[len('head.'):]: v for k, v in state.items() if k.startswith('head.')}
    backbone = {k: v for k, v in state.items() if not k.startswith('head.')}
    if not head or not backbone:
        raise RegistryError('Unexpected official checkpoint layout')
    return {'official_head_sha256': head_digest(head), 'backbone_sha256': head_digest(backbone),
            'weights_file_sha256': sha256_file(path)}


def base_section(digests) -> dict:
    from ..depth.model import CHECKPOINT, CHECKPOINT_REVISION, DA3_REVISION
    digests = digests or {}
    return {'da3_revision': DA3_REVISION, 'checkpoint': CHECKPOINT, 'checkpoint_revision': CHECKPOINT_REVISION,
            'process_hw': [504, 504], 'canonical_focal_px': 300.0,
            'official_head_sha256': digests.get('official_head_sha256'),
            'backbone_sha256': digests.get('backbone_sha256'),
            'weights_file_sha256': digests.get('weights_file_sha256')}


# ---------------------------------------------------------------- data

def describe_manifest(path, root, *, split) -> dict:
    path = Path(root) / path
    sidecar_path = path.with_suffix('.meta.json')
    side = read_json(sidecar_path)
    problems = []
    if side.get('split') != split:
        problems.append(f'{path.name}: sidecar split {side.get("split")!r}, expected {split!r}')
    if side.get('manifest_sha256') != sha256_file(path):
        problems.append(f'{path.name}: manifest bytes do not match sidecar manifest_sha256')
    if problems:
        raise RegistryError('; '.join(problems))
    return {'manifest_sha256': sha256_file(path), 'sidecar_sha256': sha256_file(sidecar_path),
            'sidecar_manifest_sha256': side['manifest_sha256'], 'count': side['count']}


def data_root_path() -> Path:
    value = os.environ.get('DATA_ROOT')
    if not value:
        raise RegistryError('DATA_ROOT must be set to re-hash the C2 pose/governing files')
    return Path(value).expanduser().absolute()


def describe_target(path, root) -> dict:
    out = describe_manifest(path, root, split='target_test')
    full = Path(root) / path
    side = read_json(full.with_suffix('.meta.json'))
    rows = [json.loads(line) for line in full.read_text().splitlines()]
    ids = [r['sample_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != side['count']:
        raise RegistryError('target_test sample IDs must be unique and match the sidecar count')
    if any(r.get('transform_id') != TARGET_GEOMETRY or r.get('split') != 'target_test' for r in rows):
        raise RegistryError('target_test rows must all use the full rectified target geometry')
    sequences = {}
    for r in rows:
        sequences[r['sequence_id']] = sequences.get(r['sequence_id'], 0) + 1
    if dict(sorted(sequences.items())) != side['per_sequence']:
        raise RegistryError('target_test per-sequence counts differ from the sidecar')
    poses = side['governing_files_sha256']
    pose_files = {k: v for k, v in poses.items() if k.endswith('pose.txt')}
    if not pose_files:
        raise RegistryError('target_test sidecar does not bind C2 pose files')
    data_root = data_root_path()
    stale = [k for k, v in poses.items() if not (data_root / k).is_file() or sha256_file(data_root / k) != v]
    if stale:
        raise RegistryError(f'{len(stale)} governing/pose file(s) on disk differ from the sidecar hashes, e.g. {stale[0]}')
    out.update(path=str(path), sidecar_path=str(Path(path).with_suffix('.meta.json')),
               per_sequence=dict(sorted(sequences.items())), sequence_count=len(sequences),
               ids_sha256=ids_digest(ids), sidecar_config_sha256=side['config_sha256'],
               sidecar_source_code_sha256=side['source_code_sha256'],
               governing_files_sha256=poses, pose_file_count=len(pose_files),
               governing_files_rehashed_on_disk=True, geometry=TARGET_GEOMETRY, hw=list(TARGET_HW))
    return out


def generated_provenance(records) -> dict:
    """The common GENERATED_PROVENANCE block (as validated by rgb_modes); per-ID fields are marked as varying."""
    from ..data.rgb_modes import GENERATED_PROVENANCE
    block = {}
    for key in GENERATED_PROVENANCE:
        if any(r.get(key) is None for r in records):
            raise RegistryError(f'generated manifest lacks provenance field {key!r}')
        values = {json.dumps(r[key], sort_keys=True) for r in records}
        block[key] = records[0][key] if len(values) == 1 else {'varies_per_record': True, 'distinct': len(values)}
    return block


def describe_data(spec, root, generated) -> dict:
    root = Path(root)
    data = spec['data']
    train = describe_manifest(data['source_train'], root, split='source_train')
    validation = describe_manifest(data['source_validation'], root, split='source_validation')
    ids_path = root / data['downstream_ids']
    ids = read_json(ids_path)
    if (ids.get('schema_version') != 1 or ids.get('purpose') != 'experiment' or ids.get('frozen') is not True
            or ids.get('source_manifest_sha256') != train['manifest_sha256']
            or ids.get('source_sidecar_sha256') != train['sidecar_sha256']):
        raise RegistryError('downstream_ids must be a frozen experiment list bound to the current source_train files')
    sample_ids = ids['sample_ids']
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise RegistryError('downstream_ids must be unique and nonempty')
    gen = {}
    for name, path in sorted(generated.items()):
        records = [json.loads(line) for line in (root / path).read_text().splitlines()]
        if {r['sample_id'] for r in records} != set(sample_ids) or len(records) != len(sample_ids):
            raise RegistryError(f'{name}: generated manifest must contain exactly one record per downstream ID')
        gen[name] = {'path': str(path), 'sha256': sha256_file(root / path), 'provenance': generated_provenance(records)}
    return {'source_train': {'path': data['source_train'], **train},
            'source_validation': {'path': data['source_validation'], **validation},
            'downstream_ids': {'path': data['downstream_ids'], 'sha256': sha256_file(ids_path),
                               'count': len(sample_ids), 'ids_sha256': ids_digest(sample_ids)},
            'generated': gen, 'target_test': describe_target(data['target_test'], root)}


# ---------------------------------------------------------------- run selection audit

def audit_run(run_dir, name, root, spec_data, sample_ids, official_head_sha256=None, head_choice='best') -> dict:
    """Prove that a trained head was selected by source validation only (``best``) or is the final step of a
    completed run (``last``), and fingerprint the run."""
    from ..depth.checkpoints import read_final_training, read_head
    root = Path(root)
    run = root / run_dir
    problems = []

    def fail(message):
        problems.append(f'{name}: {message}')

    resolved = read_json(run / 'resolved.json')
    status = read_json(run / 'status.json')
    config = resolved['config']
    state = status['state']
    evaluation = config.get('evaluation', {})
    if evaluation.get('target_views') != []:
        fail(f'resolved.json target_views is {evaluation.get("target_views")!r}, must be []')
    if evaluation.get('selection') != SELECTION:
        fail(f'selection is {evaluation.get("selection")!r}, must be {SELECTION}')
    if evaluation.get('validation_frames') is not None:
        fail('run used a validation frame subset')
    if config.get('purpose') != 'experiment' or resolved.get('smoke_only') is not False or status.get('smoke_only') is not False:
        fail('run is not a non-smoke experiment')
    if set(config.get('data', {})) != {'source_train', 'source_validation', 'matched_ids', 'mode', 'generated_manifest',
                                       'outputs_root', 'augmentation'}:
        fail('run config has unexpected dataset fields (target dataset interface?)')
    mode = config['data'].get('mode')
    if name in MODES and mode != name:
        fail(f'run mode {mode!r} does not match method name')      # the legacy names R/A/G/M pin the RGB mode
    elif mode not in MODES:
        fail(f'run mode {mode!r} is not one of {MODES}')
    for key, spec_key in (('source_train', 'source_train'), ('source_validation', 'source_validation'),
                          ('matched_ids', 'downstream_ids')):
        if config['data'].get(key) != spec_data[spec_key]:
            fail(f'run {key} path {config["data"].get(key)!r} differs from the registry spec')
    if status['completed_steps'] != status['target_steps'] or status['target_steps'] != config['training']['optimizer_steps']:
        fail('run did not complete its scheduled updates')
    if status['selected_step'] != state['best_step']:
        fail('status selected_step differs from the recorded best step')
    total = config['training']['optimizer_steps']
    lora_variant = config['model'].get('trainable') == 'head+backbone_lora'
    if head_choice == 'best':
        # best.pt must be exactly the source-validation-selected head recorded in the run state.
        head_path = run / 'best.pt'
        head = read_head(head_path, allow_backbone_lora=True)
        if head['head_sha256'] != state['best_head_sha256'] or head['selected_step'] != state['best_step']:
            fail('best.pt is not the head selected by source validation (digest or step mismatch)')
        if lora_variant and head.get('backbone_lora_sha256') != state.get('best_backbone_lora_sha256'):
            fail('best.pt backbone LoRA is not the one selected by source validation (digest mismatch)')
    else:
        # last.pt must be the training checkpoint written at the final step of the completed run.
        head_path = run / 'last.pt'
        head = read_final_training(head_path)
        if head['selected_step'] != total or status['completed_steps'] != total:
            fail(f'last.pt is at step {head["selected_step"]}, not the final step {total} of a completed run')
        if json.loads(json.dumps(head['state'])) != state:
            fail('last.pt training state differs from the final state recorded in status.json')
        if head['state']['cursor'] != total * config['training']['global_batch_size']:
            fail('last.pt sample cursor does not match the completed updates')
    if head['compatibility'] != {k: v for k, v in resolved.items()}:
        fail(f'{head_path.name} compatibility record differs from resolved.json')
    if head['compatibility']['smoke_only']:
        fail(f'{head_path.name} is a smoke head')
    if lora_variant != ('backbone_lora_state_dict' in head):
        fail(f'{head_path.name} backbone-LoRA state does not match the run configuration (model.trainable)')
    if lora_variant and head.get('backbone_lora_config') != config['model'].get('backbone_lora'):
        fail(f'{head_path.name} backbone-LoRA settings differ from the run configuration')
    if official_head_sha256 and state.get('initial_head_sha256') != official_head_sha256:
        fail('run did not start from the official pretrained head (initial_head_sha256 differs)')
    # Validation reports: every one is source-only; best step is the first strict argmin of sequence-macro raw AbsRel.
    reports = {}
    for directory in sorted((run / 'validation').glob('step_*')):
        summary = read_json(directory / 'summary.json')
        step = int(directory.name.split('_')[1])
        meta = summary['metadata']
        if meta.get('split') != 'source_validation' or meta.get('selection') != SELECTION or meta.get('smoke_only'):
            fail(f'validation report {directory.name} is not a source-validation selection report')
        if summary['frame_count'] != spec_data['_source_validation_count']:
            fail(f'validation report {directory.name} does not cover the whole source-validation manifest')
        reports[step] = (summary['sequence_macro']['abs_rel'], sha256_file(directory / 'summary.json'))
    total, every = config['training']['optimizer_steps'], config['training']['validation_every']
    expected_steps = {0, total} | {k for k in range(every, total + 1, every)}
    if set(reports) != expected_steps:
        fail(f'validation steps on disk {sorted(reports)} differ from the scheduled {sorted(expected_steps)}')
    best_step, best_value = None, math.inf
    for step in sorted(reports):
        if reports[step][0] < best_value:
            best_step, best_value = step, reports[step][0]
    if not reports or best_step != state['best_step'] or not math.isclose(best_value, state['best_abs_rel'], rel_tol=1e-12):
        fail('validation reports do not reproduce the recorded best step/AbsRel')
    # The training log may only contain matched source IDs: no target sample was ever consumed.
    log = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
    seen = {i for entry in log for i in entry['sample_ids']}
    if len(log) != status['completed_steps'] or not seen <= set(sample_ids) or any(i.startswith('c3vdv2/') for i in seen):
        fail('training log contains non-source or unmatched sample IDs')
    # Files bound at training time must still equal the files on disk and the registry data section.
    for path, expected in resolved['files_sha256'].items():
        if not (root / path).is_file() or sha256_file(root / path) != expected:
            fail(f'training input changed since training: {path}')
    # Training-time source code (data/depth/evaluation) must be unchanged, except for files the spec declares.
    now = lambda k: (sha256_file(root / 'src/generative_sim2real' / k)
                     if (root / 'src/generative_sim2real' / k).is_file() else None)
    changed = {k: {'at_training': v, 'now': now(k)} for k, v in resolved['code_sha256'].items() if now(k) != v}
    allowed = set(spec_data.get('_code_changes_since_training') or ())
    undeclared = sorted(set(changed) - allowed)
    if undeclared:
        fail(f'source files changed since training: {undeclared[:4]}')
    from ..depth import model as depth_model
    expected_model = {'da3_revision': depth_model.DA3_REVISION, 'checkpoint': depth_model.CHECKPOINT,
                      'checkpoint_revision': depth_model.CHECKPOINT_REVISION, 'process_hw': [504, 504],
                      'trainable': 'head', 'canonical_focal_px': 300.}
    expected_bblora = {**expected_model, 'trainable': 'head+backbone_lora',
                       'backbone_lora': depth_model.LOCKED_BACKBONE_LORA}
    if config['model'] != expected_model and json.dumps(config['model'], sort_keys=True) != json.dumps(
            expected_bblora, sort_keys=True):
        fail('run used a different base model protocol (neither head-only nor the locked head+backbone_lora)')
    if problems:
        raise RegistryError('; '.join(problems))
    record = {'name': name, 'kind': 'trained', 'run_dir': str(run_dir), 'mode': config['data']['mode'],
            'training_seed': config['training']['seed'], 'trainable': config['model']['trainable'],
            'head': {'path': str(Path(run_dir) / head_path.name), 'file_sha256': sha256_file(head_path),
                     'head_sha256': head['head_sha256']},
            'selected_step': state['best_step'] if head_choice == 'best' else total,
            'source_validation_sequence_macro_abs_rel': state['best_abs_rel'] if head_choice == 'best' else reports[total][0],
            'selection': SELECTION if head_choice == 'best' else FINAL_STEP_SELECTION, 'target_views': [],
            'resolved_config_sha256': digest(config),
            'run_files_sha256': {'resolved.json': sha256_file(run / 'resolved.json'),
                                 'status.json': sha256_file(run / 'status.json'),
                                 'train.jsonl': sha256_file(run / 'train.jsonl'),
                                 **{f'validation/step_{s:06d}/summary.json': h for s, (_, h) in sorted(reports.items())}},
            'training_files_sha256': resolved['files_sha256'],
            'training_code_sha256_digest': digest(resolved['code_sha256']),
            'training_environment': resolved['environment'],
            'generated_manifest': config['data']['generated_manifest'],
            '_comparison_config': _comparable_config(config)}
    # Optional blocks appear only when used, so registries of head-only/best-head specs keep their original shape.
    if head_choice != 'best':
        record['head_choice'] = head_choice
    if lora_variant:
        record['backbone_lora'] = {'config': config['model']['backbone_lora'], 'sha256': head['backbone_lora_sha256']}
    if changed:
        record['training_code_changes'] = dict(sorted(changed.items()))
    return record


def _output_scale_source_sha256(name, source, root, key=None, value=None) -> str:
    """sha256 of the pinned source file; with ``key`` (a '/'-separated path into the source JSON) the spec value must
    equal the number stored there, so the registered scale is the frozen file's number, not a hand-typed one."""
    path = Path(root) / source
    if not path.is_file():
        raise RegistryError(f'{name}: output_scale source {source} is not a file under the repository root')
    if key is not None:
        node = json.loads(path.read_text())
        try:
            for part in key.split('/'):
                node = node[int(part)] if isinstance(node, list) else node[part]
        except (KeyError, IndexError, ValueError, TypeError):
            raise RegistryError(f'{name}: output_scale key {key!r} not found in {source}') from None
        if isinstance(node, bool) or not isinstance(node, (int, float)) or float(node) != float(value):
            raise RegistryError(f'{name}: output_scale value {value!r} != {node!r} stored at {key!r} in {source}')
    return sha256_file(path)


def _comparable_config(config):
    config = json.loads(json.dumps(config))
    for group, key in VARYING_CONFIG_FIELDS:
        config[group].pop(key, None)
    return config


# ---------------------------------------------------------------- build / freeze / verify

def load_spec(path) -> dict:
    import yaml
    spec = yaml.safe_load(Path(path).read_text())
    return check_spec(spec)


def _process_hw(name, value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int or v < 14 or v % 14 for v in value)):
        raise RegistryError(f'{name}: process_hw must be [h, w], positive integer multiples of 14')
    return [int(v) for v in value]


def _output_scale(name, value):
    if (not isinstance(value, dict) or set(value) not in ({'value', 'source'}, {'value', 'source', 'key'})
            or isinstance(value['value'], bool)
            or not isinstance(value['value'], (int, float)) or not math.isfinite(value['value']) or value['value'] <= 0):
        raise RegistryError(f'{name}: output_scale must be {{value: <finite float > 0>, source: <repo-relative path>}}')
    source = value['source']
    if not isinstance(source, str) or not source or source.startswith('/') or '..' in Path(source).parts:
        raise RegistryError(f'{name}: output_scale source must be a repo-relative path without ..')
    if 'key' in value:
        if not isinstance(value['key'], str) or not value['key']:
            raise RegistryError(f'{name}: output_scale key must be a non-empty string')
        return {'value': float(value['value']), 'source': source, 'key': value['key']}
    return {'value': float(value['value']), 'source': source}


def method_modifiers(name, value) -> dict:
    """Validated diagnostic modifiers of one spec entry: ``{'process_hw': [h, w], 'output_scale': {value, source}}``
    (only the keys that are present; empty for entries without modifiers)."""
    out = {}
    if isinstance(value, dict):
        if 'process_hw' in value:
            out['process_hw'] = _process_hw(name, value['process_hw'])
        if 'output_scale' in value:
            out['output_scale'] = _output_scale(name, value['output_scale'])
    return out


def parse_method(name, value):
    """(run_dir, head_choice) of one spec method entry; (None, None) for the official pretrained head."""
    if not isinstance(name, str) or not METHOD_NAME.match(name):
        raise RegistryError(f'Method name {name!r} must match {METHOD_NAME.pattern}')
    if value is None:
        if name != 'Z':
            raise RegistryError(f'{name}: only the official pretrained head Z may have a null entry; trained methods need a run directory')
        return None, None
    if name == 'Z':
        raise RegistryError('Z is reserved for the official pretrained head and must be null')
    if isinstance(value, str) and value:
        return value, 'best'
    if isinstance(value, dict) and set(value) <= {'run_dir', 'head', *MODIFIER_KEYS} and 'run_dir' in value:
        run_dir, head = value['run_dir'], value.get('head', 'best')
        if isinstance(run_dir, str) and run_dir and head in HEAD_CHOICES:
            method_modifiers(name, value)
            return run_dir, head
    raise RegistryError(f'{name}: entry must be a run directory string or {{run_dir: ..., head: best|last, '
                        f'[process_hw: [h, w]], [output_scale: {{value, source}}]}}')


def check_spec(spec) -> dict:
    keys = set(spec) if isinstance(spec, dict) else set()
    if not isinstance(spec, dict) or not SPEC_KEYS <= keys <= SPEC_KEYS | SPEC_OPTIONAL_KEYS or spec['schema_version'] != 1:
        raise RegistryError(f'Registry spec must have exactly the keys {sorted(SPEC_KEYS)} (optional: '
                            f'{sorted(SPEC_OPTIONAL_KEYS)}) and schema_version 1')
    methods = spec['methods']
    if not isinstance(methods, dict) or 'Z' not in methods or methods['Z'] is not None:
        raise RegistryError('Spec methods must include Z (official pretrained) with no run directory')
    for name, value in methods.items():
        parse_method(name, value)
    if len(methods) < 2:
        raise RegistryError('Spec needs at least one trained method besides Z')
    if len({m.lower() for m in methods}) != len(methods):
        raise RegistryError('Method names must be unique ignoring case (they name result directories)')
    changes = spec.get('code_changes_since_training', [])
    if not isinstance(changes, list) or any(not isinstance(c, str) or c.startswith('/') or '..' in c for c in changes):
        raise RegistryError('code_changes_since_training must be a list of paths relative to src/generative_sim2real')
    scope = [g[len('src/generative_sim2real/'):] for g in CODE_GLOBS]
    outside = [c for c in changes if not any(fnmatch.fnmatchcase(c, g) and c.count('/') == g.count('/') for g in scope)]
    if outside:
        raise RegistryError(f'code_changes_since_training entries outside the hashed code scope {scope}: {outside[:4]}')
    if set(spec['data']) != SPEC_DATA_KEYS:
        raise RegistryError(f'Spec data needs exactly {sorted(SPEC_DATA_KEYS)}')
    return spec


def build_content(spec, root, code_root, digests, method_order=None) -> dict:
    """Everything the registry asserts, read from disk (never from the previous registry).

    The method order is the spec order. A stored registry file has its keys sorted, so ``verify_registry`` passes
    the stored ``method_names`` back in as ``method_order`` (it must be a permutation of the spec's methods).
    """
    root = Path(root)
    spec = check_spec(spec)
    methods = list(spec['methods'] if method_order is None else method_order)
    if sorted(methods) != sorted(spec['methods']):
        raise RegistryError('Registry method_names are not the spec methods')
    probe = {'_source_validation_count': describe_manifest(spec['data']['source_validation'], root,
                                                           split='source_validation')['count'], **spec['data']}
    missing = [c for c in spec.get('code_changes_since_training', [])
               if not (Path(code_root) / 'src/generative_sim2real' / c).is_file()]
    if missing:
        raise RegistryError(f'code_changes_since_training lists files that do not exist: {missing[:4]}')
    if spec.get('code_changes_since_training'):
        probe['_code_changes_since_training'] = list(spec['code_changes_since_training'])
    ids = read_json(root / spec['data']['downstream_ids'])['sample_ids']
    records, generated, baseline = [], {}, None
    with_modifiers = False
    for name in methods:
        if name == 'Z':
            records.append({'name': 'Z', 'kind': 'official', 'run_dir': None, 'head': None, 'selected_step': None,
                            'selection': 'none (official pretrained head, never trained here)', 'target_views': []})
            continue
        run_dir, head_choice = parse_method(name, spec['methods'][name])
        record = audit_run(run_dir, name, root, probe, ids, (digests or {}).get('official_head_sha256'), head_choice)
        comparable = record.pop('_comparison_config')
        baseline = baseline or comparable
        if comparable != baseline:
            raise RegistryError(f'{name}: training configuration differs from the other methods beyond mode/generated manifest/seed/model variant')
        modifiers = method_modifiers(name, spec['methods'][name])
        if 'process_hw' in modifiers:
            record['inference_process_hw'] = modifiers['process_hw']
        if 'output_scale' in modifiers:
            record['output_scale'] = {**modifiers['output_scale'],
                                      'sha256': _output_scale_source_sha256(
                                          name, modifiers['output_scale']['source'], root,
                                          modifiers['output_scale'].get('key'), modifiers['output_scale']['value'])}
        with_modifiers = with_modifiers or bool(modifiers)
        if (record['mode'] in ('G', 'M')) != (record['generated_manifest'] is not None):
            raise RegistryError(f'{name}: generated manifest is required for G/M only')
        if record['generated_manifest']:
            generated[name] = record['generated_manifest']
        records.append(record)
    legacy_names = set(methods) <= set(METHOD_ORDER)
    trained = [r for r in records if r['kind'] == 'trained']
    if legacy_names and len({(r['training_seed'], r['trainable']) for r in trained}) > 1:
        raise RegistryError('Methods named only Z/R/A/G/M must come from runs with one training seed and one model '
                            'variant; use descriptive method names (e.g. G_s1, G_bblora) for mixed seeds or variants')
    legacy = set(methods) <= set(METHOD_ORDER)         # 'declared absent' is only meaningful for the original names
    return {'schema_version': 1, 'kind': 'c2_evaluation_registry', 'name': spec['name'],
            'evaluator_version': EVALUATOR_VERSION, 'spec': spec,
            'method_names': methods, 'declared_absent_methods': [m for m in METHOD_ORDER if m not in methods] if legacy else [],
            'methods': records, 'base': base_section(digests), 'data': describe_data(spec, root, generated),
            'code': code_manifest(code_root), 'protocol': protocol_section(with_modifiers), 'disclosures': list(DISCLOSURES)}


def build_registry(spec, root, output, *, digests, code_root=None) -> dict:
    output = Path(output)
    if output.exists() and read_json(output).get('freeze', {}).get('frozen') is True:
        raise RegistryError(f'{output} is frozen; refusing to overwrite. A changed protocol is a new disclosed experiment.')
    registry = build_content(spec, root, code_root or root, digests)
    registry['build_environment'] = environment_info()
    registry['runtime'] = runtime_requirements()
    registry['freeze'] = {'frozen': False, 'note': 'DRAFT: not frozen; evaluation refuses until freeze_registry runs'}
    write_json(output, registry)
    return registry


def registry_hash(registry) -> str:
    return digest({k: v for k, v in registry.items() if k != 'freeze'})


def check_complete(registry):
    base = registry['base']
    if not all(base.get(k) for k in ('official_head_sha256', 'backbone_sha256', 'weights_file_sha256')):
        raise RegistryError('Official head/backbone digests missing; build with model digests before freezing')
    if 'Z' not in registry['method_names'] or len(registry['method_names']) < 2:
        raise RegistryError('A freezable registry needs Z and at least one trained method')


def verify_registry(path_or_registry, root, *, code_root=None, require_frozen=True, expected_sha256=None,
                    official_digests_fn=None) -> dict:
    """Raise RegistryError listing every problem; return the registry when all inputs still match."""
    if isinstance(path_or_registry, dict):
        registry = path_or_registry
    else:
        try:
            registry = read_json(path_or_registry)
        except (OSError, ValueError) as error:
            raise RegistryError(f'No readable frozen registry at {path_or_registry}: {error}') from error
    problems = []
    freeze = registry.get('freeze') or {}
    if registry.get('kind') != 'c2_evaluation_registry' or registry.get('evaluator_version') != EVALUATOR_VERSION:
        raise RegistryError('Not a c2_evaluation_registry for this evaluator version')
    if require_frozen and freeze.get('frozen') is not True:
        raise RegistryError('Registry is not frozen; freeze it before any C2 evaluation')
    if freeze.get('frozen') is True:
        recorded = freeze.get('registry_sha256')
        if recorded != registry_hash(registry):
            problems.append('registry content hash does not match its recorded freeze hash (file edited after freezing)')
        if expected_sha256 and recorded != expected_sha256:
            problems.append(f'registry hash {recorded} differs from the expected hash {expected_sha256}')
    elif expected_sha256:
        problems.append('an expected registry hash was supplied but the registry is not frozen')
    base = registry.get('base') or {}
    if base.get('official_head_sha256'):
        try:
            actual = (official_digests_fn or official_digests_from_files)()
        except Exception as error:
            problems.append(f'official checkpoint digests cannot be recomputed: {error}')
        else:
            problems.extend(diff({k: base.get(k) for k in actual}, actual, '/base(recomputed from weights file)'))
    try:
        rebuilt = build_content(registry['spec'], root, code_root or root, registry['base'], registry.get('method_names'))
    except Exception as error:      # any failure to re-derive an input is a refusal
        problems.append(f'inputs cannot be re-verified: {error}')
    else:
        keys = ('method_names', 'declared_absent_methods', 'methods', 'base', 'data', 'code', 'protocol', 'disclosures',
                'name', 'spec')
        problems.extend(diff({k: registry.get(k) for k in keys}, {k: rebuilt[k] for k in keys}))
    if problems:
        raise RegistryError('; '.join(problems))
    return registry


def freeze_registry(path, root, *, code_root=None, now=None, official_digests_fn=None) -> dict:
    registry = read_json(path)
    if registry.get('freeze', {}).get('frozen') is True:
        raise RegistryError('Registry is already frozen')
    check_complete(registry)
    verify_registry(registry, root, code_root=code_root, require_frozen=False, official_digests_fn=official_digests_fn)
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime('%Y-%m-%dT%H:%M:%SZ')
    registry['freeze'] = {'frozen': True, 'frozen_utc': stamp, 'registry_sha256': registry_hash(registry),
                          'hash_scope': 'sha256 of canonical JSON of the registry without its freeze block'}
    write_json(path, registry)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'freeze', 'verify'])
    parser.add_argument('--spec', default='configs/registry.yaml')
    parser.add_argument('--registry', help='registry JSON (default: the spec output path)')
    parser.add_argument('--no-model-digests', action='store_true',
                        help='skip recomputing official weight digests (drafts only; freeze refuses without them)')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    root = Path('.').resolve()
    spec = load_spec(args.spec)
    path = args.registry or spec['output']
    skip = (lambda: {}) if args.no_model_digests else None
    if args.action == 'build':
        registry = build_registry(spec, root, path, digests=None if args.no_model_digests else official_digests_from_files())
        print(json.dumps({'registry': path, 'frozen': False, 'methods': registry['method_names'],
                          'declared_absent': registry['declared_absent_methods']}, indent=2))
    elif args.action == 'freeze':
        registry = freeze_registry(path, root, official_digests_fn=skip)
        print(json.dumps({'registry': path, **registry['freeze']}, indent=2))
        print(f'REGISTRY SHA256 {registry["freeze"]["registry_sha256"]}  <- copy into docs/implementation-plan.md and EXPECTED_REGISTRY_SHA256')
    else:
        registry = verify_registry(path, root, require_frozen=False, expected_sha256=args.expected_sha256,
                                   official_digests_fn=skip)
        print(json.dumps({'registry': path, 'verified': True, 'frozen': registry['freeze'].get('frozen') is True,
                          'registry_sha256': registry_hash(registry),
                          'official_digests_recomputed_from_weights_file': not args.no_model_digests,
                          'runtime_mismatches_in_this_process': check_runtime(registry)}, indent=2))


if __name__ == '__main__':
    main()
