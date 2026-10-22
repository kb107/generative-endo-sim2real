"""P4 trainers: source LoRA, target LoRA, and the source-only depth controller.

One resumable loop (training.train_updates) for every run in configs/p4.yaml:
  source_lora         source_train RGB only, source adapter trains
  target_lora         C1 appearance_train RGB only (no depth/pose is indexed), target adapter trains
  controller_overfit  8 source_train pairs, source LoRA + backbone frozen (differentiable)
  controller_pilot    source_train pairs, same freezing
Run: `bash scripts/train_p4.sh` (RUN=<name> GPU=<index>) or
`python -m generative_sim2real.generation.train --run NAME --config configs/p4.yaml`.

Each run directory (outputs/p4-appearance/runs/NAME) holds status.json (atomic; state, step,
ETA, last checkpoint, exit reason), train.jsonl (per update: loss, grad norm, lr, step time),
eval.jsonl (fixed-noise held-out/probe losses at checkpoints), last.pt (resume authority),
checkpoints/step_NNNNNN/{lora_<adapter>|controller}.safetensors + meta.json, final.json.
Source validation never enters a training stage: it is used only for fixed-noise evaluation
losses and later gate renders. No C2 data is reachable from this module.
"""
import argparse
import collections
import json
import os
import platform
import signal
import time
from pathlib import Path

import torch
import yaml

from generative_sim2real.depth.checkpoints import atomic_write_json

from . import checkpoints as ck
from . import gate
from . import lowrank_controller as lrc
from . import model as gm
from .inputs import FrameTensors, cursor_loader, encode_image, generation_dataset
from .model import compute_context, diffusion_loss
from .training import make_optimizer, noise_generator, train_updates

RUN_KEYS = {'stage', 'adapter', 'role', 'train', 'heldout', 'lr_from', 'lr', 'steps',
            'checkpoint_every', 'warmup', 'micro_batch', 'seeds', 'subset', 'source_lora_run',
            'time_cap_hours', 'eval_steps_note', 'controller'}
FINGERPRINT_CODE = ('model.py', 'inputs.py', 'training.py', 'controlnet.py', 'checkpoints.py')
ROLE_SPLIT = {'source_lora': 'source_train', 'target_lora': 'target_appearance_train',
              'controller': 'source_train'}


class Interrupted(Exception):
    pass


def validate_p4(config, generator):
    """Reject anything outside the declared protocol before touching data or a GPU."""
    def need(condition, message):
        if not condition:
            raise ValueError(f'Unsupported P4 config: {message}')
    runs = config['runs']
    need(set(runs) >= {'source_lora', 'target_lora', 'controller_overfit', 'controller_pilot'},
         'the four runs source_lora, target_lora, controller_overfit, controller_pilot')
    seeds, init_seeds = [], []
    for name, run in runs.items():
        need(set(run) <= RUN_KEYS, f'unknown keys in run {name}: {set(run) - RUN_KEYS}')
        need(run['stage'] in {'lora', 'controller'}, f'{name}.stage')
        need(run['role'] in ROLE_SPLIT, f'{name}.role')
        need(run['stage'] == ('controller' if run['role'] == 'controller' else 'lora'),
             f'{name}: stage/role mismatch')
        need(run['adapter'] == ('target' if run['role'] == 'target_lora' else 'source'),
             f'{name}.adapter')
        need(run['lr_from'] == ('controller_lr' if run['stage'] == 'controller' else 'lora_lr'),
             f'{name}.lr_from')
        need('lr' not in run or (isinstance(run['lr'], float) and 0 < run['lr'] < 1), f'{name}.lr')
        need(run['steps'] % run['checkpoint_every'] == 0, f'{name}: checkpoint_every must divide steps')
        need(config['manifests'][run['train']].endswith(ROLE_SPLIT[run['role']] + '.jsonl'),
             f'{name}.train manifest must be the {ROLE_SPLIT[run["role"]]} role')
        need(run['train'] != 'source_validation', f'{name}: source validation cannot train')
        gate.accumulation_for(generator['training'], run['micro_batch'])
        need(set(run['seeds']) == {'loader', 'noise'}, f'{name}.seeds')
        seeds += list(run['seeds'].values())
        if run['stage'] == 'controller':
            need(run.get('source_lora_run') == 'source_lora', f'{name} must start from source_lora')
            try:
                low = controller_spec(config, run)
            except ValueError as error:
                need(False, f'{name}.controller: {error}')
            init_seeds += [low['init_seed']] if low else []
        else:
            need('subset' not in run and 'source_lora_run' not in run, f'{name}: LoRA stage options')
            need('controller' not in run, f'{name}: controller options on a LoRA stage')
    init = generator['training']['seeds']['adapter_init'].values()
    need(not set(init_seeds) & {*seeds, *init},
         'controller.init_seed must differ from every loader/noise/adapter-init seed')
    need(len(set(seeds)) == len(seeds) and not set(seeds) & set(init),
         'every stage needs its own loader/noise seeds, distinct from the adapter init seeds')
    need(runs['target_lora']['heldout'] == 'target_appearance_validation', 'target heldout')
    need(runs['source_lora']['heldout'] == 'source_validation', 'source heldout')


def controller_spec(config, spec):
    """Low-rank controller settings of a run (config `controller:` block + run override), or None
    for the full controller (absent / kind: full: today's behaviour). Controller stage only."""
    if spec['stage'] != 'controller':
        return None
    return lrc.resolve_spec(config.get('controller'), spec.get('controller'))


def load_p4(path):
    config = yaml.safe_load(Path(path).read_text())
    generator = gate.load_config(config['generator_config'])
    validate_p4(config, generator)
    return config, generator


def select_overfit_pairs(medians_by_scene, per_scene):
    """Uniform integer endpoints of median-depth ranks within each scene (same rule as the
    P1 diagnostics); wrong depth = reversed cross-scene rank pairing over the sorted list."""
    scenes = sorted(medians_by_scene)
    if len(scenes) != 2:
        raise ValueError('The overfit set balances exactly two scenes')
    ids = []
    for scene in scenes:
        ranked = sorted(medians_by_scene[scene])
        if len(ranked) < per_scene:
            raise ValueError('Candidate pool smaller than the per-scene selection')
        picks = [round(i * (len(ranked) - 1) / (per_scene - 1)) for i in range(per_scene)]
        ids += [ranked[p][1] for p in picks]
    return ids, list(range(len(ids)))[::-1]


def overfit_pairs(manifest, pool):
    """Pick the 8 overfit pairs from source_train candidates (labels read for source only)."""
    groups = collections.defaultdict(list)
    for index, row in enumerate(manifest.rows):
        groups[row['sequence_id'].split('/')[0]].append(index)
    medians = {}
    for scene, indices in groups.items():
        step = (len(indices) - 1) / (pool - 1)
        chosen = sorted({indices[round(i * step)] for i in range(pool)})
        medians[scene] = []
        for index in chosen:
            target = manifest[index]['targets']
            depth = target['depth_m'][target['depth_valid'] & (target['depth_m'] > 0)]
            medians[scene].append((depth.median().item(), manifest.rows[index]['sample_id']))
    return select_overfit_pairs(medians, 4)


def sha_files(paths):
    return {str(p): ck.file_sha256(p) for p in map(Path, paths)}


def code_hashes():
    code = Path(__file__).parent
    return {name: ck.file_sha256(code / name) for name in FINGERPRINT_CODE}


def build_compatibility(name, spec, generator, ctx, hashes, manifest_path, extra_files, *, smoke,
                        config):
    import diffusers
    import peft
    import transformers

    training = generator['training']
    return {
        'run': name, 'smoke_only': smoke,
        'deterministic_attention': bool(config.get('deterministic_attention')),
        'spec': {k: v for k, v in spec.items() if k not in {'steps', 'checkpoint_every',
                                                               'time_cap_hours', 'eval_steps_note'}},
        'training': {k: training[k] for k in ('weight_decay', 'max_grad_norm', 'effective_batch')},
        'lr': spec.get('lr', training[spec['lr_from']]), 'lora': generator['lora'],
        'controller': generator['controller'], 'revision': generator['revision'],
        'precision': generator['precision'], 'adapter_init_seeds': training['seeds']['adapter_init'],
        'base_sha256': ctx.base_sha256, 'initial_hashes': hashes,
        'files_sha256': sha_files([manifest_path, Path(manifest_path).with_suffix('.meta.json'),
                                   generator['smoke']['prompt_cache'], *extra_files]),
        'code_sha256': code_hashes(),
        'versions': {'torch': torch.__version__, 'diffusers': diffusers.__version__,
                     'peft': peft.__version__, 'transformers': transformers.__version__},
        **({'controller_spec': low} if (low := controller_spec(config, spec)) else {}),
    }


def build_eval_set(ctx, dataset, *, wrong=None, seed, workers=4):
    """Encode a fixed frame set once: RGB posterior samples (fixed generator) and, for
    labeled source sets, deterministic control latents. Kept on the GPU (tiny)."""
    from torch.utils.data import DataLoader

    generator = noise_generator(ctx.device, seed)
    rgb, control, ids = [], [], []
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=workers)
    for batch in loader:
        rgb.append(encode_image(ctx.vae, batch['rgb'].to(ctx.device), control=False,
                                generator=generator))
        if 'control' in batch:
            control.append(encode_image(ctx.vae, batch['control'].to(ctx.device), control=True))
        ids += list(batch['sample_id'])
    count = len(ids)
    return {'ids': ids, 'rgb': torch.cat(rgb), 'control': torch.cat(control) if control else None,
            'wrong': wrong if wrong is not None else [(i + count // 2) % count for i in range(count)]}


@torch.no_grad()
def eval_loss(ctx, evalset, *, mode, seed, micro=8):
    """Mean fixed-noise denoising loss. modes: correct control, off (no controller),
    wrong (control of the paired other frame), none (LoRA stages, no controller)."""
    generator = noise_generator(ctx.device, seed)
    embeds = ctx.prompt['positive'].to(ctx.device)
    mask = ctx.prompt['positive_mask'].to(ctx.device)
    total, count = 0.0, len(evalset['ids'])
    for start in range(0, count, micro):
        latents = evalset['rgb'][start:start + micro]
        size = latents.shape[0]
        control = None
        if mode == 'correct':
            control = evalset['control'][start:start + size]
        elif mode == 'wrong':
            control = evalset['control'][evalset['wrong'][start:start + size]]
        with compute_context(ctx.device, ctx.dtype):
            loss = diffusion_loss(ctx.model, ctx.train_scheduler, latents,
                                  embeds.expand(size, -1, -1), mask.expand(size, -1),
                                  generator=generator, control=control)
        total += loss.item() * size
    return total / count


def subset_indices(count, wanted):
    step = (count - 1) / (wanted - 1) if wanted > 1 else 0
    return sorted({round(i * step) for i in range(min(wanted, count))})


def resolve_run(config, generator, name, *, smoke_steps=None):
    spec = dict(config['runs'][name])
    if smoke_steps is not None:
        spec['steps'] = smoke_steps
        spec['checkpoint_every'] = max(1, smoke_steps // 2)
    return spec


def run_training(config, generator, name, *, output, resume=False, stop_after_step=None,
                 smoke_steps=None, root=None):
    root = Path(root or Path.cwd())
    smoke = smoke_steps is not None
    spec = resolve_run(config, generator, name, smoke_steps=smoke_steps)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    last = output / 'last.pt'
    if not resume and last.exists():
        raise ValueError('Existing run requires --resume or a new output directory')
    if stop_after_step is not None and not smoke:
        raise ValueError('--stop-after-step is confined to smoke runs')
    stop = {'requested': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if config.get('deterministic_attention'):
        # Flash / memory-efficient SDPA backward use atomics (not bitwise repeatable); the math
        # kernel is, which makes an interrupted run equal an uninterrupted one on the GPU.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    training = generator['training']
    micro = spec['micro_batch']
    accumulation = gate.accumulation_for(training, micro)
    status_path = output / 'status.json'
    started = time.time()
    status = {'run': name, 'stage': spec['stage'], 'state': 'starting', 'step': 0,
              'target_steps': spec['steps'], 'eta_seconds': None, 'last_checkpoint': None,
              'exit_reason': None, 'pid': os.getpid(), 'host': platform.node(),
              'started_unix': started, 'smoke_only': smoke,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}

    def publish(**fields):
        status.update(fields, updated_unix=time.time())
        ck.write_status(status_path, **status)

    publish()
    try:
        return _run(config, generator, name, spec, output, resume, stop_after_step, smoke, root,
                    micro, accumulation, stop, publish, status)
    except Interrupted:
        publish(state='interrupted', exit_reason='signal; resume with --resume')
        return status
    except BaseException as error:
        publish(state='failed', exit_reason=f'{type(error).__name__}: {error}'[:500])
        raise


def _run(config, generator, name, spec, output, resume, stop_after_step, smoke, root, micro,
         accumulation, stop, publish, status):
    training = generator['training']
    ctx = gate.start(generator, root)
    device, dtype = ctx.device, ctx.dtype
    revision, rank, alpha = generator['revision'], generator['lora']['rank'], generator['lora']['alpha']
    data_root = os.environ['DATA_ROOT']
    manifests = config['manifests']
    manifest_path = manifests[spec['train']]
    extra_files = []
    if spec['stage'] == 'controller':
        source_run = (Path(config['output_root']) / ('smoke' if smoke else 'runs')
                      / spec['source_lora_run'])
        final = json.loads((source_run / 'final.json').read_text())
        if not final['completed_all_steps'] or final['smoke_only'] != smoke:
            raise ValueError('The source LoRA run must be complete (and smoke-ness must match)')
        source_file = Path(final['final_export']) / final['export_file']
        gm.load_adapter(ctx.transformer, source_file, 'source', rank=rank, alpha=alpha,
                        revision=revision)
        extra_files.append(source_file)
    lowrank = controller_spec(config, spec)
    lowrank_info = None
    if lowrank:  # low-capacity controller: convert the clean full controller before anything is hashed
        lowrank_info = lrc.convert(ctx.controller, lowrank)
        lowrank_info['copied_block_hashes_equal_base'] = getattr(ctx, 'provenance', {}).get(
            'copied_block_hashes_equal_base')
        if not lowrank_info['zero_at_init']:
            raise RuntimeError('Low-rank controller is not exactly zero at initialisation')
    hashes_before = gate.snapshot_hashes(ctx.model, ctx.controller)
    named = gm.set_stage(ctx.transformer, ctx.controller, stage=spec['stage'], adapter=spec['adapter'])
    if lowrank:  # set_stage re-enables every controller parameter; keep only the low-rank factors
        named = lrc.restrict_trainable(ctx.controller, ctx.transformer)
    audit = gm.parameter_audit(named)
    before = gm.snapshot(named)

    manifest = generation_dataset(manifest_path, data_root, stage=spec['role'])
    pairs = None
    if spec.get('subset'):
        pair_file = output / 'overfit_pairs.json'
        if pair_file.exists():
            pairs = json.loads(pair_file.read_text())
        else:
            ids, wrong = overfit_pairs(manifest, spec['subset']['pool_per_scene'])
            pairs = {'ids': ids, 'wrong_depth_permutation': wrong}
            atomic_write_json(pairs, pair_file)
        index = {row['sample_id']: i for i, row in enumerate(manifest.rows)}
        indices = [index[i] for i in pairs['ids']]
    else:
        indices = None
    dataset = FrameTensors(manifest, with_control=spec['stage'] == 'controller', indices=indices)
    compatibility = build_compatibility(name, spec, generator, ctx, hashes_before, manifest_path,
                                        extra_files, smoke=smoke, config=config)
    compatibility['overfit_pairs'] = pairs
    lr = spec.get('lr', training[spec['lr_from']])  # `lr` overrides the shared value (v2 retrain)
    optimizer, schedule = make_optimizer(named, lr=lr, weight_decay=training['weight_decay'],
                                         warmup=spec['warmup'])
    generator_noise = noise_generator(device, spec['seeds']['noise'])
    torch.manual_seed(spec['seeds']['noise'])
    state = {'update': 0, 'cursor': 0}
    total = spec['steps']
    end = total if stop_after_step is None else stop_after_step
    if resume:
        state = ck.load_training_state(output / 'last.pt', named, optimizer, schedule,
                                       generator_noise, compatibility, accumulation=accumulation)
        if state['update'] > end:
            raise ValueError('Checkpoint is beyond the requested stop update')
        for log in ('train.jsonl', 'eval.jsonl'):
            ck.truncate_jsonl(output / log, state['update'])
    atomic_write_json({'config': spec, 'compatibility': compatibility, 'audit': audit,
                       **({'controller_lowrank': lowrank_info} if lowrank else {})},
                      output / 'resolved.json')
    atomic_write_json({**gate.environment_info(generator), 'resumed': resume, 'smoke_only': smoke},
                      output / 'run_info.json')

    # Fixed evaluation sets (never trained on; encoded once with fixed generators).
    eval_seed = config['eval_seed']
    controller = spec['stage'] == 'controller'
    heldout_manifest = generation_dataset(
        manifests[spec['heldout']], data_root, stage='validation') if spec.get('heldout') else None
    evalsets = {}
    probe_dataset = FrameTensors(manifest, with_control=controller, indices=(
        indices if indices is not None else subset_indices(len(manifest), config['train_probe_frames'])))
    wrong = pairs['wrong_depth_permutation'] if pairs else None
    evalsets['train_probe'] = build_eval_set(ctx, probe_dataset, wrong=wrong, seed=eval_seed,
                                       workers=config['num_workers'])
    if heldout_manifest is not None:
        evalsets['heldout'] = build_eval_set(
            ctx, FrameTensors(heldout_manifest, with_control=controller), seed=eval_seed + 1,
            workers=config['num_workers'])

    def evaluate(step):
        record = {'step': step}
        modes = ('correct', 'off', 'wrong') if controller else ('none',)
        for key, evalset in evalsets.items():
            for mode in modes:
                record[f'{key}_{mode}'] = eval_loss(ctx, evalset, mode=mode, seed=eval_seed + 2)
        return record

    def append(file, record):
        with (output / file).open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    recent = collections.deque(maxlen=20)
    if not resume:
        append('eval.jsonl', evaluate(0))
    publish(state='running', step=state['update'])
    loader = cursor_loader(dataset, batch_size=micro, seed=spec['seeds']['loader'],
                           workers=config['num_workers'], start=state['cursor'])
    batches = iter(loader)
    export_dir = output / 'checkpoints'

    def save(step, exports):
        state.update(update=step, cursor=step * accumulation)
        if exports:
            directory, meta = ck.export_checkpoint(
                export_dir, step, kind=spec['stage'], model=ctx.model, controller=ctx.controller,
                adapter=spec['adapter'], rank=rank, alpha=alpha, revision=revision,
                extra={'run': name, 'train_manifest': manifest_path}, controller_spec=lowrank)
            status['last_checkpoint'] = str(directory)
        ck.save_training_state(output / 'last.pt', named, optimizer, schedule, generator_noise,
                               state, compatibility)

    def callback(step, record):
        append('train.jsonl', {'step': step, **{k: record[k] for k in (
            'loss', 'lr', 'grad_norm', 'seconds', 'data_wait_seconds', 'vae_seconds')}})
        recent.append(record['seconds'])
        boundary = step % spec['checkpoint_every'] == 0 or step == end
        if boundary:
            append('eval.jsonl', evaluate(step))
            save(step, exports=True)
        elif stop['requested']:
            save(step, exports=False)
        if boundary or step % 10 == 0 or stop['requested']:
            publish(step=step, loss=record['loss'], grad_norm=record['grad_norm'],
                    seconds_per_update=sum(recent) / len(recent),
                    eta_seconds=ck.eta_seconds(recent, total - step))
        if stop['requested'] and step < end:
            raise Interrupted()

    updates = end - state['update']
    records, _ = train_updates(
        ctx.model, ctx.vae, ctx.train_scheduler, ctx.prompt, named, batches, updates=updates,
        accumulation=accumulation, lr=lr, weight_decay=training['weight_decay'],
        max_grad_norm=training['max_grad_norm'], use_control=controller, device=device,
        dtype=dtype, callback=callback, optimizer=optimizer, lr_schedule=schedule,
        generator=generator_noise, first_update=state['update'])
    del loader
    hashes_after = gate.snapshot_hashes(ctx.model, ctx.controller)
    changed = gm.changed_names(before, named)
    trained = {'source_lora': 'lora_source', 'target_lora': 'lora_target'}.get(
        spec['role'], 'controller')
    final = {
        'run': name, 'steps_completed': end, 'target_steps': total, 'smoke_only': smoke,
        'trainable': audit, 'trainable_changed_tensors': len(changed),
        'hashes_before': hashes_before, 'hashes_after': hashes_after,
        'hash_changed_as_expected': {k: (hashes_before[k] != hashes_after[k]) == (k == trained)
                                        for k in hashes_before},
        'final_export': status['last_checkpoint'], 'export_file': (
            f'lora_{spec["adapter"]}.safetensors' if spec['stage'] == 'lora'
            else lrc.file_name(lowrank)),
        'loss_last': records[-1]['loss'] if records else None,
        'peak_vram_gib': gate.vram(device) if device.type == 'cuda' else None,
        'wall_seconds': time.time() - status['started_unix'],
        'completed_all_steps': end == total}
    if lowrank:
        final['controller_lowrank'] = {
            **lowrank_info, 'frozen_blocks_unchanged_by_training': (
                lrc.frozen_hash(ctx.controller) == lowrank_info['frozen_blocks_sha256']),
            'trainable_tensors_all_lowrank': all(
                lrc.is_trainable_name(n.removeprefix('controller.')) for n, _ in named)}
    atomic_write_json(final, output / 'final.json')
    publish(state='completed' if end == total else 'stopped_early', step=end, eta_seconds=0,
            exit_reason='completed' if end == total else 'stop_after_step (smoke)')
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--config', default='configs/p4.yaml')
    parser.add_argument('--output', help='default: <output_root>/runs/<run> (smoke: smoke/<run>)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-steps', type=int, help='labeled smoke: override the step budget')
    parser.add_argument('--stop-after-step', type=int, help='smoke only: interrupt for resume tests')
    args = parser.parse_args()
    config, generator = load_p4(args.config)
    default = Path(config['output_root']) / ('smoke' if args.smoke_steps else 'runs') / args.run
    run_training(config, generator, args.run, output=args.output or default, resume=args.resume,
                 stop_after_step=args.stop_after_step, smoke_steps=args.smoke_steps)


if __name__ == '__main__':
    main()
