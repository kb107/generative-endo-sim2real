"""Direct single-process, source-only DA3 head training.

The logical global batch is loaded on CPU before microbatch backward. All three
loss denominators are computed globally, making accumulated gradients match the
same full-batch masked loss despite varying label coverage. No DataLoader
prefetch or worker RNG is involved; occurrence cursor advances only on updates.
"""
from __future__ import annotations
import argparse
import copy
from importlib import metadata
import json
import math
import os
import shutil
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch
import yaml

from ..data.collate import collate_frame_samples
from ..data.manifests import ManifestDataset
from ..data.rgb_modes import AUGMENTATION, RGBModeDataset, OccurrenceOrder, file_sha256, load_matched_ids
from ..evaluation.depth import evaluate_source, validation_indices, write_report
from .checkpoints import atomic_replace, atomic_write_json, head_digest, load_training, read_head, save_head, save_training
from .loss import canonical_depth_loss, mask_counts, validate_targets
from .model import CHECKPOINT, CHECKPOINT_REVISION, DA3_REVISION, DA3MetricDepth, verify_upstream_checkout


TRAINING_KEYS = {'seed', 'optimizer_steps', 'global_batch_size', 'microbatch_size', 'learning_rate', 'weight_decay',
                 'warmup_steps', 'gradient_clip_norm', 'validation_every', 'checkpoint_every', 'depth_weight',
                 'gradient_weight', 'precision', 'optimizer', 'schedule', 'world_size'}
# Every experiment run (R/A/G/M) must share these exactly. Only data.mode and data.generated_manifest vary;
# microbatch_size and checkpoint_every are execution details that do not change the logical batch or schedule.
LOCKED_EXPERIMENT = {
    'training': {'seed': 0, 'optimizer_steps': 3000, 'global_batch_size': 16, 'learning_rate': 1e-4,
                 'weight_decay': 0.01, 'warmup_steps': 100, 'gradient_clip_norm': 1.0, 'validation_every': 250,
                 'depth_weight': 1.0, 'gradient_weight': 1.0, 'precision': 'float32', 'optimizer': 'AdamW',
                 'schedule': 'warmup_cosine', 'world_size': 1},
    'data': {'source_train': 'outputs/manifests/milestone_v1/source_train.jsonl',
             'source_validation': 'outputs/manifests/milestone_v1/source_validation.jsonl',
             'matched_ids': 'outputs/manifests/milestone_v1/downstream_ids.json', 'outputs_root': 'outputs'}}


def apply_rgb_overrides(config, mode=None, generated_manifest=None):
    """The only per-run variation allowed across R/A/G/M, applied to a copy of the shared config."""
    config = copy.deepcopy(config)
    if mode is not None:
        config['data']['mode'] = mode
    if generated_manifest is not None:
        config['data']['generated_manifest'] = generated_manifest or None
    return config


def validate_config(config):
    if config.get('schema_version') != 1 or config.get('purpose') not in {'experiment', 'smoke'}:
        raise ValueError('Unsupported DA3 config schema or missing purpose (experiment|smoke)')
    if set(config) != {'schema_version', 'purpose', 'model', 'data', 'training', 'evaluation'}:
        raise ValueError('Unexpected top-level DA3 config fields')
    smoke = config['purpose'] == 'smoke'
    model, data, train, evaluation = (config[k] for k in ('model', 'data', 'training', 'evaluation'))
    expected = {'da3_revision': DA3_REVISION, 'checkpoint': CHECKPOINT,
                'checkpoint_revision': CHECKPOINT_REVISION, 'process_hw': [504, 504],
                'trainable': 'head', 'canonical_focal_px': 300.}
    if model != expected:
        raise ValueError('DA3 model must use the identical official metric base and 504-square head-only protocol')
    frames = evaluation.get('validation_frames') if isinstance(evaluation, dict) else None
    if evaluation != {'selection': 'source_validation_sequence_macro_raw_abs_rel',
                      'prediction_resize': 'bilinear_align_corners_false', 'target_views': [],
                      'validation_frames': frames}:
        raise ValueError('DA3 training permits only source validation; target views must be empty')
    if frames is not None and (not smoke or type(frames) is not int or frames < 1):
        raise ValueError('A validation subset is permitted only for labeled smoke runs')
    if set(data) != {'source_train', 'source_validation', 'matched_ids', 'mode', 'generated_manifest', 'outputs_root', 'augmentation'}:
        raise ValueError('Unexpected dataset fields; training has no target dataset interface')
    if data['augmentation'] != AUGMENTATION or data['mode'] not in {'R', 'A', 'G', 'M'}:
        raise ValueError('Unsupported RGB/photometric policy')
    if set(train) != TRAINING_KEYS:
        raise ValueError('Unknown or missing training settings')
    if (data['mode'] in {'G', 'M'}) != (data['generated_manifest'] is not None):
        raise ValueError('Generated manifest is required for G/M and forbidden for R/A')
    positive_ints = ('optimizer_steps', 'global_batch_size', 'microbatch_size', 'validation_every', 'checkpoint_every',
                     'world_size')
    if any(type(train[k]) is not int or train[k] < 1 for k in positive_ints):
        raise ValueError('Invalid training counts')
    if type(train['warmup_steps']) is not int or not 0 <= train['warmup_steps'] <= train['optimizer_steps']:
        raise ValueError('Invalid warmup duration')
    if type(train['seed']) is not int or train['seed'] < 0:
        raise ValueError('Seed must be a nonnegative integer')
    if train['global_batch_size'] % train['microbatch_size'] or train['world_size'] != 1:
        raise ValueError('Exact accumulation requires divisible global/microbatch sizes and one process')
    if train['optimizer'] != 'AdamW' or train['schedule'] != 'warmup_cosine' or train['precision'] != 'float32':
        raise ValueError('Only AdamW/warmup_cosine/float32 are implemented')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip_norm', 'depth_weight', 'gradient_weight'):
        if not math.isfinite(train[key]) or train[key] < 0:
            raise ValueError(f'Invalid {key}')
    if train['learning_rate'] == 0 or train['gradient_clip_norm'] == 0 or train['depth_weight'] + train['gradient_weight'] == 0:
        raise ValueError('Learning rate, clipping norm and total loss weight must be positive')
    if train['validation_every'] % train['checkpoint_every']:
        raise ValueError('checkpoint_every must divide validation_every')
    if not smoke:
        for group, values in LOCKED_EXPERIMENT.items():
            if any(config[group][key] != value or type(config[group][key]) is not type(value) for key, value in values.items()):
                raise ValueError('Experiment settings are locked across R/A/G/M; changes need a shared protocol update')


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.device(device).type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def learning_rate_factor(completed_steps, total, warmup):
    """LR for the next update; update 1 starts at 1/warmup of base LR."""
    if warmup and completed_steps < warmup:
        return (completed_steps + 1) / warmup
    progress = (completed_steps - warmup) / max(1, total - warmup)
    return .5 * (1 + math.cos(math.pi * min(1., max(0., progress))))


def code_hashes():
    package = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(package)): file_sha256(p)
            for subdir in ('data', 'depth', 'evaluation') for p in sorted((package / subdir).glob('*.py'))}


def environment_versions():
    names = ('torch', 'torchvision', 'numpy', 'opencv-python', 'opencv-python-headless',
             'scipy', 'pillow', 'depth-anything-3', 'xformers')
    result = {'python': platform.python_version()}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return result


def compatibility(config, train_source, validation_source):
    files = [train_source.path, train_source.path.with_suffix('.meta.json'), validation_source.path,
             validation_source.path.with_suffix('.meta.json'), Path(config['data']['matched_ids'])]
    generated = config['data']['generated_manifest']
    if generated:
        files.append(Path(generated))
    return {'config': config, 'files_sha256': {str(p): file_sha256(p) for p in files},
            'code_sha256': code_hashes(), 'environment': environment_versions(),
            'smoke_only': config['purpose'] == 'smoke', 'sampler': 'seed_plus_epoch_randperm_occurrence_cursor_v1',
            'rgb_rng': 'sha256_seed_id_occurrence_pcg64_v1', 'loss_reduction': 'global_valid_pixels_and_neighbor_pairs_v1'}


def build_datasets(config, data_root):
    # Validate roles from JSON before constructing either underlying adapter.
    data = config['data']
    for key, expected_split in (('source_train', 'source_train'), ('source_validation', 'source_validation')):
        sidecar = json.loads(Path(data[key]).with_suffix('.meta.json').read_text())
        if sidecar.get('split') != expected_split:
            raise ValueError(f'{key} must be an explicit {expected_split} manifest')
    source = ManifestDataset(data['source_train'], data_root, purpose='training', geometry='source_generation512')
    validation = ManifestDataset(data['source_validation'], data_root, purpose='training', geometry='source_native')
    ids = load_matched_ids(data['matched_ids'], source, smoke=config['purpose'] == 'smoke')
    training = RGBModeDataset(source, ids, mode=data['mode'], seed=config['training']['seed'],
                              generated_manifest=data['generated_manifest'], outputs_root=data['outputs_root'])
    return source, training, validation


def runtime_info(device):
    """Informational run metadata; deliberately outside the resume fingerprint."""
    info = {'device': str(device), 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'hostname': platform.node()}
    if torch.device(device).type == 'cuda':
        index = torch.device(device).index or torch.cuda.current_device()
        info.update(gpu_name=torch.cuda.get_device_name(index), cudnn=torch.backends.cudnn.version(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_benchmark=torch.backends.cudnn.benchmark)
    return info


def _truncate_logs(output, completed):
    """Drop records/reports written after the last durable checkpoint of an interrupted run."""
    log = output / 'train.jsonl'
    if log.exists():
        lines, kept = log.read_text().splitlines(), []
        for number, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if number == len(lines) - 1:
                    break                          # a write interrupted by the crash; the step is re-run
                raise
            if record['step'] <= completed:
                kept.append(line)
        atomic_replace(log, lambda temporary: Path(temporary).write_text(''.join(line + '\n' for line in kept)))
    for directory in (output / 'validation').glob('step_*'):
        if int(directory.name.split('_')[1]) > completed:
            shutil.rmtree(directory)


def _verify_best(output, model, state, run_compatibility):
    """Tie best.pt to the training state by content, repairing only a crash after last.pt."""
    path = output / 'best.pt'
    try:
        best = read_head(path)
        intact = (best['compatibility'] == run_compatibility and best['selected_step'] == state['best_step']
                  and best['head_sha256'] == state['best_head_sha256'])
    except FileNotFoundError:
        intact = False
    if intact:
        return
    # last.pt is written first; when it records a new best, its head is that best head.
    if state['best_step'] == state['optimizer_step'] and head_digest(model.head_state_dict()) == state['best_head_sha256']:
        save_head(path, model, state, run_compatibility)
        return
    raise ValueError('Resume best head does not match the training checkpoint (missing or changed)')


def train(model, dataset, validation, config, output, run_compatibility, *, device='cpu', resume=None,
          stop_after_step=None, evaluate=evaluate_source):
    """Run complete updates, then source-select/save at step zero and intervals.

    ``stop_after_step`` supports an interruption/restart smoke while retaining
    the original total schedule in the resume fingerprint.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not run_compatibility['smoke_only'] and stop_after_step is not None:
        raise ValueError('Early-stop override is confined to smoke runs')
    settings = config['training']
    total, batch_size, micro = (settings[k] for k in ('optimizer_steps', 'global_batch_size', 'microbatch_size'))
    end = total if stop_after_step is None else stop_after_step
    if type(end) is not int or not 1 <= end <= total:
        raise ValueError('Invalid stop update')
    if resume is None and (output / 'last.pt').exists():
        raise ValueError('Existing run requires explicit resume or a new output directory')
    model.to(device)
    parameters = model.configure_head_training()
    optimizer = torch.optim.AdamW(parameters, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, total, settings['warmup_steps']))
    state = {'optimizer_step': 0, 'cursor': 0, 'best_abs_rel': float('inf'), 'best_step': -1,
             'best_head_sha256': None, 'initial_head_sha256': head_digest(model.head_state_dict())}
    indices = validation_indices(len(validation), config['evaluation']['validation_frames'])
    if resume is not None:
        state = load_training(resume, model, optimizer, scheduler, run_compatibility)
        _verify_best(output, model, state, run_compatibility)
        if state['optimizer_step'] > end:
            raise ValueError('Checkpoint is beyond requested stop update')
        _truncate_logs(output, state['optimizer_step'])
    atomic_write_json(run_compatibility, output / 'resolved.json')
    atomic_write_json(runtime_info(device), output / 'run_info.json')
    order = OccurrenceOrder(len(dataset), settings['seed'])

    def write_status():
        atomic_write_json({'state': state, 'target_steps': total, 'completed_steps': state['optimizer_step'],
                           'selected_step': state['best_step'], 'smoke_only': run_compatibility['smoke_only'],
                           'device': str(device)}, output / 'status.json')

    def validate_and_save():
        rows = evaluate(model, validation, device, indices)
        summary = write_report(output / f'validation/step_{state["optimizer_step"]:06d}', rows,
                               {'smoke_only': run_compatibility['smoke_only'], 'split': 'source_validation',
                                'native_labels': True, 'selection': config['evaluation']['selection'],
                                'validation_frames': config['evaluation']['validation_frames']})
        score = summary['sequence_macro']['abs_rel']
        improved = score < state['best_abs_rel']
        if improved:
            state.update(best_abs_rel=score, best_step=state['optimizer_step'],
                         best_head_sha256=head_digest(model.head_state_dict()))
        # Training state first: a crash before best.pt is repaired from its head on resume.
        save_training(output / 'last.pt', model, optimizer, scheduler, dict(state), run_compatibility)
        if improved:
            save_head(output / 'best.pt', model, state, run_compatibility)

    if resume is None:
        validate_and_save()
        write_status()
    model.train()
    while state['optimizer_step'] < end:
        samples = [dataset.get_occurrence(order.at(cursor), cursor)
                   for cursor in range(state['cursor'], state['cursor'] + batch_size)]
        batch = collate_frame_samples(samples)
        target, valid = batch['targets']['depth_m'], batch['targets']['depth_valid']
        validate_targets(target, valid)
        counts = mask_counts(valid)
        optimizer.zero_grad(set_to_none=True)
        sums = {'total': 0., 'depth': 0., 'gradient': 0.}
        for start in range(0, batch_size, micro):
            part = slice(start, start + micro)
            rgb, K = batch['rgb'][part].to(device), batch['camera']['K'][part].to(device)
            prediction = model(rgb, K)
            canonical = model.metric_to_canonical(target[part].to(device), prediction['K_processed'])
            loss, terms = canonical_depth_loss(prediction['canonical_depth'], canonical, valid[part].to(device),
                                               depth_weight=settings['depth_weight'], gradient_weight=settings['gradient_weight'],
                                               denominators=counts)
            loss.backward()
            for key in sums:
                sums[key] += terms[key].item()
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip_norm'], error_if_nonfinite=True)
        lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        state['optimizer_step'] += 1
        state['cursor'] += batch_size
        log = {'step': state['optimizer_step'], 'sample_ids': batch['sample_id'], 'loss': sums,
               'learning_rate': lr, 'gradient_norm': float(norm), 'smoke_only': run_compatibility['smoke_only']}
        with (output / 'train.jsonl').open('a') as stream:
            stream.write(json.dumps(log, allow_nan=False) + '\n')
        if state['optimizer_step'] % settings['validation_every'] == 0 or state['optimizer_step'] == total:
            validate_and_save()
        elif state['optimizer_step'] % settings['checkpoint_every'] == 0 or state['optimizer_step'] == end:
            save_training(output / 'last.pt', model, optimizer, scheduler, dict(state), run_compatibility)
        write_status()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/da3.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume')
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--mode', choices=['R', 'A', 'G', 'M'], help='RGB mode; overrides the shared config')
    parser.add_argument('--generated-manifest', help='Generated manifest for G/M ("" for none)')
    args = parser.parse_args()
    config = apply_rgb_overrides(yaml.safe_load(Path(args.config).read_text()), args.mode, args.generated_manifest)
    validate_config(config)
    if torch.device(args.device).type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        print(subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv'], text=True), flush=True)
    source, dataset, validation = build_datasets(config, os.environ['DATA_ROOT'])
    verify_upstream_checkout()
    seed_all(config['training']['seed'], args.device)
    model = DA3MetricDepth.from_pretrained()
    result = train(model, dataset, validation, config, args.output, compatibility(config, source, validation),
                   device=args.device, resume=args.resume, stop_after_step=args.stop_after_step)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
