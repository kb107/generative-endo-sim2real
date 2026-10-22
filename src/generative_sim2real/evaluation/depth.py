"""Raw metric-depth statistics and separately labeled GT-median oracle RMSE.

Aggregation semantics are selectively adapted from endoscopy-sim2real/evaluation/depth.py.
"""
from collections import defaultdict
import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

METRICS = ('abs_rel', 'rmse_mm', 'delta_1_1', 'delta_1_25', 'oracle_median_rmse_mm')


def frame_metrics(prediction, target, valid):
    if prediction.shape != target.shape or target.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError('Metric shapes/mask mismatch')
    if not valid.any():
        raise ValueError('No valid evaluation pixels')
    p, g = prediction[valid].double(), target[valid].double()
    if not torch.isfinite(g).all() or (g <= 0).any():
        raise ValueError('Marked-valid evaluation labels must be finite and positive')
    # Never discard model failures by modifying the ground-truth mask.
    if not torch.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Nonfinite or nonpositive prediction at a valid pixel')
    error = p - g
    ratio = torch.maximum(p / g, g / p)
    # Oracle diagnostic: lower median of double values, as in the reference frame_median mode.
    scale = g.median() / p.median()
    oracle_squared = ((p * scale - g) ** 2).sum().item()
    stats = {'valid_pixels': p.numel(), 'abs_rel_sum': (error.abs() / g).sum().item(),
             'squared_error_sum_m2': error.square().sum().item(),
             'delta_1_1_count': int((ratio < 1.1).sum()), 'delta_1_25_count': int((ratio < 1.25).sum()),
             'oracle_squared_error_sum_m2': oracle_squared}
    return {**stats, **metrics_from_stats(stats), 'oracle_median_scale': scale.item()}


def metrics_from_stats(stats):
    n = stats['valid_pixels']
    return {'abs_rel': stats['abs_rel_sum'] / n,
            'rmse_mm': 1000 * (stats['squared_error_sum_m2'] / n) ** .5,
            'delta_1_1': stats['delta_1_1_count'] / n,
            'delta_1_25': stats['delta_1_25_count'] / n,
            'oracle_median_rmse_mm': 1000 * (stats['oracle_squared_error_sum_m2'] / n) ** .5}


def aggregate_frames(rows):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Evaluation needs unique nonempty frame IDs')
    groups = defaultdict(list)
    for row in rows:
        groups[row['sequence_id']].append(row)
    sequence_rows = []
    for sid, frames in sorted(groups.items()):
        sequence_rows.append({'sequence_id': sid, 'frame_count': len(frames),
                              'valid_pixels': sum(r['valid_pixels'] for r in frames),
                              **{k: sum(r[k] for r in frames) / len(frames) for k in METRICS}})
    stat_keys = ('valid_pixels', 'abs_rel_sum', 'squared_error_sum_m2', 'delta_1_1_count',
                 'delta_1_25_count', 'oracle_squared_error_sum_m2')
    pooled = {k: sum(r[k] for r in rows) for k in stat_keys}
    return {'frame_count': len(rows), 'sequence_count': len(groups), 'valid_pixels': pooled['valid_pixels'],
            'sequence_macro': {k: sum(r[k] for r in sequence_rows) / len(sequence_rows) for k in METRICS},
            'frame_macro': {k: sum(r[k] for r in rows) / len(rows) for k in METRICS},
            'pixel_global': metrics_from_stats(pooled), 'sequences': sequence_rows}


def write_report(directory, rows, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = aggregate_frames(rows)
    for name, records in (('frames', rows), ('sequences', summary['sequences'])):
        (directory / f'{name}.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
        with (directory / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (directory / 'summary.json').write_text(json.dumps({'metadata': metadata, **summary}, indent=2, allow_nan=False) + '\n')
    return summary


def validation_indices(size, limit):
    """All frames (limit None) or a deterministic evenly spaced smoke subset."""
    if limit is None:
        return list(range(size))
    if type(limit) is not int or not 1 <= limit <= size:
        raise ValueError('validation_frames must be None or an integer within the manifest size')
    indices = sorted({round(v) for v in torch.linspace(0, size - 1, limit).tolist()})
    return indices


@torch.no_grad()
def evaluate_source(model, dataset, device='cpu', indices=None):
    """Source-only evaluation, one native-size frame at a time; never constructs C2."""
    if dataset.split != 'source_validation' or dataset.geometry != 'source_native':
        raise ValueError('Training validation accepts only native source_validation')
    was_training = model.training
    model.eval()
    rows = []
    try:
        for i in (range(len(dataset)) if indices is None else indices):
            sample = dataset[i]
            rgb = sample['rgb'][None].to(device)
            K = sample['camera']['K'][None].to(device)
            prediction = model(rgb, K)['depth_m']
            target, valid = sample['targets']['depth_m'], sample['targets']['depth_valid']
            prediction = F.interpolate(prediction, size=target.shape[-2:], mode='bilinear', align_corners=False)[0].cpu()
            rows.append({'sample_id': sample['sample_id'], 'sequence_id': sample['meta']['sequence_id'],
                         **frame_metrics(prediction, target, valid)})
    finally:
        model.train(was_training)
    return rows


def require_smoke_for_subset(frames, head_smoke_only, config_purpose):
    """A frame subset is a labeled smoke diagnostic, never a source-validation result."""
    if frames is not None and not (head_smoke_only is True or config_purpose == 'smoke'):
        raise ValueError('--frames subsets require a smoke head or a smoke-purpose --config')


def main():
    """Evaluate the official metric head (no --head) or a saved head on native source validation.

    Target (C2) evaluation is a separate frozen-registry entry point (P6) and is
    deliberately not reachable from here.
    """
    from ..data.manifests import ManifestDataset
    from ..depth.checkpoints import read_head
    from ..depth.model import DA3MetricDepth, verify_upstream_checkout
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--manifest', default='outputs/manifests/milestone_v1/source_validation.jsonl')
    parser.add_argument('--head')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frames', type=int, help='labeled smoke subset size (smoke head or --config only)')
    parser.add_argument('--config', help='optional DA3 config; its purpose is checked for --frames')
    args = parser.parse_args()
    verify_upstream_checkout()
    purpose = None
    if args.config:
        import yaml
        purpose = yaml.safe_load(Path(args.config).read_text()).get('purpose')
    if not args.head:
        require_smoke_for_subset(args.frames, None, purpose)
    sidecar = json.loads(Path(args.manifest).with_suffix('.meta.json').read_text())
    if sidecar.get('split') != 'source_validation':
        raise ValueError('This entry point evaluates only source_validation')
    dataset = ManifestDataset(args.manifest, os.environ['DATA_ROOT'], purpose='evaluation', geometry='source_native')
    model = DA3MetricDepth.from_pretrained().to(args.device)
    metadata = {'split': 'source_validation', 'native_labels': True, 'smoke_subset_frames': args.frames, 'head': None}
    head_smoke = None
    if args.head:
        checkpoint = read_head(args.head)
        head_smoke = checkpoint['compatibility']['smoke_only']
        require_smoke_for_subset(args.frames, head_smoke, purpose)
        model.load_head_state_dict(checkpoint['head_state_dict'])
        metadata['head'] = {'path': args.head, 'sha256': checkpoint['head_sha256'],
                            'selected_step': checkpoint['selected_step'],
                            'smoke_only': checkpoint['compatibility']['smoke_only']}
    rows = evaluate_source(model, dataset, args.device, validation_indices(len(dataset), args.frames))
    summary = write_report(args.output, rows, metadata)
    print(json.dumps({k: summary[k] for k in ('frame_count', 'sequence_count', 'sequence_macro', 'frame_macro', 'pixel_global')}, indent=2))


if __name__ == '__main__':
    main()
