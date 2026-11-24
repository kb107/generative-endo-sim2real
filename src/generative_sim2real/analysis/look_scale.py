"""Look -> metric-scale probe: does changing only the image look move a DA3 head's predicted metric scale?

Source labels only. The images are the paired renders of the 32 source-VALIDATION conditions of a P5 matrix
(outputs/p5-diagnostics/matrix_lc3000_seed17: 512-square generation-stage images of the source-validation
frame, with refs.pt holding that frame's 512-stage depth/valid/K -- the same geometry the generator was
conditioned on). Each image version is fed to a DA3 head exactly like depth/train.py validation does it
(model(rgb, K): bicubic resize to 504, K scaled with the image, canonical focal, bilinear resample of the
prediction onto the label grid), keeping the TRUE source labels, K and valid mask of the condition.

Per image: log(median_pred / median_gt) over valid pixels (positive = predicts too far) and the scale-aligned
AbsRel (prediction multiplied by median_gt / median_pred, so a pure shape error). For gain sweeps: the slope of
log median_pred against log gain (-0.5 would mean predicted depth ~ 1/sqrt(brightness)).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import rgb_modes as rm
from ..variants import build as vb
from ..variants import color

MATRIX_DIR = 'outputs/p5-diagnostics/matrix_lc3000_seed17'
DEFAULT_HEADS = ('outputs/runs_c2/da3_R_seed0', 'outputs/runs_c2/da3_G_seed0', 'outputs/runs_c2_v4/da3_G_seed0',
                 'outputs/runs_c2_simuscope/da3_G_seed0')
GAINS = (.5, .7, 1., 1.4, 2.)
GAIN_BASES = ('real', 'target_correct')
LOOK_VERSIONS = ('real', 'source_correct', 'target_correct', 'reinhard_real', 'a_strong_real')


def quantize(image):
    """The 8-bit grid every generated image lives on."""
    return (image.clamp(0, 1) * 255).round() / 255


def apply_gain(image, gain):
    """Multiply then clip, then 8-bit quantize (a pure exposure/brightness change)."""
    return quantize(image * gain)


def load_png(path):
    from PIL import Image
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2, 0, 1).float() / 255


def version_name(base, gain):
    return f'{base}_x{gain:g}'


def build_versions(directory, condition, valid, reference_rgb, strong_params, gain_sweeps=True):
    """All image versions of one condition (dict name -> float32 [3,512,512] on the 8-bit grid)."""
    directory = Path(directory)
    real = load_png(directory / 'images' / 'real' / f'c{condition}.png')
    versions = {'real': real,
                'source_correct': load_png(directory / 'images' / 'source_correct' / f'c{condition}.png'),
                'target_correct': load_png(directory / 'images' / 'target_correct' / f'c{condition}.png')}
    versions['reinhard_real'] = quantize(color.reinhard_lab(
        real, reference_rgb, color.luminance_mask(real, valid), color.luminance_mask(reference_rgb)))
    versions['a_strong_real'] = quantize(color.apply_strong_photometric(real, strong_params))
    for base in GAIN_BASES if gain_sweeps else ():
        for gain in GAINS:
            if gain != 1.:
                versions[version_name(base, gain)] = apply_gain(versions[base], gain)
    return versions


@torch.no_grad()
def predict_frame(model, rgb, K, size_hw, device):
    """Trainer validation path: prediction at the label grid (bilinear, align_corners=False)."""
    depth = model(rgb[None].to(device), K[None].to(device))['depth_m']
    return F.interpolate(depth, size=size_hw, mode='bilinear', align_corners=False)[0].cpu()


def image_stats(prediction, target, valid):
    """log(median_pred/median_gt), scale-aligned AbsRel, and the log median prediction of one image."""
    valid = valid & (target > 0)
    p, g = prediction[valid].double(), target[valid].double()
    if not torch.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Nonfinite or nonpositive prediction at a valid pixel')
    scale = g.median() / p.median()
    return {'log_ratio': math.log(p.median().item() / g.median().item()),
            'log_median_pred': math.log(p.median().item()),
            'aligned_abs_rel': ((p * scale - g).abs() / g).mean().item()}


def slope(log_gains, log_values):
    """Least-squares slope of log median prediction against log gain."""
    x, y = np.asarray(log_gains, float), np.asarray(log_values, float)
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def summarize(rows, versions):
    """rows[condition][version] -> stats dict; returns per-version mean/median/SD/n and paired deltas vs real."""
    out = {}
    real = np.array([r['real']['log_ratio'] for r in rows])
    for v in versions:
        lr = np.array([r[v]['log_ratio'] for r in rows])
        ar = np.array([r[v]['aligned_abs_rel'] for r in rows])
        delta = lr - real
        out[v] = {'n': len(rows), 'log_ratio_mean': float(lr.mean()), 'log_ratio_median': float(np.median(lr)),
                  'log_ratio_sd': float(lr.std(ddof=1)), 'aligned_abs_rel_mean': float(ar.mean()),
                  'delta_vs_real_mean': float(delta.mean()),
                  'delta_vs_real_se': float(delta.std(ddof=1) / math.sqrt(len(rows)))}
    return out


def sweep_slopes(rows):
    result = {}
    for base in GAIN_BASES:
        names = [version_name(base, g) if g != 1. else base for g in GAINS]
        per_condition = [slope([math.log(g) for g in GAINS], [r[n]['log_median_pred'] for n in names]) for r in rows]
        result[base] = {'slope_mean': float(np.mean(per_condition)), 'slope_median': float(np.median(per_condition)),
                        'slope_se': float(np.std(per_condition, ddof=1) / math.sqrt(len(rows))),
                        'per_condition': per_condition}
    return result


def evaluate_head(model, versions_by_condition, K, depth, valid, device):
    rows = []
    for i, versions in enumerate(versions_by_condition):
        row = {}
        for name, rgb in versions.items():
            prediction = predict_frame(model, rgb, K[i], depth.shape[-2:], device)
            row[name] = image_stats(prediction, depth[i], valid[i])
        rows.append(row)
    return rows


def format_tables(results, versions, with_slopes=True):
    def table(title, cell):
        lines = [f'### {title}', '', '| head | ' + ' | '.join(versions) + ' |', '|---|' + '---|' * len(versions)]
        for head, entry in results.items():
            lines.append(f'| {head} | ' + ' | '.join(cell(entry, v) for v in versions) + ' |')
        return lines + ['']
    lines = table('log(median_pred / median_gt): mean (median); positive = predicts too far',
                  lambda e, v: f"{e['summary'][v]['log_ratio_mean']:+.3f} ({e['summary'][v]['log_ratio_median']:+.3f})")
    lines += table('Paired change of log ratio vs real (mean +- SE over conditions)',
                   lambda e, v: f"{e['summary'][v]['delta_vs_real_mean']:+.3f} +- {e['summary'][v]['delta_vs_real_se']:.3f}")
    lines += table('Scale-aligned AbsRel (shape), mean over conditions',
                   lambda e, v: f"{e['summary'][v]['aligned_abs_rel_mean']:.4f}")
    if not with_slopes:
        return '\n'.join(lines) + '\n'
    lines += ['### Gain-sweep slope of log median_pred vs log gain (mean over conditions +- SE; -0.5 = 1/sqrt(gain))', '',
              '| head | ' + ' | '.join(GAIN_BASES) + ' |', '|---|' + '---|' * len(GAIN_BASES)]
    for head, e in results.items():
        lines.append(f'| {head} | ' + ' | '.join(
            f"{e['slopes'][b]['slope_mean']:+.3f} +- {e['slopes'][b]['slope_se']:.3f}" for b in GAIN_BASES) + ' |')
    return '\n'.join(lines) + '\n'


def head_label(path):
    path = Path(path)
    return f'{path.parent.parent.name}/{path.parent.name}/{path.name}'


def load_head_state(path):
    """best.pt (kind da3_head) or last.pt (kind da3_training); both hold head_state_dict (local trusted files)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('kind') not in {'da3_head', 'da3_training'}:
        raise ValueError(f'{path} is not a DA3 head/training checkpoint')
    return checkpoint['head_state_dict'], {'kind': checkpoint['kind'], 'selected_step': checkpoint.get('selected_step'),
                                           'optimizer_step': (checkpoint.get('state') or {}).get('optimizer_step')}


def main():
    from ..data.manifests import ManifestDataset
    from ..depth.checkpoints import head_digest
    from ..depth.model import DA3MetricDepth, verify_upstream_checkout
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', default=MATRIX_DIR)
    parser.add_argument('--heads', nargs='*', help='run directories (best.pt and last.pt are both probed)')
    parser.add_argument('--out', default='outputs/analysis/look_scale_probe')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--conditions', type=int, help='first N conditions (smoke only)')
    parser.add_argument('--registry', help='frozen registry JSON: probe every registered head (Z + each trained entry, '
                                           'bbLoRA applied) under its registry name instead of --heads')
    parser.add_argument('--no-gain-sweeps', action='store_true', help='skip the gain-sweep versions (and slopes)')
    parser.add_argument('--only', nargs='*', help='with --registry: restrict to these registry names (smoke)')
    args = parser.parse_args()
    matrix, out = Path(args.matrix), Path(args.out)
    render = json.loads((matrix / 'render.json').read_text())
    ids = render['ids']
    refs = torch.load(matrix / 'refs.pt', weights_only=True)
    n = args.conditions or len(ids)
    ids, K, depth, valid = ids[:n], refs['K'][:n], refs['depth_m'][:n], refs['valid'][:n]
    data_root = os.environ['DATA_ROOT']
    # The matrix conditions are source_validation frames at the 512 stage; confirm the stored labels are exactly
    # the trainer's source_generation512 geometry for those IDs (same bytes -> same fingerprint).
    val512 = ManifestDataset(vb.SOURCE_VALIDATION, data_root, purpose='training', geometry='source_generation512')
    lookup = {r['sample_id']: i for i, r in enumerate(val512.rows)}
    for i, sid in enumerate(ids):
        sample = val512[lookup[sid]]
        if not (torch.equal(sample['targets']['depth_m'], depth[i]) and torch.equal(sample['targets']['depth_valid'], valid[i])
                and torch.equal(sample['camera']['K'], K[i])):
            raise ValueError(f'refs.pt labels differ from source_generation512 geometry for {sid}')
    target = ManifestDataset(vb.TARGET_TRAIN, data_root, purpose='training')  # C1 appearance-train, RGB only
    versions_by_condition, references = [], []
    for i, sid in enumerate(ids):
        seed = vb.variant_seed('look_scale_probe', sid)
        ref_index = vb.reference_index(seed, len(target))
        references.append(target.rows[ref_index]['sample_id'])
        params = color.draw_strong_params(np.random.default_rng(seed))
        versions_by_condition.append(build_versions(matrix, i, valid[i], target[ref_index]['rgb'], params,
                                                    gain_sweeps=not args.no_gain_sweeps))
    versions = list(versions_by_condition[0])
    verify_upstream_checkout()
    model = DA3MetricDepth.from_pretrained().to(args.device).eval().requires_grad_(False)
    pretrained = {k: v.clone() for k, v in model.head_state_dict().items()}
    heads = [('Z_pretrained', None, None)]
    if args.registry:
        from ..depth.checkpoints import apply_checkpoint
        from ..evaluation.c2 import prepare_heads
        registry = json.loads(Path(args.registry).read_text())
        states = prepare_heads(model, registry, '.')       # verifies every head file/digest, bbLoRA included
        heads = [(name, None, states[name]) for name in registry['method_names'] if not args.only or name in args.only]
    else:
        for run in (args.heads or DEFAULT_HEADS):
            for name in ('best.pt', 'last.pt'):
                heads.append((head_label(Path(run) / name), Path(run) / name, None))
    results = {}
    native = ManifestDataset(vb.SOURCE_VALIDATION, data_root, purpose='training', geometry='source_native')
    for label, path, checkpoint in heads:
        if checkpoint is not None:
            apply_checkpoint(model, checkpoint)
            info = {'kind': 'registry', 'backbone_lora': 'backbone_lora_state_dict' in checkpoint}
        else:
            if path is None:
                state, info = pretrained, {'kind': 'official_pretrained'}
            else:
                state, info = load_head_state(path)
            model.load_head_state_dict(state)
        info['head_sha256'] = head_digest(model.head_state_dict())
        rows = evaluate_head(model, versions_by_condition, K, depth, valid, args.device)
        # Reproduce the trainer's native-475 validation path on the real frames as a consistency check.
        check = []
        for i, sid in enumerate(ids):
            s = native[{r['sample_id']: j for j, r in enumerate(native.rows)}[sid]]
            prediction = predict_frame(model, s['rgb'], s['camera']['K'], s['targets']['depth_m'].shape[-2:], args.device)
            check.append(image_stats(prediction, s['targets']['depth_m'], s['targets']['depth_valid'])['log_ratio']
                         - rows[i]['real']['log_ratio'])
        results[label] = {'head': info, 'summary': summarize(rows, versions), 'slopes': None if args.no_gain_sweeps else sweep_slopes(rows),
                          'native475_vs_512_real_log_ratio_diff': {'mean': float(np.mean(check)),
                                                                   'max_abs': float(np.max(np.abs(check)))},
                          'per_condition': rows}
        print(label, 'done', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    meta = {'matrix': str(matrix), 'condition_ids': ids, 'c1_reference_sample_ids': references, 'versions': versions,
            'gains': GAINS, 'gain_bases': GAIN_BASES,
            'conventions': 'log_ratio = log(median_pred/median_gt), positive = predicts too far; aligned AbsRel '
                           'after per-image median scaling; 512 generation-stage geometry, trainer preprocessing',
            'labels': 'source_validation only; no C1/C2 labels or C2 RGB'}
    if args.registry or args.no_gain_sweeps:      # additive provenance only for the non-default modes
        meta.update(registry=args.registry, gain_sweeps=not args.no_gain_sweeps)
    (out / 'results.json').write_text(json.dumps({'meta': meta, 'results': results}, indent=1, allow_nan=False) + '\n')
    shown = ['real', 'source_correct', 'target_correct', 'reinhard_real', 'a_strong_real']
    if args.no_gain_sweeps:
        table = format_tables(results, shown, with_slopes=False)
    else:
        table = format_tables(results, shown)
        sweep = [version_name(b, g) for b in GAIN_BASES for g in GAINS if g != 1.]
        table += '\n## Gain sweeps (multiply then clip)\n\n' + format_tables(results, sweep, with_slopes=False)
    (out / 'table.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
