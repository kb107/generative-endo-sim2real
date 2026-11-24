"""Label-free (target-side) scale anatomy of every DA3 head of a frozen C2 campaign registry.

Hypothesis: a head's C2 metric-scale bias = t_h + kappa_h * ln(300/241.502), with t_h the target-look term (the
look->scale probe, mean ln(median_pred/median_gt) of the target_correct renders) and kappa_h the camera-convention
term (how the prediction moves with the pixel scale of the processed image). Only SOURCE labels are used here
(source validation for kappa, source train for c_X); no C1/C2 labels, no C2 pixels, no campaign results.

Sign convention: bias = mean ln(pred/gt); positive = predicts too far.

Subcommands (all resumable; one JSON per head / per (image set, head) under the output directory):
  kappa     source-validation process-size sweep + centre-crop secondary for every registered head
  cx        c_X scalars: mean per-image ln(median pred/median gt) on the downstream IDs under each image set
  assemble  combine kappa + probe (t_h) + c_X into predictions.json / predictions.md / predictions.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = 'outputs/analysis/scale_anatomy'
REGISTRY = 'outputs/c2_campaign/registry_c5.json'
PROBE = 'outputs/analysis/look_scale_probe_c5/results.json'
SWEEP_R = (406, 448, 504, 560, 630)
BASE_R = 504
TRAIN_FOCAL = 241.502                       # 227.604 px at 475 -> 504
CANONICAL_FOCAL = 300.0
CROP_SIDE = 382
BLOCK = 10
BOOTSTRAPS = 1000
SECONDARY_KAPPA = 0.18                      # fixed secondary prediction: t_h + 0.18
LN_C2 = math.log(CANONICAL_FOCAL / TRAIN_FOCAL)
F406 = CANONICAL_FOCAL * (406 / 504 + 504 / 630) / 2
IMAGE_SETS = {
    'G': 'outputs/generated/a9743a4845dca5be/generated_manifest.jsonl',
    'Gp': 'outputs/generated/e8413cc36b857c2d/generated_manifest.jsonl',
    'S': 'outputs/generated/316eecddb37769b5/generated_manifest.jsonl',
    'T': 'outputs/generated_variants/T_reinhard/generated_manifest.jsonl',
    'A_strong': 'outputs/generated_variants/A_strong/generated_manifest.jsonl',
    'raw': None,
}
CX_HEADS = {'R': 'R', 'R.last': 'R.last', 'R_s1': 'R_s1', 'R_s1.last': 'R_s1.last', 'G': 'G'}   # name -> registry name
DIAGNOSTIC_HEADS = ('R', 'R_s1', 'G', 'G_s1')
DOWNSTREAM_IDS = 'outputs/manifests/milestone_v1/downstream_ids.json'
SOURCE_TRAIN = 'outputs/manifests/milestone_v1/source_train.jsonl'


# ---------------------------------------------------------------- pure statistics

def ols_slope(x, y):
    """OLS slope of y on x; y may be [M] or [n, M] (slope along the last axis)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return (y - y.mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)


def block_ids(sequence_ids, frame_order, size=BLOCK):
    """Block index per frame: ``size`` consecutive frames (by frame order) within each sequence; none span sequences."""
    ids = np.zeros(len(sequence_ids), int)
    next_block = 0
    for sequence in dict.fromkeys(sequence_ids):
        members = [i for i, s in enumerate(sequence_ids) if s == sequence]
        members.sort(key=lambda i: frame_order[i])
        for position, i in enumerate(members):
            ids[i] = next_block + position // size
        next_block += math.ceil(len(members) / size)
    return ids


def bootstrap_means(values, weights, blocks, n=BOOTSTRAPS, seed=0):
    """Block bootstrap of weighted means. values [M, N] (nonfinite entries need weight 0), weights [N], blocks [N].

    Whole blocks are resampled with replacement (as many as exist); returns [n, M] resampled means.
    """
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    values = np.where(weights > 0, values, 0.)
    count = int(blocks.max()) + 1
    sums = np.zeros((count, len(values)))
    mass = np.zeros(count)
    for b in range(count):
        member = blocks == b
        sums[b] = (values[:, member] * weights[member]).sum(axis=1)
        mass[b] = weights[member].sum()
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, count, size=(n, count))
    return sums[draw].sum(axis=1) / mass[draw].sum(axis=1)[:, None]


def interval(samples, level=.95):
    lo, hi = np.percentile(samples, [50 * (1 - level), 100 - 50 * (1 - level)])
    return [float(lo), float(hi)]


def kappa_and_crop(b_sweep, b_crop, b_region, weights, blocks, radii=SWEEP_R, crop_log_focal=None, n=BOOTSTRAPS, seed=0):
    """kappa = OLS slope of mean b against ln(r/504); kappa_crop = (b_crop - b_504)/ln(f_crop/f_train).

    b_sweep [len(radii), N]; b_crop [N] (crop evaluated at 504); b_region [N] (the 504 prediction scored on the
    crop region only, the pixel-matched control). Block-bootstrap CIs use identical resamples for every statistic.
    """
    x = np.log(np.asarray(radii, float) / BASE_R)
    base = list(radii).index(BASE_R)
    stacked = np.vstack([b_sweep, b_crop[None], b_region[None]])
    point = (stacked * weights).sum(axis=1) / weights.sum()
    boot = bootstrap_means(stacked, weights, blocks, n, seed)
    k = len(radii)

    def derive(means):
        out = {'kappa': ols_slope(x, means[..., :k]), 'mean_b_504': means[..., base]}
        if crop_log_focal is not None:
            out['kappa_crop'] = (means[..., k] - means[..., base]) / crop_log_focal
            out['kappa_crop_region_matched'] = (means[..., k] - means[..., k + 1]) / crop_log_focal
        return out
    est, dist = derive(point), derive(boot)
    return {name: {'estimate': float(est[name]), 'ci95': interval(dist[name]), 'bootstrap_sd': float(dist[name].std(ddof=1))}
            for name in est}, {'mean_b': {str(r): float(point[i]) for i, r in enumerate(radii)},
                               'mean_b_crop': float(point[k]), 'mean_b_504_on_crop_region': float(point[k + 1])}


def crop_sample(rgb, depth, valid, K, side=CROP_SIDE):
    """Centre-crop RGB/labels identically (integer origin, no interpolation) and shift the principal point."""
    h, w = rgb.shape[-2:]
    top, left = (h - side) // 2, (w - side) // 2
    cut = lambda v: v[..., top:top + side, left:left + side].clone()
    K = K.clone()
    K[0, 2] -= left
    K[1, 2] -= top
    return cut(rgb), cut(depth), cut(valid), K, (top, left)


def log_ratio(prediction, target, valid):
    """ln(median_pred/median_gt) over valid pixels (lower median in double, as the evaluator) and AbsRel; None if empty."""
    valid = valid & (target > 0)
    if not bool(valid.any()):
        return None, None, 0
    p, g = prediction[valid].double(), target[valid].double()
    if not bool(torch.isfinite(p).all()) or bool((p <= 0).any()):
        raise ValueError('Nonfinite or nonpositive prediction at a valid pixel')
    return (math.log(p.median().item() / g.median().item()), ((p - g).abs() / g).mean().item(), int(valid.sum()))


# ---------------------------------------------------------------- predictions (pure)

def predict_bias(t, kappa, log_focal_ratio=LN_C2):
    """Predicted mean ln(pred/gt) on C2: t + kappa * ln(f_c2_equivalent / f_train)."""
    return t + kappa * log_focal_ratio


def secondary_bias(t):
    return t + SECONDARY_KAPPA


def arm_predictions(t, kappa, c_g=None):
    """Campaign #6A diagnostic arms for one head: <h>.406, <h>.cG, <h>.cG.406 (the last two need c_G(head))."""
    ln406 = math.log(F406 / TRAIN_FOCAL)
    out = {'406': predict_bias(t, kappa, ln406)}
    if c_g is not None:
        out['cG'] = predict_bias(t, kappa) - c_g
        out['cG.406'] = t - c_g + kappa * ln406
    return out


# ---------------------------------------------------------------- IO helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, allow_nan=False, sort_keys=True) + '\n')


def head_key(record):
    head = record.get('head') or {}
    lora = (record.get('backbone_lora') or {}).get('sha256')
    return (head.get('head_sha256'), lora) if head else ('official', None)


def load_model(device):
    from ..depth.model import DA3MetricDepth, verify_upstream_checkout
    verify_upstream_checkout()
    return DA3MetricDepth.from_pretrained().to(device).eval().requires_grad_(False)


def load_states(model, registry):
    from ..evaluation.c2 import prepare_heads
    return prepare_heads(model, registry, '.')


# ---------------------------------------------------------------- kappa

@torch.no_grad()
def run_head_kappa(model, frames, device):
    """Per-frame b at every radius, the crop at 504 and the crop-region-matched 504 control for the current head."""
    n = len(frames)
    b = np.full((len(SWEEP_R), n), np.nan)
    abs_rel_504 = np.full(n, np.nan)
    b_crop, b_region = np.full(n, np.nan), np.full(n, np.nan)
    crop_weight = np.zeros(n)
    for j, fr in enumerate(frames):
        rgb, depth, valid, K = fr['rgb'], fr['depth'], fr['valid'], fr['K']
        for i, r in enumerate(SWEEP_R):
            model.process_hw = (r, r)            # wrapper resizes RGB bicubically and scales K with the image
            out = model(rgb[None], K[None])['depth_m']
            pred = F.interpolate(out, size=depth.shape[-2:], mode='bilinear', align_corners=False)[0]
            lr, ar, _ = log_ratio(pred, depth, valid)
            b[i, j] = lr
            if r == BASE_R:
                abs_rel_504[j] = ar
                full_pred = pred
        c_rgb, c_depth, c_valid, c_K, (top, left) = crop_sample(rgb, depth, valid, K)
        lr, _, count = log_ratio(full_pred[..., top:top + CROP_SIDE, left:left + CROP_SIDE], c_depth, c_valid)
        if count:
            b_region[j] = lr
            model.process_hw = (BASE_R, BASE_R)
            out = model(c_rgb[None], c_K[None])['depth_m']
            pred = F.interpolate(out, size=c_depth.shape[-2:], mode='bilinear', align_corners=False)[0]
            b_crop[j] = log_ratio(pred, c_depth, c_valid)[0]
            crop_weight[j] = 1.
    model.process_hw = (BASE_R, BASE_R)
    return b, b_crop, b_region, crop_weight, abs_rel_504


def load_source_validation(device, limit=None):
    from ..data.manifests import ManifestDataset
    from ..variants import build as vb
    dataset = ManifestDataset(vb.SOURCE_VALIDATION, os.environ['DATA_ROOT'], purpose='training', geometry='source_native')
    count = len(dataset) if limit is None else min(limit, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count).round().astype(int).tolist() if limit else list(range(count))
    frames, seqs, order = [], [], []
    for i in indices:
        s = dataset[i]
        frames.append({'rgb': s['rgb'].to(device), 'depth': s['targets']['depth_m'].to(device),
                       'valid': s['targets']['depth_valid'].to(device), 'K': s['camera']['K'].to(device)})
        seqs.append(s['meta']['sequence_id'])
        order.append(dataset.rows[i]['frame_index'])
    return frames, seqs, order, [dataset.rows[i]['sample_id'] for i in indices]


def cmd_kappa(args):
    from ..depth.checkpoints import apply_checkpoint
    out = Path(args.out)
    registry = json.loads(Path(args.registry).read_text())
    model = load_model(args.device)
    states = load_states(model, registry)
    frames, seqs, order, ids = load_source_validation(args.device, args.frames)
    blocks = block_ids(seqs, order)
    crop_f = frames[0]['K'][0, 0].item() * BASE_R / CROP_SIDE
    crop_log_focal = math.log(crop_f / TRAIN_FOCAL)
    print(f'{len(frames)} frames, {blocks.max() + 1} blocks, crop focal {crop_f:.3f} px', flush=True)
    records = {r['name']: r for r in registry['methods']}
    done = {}
    for name in registry['method_names']:
        if args.only and name not in args.only:
            continue
        target = out / 'kappa' / f'{name}.json'
        if target.exists() and not args.force:
            continue
        key = head_key(records[name])
        if key in done:                                     # identical head (+LoRA): same numbers, recorded as such
            payload = dict(json.loads((out / 'kappa' / f'{done[key]}.json').read_text()), name=name, same_as=done[key])
            write_json(target, payload)
            print(name, 'same head as', done[key], flush=True)
            continue
        apply_checkpoint(model, states[name])
        b, b_crop, b_region, weight, abs_rel = run_head_kappa(model, frames, args.device)
        stats, means = kappa_and_crop(b, b_crop, b_region, weight, blocks, crop_log_focal=crop_log_focal)
        rec = records[name]
        seq_macro = float(np.mean([abs_rel[[i for i, s in enumerate(seqs) if s == q]].mean() for q in dict.fromkeys(seqs)]))
        payload = {'name': name, 'same_as': None, 'head_key': list(key) if key != ('official', None) else ['official'],
                   'n_frames': len(frames), 'n_blocks': int(blocks.max()) + 1, 'radii': list(SWEEP_R),
                   'crop_focal_px': crop_f, 'crop_log_focal_ratio': crop_log_focal, 'train_focal_px': TRAIN_FOCAL,
                   'stats': stats, 'means': means, 'abs_rel_504_sequence_macro': seq_macro,
                   'registry_source_validation_abs_rel': rec.get('source_validation_sequence_macro_abs_rel'),
                   'per_frame': {'sample_ids': ids, 'sequence_ids': seqs, 'b': b.tolist(), 'b_crop': b_crop.tolist(),
                                 'b_504_on_crop_region': b_region.tolist()}}
        if args.frames is None:
            done[key] = name
        write_json(target, payload)
        print(name, f"kappa={stats['kappa']['estimate']:+.4f} ci={stats['kappa']['ci95']} "
              f"crop={stats['kappa_crop']['estimate']:+.4f} regionmatched={stats['kappa_crop_region_matched']['estimate']:+.4f} "
              f"absrel504={seq_macro:.5f} (registry {rec.get('source_validation_sequence_macro_abs_rel')})", flush=True)
    return 0


# ---------------------------------------------------------------- c_X

def sequence_bootstrap(values, sequences, n=BOOTSTRAPS, seed=0):
    """Mean and 95% CI of per-image values, resampling whole source sequences with replacement."""
    values = np.asarray(values, float)
    groups = [np.flatnonzero(np.asarray(sequences) == s) for s in dict.fromkeys(sequences)]
    sums = np.array([values[g].sum() for g in groups])
    counts = np.array([len(g) for g in groups], float)
    draw = np.random.default_rng(seed).integers(0, len(groups), size=(n, len(groups)))
    boot = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return {'mean': float(values.mean()), 'ci95': interval(boot), 'n_images': int(len(values)), 'n_sequences': len(groups)}


@torch.no_grad()
def cmd_cx(args):
    from ..data.manifests import ManifestDataset
    from ..data.rgb_modes import RGBModeDataset
    from ..data.workers import configure_worker_cleanup
    from ..depth.checkpoints import apply_checkpoint
    out = Path(args.out)
    registry = json.loads(Path(args.registry).read_text())
    model = load_model(args.device)
    model.process_hw = (BASE_R, BASE_R)
    states = load_states(model, registry)
    heads = {n: states[r] for n, r in CX_HEADS.items() if not args.heads or n in args.heads}
    # Head weights stay on the device; per sample the active head is switched in place (cheap, no backbone change).
    device_states = {n: {k: v.to(args.device) for k, v in s['head_state_dict'].items()} for n, s in heads.items()}
    source = ManifestDataset(SOURCE_TRAIN, os.environ['DATA_ROOT'], purpose='training', geometry='source_generation512')
    ids = json.loads(Path(DOWNSTREAM_IDS).read_text())['sample_ids']
    if args.limit:
        ids = ids[:args.limit]
    lookup = {r['sample_id']: r for r in source.rows}
    sequences = [lookup[i]['sequence_id'] for i in ids]
    for set_name in args.sets or IMAGE_SETS:
        manifest = IMAGE_SETS[set_name]
        pending = [n for n in heads if args.force or not (out / 'cx' / f'{set_name}__{n}.json').exists()]
        if not pending:
            continue
        dataset = RGBModeDataset(source, ids, mode='R' if manifest is None else 'G', seed=0,
                                 generated_manifest=manifest, outputs_root='outputs')
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.workers,
                                             worker_init_fn=configure_worker_cleanup if args.workers else None)
        values = {n: [] for n in pending}
        for position, sample in enumerate(loader):
            if sample['sample_id'] != ids[position]:
                raise ValueError('Dataset order differs from the downstream ID list')
            rgb, K = sample['rgb'][None].to(args.device), sample['camera']['K'][None].to(args.device)
            depth, valid = sample['targets']['depth_m'].to(args.device), sample['targets']['depth_valid'].to(args.device)
            if tuple(rgb.shape[-2:]) != (BASE_R, BASE_R):
                raise ValueError('Expected the 504x504 training view')
            for n in pending:
                model.load_head_state_dict(device_states[n])
                lr = log_ratio(model(rgb, K)['depth_m'][0], depth, valid)[0]
                values[n].append(lr)
            if (position + 1) % 250 == 0:
                print(set_name, position + 1, '/', len(ids), flush=True)
        for n in pending:
            stats = sequence_bootstrap(values[n], sequences)
            write_json(out / 'cx' / f'{set_name}__{n}.json',
                       {'set': set_name, 'head': n, 'registry_name': CX_HEADS[n], 'manifest': manifest,
                        'manifest_sha256': None if manifest is None else sha256_file(manifest), **stats,
                        'sample_ids': ids, 'sequence_ids': sequences, 'values': values[n]})
            print(set_name, n, f"c_X={stats['mean']:+.4f} ci={stats['ci95']}", flush=True)
    return 0


# ---------------------------------------------------------------- assemble

def read_optional(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def build_predictions(registry, kappa, probe, cx, input_hashes):
    """Pure assembly: kappa[name], probe['results'][name], cx[(set, head)] -> the frozen predictions dict."""
    entries, absent = {}, []
    for name in registry['method_names']:
        record = {r['name']: r for r in registry['methods']}[name]
        k, p = kappa.get(name), probe.get(name) if probe else None
        if k is None or p is None:
            absent.append({'name': name, 'kappa': k is None, 'probe': p is None})
            continue
        t = p['summary']['target_correct']['log_ratio_mean']
        kap = k['stats']['kappa']['estimate']
        entries[name] = {
            't_h': t, 'kappa_h': kap, 'kappa_h_ci95': k['stats']['kappa']['ci95'],
            'kappa_crop': k['stats'].get('kappa_crop', {}).get('estimate'),
            'kappa_crop_ci95': k['stats'].get('kappa_crop', {}).get('ci95'),
            'kappa_crop_region_matched': k['stats'].get('kappa_crop_region_matched', {}).get('estimate'),
            'primary_predicted_bias': predict_bias(t, kap), 'secondary_predicted_bias': secondary_bias(t),
            'baseline_source_validation_abs_rel': record.get('source_validation_sequence_macro_abs_rel')
            if record.get('source_validation_sequence_macro_abs_rel') is not None else k.get('registry_source_validation_abs_rel'),
            'source_validation_abs_rel_recomputed_504': k.get('abs_rel_504_sequence_macro'),
            'baseline_probe_real_mean': p['summary']['real']['log_ratio_mean'],
            'head_file_sha256': (record.get('head') or {}).get('file_sha256'),
            'head_sha256': (record.get('head') or {}).get('head_sha256'),
            'backbone_lora': record.get('backbone_lora') is not None}
    arms = {}
    for h in DIAGNOSTIC_HEADS:
        e = entries.get(h)
        if e is None:
            continue
        c_g = cx.get(('G', h))
        pieces = arm_predictions(e['t_h'], e['kappa_h'], None if c_g is None else c_g['mean'])
        arms[f'{h}.406'] = pieces['406']
        if h in ('R', 'R_s1'):
            if 'cG' in pieces:
                arms[f'{h}.cG'], arms[f'{h}.cG.406'] = pieces['cG'], pieces['cG.406']
            else:
                absent.append({'name': f'{h}.cG', 'c_G': True})
    return {'schema_version': 1,
            'convention': 'bias = mean ln(pred/gt); positive = predicts too far (C2 "mean log s" in earlier docs is minus this)',
            'formulas': {'primary': 't_h + kappa_h * ln(300/241.502)', 'secondary': f't_h + {SECONDARY_KAPPA}',
                         '<h>.406': 't_h + kappa_h * ln(f406/241.502), f406 = 300*mean(406/504, 504/630)',
                         '<h>.cG': 't_h + kappa_h * ln(300/241.502) - c_G(h)',
                         '<h>.cG.406': 't_h - c_G(h) + kappa_h * ln(f406/241.502)'},
            'constants': {'train_focal_px': TRAIN_FOCAL, 'canonical_focal_px': CANONICAL_FOCAL, 'ln_300_over_train': LN_C2,
                          'f406_px': F406, 'ln_f406_over_train': math.log(F406 / TRAIN_FOCAL),
                          'sweep_radii': list(SWEEP_R), 'block': BLOCK, 'bootstraps': BOOTSTRAPS, 'seed': 0},
            'heads': entries, 'arms_6A': arms,
            'c_X': {f'{s}|{h}': {k: v[k] for k in ('mean', 'ci95', 'n_images', 'n_sequences')} for (s, h), v in sorted(cx.items())},
            'absent': absent, 'inputs': input_hashes}


def format_markdown(pred):
    f = lambda v, d=4: 'n/a' if v is None else f'{v:+.{d}f}'
    lines = ['# Scale anatomy: blind predictions of C2 metric-scale bias', '',
             pred['convention'] + '.', '',
             '| head | t_h | kappa_h [95% CI] | kappa_crop | kappa_crop (region-matched) | primary t+k*ln(300/241.502) | secondary t+0.18 | src-val AbsRel | probe real |',
             '|---|---|---|---|---|---|---|---|---|']
    for name, e in pred['heads'].items():
        ci = e['kappa_h_ci95']
        ab = e['baseline_source_validation_abs_rel']
        lines.append(f"| {name} | {f(e['t_h'])} | {f(e['kappa_h'], 3)} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {f(e['kappa_crop'], 3)} | "
                     f"{f(e['kappa_crop_region_matched'], 3)} | {f(e['primary_predicted_bias'])} | {f(e['secondary_predicted_bias'])} | "
                     f"{'n/a' if ab is None else f'{ab:.4f}'} | {f(e['baseline_probe_real_mean'])} |")
    lines += ['', '## Campaign 6A diagnostic arms (predicted bias)', '', '| arm | predicted bias |', '|---|---|']
    lines += [f'| {a} | {f(v)} |' for a, v in pred['arms_6A'].items()]
    lines += ['', '## c_X (mean per-image ln(median pred/median gt) on the 2,000 downstream IDs)', '',
              '| image set | head | c_X | 95% CI (sequence bootstrap) |', '|---|---|---|---|']
    lines += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {f(v['mean'])} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |"
              for k, v in pred['c_X'].items()]
    if pred['absent']:
        lines += ['', 'Absent: ' + json.dumps(pred['absent'])]
    return '\n'.join(lines) + '\n'


def cmd_assemble(args):
    out = Path(args.out)
    registry = json.loads(Path(args.registry).read_text())
    kappa = {n: read_optional(out / 'kappa' / f'{n}.json') for n in registry['method_names']}
    kappa = {n: v for n, v in kappa.items() if v}
    probe_doc = read_optional(args.probe)
    probe = probe_doc['results'] if probe_doc else {}
    cx = {}
    for path in sorted((out / 'cx').glob('*.json')) if (out / 'cx').exists() else []:
        v = json.loads(path.read_text())
        cx[(v['set'], v['head'])] = v
    hashes = {'registry': {'path': args.registry, 'sha256': sha256_file(args.registry)},
              'probe_results': {'path': args.probe, 'sha256': sha256_file(args.probe)} if probe_doc else None,
              'kappa_files': {n: sha256_file(out / 'kappa' / f'{n}.json') for n in sorted(kappa)},
              'cx_files': {f'{s}__{h}': sha256_file(out / 'cx' / f'{s}__{h}.json') for (s, h) in sorted(cx)},
              'cx_manifests': {s: {'path': m, 'sha256': sha256_file(m) if m else None} for s, m in IMAGE_SETS.items()},
              'heads': {r['name']: {'path': r['head']['path'], 'file_sha256': r['head']['file_sha256'],
                                    'head_sha256': r['head']['head_sha256']} for r in registry['methods'] if r.get('head')},
              'code': {p: sha256_file(p) for p in ('src/generative_sim2real/analysis/scale_anatomy.py',
                                                   'src/generative_sim2real/analysis/look_scale.py')},
              'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()}
    pred = build_predictions(registry, kappa, probe, cx, hashes)
    write_json(out / 'predictions.json', pred)
    (out / 'predictions.md').write_text(format_markdown(pred))
    digest = sha256_file(out / 'predictions.json')
    (out / 'predictions.sha256').write_text(f'{digest}  predictions.json\n')
    print((out / 'predictions.md').read_text())
    print('predictions.json sha256', digest)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('kappa', 'cx', 'assemble'))
    parser.add_argument('--out', default=OUT)
    parser.add_argument('--registry', default=REGISTRY)
    parser.add_argument('--probe', default=PROBE)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--only', nargs='*', help='kappa: restrict to these registry names')
    parser.add_argument('--frames', type=int, help='kappa: evenly spaced subset of validation frames (smoke only)')
    parser.add_argument('--limit', type=int, help='cx: first N downstream IDs (smoke only)')
    parser.add_argument('--sets', nargs='*', help='cx: image sets (default all)')
    parser.add_argument('--heads', nargs='*', help='cx: head names (default all)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    return {'kappa': cmd_kappa, 'cx': cmd_cx, 'assemble': cmd_assemble}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
