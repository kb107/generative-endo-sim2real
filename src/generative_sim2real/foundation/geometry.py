"""Geometry-fidelity metrics for an image set paired with its source SimCol frames and depth.

(a) registration vs the source RGB, on gradient magnitude (largely appearance-invariant):
    ECC affine -> translation of the image centre (px) and isotropic scale (>1 = zoomed in),
    phase-correlation translation (px), and median / p90 Farneback optical-flow magnitude (px).
(b) depth-structure fidelity: a frozen pretrained DA3-Metric (no fine-tuning; the same proxy as
    generation/p4_proxy.py, scored K-independently) is run on the image; its prediction is scored
    against the conditioning source depth with median-scaled AbsRel (lower is better) and Spearman
    rank correlation (higher is better) over valid pixels. The same score on the real SimCol frame is
    the ceiling. This measures whether geometry a depth network can read survives; it does not
    prove pixel-exact geometry.

Registration needs only numpy+OpenCV. The depth proxy is run in the DA3 venv:
  PYTHONPATH=src external/da3-venv/bin/python -m generative_sim2real.foundation.geometry --set fixed8 ...
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'))


def gradient_magnitude(rgb):
    """Float32 [H,W] Sobel magnitude of the blurred grey image, scaled by its 99th percentile."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    grey = cv2.GaussianBlur(grey, (0, 0), 1.0)
    mag = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    return np.clip(mag / max(float(np.percentile(mag, 99)), 1e-6), 0, 1).astype(np.float32)


def phase_shift(a, b):
    """(dx, dy) such that content of b appears at (x+dx, y+dy) relative to a; and the peak response."""
    window = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), window)
    return float(dx), float(dy), float(response)


def ecc_affine(a, b, iterations=200, eps=1e-6):
    """Affine W (2x3) with b(W(x)) ~ a(x); (centre shift dx, dy, isotropic scale, converged)."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            a, b, warp, cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps), None, 5)
    except cv2.error:
        return math.nan, math.nan, math.nan, False
    h, w = a.shape
    centre = np.array([(w - 1) / 2, (h - 1) / 2, 1.0])
    moved = warp @ centre
    scale = math.sqrt(abs(float(np.linalg.det(warp[:, :2]))))
    return float(moved[0] - centre[0]), float(moved[1] - centre[1]), scale, True


def flow_magnitude(a, b):
    """Median and p90 Farneback flow magnitude (px) from a to b (gradient-magnitude images)."""
    a8, b8 = (a * 255).astype(np.uint8), (b * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 4, 21, 3, 5, 1.1, 0)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    return float(np.median(mag)), float(np.percentile(mag, 90))


def registration(real_rgb, other_rgb):
    """All registration numbers of `other_rgb` against `real_rgb` (same size required)."""
    if real_rgb.shape != other_rgb.shape:
        raise ValueError(f'size mismatch {real_rgb.shape} vs {other_rgb.shape}')
    a, b = gradient_magnitude(real_rgb), gradient_magnitude(other_rgb)
    px, py, response = phase_shift(a, b)
    ex, ey, scale, ok = ecc_affine(a, b)
    med, p90 = flow_magnitude(a, b)
    return {'phase_dx': px, 'phase_dy': py, 'phase_shift_px': math.hypot(px, py),
            'phase_response': response, 'ecc_converged': ok, 'ecc_dx': ex, 'ecc_dy': ey,
            'ecc_shift_px': math.hypot(ex, ey) if ok else math.nan, 'ecc_scale': scale,
            'flow_median_px': med, 'flow_p90_px': p90}


def depth_scores(depth_pred, depth_ref, valid):
    """Median-scaled AbsRel and Spearman of a predicted depth map vs reference (torch tensors)."""
    from generative_sim2real.generation import p4_report as report
    valid = valid & (depth_ref > 0)
    return {'abs_rel': report.median_scaled_abs_rel(depth_pred, depth_ref, valid),
            'spearman': report.spearman(depth_pred, depth_ref, valid)}


class DepthProxy:
    """Frozen DA3-Metric-Large (pretrained, no fine-tuning): rgb PNG -> depth map [1,H,W] at ref size."""

    def __init__(self, device='cuda'):
        from generative_sim2real.depth.model import DA3MetricDepth, verify_upstream_checkout
        from generative_sim2real.depth.checkpoints import head_digest
        verify_upstream_checkout()
        self.device = device
        self.model = DA3MetricDepth.from_pretrained().to(device).eval().requires_grad_(False)
        self.digest = head_digest(self.model.head_state_dict())
        self._digest_fn = head_digest

    def unchanged(self):
        return self._digest_fn(self.model.head_state_dict()) == self.digest

    def predict(self, rgb_uint8, K, size):
        import torch
        import torch.nn.functional as F
        rgb = torch.from_numpy(rgb_uint8.copy()).permute(2, 0, 1).float()[None] / 255
        with torch.no_grad():
            depth = self.model(rgb.to(self.device), K[None].to(self.device))['depth_m']
            return F.interpolate(depth, size=size, mode='bilinear', align_corners=False)[0].cpu()


def summarize(rows, keys):
    out = {}
    for arm in sorted({r['arm'] for r in rows}):
        sel = [r for r in rows if r['arm'] == arm]
        out[arm] = {'n': len(sel)}
        for k in keys:
            vals = np.array([r[k] for r in sel if r.get(k) is not None and not math.isnan(r[k])])
            if len(vals):
                out[arm][k] = {'mean': float(vals.mean()), 'median': float(np.median(vals)), 'n': int(len(vals))}
    return out


METRICS = ('abs_rel', 'spearman', 'ecc_shift_px', 'ecc_scale', 'phase_shift_px', 'flow_median_px', 'flow_p90_px')


def evaluate(items, arms, proxy):
    """items: [{'key', 'real': path, 'depth': (depth_m[1,H,W], valid[1,H,W], K[3,3])}]
    arms: {name: {key: path}}; 'real' is scored on depth only (registration against itself is 0)."""
    rows = []
    for item in items:
        depth_ref, valid, K = item['depth']
        real = load_rgb(item['real'])
        for arm, files in arms.items():
            if item['key'] not in files:
                continue
            image = real if arm == 'real' else load_rgb(files[item['key']])
            row = {'arm': arm, 'key': item['key'], 'file': str(files[item['key']])}
            if arm != 'real':
                row.update(registration(real, image))
            if proxy is not None:
                pred = proxy.predict(image, K, depth_ref.shape[-2:])
                row.update(depth_scores(pred, depth_ref, valid))
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--set', required=True, choices=['fixed8', 'matrix32', 'subset300'])
    parser.add_argument('--inputs', default='outputs/foundation/inputs/inputs.json')
    parser.add_argument('--arm', action='append', default=[],
                        help='NAME=TEMPLATE extra arm; TEMPLATE uses {cond} (fixed8/matrix32) or {index} (subset300)')
    parser.add_argument('--only-arms', default=None, help='comma list restricting the built-in arms')
    parser.add_argument('--out', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-depth', action='store_true')
    args = parser.parse_args()
    spec = json.loads(Path(args.inputs).read_text())
    matrix_dir = Path(spec['matrix_dir'])
    if args.set in ('fixed8', 'matrix32'):
        import torch
        refs = torch.load(matrix_dir / 'refs.pt', weights_only=True)
        conds = [r['cond'] for r in spec['fixed8']] if args.set == 'fixed8' else list(range(refs['K'].shape[0]))
        items = [{'key': c, 'real': matrix_dir / f'images/real/c{c}.png',
                  'depth': (refs['depth_m'][c], refs['valid'][c], refs['K'][c])} for c in conds]
        arms = {'real': {c: matrix_dir / f'images/real/c{c}.png' for c in conds}}
        for name in ('source_correct', 'target_correct'):
            arms[name] = {c: matrix_dir / f'images/{name}/c{c}.png' for c in conds}
        templates = {'cond': conds}
    else:
        import os
        from generative_sim2real.generation.inputs import generation_dataset
        dataset = generation_dataset('outputs/manifests/milestone_v1/source_train.jsonl',
                                     os.environ['DATA_ROOT'], stage='generation')
        index = {row['sample_id']: i for i, row in enumerate(dataset.rows)}
        items, arms = [], {'real': {}}
        for name in ('ours_controllora', 'ours_fullctrl', 'simuscope'):
            arms[name] = {}
        for entry in spec['subset300']:
            sample = dataset[index[entry['sample_id']]]
            t = sample['targets']
            items.append({'key': entry['index'], 'real': entry['real'],
                          'depth': (t['depth_m'], t['depth_valid'], sample['camera']['K'])})
            for name in arms:
                arms[name][entry['index']] = entry['real'] if name == 'real' else entry[name]
        templates = {'index': [e['index'] for e in spec['subset300']]}
    if args.only_arms:
        keep = set(args.only_arms.split(','))
        arms = {k: v for k, v in arms.items() if k in keep}
    (field, keys), = templates.items()
    for extra in args.arm:
        name, template = extra.split('=', 1)
        arms[name] = {k: template.format(**{field: k}) for k in keys
                      if Path(template.format(**{field: k})).exists()}
    proxy = None if args.no_depth else DepthProxy(args.device)
    rows = evaluate(items, arms, proxy)
    result = {'set': args.set, 'arms': {a: len(f) for a, f in arms.items()},
              'depth_model': None if proxy is None else {
                  'checkpoint': 'depth-anything/DA3METRIC-LARGE (pretrained, frozen)',
                  'head_unchanged': proxy.unchanged(), 'scoring': 'median-scaled, K-independent'},
              'summary': summarize(rows, METRICS), 'rows': rows}
    out = Path(args.out or f'outputs/foundation/geometry/{args.set}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str, allow_nan=True) + '\n')
    for arm, s in result['summary'].items():
        print(arm, s['n'], {k: round(v['mean'], 4) for k, v in s.items() if k != 'n'})


if __name__ == '__main__':
    main()
