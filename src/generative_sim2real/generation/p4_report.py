"""Pure metric and decision logic for the P4 source-controller gate and appearance checks.

No model imports (torch + numpy + PIL only), so the pre-declared rules in configs/p4.yaml are
applied by code that fixtures can test and that runs in any of the project environments.

Proxy images are scored by the frozen official DA3Metric head (generation/p4_proxy.py, DA3
venv). Scores are scale-free: the prediction is median-scaled to the reference depth over the
reference's valid pixels, so camera focal length does not matter.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch
import yaml


def median_scaled_abs_rel(prediction, reference, valid):
    """Mean |s*p - g| / g over valid pixels with s = median(g) / median(p). Lower is better."""
    p, g = prediction[valid].double(), reference[valid].double()
    if p.numel() == 0 or not torch.isfinite(p).all() or (p <= 0).any() or (g <= 0).any():
        raise ValueError('Proxy scoring needs positive finite prediction and reference depth')
    scale = g.median() / p.median()
    return ((p * scale - g).abs() / g).mean().item()


def _ranks(values):
    order = values.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(values), dtype=torch.float64)
    return ranks


def spearman(prediction, reference, valid, *, stride=4):
    """Spearman rank correlation of depth on a stride-subsampled grid (higher is better)."""
    p = prediction[..., ::stride, ::stride]
    g = reference[..., ::stride, ::stride]
    m = valid[..., ::stride, ::stride]
    a, b = _ranks(p[m].double()), _ranks(g[m].double())
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def direction_check(gains, *, min_conditions, require_positive_mean=True):
    """gains[i] > 0 means the expected direction held for condition i."""
    better = sum(g > 0 for g in gains)
    average = mean(gains)
    passed = better >= min_conditions and (average > 0 or not require_positive_mean)
    return {'conditions_in_expected_direction': better, 'conditions': len(gains),
            'mean_gain': average, 'gains': list(gains), 'passed': bool(passed)}


def condition_means(rows, arm, key, count):
    """Mean over the paired seeds per condition for one arm and metric key."""
    values = [[r[key] for r in rows if r['arm'] == arm and r['cond'] == i] for i in range(count)]
    if any(not v for v in values):
        raise ValueError(f'Missing proxy rows for arm {arm!r}')
    return [mean(v) for v in values]


def intervention_checks(rows, partner, rules, *, metric='abs_rel'):
    """Apply R0-R3 of configs/p4.yaml. Error rows carry `<metric>_own` (score against the
    reference depth of the condition the image is named for) and `<metric>_partner` (against the
    partner condition's depth). Lower error is better (for spearman pass sign=-1 by negation)."""
    count = len(partner)
    need = rules['min_conditions']
    pos = rules['require_positive_mean_gain']
    own, other = f'{metric}_own', f'{metric}_partner'
    real_own = condition_means(rows, 'real', own, count)
    real_partner = condition_means(rows, 'real', other, count)
    correct = condition_means(rows, 'correct', own, count)
    off = condition_means(rows, 'off', own, count)
    wrong_own = condition_means(rows, 'wrong', own, count)
    wrong_partner = condition_means(rows, 'wrong', other, count)
    checks = {
        'R0_proxy_audit': direction_check([b - a for a, b in zip(real_own, real_partner)],
                                          min_conditions=need, require_positive_mean=pos),
        'R1_correct_beats_off': direction_check([b - a for a, b in zip(correct, off)],
                                                min_conditions=need, require_positive_mean=pos),
        'R2_correct_beats_wrong': direction_check([b - a for a, b in zip(correct, wrong_own)],
                                                  min_conditions=need, require_positive_mean=pos),
        'R3_wrong_tracks_supplied': direction_check([b - a for a, b in zip(wrong_partner, wrong_own)],
                                                    min_conditions=need, require_positive_mean=pos),
    }
    # `wrong` image i was generated from partner(i)'s depth, so the supplied depth is the
    # partner's: rows for arm 'wrong' therefore store `_own` = score vs ITS ORIGINAL condition
    # depth and `_partner` = score vs the SUPPLIED (partner) depth. R2 compares C and W against
    # the original depth; R3 asks whether W is closer to the supplied depth than to the original.
    interventions = all(checks[k]['passed'] for k in ('R1_correct_beats_off',
                                                      'R2_correct_beats_wrong',
                                                      'R3_wrong_tracks_supplied'))
    return {'metric': metric, 'checks': checks, 'proxy_discriminative': checks['R0_proxy_audit']['passed'],
            'interventions_all_hold': interventions,
            'proxy_verdict': ('UNINTERPRETABLE (proxy fails its real-RGB audit)'
                              if not checks['R0_proxy_audit']['passed']
                              else 'PASS' if interventions else 'FAIL')}


def overall_verdict(proxy_verdict, visual):
    """Combine the proxy result with the reviewer's visual rubric (configs/p4.yaml `overall`)."""
    if proxy_verdict.startswith('UNINTERPRETABLE'):
        return 'AMBIGUOUS'
    if visual is None:
        return 'PENDING_VISUAL_RUBRIC'
    if proxy_verdict == 'PASS' and visual:
        return 'PASS'
    if proxy_verdict == 'FAIL' and not visual:
        return 'FAIL'
    return 'AMBIGUOUS'


def wasserstein_1d(a, b, *, quantiles=1000):
    """W1 between two 1-D empirical distributions: mean |Qa(u) - Qb(u)| over a uniform grid."""
    u = (torch.arange(quantiles, dtype=torch.float64) + .5) / quantiles
    qa = torch.quantile(_subsample(a), u)
    qb = torch.quantile(_subsample(b), u)
    return (qa - qb).abs().mean().item()


def _subsample(values, limit=2_000_000):
    values = values.reshape(-1).double()
    if len(values) > limit:  # torch.quantile's input limit; deterministic even stride
        values = values[::len(values) // limit + 1]
    return values


def rgb_w1(a, b, *, quantiles=1000):
    """Mean over R,G,B of the 1-D W1 between pooled pixel intensities (images [N,3,H,W] in [0,1])."""
    return mean(wasserstein_1d(a[:, c], b[:, c], quantiles=quantiles) for c in range(3))


def pairwise_diversity(images, *, size=64):
    """Mean pairwise mean-absolute pixel difference between images (64x64 downsample)."""
    small = torch.nn.functional.adaptive_avg_pool2d(images.float(), size).flatten(1)
    count = len(small)
    if count < 2:
        raise ValueError('Diversity needs at least two images')
    total = sum((small[i] - small[j]).abs().mean().item()
                for i in range(count) for j in range(i + 1, count))
    return total / (count * (count - 1) / 2)


def appearance_checks(stats):
    """A1/A2 of configs/p4.yaml from the four W1 distances."""
    a1 = stats['target_vs_c1'] < stats['source_vs_c1']
    a2 = stats['source_vs_source'] < stats['target_vs_source']
    return {'A1_target_closer_to_c1_than_source_lora': bool(a1),
            'A2_source_closer_to_source_than_target_lora': bool(a2),
            'pixel_statistics_support_appearance_difference': bool(a1 and a2)}


def png_tensor(path):
    from PIL import Image

    import numpy as np

    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(
        2, 0, 1).float() / 255


def save_png(tensor, path):
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.detach().cpu().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path)


def visual_verdict(rubric, count=8, need=6):
    """configs/p4.yaml visual_rubric.pass: layout_follows_control and wrong_depth_moves_layout
    are 'yes' in >= 6 of 8 conditions and the images are not degenerate."""
    for key in ('layout_follows_control', 'wrong_depth_moves_layout'):
        if len(rubric[key]) != count or not set(rubric[key]) <= {'yes', 'partly', 'no'}:
            raise ValueError(f'Rubric needs {count} yes/partly/no ratings for {key}')
    return bool(sum(r == 'yes' for r in rubric['layout_follows_control']) >= need
                and sum(r == 'yes' for r in rubric['wrong_depth_moves_layout']) >= need
                and rubric['not_degenerate'] is True)


def make_gate_report(render_dir, proxy_path, config, *, visual=None, out=None, rubric=None):
    """Combine the render manifest, the proxy scores and the config rules into gate.json."""
    render_dir = Path(render_dir)
    render = json.loads((render_dir / 'render.json').read_text())
    proxy = json.loads(Path(proxy_path).read_text())
    rules = config['gate']['rules']
    partner = render['partner']
    primary = intervention_checks(proxy['rows'], partner, rules, metric='abs_rel')
    secondary = intervention_checks(
        [{**r, 'spearman_own': -r['spearman_own'], 'spearman_partner': -r['spearman_partner']}
         for r in proxy['rows']], partner, rules, metric='spearman')
    report = {
        'render': {k: render[k] for k in ('set', 'step', 'controller', 'source_lora', 'ids',
                                          'partner', 'seeds', 'control_scale')},
        'proxy_model': proxy['model'],
        'primary_abs_rel': primary, 'secondary_spearman_negated': secondary,
        'visual_rubric': {'passed': visual, 'ratings': rubric},
        'verdict': overall_verdict(primary['proxy_verdict'], visual)}
    if out:
        Path(out).write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-dir', required=True)
    parser.add_argument('--config', default='configs/p4.yaml')
    parser.add_argument('--rubric', help='JSON of reviewer ratings (see visual_verdict)')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    directory = Path(args.render_dir)
    rubric = json.loads(Path(args.rubric).read_text()) if args.rubric else None
    visual = None if rubric is None else visual_verdict(rubric)
    report = make_gate_report(directory, directory / 'proxy.json', config, visual=visual,
                              out=directory / 'gate.json', rubric=rubric)
    print(json.dumps({k: report[k] for k in ('verdict',)}, indent=1))
    for key in ('primary_abs_rel', 'secondary_spearman_negated'):
        print(key, report[key]['proxy_verdict'],
              {k: (v['conditions_in_expected_direction'], round(v['mean_gain'], 4))
               for k, v in report[key]['checks'].items()})


if __name__ == '__main__':
    main()
