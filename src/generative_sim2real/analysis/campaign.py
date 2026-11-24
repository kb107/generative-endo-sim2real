"""Post-hoc analysis of one completed C2 campaign directory (no inference, reads saved per-frame rows only).

Applies the decision rules fixed before the campaign-5 results were read:
sigma = max(|G-Gp|, |G-G_s1|, |R-R_s1|) on sequence-macro raw AbsRel;
"X ~ ref" if |dAbsRel| < max(2 sigma, 0.03) and X beats ref on 7-11 of the families;
"ref >> X" if AbsRel(X) - AbsRel(ref) > max(3 sigma, 0.05) and ref beats X on >= 14 families (mirror for "X >> ref");
otherwise "inconclusive". A family is a C2 sequence id without its `_v1`/`_v2` suffix; its AbsRel is the mean of its
sequences' AbsRel. Also reports the per-arm scale anatomy from s = median(gt)/median(pred) per frame.
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

NOISE_PAIRS = (('G', 'Gp'), ('G', 'G_s1'), ('R', 'R_s1'))
ATTRIBUTION = ('S', 'T', 'A_strong', 'A_mild', 'R', 'M', 'G_full', 'G_simuscope')
ARCHITECTURE = (('R_bb', 'R'), ('G_bb', 'G'))


def family(sequence_id):
    return re.sub(r'_v[0-9]+$', '', sequence_id)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def load_campaign(campaign_dir):
    campaign_dir = Path(campaign_dir)
    headline = {r['method']: r for r in read_csv(campaign_dir / 'comparison_headline.csv')}
    for r in read_csv(campaign_dir / 'comparison_oracle_diagnostic.csv'):
        headline[r['method']]['oracle_rmse_mm'] = r['sequence_macro_oracle_rmse_mm']
    methods = {}
    for name, row in headline.items():
        frames = [r for r in read_csv(campaign_dir / 'methods' / name / 'frames.csv') if r['status'] == 'ok']
        seqs = read_csv(campaign_dir / 'methods' / name / 'sequences.csv')
        methods[name] = {'headline': row, 'frames': frames,
                         'sequences': {r['sequence_id']: float(r['abs_rel']) for r in seqs}}
    registry = json.loads((campaign_dir / 'registry.json').read_text())
    return methods, registry


def family_abs_rel(sequences):
    fams = {}
    for sid, value in sequences.items():
        fams.setdefault(family(sid), []).append(value)
    return {k: sum(v) / len(v) for k, v in sorted(fams.items())}


def scale_anatomy(frames):
    s = np.array([float(r['oracle_median_scale']) for r in frames])
    log_s = np.log(s)
    seq = np.array([r['sequence_id'] for r in frames])
    bias = float(log_s.mean())
    within_sd = []
    seq_means = []
    for sid in np.unique(seq):
        x = log_s[seq == sid]
        seq_means.append(x.mean())
        within_sd.append(x.std())
    out = {'median_s': float(np.median(s)), 'mean_log_s': bias, 'scale_factor_pred_over_gt': float(math.exp(-bias)),
           'sd_log_s_after_global': float(np.std(log_s - bias)),
           'sd_within_sequence': float(np.mean(within_sd)),  # equal weight per sequence
           'mean_abs_sequence_log_s': float(np.mean(np.abs(seq_means)))}
    if frames and 'median_pred_m' in frames[0] and frames[0]['median_pred_m'] not in ('', None):
        lg = np.log(np.array([float(r['median_gt_m']) for r in frames]))
        lp = np.log(np.array([float(r['median_pred_m']) for r in frames]))
        slope, intercept = np.polyfit(lg, lp, 1)
        out.update({'slope_log_pred_on_log_gt': float(slope), 'intercept': float(intercept),
                    'corr_log_pred_log_gt': float(np.corrcoef(lg, lp)[0, 1])})
    return out


def compare(methods, x, ref, sigma):
    a = float(methods[x]['headline']['abs_rel'])
    b = float(methods[ref]['headline']['abs_rel'])
    fx, fr = family_abs_rel(methods[x]['sequences']), family_abs_rel(methods[ref]['sequences'])
    if fx.keys() != fr.keys():
        raise ValueError(f'family sets differ between {x} and {ref}')
    n = len(fx)
    x_wins = sum(fx[k] < fr[k] for k in fx)
    d = a - b
    lo, hi = round(7 / 18 * n), round(11 / 18 * n)
    strong = round(14 / 18 * n)
    if abs(d) < max(2 * sigma, 0.03) and lo <= x_wins <= hi:
        verdict = f'{x} ~ {ref}'
    elif d > max(3 * sigma, 0.05) and n - x_wins >= strong:
        verdict = f'{ref} >> {x}'
    elif -d > max(3 * sigma, 0.05) and x_wins >= strong:
        verdict = f'{x} >> {ref}'
    else:
        verdict = 'inconclusive'
    return {'x': x, 'ref': ref, 'abs_rel_x': a, 'abs_rel_ref': b, 'delta': d, 'x_family_wins': x_wins,
            'families': n, 'verdict': verdict}


def analyse(campaign_dir, suffix=''):
    """suffix '' uses best.pt arms; '.last' uses the last.pt arms (Z is shared)."""
    methods, registry = load_campaign(campaign_dir)

    def m(name):
        return name if name == 'Z' else name + suffix

    available = {k for k in methods}
    pairs = [(m(a), m(b)) for a, b in NOISE_PAIRS if m(a) in available and m(b) in available]
    noise = {f'{a} vs {b}': abs(float(methods[a]['headline']['abs_rel']) - float(methods[b]['headline']['abs_rel']))
             for a, b in pairs}
    sigma = max(noise.values()) if noise else float('nan')
    comparisons = []
    if m('G') in available:
        for x in ATTRIBUTION:
            if m(x) in available:
                comparisons.append(compare(methods, m(x), m('G'), sigma))
    for x, ref in ARCHITECTURE:
        if m(x) in available and m(ref) in available:
            comparisons.append(compare(methods, m(x), m(ref), sigma))
    records = {r['name']: r for r in registry['methods']}
    arms = []
    for name, data in methods.items():
        if suffix and name != 'Z' and not name.endswith(suffix):
            continue
        if not suffix and name.endswith('.last'):
            continue
        h = data['headline']
        rec = records.get(name, {})
        arms.append({'method': name, 'abs_rel': float(h['abs_rel']), 'rmse_mm': float(h['rmse_mm']),
                     'delta_1_25': float(h['delta_1_25']), 'step': rec.get('selected_step'),
                     'source_val_abs_rel': rec.get('source_validation_sequence_macro_abs_rel'),
                     'oracle_rmse_mm': float(h['oracle_rmse_mm']),  # sequence-macro, ORACLE diagnostic
                     **scale_anatomy(data['frames'])})
    return {'campaign_dir': str(campaign_dir), 'head_set': suffix or 'best', 'noise': noise, 'sigma': sigma,
            'arms': arms, 'comparisons': comparisons}


def markdown(result):
    lines = [f"## Head set: {result['head_set']}", '',
             f"Noise sigma = {result['sigma']:.3f} (" + ', '.join(f'{k}: {v:.3f}' for k, v in result['noise'].items()) + ')',
             '', '| arm | step | src-val AbsRel | C2 AbsRel | RMSE mm | d<1.25 | ORACLE RMSE mm | median gt/pred | pred/gt exp(-mean log s) | '
             'SD log s after global | mean within-seq SD | slope log pred~log gt |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a in result['arms']:
        sv, step, slope = a['source_val_abs_rel'], a['step'], a.get('slope_log_pred_on_log_gt')
        lines.append(f"| {a['method']} | {'-' if step is None else step} | {'-' if sv is None else f'{sv:.4f}'} | "
                     f"{a['abs_rel']:.3f} | {a['rmse_mm']:.1f} | {a['delta_1_25']:.3f} | {a['oracle_rmse_mm']:.2f} | "
                     f"{a['median_s']:.3f} | {a['scale_factor_pred_over_gt']:.2f}x | {a['sd_log_s_after_global']:.3f} | "
                     f"{a['sd_within_sequence']:.3f} | {'-' if slope is None else f'{slope:.2f}'} |")
    lines += ['', '| comparison | AbsRel x | AbsRel ref | delta | x wins families | verdict |', '|---|---|---|---|---|---|']
    for c in result['comparisons']:
        lines.append(f"| {c['x']} vs {c['ref']} | {c['abs_rel_x']:.3f} | {c['abs_rel_ref']:.3f} | {c['delta']:+.3f} | "
                     f"{c['x_family_wins']}/{c['families']} | {c['verdict']} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--campaign', required=True, help='outputs/c2_campaign/<hash12> (must contain COMPLETE)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not (Path(args.campaign) / 'COMPLETE').exists():
        raise SystemExit(f'{args.campaign} is not a completed campaign')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results = [analyse(args.campaign, '')]
    methods, _ = load_campaign(args.campaign)
    if any(k.endswith('.last') for k in methods):
        results.append(analyse(args.campaign, '.last'))
    (out / 'analysis.json').write_text(json.dumps(results, indent=2))
    (out / 'analysis.md').write_text('\n'.join(markdown(r) for r in results))
    print((out / 'analysis.md').read_text())


if __name__ == '__main__':
    main()
