"""Score the campaign #6 pre-registration against completed C2 campaigns.

Blind scale test on campaign #5, campaign #6A (camera-convention and look scalars), campaign #6B (FDA). Written before
campaign #5 was read. Inputs: the frozen predictions.json and the saved per-frame rows of completed campaigns (no
inference). Bias b = -mean_frames ln(median_gt/median_pred) (positive = predicts too far), as in predictions.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from generative_sim2real.analysis.campaign import analyse, compare, load_campaign, scale_anatomy

BLIND = ('R_s1', 'A_mild', 'A_strong', 'T', 'S', 'G_s1', 'Gp', 'R_bb', 'G_bb')
R_LIKE = ('R_s1', 'A_mild', 'A_strong', 'T', 'S', 'R_bb')
G_LIKE = ('G_s1', 'Gp', 'G_bb')
TRAINED = ('R', 'R_s1', 'A_mild', 'A_strong', 'T', 'S', 'G', 'G_s1', 'Gp', 'M', 'R_bb', 'G_bb')
CONV4 = ('R', 'R_s1', 'G', 'G_s1')
EXTENSION = ('A_mild', 'A_strong', 'T', 'S', 'Gp', 'M', 'R_bb', 'G_bb')
PREDICTORS = ('primary_predicted_bias', 'secondary_predicted_bias', 't_h')
BASELINES = ('baseline_source_validation_abs_rel', 'baseline_probe_real_mean')


def bias(methods, name):
    return -scale_anatomy(methods[name]['frames'])['mean_log_s']


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v, kind='mergesort')
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        for value in np.unique(v):  # average ties
            r[v == value] = r[v == value].mean()
        return r
    if len(x) < 3:
        return float('nan')
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def score_set(heads, names, observed):
    out = {'n': len(names), 'heads': list(names)}
    if not names:
        return out
    obs = [observed[h] for h in names]
    for p in PREDICTORS:
        pred = [heads[h][p] for h in names]
        out[p] = {'mae': float(np.mean(np.abs(np.subtract(pred, obs)))), 'spearman': spearman(pred, obs)}
    for b in BASELINES:
        out[b] = {'spearman': spearman([heads[h][b] for h in names], obs)}
    return out


def blind_test(pred, methods5):
    heads = pred['heads']
    observed = {h: bias(methods5, h) for h in methods5 if h in heads}
    blind = [h for h in BLIND if h in observed]
    result = {'observed_bias': observed, 'absent': [h for h in BLIND if h not in observed],
              'B': score_set(heads, blind, observed),
              'R_like': score_set(heads, [h for h in R_LIKE if h in observed], observed),
              'G_like': score_set(heads, [h for h in G_LIKE if h in observed], observed)}
    b = result['B']
    p, s = b['primary_predicted_bias'], b['secondary_predicted_bias']
    result['pass'] = {'spearman_ge_0.6': p['spearman'] >= .6, 'mae_le_0.07': p['mae'] <= .07,
                      'beats_secondary_mae': p['mae'] < s['mae'],
                      'beats_baselines_spearman': all(p['spearman'] > b[k]['spearman'] for k in BASELINES)}
    result['pass']['all'] = all(result['pass'].values())
    last = [h + '.last' for h in TRAINED if h + '.last' in observed
            and heads[h + '.last']['head_sha256'] != heads[h]['head_sha256']]
    result['last_secondary'] = score_set(heads, last, observed)
    result['per_head'] = {h: {'observed': observed[h], **{p: heads[h][p] for p in PREDICTORS}} for h in observed}
    return result


def campaign6a(pred, methods5, methods6, sigma5):
    heads, c = pred['heads'], pred['constants']
    delta_per_kappa = c['ln_f406_over_train'] - c['ln_300_over_train']
    out = {'sigma5': sigma5}
    out['anchors'] = {h: abs(float(methods6[h]['headline']['abs_rel']) - float(methods5[h]['headline']['abs_rel']))
                      for h in ('R', 'G') if h in methods6 and h in methods5}
    out['anchors_ok'] = bool(out['anchors']) and all(v <= 1e-6 for v in out['anchors'].values())

    def delta_row(h):
        d = bias(methods6, h + '.406') - bias(methods5, h)
        p406 = heads[h]['t_h'] + heads[h]['kappa_h'] * c['ln_f406_over_train']
        return {'delta': d, 'predicted_delta': heads[h]['kappa_h'] * delta_per_kappa,
                'bias_406': bias(methods6, h + '.406'), 'predicted_bias_406': p406}

    conv = {h: delta_row(h) for h in CONV4 if h + '.406' in methods6 and h in methods5}
    for r in conv.values():
        r['hit'] = abs(r['delta'] - r['predicted_delta']) <= .05
    out['convention'] = conv
    out['convention_confirmed'] = ('R' in conv and conv['R']['delta'] <= -.10
                                   and sum(r['hit'] for r in conv.values()) >= 3)
    out['convention_fails_for_G'] = 'G' in conv and not conv['G']['hit']
    ext = {h: delta_row(h) for h in EXTENSION if h + '.406' in methods6 and h in methods5}
    if ext:
        out['extension'] = ext
        out['extension_mae_delta'] = float(np.mean([abs(r['delta'] - r['predicted_delta']) for r in ext.values()]))
        out['extension_mae_bias_406'] = float(np.mean([abs(r['bias_406'] - r['predicted_bias_406'])
                                                       for r in ext.values()]))
        out['extension_pass'] = out['extension_mae_delta'] <= .05 and out['extension_mae_bias_406'] <= .05
    out['scalar_arms'] = {a: {'observed': bias(methods6, a), 'predicted': pred['arms_6A'][a]}
                          for a in pred['arms_6A'] if a in methods6}
    out['residual_ok'] = {a: abs(bias(methods6, a)) <= .08 for a in ('R.cG.406', 'R_s1.cG.406') if a in methods6}
    merged = {**methods5, **{k: v for k, v in methods6.items() if k not in methods5}}
    contrasts = []
    for x, ref in (('R.cG.406', 'G'), ('R_s1.cG.406', 'G'), ('R.cG.406', 'G.406'), ('R_s1.cG.406', 'G_s1.406'),
                   ('R.cG', 'G'), ('R.406', 'G')):
        if x in merged and ref in merged:
            row = compare(merged, x, ref, sigma5)
            row['oracle_rmse_mm_x'] = float(merged[x]['headline']['oracle_rmse_mm'])
            row['oracle_rmse_mm_ref'] = float(merged[ref]['headline']['oracle_rmse_mm'])
            contrasts.append(row)
    out['contrasts'] = contrasts
    out['seed_check'] = {f'{a} vs {b}': abs(float(methods6[a]['headline']['abs_rel'])
                                             - float(methods6[b]['headline']['abs_rel']))
                         for a, b in (('R.406', 'R_s1.406'), ('G.406', 'G_s1.406')) if a in methods6 and b in methods6}
    return out


def campaign6b(methods5, methods6b, sigma5, f_predictions=None):
    merged = {**methods5, **methods6b}
    out = {'sigma5': sigma5, 'contrasts': [], 'observed_bias': {}}
    for f in ('F_fda01', 'F_fda05'):
        for suffix in ('', '.last'):
            x = f + suffix
            if x not in methods6b:
                continue
            out['observed_bias'][x] = bias(methods6b, x)
            for ref in ('G', 'T'):
                if ref + suffix in merged:
                    out['contrasts'].append(compare(merged, x, ref + suffix, sigma5))
    if f_predictions:
        out['predicted_bias'] = {k: v for k, v in f_predictions.items() if k in out['observed_bias']}
    return out


def markdown(scores):
    lines = ['# Campaign #6 pre-registration: scores', '']
    bt = scores.get('blind_test')
    if bt:
        lines += ['## Blind scale test on campaign #5 (bias = mean ln(pred/gt))', '',
                  '| set | n | primary MAE | primary rho | secondary MAE | secondary rho | t_h MAE | src-val rho | probe-real rho |',
                  '|---|---|---|---|---|---|---|---|---|']
        for key in ('R_like', 'G_like', 'B', 'last_secondary'):
            s = bt[key]
            if not s['n']:
                continue
            p, q, t = s['primary_predicted_bias'], s['secondary_predicted_bias'], s['t_h']
            lines.append(f"| {key} | {s['n']} | {p['mae']:.3f} | {p['spearman']:.2f} | {q['mae']:.3f} | "
                         f"{q['spearman']:.2f} | {t['mae']:.3f} | {s[BASELINES[0]]['spearman']:.2f} | "
                         f"{s[BASELINES[1]]['spearman']:.2f} |")
        lines += ['', f"Pass criteria: {bt['pass']}", '', '| head | observed | primary | secondary | t_h |',
                  '|---|---|---|---|---|']
        for h, r in bt['per_head'].items():
            lines.append(f"| {h} | {r['observed']:+.3f} | {r['primary_predicted_bias']:+.3f} | "
                         f"{r['secondary_predicted_bias']:+.3f} | {r['t_h']:+.3f} |")
    a = scores.get('c6a')
    if a:
        lines += ['', '## Campaign #6A', '', f"Anchors |dAbsRel| vs #5: {a['anchors']} (ok: {a['anchors_ok']})", '',
                  '| head | b native (#5) -> b 406 | Delta | predicted Delta | hit (<=0.05) |', '|---|---|---|---|---|']
        for h, r in {**a['convention'], **a.get('extension', {})}.items():
            lines.append(f"| {h} | {r['bias_406'] - r['delta']:+.3f} -> {r['bias_406']:+.3f} | {r['delta']:+.3f} | "
                         f"{r['predicted_delta']:+.3f} | {r.get('hit', abs(r['delta'] - r['predicted_delta']) <= .05)} |")
        lines += ['', f"Convention confirmed: {a['convention_confirmed']}; fails for G: {a['convention_fails_for_G']}",
                  f"Extension: MAE Delta {a.get('extension_mae_delta', float('nan')):.3f}, MAE b406 "
                  f"{a.get('extension_mae_bias_406', float('nan')):.3f}, pass {a.get('extension_pass')}", '',
                  '| scalar arm | observed b | predicted b |', '|---|---|---|']
        for k, r in a['scalar_arms'].items():
            lines.append(f"| {k} | {r['observed']:+.3f} | {r['predicted']:+.3f} |")
        lines += ['', f"Residual |b| <= 0.08: {a['residual_ok']}; seed check: {a['seed_check']}", '']
        lines += _contrast_table(a['contrasts'], oracle=True)
    b = scores.get('c6b')
    if b:
        lines += ['', '## Campaign #6B (FDA)', '', f"Observed bias: {b['observed_bias']}",
                  f"Predicted bias: {b.get('predicted_bias')}", '']
        lines += _contrast_table(b['contrasts'])
    return '\n'.join(lines) + '\n'


def _contrast_table(rows, oracle=False):
    head = '| comparison | AbsRel x | AbsRel ref | delta | x wins families | verdict |' + (' oracle RMSE x / ref |' if oracle else '')
    lines = [head, '|---|---|---|---|---|---|' + ('---|' if oracle else '')]
    for c in rows:
        line = (f"| {c['x']} vs {c['ref']} | {c['abs_rel_x']:.3f} | {c['abs_rel_ref']:.3f} | {c['delta']:+.3f} | "
                f"{c['x_family_wins']}/{c['families']} | {c['verdict']} |")
        if oracle:
            line += f" {c['oracle_rmse_mm_x']:.2f} / {c['oracle_rmse_mm_ref']:.2f} |"
        lines.append(line)
    return lines


def _completed(path):
    if not (Path(path) / 'COMPLETE').exists():
        raise SystemExit(f'{path} is not a completed campaign')
    return load_campaign(path)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--predictions', default='outputs/analysis/scale_anatomy/predictions.json')
    parser.add_argument('--c5', required=True)
    parser.add_argument('--c6a')
    parser.add_argument('--c6b')
    parser.add_argument('--f-predictions', help='JSON {arm: predicted bias} frozen before #6B')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    pred = json.loads(Path(args.predictions).read_text())
    methods5 = _completed(args.c5)
    sigma5 = analyse(args.c5, '')['sigma']
    scores = {'predictions': args.predictions, 'c5': args.c5, 'sigma5': sigma5,
              'blind_test': blind_test(pred, methods5)}
    if args.c6a:
        scores['c6a'] = campaign6a(pred, methods5, _completed(args.c6a), sigma5)
    if args.c6b:
        fp = json.loads(Path(args.f_predictions).read_text()) if args.f_predictions else None
        scores['c6b'] = campaign6b(methods5, _completed(args.c6b), sigma5, fp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scores.json').write_text(json.dumps(scores, indent=2, default=float))
    (out / 'scores.md').write_text(markdown(scores))
    print((out / 'scores.md').read_text())


if __name__ == '__main__':
    main()
