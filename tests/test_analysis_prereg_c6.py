import math

import pytest

from generative_sim2real.analysis import prereg_c6 as P

FAMILIES = [f'fam{i:02d}' for i in range(18)]
NAMES = ('Z', 'R', 'R_s1', 'A_mild', 'A_strong', 'T', 'S', 'G', 'G_s1', 'Gp', 'M', 'R_bb', 'G_bb', 'G_full',
         'M_full', 'G_simuscope')
CONST = {'ln_300_over_train': math.log(300 / 241.502), 'ln_f406_over_train': math.log(240.8333 / 241.502)}


def method(b, abs_rel, oracle=5.0):
    frames = [{'oracle_median_scale': math.exp(-b), 'sequence_id': f'{f}_v{v}'} for f in FAMILIES for v in (1, 2)]
    seqs = {f'{f}_v{v}': abs_rel for f in FAMILIES for v in (1, 2)}
    return {'headline': {'abs_rel': abs_rel, 'oracle_rmse_mm': oracle}, 'frames': frames, 'sequences': seqs}


@pytest.fixture
def pred():
    heads = {}
    for i, h in enumerate(NAMES):
        kappa = 1.0 if h.startswith('R') else .8
        t = .05 * i - .2
        heads[h] = {'t_h': t, 'kappa_h': kappa, 'primary_predicted_bias': t + kappa * CONST['ln_300_over_train'],
                    'secondary_predicted_bias': t + .18, 'baseline_source_validation_abs_rel': .05 + .001 * (i % 3),
                    'baseline_probe_real_mean': -.01 * (i % 4), 'head_sha256': h}
    return {'heads': heads, 'constants': CONST, 'arms_6A': {'R.cG.406': 0.0, 'R.cG': .2}}


def test_blind_test_scores_exact_predictions(pred):
    methods5 = {h: method(pred['heads'][h]['primary_predicted_bias'], .5) for h in NAMES}
    bt = P.blind_test(pred, methods5)
    assert bt['B']['n'] == 9 and bt['absent'] == []
    assert bt['B']['primary_predicted_bias']['mae'] == pytest.approx(0, abs=1e-9)
    assert bt['B']['primary_predicted_bias']['spearman'] == pytest.approx(1)
    assert bt['B']['secondary_predicted_bias']['mae'] > 0
    assert bt['pass']['all']
    assert bt['R_like']['n'] == 6 and bt['G_like']['n'] == 3 and bt['last_secondary']['n'] == 0


def test_campaign6a_convention_residual_and_contrast(pred):
    heads = pred['heads']
    methods5 = {h: method(heads[h]['primary_predicted_bias'], .5 if h.startswith('R') else .35) for h in NAMES}
    per_kappa = CONST['ln_f406_over_train'] - CONST['ln_300_over_train']
    methods6 = {'R': method(0, .5), 'G': method(0, .35)}
    for h in ('R', 'R_s1', 'G', 'G_s1'):
        methods6[h + '.406'] = method(heads[h]['primary_predicted_bias'] + heads[h]['kappa_h'] * per_kappa, .4)
    methods6['G.406']['frames'][0]['oracle_median_scale'] *= 1  # unchanged
    methods6['R.cG.406'] = method(.02, .30)
    out = P.campaign6a(pred, methods5, methods6, sigma5=.01)
    assert out['anchors_ok']
    assert out['convention']['R']['delta'] == pytest.approx(per_kappa, abs=1e-9)
    assert out['convention_confirmed'] and not out['convention_fails_for_G']
    assert out['residual_ok'] == {'R.cG.406': True}
    row = next(c for c in out['contrasts'] if c['x'] == 'R.cG.406' and c['ref'] == 'G')
    assert row['delta'] == pytest.approx(-.05) and row['abs_rel_ref'] == .35  # G native comes from campaign #5
    assert out['scalar_arms']['R.cG.406']['observed'] == pytest.approx(.02)


def test_campaign6a_flags_convention_miss_for_G(pred):
    heads = pred['heads']
    methods5 = {h: method(heads[h]['primary_predicted_bias'], .5) for h in NAMES}
    methods6 = {h + '.406': method(heads[h]['primary_predicted_bias'], .5) for h in ('R', 'R_s1', 'G', 'G_s1')}
    out = P.campaign6a(pred, methods5, methods6, sigma5=.01)
    assert not out['convention_confirmed'] and out['convention_fails_for_G'] and not out['anchors_ok']
