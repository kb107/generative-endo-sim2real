"""Fixture tests for the scale anatomy (CPU, generated numbers; no dataset, no model)."""
import json
import math

import numpy as np
import torch

from generative_sim2real.analysis import scale_anatomy as sa


def test_ols_slope_recovers_line_and_vectorizes():
    x = np.log(np.array(sa.SWEEP_R) / 504)
    assert math.isclose(ols_one(x, 1.03 * x + .2), 1.03, abs_tol=1e-12)
    y = np.stack([2. * x - 1., -.5 * x, np.zeros_like(x)])
    assert np.allclose(sa.ols_slope(x, y), [2., -.5, 0.])


def ols_one(x, y):
    return float(sa.ols_slope(x, y))


def test_block_ids_stay_within_sequences_and_follow_frame_order():
    seqs = ['a'] * 23 + ['b'] * 10 + ['c'] * 3
    order = list(range(len(seqs)))
    order[:23] = order[:23][::-1]                       # shuffled storage order inside sequence a
    blocks = sa.block_ids(seqs, order)
    assert blocks.max() + 1 == 3 + 1 + 1
    for s in 'abc':
        assert len(set(blocks[[i for i, v in enumerate(seqs) if v == s]]) & set(blocks[[i for i, v in enumerate(seqs) if v != s]])) == 0
    members = [i for i, v in enumerate(seqs) if v == 'a']
    first = sorted(members, key=lambda i: order[i])[:10]
    assert len(set(blocks[first])) == 1 and sum(blocks[members] == blocks[first[0]]) == 10


def synthetic(n=200, kappa=0.8, seed=3):
    rng = np.random.default_rng(seed)
    x = np.log(np.array(sa.SWEEP_R) / 504)
    base = rng.normal(0., .05, n)
    b = base[None] + kappa * x[:, None] + rng.normal(0, .01, (len(x), n))
    seqs = [f's{i // 50}' for i in range(n)]
    return b, base, seqs, list(range(n))


def test_bootstrap_is_deterministic_and_ci_covers_truth():
    b, base, seqs, order = synthetic()
    blocks = sa.block_ids(seqs, order)
    w = np.ones(len(base))
    crop_log = math.log(300.3 / 241.5)
    kw = dict(crop_log_focal=crop_log)
    first, means = sa.kappa_and_crop(b, base + .4 * crop_log, base, w, blocks, **kw)
    second, _ = sa.kappa_and_crop(b, base + .4 * crop_log, base, w, blocks, **kw)
    assert first == second                                              # fixed seed 0 -> identical
    lo, hi = first['kappa']['ci95']
    assert lo < .8 < hi and abs(first['kappa']['estimate'] - .8) < .03
    assert abs(first['kappa_crop_region_matched']['estimate'] - .4) < 1e-9     # (b_crop - b_region)/ln f
    assert set(means['mean_b']) == {str(r) for r in sa.SWEEP_R}
    other, _ = sa.kappa_and_crop(b, base, base, w, blocks, n=50, seed=1, **kw)
    assert other['kappa']['ci95'] != first['kappa']['ci95']


def test_bootstrap_means_resample_whole_blocks_and_zero_weight_is_ignored():
    values = np.array([[1., 1., 1., 5., 5., 5.]])
    blocks = np.array([0, 0, 0, 1, 1, 1])
    draws = sa.bootstrap_means(values, np.ones(6), blocks, n=200, seed=0)
    assert set(np.round(draws[:, 0], 6)) <= {1., 3., 5.}
    weights = np.array([1., 1., 1., 1., 0., 0.])
    assert np.allclose(sa.bootstrap_means(np.array([[1., 1., 1., 1., np.nan, np.nan]]), weights, blocks, 20), 1.)


def test_crop_k_update_and_effective_focal():
    K = torch.tensor([[227.604, 0., 237.5], [0., 227.604, 237.5], [0., 0., 1.]])
    rgb, depth, valid = torch.rand(3, 475, 475), torch.rand(1, 475, 475), torch.ones(1, 475, 475, dtype=torch.bool)
    c_rgb, c_depth, c_valid, c_K, (top, left) = sa.crop_sample(rgb, depth, valid, K)
    assert (top, left) == (46, 46) and c_rgb.shape == (3, 382, 382) == (3, *c_depth.shape[-2:]) and c_valid.shape == (1, 382, 382)
    assert torch.equal(c_rgb, rgb[:, 46:428, 46:428]) and torch.equal(c_depth, depth[:, 46:428, 46:428])
    assert c_K[0, 2] == 237.5 - 46 and c_K[1, 2] == 237.5 - 46 and c_K[0, 0] == K[0, 0] and K[0, 2] == 237.5
    assert math.isclose(c_K[0, 0].item() * 504 / 382, 300.3, abs_tol=.01)         # 300.3 px after the 504 resize
    assert math.isclose(sa.TRAIN_FOCAL, 227.604 * 504 / 475, abs_tol=5e-3)


def test_log_ratio_sign_and_median():
    gt = torch.full((1, 4, 4), .05)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[0, 0, 0] = False
    lr, ar, n = sa.log_ratio(gt * math.e, gt, valid)
    assert math.isclose(lr, 1., abs_tol=1e-6) and n == 15 and math.isclose(ar, math.e - 1, rel_tol=1e-5)   # too far -> positive
    assert sa.log_ratio(gt, gt, torch.zeros_like(valid))[0] is None


def test_prediction_formulas_and_sign():
    t, kappa = -.05, .9
    assert math.isclose(sa.predict_bias(t, kappa), t + kappa * math.log(300 / 241.502))
    assert math.isclose(sa.LN_C2, .2170, abs_tol=2e-4)
    assert math.isclose(sa.secondary_bias(t), t + .18)
    f406 = 300 * (406 / 504 + 504 / 630) / 2
    assert math.isclose(sa.F406, f406)
    arms = sa.arm_predictions(t, kappa, c_g=.03)
    assert math.isclose(arms['406'], t + kappa * math.log(f406 / 241.502))
    assert math.isclose(arms['cG'], t + kappa * math.log(300 / 241.502) - .03)
    assert math.isclose(arms['cG.406'], t - .03 + kappa * math.log(f406 / 241.502))
    assert 'cG' not in sa.arm_predictions(t, kappa)
    assert sa.predict_bias(.1, 1.) > sa.predict_bias(.1, 0.)          # kappa=1 over-predicts: positive bias


def test_sequence_bootstrap_resamples_sequences():
    values = [1., 1., 3., 3., 3., 3.]
    seqs = ['a', 'a', 'b', 'b', 'c', 'c']
    out = sa.sequence_bootstrap(values, seqs, n=500)
    assert math.isclose(out['mean'], 14 / 6) and out['n_sequences'] == 3 and 1. <= out['ci95'][0] <= out['ci95'][1] <= 3.
    assert out == sa.sequence_bootstrap(values, seqs, n=500)


def _registry():
    names = ['Z', 'R', 'R_s1', 'G', 'G_s1']
    methods = [{'name': 'Z', 'kind': 'official', 'head': None}] + [
        {'name': n, 'kind': 'trained', 'head': {'path': f'{n}/best.pt', 'file_sha256': 'f' * 64, 'head_sha256': 'h' * 64},
         'source_validation_sequence_macro_abs_rel': .05} for n in names[1:]]
    return {'method_names': names, 'methods': methods}


def test_predictions_schema_and_markdown(tmp_path):
    reg = _registry()
    kappa = {n: {'stats': {'kappa': {'estimate': 1., 'ci95': [.9, 1.1]}, 'kappa_crop': {'estimate': .8, 'ci95': [.7, .9]},
                           'kappa_crop_region_matched': {'estimate': .7, 'ci95': [.6, .8]}}} for n in reg['method_names']}
    probe = {n: {'summary': {'target_correct': {'log_ratio_mean': -.1}, 'real': {'log_ratio_mean': .02}}} for n in reg['method_names']}
    cx = {('G', 'R'): {'mean': .05, 'ci95': [.04, .06], 'n_images': 2000, 'n_sequences': 7}}
    pred = sa.build_predictions(reg, kappa, probe, cx, {'registry': 'x'})
    assert list(pred['heads']) == reg['method_names'] and not [a for a in pred['absent'] if 'kappa' in a]
    z, r = pred['heads']['Z'], pred['heads']['R']
    assert set(r) >= {'t_h', 'kappa_h', 'kappa_h_ci95', 'kappa_crop', 'primary_predicted_bias', 'secondary_predicted_bias',
                      'baseline_source_validation_abs_rel', 'baseline_probe_real_mean'}
    assert math.isclose(r['primary_predicted_bias'], -.1 + math.log(300 / 241.502)) and math.isclose(r['secondary_predicted_bias'], .08)
    assert z['baseline_source_validation_abs_rel'] is None and r['baseline_source_validation_abs_rel'] == .05
    arms = pred['arms_6A']
    assert math.isclose(arms['R.cG'], r['primary_predicted_bias'] - .05)
    assert {'R.406', 'R_s1.406', 'G.406', 'G_s1.406', 'R.cG', 'R.cG.406'} <= set(arms) and 'R_s1.cG' not in arms
    assert {'name': 'R_s1.cG', 'c_G': True} in pred['absent']
    json.dumps(pred, allow_nan=False)
    assert '| R |' in sa.format_markdown(pred) and 'R.cG.406' in sa.format_markdown(pred)


def test_head_key_dedupes_identical_heads():
    a = {'head': {'head_sha256': 'x'}}
    assert sa.head_key(a) == sa.head_key({'head': {'head_sha256': 'x'}}) != sa.head_key({'head': {'head_sha256': 'x'}, 'backbone_lora': {'sha256': 'y'}})
    assert sa.head_key({'head': None}) == ('official', None)
