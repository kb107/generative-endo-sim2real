"""Fixture tests for the look->scale probe (no datasets, no real model)."""
import math

import numpy as np
from PIL import Image
import torch

from da3_fakes import fake_model, intrinsics
from generative_sim2real.analysis import look_scale as ls
from generative_sim2real.variants import color


def write_matrix(tmp_path, count=2, size=64):
    rng = np.random.default_rng(0)
    for arm in ('real', 'source_correct', 'target_correct'):
        for i in range(count):
            array = rng.integers(40, 220, (size, size, 3), dtype=np.uint8)
            (tmp_path / 'images' / arm).mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(tmp_path / 'images' / arm / f'c{i}.png')


def test_slope_recovers_exponent():
    gains = np.log(ls.GAINS)
    assert math.isclose(ls.slope(gains, -.5 * gains + 3.), -.5, abs_tol=1e-12)
    assert math.isclose(ls.slope(gains, np.zeros(5)), 0., abs_tol=1e-12)


def test_gain_is_multiply_then_clip_on_8bit_grid():
    image = torch.tensor([[[.1, .6]], [[.2, .9]], [[.3, 1.]]])
    out = ls.apply_gain(image, 2.)
    assert torch.allclose(out, torch.tensor([[[.2, 1.]], [[.4, 1.]], [[.6, 1.]]]), atol=1 / 255)
    assert torch.equal(ls.apply_gain(out, 1.), out)


def test_image_stats_conventions():
    target = torch.full((1, 8, 8), .03)
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    far = ls.image_stats(target * 2, target, valid)
    assert math.isclose(far['log_ratio'], math.log(2)) and far['aligned_abs_rel'] < 1e-9   # too far: positive
    shape = torch.linspace(.02, .04, 64).reshape(1, 8, 8)
    wrong = ls.image_stats(shape.flip(-1) * 3, shape, valid)
    assert wrong['aligned_abs_rel'] > .03
    masked_valid = valid.clone()
    masked_valid[..., :4] = False
    hidden = target.clone()
    hidden[..., :4] = 5.                                                                    # garbage outside the mask
    assert math.isclose(ls.image_stats(hidden * .5, hidden, masked_valid)['log_ratio'], math.log(.5))


def test_versions_and_head_evaluation_on_fixtures(tmp_path):
    write_matrix(tmp_path)
    valid = torch.ones(1, 64, 64, dtype=torch.bool)
    reference = ls.load_png(tmp_path / 'images' / 'real' / 'c1.png')
    params = color.draw_strong_params(np.random.default_rng(3))
    versions = [ls.build_versions(tmp_path, i, valid, reference, params) for i in range(2)]
    names = set(versions[0])
    assert {'real', 'source_correct', 'target_correct', 'reinhard_real', 'a_strong_real'} <= names
    assert {ls.version_name(b, g) for b in ls.GAIN_BASES for g in ls.GAINS if g != 1.} <= names
    assert all(v.dtype == torch.float32 and 0 <= v.min() and v.max() <= 1 for v in versions[0].values())
    assert torch.allclose(versions[0]['real_x0.5'], ls.apply_gain(versions[0]['real'], .5))
    K = torch.stack([intrinsics(64)] * 2)
    depth = torch.full((2, 1, 64, 64), .03)
    mask = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    rows = ls.evaluate_head(fake_model(), versions, K, depth, mask, 'cpu')
    assert len(rows) == 2 and set(rows[0]) == names
    summary = ls.summarize(rows, ['real', 'target_correct'])
    assert summary['real']['delta_vs_real_mean'] == 0 and summary['real']['n'] == 2
    slopes = ls.sweep_slopes(rows)
    assert set(slopes) == set(ls.GAIN_BASES) and all(np.isfinite(slopes[b]['slope_mean']) for b in slopes)
    table = ls.format_tables({'head': {'summary': ls.summarize(rows, ['real']), 'slopes': slopes}}, ['real'])
    assert 'log(median_pred / median_gt)' in table and 'slope' in table
