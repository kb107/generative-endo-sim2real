"""Pre-declared P4 gate rules: metrics, direction-of-effect checks, verdict logic."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from generative_sim2real.generation import p4_report as pr

CONFIG = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'p4.yaml').read_text())
RULES = CONFIG['gate']['rules']
PARTNER = list(range(7, -1, -1))


def depth_pair():
    g = torch.Generator().manual_seed(0)
    reference = torch.rand(1, 16, 16, generator=g) + 0.5
    return reference, torch.ones(1, 16, 16, dtype=torch.bool)


def test_median_scaled_abs_rel_is_scale_free_and_orders_errors():
    reference, valid = depth_pair()
    assert pr.median_scaled_abs_rel(reference * 7.3, reference, valid) == pytest.approx(0, abs=1e-6)
    noisy = reference * (1 + 0.3 * torch.sin(torch.arange(256.).reshape(1, 16, 16)))
    assert pr.median_scaled_abs_rel(noisy, reference, valid) > 0.05
    valid[..., :8] = False  # invalid pixels are ignored, not scored
    garbage = reference.clone()
    garbage[..., :8] = 99
    assert pr.median_scaled_abs_rel(garbage, reference, valid) == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError):
        pr.median_scaled_abs_rel(-reference, reference, valid)


def test_spearman_monotone_and_reversed():
    reference, valid = depth_pair()
    assert pr.spearman(reference ** 3, reference, valid, stride=1) == pytest.approx(1.0)
    assert pr.spearman(-reference, reference, valid, stride=1) == pytest.approx(-1.0)


def rows(*, real_good=True, correct=0.1, off=0.3, wrong_own=0.3, wrong_partner=0.1):
    out = []
    for i in range(8):
        for seed in (0, 1):
            def add(arm, own, partner, s=seed):
                out.append({'arm': arm, 'cond': i, 'seed_index': s, 'abs_rel_own': own,
                            'abs_rel_partner': partner, 'spearman_own': 1 - own,
                            'spearman_partner': 1 - partner})
            add('correct', correct, 0.4)
            add('off', off, 0.4)
            add('wrong', wrong_own, wrong_partner)
        out.append({'arm': 'real', 'cond': i, 'seed_index': None,
                    'abs_rel_own': 0.1 if real_good else 0.4,
                    'abs_rel_partner': 0.4 if real_good else 0.1,
                    'spearman_own': 0.9, 'spearman_partner': 0.6})
    return out


def test_direction_check_thresholds():
    assert pr.direction_check([1] * 7 + [-1], min_conditions=7)['passed']
    assert not pr.direction_check([1] * 6 + [-1, -1], min_conditions=7)['passed']
    assert not pr.direction_check([1] * 7 + [-100], min_conditions=7)['passed']  # mean gain <= 0
    assert pr.direction_check([1] * 7 + [-100], min_conditions=7, require_positive_mean=False)['passed']


def test_intervention_checks_pass_fail_and_uninterpretable():
    good = pr.intervention_checks(rows(), PARTNER, RULES)
    assert good['proxy_verdict'] == 'PASS' and all(c['passed'] for c in good['checks'].values())
    # Controller ignores depth: correct == off == wrong -> fails R1, R2 and R3.
    ignoring = pr.intervention_checks(rows(correct=0.3, off=0.3, wrong_own=0.3, wrong_partner=0.3),
                                      PARTNER, RULES)
    assert ignoring['proxy_verdict'] == 'FAIL' and not ignoring['checks']['R1_correct_beats_off']['passed']
    # A proxy that cannot tell true from mismatched depth on real RGB makes the gate uninterpretable.
    blind = pr.intervention_checks(rows(real_good=False), PARTNER, RULES)
    assert blind['proxy_verdict'].startswith('UNINTERPRETABLE') and not blind['proxy_discriminative']
    with pytest.raises(ValueError, match='Missing proxy rows'):
        pr.intervention_checks([r for r in rows() if r['arm'] != 'off'], PARTNER, RULES)


def test_overall_verdict_combines_proxy_and_visual_rubric():
    assert pr.overall_verdict('PASS', True) == 'PASS'
    assert pr.overall_verdict('FAIL', False) == 'FAIL'
    assert pr.overall_verdict('PASS', False) == 'AMBIGUOUS'
    assert pr.overall_verdict('FAIL', True) == 'AMBIGUOUS'
    assert pr.overall_verdict('UNINTERPRETABLE (proxy fails its real-RGB audit)', True) == 'AMBIGUOUS'
    assert pr.overall_verdict('PASS', None) == 'PENDING_VISUAL_RUBRIC'


def test_wasserstein_and_appearance_checks():
    g = torch.Generator().manual_seed(2)
    a = torch.rand(6, 3, 16, 16, generator=g) * 0.5
    assert pr.rgb_w1(a, a) == pytest.approx(0, abs=1e-6)
    assert pr.rgb_w1(a, a + 0.2, quantiles=100) == pytest.approx(0.2, abs=1e-6)
    stats = {'target_vs_c1': 0.05, 'source_vs_c1': 0.2, 'source_vs_source': 0.04, 'target_vs_source': 0.3}
    assert pr.appearance_checks(stats)['pixel_statistics_support_appearance_difference']
    swapped = copy.deepcopy(stats)
    swapped['target_vs_c1'] = 0.3
    assert not pr.appearance_checks(swapped)['pixel_statistics_support_appearance_difference']


def test_pairwise_diversity_detects_collapse():
    same = torch.ones(4, 3, 8, 8) * 0.3
    assert pr.pairwise_diversity(same) == 0.0
    varied = torch.stack([torch.full((3, 8, 8), v) for v in (0.1, 0.4, 0.7, 0.9)])
    assert pr.pairwise_diversity(varied) > 0.2


def test_png_roundtrip_is_exact_for_8bit_images(tmp_path):
    image = torch.randint(0, 256, (3, 8, 8)).float() / 255
    pr.save_png(image, tmp_path / 'a' / 'x.png')
    assert torch.equal(pr.png_tensor(tmp_path / 'a' / 'x.png'), image)


def test_visual_verdict_follows_the_declared_counts():
    good = {'layout_follows_control': ['yes'] * 6 + ['partly', 'no'],
            'wrong_depth_moves_layout': ['yes'] * 6 + ['no', 'no'], 'not_degenerate': True}
    assert pr.visual_verdict(good)
    assert not pr.visual_verdict({**good, 'not_degenerate': False})
    assert not pr.visual_verdict({**good, 'layout_follows_control': ['yes'] * 5 + ['partly'] * 3})
    with pytest.raises(ValueError):
        pr.visual_verdict({**good, 'wrong_depth_moves_layout': ['yes'] * 7})
