"""Full P5 diagnostic expansion: the declared configs differ from the reduced ones only as declared."""
import copy
import json
import math
from pathlib import Path

import pytest
import yaml

pytest.importorskip('diffusers')
pytest.importorskip('peft')

from generative_sim2real.generation import p5_full  # noqa: E402

ROOT = Path(__file__).parents[1]
load = lambda name: yaml.safe_load((ROOT / 'configs' / name).read_text())  # noqa: E731


def test_full_seed_configs_differ_from_v2_only_in_declared_keys():
    v2 = load('p5_v2.yaml')
    for seed in (17, 43):
        full = load(f'p5_full_seed{seed}.yaml')
        assert full['seed'] == seed and full['output_dir'].endswith(f'matrix_full_seed{seed}')
        assert full['p4_config'] == 'configs/p4_full_diag.yaml'
        assert full['analysis']['rules']['min_conditions'] == math.ceil(7 / 8 * 32) == 28
        # the scaled rule keeps the declared proportion
        assert 28 / 32 >= 7 / 8 > 27 / 32
        restored = copy.deepcopy(full)
        restored['seed'], restored['output_dir'] = v2['seed'], v2['output_dir']
        restored['p4_config'] = v2['p4_config']
        restored['analysis']['rules']['min_conditions'] = 7
        assert restored == v2  # adapter, scale, guidance, controller, arms, every other rule and threshold identical
    assert load('p5_full_seed17.yaml')['inputs']['target_lora'].endswith('target_lora_v2/checkpoints/step_002000/lora_target.safetensors')
    assert load('p5_full_seed17.yaml')['lora_scale'] == 1.0 and load('p5_full_seed17.yaml')['guidance'] == 4.5


def test_full_diag_p4_config_changes_only_the_condition_keys():
    base, full = load('p4.yaml'), load('p4_full_diag.yaml')
    assert full['gate']['conditions'] == 'sample_ids' and full['gate']['wrong_depth'] == 'wrong_depth_permutation'
    full['gate']['conditions'], full['gate']['wrong_depth'] = base['gate']['conditions'], base['gate']['wrong_depth']
    assert full == base


def test_declared_combined_analysis_config_and_noise_overlap_is_explicit():
    combined = load('p5_full.yaml')
    assert combined['rules'] == {'min_conditions': 28, 'require_positive_mean_gain': True}
    assert [Path(d).name for d in combined['seed_runs']] == ['matrix_full_seed17', 'matrix_full_seed43']
    seeds = [{17 + i for i in range(32)}, {43 + i for i in range(32)}]
    assert sorted(seeds[0] & seeds[1]) == list(range(43, 49))  # declared in the config comments: 6 shared draws


def test_diagnostic_derangement_is_cross_scene_when_manifests_exist():
    path = ROOT / 'outputs' / 'manifests' / 'milestone_v1' / 'diagnostics.json'
    if not path.exists():
        pytest.skip('no manifests in this checkout')
    d = json.loads(path.read_text())
    assert len(d['sample_ids']) == 32 and sorted(d['wrong_depth_permutation']) == list(range(32))
    assert all(d['scene_ids'][i] != d['scene_ids'][p] for i, p in enumerate(d['wrong_depth_permutation']))


def synthetic(good, seed_offset=0):
    rows = []
    for i in range(32):
        rows.append({'arm': 'real', 'cond': i, 'abs_rel_own': 0.1, 'abs_rel_partner': 0.5,
                     'spearman_own': 0.9, 'spearman_partner': 0.1})
        for adapter in p5_full.ADAPTERS:
            ok = good[adapter](i)
            rows += [
                {'arm': f'{adapter}_off', 'cond': i, 'abs_rel_own': 0.5, 'abs_rel_partner': 0.5,
                 'spearman_own': 0.1, 'spearman_partner': 0.1},
                {'arm': f'{adapter}_correct', 'cond': i, 'abs_rel_own': 0.2 if ok else 0.5, 'abs_rel_partner': 0.5,
                 'spearman_own': 0.8 if ok else 0.1, 'spearman_partner': 0.1},
                {'arm': f'{adapter}_wrong', 'cond': i, 'abs_rel_own': 0.5,
                 'abs_rel_partner': 0.2 if ok else 0.5, 'spearman_own': 0.1, 'spearman_partner': 0.8 if ok else 0.1}]
    return rows


def test_scaled_rule_counts_28_of_32_and_combined_means_over_seeds(tmp_path):
    rules = {'min_conditions': 28, 'require_positive_mean_gain': True}
    everywhere = lambda i: True  # noqa: E731
    # the target adapter follows depth in 28 conditions in one seed and in 27 in the other
    seeds = {17: lambda i: i >= 4, 43: lambda i: i >= 5}  # fails 4 and 5 conditions respectively
    for seed, rule in seeds.items():
        directory = tmp_path / f'seed{seed}'
        directory.mkdir()
        rows = synthetic({'none': everywhere, 'source': everywhere, 'target': rule})
        (directory / 'render.json').write_text(json.dumps({
            'ids': list(range(32)), 'partner': list(range(31, -1, -1)), 'seed': seed, 'tests': {}}))
        (directory / 'proxy.json').write_text(json.dumps({'rows': rows, 'model': {}}))
    result, cells = p5_full.full_analysis([tmp_path / 'seed17', tmp_path / 'seed43'], rules)
    primary = result['primary_abs_rel']
    assert primary['per_seed']['17']['swap_verdict'].startswith('RETAINED')       # 28/32 meets the threshold exactly
    assert primary['per_seed']['43']['swap_verdict'].startswith('NOT RETAINED')   # 27/32 does not
    combined_m1 = primary['combined']['per_adapter']['target']['checks']['M1_correct_beats_off']
    assert combined_m1['conditions_in_expected_direction'] == 28  # condition 4 is good in one seed only: its mean still gains
    assert primary['verdicts']['combined'].startswith('RETAINED') and not primary['seeds_agree_with_combined']
    assert primary['combined']['per_adapter']['source']['control_retained']
    assert result['secondary_spearman']['combined']['R0_audit_real_rgb']['passed']
    assert [c['adapter'] for c in result['failure_cells_lowest_combined_M1_gain']][:4] == ['target'] * 4
    with pytest.raises(ValueError, match='same conditions'):
        (tmp_path / 'seed43' / 'render.json').write_text(json.dumps({
            'ids': list(range(1, 33)), 'partner': list(range(31, -1, -1)), 'seed': 43, 'tests': {}}))
        p5_full.full_analysis([tmp_path / 'seed17', tmp_path / 'seed43'], rules)
