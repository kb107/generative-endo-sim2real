"""Reporting tables from generated fixtures (no outputs/ directory, no models, no data)."""
import csv
import json

import numpy as np
import pytest

from generative_sim2real.reporting import common, tables


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_write_table_md_and_csv_agree(tmp_path):
    common.write_table(tmp_path / 't', 'Title', ['a', 'b'], [['x|y', None], ['z', 0.12345], ['w', True]], ['note'])
    text = (tmp_path / 't.md').read_text()
    assert text.startswith('# Title') and 'note' in text and 'x/y' in text  # pipe escaped
    rows = list(csv.reader((tmp_path / 't.csv').open()))
    assert rows == [['a', 'b'], ['x|y', 'pending'], ['z', '0.123'], ['w', 'yes']]
    with pytest.raises(ValueError):
        common.write_table(tmp_path / 'bad', 'T', ['a', 'b'], [['only one']])


def test_depth_conversion_inverts_the_controller_encoding():
    near, far = 0.007, 0.085
    mm = common.inverse_depth_to_mm(np.array([0.0, 1.0]), near, far)
    assert mm[0] == pytest.approx(far * 1000) and mm[1] == pytest.approx(near * 1000)
    # same formula as data.preprocessing.InverseDepthEncoding.encode
    depth = 0.02
    encoded = (1 / depth - 1 / far) / (1 / near - 1 / far)
    assert common.inverse_depth_to_mm(encoded, near, far) == pytest.approx(20.0)


def test_labels():
    assert common.scene_label('simcol3d/SyntheticColon_II/Frames_B14/0770') == 'Colon II B14 #0770'
    path = 'outputs/p4/runs/target_lora_v2/checkpoints/step_002000/lora_target.safetensors'
    assert common.checkpoint_label(path) == 'target_lora_v2 step 2000'


def da3_run(root, method, steps, completed, target=1000):
    run = root / f'da3_{method}_seed0'
    dump(run / 'status.json', {'completed_steps': completed, 'target_steps': target})
    for step in steps:
        macro = {'abs_rel': 0.1 / (step + 1), 'rmse_mm': 2.0, 'delta_1_25': 0.9, 'oracle_median_rmse_mm': 2.1}
        dump(run / 'validation' / f'step_{step:06d}' / 'summary.json', {'sequence_macro': macro})


def test_da3_rows_mark_unfinished_runs_pending(tmp_path):
    da3_run(tmp_path, 'R', [0, 500, 1000], 1000)
    da3_run(tmp_path, 'G', [0], 40)
    rows = tables.da3_rows(tmp_path)
    by_method = {}
    for row in rows:
        by_method.setdefault(row[0], []).append(row)
    assert [r[1] for r in by_method['R']] == [0, 500, 1000] and by_method['R'][-1][-1] == 'complete'
    assert by_method['A'][0][3] is None and 'no run found' in by_method['A'][0][-1]  # missing run
    assert by_method['G'][-1][1] == 'final' and by_method['G'][-1][3] is None
    assert 'training 40/1000' in by_method['G'][-1][-1]
    tables.da3_table(tmp_path, tmp_path / 'out' / 'da3')
    assert 'pending' in (tmp_path / 'out' / 'da3.md').read_text()
    assert 'SOURCE validation only' in (tmp_path / 'out' / 'da3.md').read_text()


def check(k, n=8, gain=0.2):
    return {'conditions_in_expected_direction': k, 'conditions': n, 'mean_gain': gain, 'gains': [gain] * n,
            'passed': k >= 7}


def gate_json(step, r3):
    checks = {k: check(8) for k in ('R0_proxy_audit', 'R1_correct_beats_off', 'R2_correct_beats_wrong')}
    checks['R3_wrong_tracks_supplied'] = check(r3)
    return {'render': {'set': 'validation', 'step': step},
            'primary_abs_rel': {'checks': checks, 'proxy_verdict': 'PASS' if r3 >= 7 else 'FAIL'},
            'secondary_spearman_negated': {'proxy_verdict': 'PASS'},
            'visual_rubric': {'passed': r3 >= 7, 'ratings': {
                'layout_follows_control': ['yes'] * 6 + ['partly'] * 2, 'wrong_depth_moves_layout': ['yes'] * 8}},
            'verdict': 'PASS' if r3 >= 7 else 'FAIL'}


def test_gate_rows_order_and_ratings(tmp_path):
    dump(tmp_path / 'g' / 'run_step1500_validation' / 'gate.json', gate_json(1500, 6))
    dump(tmp_path / 'g' / 'run_step500_validation' / 'gate.json', gate_json(500, 7))
    blind = {'rater': 'fixture rater', 'grids': {
        'run_step1500_validation/grid_seed0.png': {'yes_count': 24, 'pass': True}}}
    rows = tables.gate_rows(tmp_path / 'g', blind)
    assert [r[0] for r in rows] == ['validation step 500', 'validation step 1500']
    last = rows[1]
    assert last[5] == '6/8 (+0.200)' and last[-1] == 'FAIL'
    assert last[8] == 'layout 6/8, wrong-depth 8/8: FAIL'
    assert last[9] == '24/24 yes: pass' and last[10] == 'not rated'  # unrated grids stay unrated


def block(m3_target, n, retained):
    def entry(k):
        return {'checks': {c: check(k if c == tables.CHECKS[2] else n, n) for c in tables.CHECKS},
                'control_retained': retained or False}
    ret = {c: {'mean_gain_source': 0.4, 'mean_gain_target': 0.2, 'ratio_target_over_source': 0.5,
               'conditions_target_gain_larger': 1} for c in tables.CHECKS}
    verdict = 'RETAINED (x)' if retained else 'NOT RETAINED (y)'
    return {'per_adapter': {'none': entry(n), 'source': entry(n), 'target': entry(m3_target)},
            'retention_source_vs_target': ret, 'swap_verdict': verdict}


def test_matrix_tables_cover_every_block(tmp_path):
    r0 = check(8)
    for name, retained in (('v1', False), ('v2', False)):
        dump(tmp_path / name / 'matrix.json', {
            'render': {'seed': 5}, 'R0_audit_real_rgb': r0, 'primary_abs_rel': block(6, 8, False),
            'secondary_spearman': block(8, 8, True)})
    analysis = {'conditions': 32, 'seeds': [17, 43], 'rules': {'min_conditions': 28},
                'primary_abs_rel': {'combined': dict(block(29, 32, True), R0_audit_real_rgb=check(32, 32)),
                                    'per_seed': {str(s): dict(block(29, 32, True), R0_audit_real_rgb=check(32, 32))
                                                 for s in (17, 43)}},
                'secondary_spearman': {'combined': block(31, 32, True),
                                       'per_seed': {str(s): block(31, 32, True) for s in (17, 43)}}}
    dump(tmp_path / 'analysis.json', analysis)
    adapter_rows, retention_rows = tables.matrix_tables(
        [('v1 reduced 8x1', tmp_path / 'v1'), ('v2 reduced 8x1', tmp_path / 'v2')], tmp_path / 'analysis.json',
        tmp_path / 'out')
    assert len(adapter_rows) == 5 * 3 and len(retention_rows) == 5 * 2
    target = [r for r in adapter_rows if r[0] == 'v2 full 32x2: seed 17' and r[4] == 'target'][0]
    assert target[8] == '29/32 (+0.200)' and target[10] == 'yes'
    verdicts = {(r[0], r[2]): r[3] for r in retention_rows}
    assert verdicts[('v1 reduced 8x1', 'primary AbsRel')] == 'NOT RETAINED'
    assert verdicts[('v1 reduced 8x1', 'secondary Spearman')] == 'RETAINED'
    assert (tmp_path / 'out' / 'swap_matrix_retention.csv').exists()


def test_appearance_rows_read_from_reference_json():
    def ref(div_t, ratio):
        entry = {'latent': {'median_ratio_generated_over_heldout': ratio},
                 'pixel': {'median_ratio_generated_over_heldout': ratio + 0.1}}
        return {'diversity_mean_pairwise_l1': {'generated_source': 0.2, 'generated_target': div_t, 'real_c1_train': 0.21,
                                               'real_c1_validation': 0.18},
                'w1_rgb_mean': {'source': {'c1_train': 0.1, 'c1_validation': 0.1, 'source_train': 0.04},
                                'target': {'c1_train': 0.08, 'c1_validation': 0.09, 'source_train': 0.12},
                                'real_c1_train_vs_c1_validation': 0.03, 'real_c1_train_vs_source_train': 0.06},
                'nearest_neighbour': {'c1_train': entry, 'source_train': entry}}
    header, body = tables.appearance_rows(ref(0.11, 1.0), ref(0.16, 1.2))
    assert [r[0] for r in body][:3] == ['Source LoRA', 'Target LoRA v1', 'Target LoRA v2']
    assert body[1][1] == '0.110' and body[2][1] == '0.160' and body[2][5] == '1.20 / 1.30'
