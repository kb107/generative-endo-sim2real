import math

import pytest
import torch

from da3_fakes import FakeSplit, fake_model
from generative_sim2real.evaluation import depth as ev


def row(sid, seq, pred, gt, valid=None):
    pred, gt = torch.tensor(pred, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32)
    valid = torch.ones_like(gt, dtype=torch.bool) if valid is None else torch.tensor(valid, dtype=torch.bool)
    return {'sample_id': sid, 'sequence_id': seq, **ev.frame_metrics(pred, gt, valid)}


def test_frame_metric_formulas_by_hand():
    gt = [[0.010, 0.020, 0.040, 0.100]]
    pred = [[0.0105, 0.020, 0.030, 0.140]]
    m = row('a', 's', pred, gt)
    p = torch.tensor(pred[0], dtype=torch.float64)
    g = torch.tensor(gt[0], dtype=torch.float64)
    assert m['valid_pixels'] == 4
    assert m['abs_rel'] == pytest.approx(((p - g).abs() / g).mean().item())
    assert m['rmse_mm'] == pytest.approx(1000 * math.sqrt(((p - g) ** 2).mean().item()))
    # symmetric ratios 1.05, 1.0, 1.333, 1.4
    assert m['delta_1_1'] == .5
    assert m['delta_1_25'] == .5
    assert m['squared_error_sum_m2'] == pytest.approx(((p - g) ** 2).sum().item())


def test_delta_thresholds_are_strict_and_ratio_symmetric():
    m = row('a', 's', [[1.0, 1.0, 1.0, 1.0]], [[1.05, 1 / 1.05, 1.2, 1 / 1.3]])
    assert m['delta_1_1'] == .5 and m['delta_1_25'] == .75


def test_invalid_pixels_are_ignored_and_failures_are_not_hidden():
    gt = torch.tensor([[1., 2., float('nan'), 0.]])
    valid = torch.tensor([[True, True, False, False]])
    pred = torch.tensor([[1.5, 2., float('nan'), -5.]])
    m = ev.frame_metrics(pred, gt, valid)
    assert m['valid_pixels'] == 2 and m['abs_rel'] == pytest.approx(.25)
    with pytest.raises(ValueError):          # nonfinite prediction at a valid pixel
        ev.frame_metrics(torch.tensor([[1.5, float('nan')]]), torch.tensor([[1., 2.]]), torch.tensor([[True, True]]))
    with pytest.raises(ValueError):          # nonpositive prediction at a valid pixel
        ev.frame_metrics(torch.tensor([[1.5, -1.]]), torch.tensor([[1., 2.]]), torch.tensor([[True, True]]))
    with pytest.raises(ValueError):
        ev.frame_metrics(pred, gt, torch.zeros_like(valid))      # no valid pixels
    with pytest.raises(ValueError):
        ev.frame_metrics(pred, torch.tensor([[1., 2., 3., 4.]]), valid.float())


def test_oracle_median_scale_is_separate_from_raw_error():
    gt = [[1., 2., 3.]]
    m = row('a', 's', [[2., 4., 6.]], gt)          # exactly a factor-two scale error
    assert m['abs_rel'] == pytest.approx(1.0)       # raw metric is uncorrected
    assert m['oracle_median_scale'] == pytest.approx(.5)
    assert m['oracle_median_rmse_mm'] == pytest.approx(0.0, abs=1e-9)
    assert m['rmse_mm'] > 1000


def test_aggregation_levels_differ_and_pooled_rmse_is_not_mean_of_rmse():
    a1 = row('a1', 'A', [[1.1] * 100], [[1.] * 100])           # long frame, small error
    a2 = row('a2', 'A', [[1.1] * 100], [[1.] * 100])
    b1 = row('b1', 'B', [[3.]], [[1.]])                       # one tiny frame, huge error
    summary = ev.aggregate_frames([a1, a2, b1])
    assert summary['frame_count'] == 3 and summary['sequence_count'] == 2 and summary['valid_pixels'] == 201
    seq = {s['sequence_id']: s for s in summary['sequences']}
    assert seq['A']['frame_count'] == 2 and seq['A']['abs_rel'] == pytest.approx(.1)
    assert summary['sequence_macro']['abs_rel'] == pytest.approx((.1 + 2.) / 2)
    assert summary['frame_macro']['abs_rel'] == pytest.approx((.1 + .1 + 2.) / 3)
    expected_pixel = (200 * .1 + 2.) / 201
    assert summary['pixel_global']['abs_rel'] == pytest.approx(expected_pixel)
    pooled_rmse = 1000 * math.sqrt((200 * .1 ** 2 + 2. ** 2) / 201)
    assert summary['pixel_global']['rmse_mm'] == pytest.approx(pooled_rmse, rel=1e-5)
    assert summary['pixel_global']['rmse_mm'] != pytest.approx(summary['sequence_macro']['rmse_mm'], rel=1e-3)


def test_aggregation_rejects_duplicate_or_empty_rows():
    r = row('a', 's', [[1.]], [[1.]])
    with pytest.raises(ValueError):
        ev.aggregate_frames([])
    with pytest.raises(ValueError):
        ev.aggregate_frames([r, r])


def test_report_files_have_counts_and_finite_json(tmp_path):
    rows = [row('a', 'A', [[1.1]], [[1.]]), row('b', 'B', [[1.2]], [[1.]])]
    summary = ev.write_report(tmp_path / 'r', rows, {'note': 'fixture'})
    assert {p.name for p in (tmp_path / 'r').iterdir()} == {
        'frames.json', 'frames.csv', 'sequences.json', 'sequences.csv', 'summary.json'}
    assert summary['sequence_count'] == 2
    assert 'sequence_id,frame_count' in (tmp_path / 'r/sequences.csv').read_text().splitlines()[0]


def test_validation_indices_subset_is_deterministic_and_bounded():
    assert ev.validation_indices(5, None) == [0, 1, 2, 3, 4]
    assert ev.validation_indices(10, 4) == [0, 3, 6, 9]
    for bad in (0, 11, 2.5):
        with pytest.raises(ValueError):
            ev.validation_indices(10, bad)


def test_evaluate_source_resizes_prediction_back_and_restores_mode(tmp_path):
    data = FakeSplit(tmp_path, 'source_validation', 'source_native', 4, 56, name='val')
    model = fake_model()
    model.configure_head_training()
    model.train()
    rows = ev.evaluate_source(model, data, 'cpu', indices=[0, 3])
    assert [r['sample_id'] for r in rows] == [data.rows[0]['sample_id'], data.rows[3]['sample_id']]
    assert all(r['valid_pixels'] == (56 - 7) * 56 for r in rows)      # the invalid band is excluded
    assert model.training and model.network.head.training and not model.network.backbone.training


@pytest.mark.parametrize('split,geometry', [('target_test', 'target_native'), ('source_train', 'source_native'),
                                            ('source_validation', 'source_da3_504')])
def test_training_validation_refuses_non_source_validation(tmp_path, split, geometry):
    data = FakeSplit(tmp_path, split, geometry, 2, 56)
    with pytest.raises(ValueError):
        ev.evaluate_source(fake_model(), data)


def test_frame_subsets_are_refused_outside_smoke():
    ev.require_smoke_for_subset(None, False, None)             # a full evaluation is always allowed
    ev.require_smoke_for_subset(16, True, None)
    ev.require_smoke_for_subset(16, None, 'smoke')
    for head, purpose in ((False, None), (None, None), (False, 'experiment'), (None, 'experiment')):
        with pytest.raises(ValueError):
            ev.require_smoke_for_subset(16, head, purpose)
