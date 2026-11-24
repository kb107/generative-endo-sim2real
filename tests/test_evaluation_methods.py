"""Locked-campaign generalisation: arbitrary method lists, last/bbLoRA heads, per-frame medians, table order."""
import csv
import json
from pathlib import Path

import pytest
import torch

from da3_fakes import fake_bb_model, fake_model
from evaluation_fakes import FakeTarget, HW, World, make_run
from generative_sim2real.depth.checkpoints import apply_checkpoint, read_final_training, read_head
from generative_sim2real.evaluation import c2, depth as ev, report
from generative_sim2real.evaluation import registry as reg

NAMES = ['Z', 'R', 'G', 'G_last', 'G_bb', 'G_bb_last', 'G_cl.s2']


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, methods=('R', 'G'), monkeypatch=monkeypatch, model_fn=fake_bb_model)
    make_run(tmp_path, 'G_bb', 6, mode='G', run_name='da3_G_bblora_seed0', lora=True, ids=w.ids, source_val_count=2)
    make_run(tmp_path, 'G_cl.s2', 7, mode='G', run_name='da3_G_seed2', train_seed=2, ids=w.ids, source_val_count=2)
    w.spec['methods'] = {
        'Z': None, 'R': 'outputs/runs/da3_R_seed0', 'G': 'outputs/runs/da3_G_seed0',
        'G_last': {'run_dir': 'outputs/runs/da3_G_seed0', 'head': 'last'},
        'G_bb': {'run_dir': 'outputs/runs/da3_G_bblora_seed0'},
        'G_bb_last': {'run_dir': 'outputs/runs/da3_G_bblora_seed0', 'head': 'last'},
        'G_cl.s2': 'outputs/runs/da3_G_seed2'}
    w.messages = []
    yield w
    torch.use_deterministic_algorithms(False)


def campaign(w, **kw):
    return c2.run_campaign(w.registry_path, root=w.root, campaign_dir=w.campaign, code_root=w.root,
                           dataset_factory=w.dataset_factory, model_factory=w.model_factory, log=w.messages.append, **kw)


def read_csv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def reference_row(w, name, index):
    """Independent evaluation of one frame with a fresh model set up by hand (no campaign code)."""
    record = next(m for m in json.loads(w.registry_path.read_text())['methods'] if m['name'] == name)
    model = fake_bb_model()
    c2.set_processing(model, HW)
    if record['kind'] == 'official':
        pass
    else:
        path = w.root / record['head']['path']
        checkpoint = read_final_training(path) if record.get('head_choice') == 'last' else read_head(
            path, allow_backbone_lora=True)
        apply_checkpoint(model, checkpoint)
    sample = FakeTarget(w.rows)[index]
    model.eval()
    with torch.no_grad():
        pred = model(sample['rgb'][None], sample['camera']['K'][None])['depth_m'][0]
    return ev.frame_metrics(pred, sample['targets']['depth_m'], sample['targets']['depth_valid'])


def test_campaign_runs_every_arm_in_spec_order_and_matches_independent_evaluation(world):
    world.freeze()
    out = Path(campaign(world)['output_dir'])
    assert json.loads((out / 'comparison.json').read_text())['methods'] == NAMES
    assert [r['method'] for r in read_csv(out / 'comparison_headline.csv')] == NAMES
    assert [r['method'] for r in read_csv(out / 'comparison_matched_subset.csv')] == NAMES
    per_sequence = read_csv(out / 'comparison_per_sequence.csv')
    assert [r['method'] for r in per_sequence[:len(NAMES)]] == NAMES and len(per_sequence) == 3 * len(NAMES)
    assert sorted(p.name for p in (out / 'methods').iterdir()) == sorted(NAMES)
    assert 'not registered' not in (out / 'comparison.md').read_text() and 'method comparison' in (out / 'comparison.md').read_text()
    assert json.loads((out / 'comparison.json').read_text())['not_registered'] == []
    provenance = {r['method']: r for r in read_csv(out / 'comparison_provenance.csv')}
    assert [r for r in provenance] == NAMES
    assert provenance['Z']['variant'] == 'official' and provenance['Z']['head'] == 'official' and provenance['Z']['training_seed'] == ''
    assert (provenance['G_bb']['variant'], provenance['G_bb']['head'], provenance['G_bb']['training_seed']) == ('head+backbone_lora', 'best', '0')
    assert (provenance['G_bb_last']['head'], provenance['G_bb_last']['selected_step']) == ('last', '4')
    assert (provenance['G_cl.s2']['training_seed'], provenance['G_cl.s2']['variant']) == ('2', 'head')
    markdown = (out / 'comparison.md').read_text()
    assert '## Method provenance' in markdown and '| G_bb | 0 | head+backbone_lora | best |' in markdown
    abs_rel = {}
    for name in NAMES:
        frames = read_csv(out / 'methods' / name / 'frames.csv')
        assert [f['sample_id'] for f in frames] == [r['sample_id'] for r in world.rows]
        abs_rel[name] = float(frames[4]['abs_rel'])
        expected = reference_row(world, name, 4)
        for key in ('abs_rel', 'rmse_mm', 'oracle_median_rmse_mm', 'median_gt_m', 'median_pred_m'):
            assert float(frames[4][key]) == pytest.approx(expected[key], rel=1e-9), (name, key)
    # best vs last of the same run and the LoRA arms are genuinely different models; head-only arms after a LoRA arm
    # (G_cl.s2 is evaluated last) carry no leftover adapter, which the independent comparison above also proves.
    assert len({round(v, 9) for v in abs_rel.values()}) == len(NAMES)
    json.loads((out / 'methods' / 'G_bb' / 'frames.json').read_text())     # allow_nan=False output is finite


def test_lora_adapter_changes_predictions_and_is_removed_for_later_arms(world):
    world.freeze()
    registry = json.loads(world.registry_path.read_text())
    model = fake_bb_model()
    c2.set_processing(model, HW)
    states = c2.prepare_heads(model, registry, world.root)
    assert set(states) == set(NAMES) and 'backbone_lora_state_dict' in states['G_bb'] and 'backbone_lora_state_dict' not in states['G']
    sample = FakeTarget(world.rows)[0]
    rgb, K = sample['rgb'][None], sample['camera']['K'][None]
    with torch.no_grad():
        apply_checkpoint(model, states['G'])
        plain = model(rgb, K)['depth_m']
        apply_checkpoint(model, {**states['G_bb'], 'head_state_dict': states['G']['head_state_dict']})
        with_lora = model(rgb, K)['depth_m']
        assert model.backbone_lora is not None and not torch.allclose(plain, with_lora)
        apply_checkpoint(model, states['G'])
        assert model.backbone_lora is None and not model._lora_hooks
        assert torch.equal(model(rgb, K)['depth_m'], plain)


def test_prepare_heads_rejects_a_registry_that_disagrees_with_the_lora_state(world):
    world.freeze()
    registry = json.loads(world.registry_path.read_text())
    record = next(m for m in registry['methods'] if m['name'] == 'G_bb')
    record['backbone_lora']['sha256'] = '0' * 64
    with pytest.raises(reg.RegistryError, match='backbone-LoRA'):
        c2.prepare_heads(fake_bb_model(), registry, world.root)
    registry = json.loads(world.registry_path.read_text())
    next(m for m in registry['methods'] if m['name'] == 'G_bb').pop('backbone_lora')
    with pytest.raises(reg.RegistryError, match='backbone-LoRA presence'):
        c2.prepare_heads(fake_bb_model(), registry, world.root)
    registry = json.loads(world.registry_path.read_text())
    next(m for m in registry['methods'] if m['name'] == 'G_last').pop('head_choice')
    with pytest.raises(reg.RegistryError, match='head choice'):
        c2.prepare_heads(fake_bb_model(), registry, world.root)


def test_lora_checkpoint_file_swapped_after_freezing_is_refused(world):
    world.freeze()
    run = world.root / 'outputs/runs/da3_G_bblora_seed0'
    value = torch.load(run / 'best.pt', weights_only=False)
    key = next(iter(value['backbone_lora_state_dict']))
    value['backbone_lora_state_dict'][key] = value['backbone_lora_state_dict'][key] + 1
    torch.save(value, run / 'best.pt')
    with pytest.raises(reg.RegistryError):
        campaign(world)
    assert not world.ledger.exists()


# ---------------------------------------------------------------- per-frame medians

def test_frame_metrics_report_raw_medians_without_changing_the_metrics():
    gt = torch.tensor([[1., 2., 3., 4., 9.]])
    pred = torch.tensor([[2., 3., 4., 8., 9.]])
    valid = torch.tensor([[True, True, True, True, False]])           # the invalid pixel must not enter the medians
    m = ev.frame_metrics(pred, gt, valid)
    assert m['median_gt_m'] == 2.0 and m['median_pred_m'] == 3.0       # lower median of 4 values
    assert m['oracle_median_scale'] == pytest.approx(2.0 / 3.0)
    assert m['abs_rel'] == pytest.approx((1 + .5 + 1 / 3 + 1) / 4)
    assert m['rmse_mm'] == pytest.approx(1000 * (19 / 4) ** .5)
    assert set(ev.METRICS) <= set(m)


def test_per_frame_csv_and_json_carry_the_medians_and_failed_rows_leave_them_empty(tmp_path):
    ok = report.ok_row(ev.frame_metrics(torch.full((1, 4), 1.2), torch.ones(1, 4), torch.ones(1, 4, dtype=torch.bool)),
                       's/0', 's')
    failed = report.failed_row('s/1', 's', 'bad', 4, 4)
    summary = report.method_summary([ok, failed], {'s': 2}, {})
    report.write_method_report(tmp_path, [ok, failed], summary)
    rows = list(csv.DictReader((tmp_path / 'frames.csv').open()))
    assert 'median_gt_m' in rows[0] and 'median_pred_m' in rows[0]
    assert float(rows[0]['median_gt_m']) == 1.0 and float(rows[0]['median_pred_m']) == pytest.approx(1.2)
    assert rows[1]['median_gt_m'] == '' and rows[1]['median_pred_m'] == ''
    frames = json.loads((tmp_path / 'frames.json').read_text())
    assert frames[0]['median_pred_m'] == pytest.approx(1.2) and frames[1]['median_gt_m'] is None


# ---------------------------------------------------------------- tables for arbitrary method lists

def fake_rows(scale, sequences=(('a', 2), ('b', 1))):
    rows = []
    for seq, count in sequences:
        for i in range(count):
            gt = torch.ones(1, 4)
            rows.append(report.ok_row(ev.frame_metrics(gt * scale, gt, torch.ones(1, 4, dtype=torch.bool)), f'{seq}/{i}', seq))
    return rows


def test_comparison_keeps_registry_order_for_arbitrary_names_and_legacy_lists_keep_absent_notes():
    counts = {'a': 2, 'b': 1}
    names = ['Z', 'zeta_arm', 'G', 'G_cl+s1']
    rows = {n: fake_rows(1.0 + .1 * i) for i, n in enumerate(names)}
    summaries = {n: report.method_summary(r, counts, {'method': n}) for n, r in rows.items()}
    tables = report.comparison(summaries, names, counts, rows)
    assert tables['methods'] == names and tables['not_registered'] == []
    for key in ('headline_sequence_macro_raw', 'secondary_frame_macro_raw', 'secondary_pixel_global_raw',
                'matched_subset_sequence_macro_raw', 'oracle_diagnostic_median_scaled'):
        assert [r['method'] for r in tables[key]] == names, key
    assert [r['method'] for r in tables['per_sequence']] == names * 2
    assert 'not registered' not in report.markdown(tables)
    legacy = ['Z', 'R']
    tables = report.comparison({n: summaries['Z'] for n in legacy}, legacy, counts, {n: rows['Z'] for n in legacy})
    assert tables['not_registered'] == ['A', 'G', 'M']
    with pytest.raises(ValueError):
        report.comparison({'Z': summaries['Z']}, ['Z', 'Z'], counts, {'Z': rows['Z']})


def test_cli_source_validation_metadata_records_the_lora_digest(tmp_path, monkeypatch):
    import sys
    from da3_fakes import FakeSplit
    from generative_sim2real.data import manifests
    from generative_sim2real.depth import checkpoints as ck, model as depth_model
    w = World(tmp_path, methods=('R', 'G'), monkeypatch=monkeypatch, model_fn=fake_bb_model)
    make_run(tmp_path, 'G_bb', 6, mode='G', run_name='da3_G_bblora_seed0', lora=True, ids=w.ids, source_val_count=2)
    split = FakeSplit(tmp_path, 'source_validation', 'source_native', 2, 56)
    monkeypatch.setattr(manifests, 'ManifestDataset', lambda *a, **k: split)
    monkeypatch.setattr(depth_model, 'verify_upstream_checkout', lambda: None)
    monkeypatch.setattr(depth_model.DA3MetricDepth, 'from_pretrained', classmethod(lambda cls, **k: fake_bb_model()))
    for run, expect_lora in (('da3_G_bblora_seed0', True), ('da3_G_seed0', False)):
        best = tmp_path / 'outputs/runs' / run / 'best.pt'
        monkeypatch.setattr(sys, 'argv', ['x', '--manifest', str(split.path), '--head', str(best), '--output',
                                          str(tmp_path / f'eval_{run}'), '--config', 'configs/da3_smoke.yaml'])
        ev.main()
        head = json.loads((tmp_path / f'eval_{run}/summary.json').read_text())['metadata']['head']
        assert ('backbone_lora_sha256' in head) == expect_lora
        if expect_lora:
            assert head['backbone_lora_sha256'] == ck.read_head(best, allow_backbone_lora=True)['backbone_lora_sha256']
