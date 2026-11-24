import csv
import fcntl
import json
import os
from pathlib import Path

import pytest
import torch

from da3_fakes import FakeNetwork, FakeSplit, fake_model, intrinsics
from evaluation_fakes import FakeTarget, HW, World, put
from generative_sim2real.depth.model import DA3MetricDepth
from generative_sim2real.evaluation import c2, depth as ev, ledger, report
from generative_sim2real.evaluation import registry as reg


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch=monkeypatch)
    w.messages = []
    yield w
    torch.use_deterministic_algorithms(False)


def campaign(w, dry_run=False, **kw):
    return c2.run_campaign(w.registry_path, root=w.root, campaign_dir=w.campaign, dry_run=dry_run, code_root=w.root,
                           dataset_factory=w.dataset_factory, model_factory=w.model_factory,
                           log=getattr(w, 'messages', []).append, **kw)


def counting_model_factory(w):
    w.model_calls = 0

    def factory():
        w.model_calls += 1
        return fake_model()
    w.model_factory = factory


def assert_no_c2_touch(w):
    assert getattr(w, 'model_calls', 0) == 0
    assert w.dataset is None or w.dataset.decoded == 0
    assert not w.ledger.exists() and not (w.campaign / '.seals').exists()


def read_csv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------- refusals

def test_refuses_without_registry(world):
    counting_model_factory(world)
    with pytest.raises(reg.RegistryError, match='No readable'):
        campaign(world)
    assert_no_c2_touch(world)


def test_refuses_unfrozen_registry(world):
    counting_model_factory(world)
    world.build()
    with pytest.raises(reg.RegistryError, match='not frozen'):
        campaign(world)
    assert_no_c2_touch(world)


def test_refuses_modified_hash_in_any_input(world):
    counting_model_factory(world)
    world.freeze()
    put(world.root / 'outputs/runs/da3_A_seed0/best.pt', 'tampered')
    with pytest.raises(reg.RegistryError, match='cannot be re-verified'):
        campaign(world)
    assert_no_c2_touch(world)


def test_refuses_modified_registry_hash_and_expected_hash(world):
    counting_model_factory(world)
    world.freeze()
    with pytest.raises(reg.RegistryError, match='expected hash'):
        campaign(world, expected_sha256='1' * 64)
    value = json.loads(world.registry_path.read_text())
    value['freeze']['registry_sha256'] = '2' * 64
    world.registry_path.write_text(json.dumps(value))
    with pytest.raises(reg.RegistryError, match='hash'):
        campaign(world)
    assert_no_c2_touch(world)


def test_refuses_a_head_whose_run_used_target_views(world):
    counting_model_factory(world)
    world.freeze()
    path = world.root / 'outputs/runs/da3_R_seed0/resolved.json'
    value = json.loads(path.read_text())
    value['config']['evaluation']['target_views'] = ['target_appearance_validation']
    path.write_text(json.dumps(value))
    with pytest.raises(reg.RegistryError, match='target_views'):
        campaign(world)
    assert_no_c2_touch(world)


def test_refuses_dataset_rows_that_differ_from_the_registry(world):
    world.freeze()
    world.dataset_factory = lambda registry, root: FakeTarget(world.rows[:-1])
    with pytest.raises(reg.RegistryError, match='differ from the registry'):
        campaign(world)


def test_refuses_weights_that_differ_from_registry_digests(world):
    world.freeze()

    def altered():
        model = fake_model()
        with torch.no_grad():
            model.network.backbone.weight.add_(1.)
        return model
    world.model_factory = altered
    with pytest.raises(reg.RegistryError, match='digests'):
        campaign(world)
    assert not world.ledger.exists()


def test_refuses_when_runtime_versions_differ_from_the_registry(world, monkeypatch):
    counting_model_factory(world)
    world.freeze()
    monkeypatch.setattr(reg, 'runtime_requirements', lambda: {'python': '3', 'torch': '0.0.1', 'numpy': '0',
                                                              'xformers': None, 'torch_cuda': '0.0'})
    with pytest.raises(reg.RegistryError, match='Runtime differs'):
        campaign(world)
    assert_no_c2_touch(world)
    assert any('runtime differs' in m for m in campaign(world, dry_run=True)['would_refuse'])


# ---------------------------------------------------------------- dry run

def test_dry_run_loads_no_model_no_pixels_and_writes_nothing(world):
    counting_model_factory(world)
    world.freeze()
    result = campaign(world, dry_run=True)
    assert result['model_loaded'] is False and result['pixels_decoded'] is False and result['would_refuse'] == []
    assert result['official_digests_recomputed_from_weights_file'] and result['governing_files_rehashed_on_disk']
    assert result['frames'] == 9 and result['methods'] == ['Z', 'R', 'A'] and result['declared_absent'] == ['G', 'M']
    assert result['process_hw'] == list(HW)
    assert result['ledger']['experiment_index'] == 1 and not result['ledger']['new_disclosed_experiment']
    assert world.dataset.decoded == 0
    assert_no_c2_touch(world)


def test_dry_run_recomputes_official_digests_instead_of_echoing_the_registry(world):
    world.freeze()
    world.digests['official_head_sha256'] = '8' * 64
    with pytest.raises(reg.RegistryError, match='recomputed'):
        campaign(world, dry_run=True)


def test_dry_run_reports_inconsistent_campaign_state(world):
    world.freeze()
    (world.campaign / 'abcdef012345').mkdir(parents=True)
    result = campaign(world, dry_run=True)
    assert result['would_refuse'] and 'no ledger start' in result['would_refuse'][0]


@pytest.mark.skipif(not (Path('outputs/runs/da3_R_seed0/best.pt').exists() and os.environ.get('DATA_ROOT')
                         and Path('outputs/manifests/milestone_v1/target_test.jsonl').exists()),
                    reason='needs the real runs, manifests and DATA_ROOT')
def test_real_adapter_dry_run_decodes_no_pixel(tmp_path, monkeypatch):
    pytest.importorskip('safetensors')
    pytest.importorskip('huggingface_hub')
    import cv2
    from PIL import Image
    root = Path('.').resolve()
    spec = reg.load_spec('configs/registry.yaml')
    try:
        digests = reg.official_digests_from_files()
    except reg.RegistryError:
        pytest.skip('pinned official checkpoint is not cached')
    registry_path = tmp_path / 'registry.json'
    reg.build_registry(spec, root, registry_path, digests=digests)
    reg.freeze_registry(registry_path, root)

    def forbidden(*args, **kwargs):
        raise AssertionError('pixel decode attempted during dry run')
    monkeypatch.setattr(cv2, 'imread', forbidden)
    monkeypatch.setattr(Image, 'open', forbidden)
    result = c2.run_campaign(registry_path, root=root, dry_run=True, campaign_dir=tmp_path / 'c2',
                             model_factory=lambda: (_ for _ in ()).throw(AssertionError('model loaded')))
    assert result['frames'] == 2589 and result['pixels_decoded'] is False and result['would_refuse'] == []


# ---------------------------------------------------------------- native 504x630-style processing (H1)

class StubNetwork(FakeNetwork):
    """Returns a constant canonical depth so the metric scale convention is observable."""
    value = .05

    def forward(self, x, **kwargs):
        self.calls.append((x.detach().clone(), kwargs))
        return {'depth': torch.full((x.shape[0], x.shape[-2], x.shape[-1]), self.value)}


def test_native_grid_is_not_squashed_to_a_square():
    h, w = 504, 630
    model = fake_model()
    c2.set_processing(model, (h, w))
    data = FakeTarget([{'sample_id': 'c3vdv2/c2/s/0000', 'sequence_id': 's'}], hw=(h, w))
    data.split, data.geometry = 'target_test', 'target_rectified_full'
    rows = c2.evaluate_dataset(model, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=(h, w),
                               exact_grid=True)
    assert rows[0]['status'] == 'ok'
    network_input = model.network.calls[0][0]
    assert tuple(network_input.shape) == (1, 1, 3, h, w)               # never 504x504


def test_metric_scale_follows_the_unstretched_focal_convention():
    h, w = HW
    model = DA3MetricDepth(StubNetwork(), process_hw=HW)
    rgb = torch.rand(1, 3, h, w)
    k300 = torch.tensor([[[300., 0, w / 2], [0, 300., h / 2], [0, 0, 1]]])
    out = model(rgb, k300)
    assert torch.equal(out['K_processed'], k300)                       # K untouched at the native grid
    assert torch.allclose(out['depth_m'], torch.full((1, 1, h, w), .05))    # canonical * 300/300
    out600 = model(rgb, k300 * torch.tensor([2., 2., 1.]).diag())
    assert torch.allclose(out600['depth_m'], torch.full((1, 1, h, w), .10))
    # A square-squashed processing grid would rescale K anisotropically and change the metric scale.
    squashed = DA3MetricDepth(StubNetwork(), process_hw=(h, h))
    assert not torch.equal(squashed(rgb, k300)['K_processed'], k300)


def test_exact_grid_aborts_instead_of_resizing(tmp_path):
    data = FakeTarget([{'sample_id': 'c3vdv2/c2/s/0000', 'sequence_id': 's'}])
    data.split, data.geometry = 'target_test', 'target_rectified_full'
    model = fake_model()                                               # default 504x504 processing, 56x70 frames
    with pytest.raises(ValueError, match='resizing is not permitted'):
        c2.evaluate_dataset(model, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=HW,
                            exact_grid=True)


def test_processing_size_must_be_a_multiple_of_14():
    with pytest.raises(ValueError):
        c2.set_processing(fake_model(), (504, 631))


def test_campaign_sets_native_processing_and_pins_it_in_the_registry(world):
    registry = world.freeze()
    assert registry['protocol']['model_input']['process_hw'] == list(HW)
    seen = []
    def factory():
        model = fake_model()
        seen.append(model)
        return model
    world.model_factory = factory
    campaign(world)
    assert seen[0].process_hw == HW
    shapes = {tuple(x.shape) for x, _ in seen[0].network.calls}
    assert shapes == {(1, 1, 3, *HW)}
    assert torch.are_deterministic_algorithms_enabled()


# ---------------------------------------------------------------- full fixture campaign

def test_campaign_outputs_equal_rows_counts_and_ledger(world):
    world.freeze()
    result = campaign(world)
    out = Path(result['output_dir'])
    assert (out / 'COMPLETE').exists() and (out / 'registry.json').exists() and (out / 'campaign.json').exists()
    ids = [r['sample_id'] for r in world.rows]
    for name in ('Z', 'R', 'A'):
        frames = read_csv(out / 'methods' / name / 'frames.csv')
        assert [f['sample_id'] for f in frames] == ids and all(f['status'] == 'ok' for f in frames)
        sequences = json.loads((out / 'methods' / name / 'sequences.json').read_text())
        assert {s['sequence_id']: s['total_frames'] for s in sequences} == {'c2_a_v1': 3, 'c2_b_v1': 2, 'c2_c_v2': 4}
        assert all(s['gt_valid_pixels'] == s['total_frames'] * 56 * (70 - 14) for s in sequences)
    summaries = {n: json.loads((out / 'methods' / n / 'summary.json').read_text()) for n in ('Z', 'R', 'A')}
    assert len({s['gt_valid_pixels'] for s in summaries.values()}) == 1
    assert summaries['R']['sequence_macro']['abs_rel'] != summaries['A']['sequence_macro']['abs_rel']
    # Independent recomputation of one frame through the model for method R at native processing.
    model = fake_model()
    c2.set_processing(model, HW)
    from generative_sim2real.depth.checkpoints import read_head
    model.load_head_state_dict(read_head(world.root / 'outputs/runs/da3_R_seed0/best.pt')['head_state_dict'])
    sample = FakeTarget(world.rows)[4]
    pred = model(sample['rgb'][None], sample['camera']['K'][None])['depth_m'][0]
    expected = ev.frame_metrics(pred, sample['targets']['depth_m'], sample['targets']['depth_valid'])
    row = read_csv(out / 'methods' / 'R' / 'frames.csv')[4]
    assert float(row['abs_rel']) == pytest.approx(expected['abs_rel'], rel=1e-9)
    assert float(row['oracle_median_rmse_mm']) == pytest.approx(expected['oracle_median_rmse_mm'], rel=1e-9)
    headline = read_csv(out / 'comparison_headline.csv')
    assert [r['method'] for r in headline] == ['Z', 'R', 'A']
    assert 'oracle_median_rmse_mm' not in headline[0] and 'abs_rel' in headline[0]
    markdown = (out / 'comparison.md').read_text()
    assert 'ORACLE' in markdown and 'G: not registered' in markdown and 'Matched subset' in markdown
    matched = read_csv(out / 'comparison_matched_subset.csv')
    assert [r['matched_frames'] for r in matched] == ['9', '9', '9']
    assert len(read_csv(out / 'comparison_per_sequence.csv')) == 9
    assert read_csv(out / 'comparison_oracle_diagnostic.csv')[0]['note'] == 'ORACLE DIAGNOSTIC'
    entries = ledger.read_ledger(world.campaign)
    assert [e['event'] for e in entries] == ['start', 'complete']
    assert entries[0]['registry_sha256'] == result['registry_sha256'] and entries[0]['experiment_index'] == 1
    assert (world.campaign / '.seals' / result['registry_sha256']).exists()
    assert (world.campaign / '.seals' / f'{result["registry_sha256"]}.complete').exists()
    assert any(result['registry_sha256'] in m and 'REGISTRY SHA256' in m for m in world.messages)
    assert not list(world.campaign.glob('*.tmp*')) and not list(world.campaign.glob('ledger.jsonl.tmp*'))
    ledger.audit(world.campaign)


def test_second_run_of_a_completed_registry_is_refused(world):
    world.freeze()
    campaign(world)
    with pytest.raises(ledger.LedgerError, match='already completed'):
        campaign(world)


def test_changed_registry_is_a_new_disclosed_experiment(world):
    first = world.freeze()
    campaign(world)
    world.registry_path.unlink()
    world.spec['methods'].pop('A')
    world.freeze()
    counting_model_factory(world)
    with pytest.raises(ledger.LedgerError, match='new disclosed experiment'):
        campaign(world)
    assert world.model_calls == 1                      # refused at the ledger, after head verification, before any C2 pixel
    assert world.dataset.decoded == 0
    dry = campaign(world, dry_run=True)
    assert dry['would_refuse']
    result = campaign(world, disclosure='drop A: its run is invalid')
    out = Path(result['output_dir'])
    assert 'NEW DISCLOSED EXPERIMENT #2' in (out / 'comparison.md').read_text()
    entries = ledger.read_ledger(world.campaign)
    assert [e['event'] for e in entries] == ['start', 'complete', 'start', 'complete']
    assert entries[2]['new_disclosed_experiment'] and entries[2]['disclosure'] == 'drop A: its run is invalid'
    assert entries[2]['earlier_registries'] == [first['freeze']['registry_sha256']]
    old = first['freeze']['registry_sha256'][:12]
    assert out.name != old and (world.campaign / old / 'COMPLETE').exists()


def broken_model_factory():
    model = fake_model()
    model.forward = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom'))
    return model


def test_abort_then_other_registry_then_retry_discloses_the_other_registry(world):
    first = world.freeze()
    sha_a = first['freeze']['registry_sha256']
    good = world.model_factory
    world.model_factory = broken_model_factory
    with pytest.raises(RuntimeError):
        campaign(world)                                # A: start, abort
    world.registry_path.unlink()
    world.spec['methods'].pop('A')
    second = world.freeze()
    sha_b = second['freeze']['registry_sha256']
    world.model_factory = good
    campaign(world, disclosure='A aborted on a code bug in the harness')    # B: new disclosed, completes
    world.registry_path.unlink()
    world.spec['methods']['A'] = 'outputs/runs/da3_A_seed0'
    assert world.freeze()['freeze']['registry_sha256'] == sha_a
    result = campaign(world)                           # A retry: same hash, no disclosure needed, but a banner
    entries = ledger.read_ledger(world.campaign)
    assert [e['event'] for e in entries] == ['start', 'abort', 'start', 'complete', 'start', 'complete']
    assert entries[4]['attempt'] == 2 and not entries[4]['new_disclosed_experiment']
    assert entries[4]['other_registries'] == [sha_b]
    banner = (Path(result['output_dir']) / 'comparison.md').read_text()
    assert 'DISCLOSURE' in banner and sha_b[:12] in banner
    assert list(world.campaign.glob('*.incomplete_*'))
    ledger.audit(world.campaign)


def test_ledger_tampering_is_detected(world):
    world.freeze()
    campaign(world)
    lines = world.ledger.read_text().splitlines()
    world.ledger.write_text(lines[1] + '\n')
    with pytest.raises(ledger.LedgerError, match='chain'):
        ledger.read_ledger(world.campaign)


# ---------------------------------------------------------------- one-shot enforcement (H2/M1)

def test_cli_has_no_ledger_or_output_overrides():
    for flag in ('--ledger', '--output-root'):
        with pytest.raises(SystemExit):
            c2.main(['run', '--registry', 'r.json', flag, 'x'])


def test_result_directory_without_a_ledger_start_is_refused(world):
    world.freeze()
    fake = world.campaign / 'abcdef012345'
    put(fake / 'registry.json', '{}')
    counting_model_factory(world)
    with pytest.raises(ledger.LedgerError, match='no ledger start'):
        campaign(world)
    assert world.dataset.decoded == 0 and not world.ledger.exists()


@pytest.mark.parametrize('damage', ['delete_ledger', 'empty_ledger', 'drop_tail_complete'])
def test_ledger_truncation_or_deletion_is_detected_by_seals(world, damage):
    world.freeze()
    campaign(world)
    lines = world.ledger.read_text().splitlines()
    if damage == 'delete_ledger':
        world.ledger.unlink()
    elif damage == 'empty_ledger':
        world.ledger.write_text('')
    else:
        world.ledger.write_text(lines[0] + '\n')               # chain stays valid: only the seal reveals the cut
    with pytest.raises(ledger.LedgerError, match='inconsistent'):
        ledger.audit(world.campaign)
    # ...and a different registry then cannot start "fresh" either.
    world.registry_path.unlink()
    world.spec['methods'].pop('A')
    world.freeze()
    with pytest.raises(ledger.LedgerError, match='inconsistent'):
        campaign(world, disclosure='trying to look first')
    assert world.dataset.decoded <= 9                           # only the first, legitimate campaign decoded frames


def test_concurrent_launch_is_refused_while_the_lock_is_held(world):
    world.freeze()
    (world.campaign / '.seals').mkdir(parents=True)
    with open(world.campaign / '.seals' / '.lock', 'a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ledger.LedgerError, match='lock'):
            campaign(world)
    assert not world.ledger.exists()


def test_double_launch_while_running_is_refused(world):
    world.freeze()
    inner = {}

    class Reentrant(FakeTarget):
        def __getitem__(self, index):
            if index == 0 and 'error' not in inner and 'ran' not in inner:
                inner['ran'] = True
                try:
                    c2.run_campaign(world.registry_path, root=world.root, campaign_dir=world.campaign,
                                    code_root=world.root, dataset_factory=lambda r, root: FakeTarget(world.rows),
                                    model_factory=fake_model, log=lambda *_: None)
                except ledger.LedgerError as error:
                    inner['error'] = str(error)
            return super().__getitem__(index)
    world.dataset_factory = lambda registry, root: Reentrant(world.rows)
    campaign(world)
    assert 'still running' in inner['error']
    assert [e['event'] for e in ledger.read_ledger(world.campaign)] == ['start', 'complete']


def test_orphaned_start_is_aborted_and_retry_is_allowed(world):
    first = world.freeze()
    sha = first['freeze']['registry_sha256']
    world.campaign.mkdir(parents=True, exist_ok=True)
    ledger.append(world.campaign, {'event': 'start', 'registry_sha256': sha, 'pid': 2 ** 22 + 12345,
                                   'experiment_index': 1, 'attempt': 1, 'new_disclosed_experiment': False,
                                   'other_registries': [], 'earlier_registries': [], 'disclosure': None, 'orphaned': []})
    (world.campaign / sha[:12]).mkdir()
    campaign(world)
    events = ledger.read_ledger(world.campaign)
    assert [e['event'] for e in events] == ['start', 'abort', 'start', 'complete']
    assert 'orphaned' in events[1]['error'] and events[2]['attempt'] == 2


# ---------------------------------------------------------------- failures, empty masks, geometry

def test_nonfinite_predictions_are_recorded_not_dropped(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch=monkeypatch, seeds={'R': 1000})       # exp(bias 99) overflows float32 to inf
    w.freeze()
    result = campaign(w)
    s = result['summaries']
    assert s['R']['complete'] is False and s['R']['failed_frames'] == 9 and s['R']['ok_frames'] == 0
    assert s['R']['sequence_macro'] is None and s['A']['complete'] and s['Z']['complete']
    out = Path(result['output_dir'])
    frames = read_csv(out / 'methods' / 'R' / 'frames.csv')
    assert len(frames) == 9 and {f['status'] for f in frames} == {'failed'} and frames[0]['failure']
    assert 'INCOMPLETE (9 failed)' in (out / 'comparison.md').read_text()
    assert json.loads((out / 'campaign.json').read_text())['complete'] is False
    assert json.loads((out / 'methods/R/sequences.json').read_text())[0]['failed_frames'] == 3
    assert ledger.read_ledger(w.campaign)[-1]['all_frames_finite'] is False
    assert [r['matched_frames'] for r in read_csv(out / 'comparison_matched_subset.csv')] == ['0', '0', '0']


def test_frame_without_valid_gt_is_excluded_for_every_method_not_aborted(world):
    world.freeze()

    class OneEmpty(FakeTarget):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            if index == 1:
                sample['targets']['depth_valid'] = torch.zeros_like(sample['targets']['depth_valid'])
                sample['targets']['depth_m'] = torch.zeros_like(sample['targets']['depth_m'])
            return sample
    world.dataset_factory = lambda registry, root: OneEmpty(world.rows)
    result = campaign(world)
    out = Path(result['output_dir'])
    for name in ('Z', 'R', 'A'):
        summary = json.loads((out / 'methods' / name / 'summary.json').read_text())
        assert summary['no_valid_gt_frames'] == 1 and summary['failed_frames'] == 0 and summary['complete']
        assert summary['no_valid_gt_sample_ids'] == [world.rows[1]['sample_id']] and summary['ok_frames'] == 8
        frame = read_csv(out / 'methods' / name / 'frames.csv')[1]
        assert frame['status'] == 'no_valid_gt' and frame['abs_rel'] == ''
        first_sequence = summary['sequences'][0]
        assert first_sequence['no_valid_gt_frames'] == 1 and first_sequence['total_frames'] == 3
    matched = read_csv(out / 'comparison_matched_subset.csv')
    assert [r['matched_frames'] for r in matched] == ['8', '8', '8']
    assert 'No valid GT' in (out / 'comparison.md').read_text()


def test_frame_geometry_must_match_registered_resolution(world):
    world.freeze()

    class Wrong(FakeTarget):
        def __getitem__(self, i):
            sample = super().__getitem__(i)
            sample['rgb'] = sample['rgb'][..., :60]
            return sample
    world.dataset_factory = lambda registry, root: Wrong(world.rows)
    with pytest.raises(ValueError, match='registered'):
        campaign(world)
    assert [e['event'] for e in ledger.read_ledger(world.campaign)] == ['start', 'abort']


# ---------------------------------------------------------------- shared evaluator path and report units

def test_evaluate_dataset_matches_training_validation_path_on_source_validation(tmp_path):
    data = FakeSplit(tmp_path, 'source_validation', 'source_native', 4, 56, name='val')
    model = fake_model()
    old = ev.evaluate_source(model, data, 'cpu')
    new = c2.evaluate_dataset(model, data, 'cpu', split='source_validation', geometry='source_native')
    for a, b in zip(old, new):
        assert b['status'] == 'ok' and a['sample_id'] == b['sample_id']
        for key in ev.METRICS + ('valid_pixels',):
            assert a[key] == pytest.approx(b[key], rel=1e-12)


def test_evaluate_dataset_refuses_wrong_split_or_geometry(tmp_path):
    data = FakeSplit(tmp_path, 'source_train', 'source_native', 2, 56)
    with pytest.raises(ValueError):
        c2.evaluate_dataset(fake_model(), data, 'cpu', split='target_test', geometry='target_rectified_full')


def _rows(prefix, n, statuses=None):
    out = []
    for i in range(n):
        if statuses and statuses.get(i) == 'failed':
            out.append(report.failed_row(f'{prefix}{i}', 'A', 'x', 4, 4))
        else:
            out.append(report.ok_row(ev.frame_metrics(torch.full((1, 4), 1.1 + .1 * i * i), torch.ones(1, 4),
                                                      torch.ones(1, 4, dtype=torch.bool)), f'{prefix}{i}', 'A'))
    return out


def test_comparison_requires_equal_counts_and_registered_methods():
    r2, r1 = _rows('a', 2), _rows('a', 1)
    s1, s2 = report.method_summary(r2, {'A': 2}, {}), report.method_summary(r1, {'A': 1}, {})
    with pytest.raises(ValueError):
        report.method_summary(r2, {'A': 3}, {})
    with pytest.raises(ValueError, match='differ'):
        report.comparison({'Z': s1, 'R': s2}, ['Z', 'R'], {'A': 2}, {'Z': r2, 'R': r1})
    with pytest.raises(ValueError, match='registry'):
        report.comparison({'Z': s1}, ['Z', 'R'], {'A': 2}, {'Z': r2})
    tables = report.comparison({'Z': s1, 'R': s1}, ['Z', 'R'], {'A': 2}, {'Z': r2, 'R': r2})
    assert tables['not_registered'] == ['A', 'G', 'M']


def test_matched_subset_uses_only_rows_ok_for_every_method():
    z, r = _rows('a', 3), _rows('a', 3, {1: 'failed'})
    summaries = {'Z': report.method_summary(z, {'A': 3}, {}), 'R': report.method_summary(r, {'A': 3}, {})}
    tables = report.comparison(summaries, ['Z', 'R'], {'A': 3}, {'Z': z, 'R': r})
    matched = {row['method']: row for row in tables['matched_subset_sequence_macro_raw']}
    assert matched['Z']['matched_frames'] == matched['R']['matched_frames'] == 2
    assert matched['Z']['abs_rel'] == pytest.approx(matched['R']['abs_rel'])      # same rows, same data
    assert summaries['Z']['sequence_macro']['abs_rel'] != pytest.approx(matched['Z']['abs_rel'])
    assert tables['headline_sequence_macro_raw'][1]['complete'] is False


def test_cli_refuses_unfrozen_registry_and_reports_reason(world, monkeypatch):
    world.build()
    monkeypatch.chdir(world.root)
    with pytest.raises(SystemExit, match='REFUSED: Registry is not frozen'):
        c2.main(['run', '--registry', str(world.registry_path), '--dry-run'])
    with pytest.raises(SystemExit, match='No readable'):
        c2.main(['run', '--registry', str(world.root / 'missing.json')])
    assert not world.ledger.exists()
