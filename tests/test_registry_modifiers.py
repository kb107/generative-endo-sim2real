"""Diagnostic per-entry modifiers (campaign #6A): process_hw and output_scale on an already-registered head."""
import copy
import csv
import json
from pathlib import Path

import pytest
import torch

from da3_fakes import FOCAL, fake_model
from evaluation_fakes import FakeTarget, HW, World, put
from generative_sim2real.depth.model import DA3MetricDepth
from generative_sim2real.evaluation import c2, depth as ev, ledger, report
from generative_sim2real.evaluation import registry as reg

SCALE_SOURCE = 'outputs/analysis/scale_anatomy/predictions.json'
PROCESS_HW = [28, 42]                      # multiples of 14, different aspect handling than HW = (56, 70)
RUN = 'outputs/runs/da3_R_seed0'


def scale_entry(value=1.5, source=SCALE_SOURCE):
    return {'value': value, 'source': source}


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch=monkeypatch)
    put(tmp_path / SCALE_SOURCE, '{"frozen": "scale anatomy"}\n')
    w.messages = []
    yield w
    torch.use_deterministic_algorithms(False)


def with_arms(w):
    """R, A plus diagnostic arms on R's registered head (the same head appears in several entries)."""
    w.spec['methods'].update({
        'R.406': {'run_dir': RUN, 'head': 'best', 'process_hw': PROCESS_HW},
        'R.cG': {'run_dir': RUN, 'head': 'best', 'output_scale': scale_entry()},
        'R.cG.406': {'run_dir': RUN, 'head': 'best', 'process_hw': PROCESS_HW, 'output_scale': scale_entry()}})
    return w


def campaign(w, dry_run=False, **kw):
    return c2.run_campaign(w.registry_path, root=w.root, campaign_dir=w.campaign, dry_run=dry_run, code_root=w.root,
                           dataset_factory=w.dataset_factory, model_factory=w.model_factory, log=w.messages.append, **kw)


def read_csv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def verify(w, **kw):
    return reg.verify_registry(w.registry_path, w.root, code_root=w.root, **kw)


def record(registry, name):
    return next(r for r in registry['methods'] if r['name'] == name)


# ---------------------------------------------------------------- (1) entries without modifiers are unchanged

PLAIN_KEYS = {'name', 'kind', 'run_dir', 'mode', 'training_seed', 'trainable', 'head', 'selected_step',
              'source_validation_sequence_macro_abs_rel', 'selection', 'target_views', 'resolved_config_sha256',
              'run_files_sha256', 'training_files_sha256', 'training_code_sha256_digest', 'training_environment',
              'generated_manifest'}


def test_plain_entries_get_no_new_keys_and_modifier_entries_do_not_change_them(world):
    plain = world.build()
    assert {r['name'] for r in plain['methods']} == {'Z', 'R', 'A'}
    for r in plain['methods'][1:]:
        assert set(r) == PLAIN_KEYS
    assert 'diagnostic_modifiers' not in plain['protocol'] and 'inference_process_hw' not in json.dumps(plain)
    # An explicit dict form with the default head is the same record as the string form.
    world.spec['methods']['R2'] = {'run_dir': RUN, 'head': 'best'}
    again = world.build()
    assert {k: v for k, v in record(again, 'R2').items() if k != 'name'} == {k: v for k, v in record(again, 'R').items()
                                                                              if k != 'name'}
    with_arms(world)
    modified = world.build()
    for name in ('Z', 'R', 'A', 'R2'):                       # plain entries keep exactly their content
        assert record(modified, name) == record(again, name)
    assert modified['protocol']['diagnostic_modifiers']['scope'].startswith('DIAGNOSTIC')


def test_plain_campaign_outputs_have_no_modifier_columns_or_fields(world):
    world.freeze()
    result = campaign(world)
    out = Path(result['output_dir'])
    assert list(read_csv(out / 'comparison_provenance.csv')[0]) == ['method', 'training_seed', 'variant', 'head', 'selected_step']
    assert 'Process HW' not in (out / 'comparison.md').read_text() and 'DIAGNOSTIC arms' not in (out / 'comparison.md').read_text()
    assert 'method_modifiers' not in json.loads((out / 'campaign.json').read_text())
    meta = json.loads((out / 'methods/R/summary.json').read_text())['metadata']
    assert meta['process_hw'] == list(HW) and 'inference_process_hw' not in meta and 'output_scale' not in meta
    assert 'method_modifiers' not in campaign(world, dry_run=True)


# ---------------------------------------------------------------- spec validation

@pytest.mark.parametrize('bad', [[30, 42], [28], [28, 42, 14], [0, 14], [-14, 28], [28.0, 42], [True, 14], '28x42', 28, None,
                                 [[28], [42]]])
def test_process_hw_must_be_a_pair_of_positive_multiples_of_14(bad):
    spec = {'schema_version': 1, 'name': 'n', 'output': 'o', 'data': {k: 'p' for k in reg.SPEC_DATA_KEYS},
            'methods': {'Z': None, 'R.406': {'run_dir': 'x', 'process_hw': bad}}}
    with pytest.raises(reg.RegistryError, match='process_hw'):
        reg.check_spec(spec)


@pytest.mark.parametrize('bad', [{'value': 0, 'source': 'a'}, {'value': -1.0, 'source': 'a'}, {'value': float('nan'), 'source': 'a'},
                                 {'value': float('inf'), 'source': 'a'}, {'value': True, 'source': 'a'},
                                 {'value': '1.5', 'source': 'a'}, {'value': 1.5}, {'source': 'a'}, 1.5, None,
                                 {'value': 1.5, 'source': ''}, {'value': 1.5, 'source': '/abs/path.json'},
                                 {'value': 1.5, 'source': '../x.json'}, {'value': 1.5, 'source': 'a', 'extra': 1}])
def test_output_scale_must_be_a_pinned_positive_finite_value(bad):
    spec = {'schema_version': 1, 'name': 'n', 'output': 'o', 'data': {k: 'p' for k in reg.SPEC_DATA_KEYS},
            'methods': {'Z': None, 'R.cG': {'run_dir': 'x', 'output_scale': bad}}}
    with pytest.raises(reg.RegistryError, match='output_scale'):
        reg.check_spec(spec)


def test_modifiers_need_a_trained_head_and_an_existing_source_file(world):
    world.spec['methods']['Z'] = {'run_dir': 'x', 'process_hw': PROCESS_HW}
    with pytest.raises(reg.RegistryError, match='Z'):
        reg.check_spec(world.spec)
    world.spec['methods']['Z'] = None
    world.spec['methods']['R.cG'] = {'run_dir': RUN, 'output_scale': scale_entry(source='outputs/analysis/missing.json')}
    with pytest.raises(reg.RegistryError, match='not a file'):
        world.build()


def test_modifier_records_state_the_modifiers_explicitly(world):
    registry = with_arms(world).freeze()
    r406, rcg, both = (record(registry, n) for n in ('R.406', 'R.cG', 'R.cG.406'))
    assert r406['inference_process_hw'] == PROCESS_HW and 'output_scale' not in r406
    assert rcg['output_scale'] == {'value': 1.5, 'source': SCALE_SOURCE, 'sha256': reg.sha256_file(world.root / SCALE_SOURCE)}
    assert 'inference_process_hw' not in rcg
    assert both['inference_process_hw'] == PROCESS_HW and both['output_scale'] == rcg['output_scale']
    # Same registered head file/digest in every entry; only the modifier fields differ.
    strip = lambda r: {k: v for k, v in r.items() if k not in ('name', 'inference_process_hw', 'output_scale')}
    assert strip(r406) == strip(rcg) == strip(both) == strip(record(registry, 'R'))
    verify(world)


# ---------------------------------------------------------------- (4) hash coverage

def test_value_source_bytes_and_process_hw_are_covered_by_the_registry_hash(world):
    with_arms(world)
    base = reg.registry_hash(world.build())
    changed = []
    for edit in (lambda m: m['R.cG'].update(output_scale=scale_entry(1.6)),
                 lambda m: m['R.406'].update(process_hw=[42, 56]),
                 lambda m: m['R.cG'].update(output_scale=scale_entry(source='outputs/analysis/other.json'))):
        put(world.root / 'outputs/analysis/other.json', '{"frozen": "scale anatomy"}\n')      # same bytes, other path
        spec = copy.deepcopy(world.spec)
        edit(world.spec['methods'])
        changed.append(reg.registry_hash(world.build()))
        world.spec = spec
    put(world.root / SCALE_SOURCE, '{"frozen": "scale anatomy", "edited": true}\n')            # same value, other bytes
    changed.append(reg.registry_hash(world.build()))
    assert base not in changed and len(set(changed)) == len(changed)


def test_verification_refuses_a_modified_source_file_and_an_edited_modifier(world):
    with_arms(world)
    world.freeze()
    verify(world)
    stored = json.loads(world.registry_path.read_text())
    stored['methods'] = [dict(r, output_scale={**r['output_scale'], 'value': 9.0}) if r['name'] == 'R.cG' else r
                         for r in stored['methods']]
    edited = world.registry_path.with_name('edited.json')
    edited.write_text(json.dumps(stored))
    with pytest.raises(reg.RegistryError, match='hash does not match'):
        reg.verify_registry(edited, world.root, code_root=world.root)
    put(world.root / SCALE_SOURCE, '{"frozen": "changed after freezing"}\n')
    with pytest.raises(reg.RegistryError, match='output_scale'):
        verify(world)
    with pytest.raises(reg.RegistryError, match='output_scale'):
        campaign(world)
    assert not world.ledger.exists()                       # refused before the ledger records a start


def test_changed_process_hw_in_a_frozen_registry_is_refused(world):
    with_arms(world)
    world.freeze()
    stored = json.loads(world.registry_path.read_text())
    stored['spec']['methods']['R.406']['process_hw'] = [14, 28]
    world.registry_path.write_text(json.dumps(stored))
    with pytest.raises(reg.RegistryError):
        verify(world)


# ---------------------------------------------------------------- (2) process_hw at inference

def test_process_hw_scales_k_per_axis_and_resizes_back_to_the_gt_grid():
    h, w = HW
    ph, pw = PROCESS_HW
    sample = FakeTarget([{'sample_id': 's/0', 'sequence_id': 's'}])[0]
    model = fake_model()
    c2.set_processing(model, PROCESS_HW)
    rgb, K = model.prepare_inputs(sample['rgb'][None], sample['camera']['K'][None])
    assert tuple(rgb.shape[-2:]) == tuple(PROCESS_HW)
    K0 = sample['camera']['K']
    sx, sy = pw / w, ph / h
    assert sx != sy
    expected = torch.tensor([[K0[0, 0] * sx, 0, K0[0, 2] * sx], [0, K0[1, 1] * sy, K0[1, 2] * sy], [0, 0, 1]])
    assert torch.allclose(K[0], expected)
    # Metric factor uses the scaled mean focal: a constant canonical depth makes it observable, on the GT grid.
    from test_evaluation_c2 import StubNetwork
    stub = DA3MetricDepth(StubNetwork(), process_hw=tuple(PROCESS_HW))
    data = FakeTarget([{'sample_id': 'c3vdv2/c2/s/0000', 'sequence_id': 's'}])
    data.split, data.geometry = 'target_test', 'target_rectified_full'
    rows = c2.evaluate_dataset(stub, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=HW)
    factor = (FOCAL * sx + FOCAL * sy) / 2 / 300
    assert rows[0]['status'] == 'ok' and rows[0]['valid_pixels'] == h * (w - w // 5)
    assert rows[0]['median_pred_m'] == pytest.approx(.05 * factor, rel=1e-6)
    assert tuple(stub.network.calls[0][0].shape) == (1, 1, 3, ph, pw)


def test_non_native_processing_needs_the_resizing_path_and_refuses_non_multiples():
    data = FakeTarget([{'sample_id': 'c3vdv2/c2/s/0000', 'sequence_id': 's'}])
    data.split, data.geometry = 'target_test', 'target_rectified_full'
    model = fake_model()
    c2.set_processing(model, PROCESS_HW)
    with pytest.raises(ValueError, match='resizing is not permitted'):
        c2.evaluate_dataset(model, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=HW, exact_grid=True)
    with pytest.raises(ValueError):
        c2.set_processing(model, (30, 42))
    with pytest.raises(ValueError):
        c2.set_processing(model, (28, 43))
    assert c2.method_inference({'name': 'R'}, HW) == (tuple(HW), True, None)
    assert c2.method_inference({'name': 'R', 'inference_process_hw': list(HW)}, HW) == (tuple(HW), True, None)
    assert c2.method_inference({'name': 'R', 'inference_process_hw': PROCESS_HW}, HW) == (tuple(PROCESS_HW), False, None)
    with pytest.raises(reg.RegistryError):
        c2.method_inference({'name': 'R', 'inference_process_hw': [30, 42]}, HW)


# ---------------------------------------------------------------- (3) output_scale

def test_output_scale_multiplies_the_prediction_before_every_metric():
    data = FakeTarget([{'sample_id': f'c3vdv2/c2/s/{i:04d}', 'sequence_id': 's'} for i in range(2)])
    data.split, data.geometry = 'target_test', 'target_rectified_full'
    model = fake_model()
    c2.set_processing(model, HW)
    plain = c2.evaluate_dataset(model, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=HW,
                                exact_grid=True)
    scaled = c2.evaluate_dataset(model, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=HW,
                                 exact_grid=True, output_scale=.75)
    none = c2.evaluate_dataset(model, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=HW,
                               exact_grid=True, output_scale=None)
    assert none == plain
    for i, (a, b) in enumerate(zip(plain, scaled)):
        sample = data[i]
        with torch.no_grad():
            prediction = model(sample['rgb'][None], sample['camera']['K'][None])['depth_m'][0] * .75
        expected = ev.frame_metrics(prediction, sample['targets']['depth_m'], sample['targets']['depth_valid'])
        for key in ev.METRICS + ('median_pred_m', 'oracle_median_scale', 'median_gt_m'):
            assert b[key] == pytest.approx(expected[key], rel=1e-12), key
        assert b['median_pred_m'] == pytest.approx(.75 * a['median_pred_m'], rel=1e-6)
        assert b['oracle_median_scale'] == pytest.approx(a['oracle_median_scale'] / .75, rel=1e-6)
        assert b['oracle_median_rmse_mm'] == pytest.approx(a['oracle_median_rmse_mm'], rel=1e-5)   # scale-invariant
        assert b['abs_rel'] != a['abs_rel'] and b['rmse_mm'] != a['rmse_mm']
    for bad in (0, -1., float('nan'), float('inf')):
        with pytest.raises(ValueError, match='output_scale'):
            c2.evaluate_dataset(model, data, 'cpu', split='target_test', geometry='target_rectified_full', hw=HW,
                                exact_grid=True, output_scale=bad)


# ---------------------------------------------------------------- campaign: rows, outputs and provenance (5)

def test_campaign_runs_modifier_arms_on_the_same_rows_and_shows_the_modifiers(world):
    with_arms(world).freeze()
    seen = []

    def factory():
        model = fake_model()
        seen.append(model)
        return model
    world.model_factory = factory
    plain_dry = campaign(world, dry_run=True)
    assert plain_dry['method_modifiers']['R.cG.406']['inference_process_hw'] == PROCESS_HW
    result = campaign(world)
    out = Path(result['output_dir'])
    names = ['Z', 'R', 'A', 'R.406', 'R.cG', 'R.cG.406']
    frames = {n: read_csv(out / 'methods' / n / 'frames.csv') for n in names}
    ids = [r['sample_id'] for r in world.rows]
    assert all([f['sample_id'] for f in frames[n]] == ids and all(f['status'] == 'ok' for f in frames[n]) for n in names)
    counts = {n: [r['gt_valid_pixels'] for r in json.loads((out / 'methods' / n / 'sequences.json').read_text())] for n in names}
    assert all(counts[n] == counts['R'] for n in names)
    # Native shapes for plain/scale-only arms, the modifier size for process_hw arms (one network pass per frame each).
    shapes = [tuple(x.shape[-2:]) for x, _ in seen[0].network.calls]
    assert shapes.count(tuple(HW)) == 9 * 4 and shapes.count(tuple(PROCESS_HW)) == 9 * 2
    # output_scale arms: same registered head and processing as R, so metrics follow exactly from scaling R's prediction.
    for plain, scaled in zip(frames['R'], frames['R.cG']):
        assert float(scaled['median_pred_m']) == pytest.approx(1.5 * float(plain['median_pred_m']), rel=1e-5)
        assert float(scaled['median_gt_m']) == float(plain['median_gt_m'])
    # process_hw arms differ from the native arm (different input resolution and K scaling).
    assert [f['abs_rel'] for f in frames['R.406']] != [f['abs_rel'] for f in frames['R']]
    summaries = {n: json.loads((out / 'methods' / n / 'summary.json').read_text()) for n in names}
    assert summaries['R']['metadata']['process_hw'] == list(HW) and 'inference_process_hw' not in summaries['R']['metadata']
    assert summaries['R.406']['metadata']['process_hw'] == PROCESS_HW
    assert summaries['R.406']['metadata']['inference_process_hw'] == PROCESS_HW
    assert summaries['R.cG']['metadata']['output_scale']['value'] == 1.5 and summaries['R.cG']['metadata']['process_hw'] == list(HW)
    # Provenance outputs.
    rows = {r['method']: r for r in read_csv(out / 'comparison_provenance.csv')}
    sha = reg.sha256_file(world.root / SCALE_SOURCE)
    assert rows['R']['inference_process_hw'] == 'native' and rows['R']['output_scale'] == ''
    assert rows['R.406']['inference_process_hw'] == '28x42' and rows['R.406']['output_scale'] == ''
    assert rows['R.cG']['output_scale'] == '1.5' and rows['R.cG']['output_scale_source'] == SCALE_SOURCE
    assert rows['R.cG.406']['output_scale_sha256'] == sha and rows['R.cG.406']['inference_process_hw'] == '28x42'
    markdown = (out / 'comparison.md').read_text()
    provenance = markdown.split('## Method provenance')[1].split('## Headline')[0]
    assert 'Process HW' in provenance and f'1.5 ({SCALE_SOURCE}, {sha[:12]})' in provenance and '28x42' in provenance
    assert 'DIAGNOSTIC arms' in provenance
    camp = json.loads((out / 'campaign.json').read_text())
    assert camp['method_modifiers'] == {'R.406': {'inference_process_hw': PROCESS_HW},
                                        'R.cG': {'output_scale': {'value': 1.5, 'source': SCALE_SOURCE, 'sha256': sha}},
                                        'R.cG.406': {'inference_process_hw': PROCESS_HW,
                                                     'output_scale': {'value': 1.5, 'source': SCALE_SOURCE, 'sha256': sha}}}
    assert json.loads((out / 'comparison.json').read_text())['provenance'][-1]['output_scale_sha256'] == sha
    ledger.audit(world.campaign)


def test_failure_policy_applies_to_scaled_arms_unchanged(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch=monkeypatch, seeds={'R': 1000})       # R's head overflows to inf at valid pixels
    put(tmp_path / SCALE_SOURCE, '{"x": 1}\n')
    w.spec['methods']['R.cG'] = {'run_dir': RUN, 'output_scale': scale_entry()}
    w.messages = []
    w.freeze()
    result = campaign(w)
    summaries = result['summaries']
    assert summaries['R']['failed_frames'] == 9 == summaries['R.cG']['failed_frames']
    assert summaries['R.cG']['complete'] is False and summaries['A']['complete']


def test_output_scale_key_binds_the_value_to_the_frozen_file(world):
    put(world.root / SCALE_SOURCE, json.dumps({'diagnostic_arms': {'R.cG': {'output_scale': 0.8125}}}) + '\n')
    key = 'diagnostic_arms/R.cG/output_scale'
    world.spec['methods']['R.cG'] = {'run_dir': RUN, 'output_scale': {'value': 0.8125, 'source': SCALE_SOURCE, 'key': key}}
    registry = world.freeze()
    assert record(registry, 'R.cG')['output_scale']['key'] == key
    def build_elsewhere():                                                      # the frozen file is never overwritten
        return reg.build_registry(world.spec, world.root, world.root / 'outputs/other_registry.json',
                                  digests=world.digests, code_root=world.root)
    world.spec['methods']['R.cG']['output_scale']['value'] = 0.8126           # hand-typed number != frozen file
    with pytest.raises(reg.RegistryError, match='!='):
        build_elsewhere()
    world.spec['methods']['R.cG']['output_scale'].update(value=0.8125, key='diagnostic_arms/R.cG/missing')
    with pytest.raises(reg.RegistryError, match='not found'):
        build_elsewhere()
