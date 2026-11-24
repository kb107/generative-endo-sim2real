import copy
import json

import pytest
import torch

from evaluation_fakes import World, make_run, put, sha
from generative_sim2real.evaluation import registry as reg


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch=monkeypatch)


def verify(w, **kw):
    return reg.verify_registry(w.registry_path, w.root, code_root=w.root, **kw)


def test_draft_is_not_frozen_and_freeze_records_a_content_hash(world):
    draft = world.build()
    assert draft['freeze']['frozen'] is False and draft['method_names'] == ['Z', 'R', 'A']
    assert draft['declared_absent_methods'] == ['G', 'M']
    with pytest.raises(reg.RegistryError, match='not frozen'):
        verify(world)
    verify(world, require_frozen=False)
    frozen = world.freeze()
    assert frozen['freeze']['frozen'] is True and frozen['freeze']['registry_sha256'] == reg.registry_hash(frozen)
    assert verify(world)['freeze']['registry_sha256'] == frozen['freeze']['registry_sha256']
    with pytest.raises(reg.RegistryError, match='already frozen'):
        reg.freeze_registry(world.registry_path, world.root, code_root=world.root)
    with pytest.raises(reg.RegistryError, match='frozen; refusing'):
        world.build()


def test_registry_contents_cover_the_required_identities(world):
    registry = world.freeze()
    r = next(m for m in registry['methods'] if m['name'] == 'R')
    assert r['head']['head_sha256'] and r['head']['file_sha256'] and r['selected_step'] == 2
    assert r['target_views'] == [] and r['selection'] == reg.SELECTION and r['resolved_config_sha256']
    assert registry['base']['official_head_sha256'] and registry['base']['backbone_sha256']
    assert registry['base']['checkpoint_revision'] and registry['base']['da3_revision']
    data = registry['data']
    for part in ('source_train', 'source_validation'):
        assert data[part]['manifest_sha256'] and data[part]['sidecar_sha256']
    assert data['downstream_ids']['sha256']
    target = data['target_test']
    assert target['sidecar_sha256'] and target['pose_file_count'] == 3 and len(target['governing_files_sha256']) == 3
    assert target['count'] == 9 and sum(target['per_sequence'].values()) == 9 and target['ids_sha256']
    assert registry['code']['sha256'] and registry['evaluator_version']
    protocol = registry['protocol']
    assert protocol['prediction']['units'] == 'meters' and 'none' in protocol['alignment']['headline']
    assert 'sequence_macro' in protocol['aggregation']['headline'] and 'ORACLE' in protocol['metrics']['oracle_median_rmse_mm']


def _edit_json(path, change):
    import json as j
    value = j.loads(path.read_text())
    change(value)
    path.write_text(j.dumps(value))


MUTATIONS = {
    'head_file': lambda w: (w.root / 'outputs/runs/da3_R_seed0/best.pt').write_bytes(
        (w.root / 'outputs/runs/da3_R_seed0/best.pt').read_bytes() + b'x'),
    'validation_report': lambda w: _edit_json(w.root / 'outputs/runs/da3_R_seed0/validation/step_000004/summary.json',
                                              lambda v: v['sequence_macro'].update(abs_rel=.299)),
    'train_log': lambda w: put(w.root / 'outputs/runs/da3_R_seed0/train.jsonl', ''),
    'downstream_ids': lambda w: _edit_json(w.root / 'outputs/manifests/downstream_ids.json', lambda v: v['sample_ids'].reverse()),
    'source_train_manifest': lambda w: (w.root / 'outputs/manifests/source_train.jsonl').write_text('{}\n'),
    'target_manifest': lambda w: (w.root / 'outputs/manifests/target_test.jsonl').write_text('{}\n'),
    'target_sidecar_pose_hash': lambda w: _edit_json(w.root / 'outputs/manifests/target_test.meta.json',
                                                     lambda v: v['governing_files_sha256'].update(
                                                         {'C3VDv2/frames/C2/c2_a_v1/pose.txt': '0' * 64})),
    'code': lambda w: put(w.root / 'src/generative_sim2real/evaluation/c.py', 'C = 2\n'),
    'training_code': lambda w: put(w.root / 'src/generative_sim2real/data/a.py', 'A = 2\n'),
}


@pytest.mark.parametrize('name', sorted(MUTATIONS))
def test_every_input_change_after_freezing_is_refused(world, name):
    world.freeze()
    verify(world)
    MUTATIONS[name](world)
    with pytest.raises(reg.RegistryError):
        verify(world)


def test_edited_or_mismatched_registry_file_is_refused(world):
    world.freeze()
    _edit_json(world.registry_path, lambda v: v['protocol']['prediction'].update(units='millimeters'))
    with pytest.raises(reg.RegistryError, match='hash'):
        verify(world)
    world.registry_path.unlink()
    frozen = world.freeze()
    with pytest.raises(reg.RegistryError, match='expected hash'):
        verify(world, expected_sha256='0' * 64)
    verify(world, expected_sha256=frozen['freeze']['registry_sha256'])


@pytest.mark.parametrize('label,mutate,message', [
    ('target_views', lambda r, s: r['config']['evaluation'].update(target_views=['target_test']), 'target_views'),
    ('selection', lambda r, s: r['config']['evaluation'].update(selection='target_abs_rel'), 'selection'),
    ('smoke', lambda r, s: (r.update(smoke_only=True), s.update(smoke_only=True)), 'non-smoke'),
    ('incomplete', lambda r, s: s.update(completed_steps=3), 'scheduled updates'),
    ('best_step', lambda r, s: (s['state'].update(best_step=4), s.update(selected_step=4)), 'selected by source validation'),
    ('target_data_key', lambda r, s: r['config']['data'].update(target_train='x'), 'dataset fields'),
    ('wrong_mode', lambda r, s: r['config']['data'].update(mode='A'), 'mode'),
])
def test_a_head_not_selected_by_source_validation_only_is_refused(world, label, mutate, message):
    run = world.root / 'outputs/runs/da3_R_seed0'
    resolved, status = json.loads((run / 'resolved.json').read_text()), json.loads((run / 'status.json').read_text())
    mutate(resolved, status)
    (run / 'resolved.json').write_text(json.dumps(resolved))
    (run / 'status.json').write_text(json.dumps(status))
    with pytest.raises(reg.RegistryError, match=message):
        world.build()


def test_training_log_with_target_ids_is_refused(world):
    put(world.root / 'outputs/runs/da3_R_seed0/train.jsonl',
        ''.join(json.dumps({'step': s, 'sample_ids': [world.rows[0]['sample_id']]}) + '\n' for s in range(4)))
    with pytest.raises(reg.RegistryError, match='non-source'):
        world.build()


def test_partial_validation_reports_and_unmatched_argmin_are_refused(world):
    _edit_json(world.root / 'outputs/runs/da3_R_seed0/validation/step_000002/summary.json', lambda v: v.update(frame_count=1))
    with pytest.raises(reg.RegistryError, match='whole source-validation'):
        world.build()


def test_methods_must_share_the_training_protocol(world):
    ids = world.ids
    make_run(world.root, 'A', 2, ids=ids, source_val_count=2, pre=lambda r: r['config']['training'].update(learning_rate=1e-4))
    with pytest.raises(reg.RegistryError, match='training configuration differs'):
        world.build()


def test_consistently_remade_run_changes_the_registry_hash_fields(world):
    world.freeze()
    # An allowed difference (microbatch size), recorded consistently in resolved.json and best.pt: only the pins change.
    make_run(world.root, 'A', 2, ids=world.ids, source_val_count=2,
             pre=lambda r: r['config']['training'].update(microbatch_size=1))
    with pytest.raises(reg.RegistryError, match='resolved_config_sha256'):
        verify(world)


def test_run_must_start_from_the_official_head(world):
    make_run(world.root, 'A', 2, ids=world.ids, source_val_count=2, official_head='0' * 64)
    with pytest.raises(reg.RegistryError, match='official pretrained head'):
        world.build()


def test_validation_report_steps_must_match_the_schedule(world):
    put(world.root / 'outputs/runs/da3_R_seed0/validation/step_000003/summary.json',
        (world.root / 'outputs/runs/da3_R_seed0/validation/step_000002/summary.json').read_text())
    with pytest.raises(reg.RegistryError, match='validation steps'):
        world.build()


def test_pose_file_changed_on_disk_is_refused_at_build_and_verify(world):
    world.freeze()
    pose = world.data_root / 'C3VDv2/frames/C2/c2_a_v1/pose.txt'
    pose.write_text('changed\n')
    with pytest.raises(reg.RegistryError, match='pose'):
        verify(world)
    with pytest.raises(reg.RegistryError, match='pose'):
        reg.describe_target('outputs/manifests/target_test.jsonl', world.root)


def test_data_root_is_required_to_rehash_pose_files(world, monkeypatch):
    monkeypatch.delenv('DATA_ROOT')
    with pytest.raises(reg.RegistryError, match='DATA_ROOT'):
        world.build()


def test_official_digests_are_recomputed_by_verify(world):
    world.freeze()
    world.digests['backbone_sha256'] = '9' * 64            # what the weights file "now" hashes to
    with pytest.raises(reg.RegistryError, match='recomputed'):
        verify(world)


def test_runtime_versions_are_recorded_and_compared(world):
    registry = world.freeze()
    assert set(registry['runtime']) == {'python', 'torch', 'numpy', 'xformers', 'torch_cuda'}
    assert reg.check_runtime(registry) == []
    other = {**registry['runtime'], 'torch': '0.0.1', 'torch_cuda': '0.0'}
    mismatches = reg.check_runtime(registry, other)
    assert len(mismatches) == 2 and 'torch' in mismatches[0]


def test_diff_names_the_changed_field_inside_lists():
    assert reg.diff({'m': [{'a': 1}, {'a': 2}]}, {'m': [{'a': 1}, {'a': 3}]}) == ['/m/1/a: registry=2 current=3']


def _spec(methods, **extra):
    return {'schema_version': 1, 'name': 'n', 'output': 'o', 'methods': methods,
            'data': {k: 'p' for k in reg.SPEC_DATA_KEYS}, **extra}


def test_spec_requires_official_method_and_valid_names():
    bad = ({'R': 'x'}, {'Z': None}, {'Z': None, 'Q': None}, {'Z': 'x', 'R': 'y'}, {'Z': None, '1x': 'x'},
           {'Z': None, 'a b': 'x'}, {'Z': None, 'x' * 41: 'x'}, {'Z': None, 'R': 'x', 'r': 'y'}, {'Z': None, 'R': ''},
           {'Z': None, 'R': {'head': 'best'}}, {'Z': None, 'R': {'run_dir': 'x', 'head': 'median'}},
           {'Z': None, 'R': {'run_dir': 'x', 'extra': 1}}, {'Z': None, 'R': 3})
    for methods in bad:
        with pytest.raises(reg.RegistryError):
            reg.check_spec(_spec(methods))
    good = {'Z': None, 'Q': 'x', 'G_cl+bb.s1-a': {'run_dir': 'y'}, 'A9': {'run_dir': 'z', 'head': 'last'}, 'x' * 40: 'w'}
    assert reg.check_spec(_spec(good))['methods'] == good
    assert [reg.parse_method(n, v) for n, v in good.items()] == [
        (None, None), ('x', 'best'), ('y', 'best'), ('z', 'last'), ('w', 'best')]
    with pytest.raises(reg.RegistryError):
        reg.check_spec(_spec({'Z': None, 'R': 'x'}, unknown=1))
    with pytest.raises(reg.RegistryError):
        reg.check_spec(_spec({'Z': None, 'R': 'x'}, code_changes_since_training=['../x.py']))
    reg.check_spec(_spec({'Z': None, 'R': 'x'}, code_changes_since_training=['depth/train.py']))


def test_freeze_requires_model_digests(world):
    world.digests = None
    world.build()
    with pytest.raises(reg.RegistryError, match='digests'):
        reg.freeze_registry(world.registry_path, world.root, code_root=world.root)


def test_generated_methods_register_their_manifest_and_must_cover_every_id(tmp_path, monkeypatch):
    w = World(tmp_path, methods=('R', 'G'), monkeypatch=monkeypatch)
    registry = w.freeze()
    g = next(m for m in registry['methods'] if m['name'] == 'G')
    assert registry['data']['generated']['G']['sha256'] == sha(tmp_path / 'outputs/manifests/generated_G.jsonl')
    assert g['generated_manifest'] == 'outputs/manifests/generated_G.jsonl'
    provenance = registry['data']['generated']['G']['provenance']
    assert provenance['base_model'] == 'base_model-value' and provenance['seed'] == {'varies_per_record': True, 'distinct': 4}
    put(tmp_path / 'outputs/manifests/generated_G.jsonl', json.dumps({'sample_id': w.ids[0]}) + '\n')
    with pytest.raises(reg.RegistryError):
        verify(w)


def test_draft_can_be_rebuilt_with_a_larger_method_set_before_freezing(tmp_path, monkeypatch):
    w = World(tmp_path, methods=('R', 'A', 'G'), monkeypatch=monkeypatch)
    spec = copy.deepcopy(w.spec)
    del w.spec['methods']['G']
    assert w.build()['method_names'] == ['Z', 'R', 'A']
    w.spec = spec
    assert w.freeze()['method_names'] == ['Z', 'R', 'A', 'G']


def test_official_digests_identify_head_and_backbone():
    from da3_fakes import fake_model
    base = reg.official_digests(fake_model())
    other = fake_model()
    with torch.no_grad():
        other.network.backbone.weight.add_(1.)
    assert reg.official_digests(other)['backbone_sha256'] != base['backbone_sha256']
    assert reg.official_digests(other)['official_head_sha256'] == base['official_head_sha256']
