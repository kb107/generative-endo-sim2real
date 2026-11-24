"""Registry generalisation: arbitrary method names, last-step heads, seeds, bbLoRA arms, declared code changes."""
import json

import pytest
import torch

from da3_fakes import fake_bb_model
from evaluation_fakes import World, make_run, put, sha
from generative_sim2real.depth.checkpoints import head_digest
from generative_sim2real.evaluation import registry as reg


def verify(w, **kw):
    return reg.verify_registry(w.registry_path, w.root, code_root=w.root, **kw)


def edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def arms(tmp_path, monkeypatch):
    """R (legacy name) plus new-style arms: another seed, 'last' heads, a bbLoRA run, a third generated seed."""
    w = World(tmp_path, methods=('R', 'G'), monkeypatch=monkeypatch, model_fn=fake_bb_model)
    ids = w.ids
    make_run(tmp_path, 'R_s1', 5, mode='R', run_name='da3_R_seed1', train_seed=1, ids=ids, source_val_count=2)
    make_run(tmp_path, 'G_bb', 6, mode='G', run_name='da3_G_bblora_seed0', lora=True, ids=ids, source_val_count=2)
    make_run(tmp_path, 'G_cl.s2', 7, mode='G', run_name='da3_G_seed2', train_seed=2, ids=ids, source_val_count=2)
    w.spec['methods'] = {
        'Z': None, 'R': 'outputs/runs/da3_R_seed0', 'R_s1': 'outputs/runs/da3_R_seed1',
        'G': 'outputs/runs/da3_G_seed0', 'G_last': {'run_dir': 'outputs/runs/da3_G_seed0', 'head': 'last'},
        'G_bb': {'run_dir': 'outputs/runs/da3_G_bblora_seed0'},
        'G_bb_last': {'run_dir': 'outputs/runs/da3_G_bblora_seed0', 'head': 'last'},
        'G_cl.s2': {'run_dir': 'outputs/runs/da3_G_seed2', 'head': 'best'}}
    return w


def test_arbitrary_methods_seeds_last_heads_and_bblora_register_in_spec_order(tmp_path, monkeypatch):
    w = arms(tmp_path, monkeypatch)
    registry = w.freeze()
    assert registry['method_names'] == list(w.spec['methods']) and registry['declared_absent_methods'] == []
    by = {m['name']: m for m in registry['methods']}
    assert {n: (by[n]['training_seed'], by[n]['trainable']) for n in by if n != 'Z'} == {
        'R': (0, 'head'), 'R_s1': (1, 'head'), 'G': (0, 'head'), 'G_last': (0, 'head'), 'G_bb': (0, 'head+backbone_lora'),
        'G_bb_last': (0, 'head+backbone_lora'), 'G_cl.s2': (2, 'head')}
    assert 'training_seed' not in by['Z'] and 'trainable' not in by['Z']
    assert by['R_s1']['mode'] == 'R' and by['G_cl.s2']['mode'] == 'G' and 'head_choice' not in by['G']
    assert 'backbone_lora' not in by['G'] and 'training_code_changes' not in by['G']
    # best vs last of the same run: different files and digests; last is the final step, flagged as no selection.
    assert by['G_last']['head']['path'].endswith('last.pt') and by['G']['head']['path'].endswith('best.pt')
    assert by['G_last']['head']['head_sha256'] != by['G']['head']['head_sha256']
    assert by['G_last']['selected_step'] == 4 and by['G_last']['head_choice'] == 'last'
    assert by['G_last']['selection'] == reg.FINAL_STEP_SELECTION and by['G']['selection'] == reg.SELECTION
    assert by['G_last']['source_validation_sequence_macro_abs_rel'] == .3        # the final-step validation report
    assert by['G']['selected_step'] == 2
    assert by['G_bb']['backbone_lora']['config'] == {'rank': 8, 'alpha': 8, 'last_k_blocks': 4,
                                                     'targets': ['attn.qkv', 'attn.proj']}
    assert by['G_bb']['backbone_lora']['sha256']
    assert by['G_bb_last']['backbone_lora']['sha256'] != by['G_bb']['backbone_lora']['sha256']
    assert by['G_bb']['resolved_config_sha256'] != by['G']['resolved_config_sha256']
    assert set(registry['data']['generated']) == {'G', 'G_last', 'G_bb', 'G_bb_last', 'G_cl.s2'}
    verify(w)


def test_verify_recovers_spec_order_from_a_key_sorted_registry_file(tmp_path, monkeypatch):
    w = arms(tmp_path, monkeypatch)
    w.freeze()
    stored = json.loads(w.registry_path.read_text())
    assert list(stored['spec']['methods']) != stored['method_names']          # keys are sorted on disk
    verify(w)
    stored['method_names'] = stored['method_names'][::-1]                     # reordered after freezing
    w.registry_path.write_text(json.dumps(stored))
    with pytest.raises(reg.RegistryError):
        verify(w)


def rewrite(path, change):
    value = torch.load(path, weights_only=False)
    change(value)
    torch.save(value, path)


G_RUN, BB_RUN = 'outputs/runs/da3_G_seed0', 'outputs/runs/da3_G_bblora_seed0'


@pytest.mark.parametrize('label,edit,message', [
    ('last_state_differs_from_status', lambda w: edit_json(w.root / G_RUN / 'status.json',
                                                            lambda v: v['state'].update(cursor=7)),
     'last.pt training state differs'),
    ('last_wrong_step', lambda w: rewrite(w.root / G_RUN / 'last.pt', lambda c: c['state'].update(optimizer_step=3)),
     'not the final step'),
    ('last_other_config', lambda w: rewrite(w.root / G_RUN / 'last.pt',
                                            lambda c: c['compatibility']['config']['training'].update(seed=2)),
     'compatibility record differs'),
    ('lora_state_missing', lambda w: rewrite(w.root / BB_RUN / 'best.pt',
                                             lambda c: c.pop('backbone_lora_state_dict')), 'LoRA'),
    ('lora_digest_wrong', lambda w: rewrite(w.root / BB_RUN / 'best.pt',
                                            lambda c: c.update(backbone_lora_sha256='0' * 64)), 'digest|LoRA'),
    ('lora_in_head_only_run', lambda w: rewrite(w.root / G_RUN / 'best.pt', lambda c: c.update(
        kind='da3_head_bblora', backbone_lora_state_dict={}, backbone_lora_sha256=head_digest({}),
        backbone_lora_config={})), 'LoRA'),
])
def test_last_and_lora_heads_are_audited(tmp_path, monkeypatch, label, edit, message):
    w = arms(tmp_path, monkeypatch)
    edit(w)
    with pytest.raises((reg.RegistryError, ValueError), match=message):
        w.build()


def test_last_head_requires_a_completed_run(tmp_path, monkeypatch):
    w = arms(tmp_path, monkeypatch)
    edit_json(w.root / G_RUN / 'status.json', lambda v: v.update(completed_steps=3))
    with pytest.raises(reg.RegistryError, match='scheduled updates'):
        w.build()


def test_legacy_mode_names_still_pin_the_run_mode_but_new_names_may_reuse_modes(tmp_path, monkeypatch):
    w = arms(tmp_path, monkeypatch)
    w.spec['methods']['A'] = 'outputs/runs/da3_R_seed1'            # method 'A' on an R-mode run is refused
    with pytest.raises(reg.RegistryError, match="mode 'R' does not match method name"):
        w.build()
    del w.spec['methods']['A']
    w.spec['methods']['arm_A'] = 'outputs/runs/da3_R_seed1'
    assert 'arm_A' in w.build()['method_names']
    make_run(tmp_path, 'weird', 3, mode='Z', run_name='da3_weird', ids=w.ids, source_val_count=2)
    w.spec['methods']['weird'] = 'outputs/runs/da3_weird'
    with pytest.raises(reg.RegistryError, match='not one of'):
        w.build()


def test_seeds_and_model_variant_may_differ_but_nothing_else(tmp_path, monkeypatch):
    w = arms(tmp_path, monkeypatch)
    w.build()
    make_run(tmp_path, 'G_bad', 9, mode='G', run_name='da3_G_bad', train_seed=1, ids=w.ids, source_val_count=2,
             pre=lambda r: r['config']['training'].update(learning_rate=2e-4))
    w.spec['methods']['G_bad'] = 'outputs/runs/da3_G_bad'
    with pytest.raises(reg.RegistryError, match='training configuration differs'):
        w.build()
    del w.spec['methods']['G_bad']
    make_run(tmp_path, 'G_bad2', 9, mode='G', run_name='da3_G_bad2', lora=True, ids=w.ids, source_val_count=2,
             pre=lambda r: r['config']['model']['backbone_lora'].update(rank=4))
    w.spec['methods']['G_bad2'] = 'outputs/runs/da3_G_bad2'
    with pytest.raises(reg.RegistryError, match='different base model protocol'):
        w.build()


def test_training_code_changes_must_be_declared_and_are_recorded(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch=monkeypatch)
    b = tmp_path / 'src/generative_sim2real/depth/b.py'
    put(b, 'B = 2\n')                                       # a hashed file changed after both runs trained
    with pytest.raises(reg.RegistryError, match='changed since training'):
        w.build()
    w.spec['code_changes_since_training'] = ['depth/b.py']
    registry = w.freeze()
    for record in registry['methods'][1:]:
        change = record['training_code_changes']['depth/b.py']
        assert change['at_training'] != change['now'] and change['now'] == sha(b)
    verify(w)
    put(tmp_path / 'src/generative_sim2real/data/a.py', 'A = 2\n')      # an undeclared change is still refused
    with pytest.raises(reg.RegistryError):
        verify(w)


def test_legacy_specs_keep_their_registry_shape(tmp_path, monkeypatch):
    """Old Z/R/A/G/M specs only gain the explicit training_seed/trainable fields (so their hashes change) and keep
    the original declared-absent listing."""
    registry = World(tmp_path, monkeypatch=monkeypatch).freeze()
    for record in registry['methods'][1:]:
        assert set(record) == {'name', 'kind', 'run_dir', 'mode', 'training_seed', 'trainable', 'head', 'selected_step',
                               'source_validation_sequence_macro_abs_rel', 'selection', 'target_views',
                               'resolved_config_sha256', 'run_files_sha256', 'training_files_sha256',
                               'training_code_sha256_digest', 'training_environment', 'generated_manifest'}
    assert registry['method_names'] == ['Z', 'R', 'A'] and registry['declared_absent_methods'] == ['G', 'M']
    assert 'code_changes_since_training' not in registry['spec']


def test_all_legacy_names_must_share_one_seed_and_variant(tmp_path, monkeypatch):
    w = World(tmp_path, methods=('R', 'G'), monkeypatch=monkeypatch, model_fn=fake_bb_model)
    make_run(tmp_path, 'A', 5, mode='A', run_name='da3_A_seed0', train_seed=1, ids=w.ids, source_val_count=2)
    w.spec['methods']['A'] = 'outputs/runs/da3_A_seed0'
    with pytest.raises(reg.RegistryError, match='one training seed and one model variant'):
        w.build()
    make_run(tmp_path, 'A', 5, mode='A', run_name='da3_A_seed0', lora=True, ids=w.ids, source_val_count=2)
    with pytest.raises(reg.RegistryError, match='one training seed and one model variant'):
        w.build()
    w.spec['methods']['A_seed1'] = w.spec['methods'].pop('A')              # a descriptive name makes the mix explicit
    assert w.build()['methods'][-1]['trainable'] == 'head+backbone_lora'


def test_declared_code_changes_must_exist_and_be_in_the_hashed_scope(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch=monkeypatch)
    for bad in (['depth/nope.py'], ['depth/sub/x.py'], ['scripts/train_da3.sh'], ['depth/b.pyc'], ['b.py']):
        w.spec['code_changes_since_training'] = bad
        with pytest.raises(reg.RegistryError):
            w.build()
    w.spec['code_changes_since_training'] = ['depth/b.py', '__init__.py']
    assert w.build()['spec']['code_changes_since_training'] == ['depth/b.py', '__init__.py']
