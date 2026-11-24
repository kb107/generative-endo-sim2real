import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from da3_fakes import FakeSplit, fake_model, small_config, write_generated, write_ids
from generative_sim2real.data import rgb_modes as rm
from generative_sim2real.depth import train as T
from generative_sim2real.depth.checkpoints import head_digest, read_head

N = 6


@pytest.fixture
def world(tmp_path):
    source = FakeSplit(tmp_path, 'source_train', 'source_generation512', N, 512)
    validation = FakeSplit(tmp_path, 'source_validation', 'source_native', 4, 56)
    ids = [r['sample_id'] for r in source.rows]
    return {'tmp': tmp_path, 'source': source, 'validation': validation, 'ids': ids,
            'ids_path': write_ids(tmp_path, source, ids),
            'generated': write_generated(tmp_path, source, ids, tmp_path / 'out')}


def make(world, mode='R', lora=False, **training):
    config = small_config(**training)
    if lora:
        config['model'].update(trainable='head+backbone_lora', backbone_lora=copy.deepcopy(T.BACKBONE_LORA))
    config['data'].update(mode=mode, matched_ids=str(world['ids_path']), outputs_root=str(world['tmp'] / 'out'),
                          source_train=str(world['source'].path), source_validation=str(world['validation'].path),
                          generated_manifest=str(world['generated']) if mode in 'GM' else None)
    T.validate_config(config)
    return config


def run(world, name, config, *, resume=None, stop=None, evaluate=None, model_fn=fake_model):
    ids = rm.load_matched_ids(config['data']['matched_ids'], world['source'], smoke=True)
    dataset = rm.RGBModeDataset(world['source'], ids, mode=config['data']['mode'], seed=config['training']['seed'],
                                generated_manifest=config['data']['generated_manifest'], outputs_root=config['data']['outputs_root'])
    compat = T.compatibility(config, world['source'], world['validation'])
    out = world['tmp'] / name
    kwargs = {} if evaluate is None else {'evaluate': evaluate}
    state = T.train(model_fn(), dataset, world['validation'], config, out, compat, resume=resume,
                    stop_after_step=stop, **kwargs)
    return out, state


def log(out):
    return [json.loads(line) for line in (out / 'train.jsonl').read_text().splitlines()]


def last(out):
    return torch.load(out / 'last.pt', weights_only=False)


# -- configuration and target exclusion ------------------------------------------------------------------

def test_default_configs_validate_and_production_schedule_is_locked():
    for name in ('da3', 'da3_smoke', 'da3_bblora'):
        T.validate_config(yaml.safe_load(Path(f'configs/{name}.yaml').read_text()))
    prod = yaml.safe_load(Path('configs/da3.yaml').read_text())
    assert prod['purpose'] == 'experiment' and prod['evaluation']['target_views'] == []
    assert prod['data']['matched_ids'].endswith('downstream_ids.json')
    prod['training']['optimizer_steps'] = 500
    with pytest.raises(ValueError):
        T.validate_config(prod)


@pytest.mark.parametrize('edit', [
    lambda c: c['evaluation'].update(target_views=['c2']),
    lambda c: c['data'].update(target_test='x.jsonl'),
    lambda c: c.update(target={'split': 'target_test'}),
    lambda c: c['evaluation'].update(validation_frames=8),
    lambda c: c['data'].update(mode='Z'),
    lambda c: c['data'].update(augmentation={'brightness': [.5, 1.5], 'color': [.9, 1.1], 'gamma': [.9, 1.1]}),
    lambda c: c['model'].update(checkpoint_revision='main'),
    lambda c: c['model'].update(trainable='all'),
    lambda c: c['training'].update(world_size=2),
    lambda c: c['training'].update(microbatch_size=3),
    lambda c: c['training'].update(learning_rate=0),
    lambda c: c.update(purpose='anything'),
    lambda c: c['training'].update(learning_rate=5e-5),
    lambda c: c['training'].update(weight_decay=0.02),
    lambda c: c['training'].update(gradient_clip_norm=2.0),
    lambda c: c['training'].update(seed=3),
    lambda c: c['training'].update(depth_weight=2.0),
    lambda c: c['training'].update(gradient_weight=0.5),
    lambda c: c['training'].update(global_batch_size=32),
    lambda c: c['training'].update(warmup_steps=50),
    lambda c: c['training'].update(validation_every=125),
    lambda c: c['training'].update(precision='bfloat16'),
    lambda c: c['training'].update(optimizer='SGD'),
    lambda c: c['training'].update(schedule='constant'),
    lambda c: c['training'].update(unknown_knob=1),
    lambda c: c['training'].pop('checkpoint_every'),
    lambda c: c['training'].update(checkpoint_every=100),
    lambda c: c['data'].update(matched_ids='other_ids.json'),
    lambda c: c['data'].update(source_train='outputs/manifests/other/source_train.jsonl'),
    lambda c: c['data'].update(outputs_root='elsewhere'),
    lambda c: c['data'].update(mode='G'),
    lambda c: c['data'].update(generated_manifest='gen.jsonl')])
def test_invalid_production_configs_rejected(edit):
    config = yaml.safe_load(Path('configs/da3.yaml').read_text())
    edit(config)
    with pytest.raises(ValueError):
        T.validate_config(config)


def test_build_datasets_rejects_target_or_swapped_manifests_before_construction(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(T, 'ManifestDataset', lambda *a, **k: called.append(a))
    config = yaml.safe_load(Path('configs/da3_smoke.yaml').read_text())

    def manifest(name, split):
        path = tmp_path / f'{name}.jsonl'
        path.write_text('')
        path.with_suffix('.meta.json').write_text(json.dumps({'split': split}))
        return str(path)

    for role in ('source_train', 'source_validation'):
        other = 'source_validation' if role == 'source_train' else 'source_train'
        for wrong in ('target_test', 'target_appearance_train', other):
            bad = copy.deepcopy(config)
            bad['data'][role] = manifest(f'{role}_{wrong}', wrong)
            bad['data'][other] = manifest(f'{other}_right', other)       # fixtures only; no real manifests touched
            with pytest.raises(ValueError):
                T.build_datasets(bad, tmp_path)
    assert called == []                       # no adapter was constructed for a wrong role


def test_training_path_has_no_target_inputs_and_validation_must_be_source(world):
    config = make(world)
    wrong = FakeSplit(world['tmp'], 'target_test', 'target_native', 2, 56, name='c2')
    world['validation'] = wrong
    with pytest.raises(ValueError):
        run(world, 'wrong_validation', config)
    import inspect
    assert not [p for p in inspect.signature(T.train).parameters if 'target' in p or 'test' in p]


# -- learning-rate schedule and optimization ---------------------------------------------------------------

def test_warmup_cosine_schedule_values():
    f = T.learning_rate_factor
    assert [f(i, 1000, 100) for i in (0, 49, 99)] == pytest.approx([.01, .5, 1.])
    assert f(100, 1000, 100) == pytest.approx(1.) and f(550, 1000, 100) == pytest.approx(.5)
    assert 0 <= f(999, 1000, 100) < 1e-4


def test_training_learns_logs_lr_and_selects_by_source_validation(world):
    out, state = run(world, 'learn', make(world))
    records = log(out)
    assert [r['step'] for r in records] == list(range(1, 7))
    assert records[-1]['loss']['total'] < .8 * records[0]['loss']['total']
    assert [r['learning_rate'] for r in records[:2]] == pytest.approx([.025, .05])
    status = json.loads((out / 'status.json').read_text())
    assert status['completed_steps'] == 6 and status['selected_step'] == state['best_step']
    assert state['best_step'] > 0                      # the bias-initialized fake head improves on step zero
    best = read_head(out / 'best.pt')
    assert best['selected_step'] == state['best_step'] and best['head_sha256'] == state['best_head_sha256']
    assert {p.name for p in (out / 'validation').iterdir()} == {'step_000000', 'step_000003', 'step_000006'}
    assert all(r['smoke_only'] for r in records)


def test_microbatching_matches_full_batch_gradients(world):
    a, _ = run(world, 'micro1', make(world, microbatch_size=1))
    b, _ = run(world, 'micro4', make(world, microbatch_size=4))
    ha, hb = last(a)['head_state_dict'], last(b)['head_state_dict']
    for k in ha:
        assert torch.allclose(ha[k], hb[k], atol=1e-5), k
    assert [r['loss']['total'] for r in log(a)] == pytest.approx([r['loss']['total'] for r in log(b)], rel=1e-5)


def test_initial_head_is_identical_and_backbone_never_changes(world):
    out, state = run(world, 'init', make(world))
    assert state['initial_head_sha256'] == head_digest(fake_model().head_state_dict())
    model = fake_model()
    saved = last(out)['head_state_dict']
    assert not torch.equal(saved['scratch.depth_conv.bias'], model.head_state_dict()['scratch.depth_conv.bias'])
    assert torch.equal(saved['scratch.sky_output_conv2.weight'], model.head_state_dict()['scratch.sky_output_conv2.weight'])


# -- same source order across modes ------------------------------------------------------------------------

def test_source_order_identical_across_rgb_modes_while_appearance_differs(world):
    orders, losses = {}, {}
    for mode in 'RAGM':
        out, _ = run(world, f'mode_{mode}', make(world, mode=mode))
        orders[mode] = [r['sample_ids'] for r in log(out)]
        losses[mode] = [r['loss']['total'] for r in log(out)]
    assert orders['R'] == orders['A'] == orders['G'] == orders['M']
    assert len({tuple(v) for v in losses.values()}) == 4          # RGB differs, order and labels do not
    flat = [s for batch in orders['R'][:1] for s in batch]
    assert len(flat) == 4 and all(s in world['ids'] for s in flat)


# -- exact resume --------------------------------------------------------------------------------------------

def assert_runs_identical(a, b):
    ca, cb = last(a), last(b)
    assert ca['state'] == cb['state']
    for k in ca['head_state_dict']:
        assert torch.equal(ca['head_state_dict'][k], cb['head_state_dict'][k]), k
    sa, sb = ca['optimizer']['state'], cb['optimizer']['state']
    assert ca['optimizer']['param_groups'] == cb['optimizer']['param_groups']
    for key in sa:
        assert torch.equal(sa[key]['exp_avg'], sb[key]['exp_avg']) and torch.equal(sa[key]['exp_avg_sq'], sb[key]['exp_avg_sq'])
    assert ca['scheduler'] == cb['scheduler']
    assert ca['rng']['python'] == cb['rng']['python'] and torch.equal(ca['rng']['torch'], cb['rng']['torch'])
    assert log(a) == log(b)
    ba, bb = torch.load(a / 'best.pt', weights_only=False), torch.load(b / 'best.pt', weights_only=False)
    assert ba['head_sha256'] == bb['head_sha256'] and ba['selected_step'] == bb['selected_step']
    for name in sorted(p.name for p in (a / 'validation').iterdir()):
        assert (a / 'validation' / name / 'frames.json').read_text() == (b / 'validation' / name / 'frames.json').read_text()


@pytest.mark.parametrize('mode,stop', [('R', 1), ('R', 4), ('A', 3), ('M', 3), ('M', 4)])
def test_resumed_run_equals_uninterrupted_run_exactly(world, mode, stop):
    config = make(world, mode=mode)
    full, _ = run(world, f'full_{mode}_{stop}', config)
    part, _ = run(world, f'part_{mode}_{stop}', config, stop=stop)
    assert json.loads((part / 'status.json').read_text())['completed_steps'] == stop
    run_dir = part
    resumed, _ = run_resume(world, run_dir, config)
    assert_runs_identical(full, resumed)


def run_resume(world, out, config, model_fn=fake_model, **kw):
    ids = rm.load_matched_ids(config['data']['matched_ids'], world['source'], smoke=True)
    dataset = rm.RGBModeDataset(world['source'], ids, mode=config['data']['mode'], seed=config['training']['seed'],
                                generated_manifest=config['data']['generated_manifest'], outputs_root=config['data']['outputs_root'])
    compat = T.compatibility(config, world['source'], world['validation'])
    state = T.train(model_fn(), dataset, world['validation'], config, out, compat, resume=out / 'last.pt', **kw)
    return out, state


def test_resume_after_two_interruptions_and_repeated_resume_is_noop(world):
    config = make(world, mode='M')
    full, _ = run(world, 'full', config)
    part, _ = run(world, 'part', config, stop=2)
    run_resume(world, part, config, stop_after_step=5)
    run_resume(world, part, config)
    assert_runs_identical(full, part)
    run_resume(world, part, config)                      # resuming a finished run changes nothing
    assert_runs_identical(full, part)


def test_resume_discards_records_written_after_last_checkpoint(world):
    config = make(world)
    part, _ = run(world, 'part', config, stop=4)
    with (part / 'train.jsonl').open('a') as stream:     # simulated crash between log append and checkpoint
        stream.write(json.dumps({'step': 5, 'sample_ids': ['x'], 'loss': {}}) + '\n')
    (part / 'validation/step_000009').mkdir()
    run_resume(world, part, config)
    full, _ = run(world, 'full', config)
    assert_runs_identical(full, part)
    assert not (part / 'validation/step_000009').exists()


def test_crash_between_last_and_best_checkpoint_is_repaired_from_the_same_head(world):
    config = make(world)
    part, state = run(world, 'part', config, stop=3)
    assert state['best_step'] == 3
    (part / 'best.pt').unlink()
    run_resume(world, part, config)
    full, _ = run(world, 'full', config)
    assert_runs_identical(full, part)


def test_tampered_or_stale_best_head_blocks_resume(world):
    config = make(world)
    part, _ = run(world, 'part', config, stop=4)       # best may be step 3 while last is step 4
    best = torch.load(part / 'best.pt', weights_only=False)
    best['head_state_dict']['scratch.depth_conv.bias'] += 1
    torch.save(best, part / 'best.pt')
    with pytest.raises(ValueError):
        run_resume(world, part, config)
    (part / 'best.pt').unlink()                        # missing and not repairable (best step != current step)
    assert last(part)['state']['best_step'] != last(part)['state']['optimizer_step']
    with pytest.raises(ValueError):
        run_resume(world, part, config)


def test_new_run_refuses_to_overwrite_and_early_stop_is_smoke_only(world):
    config = make(world)
    run(world, 'dup', config)
    with pytest.raises(ValueError):
        run(world, 'dup', config)
    with pytest.raises(ValueError):
        T.train(fake_model(), None, world['validation'], config, world['tmp'] / 'p', {'smoke_only': False}, stop_after_step=2)


# -- resume rejection on config/data/code changes -------------------------------------------------------------

def changed(world, config, edit):
    new = copy.deepcopy(config)
    edit(new)
    return new


@pytest.mark.parametrize('edit', [
    lambda c: c['training'].update(learning_rate=0.03),
    lambda c: c['training'].update(weight_decay=0.1),
    lambda c: c['training'].update(seed=1),
    lambda c: c['training'].update(global_batch_size=8),
    lambda c: c['training'].update(microbatch_size=1),
    lambda c: c['training'].update(optimizer_steps=8),
    lambda c: c['training'].update(warmup_steps=1),
    lambda c: c['training'].update(gradient_clip_norm=.5),
    lambda c: c['training'].update(gradient_weight=0.5),
    lambda c: c['training'].update(validation_every=2),
    lambda c: c['data'].update(mode='A'),
    lambda c: c['data'].update(augmentation={'brightness': [.8, 1.1], 'color': [.9, 1.1], 'gamma': [.9, 1.1]}),
    lambda c: c['evaluation'].update(validation_frames=3)])
def test_resume_rejects_changed_configuration(world, edit):
    config = make(world)
    part, _ = run(world, 'part', config, stop=3)
    with pytest.raises(ValueError, match='Resume mismatch'):
        run_resume(world, part, changed(world, config, edit))


def test_resume_rejects_changed_data_files_and_code_and_environment(world, monkeypatch):
    config = make(world, mode='G')
    part, _ = run(world, 'part', config, stop=3)
    original = Path(config['data']['generated_manifest']).read_text()
    for target, edit in ((world['ids_path'], lambda t: t.write_text(t.read_text() + ' ')),
                         (world['source'].path, lambda t: t.write_text(t.read_text() + '\n')),
                         (world['source'].path.with_suffix('.meta.json'), lambda t: t.write_text(t.read_text() + ' ')),
                         (world['validation'].path, lambda t: t.write_text(t.read_text() + '\n')),
                         (Path(config['data']['generated_manifest']), lambda t: t.write_text(t.read_text() + ' '))):
        before = target.read_text()
        edit(target)
        with pytest.raises(ValueError, match='Resume mismatch'):
            compat = T.compatibility(config, world['source'], world['validation'])
            torch.load  # noqa
            T.train(fake_model(), None, world['validation'], config, part, compat, resume=part / 'last.pt')
        target.write_text(before)
    assert Path(config['data']['generated_manifest']).read_text() == original
    monkeypatch.setattr(T, 'code_hashes', lambda: {'depth/train.py': '0' * 64})
    with pytest.raises(ValueError, match='Resume mismatch'):
        T.train(fake_model(), None, world['validation'], config, part, T.compatibility(config, world['source'], world['validation']),
                resume=part / 'last.pt')
    monkeypatch.undo()
    monkeypatch.setattr(T, 'environment_versions', lambda: {'torch': '0.0'})
    with pytest.raises(ValueError, match='Resume mismatch'):
        T.train(fake_model(), None, world['validation'], config, part, T.compatibility(config, world['source'], world['validation']),
                resume=part / 'last.pt')
    monkeypatch.undo()
    run_resume(world, part, config)                     # unchanged inputs still resume


def test_raw_checkpoint_cannot_resume_as_generated_run(world):
    raw = make(world, mode='R')
    part, _ = run(world, 'part', raw, stop=2)
    with pytest.raises(ValueError, match='Resume mismatch'):
        run_resume(world, part, make(world, mode='G'))


# -- RGB-mode overrides, checkpoint cadence and recovery -----------------------------------------------------

def test_only_mode_and_generated_manifest_vary_across_experiment_runs():
    base = yaml.safe_load(Path('configs/da3.yaml').read_text())
    variants = {m: T.apply_rgb_overrides(base, m, 'outputs/gen/manifest.jsonl' if m in 'GM' else None) for m in 'RAGM'}
    for variant in variants.values():
        T.validate_config(variant)
    def strip(config):
        config = copy.deepcopy(config)
        config['data'].pop('mode'), config['data'].pop('generated_manifest')
        return config
    assert len({json.dumps(strip(v), sort_keys=True) for v in variants.values()}) == 1
    assert base['data']['mode'] == 'R' and base['data']['generated_manifest'] is None       # input not mutated
    with pytest.raises(ValueError):
        T.validate_config(T.apply_rgb_overrides(base, 'G'))                                  # G needs a manifest
    with pytest.raises(ValueError):
        T.validate_config(T.apply_rgb_overrides(base, 'A', 'outputs/gen/manifest.jsonl'))    # A forbids one


def test_last_checkpoint_cadence_and_run_info(world):
    config = make(world, checkpoint_every=3, validation_every=6)
    out, _ = run(world, 'cadence', config, stop=4)
    assert last(out)['state']['optimizer_step'] == 4                 # written at the stop update
    out2, _ = run(world, 'cadence2', config)
    assert last(out2)['state']['optimizer_step'] == 6
    info = json.loads((out2 / 'run_info.json').read_text())
    assert isinstance(info['deterministic_algorithms'], bool) and info['deterministic_warn_only'] in (True, False) and info['device'] == 'cpu' and 'torch' in info
    with pytest.raises(ValueError):
        T.validate_config(make(world, checkpoint_every=4, validation_every=6))


@pytest.mark.parametrize('mode', ['R', 'M'])
def test_resume_from_an_older_checkpoint_after_progress_is_bitwise_identical(world, mode):
    config = make(world, mode=mode, checkpoint_every=3, validation_every=6)
    full, _ = run(world, f'full_{mode}', config)
    part, _ = run(world, f'part_{mode}', config, stop=3)
    older = world['tmp'] / f'older_{mode}.pt'
    older.write_bytes((part / 'last.pt').read_bytes())
    run_resume(world, part, config, stop_after_step=5)               # progress beyond the old checkpoint, then "crash"
    assert len(log(part)) == 5
    ids = rm.load_matched_ids(config['data']['matched_ids'], world['source'], smoke=True)
    dataset = rm.RGBModeDataset(world['source'], ids, mode=mode, seed=0, generated_manifest=config['data']['generated_manifest'],
                                outputs_root=config['data']['outputs_root'])
    T.train(fake_model(), dataset, world['validation'], config, part, T.compatibility(config, world['source'], world['validation']),
            resume=older)
    assert_runs_identical(full, part)


def test_resume_tolerates_a_truncated_final_log_line_but_not_corruption_elsewhere(world):
    config = make(world)
    part, _ = run(world, 'part', config, stop=4)
    with (part / 'train.jsonl').open('a') as stream:
        stream.write('{"step": 5, "sample_ids": ["x')              # interrupted write, no newline
    run_resume(world, part, config)
    full, _ = run(world, 'full', config)
    assert_runs_identical(full, part)
    lines = (part / 'train.jsonl').read_text().splitlines()
    lines[1] = lines[1][:20]
    (part / 'train.jsonl').write_text('\n'.join(lines) + '\n')
    with pytest.raises(json.JSONDecodeError):
        T._truncate_logs(part, 6)
