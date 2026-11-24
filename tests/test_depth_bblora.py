"""bbLoRA variant (LoRA on the last K backbone blocks + trainable head): config lock, model, checkpoints, training."""
import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from da3_fakes import fake_bb_model, fake_model
from generative_sim2real.depth import checkpoints as ck
from generative_sim2real.depth import train as T
from generative_sim2real.depth.checkpoints import head_digest, read_head
from generative_sim2real.depth.model import LOCKED_BACKBONE_LORA
from test_depth_train import assert_runs_identical, last, log, make, run, run_resume, world   # noqa: F401  (fixture)

LORA = copy.deepcopy(LOCKED_BACKBONE_LORA)


# ---------------------------------------------------------------- configuration lock

def test_bblora_config_validates_and_differs_from_head_only_only_in_the_model_block():
    head_only = yaml.safe_load(Path('configs/da3.yaml').read_text())
    bblora = yaml.safe_load(Path('configs/da3_bblora.yaml').read_text())
    T.validate_config(bblora)
    assert bblora['model']['trainable'] == 'head+backbone_lora' and bblora['model']['backbone_lora'] == LORA
    assert LORA == {'rank': 8, 'alpha': 8, 'last_k_blocks': 4, 'targets': ['attn.qkv', 'attn.proj']}
    assert {k: v for k, v in bblora.items() if k != 'model'} == {k: v for k, v in head_only.items() if k != 'model'}
    assert bblora['training']['optimizer_steps'] == 3000 and bblora['training']['global_batch_size'] == 16
    for mode in 'RAGM':
        T.validate_config(T.apply_rgb_overrides(bblora, mode, 'outputs/g.jsonl' if mode in 'GM' else None, seed=2))


@pytest.mark.parametrize('edit', [
    lambda m: m['backbone_lora'].update(rank=4),
    lambda m: m['backbone_lora'].update(alpha=16),
    lambda m: m['backbone_lora'].update(alpha=8.5),
    lambda m: m['backbone_lora'].update(last_k_blocks=2),
    lambda m: m['backbone_lora'].update(targets=['attn.qkv']),
    lambda m: m['backbone_lora'].update(targets=['attn.proj', 'attn.qkv']),
    lambda m: m['backbone_lora'].update(extra=1),
    lambda m: m.pop('backbone_lora'),
    lambda m: m.update(trainable='head'),
    lambda m: m.update(trainable='backbone_lora'),
    lambda m: m.update(trainable='all')])
def test_only_the_exact_bblora_variant_is_accepted_for_experiments(edit):
    config = yaml.safe_load(Path('configs/da3_bblora.yaml').read_text())
    edit(config['model'])
    with pytest.raises(ValueError):
        T.validate_config(config)


def test_head_only_config_with_a_lora_block_is_rejected():
    config = yaml.safe_load(Path('configs/da3.yaml').read_text())
    config['model']['backbone_lora'] = copy.deepcopy(LORA)
    with pytest.raises(ValueError):
        T.validate_config(config)


# ---------------------------------------------------------------- model

def test_lora_starts_as_identity_and_only_the_last_blocks_and_head_receive_gradients():
    plain, model = fake_bb_model(), fake_bb_model()
    rgb, K = torch.rand(2, 3, 56, 56), torch.tensor([[600., 0, 28], [0, 600., 28], [0, 0, 1]]).expand(2, 3, 3)
    before = {k: v.clone() for k, v in model.network.state_dict().items()}
    parameters = model.configure_backbone_lora_training(LORA, seed=0)
    assert torch.equal(model(rgb, K)['depth_m'], plain(rgb, K)['depth_m'])          # B = 0: exactly the official model
    assert {k: torch.equal(v, model.network.state_dict()[k]) for k, v in before.items()} == {k: True for k in before}
    names = sorted(model.backbone_lora.state_dict())
    assert names == sorted(f'{i}_{t}.lora_{ab}' for i in (1, 2, 3, 4) for t in ('attn_qkv', 'attn_proj') for ab in 'AB')
    lora = {n: p for n, p in model.backbone_lora.named_parameters()}
    assert lora['1_attn_qkv.lora_A'].shape == (8, 4) and lora['1_attn_qkv.lora_B'].shape == (12, 8)
    head_params = [q for n, q in model.network.named_parameters() if n.startswith('head.') and q.requires_grad]
    assert len(parameters) == len(head_params) + len(lora)
    trainable = {id(p) for p in parameters}
    assert all(id(p) in trainable for p in lora.values())
    assert not any(p.requires_grad for n, p in model.network.named_parameters() if not n.startswith('head.'))
    with torch.no_grad():
        for name, p in model.backbone_lora.named_parameters():
            if name.endswith('lora_B'):
                p.add_(.1)
    model(rgb, K)['canonical_depth'].sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora.values())
    assert all(p.grad is None for n, p in model.network.named_parameters() if n.startswith('backbone.'))
    assert model.network.backbone.pretrained.blocks[0].attn.qkv.weight.grad is None        # block 0 is not adapted
    assert not torch.equal(model(rgb, K)['depth_m'], plain(rgb, K)['depth_m'])


def test_lora_init_is_seeded_and_independent_of_global_rng():
    a, b, c = fake_bb_model(), fake_bb_model(), fake_bb_model()
    a.attach_backbone_lora(LORA, seed=3)
    torch.manual_seed(999)
    b.attach_backbone_lora(LORA, seed=3)
    c.attach_backbone_lora(LORA, seed=4)
    assert head_digest(a.backbone_lora_state_dict()) == head_digest(b.backbone_lora_state_dict())
    assert head_digest(a.backbone_lora_state_dict()) != head_digest(c.backbone_lora_state_dict())


def test_attach_detach_roundtrip_and_validation_errors():
    model = fake_bb_model()
    rgb, K = torch.rand(1, 3, 56, 56), torch.tensor([[600., 0, 28], [0, 600., 28], [0, 0, 1]])[None]
    reference = model(rgb, K)['depth_m']
    model.attach_backbone_lora(LORA)
    with torch.no_grad():
        model.backbone_lora['4_attn_proj'].lora_B.add_(.5)
    assert not torch.equal(model(rgb, K)['depth_m'], reference)
    with pytest.raises(ValueError, match='already attached'):
        model.attach_backbone_lora(LORA)
    model.detach_backbone_lora()
    assert model.backbone_lora is None and not model._lora_hooks and torch.equal(model(rgb, K)['depth_m'], reference)
    with pytest.raises(ValueError, match='No backbone LoRA'):
        model.backbone_lora_state_dict()
    for bad in ({**LORA, 'rank': 0}, {**LORA, 'alpha': -1}, {**LORA, 'targets': []}, {**LORA, 'targets': ['mlp.fc1']},
                {**LORA, 'last_k_blocks': 0}, {k: v for k, v in LORA.items() if k != 'rank'}):
        with pytest.raises(ValueError):
            fake_bb_model().attach_backbone_lora(bad)
    with pytest.raises(ValueError, match='exceeds'):
        fake_bb_model().attach_backbone_lora({**LORA, 'last_k_blocks': 6})
    with pytest.raises(ValueError, match='pretrained.blocks'):
        fake_model().attach_backbone_lora(LORA)


def test_failed_attach_leaves_no_hooks_and_no_lora():
    model = fake_bb_model()
    model.network.backbone.pretrained.blocks[4].attn.proj = torch.nn.Identity()      # the last target is not a Linear
    with pytest.raises(ValueError, match='not a Linear'):
        model.attach_backbone_lora(LORA)
    assert model.backbone_lora is None and not model._lora_hooks
    assert all(not block.attn.qkv._forward_hooks and not block.attn.proj._forward_hooks
               for block in model.network.backbone.pretrained.blocks)


def test_official_digests_ignore_the_lora_wrapper_and_state_dict_is_strict():
    from generative_sim2real.evaluation import registry as reg
    model = fake_bb_model()
    before = reg.official_digests(model)
    model.attach_backbone_lora(LORA)
    assert reg.official_digests(model) == before                    # hooks, not module replacement: keys are unchanged
    state = model.backbone_lora_state_dict()
    with pytest.raises(RuntimeError):
        model.load_backbone_lora_state_dict({k: v for k, v in list(state.items())[1:]})


# ---------------------------------------------------------------- checkpoints

def perturbed_lora_model(seed=0):
    model = fake_bb_model()
    model.attach_backbone_lora(LORA, seed=seed)
    with torch.no_grad():
        for name, p in model.backbone_lora.named_parameters():
            p.add_(.05 if name.endswith('lora_B') else 0.)
    return model


def test_bblora_best_head_carries_both_states_and_head_only_readers_cannot_ignore_it(tmp_path):
    model = perturbed_lora_model()
    state = {'best_step': 3, 'best_abs_rel': .1}
    ck.save_head(tmp_path / 'best.pt', model, state, {'smoke_only': True})
    with pytest.raises(ValueError, match='allow_backbone_lora'):
        read_head(tmp_path / 'best.pt')
    best = read_head(tmp_path / 'best.pt', allow_backbone_lora=True)
    assert best['kind'] == 'da3_head_bblora' and best['backbone_lora_config'] == LORA and best['selected_step'] == 3
    assert best['backbone_lora_sha256'] == head_digest(model.backbone_lora_state_dict())
    assert best['head_sha256'] == head_digest(model.head_state_dict())
    broken = torch.load(tmp_path / 'best.pt', weights_only=False)
    broken['backbone_lora_state_dict']['1_attn_qkv.lora_B'] += 1
    torch.save(broken, tmp_path / 'broken.pt')
    with pytest.raises(ValueError, match='LoRA'):
        read_head(tmp_path / 'broken.pt', allow_backbone_lora=True)
    # round trip into a fresh model reproduces predictions bit for bit
    fresh = fake_bb_model()
    ck.apply_checkpoint(fresh, best)
    rgb, K = torch.rand(1, 3, 56, 56), torch.tensor([[600., 0, 28], [0, 600., 28], [0, 0, 1]])[None]
    assert torch.equal(fresh(rgb, K)['depth_m'], model(rgb, K)['depth_m'])
    ck.apply_checkpoint(fresh, {'head_state_dict': fake_bb_model().head_state_dict()})        # head-only record removes it
    assert fresh.backbone_lora is None


def test_head_only_checkpoints_keep_the_original_format(tmp_path):
    model = fake_bb_model()
    ck.save_head(tmp_path / 'best.pt', model, {'best_step': 1, 'best_abs_rel': .2}, {'smoke_only': True})
    raw = torch.load(tmp_path / 'best.pt', weights_only=False)
    assert list(raw) == ['format_version', 'kind', 'head_state_dict', 'head_sha256', 'selected_step',
                         'source_validation_sequence_macro_abs_rel', 'compatibility'] and raw['kind'] == 'da3_head'
    assert read_head(tmp_path / 'best.pt')['head_sha256'] == head_digest(model.head_state_dict())    # old call pattern
    fresh = fake_bb_model()
    fresh.attach_backbone_lora(LORA)
    ck.apply_checkpoint(fresh, read_head(tmp_path / 'best.pt'))
    assert fresh.backbone_lora is None


# ---------------------------------------------------------------- training

def lora_run(world, name, config, **kw):
    return run(world, name, config, model_fn=fake_bb_model, **kw)


def lora_state(path):
    return torch.load(path, weights_only=False)['backbone_lora_state_dict']


def test_bblora_training_learns_selects_and_saves_both_states(world):
    out, state = lora_run(world, 'lora', make(world, lora=True))
    records = log(out)
    assert [r['step'] for r in records] == list(range(1, 7))
    assert records[-1]['loss']['total'] < .8 * records[0]['loss']['total']
    best = read_head(out / 'best.pt', allow_backbone_lora=True)
    assert best['selected_step'] == state['best_step'] > 0
    assert best['head_sha256'] == state['best_head_sha256'] and best['backbone_lora_sha256'] == state['best_backbone_lora_sha256']
    final = last(out)
    assert set(final['backbone_lora_state_dict']) == set(best['backbone_lora_state_dict'])
    assert head_digest(final['backbone_lora_state_dict']) == head_digest(best['backbone_lora_state_dict']) or state['best_step'] != 6
    initial = fake_bb_model()
    initial.attach_backbone_lora(LORA, seed=0)
    assert state['initial_backbone_lora_sha256'] == head_digest(initial.backbone_lora_state_dict())
    assert any(v.abs().sum() > 0 for k, v in final['backbone_lora_state_dict'].items() if k.endswith('lora_B'))
    base = fake_bb_model().network.state_dict()
    saved_head = final['head_state_dict']
    assert set(saved_head) == set(fake_bb_model().head_state_dict())            # the frozen backbone is never saved
    assert not torch.equal(saved_head['scratch.depth_conv.bias'], base['head.scratch.depth_conv.bias'])
    assert json.loads((out / 'resolved.json').read_text())['config']['model']['backbone_lora'] == LORA


def test_default_head_only_run_is_unchanged_by_the_bblora_code_path(world):
    out, state = run(world, 'head', make(world))
    assert 'best_backbone_lora_sha256' not in state and 'initial_backbone_lora_sha256' not in state
    assert 'backbone_lora_state_dict' not in last(out)
    assert torch.load(out / 'best.pt', weights_only=False)['kind'] == 'da3_head'
    again, _ = run(world, 'head_again', make(world))
    assert_runs_identical(out, again)


def test_bblora_head_trajectory_differs_from_head_only_but_starts_identically(world):
    head_only, _ = run(world, 'h', make(world), model_fn=fake_bb_model)
    lora, _ = lora_run(world, 'l', make(world, lora=True))
    assert [r['sample_ids'] for r in log(head_only)] == [r['sample_ids'] for r in log(lora)]        # same sampling order
    assert log(head_only)[0]['loss'] == log(lora)[0]['loss']                                         # B = 0 at step one
    assert [r['loss']['total'] for r in log(head_only)] != [r['loss']['total'] for r in log(lora)]


def assert_lora_runs_identical(a, b):
    assert_runs_identical(a, b)
    assert head_digest(lora_state(a / 'last.pt')) == head_digest(lora_state(b / 'last.pt'))
    ba, bb = torch.load(a / 'best.pt', weights_only=False), torch.load(b / 'best.pt', weights_only=False)
    assert ba['backbone_lora_sha256'] == bb['backbone_lora_sha256']


@pytest.mark.parametrize('stop', [1, 3, 4])
def test_bblora_resume_equals_uninterrupted_run_exactly(world, stop):
    config = make(world, lora=True)
    full, _ = lora_run(world, f'full{stop}', config)
    part, _ = lora_run(world, f'part{stop}', config, stop=stop)
    run_resume(world, part, config, model_fn=fake_bb_model)
    assert_lora_runs_identical(full, part)


def test_bblora_crash_between_last_and_best_is_repaired_and_tampering_blocks_resume(world):
    config = make(world, lora=True)
    part, state = lora_run(world, 'part', config, stop=3)
    assert state['best_step'] == 3
    (part / 'best.pt').unlink()                                      # best.pt missing, but last.pt records that best
    run_resume(world, part, config, model_fn=fake_bb_model)
    full, _ = lora_run(world, 'full', config)
    assert_lora_runs_identical(full, part)
    part2, _ = lora_run(world, 'part2', config, stop=4)             # best may be an earlier step than last
    best = torch.load(part2 / 'best.pt', weights_only=False)
    best['backbone_lora_state_dict']['1_attn_qkv.lora_A'] += 1
    best['backbone_lora_sha256'] = head_digest(best['backbone_lora_state_dict'])    # digest-consistent but different LoRA
    torch.save(best, part2 / 'best.pt')
    with pytest.raises(ValueError, match='Resume'):
        run_resume(world, part2, config, model_fn=fake_bb_model)


def test_resume_refuses_a_checkpoint_of_the_other_model_variant(world):
    lora_cfg, head_cfg = make(world, lora=True), make(world)
    part, _ = lora_run(world, 'part', lora_cfg, stop=2)
    with pytest.raises(ValueError, match='Resume mismatch'):
        run_resume(world, part, head_cfg, model_fn=fake_bb_model)
    head_part, _ = run(world, 'head_part', head_cfg, stop=2, model_fn=fake_bb_model)
    with pytest.raises(ValueError, match='Resume mismatch'):
        run_resume(world, head_part, lora_cfg, model_fn=fake_bb_model)


def test_different_seeds_give_different_lora_initialisations_and_runs(world):
    a, sa = lora_run(world, 'seed0', make(world, lora=True))
    b, sb = lora_run(world, 'seed1', make(world, lora=True, seed=1))
    assert sa['initial_backbone_lora_sha256'] != sb['initial_backbone_lora_sha256']
    assert sa['initial_head_sha256'] == sb['initial_head_sha256']                  # both start from the official head
