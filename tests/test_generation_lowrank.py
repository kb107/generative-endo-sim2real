"""Low-capacity (Control-LoRA style) controller: tiny CPU fixtures, no downloads.

Covers zero output at init, the exact trainable set, gradients through the frozen backbone and
source LoRA, save/load round trip and file-kind dispatch, an end-to-end resumable training run,
the unchanged default 'full' path, and recipe-hash stability."""
import json
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip('diffusers')
pytest.importorskip('peft')
import generation_tiny as tiny  # noqa: E402
from diffusers import DDPMScheduler  # noqa: E402
from test_generation_bulk import write_recipe  # noqa: E402,F401
from test_generation_p4_run import env  # noqa: E402,F401  (fixture reuse)

from generative_sim2real.generation import bulk, gate, train  # noqa: E402
from generative_sim2real.generation import lowrank_controller as lrc  # noqa: E402
from generative_sim2real.generation import model as gm  # noqa: E402
from generative_sim2real.generation import recipe as recipe_module  # noqa: E402
from generative_sim2real.generation import training  # noqa: E402

ROOT = Path(__file__).parents[1]
EMBED = tiny.tiny_prompt()
SPEC = {'kind': 'lowrank', 'block_rank': 2, 'before_rank': 3, 'after_rank': 2, 'init_seed': 7}
INNER = 16  # tiny model: 2 heads x 8


def expected_trainable_elements(spec, layers=2):
    block = 8 * spec['block_rank'] * (INNER + INNER)      # 2 attentions x (q, k, v, out)
    after = spec['after_rank'] * 2 * INNER + INNER         # down + up + up bias
    before = spec['before_rank'] * 2 * INNER + INNER
    return layers * (block + after) + before


def converted(spec=SPEC):
    model, controller, provenance = tiny.tiny_model()
    info = lrc.convert(controller, spec)
    return model, controller, provenance, info


def forward(model, control, seed=3):
    latents, ctrl = tiny.batch(seed=seed)
    with torch.no_grad():
        return model(latents, encoder_hidden_states=EMBED['positive'].expand(2, -1, -1),
                     encoder_attention_mask=EMBED['positive_mask'].expand(2, -1),
                     timestep=torch.tensor([5, 500]), return_dict=False,
                     added_cond_kwargs={'resolution': None, 'aspect_ratio': None},
                     controlnet_cond=ctrl if control else None)[0]


def test_output_is_exactly_zero_at_init_and_provenance_is_recorded():
    model, controller, provenance, info = converted()
    assert info['zero_at_init'] and provenance['projections_zero_at_init']
    assert all(provenance['copied_block_hashes_equal_base'])
    assert info['frozen_blocks_unchanged_by_conversion']
    assert torch.equal(forward(model, True), forward(model, False))
    # frozen copied blocks equal the clean base of a freshly built identical transformer
    fresh = tiny.tiny_transformer()
    for i in range(len(controller.controlnet_blocks)):
        prefix = f'controlnet_blocks.{i}.transformer_block.'
        copied = [(n[len(prefix):], t) for n, t in lrc.frozen_items(controller) if n.startswith(prefix)]
        assert gm.tensors_hash(fresh.transformer_blocks[i].state_dict().items()) == gm.tensors_hash(copied)


def test_only_the_intended_parameters_are_trainable():
    model, controller, _, info = converted()
    named = gm.set_stage(model.transformer, controller, stage='controller', adapter='source')
    named = lrc.restrict_trainable(controller, model.transformer)
    assert named and all(n.startswith('controller.') for n, _ in named)
    assert all(lrc.is_trainable_name(n.removeprefix('controller.')) for n, _ in named)
    assert sum(p.numel() for _, p in named) == expected_trainable_elements(SPEC)
    assert info['trainable_elements'] == expected_trainable_elements(SPEC)
    assert info['full_controller_trainable_elements'] > info['trainable_elements']
    # backbone (incl. both PEFT adapters) frozen; copied block weights frozen
    assert not any(p.requires_grad for p in model.transformer.parameters())
    assert not any(p.requires_grad for n, p in controller.named_parameters()
                   if not lrc.is_trainable_name(n))
    # block 0 alone has a before_proj; every block has an after_proj; 8 LoRA'd linears per block
    names = [n for n, _ in named]
    assert sum('before_proj.lr_down' in n for n in names) == 1
    assert sum('after_proj.lr_up.bias' in n for n in names) == 2
    assert sum('attn2.to_out.0.lr_up' in n for n in names) == 2 and not any('.ff.' in n for n in names)


def batches(n, size=2):
    g = torch.Generator().manual_seed(0)
    while True:
        yield {'rgb': torch.rand(size, 3, 64, 64, generator=g),
               'control': torch.rand(size, 3, 64, 64, generator=g)}


def test_gradients_reach_every_trainable_tensor_through_the_frozen_backbone():
    model, controller, _, info = converted()
    gm.set_stage(model.transformer, controller, stage='controller', adapter='source')
    named = lrc.restrict_trainable(controller, model.transformer)
    backbone = gm.state_hash(model.transformer)
    frozen = lrc.frozen_hash(controller)
    records, evidence = training.train_updates(
        model, tiny.tiny_vae(), DDPMScheduler.from_config(tiny.tiny_scheduler().config),
        tiny.tiny_prompt(), named, batches(40), updates=8, accumulation=1, lr=1e-2,
        weight_decay=0.0, max_grad_norm=1e9, use_control=True, seed=0, device='cpu',
        dtype=torch.float32, probe_gradients=True)
    assert evidence['tensors_never_receiving_gradient'] == [], evidence
    assert evidence['ever_nonzero_grad_tensors'] == len(named)
    assert gm.state_hash(model.transformer) == backbone   # backbone and source LoRA untouched
    assert lrc.frozen_hash(controller) == frozen          # copied blocks untouched
    assert not torch.equal(forward(model, True), forward(model, False))  # the controller now acts


def test_save_load_round_trip_and_kind_dispatch(tmp_path):
    model, controller, *_ = converted()
    with torch.no_grad():  # as if trained
        for n, p in controller.named_parameters():
            if lrc.is_trainable_name(n):
                p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(len(n))) * .1)
    path = tmp_path / lrc.FILE_NAMES['lowrank']
    lrc.save(controller, path, revision='rev')
    from safetensors import safe_open
    with safe_open(str(path), framework='pt') as handle:
        assert set(handle.keys()) == set(lrc.trainable_state(controller))   # no frozen blocks
        assert handle.metadata()['kind'] == 'lowrank' and handle.metadata()['block_rank'] == '2'
    model2, controller2, _ = tiny.tiny_model()
    assert not lrc.is_lowrank(controller2)
    lrc.load_any(controller2, path, revision='rev')   # converts a clean full controller, then loads
    assert lrc.is_lowrank(controller2) and lrc.spec_of(controller2) == lrc.spec_of(controller)
    assert torch.equal(forward(model, True), forward(model2, True))
    assert gm.state_hash(controller) == gm.state_hash(controller2)
    lrc.load_any(controller2, path, revision='rev')   # idempotent into an already converted one
    # no loader can confuse the two kinds
    with pytest.raises(ValueError, match='Not a controller file'):
        gm.load_controller(tiny.tiny_model()[1], path, revision='rev')
    full_path = tmp_path / 'controller.safetensors'
    gm.save_controller(tiny.tiny_model()[1], full_path, revision='rev')
    with pytest.raises(ValueError, match='Not a low-rank'):
        lrc.load(tiny.tiny_model()[1], full_path, revision='rev')
    with pytest.raises(ValueError, match='full controller file cannot be loaded'):
        lrc.load_any(controller2, full_path, revision='rev')
    plain = tiny.tiny_model()[1]
    lrc.load_any(plain, full_path, revision='rev')    # full files still go through gm.load_controller
    assert not lrc.is_lowrank(plain)
    # guards: revision, ranks, and the frozen copied blocks
    with pytest.raises(ValueError, match='base_revision'):
        lrc.load_any(tiny.tiny_model()[1], path, revision='other')
    other = converted({**SPEC, 'block_rank': 1})[1]
    with pytest.raises(ValueError, match='ranks'):
        lrc.load(other, path, revision='rev')
    drifted = converted()[1]
    with torch.no_grad():  # the feed-forward is not LoRA'd, so this is a plain frozen weight
        drifted.controlnet_blocks[0].transformer_block.ff.net[0].proj.weight.add_(1e-3)
    with pytest.raises(ValueError, match='Frozen copied blocks'):
        lrc.load(drifted, path, revision='rev')


def test_spec_resolution_rules():
    assert lrc.resolve_spec(None, None) is None
    assert lrc.resolve_spec({'kind': 'full'}, None) is None
    assert lrc.resolve_spec({'kind': 'lowrank'}, None) == {
        'kind': 'lowrank', 'block_rank': 8, 'before_rank': 32, 'after_rank': 8, 'init_seed': 5303}
    merged = lrc.resolve_spec({'kind': 'lowrank', 'block_rank': 4}, {'after_rank': 16})
    assert (merged['block_rank'], merged['before_rank'], merged['after_rank']) == (4, 32, 16)
    for bad in ({'kind': 'sparse'}, {'kind': 'lowrank', 'block_rank': 0},
                {'kind': 'lowrank', 'block_rank': 2.5}, {'kind': 'full', 'block_rank': 4},
                {'kind': 'lowrank', 'rank': 4}):
        with pytest.raises(ValueError):
            lrc.resolve_spec(bad, None)
    with pytest.raises(ValueError, match='already low-rank'):
        lrc.convert(converted()[1], SPEC)


def test_declared_configs_validate_and_default_config_is_full(monkeypatch):
    monkeypatch.chdir(ROOT)
    for name in ('p4.yaml', 'p4_sweep.yaml'):
        config, generator = train.load_p4(ROOT / 'configs' / name)
        assert all(train.controller_spec(config, r) is None for r in config['runs'].values())
    config, generator = train.load_p4(ROOT / 'configs' / 'p4_lowcap.yaml')
    specs = {n: train.controller_spec(config, r) for n, r in config['runs'].items()
             if r['stage'] == 'controller' and n.startswith('controller_lc')}
    assert len(specs) == 5 and all(s and s['kind'] == 'lowrank' for s in specs.values())
    assert config['deterministic_attention'] is False


# ---------------------------------------------------------- training runs ----

class FakeFrames:
    """Stand-in for inputs.FrameTensors with control images (the p4 test manifests have none)."""

    def __init__(self, manifest, *, with_control, indices=None):
        self.manifest, self.with_control = manifest, with_control
        self.indices = list(range(len(manifest))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.manifest[self.indices[i]]
        out = {'rgb': item['rgb'], 'sample_id': item['sample_id']}
        if self.with_control:
            out['control'] = torch.rot90(item['rgb'], 1, (1, 2)).clamp(0, 1)
        return out


def controller_run(env, name, controller=None, run_controller=None, **kwargs):
    env.monkeypatch.setattr(train, 'FrameTensors', FakeFrames)
    config = env.config
    if controller is not None:
        config['controller'] = controller
    out = env.tmp / 'out' / 'smoke'
    if not (out / 'source_lora' / 'final.json').exists():
        train.run_training(config, env.generator, 'source_lora', output=out / 'source_lora',
                           smoke_steps=6)
    config['runs']['ctl'] = {
        'stage': 'controller', 'adapter': 'source', 'role': 'controller', 'train': 'source_train',
        'heldout': 'source_validation', 'source_lora_run': 'source_lora', 'lr_from': 'controller_lr',
        'lr': 1e-2, 'steps': 6, 'checkpoint_every': 3, 'warmup': 0, 'micro_batch': 2,
        'seeds': {'loader': 13, 'noise': 14}}
    if run_controller is not None:
        config['runs']['ctl']['controller'] = run_controller
    return train.run_training(config, env.generator, 'ctl', output=env.tmp / name, smoke_steps=6,
                              **kwargs)


def exports(env, name, step='step_000006'):
    return sorted(p.name for p in (env.tmp / name / 'checkpoints' / step).iterdir())


def params(env, name):
    return torch.load(env.tmp / name / 'last.pt', map_location='cpu', weights_only=False)


def test_lowrank_training_run_exports_own_file_and_resumes_bitwise(env):
    low = {'kind': 'lowrank', 'block_rank': 2, 'before_rank': 3, 'after_rank': 2}
    final = controller_run(env, 'low', controller=low)
    assert final['completed_all_steps'] and final['steps_completed'] == 6
    assert exports(env, 'low') == ['controller_lowrank.safetensors', 'meta.json']
    assert final['export_file'] == 'controller_lowrank.safetensors'
    summary = final['controller_lowrank']
    assert summary['frozen_blocks_unchanged_by_training'] and summary['zero_at_init']
    assert summary['trainable_tensors_all_lowrank']
    assert final['trainable']['elements'] == expected_trainable_elements({**low}) == summary[
        'trainable_elements']
    assert final['hash_changed_as_expected']['controller']
    meta = json.loads((env.tmp / 'low' / 'checkpoints' / 'step_000006' / 'meta.json').read_text())
    assert meta['controller_kind'] == 'lowrank' and meta['block_rank'] == 2
    resolved = json.loads((env.tmp / 'low' / 'resolved.json').read_text())
    assert resolved['compatibility']['controller_spec']['kind'] == 'lowrank'
    assert resolved['controller_lowrank']['frozen_blocks_sha256']
    lines = [json.loads(x) for x in (env.tmp / 'low' / 'eval.jsonl').read_text().splitlines()]
    assert [r['step'] for r in lines] == [0, 3, 6] and 'heldout_wrong' in lines[0]
    # exact controller-off equality at step 0 (zero output): correct == off
    assert lines[0]['heldout_correct'] == lines[0]['heldout_off']
    # resume equals uninterrupted
    stopped = controller_run(env, 'split', controller=low, stop_after_step=3)
    assert not stopped['completed_all_steps']
    controller_run(env, 'split', controller=low, resume=True)
    a, b = params(env, 'low')['parameters'], params(env, 'split')['parameters']
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    # a changed rank is a resume mismatch
    controller_run(env, 'cut', controller={**low}, stop_after_step=3)
    with pytest.raises(ValueError, match='Resume mismatch'):
        controller_run(env, 'cut', controller={**low, 'block_rank': 1}, resume=True)


def test_per_run_override_changes_ranks_and_lr(env):
    final = controller_run(env, 'ov', controller={'kind': 'lowrank', 'block_rank': 2,
                                                  'before_rank': 3, 'after_rank': 2},
                           run_controller={'block_rank': 1, 'after_rank': 4})
    assert final['trainable']['elements'] == expected_trainable_elements(
        {'block_rank': 1, 'before_rank': 3, 'after_rank': 4})
    meta = json.loads((env.tmp / 'ov' / 'checkpoints' / 'step_000006' / 'meta.json').read_text())
    assert (meta['block_rank'], meta['after_rank'], meta['before_rank']) == (1, 4, 3)


def test_default_full_path_is_unchanged(env):
    plain = controller_run(env, 'plain')
    explicit = controller_run(env, 'explicit', controller={'kind': 'full'})
    for name, final in (('plain', plain), ('explicit', explicit)):
        assert exports(env, name) == ['controller.safetensors', 'meta.json']
        assert 'controller_lowrank' not in final and final['export_file'] == 'controller.safetensors'
    a, b = params(env, 'plain'), params(env, 'explicit')
    assert a['parameters'].keys() == b['parameters'].keys()
    assert all(torch.equal(a['parameters'][k], b['parameters'][k]) for k in a['parameters'])
    assert a['compatibility'] == b['compatibility'] and 'controller_spec' not in a['compatibility']
    assert 'controller_lowrank' not in json.loads((env.tmp / 'plain' / 'resolved.json').read_text())
    assert plain['trainable']['elements'] == sum(  # every controller parameter trains, as before
        p.numel() for p in tiny.tiny_model()[1].parameters())


# ----------------------------------------------------------- recipe hash ----

def lowrank_recipe(tmp_path, spec, **write):
    resolved = write_recipe(tmp_path)
    recipe = json.loads((tmp_path / 'recipe.json').read_text())
    directory = tmp_path / 'runs' / 'ctl' / 'checkpoints' / 'step_000007'
    controller = tiny.tiny_model()[1]
    lrc.convert(controller, {**spec, 'init_seed': 1})
    lrc.save(controller, directory / lrc.FILE_NAMES['lowrank'], revision='rev1')
    recipe['controller'] = {**recipe['controller'], **{k: spec[k] for k in ('kind', 'block_rank',
                                                       'before_rank', 'after_rank')}, **write}
    (tmp_path / 'recipe.json').write_text(json.dumps(recipe))
    return resolved, bulk.load_recipe(tmp_path / 'recipe.json', root=tmp_path)


def test_recipe_hash_unchanged_for_full_and_distinct_for_lowrank(tmp_path):
    base = write_recipe(tmp_path)
    recipe = json.loads((tmp_path / 'recipe.json').read_text())
    recipe['controller']['kind'] = 'full'
    (tmp_path / 'recipe.json').write_text(json.dumps(recipe))
    assert bulk.load_recipe(tmp_path / 'recipe.json', root=tmp_path)['recipe_hash'] == base['recipe_hash']
    recipe['controller']['block_rank'] = 4
    (tmp_path / 'recipe.json').write_text(json.dumps(recipe))
    with pytest.raises(bulk.BulkError, match='rank'):
        bulk.load_recipe(tmp_path / 'recipe.json', root=tmp_path)
    spec = {'kind': 'lowrank', 'block_rank': 2, 'before_rank': 3, 'after_rank': 2}
    _, low = lowrank_recipe(tmp_path, spec)
    assert low['recipe_hash'] != base['recipe_hash'] and low['defining']['controller_spec'] == spec
    assert low['files']['controller']['path'].endswith('controller_lowrank.safetensors')
    assert 'controller_spec' not in base['defining']
    other = tmp_path / 'other'
    other.mkdir()
    _, low2 = lowrank_recipe(other, {**spec, 'after_rank': 4})
    assert low2['recipe_hash'] not in (low['recipe_hash'], base['recipe_hash'])
    # a recipe whose ranks disagree with the file's own metadata is refused
    with pytest.raises(bulk.BulkError, match='ranks differ'):
        lowrank_recipe(tmp_path / 'other', spec, block_rank=5)
    with pytest.raises(bulk.BulkError, match='positive int'):
        lowrank_recipe(tmp_path / 'other', spec, kind='lowrank', after_rank=None)


@pytest.mark.skipif(not (ROOT / 'outputs/p5-diagnostics/recipe_v3_production.json').exists(),
                    reason='campaign outputs not present')
def test_production_recipe_hash_is_still_the_campaign_one():
    resolved = bulk.load_recipe(ROOT / 'outputs/p5-diagnostics/recipe_v3_production.json', root=ROOT)
    assert resolved['recipe_hash'] == 'c80ff98494cdb36d'
    assert 'controller_spec' not in resolved['defining']


def test_recipe_builder_records_lowrank_kind_only_for_lowrank_files(tmp_path):
    controller = tiny.tiny_model()[1]
    lrc.convert(controller, SPEC)
    low = tmp_path / 'runs' / 'c' / 'checkpoints' / 'step_000010' / lrc.FILE_NAMES['lowrank']
    lora = tmp_path / 'runs' / 't' / 'checkpoints' / 'step_000010' / 'lora_target.safetensors'
    lrc.save(controller, low, revision='r')
    lora.parent.mkdir(parents=True)
    lora.write_bytes(b'lora')
    build = lambda path: recipe_module.build_recipe(  # noqa: E731
        'v', target_lora=lora, controller=path, lora_scale=1.0, guidance=1.0, control_scale=1.0,
        generator_config=ROOT / 'configs' / 'generator.yaml', evidence={}, caveats=[])
    assert build(low)['controller']['kind'] == 'lowrank'
    assert build(low)['controller']['block_rank'] == 2
    full = tmp_path / 'runs' / 'f' / 'checkpoints' / 'step_000010' / 'controller.safetensors'
    full.parent.mkdir(parents=True)
    full.write_bytes(b'full')
    assert 'kind' not in build(full)['controller']


def test_load_weights_dispatches_on_file_kind(tmp_path):
    """p4_eval.load_weights (used by p4_eval, p5_matrix and bulk) loads either controller kind."""
    from types import SimpleNamespace

    from generative_sim2real.generation import p4_eval

    source = tiny.tiny_model()[1]
    lrc.convert(source, SPEC)
    low = tmp_path / lrc.FILE_NAMES['lowrank']
    lrc.save(source, low, revision='rev')
    full = tmp_path / 'controller.safetensors'
    gm.save_controller(tiny.tiny_model()[1], full, revision='rev')
    generator = {'lora': {'rank': 2, 'alpha': 2}, 'revision': 'rev'}
    model, controller, _ = tiny.tiny_model()
    ctx = SimpleNamespace(controller=controller, transformer=model.transformer)
    loaded = p4_eval.load_weights(ctx, None, generator, controller=low)
    assert loaded['controller']['kind'] == 'lowrank' and loaded['controller']['block_rank'] == 2
    assert lrc.is_lowrank(controller)
    ctx = SimpleNamespace(controller=tiny.tiny_model()[1], transformer=model.transformer)
    assert 'kind' not in p4_eval.load_weights(ctx, None, generator, controller=full)['controller']
