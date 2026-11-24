import pytest
import torch

pytest.importorskip('diffusers')
pytest.importorskip('peft')
import generation_tiny as tiny  # noqa: E402

from generative_sim2real.generation import model as gm  # noqa: E402

EMBED = tiny.tiny_prompt()


def loss_for(model, control, seed=0):
    latents, ctrl = tiny.batch()
    generator = torch.Generator().manual_seed(seed)
    return gm.diffusion_loss(
        model, tiny.tiny_scheduler(), latents, EMBED['positive'].expand(2, -1, -1),
        EMBED['positive_mask'].expand(2, -1), generator=generator,
        control=ctrl if control else None)


def test_controller_is_clean_base_copy_with_zero_projections():
    model, controller, provenance = tiny.tiny_model()
    assert provenance['base_blocks_copied'] == [0, 1]
    assert all(provenance['copied_block_hashes_equal_base'])
    assert provenance['projections_zero_at_init']
    # Zero projections: an untrained controller must not alter the backbone.
    latents, ctrl = tiny.batch()
    kwargs = dict(encoder_hidden_states=EMBED['positive'].expand(2, -1, -1),
                  encoder_attention_mask=EMBED['positive_mask'].expand(2, -1),
                  timestep=torch.tensor([5, 500]), return_dict=False,
                  added_cond_kwargs={'resolution': None, 'aspect_ratio': None})
    with torch.no_grad():
        on = model(latents, controlnet_cond=ctrl, **kwargs)[0]
        off = model(latents, **kwargs)[0]
    assert torch.equal(on, off)


def test_controller_refuses_adapter_contaminated_base():
    from generative_sim2real.generation.controlnet import PixArtControlNetAdapterModel

    model, _, _ = tiny.tiny_model()
    with pytest.raises(ValueError, match='clean base'):
        PixArtControlNetAdapterModel.from_transformer(model.transformer, 2)


def test_adapters_independent_and_counts_match_formula():
    model, controller, _ = tiny.tiny_model()
    transformer = model.transformer
    expected, modules = gm.expected_lora_elements(transformer, rank=2)
    assert modules == 4 * 2 * 4  # layers x (self, cross attention) x (q, k, v, out)
    for adapter in gm.ADAPTERS:
        named = gm.set_stage(transformer, controller, stage='lora', adapter=adapter)
        assert len(named) == 2 * modules
        assert sum(p.numel() for _, p in named) == expected
        assert all(f'.{adapter}.' in n and 'lora_' in n for n, _ in named)
        assert all(p.dtype == torch.float32 for _, p in named)
    # Distinct seeds: the two adapters do not share their random A matrices.
    a = dict(transformer.named_parameters())
    key = 'transformer_blocks.0.attn1.to_q.lora_A.%s.weight'
    assert not torch.equal(a[key % 'source'], a[key % 'target'])
    assert not any(p.any() for n, p in a.items() if 'lora_B' in n)


def test_lora_stage_changes_only_active_adapter():
    model, controller, _ = tiny.tiny_model()
    transformer = model.transformer
    base = gm.state_hash(transformer, exclude_lora=True)
    target_before = gm.state_hash(transformer, only_lora='target')
    controller_before = gm.state_hash(controller)
    named = gm.set_stage(transformer, controller, stage='lora', adapter='source')
    before = gm.snapshot(named)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        loss_for(model, control=False).backward()
        optimizer.step()
    changed = gm.changed_names(before, named)
    assert len(changed) == len(named)  # B moves first, then A: all change by step 3
    assert gm.state_hash(transformer, exclude_lora=True) == base
    assert gm.state_hash(transformer, only_lora='target') == target_before
    assert gm.state_hash(controller) == controller_before


def test_controller_stage_gradient_path_through_frozen_backbone():
    model, controller, _ = tiny.tiny_model()
    transformer = model.transformer
    # Give the source adapter weight first so "frozen source LoRA" is nontrivial.
    named = gm.set_stage(transformer, controller, stage='lora', adapter='source')
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-2)
    for _ in range(2):
        optimizer.zero_grad()
        loss_for(model, control=False).backward()
        optimizer.step()
    frozen = (gm.state_hash(transformer, exclude_lora=True),
              gm.state_hash(transformer, only_lora='source'),
              gm.state_hash(transformer, only_lora='target'))
    named = gm.set_stage(transformer, controller, stage='controller', adapter='source')
    assert named and all(n.startswith('controller.') for n, _ in named)
    assert len(named) == len(list(controller.parameters()))
    before = gm.snapshot(named)
    optimizer = torch.optim.AdamW([p for _, p in named], lr=1e-2)
    ever = {n: False for n, _ in named}
    first_nonzero = None
    for update in range(4):
        optimizer.zero_grad()
        loss_for(model, control=True).backward()
        for n, p in named:
            ever[n] |= bool(p.grad is not None and p.grad.abs().sum() > 0)
        if update == 0:  # zero-initialised projections: only after_proj sees gradient
            first_nonzero = {n for n, p in named if p.grad is not None and p.grad.abs().sum() > 0}
        optimizer.step()
    assert first_nonzero and all('after_proj' in n for n in first_nonzero)
    assert all(ever.values())
    assert len(gm.changed_names(before, named)) == len(named)
    assert frozen == (gm.state_hash(transformer, exclude_lora=True),
                      gm.state_hash(transformer, only_lora='source'),
                      gm.state_hash(transformer, only_lora='target'))
    assert all(p.grad is None for n, p in transformer.named_parameters())


def test_controller_stage_rejects_target_adapter():
    model, controller, _ = tiny.tiny_model()
    with pytest.raises(ValueError, match='source-only'):
        gm.set_stage(model.transformer, controller, stage='controller', adapter='target')


def trained_state(model, controller):
    """Nonzero source/target adapters and controller so swaps are observable."""
    for stage, adapter, use_control in (('lora', 'source', False), ('lora', 'target', False),
                                        ('controller', 'source', True)):
        named = gm.set_stage(model.transformer, controller, stage=stage, adapter=adapter)
        optimizer = torch.optim.AdamW([p for _, p in named], lr=2e-2)
        for step in range(4):
            optimizer.zero_grad()
            loss_for(model, control=use_control, seed=step).backward()
            optimizer.step()
    gm.set_stage(model.transformer, controller, stage='inference', adapter='source')


def render(model, vae, seed=7, adapter='source', control=True, scale=1.0, ctrl=None):
    gm.set_stage(model.transformer, model.controlnet, stage='inference', adapter=adapter)
    if ctrl is None:
        ctrl = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(5))
    return gm.generate(model, vae, tiny.tiny_scheduler(), EMBED, seeds=[seed], steps=4,
                       guidance=3.0, control=ctrl if control else None,
                       control_scale=scale, size=64)


def test_swap_recovers_source_and_leaves_controller_untouched(tmp_path):
    model, controller, _ = tiny.tiny_model()
    vae = tiny.tiny_vae()
    trained_state(model, controller)
    controller_hash = gm.state_hash(controller)
    base_hash = gm.state_hash(model.transformer, exclude_lora=True)
    source = render(model, vae)
    target = render(model, vae, adapter='target')
    recovered = render(model, vae)
    assert (source - target).abs().mean() > 1e-4  # the swap is observable
    assert torch.allclose(source, recovered, atol=1e-5, rtol=1e-5)
    assert gm.state_hash(controller) == controller_hash
    assert gm.state_hash(model.transformer, exclude_lora=True) == base_hash
    # Controller on vs off differs after training: the control path is live.
    assert (render(model, vae) - render(model, vae, control=False)).abs().mean() > 1e-5


def test_separate_files_reload_into_fresh_model(tmp_path):
    model, controller, _ = tiny.tiny_model()
    vae = tiny.tiny_vae()
    trained_state(model, controller)
    reference = {a: render(model, vae, adapter=a) for a in gm.ADAPTERS}
    for adapter in gm.ADAPTERS:
        gm.save_adapter(model.transformer, tmp_path / f'{adapter}.safetensors', adapter,
                        rank=2, alpha=2, revision='r')
    gm.save_controller(controller, tmp_path / 'controller.safetensors', revision='r')
    fresh, fresh_controller, _ = tiny.tiny_model()
    for adapter in gm.ADAPTERS:
        assert gm.load_adapter(fresh.transformer, tmp_path / f'{adapter}.safetensors',
                               adapter, rank=2, alpha=2, revision='r')['adapter'] == adapter
    gm.load_controller(fresh_controller, tmp_path / 'controller.safetensors', revision='r')
    for adapter in gm.ADAPTERS:
        assert torch.allclose(reference[adapter], render(fresh, vae, adapter=adapter),
                              atol=1e-5, rtol=1e-5)
    # An adapter file never contains controller tensors and vice versa.
    from safetensors.torch import load_file
    assert not any('controlnet' in k or 'after_proj' in k
                   for k in load_file(str(tmp_path / 'source.safetensors')))
    assert not any('lora_' in k for k in load_file(str(tmp_path / 'controller.safetensors')))
    with pytest.raises(ValueError, match='controller file'):
        gm.load_controller(fresh_controller, tmp_path / 'source.safetensors', revision='r')
    with pytest.raises(ValueError, match='LoRA file'):
        gm.load_adapter(fresh.transformer, tmp_path / 'controller.safetensors', 'source',
                        rank=2, alpha=2, revision='r')


def test_controller_off_ignores_nominal_depth_and_fixed_seed_repeats():
    model, controller, _ = tiny.tiny_model()
    vae = tiny.tiny_vae()
    trained_state(model, controller)
    a = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    b = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(2))
    assert torch.equal(render(model, vae, scale=0.0, ctrl=a), render(model, vae, scale=0.0, ctrl=b))
    assert torch.equal(render(model, vae, control=False), render(model, vae, scale=0.0, ctrl=a))
    assert torch.equal(render(model, vae, ctrl=a), render(model, vae, ctrl=a))
    assert not torch.equal(render(model, vae, ctrl=a), render(model, vae, ctrl=b))
    assert not torch.equal(render(model, vae, seed=1), render(model, vae, seed=2))


def test_batched_noise_does_not_depend_on_batch_mates():
    model, controller, _ = tiny.tiny_model()
    vae = tiny.tiny_vae()
    ctrl = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(9))
    scheduler = tiny.tiny_scheduler()
    kwargs = dict(steps=4, guidance=3.0, size=64)
    pair = gm.generate(model, vae, scheduler, EMBED, seeds=[3, 4], control=ctrl, **kwargs)
    single = gm.generate(model, vae, scheduler, EMBED, seeds=[3], control=ctrl[:1], **kwargs)
    assert torch.allclose(pair[:1], single, atol=1e-4)


def test_v_prediction_and_unknown_prediction_types():
    model, _, _ = tiny.tiny_model()
    scheduler = tiny.tiny_scheduler()
    latents, _ = tiny.batch()
    args = (model, scheduler, latents, EMBED['positive'].expand(2, -1, -1),
            EMBED['positive_mask'].expand(2, -1))
    scheduler.register_to_config(prediction_type='v_prediction')
    assert torch.isfinite(gm.diffusion_loss(*args, generator=torch.Generator().manual_seed(0)))
    scheduler.register_to_config(prediction_type='sample')
    with pytest.raises(ValueError, match='prediction_type'):
        gm.diffusion_loss(*args, generator=torch.Generator().manual_seed(0))


def test_training_noise_is_independent_of_inference_scheduler_state():
    model, _, _ = tiny.tiny_model()
    latents, _ = tiny.batch()
    args = (model, None, latents, EMBED['positive'].expand(2, -1, -1),
            EMBED['positive_mask'].expand(2, -1))
    fresh = tiny.tiny_scheduler()
    used = tiny.tiny_scheduler()
    used.set_timesteps(4)  # what a validation render does
    with torch.no_grad():
        a = gm.diffusion_loss(args[0], fresh, *args[2:], generator=torch.Generator().manual_seed(0))
        b = gm.diffusion_loss(args[0], used, *args[2:], generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
    # And it matches DDPMScheduler.add_noise, which upstream trainers use for the same schedule.
    from diffusers import DDPMScheduler

    noise = torch.randn(latents.shape, generator=torch.Generator().manual_seed(1))
    steps = torch.tensor([10, 900])
    reference = DDPMScheduler.from_config(fresh.config).add_noise(latents, noise, steps)
    ac = fresh.alphas_cumprod[steps].view(-1, 1, 1, 1)
    assert torch.allclose(reference, ac.sqrt() * latents + (1 - ac).sqrt() * noise, atol=1e-5)


def test_load_validates_recorded_metadata(tmp_path):
    model, controller, _ = tiny.tiny_model()
    gm.save_adapter(model.transformer, tmp_path / 'a.safetensors', 'source', rank=2, alpha=2,
                    revision='r')
    gm.save_controller(controller, tmp_path / 'c.safetensors', revision='r')
    ok = dict(rank=2, alpha=2, revision='r')
    gm.load_adapter(model.transformer, tmp_path / 'a.safetensors', 'source', **ok)
    for bad in (dict(ok, rank=4), dict(ok, alpha=8), dict(ok, revision='other')):
        with pytest.raises(ValueError, match='metadata mismatch'):
            gm.load_adapter(model.transformer, tmp_path / 'a.safetensors', 'source', **bad)
    with pytest.raises(ValueError, match='metadata mismatch for adapter'):
        gm.load_adapter(model.transformer, tmp_path / 'a.safetensors', 'target', **ok)
    gm.load_controller(controller, tmp_path / 'c.safetensors', revision='r')
    with pytest.raises(ValueError, match='base_revision'):
        gm.load_controller(controller, tmp_path / 'c.safetensors', revision='other')
    shallow, shallow_controller, _ = tiny.tiny_model(controller_layers=1)
    with pytest.raises(ValueError, match='num_layers'):
        gm.load_controller(shallow_controller, tmp_path / 'c.safetensors', revision='r')


def test_control_scale_is_per_call_not_model_state():
    model, controller, _ = tiny.tiny_model()
    vae = tiny.tiny_vae()
    trained_state(model, controller)
    assert not hasattr(model, 'conditioning_scale')
    before = loss_for(model, control=True, seed=11).item()
    render(model, vae, scale=0.0)  # an eval render at scale 0 must not leak into training
    render(model, vae, scale=0.5)
    gm.set_stage(model.transformer, controller, stage='controller', adapter='source')
    assert loss_for(model, control=True, seed=11).item() == before


def test_perturbing_target_lora_leaves_source_render_unchanged():
    model, controller, _ = tiny.tiny_model()
    vae = tiny.tiny_vae()
    trained_state(model, controller)
    source = render(model, vae)
    target = render(model, vae, adapter='target')
    with torch.no_grad():
        for name, p in model.transformer.named_parameters():
            if 'lora_B' in name and '.target.' in name:
                p.add_(0.5)
    assert torch.equal(render(model, vae), source)
    assert not torch.equal(render(model, vae, adapter='target'), target)
    with torch.no_grad():
        for name, p in model.transformer.named_parameters():
            if 'lora_B' in name and '.source.' in name:
                p.add_(0.5)
    assert not torch.equal(render(model, vae), source)


def test_adapter_init_seeds_must_be_distinct_and_are_honoured():
    from generative_sim2real.generation import model as m

    with pytest.raises(ValueError, match='distinct'):
        m.assemble(tiny.tiny_transformer(), controller_layers=2, rank=2, alpha=2,
                   seeds={'source': 1, 'target': 1})
    a, _, _ = m.assemble(tiny.tiny_transformer(), controller_layers=2, rank=2, alpha=2,
                         seeds={'source': 5, 'target': 6})
    b, _, _ = m.assemble(tiny.tiny_transformer(), controller_layers=2, rank=2, alpha=2,
                         seeds={'source': 6, 'target': 5})
    key = 'transformer_blocks.0.attn1.to_q.lora_A.%s.weight'
    pa, pb = dict(a.transformer.named_parameters()), dict(b.transformer.named_parameters())
    assert torch.equal(pa[key % 'source'], pb[key % 'target'])
    assert torch.equal(pa[key % 'target'], pb[key % 'source'])
