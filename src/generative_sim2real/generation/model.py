"""PixArt LoRA/controller assembly and diffusion operations, derived from upstream examples.

The controller follows Diffusers' pinned research implementation; appearance LoRA
uses PEFT attention projections as a subset of the official PixArt LoRA recipe.
Precision policy: the frozen backbone is stored in `dtype` (bfloat16 on GPU); LoRA
and controller parameters stay float32 masters; every forward runs under autocast.
"""
import contextlib
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

LORA_MODULES = ['to_q', 'to_k', 'to_v', 'to_out.0']
ADAPTERS = ('source', 'target')


def compute_context(device, dtype):
    """One autocast policy for training and inference; a no-op for float32 fixtures."""
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def tensors_hash(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_hash(module, *, exclude_lora=False, only_lora=None):
    """Hash a state dict; PEFT's `.base_layer.` wrapper does not change base contents.

    `only_lora` selects the tensors of one named adapter (e.g. 'source')."""
    items = []
    for name, tensor in module.state_dict().items():
        is_lora = 'lora_' in name
        if exclude_lora and is_lora:
            continue
        if only_lora is not None and not (is_lora and f'.{only_lora}.' in name):
            continue
        items.append((name.replace('.base_layer.', '.'), tensor))
    return tensors_hash(items)


def add_appearance_adapters(transformer, *, rank=16, alpha=16, seeds=None, dtype=torch.float32):
    """Attach independent source/target LoRAs to one identical clean base.

    The adapters are created in float32 before the frozen base is cast to `dtype`;
    each adapter draws from its own named seed (`seeds`: adapter -> int), so neither
    inherits the other's weights."""
    from peft import LoraConfig, inject_adapter_in_model

    transformer.requires_grad_(False)
    seeds = seeds or {'source': 42, 'target': 43}
    if set(seeds) != set(ADAPTERS) or seeds['source'] == seeds['target']:
        raise ValueError('Provide distinct init seeds for the source and target adapters')
    for name in ADAPTERS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seeds[name])
            # PixArtTransformer2DModel has no diffusers PeftAdapterMixin in 0.35.2.
            inject_adapter_in_model(LoraConfig(
                r=rank, lora_alpha=alpha, init_lora_weights='gaussian',
                target_modules=LORA_MODULES), transformer, adapter_name=name)
    for name, parameter in transformer.named_parameters():
        parameter.data = parameter.data.to(torch.float32 if 'lora_' in name else dtype)
    transformer.requires_grad_(False)
    set_stage(transformer, None, stage='inference', adapter='source')


def assemble(transformer, *, controller_layers=13, rank=16, alpha=16, seeds=None,
             dtype=torch.float32):
    """Build the controller from the clean fp32 base, then attach both adapters.

    Returns (combined_model, controller, provenance). `transformer` must be clean."""
    from .controlnet import (PixArtControlNetAdapterModel,
                             PixArtControlNetTransformerModel)

    controller = PixArtControlNetAdapterModel.from_transformer(transformer, controller_layers)
    copied = []
    for i, block in enumerate(controller.controlnet_blocks):
        source = tensors_hash(transformer.transformer_blocks[i].state_dict().items())
        copied.append(source == tensors_hash(block.transformer_block.state_dict().items()))
    zero_projections = all(
        not p.any() for n, p in controller.named_parameters()
        if n.split('.')[-2] in ('before_proj', 'after_proj'))
    provenance = {
        'base_blocks_copied': list(range(controller_layers)),
        'copied_block_hashes_equal_base': copied,
        'projections_zero_at_init': bool(zero_projections),
        'base_transformer_sha256_fp32': state_hash(transformer),
        'controller_init_sha256': state_hash(controller),
    }
    add_appearance_adapters(transformer, rank=rank, alpha=alpha, seeds=seeds, dtype=dtype)
    model = PixArtControlNetTransformerModel(transformer, controller)
    model.requires_grad_(False)
    return model, controller, provenance


def activate_adapter(transformer, adapter):
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [m for m in transformer.modules() if isinstance(m, BaseTunerLayer)]
    if not layers:
        raise ValueError('No adapter layers are attached')
    for layer in layers:
        layer.set_adapter([adapter])


def active_adapters(transformer):
    """Active adapter names; all adapter layers must agree."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    active = {tuple(m.active_adapters) for m in transformer.modules()
              if isinstance(m, BaseTunerLayer)}
    if len(active) != 1:
        raise RuntimeError(f'Adapter layers disagree about the active adapter: {active}')
    return list(active.pop())


def set_stage(transformer, controller, *, stage, adapter):
    """Select the active appearance adapter and exactly the trainable group.

    lora: only the named adapter. controller: only the controller (source adapter,
    frozen but still differentiable). inference: nothing trains."""
    if adapter not in ADAPTERS:
        raise ValueError('Appearance must be source or target')
    if stage not in {'lora', 'controller', 'inference'}:
        raise ValueError('Unknown training stage')
    if stage == 'controller' and adapter != 'source':
        raise ValueError('Controller training is source-only')
    activate_adapter(transformer, adapter)
    transformer.zero_grad(set_to_none=True)  # no stale gradients from another stage
    if controller is not None:
        controller.zero_grad(set_to_none=True)
    transformer.requires_grad_(False)  # PEFT enables grads on the active adapter
    transformer.eval()
    if controller is not None:
        controller.requires_grad_(stage == 'controller')
        controller.train(stage == 'controller')
    if stage == 'lora':
        transformer.train()
        for name, parameter in transformer.named_parameters():
            if 'lora_' in name and f'.{adapter}.' in name:
                parameter.requires_grad_(True)
    return trainable_parameters(transformer, controller)


def trainable_parameters(transformer, controller=None):
    named = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
    if controller is not None:
        named += [(f'controller.{n}', p) for n, p in controller.named_parameters()
                  if p.requires_grad]
    return named


def parameter_audit(named):
    """Names, tensor counts and element counts of a trainable set (stage evidence)."""
    return {'tensors': len(named), 'elements': sum(p.numel() for _, p in named),
            'dtypes': sorted({str(p.dtype) for _, p in named}),
            'first_names': [n for n, _ in named[:2]], 'last_names': [n for n, _ in named[-2:]]}


def expected_lora_elements(transformer, rank):
    """Independent count: r * (in + out) for every injected target Linear."""
    total, modules = 0, 0
    for name, module in transformer.named_modules():
        if hasattr(module, 'lora_A') and 'source' in module.lora_A:
            base = module.base_layer
            total += rank * (base.in_features + base.out_features)
            modules += 1
    return total, modules


def snapshot(named):
    return {n: p.detach().clone().cpu() for n, p in named}


def changed_names(before, named, *, atol=0.0):
    """Names whose tensors differ from a prior `snapshot` (actual weight change)."""
    current = {n: p.detach().cpu() for n, p in named}
    return [n for n, old in before.items() if (current[n] - old).abs().max() > atol]


def _adapter_metadata(name, rank, alpha, revision):
    return {'format': 'generative_sim2real_lora_v1', 'adapter': name, 'rank': str(rank),
            'alpha': str(alpha), 'target_modules': ','.join(LORA_MODULES),
            'base_revision': revision}


def adapter_state(transformer, name):
    """One adapter's tensors keyed in PEFT's file convention (adapter name removed)."""
    return {key.replace(f'.{name}.', '.'): tensor.detach().cpu().contiguous()
            for key, tensor in transformer.state_dict().items()
            if 'lora_' in key and f'.{name}.' in key}


def save_adapter(transformer, path, name, *, rank=16, alpha=16, revision=''):
    from safetensors.torch import save_file

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(adapter_state(transformer, name), str(path),
              metadata=_adapter_metadata(name, rank, alpha, revision))


def load_adapter(transformer, path, name, *, rank, alpha, revision):
    """Load one adapter file; its recorded rank/alpha/modules/revision must match the model."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework='pt') as handle:
        metadata = handle.metadata() or {}
    if metadata.get('format') != 'generative_sim2real_lora_v1':
        raise ValueError('Not an appearance LoRA file written by this project')
    wanted = _adapter_metadata(name, rank, alpha, revision)
    for key, value in wanted.items():
        if metadata.get(key) != value:
            raise ValueError(f'Adapter metadata mismatch for {key}: file has '
                             f'{metadata.get(key)!r}, model expects {value!r}')
    state = load_file(str(path))
    expected = adapter_state(transformer, name)
    if state.keys() != expected.keys():
        raise ValueError('Adapter keys do not match the configured backbone module set')
    if any(state[k].shape != expected[k].shape for k in state):
        raise ValueError('Adapter tensor shapes do not match the configured rank')
    parameters = dict(transformer.named_parameters())
    with torch.no_grad():
        for key, tensor in state.items():
            head, tail = key.rsplit('.', 1)  # '...lora_A' + 'weight'
            target = parameters[f'{head}.{name}.{tail}']
            target.copy_(tensor.to(target.dtype))
    return metadata


def save_controller(controller, path, *, revision=''):
    from safetensors.torch import save_file

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({n: p.detach().cpu().contiguous() for n, p in controller.state_dict().items()},
              str(path), metadata={'format': 'generative_sim2real_controller_v1',
                                   'num_layers': str(controller.num_layers),
                                   'base_revision': revision})


def load_controller(controller, path, *, revision):
    """Load controller weights; recorded depth and base revision must match the model."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework='pt') as handle:
        metadata = handle.metadata() or {}
    if metadata.get('format') != 'generative_sim2real_controller_v1':
        raise ValueError('Not a controller file written by this project')
    if metadata.get('num_layers') != str(controller.num_layers):
        raise ValueError('Controller metadata mismatch for num_layers')
    if metadata.get('base_revision') != revision:
        raise ValueError('Controller metadata mismatch for base_revision')
    controller.load_state_dict(load_file(str(path)), strict=True)
    return metadata


def noise_prediction(output, channels):
    # PixArt predicts learned sigma in its second half. Upstream LoRA/control
    # recipes train only denoising MSE and the DPM solver consumes that first half.
    if output.shape[1] == 2 * channels:
        return output.chunk(2, dim=1)[0]
    if output.shape[1] != channels:
        raise ValueError('Unexpected PixArt prediction channel count')
    return output


def diffusion_loss(model, scheduler, latents, embeddings, mask, *, generator, control=None,
                   control_scale=1.0):
    """Denoising MSE on the first (noise) half of the output.

    Never wrap the frozen backbone in no_grad: controller gradients flow through it."""
    noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device,
                        generator=generator)
    timesteps = torch.randint(scheduler.config.num_train_timesteps,
                              (latents.shape[0],), device=latents.device, generator=generator)
    # Do NOT call add_noise on a DPMSolverMultistepScheduler: it indexes the *inference*
    # timesteps (misaligned even when fresh, and changed by any set_timesteps() call).
    # This state-free DDPM forward process equals DDPMScheduler.add_noise, which the
    # upstream trainers use; training should pass a DDPMScheduler built from the same config.
    cumulative = scheduler.alphas_cumprod.to(latents.device)[timesteps].float()
    cumulative = cumulative.view(-1, *([1] * (latents.ndim - 1)))
    noisy = cumulative.sqrt() * latents + (1 - cumulative).sqrt() * noise
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        target = noise
    elif kind == 'v_prediction':
        target = cumulative.sqrt() * noise - (1 - cumulative).sqrt() * latents
    else:
        raise ValueError(f'Unsupported scheduler prediction_type: {kind}')
    kwargs = dict(encoder_hidden_states=embeddings, encoder_attention_mask=mask,
                  timestep=timesteps, added_cond_kwargs={'resolution': None, 'aspect_ratio': None},
                  return_dict=False)
    if control is not None:
        kwargs.update(controlnet_cond=control, conditioning_scale=control_scale)
    prediction = noise_prediction(model(noisy, **kwargs)[0], latents.shape[1])
    return F.mse_loss(prediction.float(), target.float())


@torch.no_grad()
def generate(model, vae, scheduler, prompt, *, seeds, steps=20, guidance=4.5, control=None,
             control_scale=1.0, size=512, dtype=torch.float32):
    """Fixed-seed batched inference from cached prompt and deterministic control latents.

    Each sample draws its initial noise from its own seed, so a sample's noise does not
    depend on the batch it is generated in. `control` is a [B,4,h,w] control latent."""
    if steps < 2 or size % 8:
        raise ValueError('Use at least two DPM steps and size divisible by eight')
    device = next(model.parameters()).device
    batch = len(seeds)
    noise = [torch.randn((4, size // 8, size // 8), device=device, dtype=torch.float32,
                         generator=torch.Generator(device=device).manual_seed(int(seed)))
             for seed in seeds]
    latents = torch.stack(noise)
    scheduler.set_timesteps(steps, device=device)
    latents = latents * scheduler.init_noise_sigma
    embeds = torch.cat([prompt['negative'].expand(batch, -1, -1),
                        prompt['positive'].expand(batch, -1, -1)]).to(device)
    mask = torch.cat([prompt['negative_mask'].expand(batch, -1),
                      prompt['positive_mask'].expand(batch, -1)]).to(device)
    condition = None
    if control is not None:
        if control.shape[0] != batch:
            raise ValueError('One control latent per seed is required')
        condition = torch.cat([control, control]).to(device)
    model.eval()
    for step in scheduler.timesteps:
        inputs = scheduler.scale_model_input(torch.cat([latents, latents]), step)
        with compute_context(device, dtype):
            prediction = model(
                inputs, encoder_hidden_states=embeds, encoder_attention_mask=mask,
                timestep=step.expand(2 * batch), controlnet_cond=condition,
                conditioning_scale=control_scale,
                added_cond_kwargs={'resolution': None, 'aspect_ratio': None},
                return_dict=False)[0]
        negative, positive = prediction.float().chunk(2)
        prediction = noise_prediction(negative + guidance * (positive - negative), 4)
        latents = scheduler.step(prediction, step, latents, return_dict=False)[0]
    image = vae.decode((latents / vae.config.scaling_factor).to(vae.dtype), return_dict=False)[0]
    return (image.float() / 2 + .5).clamp(0, 1)
