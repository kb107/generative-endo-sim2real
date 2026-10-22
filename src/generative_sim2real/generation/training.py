"""One update loop for the LoRA and controller stages (smoke, profile, P4 training).

The stage is chosen beforehand with model.set_stage; this loop trains exactly the
parameters it is given, with gradient accumulation to the effective batch.
A resumable caller (generation/train.py) passes its own optimizer, scheduler and noise
generator so their state can be checkpointed and restored between calls."""
import math
import time

import torch

from .inputs import encode_image
from .model import active_adapters, compute_context, diffusion_loss


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def make_optimizer(named, *, lr, weight_decay, warmup=0):
    """AdamW plus the linear-warmup schedule (0 = constant LR) used by every stage."""
    optimizer = torch.optim.AdamW([p for _, p in named], lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) if warmup else 1.0)
    return optimizer, schedule


def noise_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def train_updates(model, vae, scheduler, prompt, named, batches, *, updates, accumulation,
                  lr, weight_decay, max_grad_norm, use_control, seed=None, device, dtype,
                  probe_gradients=False, warmup=0, callback=None, optimizer=None,
                  lr_schedule=None, generator=None, first_update=0):
    """Run `updates` optimizer steps; return per-update records and gradient evidence.

    Resumable use: pass `optimizer`/`lr_schedule` (restored from a checkpoint), the noise
    `generator` (state restored) and `first_update` (completed updates), so numbering and the
    callback's update number continue; omit them for a fresh run seeded by `seed`.

    `warmup` is a linear LR warmup in optimizer steps (0 = constant LR, the smoke default).
    `callback(update_number, record)` runs after each optimizer step (eval/checkpoints);
    it must leave the same parameters trainable and the same adapter active, else
    RuntimeError (a render helper that switches to inference mode must restore the stage).
    `batches` yields dicts with 'rgb' [B,3,H,W] and, if use_control, 'control'."""
    parameters = [p for _, p in named]
    if not parameters or any(not p.requires_grad for p in parameters):
        raise ValueError('Pass exactly the trainable parameters of the current stage')
    if optimizer is None:
        optimizer, schedule = make_optimizer(named, lr=lr, weight_decay=weight_decay,
                                             warmup=warmup)
    else:
        schedule = lr_schedule
        if schedule is None or {id(p) for g in optimizer.param_groups for p in g['params']} != {
                id(p) for p in parameters}:
            raise ValueError('The optimizer must cover exactly the trainable parameters')
    if generator is None:
        if seed is None:
            raise ValueError('A noise seed or generator is required')
        generator = noise_generator(device, seed)
    embeds, mask = prompt['positive'].to(device), prompt['positive_mask'].to(device)
    expected_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    if expected_trainable != {id(p) for p in parameters}:
        raise ValueError('The trainable set of the model differs from the parameters passed')
    expected_adapters = active_adapters(model.transformer)
    ever = {name: False for name, _ in named}
    first_nonzero, records = None, []
    for update in range(updates):
        _sync(device)
        start = time.perf_counter()
        wait = vae_time = 0.0
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            t0 = time.perf_counter()
            batch = next(batches)
            wait += time.perf_counter() - t0
            rgb = batch['rgb'].to(device, non_blocking=True)
            _sync(device)
            t1 = time.perf_counter()
            latents = encode_image(vae, rgb, control=False, generator=generator)
            control = None
            if use_control:
                control = encode_image(vae, batch['control'].to(device), control=True)
            _sync(device)
            vae_time += time.perf_counter() - t1
            size = latents.shape[0]
            with compute_context(device, dtype):
                loss = diffusion_loss(model, scheduler, latents, embeds.expand(size, -1, -1),
                                      mask.expand(size, -1), generator=generator,
                                      control=control)
            (loss / accumulation).backward()
            total += loss.item() / accumulation
        if not math.isfinite(total):
            raise FloatingPointError(f'Non-finite loss at update {first_update + update + 1}')
        if probe_gradients:
            norms = torch.stack([p.grad.norm() if p.grad is not None else p.new_zeros(())
                                 for p in parameters]).tolist()
            for (name, _), value in zip(named, norms):
                ever[name] |= value > 0
            if first_nonzero is None:
                first_nonzero = [n for (n, _), v in zip(named, norms) if v > 0]
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm,
                                              error_if_nonfinite=True).item()
        used_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        schedule.step()
        _sync(device)
        records.append({'lr': used_lr, 'loss': total, 'grad_norm': norm,
                        'seconds': time.perf_counter() - start,
                        'data_wait_seconds': wait, 'vae_seconds': vae_time})
        if callback is not None:
            callback(first_update + update + 1, records[-1])
            now_trainable = {id(p) for p in model.parameters() if p.requires_grad}
            if now_trainable != expected_trainable or (
                    active_adapters(model.transformer) != expected_adapters):
                raise RuntimeError('Callback changed the trainable set or active adapter')
    evidence = {}
    if probe_gradients:
        evidence = {'first_update_nonzero_grad_tensors': len(first_nonzero),
                    'ever_nonzero_grad_tensors': sum(ever.values()),
                    'trainable_tensors': len(named),
                    'tensors_never_receiving_gradient': [n for n, v in ever.items() if not v]}
    return records, evidence
