"""Atomic resumable training state and separate adapter/controller exports (P4).

Follows depth/checkpoints.py: one `last.pt` holds everything needed so an interrupted run
continues bitwise like an uninterrupted one (trainable weights, optimizer, LR schedule, noise
generator, global RNG, update count and data cursor, plus a compatibility fingerprint that
must match exactly). Exports are separate safetensors files, one kind per file:
`lora_<adapter>.safetensors` or `controller.safetensors` (+ `meta.json`); a LoRA export holds
no controller tensors and vice versa. Exports are derived artifacts; `last.pt` is the resume
authority.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from generative_sim2real.depth.checkpoints import (atomic_replace, atomic_save,
                                                   atomic_write_json, capture_rng, restore_rng)

from . import model as gm

FORMAT_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_training_state(path, named, optimizer, schedule, generator, state, compatibility):
    """`state`: {'update': completed optimizer updates, 'cursor': micro-batches consumed}."""
    atomic_save({
        'format_version': FORMAT_VERSION, 'kind': 'generator_training',
        'names': [n for n, _ in named],
        'parameters': {n: p.detach().cpu().clone() for n, p in named},
        'optimizer': optimizer.state_dict(), 'scheduler': schedule.state_dict(),
        'generator': generator.get_state(), 'rng': capture_rng(), 'state': dict(state),
        'compatibility': compatibility}, path)


def load_training_state(path, named, optimizer, schedule, generator, compatibility, *,
                        accumulation):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint.get('format_version') != FORMAT_VERSION
            or checkpoint.get('kind') != 'generator_training'):
        raise ValueError('Expected a resumable generator training checkpoint')
    if checkpoint['compatibility'] != compatibility:
        raise ValueError('Resume mismatch: configuration, data, weights, code or '
                         'environment changed since the checkpoint')
    if checkpoint['names'] != [n for n, _ in named]:
        raise ValueError('Resume mismatch: trainable parameter names differ')
    state = checkpoint['state']
    if state['cursor'] != state['update'] * accumulation:
        raise ValueError('Checkpoint data cursor does not match completed updates')
    with torch.no_grad():
        for name, parameter in named:
            parameter.copy_(checkpoint['parameters'][name].to(parameter.dtype))
    optimizer.load_state_dict(checkpoint['optimizer'])
    schedule.load_state_dict(checkpoint['scheduler'])
    generator.set_state(checkpoint['generator'])
    restore_rng(checkpoint['rng'])
    return state


def export_checkpoint(directory, step, *, kind, model, controller, adapter, rank, alpha,
                      revision, extra=None, controller_spec=None):
    """Write the stage's separate adapter/controller file into `directory/step_NNNNNN/`
    atomically (temporary directory renamed into place). A low-rank controller (`controller_spec`
    from lowrank_controller.resolve_spec) goes to its own file name, never `controller.safetensors`."""
    directory = Path(directory)
    final = directory / f'step_{step:06d}'
    temporary = directory / f'.step_{step:06d}.tmp'
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    if kind == 'lora':
        name = f'lora_{adapter}.safetensors'
        gm.save_adapter(model.transformer, temporary / name, adapter, rank=rank, alpha=alpha,
                        revision=revision)
    elif kind == 'controller' and controller_spec:
        from . import lowrank_controller as lrc

        name = lrc.file_name(controller_spec)
        lrc.save(controller, temporary / name, revision=revision)
        extra = {**(extra or {}), 'controller_kind': 'lowrank',
                 **{k: controller_spec[k] for k in lrc.RANK_KEYS}}
    elif kind == 'controller':
        name = 'controller.safetensors'
        gm.save_controller(controller, temporary / name, revision=revision)
    else:
        raise ValueError('kind must be lora or controller')
    meta = {'step': step, 'kind': kind, 'adapter': adapter if kind == 'lora' else None,
            'file': name, 'sha256': file_sha256(temporary / name),
            'bytes': (temporary / name).stat().st_size, 'saved_unix': time.time(),
            **(extra or {})}
    (temporary / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    shutil.rmtree(final, ignore_errors=True)
    os.replace(temporary, final)
    return final, meta


def write_status(path, **fields):
    atomic_write_json(fields, path)


def eta_seconds(recent_seconds, remaining):
    """Remaining updates times the mean of the recent update times (None before any data)."""
    if not recent_seconds:
        return None
    return remaining * sum(recent_seconds) / len(recent_seconds)


def truncate_jsonl(path, completed):
    """Drop records past the last durable checkpoint (and one torn final line)."""
    path = Path(path)
    if not path.exists():
        return
    lines, kept = path.read_text().splitlines(), []
    for number, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines) - 1:
                break
            raise
        if record['step'] <= completed:
            kept.append(line)
    atomic_replace(path, lambda temporary: Path(temporary).write_text(
        ''.join(line + '\n' for line in kept)))
