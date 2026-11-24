"""Atomic head-only artifacts and exact single-process training resume.

Selectively adapted from the reference training/checkpoints.py; local checkpoint
files use torch's trusted pickle format and must originate from this project.
Only the DPT head is saved (plus, for the optional bbLoRA variant, the small
backbone-LoRA state): the frozen backbone always comes from the pinned official
checkpoint. Head-only files are byte-compatible with the original format; bbLoRA
best files use kind ``da3_head_bblora`` so a head-only reader cannot silently
ignore the adapters.
"""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

FORMAT_VERSION = 1
HEAD_KIND, HEAD_BBLORA_KIND = 'da3_head', 'da3_head_bblora'


def atomic_replace(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        writer(temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(value, path):
    atomic_replace(path, lambda temporary: torch.save(value, temporary))


def atomic_write_json(value, path):
    atomic_replace(path, lambda temporary: Path(temporary).write_text(
        json.dumps(value, indent=2, allow_nan=False) + '\n'))


def head_digest(state):
    """SHA-256 over sorted names, dtypes, shapes and raw bytes of a state dict."""
    value = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        value.update(f'{name}|{tensor.dtype}|{tuple(tensor.shape)}|'.encode())
        value.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _head_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.head_state_dict().items()}


def _lora_copy(model):
    return {k: v.detach().cpu().clone() for k, v in model.backbone_lora_state_dict().items()}


def save_training(path, model, optimizer, scheduler, state, compatibility):
    payload = {'format_version': FORMAT_VERSION, 'kind': 'da3_training', 'head_state_dict': _head_copy(model),
               'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
               'state': state, 'rng': capture_rng(), 'compatibility': compatibility}
    if model.backbone_lora is not None:
        payload['backbone_lora_state_dict'] = _lora_copy(model)
    atomic_save(payload, path)


def load_training(path, model, optimizer, scheduler, compatibility):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != FORMAT_VERSION or checkpoint.get('kind') != 'da3_training':
        raise ValueError('Expected resumable DA3 training checkpoint')
    if checkpoint.get('compatibility') != compatibility:
        raise ValueError('Resume mismatch: configuration, data, environment, code or batch topology changed')
    if (model.backbone_lora is not None) != ('backbone_lora_state_dict' in checkpoint):
        raise ValueError('Resume mismatch: backbone-LoRA presence differs between model and checkpoint')
    model.load_head_state_dict(checkpoint['head_state_dict'])
    if model.backbone_lora is not None:
        model.load_backbone_lora_state_dict(checkpoint['backbone_lora_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    restore_rng(checkpoint['rng'])
    state = checkpoint['state']
    if state['cursor'] != state['optimizer_step'] * compatibility['config']['training']['global_batch_size']:
        raise ValueError('Checkpoint sample cursor does not match completed optimizer updates')
    return state


def save_head(path, model, state, compatibility):
    head = _head_copy(model)
    payload = {'format_version': FORMAT_VERSION, 'kind': HEAD_KIND, 'head_state_dict': head,
               'head_sha256': head_digest(head), 'selected_step': state['best_step'],
               'source_validation_sequence_macro_abs_rel': state['best_abs_rel'],
               'compatibility': compatibility}
    if model.backbone_lora is not None:
        lora = _lora_copy(model)
        payload.update(kind=HEAD_BBLORA_KIND, backbone_lora_state_dict=lora, backbone_lora_sha256=head_digest(lora),
                       backbone_lora_config=dict(model.backbone_lora_config))
    atomic_save(payload, path)


def _check_lora_payload(checkpoint):
    lora = checkpoint.get('backbone_lora_state_dict')
    if lora is None or not checkpoint.get('backbone_lora_config') or head_digest(lora) != checkpoint.get('backbone_lora_sha256'):
        raise ValueError('Backbone-LoRA checkpoint content does not match its recorded digest')


def read_head(path, allow_backbone_lora=False):
    """Load a saved head and verify that its payload still matches its recorded digest(s).

    A bbLoRA checkpoint is refused unless the caller opts in: a head-only reader would otherwise drop the adapters.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    kind = checkpoint.get('kind')
    if checkpoint.get('format_version') != FORMAT_VERSION or kind not in (HEAD_KIND, HEAD_BBLORA_KIND):
        raise ValueError('Expected a DA3 head checkpoint')
    if kind == HEAD_BBLORA_KIND:
        if not allow_backbone_lora:
            raise ValueError('This head checkpoint also carries backbone LoRA; read it with allow_backbone_lora=True '
                             'and apply it with apply_checkpoint')
        _check_lora_payload(checkpoint)
    if head_digest(checkpoint['head_state_dict']) != checkpoint['head_sha256']:
        raise ValueError('Head checkpoint content does not match its recorded digest')
    return checkpoint


def read_final_training(path):
    """The head (and LoRA) state of a resumable ``last.pt`` as a head-style record (digests computed, not trusted).

    ``selected_step`` is the optimizer step the file was written at; callers must prove it is the run's final step.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != FORMAT_VERSION or checkpoint.get('kind') != 'da3_training':
        raise ValueError('Expected resumable DA3 training checkpoint')
    out = {'format_version': FORMAT_VERSION, 'kind': 'da3_training_final', 'head_state_dict': checkpoint['head_state_dict'],
           'head_sha256': head_digest(checkpoint['head_state_dict']), 'selected_step': checkpoint['state']['optimizer_step'],
           'compatibility': checkpoint['compatibility'], 'state': checkpoint['state']}
    lora = checkpoint.get('backbone_lora_state_dict')
    if lora is not None:
        out.update(backbone_lora_state_dict=lora, backbone_lora_sha256=head_digest(lora),
                   backbone_lora_config=checkpoint['compatibility']['config']['model']['backbone_lora'])
    return out


def apply_checkpoint(model, checkpoint):
    """Make ``model`` exactly official backbone + the checkpoint's head (+ its backbone LoRA, or none)."""
    if 'backbone_lora_state_dict' in checkpoint:
        model.set_backbone_lora(checkpoint['backbone_lora_config'], checkpoint['backbone_lora_state_dict'])
    else:
        model.set_backbone_lora(None)
    model.load_head_state_dict(checkpoint['head_state_dict'])
