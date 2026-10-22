"""Low-capacity ("Control-LoRA style") depth controller, an option beside the full controller.

The full controller (controlnet.py) is 13 fully trainable copied PixArt blocks plus zero-initialised
full 1152x1152 projections. This module converts such a controller IN PLACE into a restricted one:

  * the 13 copied blocks stay exactly the clean-base weights and are FROZEN;
  * trainable: a LoRA (rank `block_rank`, alpha = rank) on every attention projection of each copied
    block (to_q, to_k, to_v, to_out.0 of self- and cross-attention, the backbone LoRA's target set);
  * trainable: each `after_proj` (13) and block 0's `before_proj` become a zero-at-init low-rank
    factorisation (down: in->r random, up: r->out zero + zero bias). Ranks `after_rank`/`before_rank`.

The controller output is therefore exactly zero at initialisation, like the full controller's.
The LoRA wrapper is self-contained (no PEFT), so it cannot clash with the PEFT adapters on the
backbone. The forward path (PixArtControlNetTransformerModel) is unchanged.

Files: `controller_lowrank.safetensors` holds ONLY the trainable tensors plus metadata (kind, ranks,
base revision, hash of the frozen copied blocks); the frozen blocks are re-created from the clean
base at start-up and verified against that hash. `gm.load_controller` rejects such files (different
`format`), and `load_any` here dispatches on that format.
"""
import math
from pathlib import Path

import torch
from torch import nn

from . import model as gm

KINDS = ('full', 'lowrank')
FILE_NAMES = {'full': 'controller.safetensors', 'lowrank': 'controller_lowrank.safetensors'}
FORMAT = 'generative_sim2real_controller_lowrank_v1'
RANK_KEYS = ('block_rank', 'before_rank', 'after_rank')
DEFAULTS = {'block_rank': 8, 'before_rank': 32, 'after_rank': 8}
SPEC_KEYS = ('kind', *RANK_KEYS, 'init_seed')
DEFAULT_INIT_SEED = 5303
TRAINABLE_MARK = ('lr_down', 'lr_up')
ATTENTION_TARGETS = tuple(gm.LORA_MODULES)  # to_q, to_k, to_v, to_out.0 (as the backbone LoRA)


def is_trainable_name(name):
    return any(f'.{mark}.' in f'.{name}.' for mark in TRAINABLE_MARK)


class LowRankLinear(nn.Module):
    """y = base(x) + (alpha / r) * up(down(x)); base frozen, up zero-initialised, alpha = r."""

    def __init__(self, base, rank, generator):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = 1.0  # alpha == rank
        self.lr_down = nn.Linear(base.in_features, rank, bias=False)
        self.lr_up = nn.Linear(rank, base.out_features, bias=False)
        _init_down(self.lr_down, generator)
        nn.init.zeros_(self.lr_up.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lr_up(self.lr_down(x))


class LowRankProjection(nn.Module):
    """Zero-at-init replacement of a full Linear: down (in->r, random), up (r->out, zero + zero bias)."""

    def __init__(self, in_features, out_features, rank, generator):
        super().__init__()
        self.rank = rank
        self.lr_down = nn.Linear(in_features, rank, bias=False)
        self.lr_up = nn.Linear(rank, out_features, bias=True)
        _init_down(self.lr_down, generator)
        nn.init.zeros_(self.lr_up.weight)
        nn.init.zeros_(self.lr_up.bias)

    def forward(self, x):
        return self.lr_up(self.lr_down(x))


def _init_down(layer, generator):
    """Uniform(+-1/sqrt(in)) (the nn.Linear default bound) from an explicit generator."""
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.copy_((torch.rand(layer.weight.shape, generator=generator) * 2 - 1) * bound)


def _get(module, dotted):
    for part in dotted.split('.'):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set(module, dotted, value):
    *head, last = dotted.split('.')
    parent = _get(module, '.'.join(head)) if head else module
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


# ------------------------------------------------------------------ spec ----

def _check_rank(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'controller.{name} must be a positive int, got {value!r}')


def resolve_spec(top=None, run=None):
    """Merge the config-level `controller:` block with a run-level override.

    Returns None for the full controller (absent block or kind: full, also as a run override of a
    config-level lowrank block: today's behaviour, no extra
    keys anywhere) and a complete dict {kind, block_rank, before_rank, after_rank, init_seed} for
    lowrank. Unknown keys and ranks given for kind: full are rejected."""
    merged = {}
    if run is not None and run.get('kind') == 'full':
        top = None  # a run that says kind: full ignores the config-level low-rank defaults
    for block in (top, run):
        if block is None:
            continue
        unknown = set(block) - set(SPEC_KEYS)
        if unknown:
            raise ValueError(f'Unknown controller keys: {sorted(unknown)}')
        merged.update(block)
    kind = merged.get('kind', 'full')
    if kind not in KINDS:
        raise ValueError(f'controller.kind must be one of {KINDS}, got {kind!r}')
    if kind == 'full':
        if set(merged) - {'kind'}:
            raise ValueError('controller.kind: full takes no rank or seed keys')
        return None
    spec = {'kind': 'lowrank', **{k: merged.get(k, DEFAULTS[k]) for k in RANK_KEYS},
            'init_seed': merged.get('init_seed', DEFAULT_INIT_SEED)}
    for key in RANK_KEYS:
        _check_rank(key, spec[key])
    if type(spec['init_seed']) is not int:
        raise ValueError('controller.init_seed must be an int')
    return spec


def recipe_spec(controller):
    """Spec of a bulk recipe's `controller` entry (optional kind + ranks); None for full.

    Included in the recipe hash only when not None, so full-controller hashes do not change."""
    keys = {k: controller[k] for k in ('kind', *RANK_KEYS) if k in controller}
    if keys.get('kind', 'full') == 'full':
        if set(keys) - {'kind'}:
            raise ValueError('controller.kind: full takes no rank keys')
        return None
    if keys.get('kind') != 'lowrank' or set(keys) != {'kind', *RANK_KEYS}:
        raise ValueError('A lowrank recipe controller needs kind and block_rank, before_rank, after_rank')
    for key in RANK_KEYS:
        _check_rank(key, keys[key])
    return {k: keys[k] for k in ('kind', *RANK_KEYS)}


def file_name(spec):
    return FILE_NAMES['lowrank' if spec else 'full']


# ------------------------------------------------------------ conversion ----

def is_lowrank(controller):
    return any(isinstance(m, (LowRankLinear, LowRankProjection)) for m in controller.modules())


def frozen_items(controller):
    """(name, tensor) of the copied blocks' base weights under their original (full) names."""
    return [(name.replace('.base.', '.'), tensor) for name, tensor in controller.state_dict().items()
            if not is_trainable_name(name) and '.transformer_block.' in name]


def frozen_hash(controller):
    return gm.tensors_hash(frozen_items(controller))


def count_parameters(controller):
    total = sum(p.numel() for p in controller.parameters())
    trainable = sum(p.numel() for n, p in controller.named_parameters() if is_trainable_name(n))
    return {'total_elements': total, 'trainable_elements': trainable,
            'frozen_elements': total - trainable}


def convert(controller, spec):
    """Turn a clean full controller into the low-rank one, in place. Returns provenance.

    The copied blocks' weights are untouched (hash compared before/after); the full projections
    are replaced (their zero values are discarded). Refuses a controller that is not at its clean
    initialisation (non-zero projections) or already converted."""
    if is_lowrank(controller):
        raise ValueError('Controller is already low-rank')
    if spec is None or spec.get('kind') != 'lowrank':
        raise ValueError('convert needs a lowrank spec')
    blocks = controller.controlnet_blocks
    zero_full = all(not p.any() for n, p in controller.named_parameters()
                    if n.split('.')[-2] in ('before_proj', 'after_proj'))
    if not zero_full:
        raise ValueError('Convert only a freshly initialised controller (zero projections)')
    full_trainable = sum(p.numel() for p in controller.parameters())
    before_hash = gm.tensors_hash(
        [(n, t) for n, t in controller.state_dict().items() if '.transformer_block.' in n])
    reference = next(controller.parameters())
    device, dtype = reference.device, reference.dtype
    generator = torch.Generator().manual_seed(spec['init_seed'])
    for block in blocks:
        inner = block.inner_dim
        for attention in ('attn1', 'attn2'):
            for target in ATTENTION_TARGETS:
                path = f'transformer_block.{attention}.{target}'
                _set(block, path, LowRankLinear(_get(block, path), spec['block_rank'], generator))
        block.after_proj = LowRankProjection(inner, inner, spec['after_rank'], generator)
        if block.block_index == 0:
            block.before_proj = LowRankProjection(inner, inner, spec['before_rank'], generator)
    controller.to(device=device, dtype=dtype)
    restrict_trainable(controller)
    after_hash = frozen_hash(controller)
    if after_hash != before_hash:
        raise RuntimeError('Conversion changed the copied block weights')
    return {'kind': 'lowrank', **{k: spec[k] for k in RANK_KEYS}, 'init_seed': spec['init_seed'],
            'frozen_blocks_sha256': after_hash, 'frozen_blocks_unchanged_by_conversion': True,
            'zero_at_init': zero_at_init(controller),
            'full_controller_trainable_elements': full_trainable,
            **count_parameters(controller)}


def zero_at_init(controller):
    """True iff every projection's up weights and bias are exactly zero (=> controller output 0)."""
    ups = [p for n, p in controller.named_parameters()
           if ('before_proj.lr_up' in n or 'after_proj.lr_up' in n)]
    return bool(ups) and all(not p.any() for p in ups)


def restrict_trainable(controller, transformer=None):
    """Freeze everything but the low-rank factors. `gm.set_stage(stage='controller')` re-enables
    every controller parameter, so call this after it. Returns gm.trainable_parameters(...)."""
    for name, parameter in controller.named_parameters():
        parameter.requires_grad_(is_trainable_name(name))
    if transformer is not None:
        return gm.trainable_parameters(transformer, controller)
    return [(f'controller.{n}', p) for n, p in controller.named_parameters() if p.requires_grad]


def spec_of(controller):
    """Ranks of an already converted controller (read from the modules)."""
    first = controller.controlnet_blocks[0]
    return {'kind': 'lowrank', 'block_rank': first.transformer_block.attn1.to_q.rank,
            'before_rank': first.before_proj.rank, 'after_rank': first.after_proj.rank}


# --------------------------------------------------------------- storage ----

def trainable_state(controller):
    return {n: t.detach().cpu().contiguous() for n, t in controller.state_dict().items()
            if is_trainable_name(n)}


def metadata_of(controller, revision):
    spec = spec_of(controller)
    return {'format': FORMAT, 'kind': 'lowrank', **{k: str(spec[k]) for k in RANK_KEYS},
            'alpha': 'rank', 'num_layers': str(controller.num_layers), 'base_revision': revision,
            'attention_targets': ','.join(ATTENTION_TARGETS),
            'frozen_blocks_sha256': frozen_hash(controller)}


def save(controller, path, *, revision=''):
    from safetensors.torch import save_file

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(controller), str(path), metadata=metadata_of(controller, revision))


def read_metadata(path):
    from safetensors import safe_open

    with safe_open(str(path), framework='pt') as handle:
        return handle.metadata() or {}


def spec_from_metadata(metadata):
    return {'kind': 'lowrank', **{k: int(metadata[k]) for k in RANK_KEYS}}


def check_file_matches_spec(path, spec):
    """A recipe's declared kind/ranks must equal the file's own metadata."""
    metadata = read_metadata(path)
    if metadata.get('format') != FORMAT:
        raise ValueError(f'{path} is not a low-rank controller file')
    if spec_from_metadata(metadata) != {k: spec[k] for k in ('kind', *RANK_KEYS)}:
        raise ValueError(f'{path}: recipe controller ranks differ from the file metadata')


def load(controller, path, *, revision):
    """Load a low-rank controller file, converting a clean full controller first if needed.

    Checks the format, base revision, layer count, ranks, the hash of the frozen copied blocks
    (the freshly built clean-base copy must be byte-identical to the one trained) and every tensor
    name/shape."""
    from safetensors.torch import load_file

    metadata = read_metadata(path)
    if metadata.get('format') != FORMAT or metadata.get('kind') != 'lowrank':
        raise ValueError('Not a low-rank controller file written by this project')
    if metadata.get('base_revision') != revision:
        raise ValueError('Controller metadata mismatch for base_revision')
    if metadata.get('num_layers') != str(controller.num_layers):
        raise ValueError('Controller metadata mismatch for num_layers')
    wanted = spec_from_metadata(metadata)
    if is_lowrank(controller):
        if spec_of(controller) != wanted:
            raise ValueError(f'Controller is low-rank with ranks {spec_of(controller)}, '
                             f'file has {wanted}')
    else:
        convert(controller, {**wanted, 'init_seed': DEFAULT_INIT_SEED})
    if frozen_hash(controller) != metadata.get('frozen_blocks_sha256'):
        raise ValueError('Frozen copied blocks differ from those the controller was trained with')
    state, expected = load_file(str(path)), trainable_state(controller)
    if state.keys() != expected.keys() or any(state[k].shape != expected[k].shape for k in state):
        raise ValueError('Low-rank controller tensors do not match the configured ranks')
    parameters = dict(controller.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor.to(parameters[name].dtype))
    return metadata


def load_any(controller, path, *, revision):
    """Load either controller file kind into `controller`, dispatching on the file's own format."""
    metadata = read_metadata(path)
    if metadata.get('format') == FORMAT:
        return load(controller, path, revision=revision)
    if is_lowrank(controller):
        raise ValueError('A full controller file cannot be loaded into a low-rank controller')
    return gm.load_controller(controller, path, revision=revision)
