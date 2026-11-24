"""Metric-only DA3 wrapper, selectively ported from endoscopy-sim2real/models/da3.py.

Working-file provenance is recorded in docs/code-reuse.md and the P2 evidence.
"""
import math
import subprocess

import torch
import torch.nn.functional as F
from torch import nn

DA3_REVISION = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
CHECKPOINT = 'depth-anything/DA3METRIC-LARGE'
CHECKPOINT_REVISION = '4010e39f3634a45bc60553321fb49fb760bd594e'
# Optional "bbLoRA" variant: low-rank adapters on the last K ViT blocks of the frozen backbone, trained next to the
# DPT head. Absent by default; the head-only protocol never touches any of this code path.
BACKBONE_LORA_TARGETS = ('attn.qkv', 'attn.proj')
BACKBONE_LORA_KEYS = {'rank', 'alpha', 'last_k_blocks', 'targets'}
# The one adapter setting permitted in experiments (configs/da3_bblora.yaml); see depth/train.py validate_config.
LOCKED_BACKBONE_LORA = {'rank': 8, 'alpha': 8, 'last_k_blocks': 4, 'targets': ['attn.qkv', 'attn.proj']}


class LoRADelta(nn.Module):
    """Low-rank update ``scale * B(A(x))`` added to the output of a frozen Linear (B starts at zero)."""

    def __init__(self, in_features, out_features, rank, scale, *, generator=None, device=None, dtype=None):
        super().__init__()
        self.scale = float(scale)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        bound = 1 / math.sqrt(in_features)               # PEFT/Kaiming-uniform(a=sqrt(5)) bound for A
        with torch.no_grad():
            self.lora_A.copy_(torch.empty(rank, in_features).uniform_(-bound, bound, generator=generator))

    def forward(self, x):
        return (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def check_backbone_lora_config(config):
    if not isinstance(config, dict) or set(config) != BACKBONE_LORA_KEYS:
        raise ValueError(f'backbone_lora needs exactly the keys {sorted(BACKBONE_LORA_KEYS)}')
    rank, alpha, last = config['rank'], config['alpha'], config['last_k_blocks']
    if type(rank) is not int or rank < 1 or type(last) is not int or last < 1:
        raise ValueError('backbone_lora rank and last_k_blocks must be positive integers')
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('backbone_lora alpha must be a positive number')
    targets = config['targets']
    if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
            or any(t not in BACKBONE_LORA_TARGETS for t in targets)):
        raise ValueError(f'backbone_lora targets must be a nonempty list drawn from {list(BACKBONE_LORA_TARGETS)}')
    return config


class DA3MetricDepth(nn.Module):
    canonical_focal_px = 300.0

    def __init__(self, network: nn.Module, process_hw=(504, 504)):
        super().__init__()
        self.network = network
        self.process_hw = tuple(process_hw)
        self._validate_hw(self.process_hw)
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406])[None, :, None, None], persistent=False)
        self.register_buffer('image_std', torch.tensor([.229, .224, .225])[None, :, None, None], persistent=False)
        self._head_only = False
        self.backbone_lora = None            # nn.ModuleDict of LoRADelta while a bbLoRA variant is attached
        self.backbone_lora_config = None
        self._lora_hooks = []
        self._disable_sky()

    @classmethod
    def from_pretrained(cls, process_hw=(504, 504)):
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError('Run scripts/setup_da3.sh and use its isolated Python') from exc
        official = DepthAnything3.from_pretrained(CHECKPOINT, revision=CHECKPOINT_REVISION)
        return cls(official.model, process_hw)

    @staticmethod
    def _validate_hw(hw):
        if len(hw) != 2 or any(type(v) is not int or v < 1 or v % 14 for v in hw):
            raise ValueError('DA3 dimensions must be positive integer multiples of 14')

    def prepare_inputs(self, rgb, K):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError('Expected floating RGB [B,3,H,W]')
        if not torch.isfinite(rgb).all() or bool(((rgb < 0) | (rgb > 1)).any()):
            raise ValueError('RGB must be finite in [0,1]')
        self._focal(K, len(rgb))
        h, w = rgb.shape[-2:]
        K = K.clone()
        K[:, 0] *= self.process_hw[1] / w
        K[:, 1] *= self.process_hw[0] / h
        if (h, w) != self.process_hw:
            rgb = F.interpolate(rgb, self.process_hw, mode='bicubic', align_corners=False).clamp(0, 1)
        return rgb, K

    def forward(self, rgb, K):
        rgb, K = self.prepare_inputs(rgb, K)
        output = self.network(((rgb - self.image_mean) / self.image_std)[:, None],
                              extrinsics=None, intrinsics=None, export_feat_layers=[],
                              infer_gs=False, use_ray_pose=False, ref_view_strategy='saddle_balanced')
        depth = output['depth']
        if depth.ndim == 3:
            depth = depth[:, None]
        if depth.ndim != 4 or depth.shape[:2] != (len(rgb), 1) or depth.shape[-2:] != self.process_hw:
            raise ValueError('Expected monocular DA3 canonical depth [B,1,H,W] at processed size')
        return {'canonical_depth': depth, 'depth_m': self.canonical_to_metric(depth, K), 'K_processed': K}

    @staticmethod
    def _focal(K, batch):
        if K is None or K.shape != (batch, 3, 3) or not torch.isfinite(K).all():
            raise ValueError('Expected finite pixel intrinsics [B,3,3]')
        if bool(((K[:, 0, 0] <= 0) | (K[:, 1, 1] <= 0)).any()):
            raise ValueError('Focal lengths must be positive')
        return ((K[:, 0, 0] + K[:, 1, 1]) * .5).reshape(-1, 1, 1, 1)

    def canonical_to_metric(self, depth, K):
        return depth * (self._focal(K, len(depth)).to(depth) / self.canonical_focal_px)

    def metric_to_canonical(self, depth, K):
        return depth * (self.canonical_focal_px / self._focal(K, len(depth)).to(depth))

    def _disable_sky(self):
        head = getattr(self.network, 'head', None)
        if not isinstance(head, nn.Module) or not hasattr(head, 'use_sky_head'):
            raise ValueError('Expected DA3 DPT head with use_sky_head')
        head.use_sky_head = False
        sky = getattr(getattr(head, 'scratch', None), 'sky_output_conv2', None)
        if sky is not None:
            sky.requires_grad_(False)

    def configure_head_training(self):
        if not isinstance(getattr(self.network, 'backbone', None), nn.Module):
            raise ValueError('Expected DA3 backbone')
        self.network.requires_grad_(False)
        self.network.head.requires_grad_(True)
        self._disable_sky()
        self._head_only = True
        self.train()
        return [p for p in self.parameters() if p.requires_grad]

    # ---- optional backbone LoRA (hooks keep network.state_dict(), hence the official digests, untouched)

    def _backbone_blocks(self):
        blocks = getattr(getattr(getattr(self.network, 'backbone', None), 'pretrained', None), 'blocks', None)
        if not isinstance(blocks, nn.ModuleList) or not len(blocks):
            raise ValueError('Expected DA3 backbone.pretrained.blocks')
        return blocks

    def attach_backbone_lora(self, config, seed=0):
        """Add LoRA deltas to ``targets`` of the last ``last_k_blocks`` blocks; the base weights stay untouched."""
        check_backbone_lora_config(config)
        if self.backbone_lora is not None:
            raise ValueError('Backbone LoRA is already attached')
        blocks = self._backbone_blocks()
        if config['last_k_blocks'] > len(blocks):
            raise ValueError('last_k_blocks exceeds the number of backbone blocks')
        generator = torch.Generator().manual_seed(int(seed))
        scale = config['alpha'] / config['rank']
        deltas, hooks = nn.ModuleDict(), []
        try:
            for index in range(len(blocks) - config['last_k_blocks'], len(blocks)):
                for target in config['targets']:
                    layer = blocks[index].get_submodule(target)
                    if not isinstance(layer, nn.Linear):
                        raise ValueError(f'blocks.{index}.{target} is not a Linear layer')
                    delta = LoRADelta(layer.in_features, layer.out_features, config['rank'], scale, generator=generator,
                                      device=layer.weight.device, dtype=layer.weight.dtype)
                    deltas[f'{index}_{target.replace(".", "_")}'] = delta
                    hooks.append(layer.register_forward_hook(lambda module, args, output, delta=delta: output + delta(args[0])))
        except BaseException:
            for hook in hooks:                       # a failed attach leaves no hook behind
                hook.remove()
            raise
        self.backbone_lora, self._lora_hooks = deltas, hooks
        self.backbone_lora_config = copy_config(config)
        return list(deltas.parameters())

    def detach_backbone_lora(self):
        for hook in self._lora_hooks:
            hook.remove()
        self._lora_hooks, self.backbone_lora, self.backbone_lora_config = [], None, None

    def configure_backbone_lora_training(self, config, seed=0):
        """Head + backbone-LoRA training: the head and the new deltas train, every base weight stays frozen."""
        self.configure_head_training()
        self.attach_backbone_lora(config, seed)
        for parameter in self.backbone_lora.parameters():
            parameter.requires_grad_(True)
        self.train()
        return [p for p in self.parameters() if p.requires_grad]

    def backbone_lora_state_dict(self):
        if self.backbone_lora is None:
            raise ValueError('No backbone LoRA is attached')
        return self.backbone_lora.state_dict()

    def load_backbone_lora_state_dict(self, state):
        if self.backbone_lora is None:
            raise ValueError('No backbone LoRA is attached')
        self.backbone_lora.load_state_dict(state, strict=True)

    def set_backbone_lora(self, config=None, state=None):
        """Evaluation helper: make the model exactly (official backbone + [LoRA state]); None removes any LoRA."""
        if self.backbone_lora is not None:
            self.detach_backbone_lora()
        if config is not None:
            self.attach_backbone_lora(config)
            self.load_backbone_lora_state_dict(state)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._head_only:
            self.network.backbone.eval()
            self.network.head.train()
        return self

    def head_state_dict(self):
        return self.network.head.state_dict()

    def load_head_state_dict(self, state):
        self.network.head.load_state_dict(state, strict=True)
        self._disable_sky()


def copy_config(config):
    return {**config, 'targets': list(config['targets'])}


def verify_upstream_checkout(root='external/Depth-Anything-3'):
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != DA3_REVISION or dirty:
        raise ValueError('DA3 checkout must be the clean pinned official revision')
