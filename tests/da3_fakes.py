"""Generated CPU fixtures for the DA3 wrapper, trainer and RGB modes; no data or model downloads."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from generative_sim2real.data import rgb_modes as rm
from generative_sim2real.depth.model import DA3MetricDepth

FOCAL = 600.0


class FakeScratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_conv = nn.Conv2d(4, 1, 1)
        self.sky_output_conv2 = nn.Conv2d(4, 1, 1)


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_sky_head = True
        self.scratch = FakeScratch()

    def forward(self, features):
        return torch.exp(self.scratch.depth_conv(features))[:, 0]


class FakeNetwork(nn.Module):
    """Mimics the DA3 call contract: input [B,1,3,H,W], output {'depth': [B,H,W]}."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, 1)
        self.head = FakeHead()
        self.calls = []

    def forward(self, x, **kwargs):
        self.calls.append((x.detach().clone(), kwargs))
        assert x.ndim == 5 and x.shape[1] == 1
        features = self.backbone(x[:, 0])
        return {'depth': self.head(features)}


class FakeBlock(nn.Module):
    """A transformer-like block exposing the DA3 module names attn.qkv / attn.proj (no real attention)."""
    def __init__(self, dim=4):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, 3 * dim)
        self.attn.proj = nn.Linear(dim, dim)

    def forward(self, x):
        _, _, v = self.attn.qkv(x).chunk(3, dim=-1)
        return x + self.attn.proj(v)


class FakeBlocksBackbone(nn.Module):
    """backbone.pretrained.blocks like the real DA3 backbone; maps [B,3,H,W] to 4-channel features."""
    def __init__(self, blocks=5, dim=4):
        super().__init__()
        self.stem = nn.Conv2d(3, dim, 1)
        self.pretrained = nn.Module()
        self.pretrained.blocks = nn.ModuleList([FakeBlock(dim) for _ in range(blocks)])

    def forward(self, x):
        features = self.stem(x)
        batch, channels, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        for block in self.pretrained.blocks:
            tokens = block(tokens)
        return tokens.transpose(1, 2).reshape(batch, channels, h, w)


def fake_bb_model(seed=0):
    """Same head as fake_model(seed) (identical RNG draws), but with a block-structured backbone for bbLoRA tests."""
    model = fake_model(seed)
    model.network.backbone = FakeBlocksBackbone()
    return model


def fake_model(seed=0):
    torch.manual_seed(seed)
    network = FakeNetwork()
    with torch.no_grad():
        network.head.scratch.depth_conv.bias.fill_(-1.)
    return DA3MetricDepth(network)


def intrinsics(size=512):
    return torch.tensor([[FOCAL, 0, size / 2], [0, FOCAL, size / 2], [0, 0, 1]], dtype=torch.float32)


class FakeSplit:
    """Duck-typed ManifestDataset: a deterministic labeled sample per index."""
    def __init__(self, directory, split, geometry, count, size, sequences=2, name=None):
        self.split, self.geometry, self.size = split, geometry, size
        self.rows = [{'sample_id': f'fake/seq{i % sequences}/{i:04d}', 'sequence_id': f'seq{i % sequences}',
                      'depth_path': f'fake/depth_{i:04d}.png'} for i in range(count)]
        self.sidecar = {'manifest_sha256': 'f' * 64, 'transform_definitions': {'x': 1}, 'split': split}
        self.path = Path(directory) / f'{name or split}.jsonl'
        self.path.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        self.path.with_suffix('.meta.json').write_text(json.dumps(self.sidecar))
        self.depth_bias = 0.

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rng = np.random.default_rng(1000 + index)
        n = self.size
        rgb = torch.from_numpy(rng.uniform(.05, .95, (3, n, n)).astype(np.float32))
        yy = np.linspace(0, 1, n, dtype=np.float32)[:, None] * np.ones((1, n), np.float32)
        depth = (.02 + .03 * yy + .004 * (index % 3) + self.depth_bias)[None]
        valid = np.ones((1, n, n), bool)
        valid[:, : n // 8, :] = False          # invalid band, labels there are garbage
        depth = np.where(valid, depth, np.nan if index % 2 else 0.).astype(np.float32)
        return {'sample_id': self.rows[index]['sample_id'], 'rgb': rgb,
                'camera': {'K': intrinsics(n), 'T_world_from_camera': None, 'distortion': None, 'model': 'pinhole'},
                'targets': {'depth_m': torch.from_numpy(depth), 'depth_valid': torch.from_numpy(valid),
                            'sparse_depth': None, 'instances': None},
                'meta': {'dataset': 'fake', 'domain': 'source', 'split': self.split, 'scene_id': None,
                         'sequence_id': self.rows[index]['sequence_id'], 'frame_index': index,
                         'timestamp_s': None, 'original_hw': (n, n), 'depth_type': 'camera_z',
                         'transform_history': []}}


def write_generated(directory, source, ids, outputs_root, *, mutate=None):
    """Mimics the P3 generated-manifest contract (rgb_modes.py docstring)."""
    outputs_root = Path(outputs_root)
    (outputs_root / 'gen').mkdir(parents=True, exist_ok=True)
    lookup = {r['sample_id']: i for i, r in enumerate(source.rows)}
    records = []
    for k, sid in enumerate(ids):
        row = source.rows[lookup[sid]]
        rng = np.random.default_rng(5000 + k)
        image = outputs_root / 'gen' / f'{k:04d}.png'
        Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).save(image)
        records.append({'sample_id': sid, 'source_sample_id': sid, 'source_depth_path': row['depth_path'],
                        'source_transform_id': 'source_generation512',
                        'preprocessing_sha256': rm.source_fingerprint(row, source.sidecar),
                        'source_geometry_sha256': rm.geometry_sha256(source[lookup[sid]]),
                        'generated_rgb_path': f'gen/{k:04d}.png', 'generated_rgb_sha256': rm.file_sha256(image),
                        'base_model': 'fake', 'base_revision': 'r', 'source_adapter': 's', 'target_adapter': 't',
                        'adapter_weight': 1., 'controller': 'c', 'controller_scale': 1., 'prompt': 'p', 'seed': k,
                        'scheduler': 'x', 'inference_steps': 1, 'precision': 'float32', 'size_hw': [512, 512]})
    if mutate:
        mutate(records)
    path = Path(directory) / 'generated.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    return path


def write_ids(directory, source, ids, *, smoke=True, frozen=False):
    path = Path(directory) / 'ids.json'
    path.write_text(json.dumps({'schema_version': 1, 'purpose': 'smoke' if smoke else 'experiment', 'frozen': frozen,
                                'source_manifest_sha256': rm.file_sha256(source.path),
                                'source_sidecar_sha256': rm.file_sha256(source.path.with_suffix('.meta.json')),
                                'sample_ids': list(ids)}))
    return path


def small_config(**training):
    config = yaml.safe_load(Path('configs/da3_smoke.yaml').read_text())
    config['training'].update({'optimizer_steps': 6, 'global_batch_size': 4, 'microbatch_size': 2, 'warmup_steps': 2,
                               'validation_every': 3, 'checkpoint_every': 1, 'learning_rate': 0.05, **training})
    config['evaluation']['validation_frames'] = None
    return config
