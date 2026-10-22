"""Image sources: generated PNG directories / manifests, and role-checked real reference manifests.

Real reference frames go through generation.inputs.generation_dataset + FrameTensors with
`with_control=False` (RGB only; C1 manifests carry no labels at all). generation_dataset allows
only source_train, source_validation and target_appearance_{train,validation} manifests, so
target_test / C2 manifests are refused before any frame is decoded."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .pixel import evenly_spaced

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}
MANIFEST_PATH_KEYS = ('generated_rgb_path', 'image_path', 'rgb_path', 'path', 'file')
# split -> stage name understood by generation.inputs.generation_dataset
REFERENCE_STAGES = {'source_train': 'source_lora', 'source_validation': 'validation',
                    'target_appearance_train': 'target_lora',
                    'target_appearance_validation': 'validation'}
FLOOR_PARTNER = {'source_validation': 'source_train.jsonl',
                 'target_appearance_validation': 'target_appearance_train.jsonl'}


def load_png(path):
    array = np.asarray(Image.open(path).convert('RGB')).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


class _Files(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return {'rgb': load_png(self.paths[index])}


class _Frames(Dataset):
    def __init__(self, frames, indices):
        self.frames, self.indices = frames, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {'rgb': self.frames[self.indices[index]]['rgb']}


class ImageSet:
    """A list of image ids plus a way to stream them as [B,3,H,W] float tensors in [0,1]."""

    def __init__(self, name, ids, dataset, *, kind, source):
        self.name, self.ids, self.dataset, self.kind, self.source = name, list(ids), dataset, kind, source

    def __len__(self):
        return len(self.ids)

    def batches(self, batch_size, workers=0):
        for batch in DataLoader(self.dataset, batch_size=batch_size, shuffle=False,
                                num_workers=workers):
            yield batch['rgb']

    def fingerprint(self):
        return hashlib.sha256('\n'.join(self.ids).encode()).hexdigest()


def pick_even(items, limit):
    return items if limit is None or limit >= len(items) else [
        items[i] for i in evenly_spaced(len(items), limit)]


def generated_set(path, *, root='outputs', limit=None):
    """`path` is a directory of images (searched recursively) or a manifest .jsonl whose rows hold
    a relative image path (generated_rgb_path/image_path/path/file, relative to `root`)."""
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
    elif path.suffix == '.jsonl':
        files = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = next((k for k in MANIFEST_PATH_KEYS if k in row), None)
            if key is None:
                raise ValueError(f'Manifest row has none of {MANIFEST_PATH_KEYS}')
            files.append(Path(root) / row[key])
    else:
        raise ValueError('--generated must be a directory or a .jsonl manifest')
    if not files:
        raise ValueError(f'No images found in {path}')
    files = pick_even(files, limit)
    return ImageSet(path.name, [str(f) for f in files], _Files(files), kind='generated',
                    source=str(path))


def reference_set(manifest, *, data_root=None, limit=None):
    """Real RGB frames from a project manifest (role-checked). `limit` takes evenly spaced frames."""
    from generative_sim2real.generation.inputs import FrameTensors, generation_dataset

    manifest = Path(manifest)
    split = json.loads(manifest.with_suffix('.meta.json').read_text())['split']
    if split not in REFERENCE_STAGES:
        raise ValueError(f'Manifest role {split!r} is not an allowed reference set')
    data_root = data_root or os.environ['DATA_ROOT']
    frames = generation_dataset(manifest, data_root, stage=REFERENCE_STAGES[split])
    # FrameTensors rejects control for appearance manifests; we never request it.
    FrameTensors(frames, with_control=False, indices=[0])
    indices = pick_even(list(range(len(frames))), limit)
    ids = [frames.rows[i]['sample_id'] for i in indices]
    return ImageSet(f'{manifest.stem}', ids, _Frames(frames, indices), kind='reference',
                    source=str(manifest)), split


def floor_manifest_for(reference_manifest):
    """The train manifest paired with a validation manifest (same directory), or None."""
    reference_manifest = Path(reference_manifest)
    split = json.loads(reference_manifest.with_suffix('.meta.json').read_text())['split']
    partner = FLOOR_PARTNER.get(split)
    return None if partner is None else reference_manifest.with_name(partner)
