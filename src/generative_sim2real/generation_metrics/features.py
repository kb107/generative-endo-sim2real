"""Feature extraction: DINOv2 (HF transformers) CLS embeddings, plus descriptor caching.

The metrics are DINOv2-feature variants of KID / Frechet distance, NOT standard Inception FID/KID:
their scale is not comparable with published FID numbers."""
import hashlib
from pathlib import Path

import torch

from .pixel import pixel_descriptors, to_size

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL = 'facebook/dinov2-base'


class Dinov2Extractor:
    """CLS token after the final layernorm (`pooler_output`) of an HF Dinov2Model at image_size."""

    def __init__(self, model_id=DEFAULT_MODEL, *, image_size=224, device='cpu'):
        from transformers import AutoModel

        try:
            self.model = AutoModel.from_pretrained(model_id, local_files_only=True)
        except OSError:
            self.model = AutoModel.from_pretrained(model_id)
        self.model = self.model.to(device).eval().requires_grad_(False)
        self.model_id, self.image_size, self.device = model_id, image_size, device
        self.name = f'{model_id}:cls:{image_size}'
        self.revision = getattr(self.model.config, '_commit_hash', None)
        self._mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images):
        x = to_size(images.float(), self.image_size).to(self.device)
        x = (x - self._mean) / self._std
        return self.model(pixel_values=x).pooler_output.float().cpu()


def describe(image_set, extractor, *, batch_size=32, workers=0):
    """Features + pixel descriptors for every image of the set (images are streamed, not kept)."""
    parts = {}
    for images in image_set.batches(batch_size, workers):
        if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
            raise ValueError('Images must be finite [B,3,H,W]')
        row = pixel_descriptors(images)
        row['features'] = extractor(images)
        for key, value in row.items():
            parts.setdefault(key, []).append(value.cpu())
    desc = {key: torch.cat(value) for key, value in parts.items()}
    desc['ids'] = list(image_set.ids)
    return desc


def describe_cached(image_set, extractor, cache_dir, *, batch_size=32, workers=0, tag=''):
    """describe() with an on-disk cache keyed by image ids, extractor and tag (use for reference sets
    that are reused across many candidate comparisons)."""
    if cache_dir is None:
        return describe(image_set, extractor, batch_size=batch_size, workers=workers)
    key = hashlib.sha256(f'{image_set.fingerprint()}|{extractor.name}|{tag}'.encode()).hexdigest()[:20]
    path = Path(cache_dir) / f'{image_set.name}_{key}.pt'
    if path.exists():
        return torch.load(path, weights_only=False)
    desc = describe(image_set, extractor, batch_size=batch_size, workers=workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(desc, tmp)
    tmp.replace(path)
    return desc
