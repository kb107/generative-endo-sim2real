"""Strict manifest roles and a single train/inference control preprocessing path."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from generative_sim2real.data.manifests import ManifestDataset
from generative_sim2real.data.workers import configure_worker_cleanup
from generative_sim2real.data.preprocessing import InverseDepthEncoding


def generation_dataset(path, data_root, *, stage):
    """Reject forbidden roles before constructing any decoder (including C2 audit)."""
    import json

    split = json.loads(Path(path).with_suffix('.meta.json').read_text())['split']
    allowed = {
        'source_lora': {'source_train'}, 'target_lora': {'target_appearance_train'},
        'controller': {'source_train'},
        'validation': {'source_validation', 'target_appearance_validation'},
        'generation': {'source_train', 'source_validation'},
    }
    if stage not in allowed or split not in allowed[stage]:
        raise ValueError(f'Manifest role {split!r} is forbidden for {stage!r}')
    geometry = 'source_generation512' if split == 'source_validation' else None
    return ManifestDataset(path, data_root, purpose='training', geometry=geometry)


def require_camera_z(sample):
    """The control encoding is defined on axial camera-z, never ray range."""
    depth_type = sample.get('meta', {}).get('depth_type')
    if depth_type != 'camera_z':
        raise ValueError(f'Control encoding requires camera_z depth, got {depth_type!r}')


def control_image(sample, sidecar):
    require_camera_z(sample)
    encoding = InverseDepthEncoding.from_dict(sidecar['depth_control_encoding']['constants'])
    target = sample['targets']
    if target.get('depth_m') is None or target.get('depth_valid') is None:
        raise ValueError('A depth controller requires labeled source data')
    return encoding.encode(target['depth_m'], target['depth_valid'])


def vae_pixels(image):
    """Only normalized RGB/control tensors enter the VAE; no implicit PIL quantization."""
    if image.ndim not in (3, 4) or image.shape[-3] != 3:
        raise ValueError('Expected RGB/control [3,H,W] or [B,3,H,W]')
    if image.dtype != torch.float32 or not torch.isfinite(image).all():
        raise ValueError('RGB/control must be finite float32')
    if image.min() < 0 or image.max() > 1:
        raise ValueError('RGB/control must lie in [0,1]')
    return (image.unsqueeze(0) if image.ndim == 3 else image) * 2 - 1


@torch.no_grad()
def encode_image(vae, image, *, control, generator=None):
    """Control: deterministic mode. RGB: sample each update (or cache posterior moments)."""
    pixels = vae_pixels(image).to(device=vae.device, dtype=vae.dtype)
    posterior = vae.encode(pixels).latent_dist
    latent = posterior.mode() if control else posterior.sample(generator=generator)
    return latent * vae.config.scaling_factor


def tensor_trace(tensor):
    value = tensor.detach().float().cpu()
    return {'shape': list(value.shape), 'dtype': str(tensor.dtype),
            'min': value.min().item(), 'max': value.max().item(),
            'mean': value.mean().item(), 'finite': bool(torch.isfinite(value).all())}


class FrameTensors(Dataset):
    """RGB (and, for labeled source roles only, the encoded control image) per frame.

    Appearance manifests carry no label references, so `with_control=True` on them
    fails at construction rather than silently returning nothing."""

    def __init__(self, manifest, *, with_control, indices=None):
        self.manifest, self.with_control = manifest, with_control
        if with_control:
            if not manifest.split.startswith('source'):
                raise ValueError('Control images exist only for labeled source roles')
            self.encoding = InverseDepthEncoding.from_dict(
                manifest.sidecar['depth_control_encoding']['constants'])
        self.indices = list(range(len(manifest))) if indices is None else list(indices)
        if not self.indices or any(not 0 <= i < len(manifest) for i in self.indices):
            raise ValueError('Frame indices must be a nonempty subset of the manifest')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        sample = self.manifest[self.indices[index]]
        item = {'rgb': sample['rgb'], 'sample_id': sample['sample_id']}
        if self.with_control:
            item['control'] = control_image(sample, self.manifest.sidecar)
        return item


def training_loader(dataset, *, batch_size, seed, workers):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers,
        generator=torch.Generator().manual_seed(seed), persistent_workers=workers > 0,
        worker_init_fn=configure_worker_cleanup if workers else None)


class BatchOrder:
    """Deterministic batch indices: global micro-batch number k -> dataset indices.

    Epoch e shuffles with seed+e and drops the incomplete tail, so the sample order is a
    pure function of (size, batch, seed, k). A resumed run restarts at its cursor and sees
    exactly the batches an uninterrupted run would (no loader-internal RNG to restore)."""

    def __init__(self, size, batch, seed):
        if batch < 1 or size < batch:
            raise ValueError('The dataset must hold at least one full micro-batch')
        self.size, self.batch, self.seed = size, batch, seed
        self.per_epoch = size // batch
        self._epoch, self._perm = None, None

    def indices(self, k):
        epoch, offset = divmod(k, self.per_epoch)
        if epoch != self._epoch:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self._epoch, self._perm = epoch, torch.randperm(self.size, generator=generator)
        return self._perm[offset * self.batch:(offset + 1) * self.batch].tolist()


class CursorSampler:
    """Endless batch sampler that starts at micro-batch `start`."""

    def __init__(self, order, start=0):
        self.order, self.start = order, start

    def __iter__(self):
        k = self.start
        while True:
            yield self.order.indices(k)
            k += 1


def cursor_loader(dataset, *, batch_size, seed, workers, start=0):
    """Endless loader over `BatchOrder`, resumable at micro-batch `start`."""
    return DataLoader(
        dataset, batch_sampler=CursorSampler(BatchOrder(len(dataset), batch_size, seed), start),
        num_workers=workers, worker_init_fn=configure_worker_cleanup if workers else None)


def endless(loader):
    while True:
        yield from loader


def encode_prompts(tokenizer, text_encoder, texts, device, *, max_length=120):
    """Tokenize to a fixed length and return fp32 embeddings + attention masks on CPU."""
    tokens = tokenizer(texts, max_length=max_length, padding='max_length', truncation=True,
                       return_tensors='pt')
    ids, mask = tokens.input_ids.to(device), tokens.attention_mask.to(device)
    with torch.no_grad():
        embeds = text_encoder(ids, attention_mask=mask)[0].float().cpu()
    if not torch.isfinite(embeds).all():
        raise ValueError('Non-finite text embeddings')
    return embeds, mask.cpu()


def save_prompt_cache(path, positive, negative, *, prompt, revision):
    from safetensors.torch import save_file

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({'positive': positive[0].contiguous(), 'positive_mask': positive[1].contiguous(),
               'negative': negative[0].contiguous(), 'negative_mask': negative[1].contiguous()},
              str(path), metadata={'prompt': prompt, 'negative_prompt': '',
                                   'base_revision': revision, 'max_length': '120'})


def load_prompt_cache(path, *, prompt, revision, device='cpu'):
    """Load cached embeddings; refuse a cache made for another prompt or revision."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework='pt') as handle:
        metadata = handle.metadata() or {}
    if metadata.get('prompt') != prompt or metadata.get('base_revision') != revision:
        raise ValueError('Prompt cache does not match the configured prompt/revision')
    return {k: v.to(device) for k, v in load_file(str(path)).items()}


def conditioning_trace(vae, encoding, sample, *, scratch):
    """Record range/encoding at every hop: camera-z -> inverse-depth image -> VAE latent.

    The control image is stored and reloaded as float32 (no 8-bit quantization)."""
    require_camera_z(sample)
    depth, valid = sample['targets']['depth_m'], sample['targets']['depth_valid']
    image = encoding.encode(depth, valid)
    path = Path(scratch)
    torch.save(image, path)
    reloaded = torch.load(path, weights_only=True)
    pixels = vae_pixels(reloaded)
    latent = encode_image(vae, reloaded, control=True)
    again = encode_image(vae, reloaded, control=True)
    stochastic = encode_image(vae, reloaded, control=False, generator=torch.Generator(
        device=vae.device).manual_seed(0))
    near = valid.bool() & (depth > 0)
    # Near-is-bright: encoded intensity must fall as camera-z grows over valid pixels.
    corr = torch.corrcoef(torch.stack([depth[near].float(), image[0][near[0]].float()]))[0, 1]
    return {
        'depth_m_valid': {**tensor_trace(depth[near]), 'meaning': 'camera-z, meters'},
        'valid_fraction': near.float().mean().item(),
        'encoding_constants_m': {'near': encoding.near_m, 'far': encoding.far_m},
        'saturated_near_fraction': (depth[near] <= encoding.near_m).float().mean().item(),
        'saturated_far_fraction': (depth[near] >= encoding.far_m).float().mean().item(),
        'control_image': {**tensor_trace(image), 'channels_identical': bool(
            (image[0] == image[1]).all() and (image[0] == image[2]).all())},
        'depth_vs_intensity_correlation': corr.item(),
        'float32_roundtrip_exact': bool(torch.equal(image, reloaded)),
        'vae_input': tensor_trace(pixels),
        'control_latent_mode_scaled': tensor_trace(latent),
        'mode_is_deterministic': bool(torch.equal(latent, again)),
        'rgb_style_sample_latent': tensor_trace(stochastic),
        'mode_vs_sample_mean_abs_diff': (latent.cpu() - stochastic.cpu()).abs().mean().item(),
    }
