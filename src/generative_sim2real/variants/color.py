"""Pure image functions for the attribution controls: Reinhard Lab transfer and strong photometric jitter.

All images are float32 ``[3,H,W]`` tensors in ``[0,1]`` (the project RGB convention); outputs are clipped to
``[0,1]``. Functions are deterministic given their arguments; randomness enters only through a numpy Generator.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

LUMA = np.array([.299, .587, .114], dtype=np.float32)
# C1 frames can carry a black lens border (and SimCol frames dark corners); statistics skip pixels whose mean
# channel value is below this threshold.
BORDER_LUMINANCE = .03
MIN_MASK_FRACTION = .25

# One fixed draw per ID from these ranges.
STRONG_RANGES = {'brightness': (.6, 1.4), 'contrast': (.7, 1.3), 'gamma': (.7, 1.4), 'saturation': (.6, 1.4),
                 'hue_shift': (-.05, .05), 'channel_gain': (.85, 1.15), 'blur_sigma': (0., 1.5),
                 'noise_sigma': (0., .03)}
BLUR_PROBABILITY = .5


def _hwc(rgb):
    if not isinstance(rgb, torch.Tensor) or rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != torch.float32:
        raise ValueError('Expected float32 RGB tensor [3,H,W]')
    if not torch.isfinite(rgb).all():
        raise ValueError('RGB must be finite')
    return np.ascontiguousarray(rgb.detach().cpu().numpy().transpose(1, 2, 0))


def _chw(array):
    return torch.from_numpy(np.ascontiguousarray(np.clip(array, 0, 1).astype(np.float32).transpose(2, 0, 1)))


def lab_of(rgb):
    """CIE-Lab (OpenCV float convention: L in [0,100], a/b about [-127,127]) as float32 HWC."""
    return cv2.cvtColor(np.clip(_hwc(rgb), 0, 1), cv2.COLOR_RGB2LAB)


def luminance_mask(rgb, valid=None, threshold=BORDER_LUMINANCE):
    """[H,W] bool: not-black pixels (and inside ``valid`` when given). Refuses a mostly-masked frame."""
    mask = (rgb.mean(0) > threshold)
    if valid is not None:
        valid = valid.reshape(mask.shape).bool()
        mask = mask & valid
    if mask.float().mean() < MIN_MASK_FRACTION:
        raise ValueError(f'Usable pixel fraction {mask.float().mean():.3f} < {MIN_MASK_FRACTION}')
    return mask.numpy()


def lab_stats(lab, mask):
    pixels = lab[mask]
    return pixels.mean(0), pixels.std(0)


def reinhard_lab(source, reference, source_mask, reference_mask, eps=1e-3):
    """Reinhard et al. colour transfer: per-channel mean/std of ``source`` Lab matched to ``reference`` Lab.

    Statistics use only the masked (non-border/valid) pixels; the affine map is applied to every source pixel,
    then converted back and clipped to [0,1]."""
    source_lab, reference_lab = lab_of(source), lab_of(reference)
    mu_s, sd_s = lab_stats(source_lab, source_mask)
    mu_r, sd_r = lab_stats(reference_lab, reference_mask)
    out = (source_lab - mu_s) * (sd_r / np.maximum(sd_s, eps)) + mu_r
    # Out-of-gamut Lab values are clipped to the valid Lab box before the inverse conversion.
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    out[..., 1:] = np.clip(out[..., 1:], -127, 127)
    return _chw(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB))


def fda_transfer(source, reference, beta):
    """Fourier Domain Adaptation (Yang & Soatto, CVPR 2020, https://arxiv.org/abs/2004.05498).

    Per channel, the centred low-frequency amplitude box of half-size floor(min(H, W) * beta) of ``source`` is replaced
    by the reference's; the source phase is kept. Inverse FFT, real part, clip to [0,1]. Same index convention as the
    authors' ``low_freq_mutate_np`` (box [c - b, c + b] inclusive after fftshift)."""
    if not 0 <= beta < .5:
        raise ValueError('FDA beta must be in [0, 0.5)')
    src, ref = _hwc(source).astype(np.float64), _hwc(reference).astype(np.float64)
    if src.shape != ref.shape:
        raise ValueError('FDA needs source and reference of the same shape')
    spectrum = np.fft.fft2(src, axes=(0, 1))
    amplitude = np.fft.fftshift(np.abs(spectrum), axes=(0, 1))
    reference_amplitude = np.fft.fftshift(np.abs(np.fft.fft2(ref, axes=(0, 1))), axes=(0, 1))
    h, w = src.shape[:2]
    b, ch, cw = int(np.floor(min(h, w) * beta)), h // 2, w // 2
    amplitude[ch - b:ch + b + 1, cw - b:cw + b + 1] = reference_amplitude[ch - b:ch + b + 1, cw - b:cw + b + 1]
    amplitude = np.fft.ifftshift(amplitude, axes=(0, 1))
    out = np.real(np.fft.ifft2(amplitude * np.exp(1j * np.angle(spectrum)), axes=(0, 1)))
    return _chw(np.clip(out, 0, 1).astype(np.float32))


def draw_strong_params(rng):
    """One fixed draw (fixed consumption order) for strong generic photometric augmentation."""
    r = STRONG_RANGES
    params = {'brightness': rng.uniform(*r['brightness']), 'contrast': rng.uniform(*r['contrast']),
              'gamma': rng.uniform(*r['gamma']), 'saturation': rng.uniform(*r['saturation']),
              'hue_shift': rng.uniform(*r['hue_shift']),
              'channel_gain': rng.uniform(*r['channel_gain'], size=3).tolist(),
              'blur_sigma': rng.uniform(*r['blur_sigma']), 'blur_applied': bool(rng.random() < BLUR_PROBABILITY),
              'noise_sigma': rng.uniform(*r['noise_sigma']), 'noise_seed': int(rng.integers(0, 2 ** 62))}
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in params.items()}


IDENTITY_PARAMS = {'brightness': 1., 'contrast': 1., 'gamma': 1., 'saturation': 1., 'hue_shift': 0.,
                   'channel_gain': [1., 1., 1.], 'blur_sigma': 0., 'blur_applied': False, 'noise_sigma': 0.,
                   'noise_seed': 0}


def apply_strong_photometric(rgb, params):
    """brightness -> contrast (about the image mean) -> gamma -> saturation -> hue -> channel gain -> blur -> noise,
    clipping to [0,1] after every step that can leave the range."""
    x = _hwc(rgb)
    x = np.clip(x * np.float32(params['brightness']), 0, 1)
    mean = x.mean()
    x = np.clip((x - mean) * np.float32(params['contrast']) + mean, 0, 1)
    x = np.power(x, np.float32(params['gamma']))
    gray = (x * LUMA).sum(-1, keepdims=True)
    x = np.clip(gray + (x - gray) * np.float32(params['saturation']), 0, 1)
    if params['hue_shift']:
        hsv = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2HSV)  # H in degrees [0,360)
        hsv[..., 0] = (hsv[..., 0] + 360. * params['hue_shift']) % 360.
        x = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1)
    x = np.clip(x * np.asarray(params['channel_gain'], dtype=np.float32), 0, 1)
    if params['blur_applied'] and params['blur_sigma'] > 0:
        x = cv2.GaussianBlur(x, (0, 0), float(params['blur_sigma']), borderType=cv2.BORDER_REFLECT)
    if params['noise_sigma'] > 0:
        noise = np.random.default_rng(params['noise_seed']).standard_normal(x.shape).astype(np.float32)
        x = x + noise * np.float32(params['noise_sigma'])
    return _chw(x)
