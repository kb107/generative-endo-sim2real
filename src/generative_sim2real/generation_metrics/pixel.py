"""Per-image pixel descriptors and set-level summaries (diversity, HSV saturation, brightness)."""
import torch
import torch.nn.functional as F

from .distance import pairwise_l1

DIVERSITY_SIZE = 64
W1_STRIDE = 8        # matches configs/p4.yaml appearance_review.w1.pixel_stride
PIXEL_SIZE = 512     # native generation size; other sizes are resized to this


def hsv_saturation(images):
    """HSV saturation (max-min)/max of RGB in [0,1]; 0 where max is 0. [B,3,H,W] -> [B,H,W]."""
    high, low = images.max(1).values, images.min(1).values
    return torch.where(high > 0, (high - low) / high.clamp_min(1e-12), torch.zeros_like(high))


def to_size(images, size):
    if images.shape[-2:] == (size, size):
        return images
    return F.interpolate(images, size=(size, size), mode='bilinear', antialias=True,
                         align_corners=False).clamp(0, 1)


def pixel_descriptors(images):
    """Per-image descriptors from [B,3,H,W] float images in [0,1] (resized to PIXEL_SIZE)."""
    images = to_size(images.float(), PIXEL_SIZE)
    saturation = hsv_saturation(images).flatten(1)
    return {
        'pix64': F.adaptive_avg_pool2d(images, DIVERSITY_SIZE).flatten(1),
        'strided': images[..., ::W1_STRIDE, ::W1_STRIDE].contiguous(),
        'sat_mean': saturation.mean(1),
        'sat_p95': torch.quantile(saturation, 0.95, dim=1),
        'brightness': images.mean((1, 2, 3)),
    }


def evenly_spaced(count, wanted):
    """Sorted unique indices evenly spaced over range(count) (same rule as train.subset_indices)."""
    step = (count - 1) / (wanted - 1) if wanted > 1 else 0
    return sorted({round(i * step) for i in range(min(wanted, count))})


def summarize_pixels(desc, index=None):
    """Set-level pixel statistics for the images `index` (default all) of a descriptor dict."""
    take = (lambda t: t) if index is None else (lambda t: t[index])
    pix64 = take(desc['pix64'])
    out = {'n': len(pix64), 'diversity_l1_64': pairwise_l1(pix64)}
    if len(pix64) > 64:
        out['diversity_l1_64_first_even_64'] = pairwise_l1(pix64[evenly_spaced(len(pix64), 64)])
    for key in ('sat_mean', 'sat_p95', 'brightness'):
        values = take(desc[key])
        out[key] = values.mean().item()
        out[f'{key}_std_over_images'] = values.std().item() if len(values) > 1 else 0.0
    return out
