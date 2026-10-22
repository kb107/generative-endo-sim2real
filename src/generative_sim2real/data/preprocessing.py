"""Source-fitted inverse-depth controls, kept separate from metric labels."""
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import torch


def uniform_indices(length: int, cap: int) -> list[int]:
    """Include both endpoints with integer arithmetic and no duplicated indices."""
    if length < 0 or cap < 1:
        raise ValueError("length must be nonnegative and cap positive")
    if length <= cap:
        return list(range(length))
    if cap == 1:
        return [0]
    return [i * (length - 1) // (cap - 1) for i in range(cap)]


@dataclass(frozen=True)
class InverseDepthEncoding:
    near_m: float
    far_m: float
    invalid_fill: float = 0.0

    def __post_init__(self):
        if not (np.isfinite(self.near_m) and np.isfinite(self.far_m)
                and 0 < self.near_m < self.far_m):
            raise ValueError("Expected finite 0 < near_m < far_m")
        if self.invalid_fill != 0:
            raise ValueError("The controller contract requires invalid fill zero")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict):
        return cls(**values)

    def encode(self, depth_m: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if depth_m.ndim != 3 or depth_m.shape[0] != 1 or valid.shape != depth_m.shape:
            raise ValueError("Expected matching [1,H,W] depth and validity")
        good = valid.bool() & torch.isfinite(depth_m) & (depth_m > 0)
        safe = torch.where(good, depth_m.float(), torch.ones_like(depth_m, dtype=torch.float32))
        encoded = ((safe.reciprocal() - 1 / self.far_m)
                   / (1 / self.near_m - 1 / self.far_m)).clamp(0, 1)
        return torch.where(good, encoded, self.invalid_fill).repeat(3, 1, 1)


def valid_depth_pixels(sample: dict, cap: int | None = None) -> np.ndarray:
    depth, valid = sample['targets']['depth_m'], sample['targets']['depth_valid']
    if depth is None or valid is None:
        raise ValueError("Depth statistics require labeled source or audit samples")
    array = depth.detach().cpu().numpy().reshape(-1)
    mask = valid.detach().cpu().numpy().reshape(-1).astype(bool)
    pixels = array[mask & np.isfinite(array) & (array > 0)]
    if cap is not None and len(pixels):
        pixels = pixels[uniform_indices(len(pixels), cap)]
    return pixels


def fit_inverse_depth(
    items: Iterable[tuple[dict, dict]], *, near_quantile: float = .01,
    far_quantile: float = .99, pixels_per_frame: int = 4096, max_frames: int = 128,
) -> tuple[InverseDepthEncoding, dict]:
    """Fit only explicitly identified source_train records, using raw adapter masks."""
    if not 0 <= near_quantile < far_quantile <= 1:
        raise ValueError("Invalid quantiles")
    if max_frames < 1 or pixels_per_frame < 1:
        raise ValueError("Sampling limits must be positive")
    arrays, ids, counts = [], [], []
    for row, sample in items:
        if row.get('split') != 'source_train' or row.get('dataset') != 'simcol3d':
            raise ValueError("Control constants may only be fitted on source_train")
        if len(ids) >= max_frames:
            raise ValueError("Select at most max_frames deterministically before fitting")
        if sample['sample_id'] != row['sample_id']:
            raise ValueError("Fit sample identity mismatch")
        values = valid_depth_pixels(sample, pixels_per_frame)
        ids.append(row['sample_id'])
        counts.append(len(values))
        if len(values):
            arrays.append(values)
    if not arrays:
        raise ValueError("No valid positive source depth pixels")
    near, far = np.quantile(np.concatenate(arrays), [near_quantile, far_quantile])
    encoding = InverseDepthEncoding(float(near), float(far))
    return encoding, {
        'constants': encoding.to_dict(), 'fit_sample_ids': ids,
        'pixel_counts': counts, 'near_quantile': near_quantile,
        'far_quantile': far_quantile, 'max_frames': max_frames,
        'pixels_per_frame': pixels_per_frame,
        'pixel_sampling': 'uniform integer endpoints of finite-positive raw-valid raster pixels',
        'frame_sampling': 'uniform integer endpoints of ordered source_train manifest',
        'formula': 'clip((1/d-1/far)/(1/near-1/far),0,1)',
        'polarity': 'near_is_bright', 'channels': 3, 'dtype': 'float32',
        'metric_labels_modified': False,
    }


def depth_audit(samples: Iterable[dict], encoding: InverseDepthEncoding) -> dict:
    total = good_count = below = above = raw_saturated = frames = 0
    for sample in samples:
        depth = sample['targets']['depth_m']
        if depth is None:
            raise ValueError("RGB-only appearance depth statistics are unavailable")
        values = valid_depth_pixels(sample)
        frames += 1
        total += depth.numel()
        good_count += len(values)
        below += int((values < encoding.near_m).sum())
        above += int((values > encoding.far_m).sum())
        raw_saturated += sample['meta'].get('depth_saturated_pixels', 0)
    return {'frames': frames, 'pixels': total, 'valid_pixels': good_count,
            'invalid_fraction': 1 - good_count / total if total else None,
            'below_near_fraction_of_valid': below / good_count if good_count else None,
            'above_far_fraction_of_valid': above / good_count if good_count else None,
            'adapter_view_raw_saturated_pixels': raw_saturated}
