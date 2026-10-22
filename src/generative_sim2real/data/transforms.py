"""Paired CPU transforms with explicit geometry history."""

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .types import FrameSample


def _validate_sample(sample: FrameSample) -> tuple[int, int]:
    rgb = sample["rgb"]
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected RGB shaped [3, H, W], got {tuple(rgb.shape)}")
    if rgb.device.type != "cpu":
        raise ValueError("Paired transforms require a CPU dataset sample")
    height, width = rgb.shape[-2:]
    targets = sample["targets"]
    depth, valid = targets["depth_m"], targets["depth_valid"]
    if (depth is None) != (valid is None):
        raise ValueError("depth_m and depth_valid must both be present or both be None")
    if depth is not None and (
        depth.shape != (1, height, width) or valid.shape != depth.shape
    ):
        raise ValueError("Expected matching depth and mask shaped [1, H, W]")
    if targets.get("sparse_depth") is not None or targets.get("instances"):
        raise ValueError("Sparse depth and instance spatial transforms are not supported")
    K = sample["camera"]["K"]
    if K is not None and K.shape != (3, 3):
        raise ValueError(f"Expected K shaped [3, 3], got {tuple(K.shape)}")
    return height, width


def _result(
    sample: FrameSample,
    rgb: torch.Tensor,
    depth: torch.Tensor | None,
    valid: torch.Tensor | None,
    K: torch.Tensor | None,
    history: dict[str, object],
) -> FrameSample:
    meta = sample["meta"].copy()
    meta["current_hw"] = tuple(rgb.shape[-2:])
    meta["transform_history"] = [*meta.get("transform_history", []), history]
    return {
        **sample,
        "rgb": rgb,
        "camera": {**sample["camera"], "K": K},
        "targets": {**sample["targets"], "depth_m": depth, "depth_valid": valid},
        "meta": meta,
    }


@dataclass(frozen=True)
class ResizeSample:
    """Resize RGB cubic, dense labels nearest, and scale pixel intrinsics."""

    size_hw: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.size_hw) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.size_hw
        ):
            raise ValueError(
                f"Resize dimensions must be positive integers, got {self.size_hw}"
            )

    def __call__(self, sample: FrameSample) -> FrameSample:
        old_height, old_width = _validate_sample(sample)
        new_height, new_width = self.size_hw
        targets = sample["targets"]
        depth, valid = targets["depth_m"], targets["depth_valid"]
        K = sample["camera"]["K"]
        resized_K = None if K is None else K.clone()
        if resized_K is not None:
            resized_K[0] *= new_width / old_width
            resized_K[1] *= new_height / old_height
        resized_rgb = _resize_chw(sample["rgb"], self.size_hw, cv2.INTER_CUBIC)
        resized_depth = (
            None if depth is None
            else _resize_chw(depth, self.size_hw, cv2.INTER_NEAREST)
        )
        resized_valid = (
            None if valid is None
            else _resize_chw(
                valid.to(torch.uint8), self.size_hw, cv2.INTER_NEAREST
            ).bool()
        )
        return _result(
            sample,
            resized_rgb.clamp_(0, 1),
            resized_depth,
            resized_valid,
            resized_K,
            {
                "name": "resize",
                "input_hw": (old_height, old_width),
                "output_hw": self.size_hw,
                "rgb_interpolation": "cubic",
                "label_interpolation": "nearest",
            },
        )


@dataclass(frozen=True)
class CenterCropSquare:
    """Center-crop paired fields; shift K by the integer crop origin."""

    def __call__(self, sample: FrameSample) -> FrameSample:
        height, width = _validate_sample(sample)
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2

        def crop(value: torch.Tensor | None) -> torch.Tensor | None:
            if value is None:
                return None
            return value[:, top:top + side, left:left + side].clone()

        K = sample["camera"]["K"]
        cropped_K = None if K is None else K.clone()
        if cropped_K is not None:
            cropped_K[0, 2] -= left
            cropped_K[1, 2] -= top
        return _result(
            sample,
            crop(sample["rgb"]),
            crop(sample["targets"]["depth_m"]),
            crop(sample["targets"]["depth_valid"]),
            cropped_K,
            {
                "name": "center_crop_square",
                "input_hw": (height, width),
                "output_hw": (side, side),
                "top": top,
                "left": left,
            },
        )


def _resize_chw(
    tensor: torch.Tensor, size_hw: tuple[int, int], interpolation: int
) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.device.type != "cpu":
        raise ValueError(f"Expected a CPU CHW tensor, got shape={tuple(tensor.shape)}")
    height, width = size_hw
    array = np.moveaxis(tensor.detach().numpy(), 0, -1)
    resized = cv2.resize(array, (width, height), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[..., None]
    return torch.from_numpy(
        np.ascontiguousarray(np.moveaxis(resized, -1, 0))
    ).to(dtype=tensor.dtype)
