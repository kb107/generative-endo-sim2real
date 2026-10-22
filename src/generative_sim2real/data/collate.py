"""Explicit collation for shared frame samples."""

import torch
from torch import Tensor

from .types import FrameBatch, FrameSample, OptionalTensorBatch


def collate_frame_samples(samples: list[FrameSample]) -> FrameBatch:
    """Stack compatible tensors without inventing values for missing fields."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    try:
        rgb = torch.stack([sample["rgb"] for sample in samples])
    except RuntimeError as error:
        shapes = [tuple(sample["rgb"].shape) for sample in samples]
        raise ValueError(
            f"RGB tensors must have matching shapes before collation, got {shapes}"
        ) from error

    instance_values = [sample["targets"]["instances"] for sample in samples]
    instances = (
        None if all(value is None for value in instance_values) else instance_values
    )
    sparse_values = [sample["targets"].get("sparse_depth") for sample in samples]
    sparse_depth = (
        None if all(value is None for value in sparse_values) else sparse_values
    )

    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "rgb": rgb,
        "camera": {
            "K": _collate_optional_tensors(
                [sample["camera"]["K"] for sample in samples]
            ),
            "T_world_from_camera": _collate_optional_tensors(
                [sample["camera"]["T_world_from_camera"] for sample in samples]
            ),
            "distortion": _collate_optional_tensors(
                [sample["camera"]["distortion"] for sample in samples]
            ),
            "model": [sample["camera"]["model"] for sample in samples],
        },
        "targets": {
            "depth_m": _collate_optional_tensors(
                [sample["targets"]["depth_m"] for sample in samples]
            ),
            "depth_valid": _collate_optional_tensors(
                [sample["targets"]["depth_valid"] for sample in samples]
            ),
            "sparse_depth": sparse_depth,
            "instances": instances,
        },
        "meta": [sample["meta"].copy() for sample in samples],
    }


def _collate_optional_tensors(values: list[Tensor | None]) -> OptionalTensorBatch:
    present = [value for value in values if value is not None]
    if not present:
        return None

    first = present[0]
    compatible = len(present) == len(values) and all(
        value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        for value in present[1:]
    )
    return torch.stack(present) if compatible else list(values)
