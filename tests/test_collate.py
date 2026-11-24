from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader

from generative_sim2real.data import ResizeSample, collate_frame_samples
from generative_sim2real.data.types import FrameSample


def _sample(index: int) -> FrameSample:
    return {
        "sample_id": f"example/{index:04d}",
        "rgb": torch.full((3, 2, 3), index / 10.0),
        "camera": {
            "K": torch.tensor(
                [[10.0, 0.0, 1.0], [0.0, 20.0, 0.5], [0.0, 0.0, 1.0]]
            ),
            "T_world_from_camera": torch.eye(4),
            "distortion": None,
            "model": "pinhole",
        },
        "targets": {
            "depth_m": torch.full((1, 2, 3), float(index + 1)),
            "depth_valid": torch.ones((1, 2, 3), dtype=torch.bool),
            "sparse_depth": None,
            "instances": None,
        },
        "meta": {
            "dataset": "fixture",
            "domain": "synthetic",
            "split": "train",
            "scene_id": "scene",
            "sequence_id": "sequence",
            "frame_index": index,
            "timestamp_s": None,
            "original_hw": (2, 3),
            "depth_type": "camera_z",
        },
    }


def test_collates_complete_samples() -> None:
    batch = collate_frame_samples([_sample(0), _sample(1)])

    assert batch["sample_id"] == ["example/0000", "example/0001"]
    assert batch["rgb"].shape == (2, 3, 2, 3)
    assert isinstance(batch["camera"]["K"], torch.Tensor)
    assert batch["camera"]["K"].shape == (2, 3, 3)
    assert isinstance(batch["camera"]["T_world_from_camera"], torch.Tensor)
    assert batch["camera"]["T_world_from_camera"].shape == (2, 4, 4)
    assert batch["camera"]["distortion"] is None
    assert batch["camera"]["model"] == ["pinhole", "pinhole"]
    assert isinstance(batch["targets"]["depth_m"], torch.Tensor)
    assert batch["targets"]["depth_m"].shape == (2, 1, 2, 3)
    assert isinstance(batch["targets"]["depth_valid"], torch.Tensor)
    assert batch["targets"]["depth_valid"].dtype == torch.bool
    assert batch["targets"]["sparse_depth"] is None
    assert batch["targets"]["instances"] is None
    assert [meta["frame_index"] for meta in batch["meta"]] == [0, 1]


def test_preserves_mixed_optional_values_as_lists() -> None:
    first = _sample(0)
    second = _sample(1)
    second["camera"]["K"] = None
    second["targets"]["depth_m"] = None
    second["targets"]["depth_valid"] = None
    second["targets"]["instances"] = []

    batch = collate_frame_samples([first, second])

    assert isinstance(batch["camera"]["K"], list)
    assert batch["camera"]["K"][0] is first["camera"]["K"]
    assert batch["camera"]["K"][1] is None
    assert isinstance(batch["targets"]["depth_m"], list)
    assert batch["targets"]["depth_m"][1] is None
    assert batch["targets"]["instances"] == [None, []]


def test_rejects_mismatched_rgb_shapes() -> None:
    second = _sample(1)
    second["rgb"] = torch.zeros(3, 4, 4)

    with pytest.raises(ValueError, match="matching shapes"):
        collate_frame_samples([_sample(0), second])


def test_preserves_variable_sparse_depth_records() -> None:
    first = _sample(0)
    second = _sample(1)
    first["targets"]["sparse_depth"] = {
        "xy": torch.tensor([[1.0, 1.0]]),
        "depth_m": torch.tensor([0.02]),
        "valid": torch.tensor([True]),
    }
    second["targets"]["sparse_depth"] = {
        "xy": torch.tensor([[0.0, 0.0], [2.0, 1.0]]),
        "depth_m": torch.tensor([0.01, 0.03]),
        "valid": torch.tensor([True, True]),
    }

    batch = collate_frame_samples([first, second])

    sparse = batch["targets"]["sparse_depth"]
    assert sparse is not None
    assert sparse[0] is not None and sparse[0]["xy"].shape == (1, 2)
    assert sparse[1] is not None and sparse[1]["xy"].shape == (2, 2)


def test_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_frame_samples([])


def test_dataloader_keeps_order_and_shapes() -> None:
    resize = ResizeSample((4, 6))
    samples = [resize(deepcopy(_sample(index))) for index in range(4)]
    loader = DataLoader(
        samples,
        batch_size=2,
        num_workers=0,
        collate_fn=collate_frame_samples,
    )

    batches = list(loader)

    assert [sample_id for batch in batches for sample_id in batch["sample_id"]] == [
        "example/0000",
        "example/0001",
        "example/0002",
        "example/0003",
    ]
    assert all(batch["rgb"].shape == (2, 3, 4, 6) for batch in batches)
    assert all(
        isinstance(batch["targets"]["depth_m"], torch.Tensor)
        and batch["targets"]["depth_m"].shape == (2, 1, 4, 6)
        for batch in batches
    )
