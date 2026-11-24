import pytest
import torch

from generative_sim2real.data import ResizeSample
from generative_sim2real.data.types import FrameSample


def _sample() -> FrameSample:
    rgb = torch.linspace(0.0, 1.0, 18, dtype=torch.float32).reshape(3, 2, 3)
    depth = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    valid = torch.tensor([[[True, True, False], [True, False, True]]])
    return {
        "sample_id": "example/0000",
        "rgb": rgb,
        "camera": {
            "K": torch.tensor(
                [[10.0, 0.0, 1.0], [0.0, 20.0, 0.5], [0.0, 0.0, 1.0]]
            ),
            "T_world_from_camera": torch.eye(4),
            "distortion": None,
            "model": "pinhole",
        },
        "targets": {
            "depth_m": depth,
            "depth_valid": valid,
            "sparse_depth": None,
            "instances": None,
        },
        "meta": {
            "dataset": "fixture",
            "domain": "synthetic",
            "split": "train",
            "scene_id": "scene",
            "sequence_id": "sequence",
            "frame_index": 0,
            "timestamp_s": None,
            "original_hw": (2, 3),
            "depth_type": "camera_z",
        },
    }


def test_resizes_spatial_fields_and_pixel_intrinsics() -> None:
    sample = _sample()
    original_K = sample["camera"]["K"].clone()
    resized = ResizeSample((4, 9))(sample)

    assert resized["rgb"].shape == (3, 4, 9)
    assert resized["rgb"].dtype == torch.float32
    assert 0.0 <= float(resized["rgb"].min()) <= float(resized["rgb"].max()) <= 1.0

    depth = resized["targets"]["depth_m"]
    valid = resized["targets"]["depth_valid"]
    assert depth is not None and valid is not None
    assert depth.shape == valid.shape == (1, 4, 9)
    assert depth.dtype == torch.float32
    assert valid.dtype == torch.bool
    assert depth[0, 0, 0] == 1.0
    assert depth[0, 0, 3] == 2.0
    assert depth[0, 2, 0] == 4.0
    assert not valid[0, 0, 6]

    K = resized["camera"]["K"]
    assert K is not None
    torch.testing.assert_close(
        K,
        torch.tensor([[30.0, 0.0, 3.0], [0.0, 40.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(sample["camera"]["K"], original_K)
    assert sample["rgb"].shape == (3, 2, 3)
    assert resized["meta"]["original_hw"] == (2, 3)
    torch.testing.assert_close(
        resized["camera"]["T_world_from_camera"],
        sample["camera"]["T_world_from_camera"],
    )


def test_preserves_missing_geometry_and_targets() -> None:
    sample = _sample()
    sample["camera"]["K"] = None
    sample["targets"]["depth_m"] = None
    sample["targets"]["depth_valid"] = None
    sample["targets"]["instances"] = []

    resized = ResizeSample((4, 6))(sample)

    assert resized["camera"]["K"] is None
    assert resized["targets"]["depth_m"] is None
    assert resized["targets"]["depth_valid"] is None
    assert resized["targets"]["instances"] == []


def test_rejects_incomplete_depth_target() -> None:
    sample = _sample()
    sample["targets"]["depth_valid"] = None

    with pytest.raises(ValueError, match="both be present"):
        ResizeSample((4, 6))(sample)


def test_rejects_unimplemented_spatial_annotations() -> None:
    sample = _sample()
    sample["targets"]["instances"] = [{"instance_id": "unsupported"}]
    with pytest.raises(ValueError, match="not supported"):
        ResizeSample((4, 6))(sample)


@pytest.mark.parametrize("size", [(0, 4), (4, 0), (-1, 4)])
def test_rejects_invalid_resize_size(size) -> None:
    with pytest.raises(ValueError, match="positive"):
        ResizeSample(size)


def test_center_crop_square_pairs_labels_k_and_history():
    from generative_sim2real.data import CenterCropSquare
    sample = _sample()
    cropped = CenterCropSquare()(sample)
    torch.testing.assert_close(cropped["rgb"], sample["rgb"][:, :, :2])
    torch.testing.assert_close(
        cropped["targets"]["depth_m"], sample["targets"]["depth_m"][:, :, :2]
    )
    assert cropped["meta"]["transform_history"] == [{
        "name": "center_crop_square", "input_hw": (2, 3), "output_hw": (2, 2),
        "top": 0, "left": 0}]
    assert "transform_history" not in sample["meta"]
    resized = ResizeSample((4, 4))(cropped)
    assert [h["name"] for h in resized["meta"]["transform_history"]] == [
        "center_crop_square", "resize"
    ]
    assert resized["meta"]["current_hw"] == (4, 4)
    assert cropped["meta"]["current_hw"] == (2, 2)
    torch.testing.assert_close(resized["camera"]["K"][:2], sample["camera"]["K"][:2] * 2)


def test_center_crop_nonzero_origin_and_missing_targets():
    from generative_sim2real.data import CenterCropSquare
    sample = _sample()
    sample["rgb"] = torch.arange(36, dtype=torch.float32).reshape(3, 2, 6) / 36
    sample["targets"]["depth_m"] = None
    sample["targets"]["depth_valid"] = None
    cropped = CenterCropSquare()(sample)
    torch.testing.assert_close(cropped["rgb"], sample["rgb"][:, :, 2:4])
    assert cropped["camera"]["K"][0, 2] == -1
    assert cropped["targets"]["depth_m"] is None
    assert sample["camera"]["K"][0, 2] == 1


def test_resize_matches_reference_cubic_and_nearest():
    import cv2
    import numpy as np
    sample = _sample()
    resized = ResizeSample((5, 7))(sample)
    expected_rgb = cv2.resize(
        np.moveaxis(sample["rgb"].numpy(), 0, -1), (7, 5), interpolation=cv2.INTER_CUBIC
    )
    np.testing.assert_array_equal(
        resized["rgb"].numpy(), np.moveaxis(np.clip(expected_rgb, 0, 1), -1, 0)
    )
    expected_depth = cv2.resize(
        sample["targets"]["depth_m"].numpy()[0], (7, 5), interpolation=cv2.INTER_NEAREST
    )
    np.testing.assert_array_equal(resized["targets"]["depth_m"].numpy()[0], expected_depth)


@pytest.mark.parametrize("height,width", [(3, 7), (7, 3)])
def test_nonzero_crop_origin_keeps_dense_geometry_paired(height, width):
    from generative_sim2real.data import CenterCropSquare

    sample = _sample()
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    label = torch.arange(height * width, dtype=torch.float32).reshape(1, height, width)
    sample["rgb"] = label.repeat(3, 1, 1) / (height * width)
    sample["targets"]["depth_m"] = label
    sample["targets"]["depth_valid"] = label.remainder(2).bool()
    before_K = sample["camera"]["K"].clone()
    cropped = CenterCropSquare()(sample)
    rows, columns = slice(top, top + side), slice(left, left + side)
    torch.testing.assert_close(
        cropped["targets"]["depth_m"], label[:, rows, columns]
    )
    assert torch.equal(
        cropped["targets"]["depth_valid"], label[:, rows, columns].remainder(2).bool()
    )
    torch.testing.assert_close(
        cropped["rgb"][0], cropped["targets"]["depth_m"][0] / (height * width)
    )
    assert cropped["camera"]["K"][0, 2] == before_K[0, 2] - left
    assert cropped["camera"]["K"][1, 2] == before_K[1, 2] - top
