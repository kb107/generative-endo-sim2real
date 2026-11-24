import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook
from PIL import Image
import pytest
import torch

from generative_sim2real.data import c3vdv2 as m
from generative_sim2real.data.c3vdv2 import (
    C3VDv2,
    _OCAM_CENTER,
    _OCAM_STRETCH,
    _build_pinhole_remap,
    _center_crop_square_remap,
    _load_rectified_depth,
    _ocam_z,
)


@pytest.fixture
def c3vdv2_root(tmp_path: Path) -> Path:
    root = tmp_path / "C3VDv2"
    frames = root / "frames" / "C1"
    frames.mkdir(parents=True)
    sequences = [
        ("c1_ascending_t1_v1", 1, 3, False),
        ("c1_ascending_t1_v2", 2, 2, True),
        ("c1_ascending_t1_v3", 3, 1, False),
    ]
    for sequence_id, _, frame_count, nested in sequences:
        sequence_dir = frames / sequence_id
        payload = sequence_dir / sequence_id if nested else sequence_dir
        (payload / "rgb").mkdir(parents=True)
        (payload / "depth").mkdir()
        pose_rows = []
        for frame_index in range(frame_count):
            rgb = np.full((1080, 1350, 3), 64 + frame_index, dtype=np.uint8)
            assert cv2.imwrite(str(payload / "rgb" / f"{frame_index:04d}.png"), rgb)
            depth = np.full((1080, 1350), 16_384, dtype=np.uint16)
            Image.fromarray(depth).save(
                payload / "depth" / f"{frame_index:04d}_depth.tiff"
            )

            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = [100.0 + frame_index, 200.0, 300.0]
            pose_rows.append(",".join(str(value) for value in pose.T.reshape(-1)))
        (payload / "pose.txt").write_text("\n".join(pose_rows) + "\n")

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        [
            "Colon",
            "Segment",
            "Phantom Number",
            "Video Number ",
            "Video Name",
            "Qualitative Score",
            "Total Frames",
        ]
    )
    for sequence_id, quality, frame_count, _ in sequences:
        version = sequence_id.rsplit("_", 1)[-1]
        sheet.append(
            ["c1", "ascending", "t1", version, sequence_id, quality, frame_count]
        )
    workbook.save(root / "metadata.xlsx")
    workbook.close()
    return root


def _dataset(root: Path, **kwargs) -> C3VDv2:
    options = {
        "split": "train",
        "colons": ("c1",),
        "textures": ("t1",),
        "video_versions": ("v1", "v2"),
        "quality_scores": (1, 2),
        "allow_v3_clean_proxy_depth": False,
        "show_progress": False,
    }
    options.update(kwargs)
    return C3VDv2(root, **options)


def test_loads_shared_sample_and_nested_sequence(c3vdv2_root: Path) -> None:
    dataset = _dataset(c3vdv2_root)

    assert dataset.sequence_ids == (
        "c1_ascending_t1_v1",
        "c1_ascending_t1_v2",
    )
    assert len(dataset) == 5
    assert dataset.records[-1].sequence_id == "c1_ascending_t1_v2"

    sample = dataset[0]
    assert sample["sample_id"] == "c3vdv2/c1/c1_ascending_t1_v1/0000"
    assert sample["rgb"].shape == (3, 504, 630)
    assert sample["rgb"].dtype == torch.float32
    assert torch.allclose(sample["rgb"], torch.full_like(sample["rgb"], 64 / 255))

    depth = sample["targets"]["depth_m"]
    valid = sample["targets"]["depth_valid"]
    assert depth is not None and valid is not None
    assert valid.all()
    assert torch.allclose(
        depth,
        torch.full_like(depth, 16_384 / 65_535 * 0.1),
    )
    assert torch.equal(
        sample["camera"]["K"],
        torch.tensor(
            [[300.0, 0.0, 314.5], [0.0, 300.0, 251.5], [0.0, 0.0, 1.0]]
        ),
    )
    pose = sample["camera"]["T_world_from_camera"]
    assert pose is not None
    assert torch.allclose(pose[:3, 3], torch.tensor([0.1, 0.2, 0.3]))
    assert {key: value for key, value in sample["meta"].items()
            if key not in {"current_hw", "transform_history"}} == {
        "dataset": "c3vdv2",
        "domain": "phantom",
        "split": "train",
        "scene_id": "c1",
        "sequence_id": "c1_ascending_t1_v1",
        "frame_index": 0,
        "timestamp_s": None,
        "original_hw": (1080, 1350),
        "depth_type": "camera_z",
        "depth_saturated_pixels": 0,
    }


def test_filters_sequences_and_uses_per_sequence_stride(c3vdv2_root: Path) -> None:
    strided = _dataset(c3vdv2_root, frame_stride=2)
    included = _dataset(
        c3vdv2_root,
        include_sequences=("c1_ascending_t1_v2",),
    )
    excluded = _dataset(
        c3vdv2_root,
        exclude_sequences=("c1_ascending_t1_v2",),
    )

    assert [record.frame_index for record in strided.records] == [0, 2, 0]
    assert included.sequence_ids == ("c1_ascending_t1_v2",)
    assert excluded.sequence_ids == ("c1_ascending_t1_v1",)


def test_requires_explicit_v3_clean_proxy_choice(c3vdv2_root: Path) -> None:
    with pytest.raises(ValueError, match="clean phantom"):
        _dataset(
            c3vdv2_root,
            video_versions=("v3",),
            quality_scores=(3,),
        )

    dataset = _dataset(
        c3vdv2_root,
        video_versions=("v3",),
        quality_scores=(3,),
        allow_v3_clean_proxy_depth=True,
    )
    assert dataset.sequence_ids == ("c1_ascending_t1_v3",)


def test_depth_mask_excludes_zero_and_clamped_max(tmp_path: Path) -> None:
    raw = np.zeros((1080, 1350), dtype=np.uint16)
    raw[0, 0] = 16_384
    raw[0, 1] = 65_535
    raw[1, 1] = 32_768
    path = tmp_path / "depth.tiff"
    Image.fromarray(raw).save(path)
    map_x = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    map_y = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    depth, valid, saturated = _load_rectified_depth(
        path,
        map_x,
        map_y,
        np.ones((2, 2), dtype=bool),
    )

    assert valid.tolist() == [[[True, False], [False, True]]]
    assert saturated == 1
    assert depth[0, 0, 0].item() == pytest.approx(16_384 / 65_535 * 0.1)
    assert depth[0, 1, 1].item() == pytest.approx(32_768 / 65_535 * 0.1)
    assert depth[0, 0, 1].item() == 0.0
    assert depth[0, 1, 0].item() == 0.0


def test_pinhole_remap_round_trips_through_official_camera_model() -> None:
    height, width = 504, 630
    focal = 300.0
    map_x, map_y, inside, K = _build_pinhole_remap((height, width), focal)
    assert inside.all()
    assert K.tolist() == [
        [300.0, 0.0, 314.5],
        [0.0, 300.0, 251.5],
        [0.0, 0.0, 1.0],
    ]

    inverse_stretch = np.linalg.inv(_OCAM_STRETCH)
    points = (
        (0, 0),
        (0, width - 1),
        (height // 2, width // 2),
        (height - 1, width - 1),
    )
    for row, column in points:
        source_offset = np.array(
            [map_x[row, column], map_y[row, column]], dtype=np.float64
        ) - _OCAM_CENTER
        ideal = inverse_stretch @ source_offset
        rho = np.linalg.norm(ideal)
        source_ray = np.array([ideal[0], ideal[1], _ocam_z(np.array(rho))])
        source_ray /= np.linalg.norm(source_ray)

        destination_ray = np.array(
            [(column - 314.5) / focal, (row - 251.5) / focal, 1.0]
        )
        destination_ray /= np.linalg.norm(destination_ray)
        assert np.dot(source_ray, destination_ray) > 1.0 - 1e-10


def test_square_center_crop_keeps_focal_and_shifts_principal_point() -> None:
    map_x, map_y, inside, K = _build_pinhole_remap((504, 630), 300.0)

    cropped_x, cropped_y, cropped_inside, cropped_K = _center_crop_square_remap(
        map_x, map_y, inside, K
    )

    assert cropped_x.shape == (504, 504)
    assert cropped_y.shape == (504, 504)
    assert cropped_inside.shape == (504, 504)
    np.testing.assert_array_equal(cropped_x, map_x[:, 63:567])
    np.testing.assert_array_equal(cropped_y, map_y[:, 63:567])
    assert cropped_K.tolist() == [
        [300.0, 0.0, 251.5],
        [0.0, 300.0, 251.5],
        [0.0, 0.0, 1.0],
    ]


def test_dataset_can_return_square_center_crop(c3vdv2_root: Path) -> None:
    sample = _dataset(c3vdv2_root, center_crop_square=True)[0]
    depth = sample["targets"]["depth_m"]
    valid = sample["targets"]["depth_valid"]

    assert sample["rgb"].shape == (3, 504, 504)
    assert depth is not None and depth.shape == (1, 504, 504)
    assert valid is not None and valid.shape == (1, 504, 504)
    assert sample["camera"]["K"].tolist() == [
        [300.0, 0.0, 251.5],
        [0.0, 300.0, 251.5],
        [0.0, 0.0, 1.0],
    ]


def _rgb_dataset(root, **kwargs):
    from generative_sim2real.data import C3VDv2RGB
    options = dict(split="train", colons=("c1",), textures=("t1",),
                   video_versions=("v1", "v2"), quality_scores=(1, 2),
                   allow_v3_clean_proxy_depth=False, show_progress=False)
    options.update(kwargs)
    return C3VDv2RGB(root, **options)


def test_rgb_only_never_constructs_discovers_or_opens_labels(c3vdv2_root, monkeypatch):
    """Deny label path construction, not only decoding, in direct/nested payloads."""
    import shutil
    labeled = _dataset(c3vdv2_root)
    expected_rgb = labeled[0]["rgb"]
    for path in c3vdv2_root.rglob("depth"):
        shutil.rmtree(path)
    for path in c3vdv2_root.rglob("pose.txt"):
        path.unlink()
    divide = Path.__truediv__
    def guarded_divide(path, key):
        if str(key) == "depth" or "pose" in str(key).lower() or "_depth" in str(key):
            raise AssertionError(f"RGB loader constructed label path: {key}")
        return divide(path, key)
    monkeypatch.setattr(Path, "__truediv__", guarded_divide)
    from generative_sim2real.data import c3vdv2 as module
    def forbidden(*args, **kwargs):
        raise AssertionError("RGB loader read labels")
    monkeypatch.setattr(module, "_load_poses", forbidden)
    monkeypatch.setattr(module, "_load_rectified_depth", forbidden)
    monkeypatch.setattr(module.Image, "open", forbidden)
    dataset = _rgb_dataset(c3vdv2_root, frame_stride=2)
    assert [r.frame_index for r in dataset.records] == [0, 2, 0]
    assert all(not hasattr(r, "depth_path") for r in dataset.records)
    sample = dataset[0]
    torch.testing.assert_close(sample["rgb"], expected_rgb, rtol=0, atol=0)
    assert sample["targets"]["depth_m"] is None
    assert sample["targets"]["depth_valid"] is None
    assert sample["camera"]["T_world_from_camera"] is None
    assert sample["meta"]["depth_type"] is None
    assert dataset[-1]["rgb"].shape == (3, 504, 630)


def test_rgb_only_refuses_c2_before_data_access(tmp_path):
    with pytest.raises(ValueError, match="only colons"):
        _rgb_dataset(tmp_path / "unavailable", colons=("c2",))


def test_logical_frames_symlink_paths_are_preserved(c3vdv2_root, tmp_path):
    root = tmp_path / "logical-view"
    root.mkdir()
    (root / "frames").symlink_to(c3vdv2_root / "frames", target_is_directory=True)
    (root / "metadata.xlsx").symlink_to(c3vdv2_root / "metadata.xlsx")
    for dataset in (_dataset(root), _rgb_dataset(root)):
        assert dataset.root == root
        assert all(record.rgb_path.is_relative_to(root / "frames") for record in dataset.records)
        assert dataset[0]["rgb"].shape == (3, 504, 630)


@pytest.mark.parametrize("stride", [True, 0, -1, 1.5])
def test_rejects_invalid_stride(c3vdv2_root, stride):
    with pytest.raises(ValueError, match="frame_stride"):
        _rgb_dataset(c3vdv2_root, frame_stride=stride)


def test_rejects_missing_labeled_pair(c3vdv2_root):
    (c3vdv2_root / "frames/C1/c1_ascending_t1_v1/depth/0000_depth.tiff").unlink()
    with pytest.raises(ValueError, match="Unpaired"):
        _dataset(c3vdv2_root)


def test_index_bounds(c3vdv2_root):
    dataset = _rgb_dataset(c3vdv2_root)
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-len(dataset)-1]


def digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# Recorded with the adapter before the helpers were factored out (504x630, f = 300).
MAP_X = "240c48ba53e60b36a160bcc1da95859ca436079d76dfd0f19d05eaa8b8938d91"
MAP_Y = "c7f540b89cc65ab6b901cd1f7b7975907b8357659132838c81b3ec71e67e8500"
INSIDE = "30a9ffc80a0c84fb1e62b796d8757a849d81a2e4875c42a0003992af14609766"
REAL_RGB = "4783994dabeaa8a507cfc66dae707446f7c74f82c9ec7d25627652361472b530"
REAL_DEPTH = "10c867865307f7078c0f2bf57f0447031c99232b317d0fbd876af88e41940b64"
REAL_VALID = "d98e417205129110ef3d59114cbe80994461de8eb5dbec456ea00be5bd654963"


def test_remap_tables_unchanged():
    map_x, map_y, inside, K = m._build_pinhole_remap((504, 630), 300.0)
    assert digest(map_x) == MAP_X and digest(map_y) == MAP_Y and digest(inside) == INSIDE
    assert K.tolist() == [[300.0, 0.0, 314.5], [0.0, 300.0, 251.5], [0.0, 0.0, 1.0]]


def test_ideal_coordinates_reproduce_the_source_maps():
    ideal = m.pinhole_ideal_coordinates((504, 630), 300.0)
    offset = ideal @ m.OCAM_STRETCH.T
    map_x, map_y, _, _ = m._build_pinhole_remap((504, 630), 300.0)
    np.testing.assert_array_equal((offset[..., 0] + m.OCAM_CENTER[0]).astype(np.float32), map_x)
    np.testing.assert_array_equal((offset[..., 1] + m.OCAM_CENTER[1]).astype(np.float32), map_y)


def test_helpers_equal_direct_cv2_remap():
    rng = np.random.default_rng(0)
    map_x, map_y, _, _ = m._build_pinhole_remap((504, 630), 300.0)
    rgb = rng.integers(0, 255, (1080, 1350, 3), dtype=np.uint8)
    raw = rng.integers(0, 65535, (1080, 1350), dtype=np.uint16)
    np.testing.assert_array_equal(
        m.rectify_cubic(rgb, map_x, map_y),
        cv2.remap(
            rgb, map_x, map_y, interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ),
    )
    np.testing.assert_array_equal(
        m.rectify_nearest(raw, map_x, map_y),
        cv2.remap(
            raw, map_x, map_y, interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ),
    )


def test_stored_real_frame_unchanged():
    """Read-only check on a real C3VDv2 frame (skipped without $DATA_ROOT)."""
    root = os.environ.get("DATA_ROOT")
    frame = Path(root or "/nonexistent") / "C3VDv2/frames/C1/c1_ascending_t1_v1"
    if not (frame / "rgb/0000.png").is_file():
        pytest.skip("real C3VDv2 frame not available")
    map_x, map_y, inside, _ = m._build_pinhole_remap((504, 630), 300.0)
    assert digest(m._load_rectified_rgb(frame / "rgb/0000.png", map_x, map_y).numpy()) == REAL_RGB
    depth, valid, _ = m._load_rectified_depth(frame / "depth/0000_depth.tiff", map_x, map_y, inside)
    assert digest(depth.numpy()) == REAL_DEPTH and digest(valid.numpy()) == REAL_VALID
