from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from generative_sim2real.data import SimCol3D


PNG_SCALE = 256 * 255


def _write_trajectory(scene: Path, trajectory_id: str, frame_count: int = 2) -> None:
    frames = scene / f"Frames_{trajectory_id}"
    frames.mkdir(parents=True)

    positions = np.array([[100.0, 200.0, 300.0], [0.0, 0.0, 0.0]])[
        :frame_count
    ]
    quaternions = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        ]
    )[:frame_count]
    np.savetxt(scene / f"SavedPosition_{trajectory_id}.txt", positions)
    np.savetxt(scene / f"SavedRotationQuaternion_{trajectory_id}.txt", quaternions)

    for frame_index in range(frame_count):
        bgra = np.zeros((2, 3, 4), dtype=np.uint16)
        bgra[..., 0] = 10 * 256  # blue
        bgra[..., 1] = 20 * 256  # green
        bgra[..., 2] = 30 * 256  # red
        bgra[..., 3] = PNG_SCALE
        assert cv2.imwrite(str(frames / f"FrameBuffer_{frame_index:04d}.png"), bgra)

        depth = np.array(
            [[0, 256, PNG_SCALE], [512, 1024, 2048]], dtype=np.uint16
        )
        assert cv2.imwrite(str(frames / f"Depth_{frame_index:04d}.png"), depth)


@pytest.fixture
def simcol_root(tmp_path: Path) -> Path:
    root = tmp_path / "SimCol3D"
    scene = root / "SyntheticColon_I"
    misc = root / "misc"
    misc.mkdir(parents=True)
    scene.mkdir()

    K = np.array([[227.6, 0.0, 227.6], [0.0, 237.5, 237.5], [0.0, 0.0, 1.0]])
    np.savetxt(scene / "cam.txt", K.reshape(1, -1))
    _write_trajectory(scene, "S1")
    _write_trajectory(scene, "S2")

    (misc / "train_file.txt").write_text("SyntheticColon_I/Frames_S1/\n")
    (misc / "test_file.txt").write_text("SyntheticColon_I/Frames_S2/\n")
    return root


def test_loads_canonical_frame(simcol_root: Path) -> None:
    dataset = SimCol3D(simcol_root, split="train")
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["sample_id"] == "simcol3d/SyntheticColon_I/Frames_S1/0000"
    assert dataset.records[0].sample_id == sample["sample_id"]
    assert sample["rgb"].shape == (3, 2, 3)
    assert sample["rgb"].dtype == torch.float32
    torch.testing.assert_close(
        sample["rgb"][:, 0, 0], torch.tensor([30.0, 20.0, 10.0]) / 255.0
    )

    depth = sample["targets"]["depth_m"]
    valid = sample["targets"]["depth_valid"]
    assert depth is not None and valid is not None
    assert depth.shape == (1, 2, 3)
    assert depth.dtype == torch.float32
    assert valid.dtype == torch.bool
    assert depth[0, 0, 0] == 0.0
    assert not valid[0, 0, 0]
    torch.testing.assert_close(depth[0, 0, 2], torch.tensor(0.20))
    assert valid[0, 0, 2]
    assert sample["targets"]["instances"] is None

    K = sample["camera"]["K"]
    pose = sample["camera"]["T_world_from_camera"]
    assert K is not None and pose is not None
    torch.testing.assert_close(
        K,
        torch.tensor([[227.6, 0.0, 227.6], [0.0, 237.5, 237.5], [0.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(pose[:3, 3], torch.tensor([1.0, -2.0, 3.0]))
    torch.testing.assert_close(pose[:3, :3], torch.eye(3))

    assert sample["camera"]["model"] == "pinhole"
    assert sample["camera"]["distortion"] is None
    assert sample["meta"]["original_hw"] == (2, 3)
    assert sample["meta"]["depth_type"] == "camera_z"
    assert sample["meta"]["timestamp_s"] is None
    assert "source" not in sample["meta"]

    record = dataset.records[0]
    assert record.rgb_path.name == "FrameBuffer_0000.png"
    assert record.depth_path.name == "Depth_0000.png"
    assert record.position_path.name == "SavedPosition_S1.txt"
    assert record.rotation_path.name == "SavedRotationQuaternion_S1.txt"
    assert record.intrinsics_path.name == "cam.txt"


def test_converts_rotation_handedness(simcol_root: Path) -> None:
    pose = SimCol3D(simcol_root, split="train")[1]["camera"][
        "T_world_from_camera"
    ]
    assert pose is not None
    expected = torch.tensor(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    torch.testing.assert_close(pose[:3, :3], expected, atol=1e-6, rtol=0.0)


def test_uses_official_split_and_applies_transform(simcol_root: Path) -> None:
    def add_marker(sample):
        sample["meta"]["domain"] = "transformed"
        return sample

    dataset = SimCol3D(simcol_root, split="test", transform=add_marker)

    assert dataset.records[0].sequence_id == "Frames_S2"
    assert dataset[0]["meta"]["split"] == "test"
    assert dataset[0]["meta"]["domain"] == "transformed"


def test_filters_trajectories_within_official_split(simcol_root: Path) -> None:
    train_file = simcol_root / "misc" / "train_file.txt"
    train_file.write_text(
        "SyntheticColon_I/Frames_S1/\nSyntheticColon_I/Frames_S2/\n"
    )

    included = SimCol3D(
        simcol_root,
        split="train",
        include_sequences=("SyntheticColon_I/Frames_S2",),
    )
    excluded = SimCol3D(
        simcol_root,
        split="train",
        exclude_sequences=("SyntheticColon_I/Frames_S2",),
    )

    assert {record.sequence_key for record in included.records} == {
        "SyntheticColon_I/Frames_S2"
    }
    assert {record.sequence_key for record in excluded.records} == {
        "SyntheticColon_I/Frames_S1"
    }


def test_frame_stride_is_applied_within_each_trajectory(simcol_root: Path) -> None:
    train_file = simcol_root / "misc" / "train_file.txt"
    train_file.write_text(
        "SyntheticColon_I/Frames_S1/\nSyntheticColon_I/Frames_S2/\n"
    )

    dataset = SimCol3D(simcol_root, split="train", frame_stride=2)

    assert [(record.sequence_id, record.frame_index) for record in dataset.records] == [
        ("Frames_S1", 0),
        ("Frames_S2", 0),
    ]


def test_rejects_bad_sequence_filter(simcol_root: Path) -> None:
    with pytest.raises(ValueError, match="not present"):
        SimCol3D(
            simcol_root,
            split="train",
            include_sequences=("SyntheticColon_I/Frames_missing",),
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        SimCol3D(
            simcol_root,
            split="train",
            include_sequences=("SyntheticColon_I/Frames_S1",),
            exclude_sequences=("SyntheticColon_I/Frames_S1",),
        )


def test_sample_ids_are_unique_across_splits(simcol_root: Path) -> None:
    train = SimCol3D(simcol_root, split="train")
    test = SimCol3D(simcol_root, split="test")
    sample_ids = [record.sample_id for record in (*train.records, *test.records)]

    assert len(sample_ids) == len(set(sample_ids))
    assert all(
        dataset[index]["sample_id"] == record.sample_id
        for dataset in (train, test)
        for index, record in enumerate(dataset.records)
    )


def test_rejects_unpaired_frames(simcol_root: Path) -> None:
    missing_depth = (
        simcol_root / "SyntheticColon_I" / "Frames_S1" / "Depth_0001.png"
    )
    missing_depth.unlink()

    with pytest.raises(ValueError, match="Unpaired frames"):
        SimCol3D(simcol_root, split="train")


def test_rejects_pose_count_mismatch(simcol_root: Path) -> None:
    position_path = (
        simcol_root / "SyntheticColon_I" / "SavedPosition_S1.txt"
    )
    np.savetxt(position_path, np.array([[0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="Frame/pose count mismatch"):
        SimCol3D(simcol_root, split="train")


def test_rejects_unknown_split(simcol_root: Path) -> None:
    with pytest.raises(ValueError, match="split must be"):
        SimCol3D(simcol_root, split="validation")


def test_simcol_saturated_uint16_is_not_c3vd_cap(simcol_root):
    from generative_sim2real.data.simcol3d import _load_depth
    path = simcol_root / "SyntheticColon_I/Frames_S1/Depth_0000.png"
    raw = np.array([[0, 65535]], dtype=np.uint16)
    assert cv2.imwrite(str(path), raw)
    depth, valid = _load_depth(path)
    assert valid.tolist() == [[[False, True]]]
    assert depth[0, 0, 1].item() == pytest.approx(65535 / (256*255) * .2)


def test_split_path_cannot_escape_root(simcol_root):
    (simcol_root / "misc/train_file.txt").write_text("../Frames_S1\n")
    with pytest.raises(ValueError, match="Malformed|escapes"):
        SimCol3D(simcol_root, show_progress=False)


def test_logical_root_symlink_preserved(simcol_root, tmp_path):
    logical = tmp_path / "SimColView"
    logical.symlink_to(simcol_root, target_is_directory=True)
    dataset = SimCol3D(logical, show_progress=False)
    assert dataset.root == logical
    assert dataset.records[0].rgb_path.is_relative_to(logical)
    assert dataset[0]["rgb"].shape == (3, 2, 3)
