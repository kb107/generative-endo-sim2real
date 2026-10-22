"""PyTorch dataset for the official SimCol3D train and test splits."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .types import FrameSample


# SimCol3D stores 8-bit intensity levels in 16-bit PNGs as value * 256.
_PNG_SCALE = float(256 * 255)
_DEPTH_MAX_METERS = 0.20
_FRAME_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)\.png$")
_HANDEDNESS_FLIP = np.diag([1.0, -1.0, 1.0, 1.0])


@dataclass(frozen=True)
class SimCol3DFrameRecord:
    """Paths and identifiers needed to load one SimCol3D frame."""

    sample_id: str
    scene_id: str
    sequence_id: str
    sequence_key: str
    frame_index: int
    rgb_path: Path
    depth_path: Path
    position_path: Path
    rotation_path: Path
    intrinsics_path: Path


class SimCol3D(Dataset[FrameSample]):
    """Load one RGB, depth, intrinsics, and camera pose sample per frame.

    Depth is returned in meters and K remains in pixels. Source camera-to-world
    poses are converted to right-handed OpenCV camera coordinates (x right,
    y down, z forward), with translation in meters. A raw depth value of zero is
    treated as invalid because it places the surface at the camera center.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable[[FrameSample], FrameSample] | None = None,
        include_sequences: tuple[str, ...] | None = None,
        exclude_sequences: tuple[str, ...] | None = None,
        frame_stride: int = 1,
        show_progress: bool = True,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if (
            isinstance(frame_stride, bool)
            or not isinstance(frame_stride, int)
            or frame_stride < 1
        ):
            raise ValueError("frame_stride must be a positive integer")

        self.root = Path(root).expanduser().absolute()
        self.split = split
        self.transform = transform

        included = set(include_sequences or ())
        excluded = set(exclude_sequences or ())
        if included and excluded:
            raise ValueError(
                "include_sequences and exclude_sequences are mutually exclusive"
            )

        if not self.root.is_dir():
            raise FileNotFoundError(f"SimCol3D root does not exist: {self.root}")

        split_path = self.root / "misc" / f"{split}_file.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"SimCol3D split file does not exist: {split_path}")

        self._intrinsics: dict[str, torch.Tensor] = {}
        self._poses: dict[str, torch.Tensor] = {}
        records: list[SimCol3DFrameRecord] = []
        seen_sequences: set[str] = set()

        split_lines = split_path.read_text().splitlines()
        listed_sequences = {
            line.strip().rstrip("/") for line in split_lines if line.strip()
        }
        requested = included or excluded
        missing = sorted(requested - listed_sequences)
        if missing:
            raise ValueError(
                f"Sequences are not present in the official {split} split: {missing}"
            )

        indexed_lines = enumerate(split_lines, start=1)
        for line_number, line in tqdm(
            indexed_lines,
            total=len(split_lines),
            desc=f"Indexing SimCol3D {split}",
            unit="trajectory",
            disable=not show_progress,
        ):
            entry = line.strip().rstrip("/")
            if not entry:
                continue
            if included and entry not in included:
                continue
            if entry in excluded:
                continue

            relative = Path(entry)
            if relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts:
                raise ValueError(
                    f"Malformed trajectory at {split_path}:{line_number}: {line!r}"
                )

            trajectory_dir = self.root / relative
            if not trajectory_dir.is_relative_to(self.root):
                raise ValueError(
                    f"Trajectory escapes the dataset root at {split_path}:{line_number}"
                )
            if not trajectory_dir.is_dir():
                raise FileNotFoundError(
                    f"Trajectory from {split_path}:{line_number} does not exist: "
                    f"{trajectory_dir}"
                )

            sequence_key = relative.as_posix()
            if sequence_key in seen_sequences:
                raise ValueError(f"Duplicate trajectory in {split_path}: {entry}")
            seen_sequences.add(sequence_key)

            trajectory_records = self._index_trajectory(
                trajectory_dir, sequence_key
            )
            records.extend(trajectory_records[::frame_stride])

        if not records:
            raise ValueError(f"No trajectories were listed in {split_path}")

        self.records = tuple(records)

    def _index_trajectory(
        self, trajectory_dir: Path, sequence_key: str
    ) -> list[SimCol3DFrameRecord]:
        scene_dir = trajectory_dir.parent
        scene_id = scene_dir.name
        sequence_id = trajectory_dir.name
        trajectory_id = sequence_id.removeprefix("Frames_")
        if trajectory_id == sequence_id:
            raise ValueError(f"Unexpected SimCol3D trajectory name: {trajectory_dir}")

        rgb_paths = _indexed_pngs(trajectory_dir, "FrameBuffer")
        depth_paths = _indexed_pngs(trajectory_dir, "Depth")
        if not rgb_paths:
            raise ValueError(f"No RGB frames found in {trajectory_dir}")

        rgb_indices = set(rgb_paths)
        depth_indices = set(depth_paths)
        if rgb_indices != depth_indices:
            missing_depth = sorted(rgb_indices - depth_indices)
            missing_rgb = sorted(depth_indices - rgb_indices)
            raise ValueError(
                f"Unpaired frames in {trajectory_dir}: "
                f"missing depth={missing_depth[:5]}, missing RGB={missing_rgb[:5]}"
            )

        indices = sorted(rgb_indices)
        expected_indices = list(range(len(indices)))
        if indices != expected_indices:
            raise ValueError(
                f"Frame indices in {trajectory_dir} must be contiguous from zero; "
                f"found {indices[:5]}...{indices[-5:]}"
            )

        intrinsics_path = scene_dir / "cam.txt"
        position_path = scene_dir / f"SavedPosition_{trajectory_id}.txt"
        rotation_path = scene_dir / f"SavedRotationQuaternion_{trajectory_id}.txt"

        if scene_id not in self._intrinsics:
            intrinsics = _load_table(intrinsics_path, columns=9, label="intrinsics")
            if intrinsics.shape[0] != 1:
                raise ValueError(
                    f"Expected one intrinsics row in {intrinsics_path}, "
                    f"found {intrinsics.shape[0]}"
                )
            self._intrinsics[scene_id] = torch.from_numpy(
                intrinsics.reshape(3, 3).astype(np.float32)
            )

        positions = _load_table(position_path, columns=3, label="positions")
        quaternions = _load_table(rotation_path, columns=4, label="quaternions")
        frame_count = len(indices)
        if len(positions) != frame_count or len(quaternions) != frame_count:
            raise ValueError(
                f"Frame/pose count mismatch in {trajectory_dir}: frames={frame_count}, "
                f"positions={len(positions)}, quaternions={len(quaternions)}"
            )

        try:
            rotations = Rotation.from_quat(quaternions).as_matrix()
        except ValueError as error:
            raise ValueError(f"Invalid quaternion in {rotation_path}: {error}") from error

        poses = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = positions / 100.0
        # Convert the release left-handed poses into the shared camera axes.
        poses = _HANDEDNESS_FLIP @ poses @ _HANDEDNESS_FLIP
        self._poses[sequence_key] = torch.from_numpy(poses.astype(np.float32))

        return [
            SimCol3DFrameRecord(
                sample_id=(
                    f"simcol3d/{scene_id}/{sequence_id}/{index:04d}"
                ),
                scene_id=scene_id,
                sequence_id=sequence_id,
                sequence_key=sequence_key,
                frame_index=index,
                rgb_path=rgb_paths[index],
                depth_path=depth_paths[index],
                position_path=position_path,
                rotation_path=rotation_path,
                intrinsics_path=intrinsics_path,
            )
            for index in indices
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FrameSample:
        record = self.records[index]
        rgb = _load_rgb(record.rgb_path)
        depth_m, depth_valid = _load_depth(record.depth_path)

        if rgb.shape[-2:] != depth_m.shape[-2:]:
            raise ValueError(
                f"RGB/depth shape mismatch for {record.sequence_key} frame "
                f"{record.frame_index}: RGB={tuple(rgb.shape[-2:])}, "
                f"depth={tuple(depth_m.shape[-2:])}"
            )

        height, width = depth_m.shape[-2:]
        sample: FrameSample = {
            "sample_id": record.sample_id,
            "rgb": rgb,
            "camera": {
                "K": self._intrinsics[record.scene_id].clone(),
                "T_world_from_camera": self._poses[record.sequence_key][
                    record.frame_index
                ].clone(),
                "distortion": None,
                "model": "pinhole",
            },
            "targets": {
                "depth_m": depth_m,
                "depth_valid": depth_valid,
                "sparse_depth": None,
                "instances": None,
            },
            "meta": {
                "dataset": "simcol3d",
                "domain": "synthetic",
                "split": self.split,
                "scene_id": record.scene_id,
                "sequence_id": record.sequence_id,
                "frame_index": record.frame_index,
                "timestamp_s": None,
                "original_hw": (height, width),
                "depth_type": "camera_z",
            },
        }

        return self.transform(sample) if self.transform is not None else sample


def _indexed_pngs(directory: Path, prefix: str) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in directory.glob(f"{prefix}_*.png"):
        match = _FRAME_PATTERN.fullmatch(path.name)
        if match is None or match.group("prefix") != prefix:
            continue
        index = int(match.group("index"))
        if index in indexed:
            raise ValueError(
                f"Duplicate {prefix} frame index {index} in {directory}: "
                f"{indexed[index].name}, {path.name}"
            )
        indexed[index] = path
    return indexed


def _load_table(path: Path, columns: int, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"SimCol3D {label} file does not exist: {path}")

    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError(f"Could not parse SimCol3D {label} file {path}: {error}") from error

    if values.ndim != 2 or values.shape[1] != columns:
        raise ValueError(
            f"Expected {columns} columns in {path}, found shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite value found in {path}")
    return values


def _load_rgb(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read SimCol3D RGB image: {path}")
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(
            f"Expected a 16-bit RGBA PNG at {path}, found shape={image.shape}, "
            f"dtype={image.dtype}"
        )

    rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB).astype(np.float32)
    rgb = np.clip(rgb / _PNG_SCALE, 0.0, 1.0)
    return torch.from_numpy(np.moveaxis(rgb, -1, 0).copy())


def _load_depth(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read SimCol3D depth image: {path}")
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError(
            f"Expected a 16-bit grayscale depth PNG at {path}, "
            f"found shape={raw.shape}, dtype={raw.dtype}"
        )

    depth_m = raw.astype(np.float32) / _PNG_SCALE * _DEPTH_MAX_METERS
    depth_valid = raw > 0
    return torch.from_numpy(depth_m[None]), torch.from_numpy(depth_valid[None])
