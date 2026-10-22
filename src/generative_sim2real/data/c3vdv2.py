"""Monocular metric-depth adapter for registered C3VDv2 videos."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np
from openpyxl import load_workbook
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .types import FrameSample


_RAW_HEIGHT = 1080
_RAW_WIDTH = 1350
_DEPTH_MAX_METERS = 0.1
_RGB_PATTERN = re.compile(r"^(?P<index>\d{4})\.png$")
_DEPTH_PATTERN = re.compile(r"^(?P<index>\d{4})_depth\.tiff$")
_SEQUENCE_PATTERN = re.compile(
    r"^(?P<colon>c[12])_(?P<segment>[a-z]+\d?)_"
    r"(?P<texture>t[1-4])_(?P<version>v[1-3])$"
)

# Official Olympus CF-HQ190L Scaramuzza calibration supplied with C3VDv2.
_OCAM_CENTER = np.array([678.544839263292, 542.975887548343], dtype=np.float64)
_OCAM_POLYNOMIAL = np.array(
    [
        769.243600037458,
        0.0,
        -0.000812770624150226,
        6.25674244578925e-07,
        -1.19662182144280e-09,
    ],
    dtype=np.float64,
)

# The C3VD renderer constructs glm::mat2(c, d, e, 1). GLM fills matrices by
# columns, so this is the exact stretch matrix used to render the ground truth.
_OCAM_STRETCH = np.array(
    [
        [0.999986882249990, -0.00296316513429569],
        [0.00288273829525059, 1.0],
    ],
    dtype=np.float64,
)

# Public names for code that renders or rectifies synthetic C3VDv2 views (sim/fisheye.py).
OCAM_CENTER = _OCAM_CENTER
OCAM_POLYNOMIAL = _OCAM_POLYNOMIAL
OCAM_STRETCH = _OCAM_STRETCH
RAW_SIZE_HW = (_RAW_HEIGHT, _RAW_WIDTH)


@dataclass(frozen=True)
class C3VDv2Sequence:
    sequence_id: str
    colon: str
    segment: str
    texture: str
    version: str
    quality_score: int
    total_frames: int


@dataclass(frozen=True)
class C3VDv2FrameRecord:
    sample_id: str
    scene_id: str
    sequence_id: str
    frame_index: int
    rgb_path: Path
    depth_path: Path


class C3VDv2(Dataset[FrameSample]):
    """Return rectified C3VDv2 RGB, metric depth, intrinsics, and pose.

    The raw clinical-colonoscope images use a Scaramuzza omnidirectional
    camera. RGB, depth, and validity are remapped into a virtual pinhole camera
    with the same center and axes. Returned poses are camera-to-world, use
    right-handed OpenCV camera coordinates, and have translation in meters.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        colons: tuple[str, ...],
        textures: tuple[str, ...],
        video_versions: tuple[str, ...],
        quality_scores: tuple[int, ...],
        allow_v3_clean_proxy_depth: bool,
        rectified_hw: tuple[int, int] = (504, 630),
        rectified_focal_px: float = 300.0,
        center_crop_square: bool = False,
        include_sequences: tuple[str, ...] | None = None,
        exclude_sequences: tuple[str, ...] | None = None,
        frame_stride: int = 1,
        transform: Callable[[FrameSample], FrameSample] | None = None,
        show_progress: bool = True,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "C3VDv2 split must be 'train', 'validation', or 'test', "
                f"got {split!r}"
            )
        _validate_choices("colons", colons, {"c1", "c2"})
        _validate_choices("textures", textures, {"t1", "t2", "t3", "t4"})
        _validate_choices("video_versions", video_versions, {"v1", "v2", "v3"})
        _validate_choices("quality_scores", quality_scores, {1, 2, 3})
        if "v3" in video_versions and not allow_v3_clean_proxy_depth:
            raise ValueError(
                "C3VDv2 v3 depth represents the clean phantom; set "
                "allow_v3_clean_proxy_depth=true to use it explicitly"
            )
        if (
            isinstance(frame_stride, bool)
            or not isinstance(frame_stride, int)
            or frame_stride < 1
        ):
            raise ValueError("C3VDv2 frame_stride must be at least 1")

        included = set(include_sequences or ())
        excluded = set(exclude_sequences or ())
        if included and excluded:
            raise ValueError(
                "include_sequences and exclude_sequences are mutually exclusive"
            )

        self.root = Path(root).expanduser().absolute()
        self.frames_root = self.root / "frames"
        self.metadata_path = self.root / "metadata.xlsx"
        self.split = split
        self.frame_stride = frame_stride
        self.transform = transform
        self.rectified_hw = rectified_hw
        self.rectified_focal_px = float(rectified_focal_px)
        self.center_crop_square = center_crop_square

        if not self.root.is_dir():
            raise FileNotFoundError(f"C3VDv2 root does not exist: {self.root}")
        if not self.frames_root.is_dir():
            raise FileNotFoundError(
                f"C3VDv2 frames link does not resolve to a directory: "
                f"{self.root / 'frames'}"
            )
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"C3VDv2 metadata link does not resolve to a file: "
                f"{self.metadata_path}"
            )

        map_x, map_y, map_inside, K = _build_pinhole_remap(
            rectified_hw, self.rectified_focal_px
        )
        if center_crop_square:
            map_x, map_y, map_inside, K = _center_crop_square_remap(
                map_x, map_y, map_inside, K
            )
        self._map_x = map_x
        self._map_y = map_y
        self._map_inside = map_inside
        self._K = K

        all_sequences = _load_sequence_metadata(self.metadata_path)
        selected = [
            sequence
            for sequence in all_sequences.values()
            if sequence.colon in colons
            and sequence.texture in textures
            and sequence.version in video_versions
            and sequence.quality_score in quality_scores
        ]
        eligible_ids = {sequence.sequence_id for sequence in selected}
        requested = included or excluded
        missing = sorted(requested - eligible_ids)
        if missing:
            raise ValueError(
                "C3VDv2 sequence filters are not eligible under the configured "
                f"colon/texture/version/quality filters: {missing}"
            )
        if included:
            selected = [s for s in selected if s.sequence_id in included]
        elif excluded:
            selected = [s for s in selected if s.sequence_id not in excluded]
        selected.sort(key=lambda sequence: sequence.sequence_id)
        if not selected:
            raise ValueError("C3VDv2 filters selected no sequences")

        self._poses: dict[str, torch.Tensor] = {}
        records: list[C3VDv2FrameRecord] = []
        for sequence in tqdm(
            selected,
            desc=f"Indexing C3VDv2 {split}",
            unit="sequence",
            disable=not show_progress,
        ):
            records.extend(self._index_sequence(sequence))

        if not records:
            raise ValueError("C3VDv2 filters selected no frames")
        self.records = tuple(records)
        self.sequence_ids = tuple(sequence.sequence_id for sequence in selected)

    def _index_sequence(
        self, sequence: C3VDv2Sequence
    ) -> list[C3VDv2FrameRecord]:
        colon_dir = self.frames_root / sequence.colon.upper()
        sequence_dir = colon_dir / sequence.sequence_id
        if not sequence_dir.is_dir():
            raise FileNotFoundError(
                f"C3VDv2 sequence from metadata does not exist: {sequence_dir}"
            )
        payload = _resolve_sequence_payload(sequence_dir)

        rgb_paths = _indexed_files(payload / "rgb", _RGB_PATTERN, "RGB")
        depth_paths = _indexed_files(payload / "depth", _DEPTH_PATTERN, "depth")
        rgb_indices = set(rgb_paths)
        depth_indices = set(depth_paths)
        if rgb_indices != depth_indices:
            raise ValueError(
                f"Unpaired C3VDv2 frames in {payload}: "
                f"missing depth={sorted(rgb_indices - depth_indices)[:5]}, "
                f"missing RGB={sorted(depth_indices - rgb_indices)[:5]}"
            )

        indices = sorted(rgb_indices)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"C3VDv2 frame indices must be contiguous from zero in {payload}"
            )
        if len(indices) != sequence.total_frames:
            raise ValueError(
                f"C3VDv2 metadata/frame mismatch for {sequence.sequence_id}: "
                f"metadata={sequence.total_frames}, files={len(indices)}"
            )

        poses = _load_poses(payload / "pose.txt")
        if len(poses) != len(indices):
            raise ValueError(
                f"C3VDv2 frame/pose mismatch for {sequence.sequence_id}: "
                f"frames={len(indices)}, poses={len(poses)}"
            )
        self._poses[sequence.sequence_id] = poses

        return [
            C3VDv2FrameRecord(
                sample_id=(
                    f"c3vdv2/{sequence.colon}/{sequence.sequence_id}/{index:04d}"
                ),
                scene_id=sequence.colon,
                sequence_id=sequence.sequence_id,
                frame_index=index,
                rgb_path=rgb_paths[index],
                depth_path=depth_paths[index],
            )
            for index in indices[:: self.frame_stride]
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> FrameSample:
        record = self.records[index]
        rgb = _load_rectified_rgb(record.rgb_path, self._map_x, self._map_y)
        depth_m, depth_valid, depth_saturated = _load_rectified_depth(
            record.depth_path,
            self._map_x,
            self._map_y,
            self._map_inside,
        )
        pose = self._poses[record.sequence_id][record.frame_index].clone()

        sample: FrameSample = {
            "sample_id": record.sample_id,
            "rgb": rgb,
            "camera": {
                "K": self._K.clone(),
                "T_world_from_camera": pose,
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
                "dataset": "c3vdv2",
                "domain": "phantom",
                "split": self.split,
                "scene_id": record.scene_id,
                "sequence_id": record.sequence_id,
                "frame_index": record.frame_index,
                "timestamp_s": None,
                "original_hw": (_RAW_HEIGHT, _RAW_WIDTH),
                "depth_type": "camera_z",
                "depth_saturated_pixels": depth_saturated,
                "current_hw": tuple(self._map_x.shape),
                "transform_history": self._geometry_history(),
            },
        }
        return self.transform(sample) if self.transform is not None else sample


    def _geometry_history(self) -> list[dict[str, object]]:
        history = [{
            "name": "pinhole_rectify",
            "input_hw": (_RAW_HEIGHT, _RAW_WIDTH),
            "output_hw": self.rectified_hw,
            "focal_px": self.rectified_focal_px,
            "rgb_interpolation": "cubic",
            "label_interpolation": "nearest",
        }]
        if self.center_crop_square:
            height, width = self.rectified_hw
            side = min(height, width)
            history.append({
                "name": "center_crop_square",
                "input_hw": self.rectified_hw,
                "output_hw": (side, side),
                "top": (height - side) // 2,
                "left": (width - side) // 2,
            })
        return history


@dataclass(frozen=True)
class C3VDv2RGBFrameRecord:
    """RGB-only identity: label paths are deliberately absent."""

    sample_id: str
    scene_id: str
    sequence_id: str
    frame_index: int
    rgb_path: Path


class C3VDv2RGB(C3VDv2):
    """C1 appearance view that only indexes and decodes RGB and calibration.

    Shares the labeled adapter's explicit constructor and metadata/remap policy;
    overridden indexing and loading never construct depth or pose paths.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        colons: tuple[str, ...],
        textures: tuple[str, ...],
        video_versions: tuple[str, ...],
        quality_scores: tuple[int, ...],
        allow_v3_clean_proxy_depth: bool,
        rectified_hw: tuple[int, int] = (504, 630),
        rectified_focal_px: float = 300.0,
        center_crop_square: bool = False,
        include_sequences: tuple[str, ...] | None = None,
        exclude_sequences: tuple[str, ...] | None = None,
        frame_stride: int = 1,
        transform: Callable[[FrameSample], FrameSample] | None = None,
        show_progress: bool = True,
    ) -> None:
        if colons != ("c1",):
            raise ValueError("C3VDv2RGB appearance loading permits only colons=('c1',)")
        super().__init__(
            root,
            split=split,
            colons=colons,
            textures=textures,
            video_versions=video_versions,
            quality_scores=quality_scores,
            allow_v3_clean_proxy_depth=allow_v3_clean_proxy_depth,
            rectified_hw=rectified_hw,
            rectified_focal_px=rectified_focal_px,
            center_crop_square=center_crop_square,
            include_sequences=include_sequences,
            exclude_sequences=exclude_sequences,
            frame_stride=frame_stride,
            transform=transform,
            show_progress=show_progress,
        )

    def _index_sequence(self, sequence: C3VDv2Sequence) -> list[C3VDv2RGBFrameRecord]:
        sequence_dir = self.frames_root / sequence.colon.upper() / sequence.sequence_id
        payload = _resolve_sequence_payload(sequence_dir)
        rgb_paths = _indexed_files(payload / "rgb", _RGB_PATTERN, "RGB")
        indices = sorted(rgb_paths)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"C3VDv2 frame indices must be contiguous from zero in {payload}"
            )
        if len(indices) != sequence.total_frames:
            raise ValueError(
                f"C3VDv2 metadata/frame mismatch for {sequence.sequence_id}: "
                f"metadata={sequence.total_frames}, files={len(indices)}"
            )
        return [
            C3VDv2RGBFrameRecord(
                sample_id=f"c3vdv2/{sequence.colon}/{sequence.sequence_id}/{index:04d}",
                scene_id=sequence.colon,
                sequence_id=sequence.sequence_id,
                frame_index=index,
                rgb_path=rgb_paths[index],
            )
            for index in indices[::self.frame_stride]
        ]

    def __getitem__(self, index: int) -> FrameSample:
        record = self.records[index]
        sample: FrameSample = {
            "sample_id": record.sample_id,
            "rgb": _load_rectified_rgb(record.rgb_path, self._map_x, self._map_y),
            "camera": {
                "K": self._K.clone(),
                "T_world_from_camera": None,
                "distortion": None,
                "model": "pinhole",
            },
            "targets": {
                "depth_m": None,
                "depth_valid": None,
                "sparse_depth": None,
                "instances": None,
            },
            "meta": {
                "dataset": "c3vdv2",
                "domain": "phantom",
                "split": self.split,
                "scene_id": record.scene_id,
                "sequence_id": record.sequence_id,
                "frame_index": record.frame_index,
                "timestamp_s": None,
                "original_hw": (_RAW_HEIGHT, _RAW_WIDTH),
                "depth_type": None,
                "current_hw": tuple(self._map_x.shape),
                "transform_history": self._geometry_history(),
            },
        }
        return self.transform(sample) if self.transform is not None else sample



def _load_sequence_metadata(path: Path) -> dict[str, C3VDv2Sequence]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook.active.iter_rows(values_only=True)
        try:
            header = next(rows)
        except StopIteration as error:
            raise ValueError(f"Empty C3VDv2 metadata workbook: {path}") from error
        names = [str(value).strip() if value is not None else "" for value in header]
        required = {
            "Colon",
            "Segment",
            "Phantom Number",
            "Video Number",
            "Video Name",
            "Qualitative Score",
            "Total Frames",
        }
        missing = sorted(required - set(names))
        if missing:
            raise ValueError(f"C3VDv2 metadata is missing columns: {missing}")

        metadata: dict[str, C3VDv2Sequence] = {}
        for row_number, values in enumerate(rows, start=2):
            row = dict(zip(names, values))
            sequence_id = row.get("Video Name")
            if not isinstance(sequence_id, str) or not sequence_id.strip():
                continue
            sequence_id = sequence_id.strip()
            match = _SEQUENCE_PATTERN.fullmatch(sequence_id)
            if match is None:
                continue
            quality = row.get("Qualitative Score")
            if quality is None:
                continue
            sequence = C3VDv2Sequence(
                sequence_id=sequence_id,
                colon=_required_metadata_string(row, "Colon", row_number),
                segment=_required_metadata_string(row, "Segment", row_number),
                texture=_required_metadata_string(
                    row, "Phantom Number", row_number
                ),
                version=_required_metadata_string(row, "Video Number", row_number),
                quality_score=_metadata_integer(
                    quality, "Qualitative Score", row_number
                ),
                total_frames=_metadata_integer(
                    row.get("Total Frames"), "Total Frames", row_number
                ),
            )
            parsed = match.groupdict()
            expected = (
                parsed["colon"],
                parsed["segment"],
                parsed["texture"],
                parsed["version"],
            )
            actual = (
                sequence.colon,
                sequence.segment,
                sequence.texture,
                sequence.version,
            )
            if actual != expected:
                raise ValueError(
                    f"C3VDv2 metadata disagrees with {sequence_id} at row "
                    f"{row_number}: fields={actual}, name={expected}"
                )
            if sequence_id in metadata:
                raise ValueError(f"Duplicate C3VDv2 metadata row: {sequence_id}")
            metadata[sequence_id] = sequence
    finally:
        workbook.close()

    if not metadata:
        raise ValueError(f"No registered C3VDv2 sequences found in {path}")
    return metadata


def _required_metadata_string(
    row: dict[str, object], key: str, row_number: int
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid C3VDv2 metadata {key!r} at row {row_number}")
    return value.strip()


def _metadata_integer(value: object, key: str, row_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid C3VDv2 metadata {key!r} at row {row_number}")
    integer = int(value)
    if float(value) != integer or integer < 1:
        raise ValueError(f"Invalid C3VDv2 metadata {key!r} at row {row_number}")
    return integer


def _resolve_sequence_payload(sequence_dir: Path) -> Path:
    if (sequence_dir / "rgb").is_dir():
        return sequence_dir
    nested = sequence_dir / sequence_dir.name
    if (nested / "rgb").is_dir():
        return nested
    raise FileNotFoundError(
        f"C3VDv2 sequence has no RGB payload at {sequence_dir} or {nested}"
    )


def _indexed_files(
    directory: Path, pattern: re.Pattern[str], label: str
) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"C3VDv2 {label} directory does not exist: {directory}")
    indexed: dict[int, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        frame_index = int(match.group("index"))
        if frame_index in indexed:
            raise ValueError(
                f"Duplicate C3VDv2 {label} frame {frame_index} in {directory}"
            )
        indexed[frame_index] = path
    if not indexed:
        raise ValueError(f"No C3VDv2 {label} frames found in {directory}")
    return indexed


def _load_poses(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"C3VDv2 pose file does not exist: {path}")
    try:
        values = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except ValueError as error:
        raise ValueError(f"Could not parse C3VDv2 poses in {path}: {error}") from error
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"Expected 16 pose values per row in {path}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite C3VDv2 pose in {path}")

    poses = values.reshape(-1, 4, 4).transpose(0, 2, 1).copy()
    poses[:, :3, 3] /= 1000.0
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(poses[:, 3], expected_last_row, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"C3VDv2 poses do not become homogeneous after column-major parsing: {path}"
        )
    return torch.from_numpy(poses.astype(np.float32))


def rectify_cubic(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Array-level RGB rectification used for real and rendered frames (cubic, zero border)."""
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rectify_nearest(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Array-level nearest-neighbour rectification (depth, masks, normals), zero border."""
    return cv2.remap(
        array,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _load_rectified_rgb(
    path: Path, map_x: np.ndarray, map_y: np.ndarray
) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read C3VDv2 RGB image: {path}")
    if image.dtype != np.uint8 or image.shape != (_RAW_HEIGHT, _RAW_WIDTH, 3):
        raise ValueError(
            f"Expected C3VDv2 RGB shape ({_RAW_HEIGHT}, {_RAW_WIDTH}, 3) uint8 "
            f"at {path}, got shape={image.shape}, dtype={image.dtype}"
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rectified = rectify_cubic(rgb, map_x, map_y)
    array = np.moveaxis(rectified.astype(np.float32) / 255.0, -1, 0).copy()
    return torch.from_numpy(array)


def _load_rectified_depth(
    path: Path,
    map_x: np.ndarray,
    map_y: np.ndarray,
    map_inside: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return depth, validity, and the count of pixels stored at the 100 mm cap."""
    try:
        with Image.open(path) as image:
            raw = np.asarray(image).copy()
    except OSError as error:
        raise ValueError(f"Could not read C3VDv2 depth image {path}: {error}") from error
    if raw.dtype != np.uint16 or raw.shape != (_RAW_HEIGHT, _RAW_WIDTH):
        raise ValueError(
            f"Expected C3VDv2 depth shape ({_RAW_HEIGHT}, {_RAW_WIDTH}) uint16 "
            f"at {path}, got shape={raw.shape}, dtype={raw.dtype}"
        )

    rectified = rectify_nearest(raw, map_x, map_y)
    saturated = (rectified == np.iinfo(np.uint16).max) & map_inside
    valid = (rectified > 0) & ~saturated & map_inside
    depth_m = rectified.astype(np.float32) / np.iinfo(np.uint16).max
    depth_m *= _DEPTH_MAX_METERS
    depth_m[~valid] = 0.0
    return (
        torch.from_numpy(depth_m[None]),
        torch.from_numpy(valid[None]),
        int(saturated.sum()),
    )


def pinhole_ideal_coordinates(size_hw: tuple[int, int], focal_px: float) -> np.ndarray:
    """Pre-stretch fisheye offsets (pixels, ``[H, W, 2]`` x then y) seen by each pinhole pixel.

    The real adapter maps these to raw pixels with the stretch matrix and the OCAM center;
    synthetic renders made in the pre-stretch frame sample them directly.
    """
    height, width = size_hw
    center_x = (width - 1.0) * 0.5
    center_y = (height - 1.0) * 0.5
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx - center_x) / focal_px
    y = (yy - center_y) / focal_px
    radius = np.hypot(x, y)

    rho = _invert_ocam_radius(radius)
    radial_scale = np.divide(
        rho,
        radius,
        out=np.zeros_like(rho),
        where=radius > 0.0,
    )
    return np.stack([x * radial_scale, y * radial_scale], axis=-1)


def _build_pinhole_remap(
    size_hw: tuple[int, int], focal_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    height, width = size_hw
    if height < 1 or width < 1:
        raise ValueError(f"C3VDv2 rectified dimensions must be positive, got {size_hw}")
    if focal_px <= 0.0 or not np.isfinite(focal_px):
        raise ValueError("C3VDv2 rectified focal length must be finite and positive")

    center_x = (width - 1.0) * 0.5
    center_y = (height - 1.0) * 0.5
    ideal = pinhole_ideal_coordinates(size_hw, focal_px)
    source_offset = ideal @ _OCAM_STRETCH.T
    map_x = (source_offset[..., 0] + _OCAM_CENTER[0]).astype(np.float32)
    map_y = (source_offset[..., 1] + _OCAM_CENTER[1]).astype(np.float32)
    inside = (
        (map_x >= 0.0)
        & (map_x <= _RAW_WIDTH - 1.0)
        & (map_y >= 0.0)
        & (map_y <= _RAW_HEIGHT - 1.0)
    )

    K = torch.tensor(
        [
            [focal_px, 0.0, center_x],
            [0.0, focal_px, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return map_x, map_y, inside, K


def _center_crop_square_remap(
    map_x: np.ndarray,
    map_y: np.ndarray,
    map_inside: np.ndarray,
    K: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Center-crop a rectification map and shift its principal point."""
    if map_x.shape != map_y.shape or map_x.shape != map_inside.shape:
        raise ValueError("C3VDv2 rectification maps must have matching shapes")
    height, width = map_x.shape
    side = min(height, width)
    crop_top = (height - side) // 2
    crop_left = (width - side) // 2
    rows = slice(crop_top, crop_top + side)
    columns = slice(crop_left, crop_left + side)

    cropped_K = K.clone()
    cropped_K[0, 2] -= crop_left
    cropped_K[1, 2] -= crop_top
    return (
        map_x[rows, columns].copy(),
        map_y[rows, columns].copy(),
        map_inside[rows, columns].copy(),
        cropped_K,
    )


def _invert_ocam_radius(pinhole_radius: np.ndarray) -> np.ndarray:
    """Solve rho / z(rho) = pinhole_radius on the forward-facing branch."""
    roots = np.roots(_OCAM_POLYNOMIAL[::-1])
    positive_z_roots = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-7 and root.real > 0.0
    )
    if not positive_z_roots:
        raise RuntimeError("C3VDv2 calibration polynomial has no positive z root")
    upper = positive_z_roots[0] * (1.0 - 1e-9)

    low = np.zeros_like(pinhole_radius, dtype=np.float64)
    high = np.full_like(pinhole_radius, upper, dtype=np.float64)
    for _ in range(48):
        middle = (low + high) * 0.5
        value = middle - pinhole_radius * _ocam_z(middle)
        lower_half = value < 0.0
        low = np.where(lower_half, middle, low)
        high = np.where(lower_half, high, middle)
    rho = (low + high) * 0.5
    rho[pinhole_radius == 0.0] = 0.0
    return rho


def _ocam_z(rho: np.ndarray) -> np.ndarray:
    a0, a1, a2, a3, a4 = _OCAM_POLYNOMIAL
    return a0 + rho * (a1 + rho * (a2 + rho * (a3 + rho * a4)))


def _validate_choices(
    name: str, values: tuple[object, ...], allowed: set[object]
) -> None:
    if not values:
        raise ValueError(f"C3VDv2 {name} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"C3VDv2 {name} contains duplicates")
    unsupported = sorted(set(values) - allowed)
    if unsupported:
        raise ValueError(f"Unsupported C3VDv2 {name}: {unsupported}")
