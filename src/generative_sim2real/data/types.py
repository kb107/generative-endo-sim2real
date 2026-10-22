"""Shared types for dataset samples."""

from typing import NotRequired, TypeAlias, TypedDict

from torch import Tensor


class CameraData(TypedDict):
    K: Tensor | None
    T_world_from_camera: Tensor | None
    distortion: Tensor | None
    model: str | None


class SparseDepthData(TypedDict):
    xy: Tensor
    depth_m: Tensor
    valid: Tensor


class InstanceData(TypedDict, total=False):
    instance_id: str
    mask: Tensor
    bbox_xyxy: Tensor
    size_mm: float
    pathology: str
    anatomical_site: str
    annotation_points_xy: Tensor
    annotation_point_labels: list[str]
    geometry_valid: bool
    annotation_outside_count: int


class TargetData(TypedDict):
    depth_m: Tensor | None
    depth_valid: Tensor | None
    sparse_depth: SparseDepthData | None
    instances: list[InstanceData] | None


class SampleMeta(TypedDict):
    transform_history: NotRequired[list[dict[str, object]]]
    dataset: str
    domain: str
    split: str
    scene_id: str | None
    sequence_id: str
    frame_index: int
    timestamp_s: float | None
    original_hw: tuple[int, int]
    depth_type: str | None
    calibration_id: NotRequired[int | None]
    calibration_requested_id: NotRequired[int | None]
    calibration_source: NotRequired[str]
    projection: NotRequired[str]
    current_hw: NotRequired[tuple[int, int]]
    content_hw: NotRequired[tuple[int, int]]
    content_scale: NotRequired[float]
    padding_tlbr: NotRequired[tuple[int, int, int, int]]
    fold: NotRequired[int]
    # C3VDv2: pixels stored at the 100 mm depth cap in the adapter's own view,
    # counted before any crop/resize transform. They are invalid in depth_valid.
    depth_saturated_pixels: NotRequired[int]


class FrameSample(TypedDict):
    sample_id: str
    rgb: Tensor
    camera: CameraData
    targets: TargetData
    meta: SampleMeta


OptionalTensorBatch: TypeAlias = Tensor | list[Tensor | None] | None


class CameraBatch(TypedDict):
    K: OptionalTensorBatch
    T_world_from_camera: OptionalTensorBatch
    distortion: OptionalTensorBatch
    model: list[str | None]


class TargetBatch(TypedDict):
    depth_m: OptionalTensorBatch
    depth_valid: OptionalTensorBatch
    sparse_depth: list[SparseDepthData | None] | None
    instances: list[list[InstanceData] | None] | None


class FrameBatch(TypedDict):
    sample_id: list[str]
    rgb: Tensor
    camera: CameraBatch
    targets: TargetBatch
    meta: list[SampleMeta]
