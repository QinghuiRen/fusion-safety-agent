"""把神经 BEV 3D 框转换为现有 Radar/KF 跟踪器的观测类型。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fusion_tracking_carla.types import FusionDetection, RadarReturn

from .schema import BEVFusionFrameResult


VEHICLE_CLASSES = {"car", "truck", "construction_vehicle", "bus", "trailer"}
VRU_CLASSES = {"motorcycle", "bicycle", "pedestrian"}
DEFAULT_TRACK_CLASSES = VEHICLE_CLASSES | VRU_CLASSES


@dataclass(frozen=True)
class BEVFusionAdapterConfig:
    """常调参数集中区。

    ``score_threshold`` 是首个应调参数，建议先保持 0.25；过低会产生碎片轨迹，
    过高会损失远处目标。位置标准差是工程启发式观测噪声，不是网络直接输出的
    概率协方差，后续可通过 nuScenes 验证集标定。
    """

    score_threshold: float = 0.25
    tracked_classes: frozenset[str] = frozenset(DEFAULT_TRACK_CLASSES)
    radar_association_gate_m: float = 3.2
    min_position_std_m: float = 0.60
    max_position_std_m: float = 3.00
    score_std_gain_m: float = 1.40
    support_std_gain_m: float = 1.00
    use_model_velocity: bool = True
    # 同一物体可能由检测头输出多个高度重叠的3D框。这里仅在同一类别族内做
    # BEV IoU NMS；阈值越低去重越强。车辆相邻车道通常IoU接近0，不会被合并。
    vehicle_nms_iou_threshold: float = 0.15
    vru_nms_iou_threshold: float = 0.10
    # BEVFusion速度头仅作保守Cartesian先验；应明显弱于Radar径向观测。
    model_velocity_std_mps: float = 4.0

    def validate(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold 必须位于 [0, 1]")
        if self.radar_association_gate_m <= 0:
            raise ValueError("radar_association_gate_m 必须大于 0")
        if not 0 < self.min_position_std_m <= self.max_position_std_m:
            raise ValueError("位置标准差上下限无效")
        if self.model_velocity_std_mps <= 0:
            raise ValueError("model_velocity_std_mps必须大于0")
        for name, value in (
            ("vehicle_nms_iou_threshold", self.vehicle_nms_iou_threshold),
            ("vru_nms_iou_threshold", self.vru_nms_iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}必须位于[0, 1]")


def _transform_xyz(points_xyz: np.ndarray, destination_from_source: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=float)
    transform = np.asarray(destination_from_source, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("global_from_lidar 必须为 4x4 齐次变换矩阵")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz 必须为 N×3")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (transform @ homogeneous.T).T[:, :3]


def _lidar_support_counts(
    result: BEVFusionFrameResult,
    lidar_points_lidar: np.ndarray | None,
) -> list[int]:
    """统计每个定向框内的点数，只作为证据量，不参与网络检测结果生成。"""

    if lidar_points_lidar is None:
        return [0] * len(result.detections)
    points = np.asarray(lidar_points_lidar, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("lidar_points_lidar 必须至少为 N×3")
    xyz = points[:, :3]
    counts: list[int] = []
    for box in result.detections:
        center = np.asarray(box.center_lidar_m)
        length, width, height = box.size_lwh_m
        delta = xyz - center
        cosine, sine = np.cos(box.yaw_lidar_rad), np.sin(box.yaw_lidar_rad)
        # 将点旋转到框局部坐标；LIDAR_TOP 为 x前/y左/z上。
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        inside = (
            (np.abs(local_x) <= length / 2)
            & (np.abs(local_y) <= width / 2)
            & (np.abs(delta[:, 2]) <= height / 2)
        )
        counts.append(int(np.count_nonzero(inside)))
    return counts


def _class_family(class_name: str) -> str:
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    if class_name in VRU_CLASSES:
        return "vru"
    raise ValueError(f"类别 {class_name!r} 不属于当前跟踪范围")


def _bev_iou(first, second) -> float:
    """计算两个有向3D框在地面的IoU，只用于同族候选去重。"""

    first_rect = (
        (float(first.center_lidar_m[0]), float(first.center_lidar_m[1])),
        (float(first.size_lwh_m[0]), float(first.size_lwh_m[1])),
        float(np.degrees(first.yaw_lidar_rad)),
    )
    second_rect = (
        (float(second.center_lidar_m[0]), float(second.center_lidar_m[1])),
        (float(second.size_lwh_m[0]), float(second.size_lwh_m[1])),
        float(np.degrees(second.yaw_lidar_rad)),
    )
    _, polygon = cv2.rotatedRectangleIntersection(first_rect, second_rect)
    if polygon is None:
        return 0.0
    intersection = float(abs(cv2.contourArea(polygon)))
    first_area = float(first.size_lwh_m[0] * first.size_lwh_m[1])
    second_area = float(second.size_lwh_m[0] * second.size_lwh_m[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _nms_indices(result: BEVFusionFrameResult, indices: list[int], cfg) -> list[int]:
    """按置信度执行同类别族BEV NMS，并恢复原始顺序供下游稳定消费。"""

    selected: list[int] = []
    for index in sorted(indices, key=lambda item: result.detections[item].score, reverse=True):
        box = result.detections[index]
        family = _class_family(box.class_name)
        threshold = (
            cfg.vehicle_nms_iou_threshold
            if family == "vehicle"
            else cfg.vru_nms_iou_threshold
        )
        duplicate = any(
            _class_family(result.detections[kept].class_name) == family
            and _bev_iou(box, result.detections[kept]) > threshold
            for kept in selected
        )
        if not duplicate:
            selected.append(index)
    return sorted(selected)


def detections_from_bevfusion(
    result: BEVFusionFrameResult,
    global_from_lidar: np.ndarray,
    *,
    radar_returns: list[RadarReturn] | None = None,
    lidar_points_lidar: np.ndarray | None = None,
    config: BEVFusionAdapterConfig | None = None,
) -> list[FusionDetection]:
    """转换 BEVFusion 结果，并在目标层关联当前帧 Radar 回波。

    BEVFusion 负责 Camera+LiDAR 中层特征融合；这里的 Radar 仍严格保留为
    一维径向速度证据，后续由 ``StreamingFusionTracker`` 的物理观测模型更新，
    不把 Doppler 标量伪装成完整二维速度。
    """

    cfg = config or BEVFusionAdapterConfig()
    cfg.validate()
    returns = radar_returns or []
    support_counts = _lidar_support_counts(result, lidar_points_lidar)
    centers_lidar = np.asarray(
        [box.center_lidar_m for box in result.detections], dtype=float
    ).reshape((-1, 3))
    centers_global = _transform_xyz(centers_lidar, global_from_lidar)
    global_rotation_xy = np.asarray(global_from_lidar, dtype=float)[:2, :2]

    thresholded_indices = [
        index
        for index, box in enumerate(result.detections)
        if box.score >= cfg.score_threshold and box.class_name in cfg.tracked_classes
    ]
    accepted_indices = _nms_indices(result, thresholded_indices, cfg)
    radar_by_detection: dict[int, list[RadarReturn]] = {
        index: [] for index in accepted_indices
    }
    # 一个Doppler回波只能支持一个目标，避免相邻3D框重复消费同一速度证据。
    if accepted_indices and returns:
        accepted_centers = centers_global[accepted_indices, :2]
        for radar in returns:
            distances = np.linalg.norm(
                accepted_centers
                - np.asarray(radar.position_global, dtype=float)[:2],
                axis=1,
            )
            local_index = int(np.argmin(distances))
            if float(distances[local_index]) <= cfg.radar_association_gate_m:
                radar_by_detection[accepted_indices[local_index]].append(radar)

    output: list[FusionDetection] = []
    for index in accepted_indices:
        box = result.detections[index]
        center_global = centers_global[index]
        support = support_counts[index]
        associated_radar = radar_by_detection[index]
        position_std = np.clip(
            cfg.min_position_std_m
            + cfg.score_std_gain_m * (1.0 - box.score)
            + cfg.support_std_gain_m / np.sqrt(support + 1.0),
            cfg.min_position_std_m,
            cfg.max_position_std_m,
        )
        model_velocity_global = None
        if cfg.use_model_velocity and box.velocity_lidar_mps is not None:
            model_velocity_global = (
                global_rotation_xy
                @ np.asarray(box.velocity_lidar_mps, dtype=float)
            )
        output.append(
            FusionDetection(
                timestamp_s=result.timestamp_s,
                position_global=center_global[:2].copy(),
                extent_xy=np.asarray(box.size_lwh_m[:2], dtype=float),
                class_name=box.class_name,
                family=_class_family(box.class_name),
                score=float(box.score),
                position_std_m=float(position_std),
                velocity_global=model_velocity_global,
                velocity_std_mps=(
                    cfg.model_velocity_std_mps
                    if model_velocity_global is not None
                    else None
                ),
                velocity_source=(
                    "bevfusion_velocity_head"
                    if model_velocity_global is not None
                    else None
                ),
                camera_channels=(
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK_RIGHT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                    "CAM_FRONT_LEFT",
                ),
                lidar_support=support,
                radar_returns=associated_radar,
                source_boxes=[{
                    "source": "mmdetection3d_bevfusion",
                    "center_lidar_m": list(box.center_lidar_m),
                    "size_lwh_m": list(box.size_lwh_m),
                    "yaw_lidar_rad": box.yaw_lidar_rad,
                    "checkpoint": result.checkpoint,
                }],
            )
        )
    return output
