"""六相机检测、LiDAR 3D 抬升、多相机融合和 Radar 支持点分配。"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from fusion_tracking_core.lidar_center import estimate_lidar_observation
from fusion_tracking_core.multicamera_fusion import fuse_multicamera_detections

from .config import CAMERA_CHANNELS, PerceptionConfig, SensorConfig, TrackerConfig
from .geometry import (
    camera_intrinsic,
    carla_xyz_to_internal,
    ego_to_global_xy,
    global_to_ego_xy,
    internal_xyz_to_carla,
    project_carla_world_to_camera,
)
from .sensors import SynchronizedSensorFrame
from .types import FusionDetection, RadarReturn


COCO_TO_TRACKING = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
}

# 只在点云尺寸证据不足时使用的工程先验；单位为ego轴对齐全长/全宽（m）。
# car/truck/bus共用保守道路车辆尺寸，避免COCO类别在相邻帧跳变时尺寸随之振荡。
_FAMILY_EXTENT_PRIOR_M = {
    "vehicle": np.array([4.8, 2.0], dtype=float),
    "motorcycle": np.array([2.1, 0.8], dtype=float),
    "bicycle": np.array([1.8, 0.7], dtype=float),
    "pedestrian": np.array([0.7, 0.7], dtype=float),
}


def _extent_prior_for_class(class_name: str) -> np.ndarray:
    """Return a stable footprint prior without assuming every family is a key.

    ``object_family`` deliberately groups pedestrians, bicycles and motorcycles
    into ``vru`` for tracking/risk policy.  Their physical footprints remain
    class-specific, so the extent table has no generic ``vru`` entry.
    """

    family = object_family(class_name)
    return _FAMILY_EXTENT_PRIOR_M.get(
        family,
        _FAMILY_EXTENT_PRIOR_M.get(class_name, np.array([1.0, 1.0], dtype=float)),
    )


def stabilize_sparse_extent(
    extent_xy: np.ndarray,
    class_name: str,
    lidar_support: int,
    depth_spread_m: float,
    config: PerceptionConfig,
) -> tuple[np.ndarray, bool]:
    """Replace only physically unsupported footprint size with a class-family prior."""

    prior = _extent_prior_for_class(class_name)
    extent = np.asarray(extent_xy, dtype=float).reshape(2)
    incoherent = (
        not np.all(np.isfinite(extent))
        or int(lidar_support) < config.extent_prior_min_lidar_points
        or float(depth_spread_m) > config.extent_incoherent_depth_spread_m
        or np.any(extent > prior * config.extent_prior_max_scale)
    )
    if incoherent:
        return prior.copy(), True
    return np.clip(extent, [0.6, 0.6], prior * config.extent_prior_max_scale), False


@dataclass(frozen=True)
class RadarOnlyLiftObservation:
    """高置信视觉框内、空间一致Radar回波形成的远距位置观测。"""

    center_xy: np.ndarray
    radar_indices: np.ndarray
    radial_spread_m: float


@dataclass(frozen=True)
class GroundPlaneVRUObservation:
    """Calibrated box-footpoint intersection with the local road plane."""

    center_ego: np.ndarray
    range_m: float
    bottom_center_px: np.ndarray


def estimate_ground_plane_vru_observation(
    *,
    box2d: np.ndarray,
    class_name: str,
    visual_score: float,
    camera_world_to_sensor: np.ndarray,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    ground_z_world_carla: float,
    ego_pose_xyyaw: np.ndarray,
    config: PerceptionConfig,
) -> tuple[GroundPlaneVRUObservation | None, str]:
    """Lift a high-confidence VRU box through calibrated flat-ground IPM.

    The bottom-centre image point approximates the road contact point.  Its
    camera ray is transformed by the published sensor extrinsic and
    intersected with the local ego road plane.  This is an intentionally
    uncertain fallback for a dedicated VRU safety channel; it is not learned
    monocular depth and never consumes simulator object truth.
    """

    if not config.ground_plane_vru_lift_enabled:
        return None, "disabled"
    if object_family(class_name) != "vru":
        return None, "not_vru"
    if float(visual_score) < config.ground_plane_vru_min_visual_score:
        return None, "low_visual_score"
    x1, y1, x2, y2 = np.asarray(box2d, dtype=float).reshape(4)
    u_px = float(np.clip(0.5 * (x1 + x2), 0.0, image_width - 1.0))
    v_px = float(np.clip(y2 - 1.0, 0.0, image_height - 1.0))
    horizon_px = float(intrinsic[1, 2])
    if v_px <= horizon_px + config.ground_plane_vru_horizon_margin_px:
        return None, "footpoint_above_horizon"

    # CV camera ray [right, down, forward] -> CARLA sensor [forward,right,up].
    ray_cv = np.linalg.inv(np.asarray(intrinsic, dtype=float)) @ np.array(
        [u_px, v_px, 1.0], dtype=float
    )
    ray_sensor_carla = np.array(
        [ray_cv[2], ray_cv[0], -ray_cv[1]], dtype=float
    )
    sensor_to_world = np.linalg.inv(
        np.asarray(camera_world_to_sensor, dtype=float)
    )
    origin = sensor_to_world[:3, 3]
    direction = sensor_to_world[:3, :3] @ ray_sensor_carla
    if direction[2] >= -1e-6:
        return None, "ray_does_not_hit_ground"
    scale = (float(ground_z_world_carla) - float(origin[2])) / float(direction[2])
    if scale <= 0.0 or not np.isfinite(scale):
        return None, "invalid_ground_intersection"
    point_world_carla = origin + scale * direction
    point_internal = carla_xyz_to_internal(point_world_carla[None, :])[0]
    center_ego = global_to_ego_xy(
        point_internal[None, :2], np.asarray(ego_pose_xyyaw, dtype=float)
    )[0]
    range_m = float(np.linalg.norm(center_ego))
    if not (
        config.ground_plane_vru_min_range_m
        <= range_m
        <= config.ground_plane_vru_max_range_m
    ):
        return None, "outside_range"
    if center_ego[0] <= 0.0:
        return None, "behind_ego"
    if abs(float(center_ego[1])) > config.ground_plane_vru_max_lateral_m:
        return None, "outside_lateral_monitor"
    return GroundPlaneVRUObservation(
        center_ego=np.asarray(center_ego, dtype=float),
        range_m=range_m,
        bottom_center_px=np.array([u_px, v_px], dtype=float),
    ), "accepted"


def estimate_radar_only_observation(
    radar_ego_xy: np.ndarray,
    radar_inside: np.ndarray,
    visual_score: float,
    class_name: str,
    config: PerceptionConfig,
) -> tuple[RadarOnlyLiftObservation | None, str]:
    """在LiDAR抬升失败后尝试Radar几何回退，并返回可审计拒绝原因。

    该函数不做单目深度估计。只有高分2D道路目标框与至少N个投影同框、
    空间聚集的真实Radar回波同时成立时才返回观测；近距仍必须走LiDAR主链。
    """

    if not config.radar_only_lift_enabled:
        return None, "disabled"
    if float(visual_score) < config.radar_only_lift_min_visual_score:
        return None, "low_visual_score"

    radar_ego_xy = np.asarray(radar_ego_xy, dtype=float).reshape(-1, 2)
    inside = np.asarray(radar_inside, dtype=bool).reshape(-1)
    if len(inside) != len(radar_ego_xy):
        raise ValueError("radar_inside长度必须与radar_ego_xy一致")
    indices = np.flatnonzero(inside & np.all(np.isfinite(radar_ego_xy), axis=1))
    min_points = (
        config.radar_only_lift_min_points
        if object_family(class_name) == "vehicle"
        else config.radar_only_lift_vru_min_points
    )
    if len(indices) < min_points:
        return None, "insufficient_radar_points"

    points = radar_ego_xy[indices]
    ranges = np.linalg.norm(points, axis=1)
    far_enough = ranges >= config.radar_only_lift_min_range_m
    indices = indices[far_enough]
    points = points[far_enough]
    if len(indices) < min_points:
        return None, "inside_near_range"

    # 先以二维中位数为锚点，再剔除同一2D框内可能来自道路/邻车的离散回波。
    anchor = np.median(points, axis=0)
    residual = np.linalg.norm(points - anchor[None, :], axis=1)
    coherent = residual <= config.radar_only_lift_cluster_radius_m
    indices = indices[coherent]
    points = points[coherent]
    if len(indices) < min_points:
        return None, "incoherent_radar_cluster"

    center = np.median(points, axis=0)
    radial_ranges = np.linalg.norm(points, axis=1)
    spread = (
        0.0
        if len(points) == 1
        else float(np.percentile(radial_ranges, 90) - np.percentile(radial_ranges, 10))
    )
    return RadarOnlyLiftObservation(
        center_xy=np.asarray(center, dtype=float),
        radar_indices=np.asarray(indices, dtype=int),
        radial_spread_m=spread,
    ), "accepted"


def object_family(class_name: str) -> str:
    return "vehicle" if class_name in {"car", "truck", "bus"} else "vru"


def _ego_body_mask(
    image_width: int,
    image_height: int,
    config: PerceptionConfig,
) -> np.ndarray:
    """根据归一化标定点生成后相机中的自车可见区域掩码。"""

    polygon_norm = np.asarray(config.ego_body_mask_polygon_norm, dtype=float)
    polygon_px = np.rint(
        polygon_norm * np.array([image_width - 1, image_height - 1], dtype=float)
    ).astype(np.int32)
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_px], 1)
    return mask


def mask_ego_body_for_detector(
    image_rgb: np.ndarray,
    channel: str,
    config: PerceptionConfig,
) -> np.ndarray:
    """只修改检测器输入副本，避免把固定可见的自车车身检测为外部车辆。"""

    if channel not in config.ego_body_mask_channels:
        return image_rgb
    height, width = image_rgb.shape[:2]
    mask = _ego_body_mask(width, height, config)
    dilation_px = max(int(config.ego_body_detector_mask_dilation_px), 0)
    if dilation_px:
        kernel_size = 2 * dilation_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    radius = max(int(config.ego_body_detector_inpaint_radius_px), 1)
    return cv2.inpaint(image_rgb, mask * 255, radius, cv2.INPAINT_TELEA)


def is_likely_ego_body_detection(
    detection: dict,
    channel: str,
    image_width: int,
    image_height: int,
    config: PerceptionConfig,
) -> bool:
    """识别固定出现在后相机底部的自车车身误检。

    仅对配置中的相机和vehicle类别生效，并同时要求贴底、足够宽、中心足够低，
    避免因为单一“靠近底边”条件删除正常远处车辆。
    """

    if channel not in config.ego_body_mask_channels:
        return False
    if object_family(detection["class_name"]) != "vehicle":
        return False
    x1, y1, x2, y2 = np.asarray(detection["box2d"], dtype=float)
    center_y_ratio = 0.5 * (y1 + y2) / image_height
    if center_y_ratio < config.ego_body_box_min_center_y_ratio:
        return False

    ix1 = int(np.clip(np.floor(x1), 0, image_width))
    iy1 = int(np.clip(np.floor(y1), 0, image_height))
    ix2 = int(np.clip(np.ceil(x2), 0, image_width))
    iy2 = int(np.clip(np.ceil(y2), 0, image_height))
    if ix2 <= ix1 or iy2 <= iy1:
        return False

    ego_mask = _ego_body_mask(image_width, image_height, config)
    overlap_ratio = float(ego_mask[iy1:iy2, ix1:ix2].mean())
    return overlap_ratio >= config.ego_body_box_min_overlap_ratio


@dataclass(frozen=True)
class PerceptionJob:
    sensor_frame: SynchronizedSensorFrame
    ego_pose_xyyaw: np.ndarray
    ego_z_m: float
    ego_velocity_global: np.ndarray
    submitted_wall_s: float = 0.0


@dataclass(frozen=True)
class PerceptionResult:
    frame: int
    timestamp_s: float
    detections: tuple[FusionDetection, ...]
    camera_boxes: dict[str, tuple[dict, ...]]
    latency_s: float
    ego_velocity_global: np.ndarray
    queue_latency_s: float = 0.0
    preprocess_latency_s: float = 0.0
    detector_latency_s: float = 0.0
    fusion_latency_s: float = 0.0
    error: str | None = None
    # 前端特有的只读诊断；不进入tracker计算。Stage6.6记录原始类别和过滤数量。
    diagnostics: dict | None = None


class CameraRoadUserDetector:
    """torchvision COCO 预训练检测器；只在工作线程中加载一次。"""

    def __init__(self, config: PerceptionConfig):
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            FasterRCNN_ResNet50_FPN_V2_Weights,
            SSDLite320_MobileNet_V3_Large_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
            fasterrcnn_resnet50_fpn_v2,
            ssdlite320_mobilenet_v3_large,
        )

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.detector_model == "fasterrcnn_resnet50_fpn_v2":
            self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=self.weights)
        elif config.detector_model == "fasterrcnn_mobilenet_v3_large_320_fpn":
            self.weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=self.weights)
        elif config.detector_model == "ssdlite320_mobilenet_v3_large":
            self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
        else:
            raise ValueError(f"未知 detector_model: {config.detector_model}")
        self.model = self.model.to(self.device).eval()
        self.categories = self.weights.meta["categories"]
        self.config = config
        self.requested_precision = config.detector_precision
        self.active_precision = (
            "amp_fp16"
            if self.requested_precision == "amp_fp16" and self.device.type == "cuda"
            else "fp32"
        )
        self.amp_fallback_count = 0
        self.amp_fallback_reason: str | None = None
        if self.requested_precision == "amp_fp16" and self.device.type != "cuda":
            self.amp_fallback_count = 1
            self.amp_fallback_reason = (
                f"device={self.device.type}不支持CUDA AMP，已使用FP32"
            )

    def _forward_batch(self, tensors):
        """执行一个检测批次；AMP不兼容时只回退一次并立即用FP32重试。"""

        use_amp = self.active_precision == "amp_fp16"
        try:
            with self.torch.autocast(
                device_type=self.device.type,
                dtype=self.torch.float16,
                enabled=use_amp,
            ):
                return self.model(tensors)
        except (RuntimeError, TypeError) as exc:
            if not use_amp:
                raise
            self.active_precision = "fp32"
            self.amp_fallback_count += 1
            message = str(exc).replace("\n", " ")
            self.amp_fallback_reason = f"{type(exc).__name__}: {message[:400]}"
            # 同一输入立即按稳定基线重试；若错误并非AMP兼容性问题，FP32异常继续上抛。
            with self.torch.autocast(
                device_type=self.device.type,
                dtype=self.torch.float16,
                enabled=False,
            ):
                return self.model(tensors)

    def __call__(self, images_rgb: list[np.ndarray]) -> list[list[dict]]:
        from PIL import Image

        tensors = [
            self.weights.transforms()(Image.fromarray(image)).to(self.device)
            for image in images_rgb
        ]
        outputs = []
        with self.torch.inference_mode():
            for start in range(0, len(tensors), self.config.detector_batch_size):
                outputs.extend(
                    self._forward_batch(
                        tensors[start:start + self.config.detector_batch_size]
                    )
                )
        batches = []
        for output in outputs:
            detections = []
            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            for box, label, score in zip(boxes, labels, scores):
                if score < self.config.detector_score_threshold:
                    break
                coco_name = self.categories[int(label)]
                if coco_name not in COCO_TO_TRACKING:
                    continue
                detections.append({
                    "box2d": box.astype(float),
                    "class_name": COCO_TO_TRACKING[coco_name],
                    "score": float(score),
                })
                if len(detections) >= self.config.detector_max_objects_per_camera:
                    break
            batches.append(detections)
        return batches


class OnlinePerception:
    """把 CARLA 同步帧转换成与 Advanced A 对齐的 FusionDetection。"""

    def __init__(self, perception: PerceptionConfig, sensors: SensorConfig, tracker: TrackerConfig):
        self.config = perception
        self.sensor_config = sensors
        self.tracker_config = tracker
        self.intrinsic = camera_intrinsic(
            sensors.camera_width, sensors.camera_height, sensors.camera_fov_deg
        )
        self.detector = CameraRoadUserDetector(perception)
        self.detector_warmup_latency_s = 0.0
        if perception.detector_warmup_inference:
            # 权重加载不等于 CUDA 已就绪。用与真实相机同尺寸、同批量的空白图像
            # 跑一次完整前向，把 kernel 初始化和显存缓存移到驾驶计时之前。
            warmup_started = time.perf_counter()
            blank = np.zeros(
                (sensors.camera_height, sensors.camera_width, 3), dtype=np.uint8
            )
            warmup_batch = [blank] * max(1, perception.detector_batch_size)
            self.detector(warmup_batch)
            self.detector_warmup_latency_s = time.perf_counter() - warmup_started

    def process(self, job: PerceptionJob) -> PerceptionResult:
        started = time.perf_counter()
        frame = job.sensor_frame
        detector_channels = self.config.detector_camera_channels
        # 检测器使用遮掉自车轮廓的副本；frame 中的原始图像不变，仍供仪表盘显示。
        images = [
            mask_ego_body_for_detector(
                frame.cameras[channel].image_rgb,
                channel,
                self.config,
            )
            for channel in detector_channels
        ]
        preprocess_finished = time.perf_counter()
        detected = self.detector(images)
        detector_finished = time.perf_counter()
        camera_boxes = {
            channel: tuple(
                item for item in items
                if not is_likely_ego_body_detection(
                    item,
                    channel,
                    self.sensor_config.camera_width,
                    self.sensor_config.camera_height,
                    self.config,
                )
            )
            for channel, items in zip(detector_channels, detected)
        }

        world_carla = frame.lidar.points_world_carla
        world_internal = carla_xyz_to_internal(world_carla)
        ego_xy = global_to_ego_xy(world_internal[:, :2], job.ego_pose_xyyaw)
        lidar_ego = np.column_stack([ego_xy, world_internal[:, 2] - job.ego_z_m])
        all_radar = [ret for sample in frame.radars.values() for ret in sample.returns]
        radar_world_internal = np.asarray(
            [ret.position_global for ret in all_radar], dtype=float
        ).reshape(-1, 3)
        radar_world_carla = internal_xyz_to_carla(radar_world_internal)
        radar_ego_xy = (
            global_to_ego_xy(radar_world_internal[:, :2], job.ego_pose_xyyaw)
            if len(radar_world_internal)
            else np.empty((0, 2), dtype=float)
        )
        candidates = []
        radar_only_evidence = []
        lifting_attempts = []
        for channel, detections in camera_boxes.items():
            camera = frame.cameras[channel]
            uv, depth, visible = project_carla_world_to_camera(
                world_carla, camera.world_to_sensor, self.intrinsic
            )
            radar_uv, _, radar_visible = project_carla_world_to_camera(
                radar_world_carla, camera.world_to_sensor, self.intrinsic
            )
            for det in detections:
                x1, y1, x2, y2 = det["box2d"]
                margin = self.config.lidar_projection_margin_px
                inside = visible & (
                    (uv[:, 0] >= x1 + margin) & (uv[:, 0] <= x2 - margin)
                    & (uv[:, 1] >= y1 + margin) & (uv[:, 1] <= y2 - margin)
                )
                indices = np.flatnonzero(inside)
                # 只使用投影到同一个 2D 框内的 Radar 回波来选择 LiDAR 深度簇。
                # 这是证据级引导，不使用 CARLA actor 真值；无回波时保持旧算法回退。
                radar_inside = radar_visible & (
                    (radar_uv[:, 0] >= x1) & (radar_uv[:, 0] <= x2)
                    & (radar_uv[:, 1] >= y1) & (radar_uv[:, 1] <= y2)
                )
                lift_attempt = {
                    "camera_channel": channel,
                    "class_name": det["class_name"],
                    "score": float(det["score"]),
                    "box2d": np.asarray(det["box2d"], dtype=float).tolist(),
                    "projected_lidar_points": int(len(indices)),
                    "projected_radar_points": int(np.count_nonzero(radar_inside)),
                    "decision": "pending",
                }
                lifting_attempts.append(lift_attempt)
                ground_observation, ground_reason = (
                    estimate_ground_plane_vru_observation(
                        box2d=det["box2d"],
                        class_name=det["class_name"],
                        visual_score=det["score"],
                        camera_world_to_sensor=camera.world_to_sensor,
                        intrinsic=self.intrinsic,
                        image_width=self.sensor_config.camera_width,
                        image_height=self.sensor_config.camera_height,
                        ground_z_world_carla=float(job.ego_z_m),
                        ego_pose_xyyaw=job.ego_pose_xyyaw,
                        config=self.config,
                    )
                )
                if self.config.ground_plane_vru_lift_enabled:
                    lift_attempt["ground_plane_decision"] = ground_reason
                    if ground_observation is not None:
                        lift_attempt["ground_plane_bottom_center_px"] = (
                            ground_observation.bottom_center_px.tolist()
                        )
                        lift_attempt["ground_plane_center_ego_m"] = (
                            ground_observation.center_ego.tolist()
                        )
                        lift_attempt["ground_plane_range_m"] = float(
                            ground_observation.range_m
                        )
                observation = estimate_lidar_observation(
                    lidar_ego,
                    depth,
                    indices,
                    self.config.lidar_center,
                    radar_reference_xy=radar_ego_xy[radar_inside],
                )
                if (
                    ground_observation is not None
                    and self.config.ground_plane_vru_prefer_over_lidar
                ):
                    prior = _extent_prior_for_class(det["class_name"])
                    foreground = (
                        np.empty(0, dtype=int)
                        if observation is None else observation.foreground_indices
                    )
                    if observation is not None:
                        lift_attempt["lidar_ground_disagreement_m"] = float(
                            np.linalg.norm(
                                observation.center_xy - ground_observation.center_ego
                            )
                        )
                    lift_attempt["decision"] = (
                        "accepted_ground_plane_vru_with_lidar_support"
                        if observation is not None
                        else "accepted_ground_plane_vru"
                    )
                    lift_attempt["accepted_lidar_points"] = int(len(foreground))
                    candidates.append({
                        "camera_channel": channel,
                        "camera_channels": [channel],
                        "box2d": det["box2d"],
                        "class_name": det["class_name"],
                        "score": det["score"],
                        "center_ego": ground_observation.center_ego,
                        "extent_xy": prior.copy(),
                        "depth_m": float(ground_observation.range_m),
                        "lidar_depth_spread_m": (
                            0.0 if observation is None
                            else float(observation.depth_spread_m)
                        ),
                        "lidar_indices": foreground,
                        "n_lidar": int(len(foreground)),
                        "lidar_ground_filtered": (
                            False if observation is None
                            else bool(observation.used_ground_filter)
                        ),
                        "lidar_xy_trimmed": (
                            False if observation is None
                            else bool(observation.used_xy_trim)
                        ),
                        "lidar_radar_guided": (
                            False if observation is None
                            else bool(observation.used_radar_guidance)
                        ),
                        "lidar_extent_prior": True,
                        "lift_source": (
                            "ground_plane_vru_with_lidar_support"
                            if observation is not None else "ground_plane_vru"
                        ),
                        "position_std_m_override": float(np.clip(
                            self.config.ground_plane_vru_position_std_base_m
                            + self.config.ground_plane_vru_position_std_per_range
                            * ground_observation.range_m,
                            1.0,
                            4.0,
                        )),
                    })
                    continue
                if observation is None:
                    radar_observation, radar_reason = estimate_radar_only_observation(
                        radar_ego_xy,
                        radar_inside,
                        det["score"],
                        det["class_name"],
                        self.config,
                    )
                    if self.config.radar_only_lift_enabled:
                        radar_only_evidence.append({
                            "camera_channel": channel,
                            "class_name": det["class_name"],
                            "score": float(det["score"]),
                            "projected_radar_points": int(np.count_nonzero(radar_inside)),
                            "accepted_radar_points": (
                                0 if radar_observation is None
                                else int(len(radar_observation.radar_indices))
                            ),
                            "decision": radar_reason,
                        })
                    if radar_observation is None:
                        lift_attempt["radar_decision"] = radar_reason
                        if ground_observation is None:
                            lift_attempt["decision"] = "rejected_no_3d_evidence"
                            continue
                        lift_attempt["decision"] = "accepted_ground_plane_vru"
                        prior = _extent_prior_for_class(det["class_name"])
                        candidates.append({
                            "camera_channel": channel,
                            "camera_channels": [channel],
                            "box2d": det["box2d"],
                            "class_name": det["class_name"],
                            "score": det["score"],
                            "center_ego": ground_observation.center_ego,
                            "extent_xy": prior.copy(),
                            "depth_m": float(ground_observation.range_m),
                            "lidar_depth_spread_m": 0.0,
                            "lidar_indices": np.empty(0, dtype=int),
                            "n_lidar": 0,
                            "lidar_ground_filtered": False,
                            "lidar_xy_trimmed": False,
                            "lidar_radar_guided": False,
                            "lidar_extent_prior": True,
                            "lift_source": "ground_plane_vru",
                            "position_std_m_override": float(np.clip(
                                self.config.ground_plane_vru_position_std_base_m
                                + self.config.ground_plane_vru_position_std_per_range
                                * ground_observation.range_m,
                                1.0,
                                4.0,
                            )),
                        })
                        continue
                    lift_attempt["decision"] = "accepted_radar_only"
                    lift_attempt["radar_decision"] = radar_reason
                    prior = _extent_prior_for_class(det["class_name"])
                    candidates.append({
                        "camera_channel": channel,
                        "camera_channels": [channel],
                        "box2d": det["box2d"],
                        "class_name": det["class_name"],
                        "score": det["score"],
                        "center_ego": radar_observation.center_xy,
                        "extent_xy": prior.copy(),
                        "depth_m": float(np.linalg.norm(radar_observation.center_xy)),
                        "lidar_depth_spread_m": 0.0,
                        "lidar_indices": np.empty(0, dtype=int),
                        "n_lidar": 0,
                        "lidar_ground_filtered": False,
                        "lidar_xy_trimmed": False,
                        "lidar_radar_guided": False,
                        "lidar_extent_prior": True,
                        "lift_source": "radar_only",
                        "radar_lift_indices": radar_observation.radar_indices,
                        "radar_radial_spread_m": radar_observation.radial_spread_m,
                        "position_std_m_override": (
                            self.config.radar_only_lift_position_std_m
                            + 0.25 * radar_observation.radial_spread_m
                        ),
                    })
                    continue
                lift_attempt["decision"] = "accepted_lidar"
                lift_attempt["accepted_lidar_points"] = int(
                    len(observation.foreground_indices)
                )
                stable_extent, used_extent_prior = stabilize_sparse_extent(
                    observation.extent_xy,
                    det["class_name"],
                    len(observation.foreground_indices),
                    observation.depth_spread_m,
                    self.config,
                )
                candidates.append({
                    "camera_channel": channel,
                    "camera_channels": [channel],
                    "box2d": det["box2d"],
                    "class_name": det["class_name"],
                    "score": det["score"],
                    "center_ego": observation.center_xy,
                    "extent_xy": stable_extent,
                    "depth_m": observation.median_depth_m,
                    "lidar_depth_spread_m": observation.depth_spread_m,
                    "lidar_indices": observation.foreground_indices,
                    "n_lidar": len(observation.foreground_indices),
                    "lidar_ground_filtered": observation.used_ground_filter,
                    "lidar_xy_trimmed": observation.used_xy_trim,
                    "lidar_radar_guided": observation.used_radar_guidance,
                    "lidar_extent_prior": used_extent_prior,
                    "lift_source": "lidar",
                })

        fused, fusion_stats = fuse_multicamera_detections(
            candidates, self.config.multicamera
        )
        output = []
        for det in fused:
            position_global = ego_to_global_xy(det["center_ego"], job.ego_pose_xyyaw)
            radar_returns = [
                ret for ret in all_radar
                if np.linalg.norm(ret.position_global[:2] - position_global) <= self.tracker_config.radar_association_gate_m
            ]
            spread = float(det.get("lidar_depth_spread_m", 1.0))
            support = max(int(det["n_lidar"]), 1)
            position_std = float(det.get(
                "position_std_m_override",
                np.clip(0.55 + 0.25 * spread + 1.0 / np.sqrt(support), 0.6, 3.0),
            ))
            output.append(FusionDetection(
                timestamp_s=frame.timestamp_s,
                position_global=np.asarray(position_global, dtype=float),
                extent_xy=np.asarray(det["extent_xy"], dtype=float),
                class_name=det["class_name"],
                family=object_family(det["class_name"]),
                score=float(det["score"]),
                position_std_m=position_std,
                camera_channels=tuple(det.get("camera_channels", [det["camera_channel"]])),
                lidar_support=int(det["n_lidar"]),
                radar_returns=radar_returns,
                source_boxes=[{
                    "camera_channel": det["camera_channel"],
                    "box2d": np.asarray(det["box2d"]).tolist(),
                }],
            ))
        finished = time.perf_counter()
        return PerceptionResult(
            frame=frame.frame,
            timestamp_s=frame.timestamp_s,
            detections=tuple(output),
            camera_boxes=camera_boxes,
            latency_s=finished - started,
            ego_velocity_global=job.ego_velocity_global.copy(),
            queue_latency_s=(
                max(started - job.submitted_wall_s, 0.0)
                if job.submitted_wall_s > 0.0 else 0.0
            ),
            preprocess_latency_s=preprocess_finished - started,
            detector_latency_s=detector_finished - preprocess_finished,
            fusion_latency_s=finished - detector_finished,
            diagnostics={
                # legacy视觉前端也遵守与BEVFusion一致的诊断字段契约，避免汇总把
                # “字段缺失”误写成“检测器输出0”。
                "backend_raw_detections": sum(len(items) for items in detected),
                "backend_trackable_raw": sum(
                    len(items) for items in camera_boxes.values()
                ),
                "accepted_detections": len(output),
                "class_counts": {
                    name: sum(det.class_name == name for det in output)
                    for name in sorted({det.class_name for det in output})
                },
                "lidar_lifted_candidates": sum(
                    item.get("lift_source", "lidar") == "lidar" for item in candidates
                ),
                "lidar_geometry_candidates": sum(
                    item.get("lift_source", "lidar") == "lidar" for item in candidates
                ),
                "radar_only_lifted_candidates": sum(
                    item.get("lift_source") == "radar_only" for item in candidates
                ),
                "ground_plane_vru_lifted_candidates": sum(
                    str(item.get("lift_source", "")).startswith("ground_plane_vru")
                    for item in candidates
                ),
                "radar_only_lift_evidence": radar_only_evidence,
                # Includes rejected 2D boxes as well as accepted 3D lifts.  It
                # is diagnostic-only and makes camera-vs-depth failures
                # distinguishable without recording raw images.
                "lifting_attempts": lifting_attempts,
                "multicamera_merged_detections": fusion_stats.merged_detections,
                "lifting_evidence": [
                    {
                        "camera_channel": item["camera_channel"],
                        "class_name": item["class_name"],
                        "score": float(item["score"]),
                        "center_ego": np.asarray(
                            item["center_ego"], dtype=float
                        ).tolist(),
                        "extent_xy": np.asarray(
                            item["extent_xy"], dtype=float
                        ).tolist(),
                        "n_lidar": int(item["n_lidar"]),
                        "median_depth_m": float(item["depth_m"]),
                        "depth_spread_m": float(
                            item.get("lidar_depth_spread_m", 0.0)
                        ),
                        "ground_filtered": bool(
                            item.get("lidar_ground_filtered", False)
                        ),
                        "xy_trimmed": bool(
                            item.get("lidar_xy_trimmed", False)
                        ),
                        "radar_guided": bool(
                            item.get("lidar_radar_guided", False)
                        ),
                        "extent_prior": bool(
                            item.get("lidar_extent_prior", False)
                        ),
                        "lift_source": item.get("lift_source", "lidar"),
                        "radar_lift_support": int(len(
                            item.get("radar_lift_indices", [])
                        )),
                    }
                    for item in candidates
                ],
                "detector_camera_channels": list(detector_channels),
                "detector_camera_count": len(detector_channels),
                "sensor_camera_count": len(frame.cameras),
                "detector_precision_requested": self.detector.requested_precision,
                "detector_precision_active": self.detector.active_precision,
                "detector_amp_fallback_count": self.detector.amp_fallback_count,
                "detector_amp_fallback_reason": self.detector.amp_fallback_reason,
            },
        )


class LatestOnlyPerceptionWorker:
    """单后台线程和容量 1 队列：丢弃旧帧，手动控制绝不等待神经网络。"""

    def __init__(
        self,
        perception: PerceptionConfig,
        sensors: SensorConfig,
        tracker: TrackerConfig,
        *,
        inference_arbiter=None,
        inference_priority: str = "normal",
    ):
        self.perception_config = perception
        self.sensor_config = sensors
        self.tracker_config = tracker
        self.inference_arbiter = inference_arbiter
        self.inference_priority = inference_priority
        self.jobs: queue.Queue[PerceptionJob | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: PerceptionResult | None = None
        self._ready = threading.Event()
        self._initialization_error: str | None = None
        self.startup_diagnostics: dict[str, float] = {}
        self._thread = threading.Thread(target=self._run, name="stage6_1_perception", daemon=True)
        self._thread.start()

    def wait_until_ready(self, timeout_s: float) -> None:
        """等待模型和权重完成加载，避免与 CARLA 第一批渲染帧争抢 GPU。"""

        if not self._ready.wait(timeout_s):
            raise TimeoutError(f"检测器初始化超过 {timeout_s:.0f}s")
        if self._initialization_error is not None:
            raise RuntimeError(self._initialization_error)

    def submit(self, job: PerceptionJob) -> None:
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            self.jobs.put_nowait(job)

    def latest(self) -> PerceptionResult | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        try:
            processor = OnlinePerception(
                self.perception_config, self.sensor_config, self.tracker_config
            )
            self.startup_diagnostics = {
                "detector_warmup_inference_s": processor.detector_warmup_latency_s,
                "detector_precision_requested": processor.detector.requested_precision,
                "detector_precision_active_after_warmup": processor.detector.active_precision,
                "detector_amp_fallback_count_after_warmup": (
                    processor.detector.amp_fallback_count
                ),
                "detector_amp_fallback_reason_after_warmup": (
                    processor.detector.amp_fallback_reason
                ),
            }
        except Exception as exc:
            self._initialization_error = f"检测器加载失败: {type(exc).__name__}: {exc}"
            with self._lock:
                self._latest = PerceptionResult(
                    frame=0,
                    timestamp_s=0.0,
                    detections=(),
                    camera_boxes={},
                    latency_s=0.0,
                    ego_velocity_global=np.zeros(2),
                    error=self._initialization_error,
                )
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                if self.inference_arbiter is None:
                    result = processor.process(job)
                else:
                    with self.inference_arbiter.lease(self.inference_priority):
                        result = processor.process(job)
            except Exception as exc:
                result = PerceptionResult(
                    job.sensor_frame.frame, job.sensor_frame.timestamp_s,
                    (), {}, 0.0, job.ego_velocity_global.copy(),
                    0.0, 0.0, 0.0, 0.0,
                    f"感知帧失败: {type(exc).__name__}: {exc}",
                )
            with self._lock:
                self._latest = result

    def close(self) -> None:
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            self.jobs.put_nowait(None)
        self._thread.join(timeout=3.0)
