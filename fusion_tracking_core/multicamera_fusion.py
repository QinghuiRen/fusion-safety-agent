"""多相机空间观测去重与融合。

``baseline`` 精确保留 Stage4.1 的固定距离贪心去重。
``overlap_aware`` 只在相邻相机之间建立重复关系，并联合使用共享 LiDAR 点、
BEV轴对齐范围重叠和中心距离。重复观测不是简单丢弃，而是按检测置信度与
LiDAR点数加权融合，保留多视角来源信息。
"""

from dataclasses import dataclass

import numpy as np


DEFAULT_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)


@dataclass(frozen=True)
class MultiCameraFusionConfig:
    mode: str = "overlap_aware"
    camera_order: tuple = DEFAULT_CAMERA_ORDER

    # baseline固定半径；用于严格复现Stage4.1旧方法。
    baseline_vehicle_radius_m: float = 3.0
    baseline_vru_radius_m: float = 1.3

    # overlap-aware参数。
    lidar_overlap_ratio_min: float = 0.30
    vehicle_center_gate_m: float = 2.5
    vru_center_gate_m: float = 1.0
    bev_iou_min: float = 0.15
    max_depth_difference_m: float = 3.0
    # 0=完全保留最高分主观测；1=完全多视角加权平均。
    multiview_center_blend: float = 0.25

    def __post_init__(self):
        if self.mode not in {"baseline", "overlap_aware"}:
            raise ValueError(f"未知多相机融合模式: {self.mode}")
        if not 0.0 <= self.lidar_overlap_ratio_min <= 1.0:
            raise ValueError("lidar_overlap_ratio_min 必须在 [0, 1]")
        if not 0.0 <= self.bev_iou_min <= 1.0:
            raise ValueError("bev_iou_min 必须在 [0, 1]")
        if not 0.0 <= self.multiview_center_blend <= 1.0:
            raise ValueError("multiview_center_blend 必须在 [0, 1]")


@dataclass(frozen=True)
class MultiCameraFusionStats:
    input_detections: int
    output_detections: int
    merged_detections: int
    multi_view_objects: int
    lidar_overlap_links: int
    geometry_links: int


def _family(name):
    return "vehicle" if name in {"car", "truck", "bus"} else "vru"


def _camera_adjacent(channel_a, channel_b, order):
    if channel_a == channel_b:
        return False
    try:
        ia, ib = order.index(channel_a), order.index(channel_b)
    except ValueError:
        return False
    distance = abs(ia - ib)
    return distance == 1 or distance == len(order) - 1


def _lidar_overlap_ratio(det_a, det_b):
    a = set(np.asarray(det_a.get("lidar_indices", []), dtype=int).tolist())
    b = set(np.asarray(det_b.get("lidar_indices", []), dtype=int).tolist())
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _bev_iou(det_a, det_b):
    center_a = np.asarray(det_a["center_ego"], dtype=float)
    center_b = np.asarray(det_b["center_ego"], dtype=float)
    extent_a = np.maximum(np.asarray(det_a["extent_xy"], dtype=float), 1e-3)
    extent_b = np.maximum(np.asarray(det_b["extent_xy"], dtype=float), 1e-3)
    lo_a, hi_a = center_a - extent_a / 2, center_a + extent_a / 2
    lo_b, hi_b = center_b - extent_b / 2, center_b + extent_b / 2
    intersection_extent = np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0)
    intersection = float(np.prod(intersection_extent))
    union = float(np.prod(extent_a) + np.prod(extent_b) - intersection)
    return intersection / max(union, 1e-9)


def _with_provenance(det):
    result = dict(det)
    channels = det.get("camera_channels", [det["camera_channel"]])
    result["camera_channels"] = list(dict.fromkeys(channels))
    result["n_camera_views"] = len(result["camera_channels"])
    return result


def _baseline(detections, cfg):
    kept = []
    for det in sorted(detections, key=lambda item: item["score"], reverse=True):
        duplicate = False
        for previous in kept:
            if _family(det["class_name"]) != _family(previous["class_name"]):
                continue
            radius = (
                cfg.baseline_vehicle_radius_m
                if _family(det["class_name"]) == "vehicle"
                else cfg.baseline_vru_radius_m
            )
            if np.linalg.norm(det["center_ego"] - previous["center_ego"]) < radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(_with_provenance(det))
    stats = MultiCameraFusionStats(
        input_detections=len(detections),
        output_detections=len(kept),
        merged_detections=len(detections) - len(kept),
        multi_view_objects=0,
        lidar_overlap_links=0,
        geometry_links=0,
    )
    return kept, stats


def _merge_cluster(cluster, cfg):
    if len(cluster) == 1:
        return _with_provenance(cluster[0])

    weights = np.array([
        max(float(det["score"]), 1e-3) * np.sqrt(max(int(det["n_lidar"]), 1))
        for det in cluster
    ])
    weights /= weights.sum()
    primary = max(cluster, key=lambda item: item["score"])
    merged = dict(primary)
    weighted_center = np.sum(
        [weight * det["center_ego"] for weight, det in zip(weights, cluster)], axis=0
    )
    weighted_extent = np.sum(
        [weight * det["extent_xy"] for weight, det in zip(weights, cluster)], axis=0
    )
    alpha = cfg.multiview_center_blend
    merged["center_ego"] = (
        (1.0 - alpha) * primary["center_ego"] + alpha * weighted_center
    )
    merged["extent_xy"] = (
        (1.0 - alpha) * primary["extent_xy"] + alpha * weighted_extent
    )
    merged["depth_m"] = float(sum(
        weight * det["depth_m"] for weight, det in zip(weights, cluster)
    ))
    merged["lidar_depth_spread_m"] = float(sum(
        weight * det.get("lidar_depth_spread_m", 0.0)
        for weight, det in zip(weights, cluster)
    ))
    all_indices = sorted({
        int(index)
        for det in cluster
        for index in np.asarray(det.get("lidar_indices", []), dtype=int)
    })
    merged["lidar_indices"] = np.asarray(all_indices, dtype=int)
    merged["n_lidar"] = len(all_indices) if all_indices else max(det["n_lidar"] for det in cluster)
    merged["camera_channels"] = list(dict.fromkeys(
        channel for det in cluster
        for channel in det.get("camera_channels", [det["camera_channel"]])
    ))
    merged["n_camera_views"] = len(merged["camera_channels"])
    merged["lidar_ground_filtered"] = any(
        det.get("lidar_ground_filtered", False) for det in cluster
    )
    merged["lidar_xy_trimmed"] = any(
        det.get("lidar_xy_trimmed", False) for det in cluster
    )
    return merged


def _overlap_aware(detections, cfg):
    count = len(detections)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    lidar_links = geometry_links = 0
    for i in range(count):
        for j in range(i + 1, count):
            det_a, det_b = detections[i], detections[j]
            if _family(det_a["class_name"]) != _family(det_b["class_name"]):
                continue
            overlap = _lidar_overlap_ratio(det_a, det_b)
            distance = float(np.linalg.norm(det_a["center_ego"] - det_b["center_ego"]))
            iou = _bev_iou(det_a, det_b)
            gate = (
                cfg.vehicle_center_gate_m
                if _family(det_a["class_name"]) == "vehicle"
                else cfg.vru_center_gate_m
            )

            same_camera = det_a["camera_channel"] == det_b["camera_channel"]
            if same_camera:
                # COCO检测器可能对同一目标给出car/truck等同family重复框。
                # 同相机不存在“跨视场重复”，沿用旧半径抑制以消除这类框。
                baseline_gate = (
                    cfg.baseline_vehicle_radius_m
                    if _family(det_a["class_name"]) == "vehicle"
                    else cfg.baseline_vru_radius_m
                )
                if overlap >= cfg.lidar_overlap_ratio_min or distance < baseline_gate:
                    union(i, j)
                    lidar_links += int(overlap >= cfg.lidar_overlap_ratio_min)
                    geometry_links += int(overlap < cfg.lidar_overlap_ratio_min)
                continue

            if not _camera_adjacent(
                det_a["camera_channel"], det_b["camera_channel"], cfg.camera_order
            ):
                continue
            # camera depth是各相机光轴分量，不能跨相机直接比较；统一改用ego径向距离。
            radial_a = float(np.linalg.norm(det_a["center_ego"]))
            radial_b = float(np.linalg.norm(det_b["center_ego"]))
            if abs(radial_a - radial_b) > cfg.max_depth_difference_m:
                continue

            linked_by_lidar = overlap >= cfg.lidar_overlap_ratio_min
            linked_by_geometry = distance <= gate and iou >= cfg.bev_iou_min
            if linked_by_lidar or linked_by_geometry:
                union(i, j)
                lidar_links += int(linked_by_lidar)
                geometry_links += int(linked_by_geometry and not linked_by_lidar)

    clusters = {}
    for index, det in enumerate(detections):
        clusters.setdefault(find(index), []).append(det)
    fused = [_merge_cluster(cluster, cfg) for cluster in clusters.values()]
    fused.sort(key=lambda item: item["score"], reverse=True)
    stats = MultiCameraFusionStats(
        input_detections=count,
        output_detections=len(fused),
        merged_detections=count - len(fused),
        multi_view_objects=sum(len(cluster) > 1 for cluster in clusters.values()),
        lidar_overlap_links=lidar_links,
        geometry_links=geometry_links,
    )
    return fused, stats


def fuse_multicamera_detections(detections, cfg):
    """返回融合后的检测和本帧统计；输入字典不会被原地修改。"""

    if cfg.mode == "baseline":
        return _baseline(detections, cfg)
    return _overlap_aware(detections, cfg)
