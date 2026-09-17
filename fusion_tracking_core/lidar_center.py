"""由相机2D框内的 LiDAR 点估计目标二维中心和范围。

``baseline`` 完整保留 Stage4.1 原始逻辑，用于受控对照。
``robust`` 依次执行可回退的地面过滤、深度间隙聚类和 XY 离群点裁剪，
目标是降低前后景混合造成的空间中心跳动，而不是训练新的3D检测器。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LidarCenterConfig:
    """LiDAR 中心估计参数；通常从 Stage4.1 顶部常调参数区构造。"""

    mode: str = "robust"
    min_points: int = 2
    near_depth_percentile: float = 20.0
    baseline_depth_band_m: float = 3.0

    # robust参数：仅在过滤后仍有足够点时生效，否则自动退回未过滤点。
    ground_min_z_m: float = 0.15
    depth_cluster_gap_m: float = 1.0
    depth_cluster_half_width_m: float = 1.5
    # 相机框内同时存在前景/背景深度簇时，优先选择与框内 Radar 回波空间一致的簇。
    # 没有 Radar 证据或最近回波超过该门限时，自动退回原来的近深度策略。
    radar_guidance_gate_m: float = 3.5
    xy_mad_scale: float = 3.5
    xy_mad_floor_m: float = 0.20
    # 0=完全沿用baseline中心，1=完全使用robust中心；中间值减少可见表面偏置。
    robust_center_blend: float = 0.5
    # baseline与robust中心相差过大时，说明2D框内混入了另一个深度层；此时禁止
    # 把受污染的baseline中心混回前景簇。该门限只控制回退，不改变深度聚类。
    max_center_blend_offset_m: float = 2.0

    extent_low_percentile: float = 10.0
    extent_high_percentile: float = 90.0

    def __post_init__(self):
        if self.mode not in {"baseline", "robust"}:
            raise ValueError(f"未知 LiDAR 中心模式: {self.mode}")
        if self.min_points < 1:
            raise ValueError("min_points 必须 >= 1")
        if self.depth_cluster_gap_m <= 0 or self.depth_cluster_half_width_m <= 0:
            raise ValueError("深度聚类阈值必须为正数")
        if self.radar_guidance_gate_m <= 0:
            raise ValueError("Radar 引导门限必须为正数")
        if not 0.0 <= self.robust_center_blend <= 1.0:
            raise ValueError("robust_center_blend 必须在 [0, 1]")
        if self.max_center_blend_offset_m <= 0:
            raise ValueError("max_center_blend_offset_m 必须大于 0")


@dataclass(frozen=True)
class LidarObservation:
    center_xy: np.ndarray
    extent_xy: np.ndarray
    foreground_indices: np.ndarray
    median_depth_m: float
    depth_spread_m: float
    used_ground_filter: bool
    used_xy_trim: bool
    used_radar_guidance: bool = False


def _extent(points_xy: np.ndarray, cfg: LidarCenterConfig) -> np.ndarray:
    lo = np.percentile(points_xy, cfg.extent_low_percentile, axis=0)
    hi = np.percentile(points_xy, cfg.extent_high_percentile, axis=0)
    return np.maximum(hi - lo, [0.6, 0.6])


def _make_observation(
    lidar_xyz: np.ndarray,
    depth: np.ndarray,
    indices: np.ndarray,
    cfg: LidarCenterConfig,
    used_ground_filter: bool,
    used_xy_trim: bool,
    used_radar_guidance: bool = False,
) -> LidarObservation:
    points_xy = lidar_xyz[indices, :2]
    selected_depth = depth[indices]
    return LidarObservation(
        center_xy=np.median(points_xy, axis=0),
        extent_xy=_extent(points_xy, cfg),
        foreground_indices=indices,
        median_depth_m=float(np.median(selected_depth)),
        depth_spread_m=float(np.percentile(selected_depth, 90) - np.percentile(selected_depth, 10)),
        used_ground_filter=used_ground_filter,
        used_xy_trim=used_xy_trim,
        used_radar_guidance=used_radar_guidance,
    )


def _estimate_baseline(
    lidar_xyz: np.ndarray,
    depth: np.ndarray,
    candidate_indices: np.ndarray,
    cfg: LidarCenterConfig,
) -> LidarObservation:
    box_depth = depth[candidate_indices]
    near_depth = np.percentile(box_depth, cfg.near_depth_percentile)
    keep = (
        (box_depth >= max(1.0, near_depth - 0.5))
        & (box_depth <= near_depth + cfg.baseline_depth_band_m)
    )
    selected = candidate_indices[keep]
    if len(selected) < cfg.min_points:
        selected = candidate_indices[np.argsort(box_depth)[: cfg.min_points]]
    return _make_observation(lidar_xyz, depth, selected, cfg, False, False)


def _depth_cluster_indices(
    lidar_xyz: np.ndarray,
    depth: np.ndarray,
    indices: np.ndarray,
    cfg: LidarCenterConfig,
    radar_reference_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """选择最接近稳健近深度锚点的连续深度簇。"""

    order = np.argsort(depth[indices])
    sorted_indices = indices[order]
    sorted_depth = depth[sorted_indices]
    split_at = np.flatnonzero(np.diff(sorted_depth) > cfg.depth_cluster_gap_m) + 1
    clusters = [x for x in np.split(sorted_indices, split_at) if len(x) >= cfg.min_points]
    if not clusters:
        return indices, False

    selected = None
    used_radar_guidance = False
    references = np.asarray(
        [] if radar_reference_xy is None else radar_reference_xy, dtype=float
    ).reshape(-1, 2)
    if len(references):
        # 用各深度簇的稳健 XY 中心匹配同一 2D 框内的 Radar 回波。Radar 只负责
        # 消除深度歧义，不直接替代 LiDAR 中心；超过门限即视为不相关证据。
        cluster_distances = []
        for cluster in clusters:
            center_xy = np.median(lidar_xyz[cluster, :2], axis=0)
            cluster_distances.append(float(np.min(np.linalg.norm(
                references - center_xy[None, :], axis=1
            ))))
        best_index = int(np.argmin(cluster_distances))
        if cluster_distances[best_index] <= cfg.radar_guidance_gate_m:
            selected = clusters[best_index]
            used_radar_guidance = True

    if selected is None:
        anchor = float(np.percentile(depth[indices], cfg.near_depth_percentile))
        selected = min(
            clusters,
            key=lambda x: abs(float(np.median(depth[x])) - anchor),
        )
    median_depth = float(np.median(depth[selected]))
    compact = selected[
        np.abs(depth[selected] - median_depth) <= cfg.depth_cluster_half_width_m
    ]
    selected = compact if len(compact) >= cfg.min_points else selected
    return selected, used_radar_guidance


def _trim_xy_outliers(
    lidar_xyz: np.ndarray,
    indices: np.ndarray,
    cfg: LidarCenterConfig,
) -> tuple[np.ndarray, bool]:
    points_xy = lidar_xyz[indices, :2]
    center = np.median(points_xy, axis=0)
    radius = np.linalg.norm(points_xy - center, axis=1)
    median_radius = float(np.median(radius))
    mad = float(np.median(np.abs(radius - median_radius)))
    limit = median_radius + cfg.xy_mad_scale * max(mad, cfg.xy_mad_floor_m)
    trimmed = indices[radius <= limit]
    if len(trimmed) < cfg.min_points:
        return indices, False
    return trimmed, len(trimmed) < len(indices)


def estimate_lidar_observation(
    lidar_xyz: np.ndarray,
    depth: np.ndarray,
    candidate_indices: np.ndarray,
    cfg: LidarCenterConfig,
    radar_reference_xy: np.ndarray | None = None,
) -> Optional[LidarObservation]:
    """估计一个2D检测框对应的 LiDAR 空间观测。

    参数均使用完整点云的行索引，返回的 ``foreground_indices`` 可直接用于
    调试可视化。点数不足返回 ``None``。
    """

    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if len(candidate_indices) < cfg.min_points:
        return None
    if cfg.mode == "baseline":
        return _estimate_baseline(lidar_xyz, depth, candidate_indices, cfg)

    baseline = _estimate_baseline(lidar_xyz, depth, candidate_indices, cfg)

    ground_kept = candidate_indices[
        lidar_xyz[candidate_indices, 2] >= cfg.ground_min_z_m
    ]
    used_ground_filter = len(ground_kept) >= cfg.min_points and len(ground_kept) < len(candidate_indices)
    working = ground_kept if len(ground_kept) >= cfg.min_points else candidate_indices
    selected, used_radar_guidance = _depth_cluster_indices(
        lidar_xyz,
        depth,
        working,
        cfg,
        radar_reference_xy=radar_reference_xy,
    )
    selected, used_xy_trim = _trim_xy_outliers(lidar_xyz, selected, cfg)
    robust = _make_observation(
        lidar_xyz,
        depth,
        selected,
        cfg,
        used_ground_filter,
        used_xy_trim,
        used_radar_guidance,
    )
    # Radar已明确消除深度歧义，或baseline/robust中心互相矛盾时，不再混回
    # 可能来自路面、护栏或背景的baseline中心。
    center_disagreement = float(np.linalg.norm(
        baseline.center_xy - robust.center_xy
    ))
    alpha = (
        1.0
        if used_radar_guidance
        or center_disagreement > cfg.max_center_blend_offset_m
        else cfg.robust_center_blend
    )
    return LidarObservation(
        center_xy=(1.0 - alpha) * baseline.center_xy + alpha * robust.center_xy,
        # 尺寸描述的是已选前景簇；混入baseline范围会把被排除的背景重新带回，
        # 曾在Town01前车场景产生30m长的虚假车辆尺寸。
        extent_xy=robust.extent_xy,
        foreground_indices=robust.foreground_indices,
        median_depth_m=robust.median_depth_m,
        depth_spread_m=robust.depth_spread_m,
        used_ground_filter=robust.used_ground_filter,
        used_xy_trim=robust.used_xy_trim,
        used_radar_guidance=robust.used_radar_guidance,
    )
