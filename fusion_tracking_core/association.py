"""多目标检测-轨迹关联代价与匈牙利匹配。

baseline复现原始欧氏位置距离+未归一化速度差；mahalanobis使用轨迹预测协方差
和观测协方差归一化创新，并加入弱尺寸一致性项。该模块只返回索引匹配，不管理
轨迹生命周期。
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssociationConfig:
    mode: str = "mahalanobis"
    vehicle_hard_gate_m: float = 5.5
    vru_hard_gate_m: float = 3.0
    baseline_velocity_cost_weight: float = 0.12

    position_mahalanobis_gate: float = 3.0
    velocity_mahalanobis_weight: float = 0.25
    size_log_ratio_weight: float = 0.35
    max_size_log_ratio: float = 2.0

    def __post_init__(self):
        if self.mode not in {"baseline", "mahalanobis"}:
            raise ValueError(f"未知关联模式: {self.mode}")
        if self.position_mahalanobis_gate <= 0:
            raise ValueError("position_mahalanobis_gate必须为正数")


@dataclass(frozen=True)
class AssociationResult:
    matches: list
    unmatched_tracks: list
    unmatched_detections: list
    feasible_pairs: int
    mean_matched_cost: float


def _mahalanobis(delta, covariance):
    covariance = 0.5 * (covariance + covariance.T)
    try:
        solved = np.linalg.solve(covariance, delta)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ delta
    return float(np.sqrt(max(delta @ solved, 0.0)))


def _hard_gate(family, cfg):
    return cfg.vehicle_hard_gate_m if family == "vehicle" else cfg.vru_hard_gate_m


def _pair_cost(track, detection, cfg):
    if track["family"] != detection["family"]:
        return None
    position_delta = np.asarray(track["position"]) - np.asarray(detection["position"])
    position_distance = float(np.linalg.norm(position_delta))
    if position_distance > _hard_gate(detection["family"], cfg):
        return None

    if cfg.mode == "baseline":
        velocity_cost = 0.0
        if detection["has_velocity"]:
            velocity_cost = float(np.linalg.norm(
                np.asarray(track["velocity"]) - np.asarray(detection["velocity"])
            ))
        return position_distance + cfg.baseline_velocity_cost_weight * velocity_cost

    position_covariance = (
        np.asarray(track["position_covariance"], dtype=float)
        + np.eye(2) * float(detection["position_std_m"]) ** 2
    )
    position_maha = _mahalanobis(position_delta, position_covariance)
    if position_maha > cfg.position_mahalanobis_gate:
        return None

    velocity_cost = 0.0
    if detection["has_velocity"]:
        velocity_delta = (
            np.asarray(track["velocity"]) - np.asarray(detection["velocity"])
        )
        velocity_covariance = (
            np.asarray(track["velocity_covariance"], dtype=float)
            + np.eye(2) * float(detection["velocity_std_mps"]) ** 2
        )
        velocity_cost = _mahalanobis(velocity_delta, velocity_covariance)

    track_area = max(float(np.prod(np.asarray(track["extent_xy"]))), 1e-3)
    detection_area = max(float(np.prod(np.asarray(detection["extent_xy"]))), 1e-3)
    size_cost = min(
        abs(float(np.log(detection_area / track_area))), cfg.max_size_log_ratio
    )
    return (
        position_maha
        + cfg.velocity_mahalanobis_weight * velocity_cost
        + cfg.size_log_ratio_weight * size_cost
    )


def associate_tracks(tracks, detections, cfg):
    """执行全局一对一匹配；tracks/detections均为轻量字典视图。"""

    if not tracks or not detections:
        return AssociationResult(
            matches=[],
            unmatched_tracks=list(range(len(tracks))),
            unmatched_detections=list(range(len(detections))),
            feasible_pairs=0,
            mean_matched_cost=float("nan"),
        )

    cost = np.full((len(tracks), len(detections)), 1e6, dtype=float)
    feasible_pairs = 0
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            value = _pair_cost(track, detection, cfg)
            if value is not None and np.isfinite(value):
                cost[track_index, detection_index] = value
                feasible_pairs += 1

    rows, cols = linear_sum_assignment(cost)
    matches = [
        (int(row), int(col))
        for row, col in zip(rows, cols)
        if cost[row, col] < 1e5
    ]
    matched_tracks = {item[0] for item in matches}
    matched_detections = {item[1] for item in matches}
    matched_costs = [cost[row, col] for row, col in matches]
    return AssociationResult(
        matches=matches,
        unmatched_tracks=[i for i in range(len(tracks)) if i not in matched_tracks],
        unmatched_detections=[i for i in range(len(detections)) if i not in matched_detections],
        feasible_pairs=feasible_pairs,
        mean_matched_cost=(
            float(np.mean(matched_costs)) if matched_costs else float("nan")
        ),
    )
