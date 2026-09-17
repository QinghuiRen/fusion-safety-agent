# -*- coding: utf-8 -*-
"""轨迹预测、物理一致性与安全裕度评估。

设计思想受 MarginAD 的 physics-aware / safety-margin-aware / horizon-aware
规划约束启发，但本文件是面向本项目轨迹 schema 的独立 NumPy 实现，不复制其
PyTorch/mmcv 代码，也不依赖训练环境。

输入：单条轨迹的 ego 坐标位置/速度、4x4 CV-KF 协方差、目标尺寸和 ego 速度。
输出：未来位置与不确定性、椭圆安全裕度、风险等级，以及相邻帧物理一致性。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MarginSafetyConfig:
    """常调参数；所有阈值都是工程先验，不声称来自当前数据集直接估计。"""

    horizon_s: float = 3.0
    step_s: float = 0.5
    horizon_weights: tuple[float, ...] = (1.3, 1.0, 0.8, 0.8, 1.0, 1.3)
    ego_length_m: float = 4.5
    ego_width_m: float = 2.0
    base_margin_m: float = 0.5
    uncertainty_sigma_scale: float = 1.5
    # 防止稀疏速度观测把3秒远端包络无限放大；达到上限后仍保留原始std供诊断。
    max_uncertainty_margin_m: float = 4.0
    process_accel_std_mps2: float = 0.5
    max_vehicle_acceleration_mps2: float = 8.0
    max_vru_acceleration_mps2: float = 12.0
    max_vehicle_curvature_inv_m: float = 0.35
    max_vru_curvature_inv_m: float = 1.0
    center_prediction_residual_m: float = 2.0
    min_speed_for_heading_mps: float = 0.5

    def validate(self) -> None:
        if self.horizon_s <= 0 or self.step_s <= 0:
            raise ValueError("horizon_s和step_s必须大于0")
        steps = int(round(self.horizon_s / self.step_s))
        if steps < 1 or not np.isclose(steps * self.step_s, self.horizon_s):
            raise ValueError("horizon_s必须是step_s的整数倍")
        if len(self.horizon_weights) != steps:
            raise ValueError(
                f"horizon_weights长度应为{steps}，当前为{len(self.horizon_weights)}"
            )
        positive = (
            self.ego_length_m,
            self.ego_width_m,
            self.uncertainty_sigma_scale,
            self.max_uncertainty_margin_m,
            self.process_accel_std_mps2,
        )
        if any(value <= 0 for value in positive) or self.base_margin_m < 0:
            raise ValueError("车辆尺寸/不确定性参数必须为正，base_margin_m不可为负")


def _as_xy(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}必须是两个有限数")
    return array


def _safe_covariance(covariance: Sequence[Sequence[float]] | None) -> np.ndarray:
    if covariance is None:
        return np.diag([4.0, 4.0, 16.0, 16.0])
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return np.diag([4.0, 4.0, 16.0, 16.0])
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T


def rotate_cv_covariance(
    covariance: Sequence[Sequence[float]] | None,
    rotation_2d: Sequence[Sequence[float]],
) -> np.ndarray:
    """Rotate a ``[px, py, vx, vy]`` covariance into another XY frame.

    Position and velocity must be expressed in the same coordinate frame as
    their covariance.  This helper is used when global-frame KF tracks are
    converted to the ego frame for safety-envelope prediction.
    """
    rotation = np.asarray(rotation_2d, dtype=float)
    if rotation.shape != (2, 2) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_2d must be a finite 2x2 matrix")
    transform = np.zeros((4, 4), dtype=float)
    transform[:2, :2] = rotation
    transform[2:, 2:] = rotation
    rotated = transform @ _safe_covariance(covariance) @ transform.T
    return 0.5 * (rotated + rotated.T)


def _cv_position_covariance(
    covariance: np.ndarray,
    time_s: float,
    process_accel_std_mps2: float,
) -> np.ndarray:
    """把当前[px,py,vx,vy]协方差传播到未来时刻的位置子空间。"""
    transition = np.array(
        [[1.0, 0.0, time_s, 0.0], [0.0, 1.0, 0.0, time_s],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )
    t2, t3, t4 = time_s**2, time_s**3, time_s**4
    q = process_accel_std_mps2**2
    process = q * np.array(
        [[t4 / 4, 0, t3 / 2, 0], [0, t4 / 4, 0, t3 / 2],
         [t3 / 2, 0, t2, 0], [0, t3 / 2, 0, t2]],
        dtype=float,
    )
    propagated = transition @ covariance @ transition.T + process
    return 0.5 * (propagated[:2, :2] + propagated[:2, :2].T)


def _ellipse_boundary_radius(position: np.ndarray, long_axis: float, lat_axis: float) -> float:
    distance = float(np.linalg.norm(position))
    if distance < 1e-9:
        return min(long_axis, lat_axis)
    direction = position / distance
    denominator = np.sqrt(
        (direction[0] / long_axis) ** 2 + (direction[1] / lat_axis) ** 2
    )
    return float(1.0 / max(denominator, 1e-9))


def predict_margin_safety(
    position_ego: Sequence[float],
    velocity_ego: Sequence[float],
    covariance: Sequence[Sequence[float]] | None,
    extent_xy: Sequence[float],
    ego_velocity_ego: Sequence[float] = (0.0, 0.0),
    config: MarginSafetyConfig | None = None,
) -> dict:
    """预测单目标相对ego轨迹，并计算不确定性感知安全裕度。"""
    config = config or MarginSafetyConfig()
    config.validate()
    position = _as_xy(position_ego, "position_ego")
    velocity = _as_xy(velocity_ego, "velocity_ego")
    ego_velocity = _as_xy(ego_velocity_ego, "ego_velocity_ego")
    extent = np.maximum(_as_xy(extent_xy, "extent_xy"), 0.1)
    cov = _safe_covariance(covariance)

    steps = len(config.horizon_weights)
    times = config.step_s * np.arange(1, steps + 1, dtype=float)
    relative_velocity = velocity - ego_velocity
    future_positions = position[None, :] + times[:, None] * relative_velocity[None, :]

    # 无可靠yaw时用目标BEV对角线的一半作保守圆半径，避免假装掌握朝向。
    target_radius = 0.5 * float(np.linalg.norm(extent))
    std_values, normalized_distances, clearances, step_risks = [], [], [], []
    for time_s, future_position in zip(times, future_positions):
        pos_cov = _cv_position_covariance(
            cov, float(time_s), config.process_accel_std_mps2
        )
        sigma = float(np.sqrt(max(np.linalg.eigvalsh(pos_cov).max(), 1e-9)))
        uncertainty_margin = min(
            config.uncertainty_sigma_scale * sigma,
            config.max_uncertainty_margin_m,
        )
        safe_long = (
            config.ego_length_m / 2 + target_radius
            + config.base_margin_m + uncertainty_margin
        )
        safe_lat = (
            config.ego_width_m / 2 + target_radius
            + config.base_margin_m + uncertainty_margin
        )
        normalized = float(np.sqrt(
            (future_position[0] / safe_long) ** 2
            + (future_position[1] / safe_lat) ** 2
        ))
        boundary = _ellipse_boundary_radius(future_position, safe_long, safe_lat)
        clearance = float(np.linalg.norm(future_position) - boundary)
        step_risk = float(np.clip((1.5 - normalized) / 0.5, 0.0, 1.0))
        std_values.append(sigma)
        normalized_distances.append(normalized)
        clearances.append(clearance)
        step_risks.append(step_risk)

    weights = np.asarray(config.horizon_weights, dtype=float)
    weights = weights / weights.sum()
    risks = np.asarray(step_risks)
    aggregate_risk = 0.7 * float(risks.max()) + 0.3 * float(weights @ risks)
    collision_steps = np.flatnonzero(np.asarray(normalized_distances) <= 1.0)
    time_to_margin = float(times[collision_steps[0]]) if len(collision_steps) else None
    risk_score = 100.0 * aggregate_risk
    if time_to_margin is not None and time_to_margin <= 1.5:
        risk_level = "critical"
    elif risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "model": "uncertainty-aware constant-velocity safety envelope",
        "times_s": times.tolist(),
        "future_positions_ego": future_positions.tolist(),
        "future_position_std_m": std_values,
        "uncertainty_margin_capped_at_m": config.max_uncertainty_margin_m,
        "relative_velocity_ego_mps": relative_velocity.tolist(),
        "normalized_margin_distance": normalized_distances,
        "approx_clearance_m": clearances,
        "step_risk": step_risks,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "predicted_margin_breach": bool(len(collision_steps)),
        "time_to_margin_s": time_to_margin,
        "min_clearance_m": float(min(clearances)),
    }


def assess_kinematic_update(
    previous_position_ego: Sequence[float],
    previous_velocity_ego: Sequence[float],
    current_position_ego: Sequence[float],
    current_velocity_ego: Sequence[float],
    dt_s: float,
    class_name: str,
    config: MarginSafetyConfig | None = None,
) -> dict:
    """评估一次轨迹更新的加速度、曲率和CV预测残差是否物理可疑。"""
    config = config or MarginSafetyConfig()
    config.validate()
    if dt_s <= 0:
        raise ValueError("dt_s必须大于0")
    prev_pos = _as_xy(previous_position_ego, "previous_position_ego")
    prev_vel = _as_xy(previous_velocity_ego, "previous_velocity_ego")
    cur_pos = _as_xy(current_position_ego, "current_position_ego")
    cur_vel = _as_xy(current_velocity_ego, "current_velocity_ego")

    acceleration = float(np.linalg.norm(cur_vel - prev_vel) / dt_s)
    expected_position = prev_pos + prev_vel * dt_s
    prediction_residual = float(np.linalg.norm(cur_pos - expected_position))
    prev_speed, cur_speed = float(np.linalg.norm(prev_vel)), float(np.linalg.norm(cur_vel))
    curvature = 0.0
    if min(prev_speed, cur_speed) >= config.min_speed_for_heading_mps:
        prev_heading = float(np.arctan2(prev_vel[1], prev_vel[0]))
        cur_heading = float(np.arctan2(cur_vel[1], cur_vel[0]))
        heading_delta = float(np.arctan2(
            np.sin(cur_heading - prev_heading), np.cos(cur_heading - prev_heading)
        ))
        travelled = max(0.5 * (prev_speed + cur_speed) * dt_s, 1e-3)
        curvature = abs(heading_delta) / travelled

    is_vru = class_name in {"pedestrian", "bicycle", "motorcycle"}
    max_acceleration = (
        config.max_vru_acceleration_mps2 if is_vru
        else config.max_vehicle_acceleration_mps2
    )
    max_curvature = (
        config.max_vru_curvature_inv_m if is_vru
        else config.max_vehicle_curvature_inv_m
    )
    acceleration_ratio = acceleration / max_acceleration
    curvature_ratio = curvature / max_curvature
    residual_ratio = prediction_residual / config.center_prediction_residual_m
    consistency_score = 100.0 * float(np.exp(
        -0.5 * max(acceleration_ratio - 1.0, 0.0)
        -0.5 * max(curvature_ratio - 1.0, 0.0)
        -0.7 * max(residual_ratio - 1.0, 0.0)
    ))
    flags = []
    if acceleration > max_acceleration:
        flags.append("acceleration_violation")
    if curvature > max_curvature:
        flags.append("curvature_violation")
    if prediction_residual > config.center_prediction_residual_m:
        flags.append("cv_prediction_residual")
    return {
        "acceleration_mps2": acceleration,
        "curvature_inv_m": curvature,
        "cv_prediction_residual_m": prediction_residual,
        "physics_consistency_score": consistency_score,
        "flags": flags,
    }
