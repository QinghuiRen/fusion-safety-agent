"""从目标邻域 Radar 点生成 Cartesian 速度观测及不确定性。

nuScenes ``vx_comp/vy_comp`` 是经过处理的二维 Cartesian 速度估计，不是目标
真值，也不应被错误限制为仅沿径向。本模块只做空间关联、离群点抑制和观测
不确定性估计。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RadarVelocityConfig:
    mode: str = "robust"
    base_gate_m: float = 2.5
    gate_scale: float = 0.45

    # baseline：完整复现原来的 median + 2.5/sqrt(n) 协方差。
    baseline_std_mps: float = 2.5
    velocity_std_floor_mps: float = 1.0

    # robust：先保留距最近回波不超过该带宽的点，降低邻车/护栏混入。
    nearest_spatial_band_m: float = 1.5
    velocity_mad_scale: float = 3.0
    velocity_residual_floor_mps: float = 1.0
    single_point_std_mps: float = 4.0
    dispersion_std_scale: float = 1.5
    # robust观测整体保守系数；>1降低Radar速度对KF位置预测的间接影响。
    robust_std_multiplier: float = 1.25
    max_accepted_std_mps: float = 6.0
    max_plausible_speed_mps: float = 60.0

    def __post_init__(self):
        if self.mode not in {"baseline", "robust"}:
            raise ValueError(f"未知Radar速度模式: {self.mode}")
        if self.base_gate_m <= 0 or self.nearest_spatial_band_m <= 0:
            raise ValueError("Radar空间门限必须为正数")
        if self.velocity_mad_scale <= 0:
            raise ValueError("velocity_mad_scale必须为正数")
        if self.robust_std_multiplier < 1.0:
            raise ValueError("robust_std_multiplier建议且必须>=1")


@dataclass(frozen=True)
class RadarVelocityObservation:
    velocity_xy: np.ndarray
    velocity_std_mps: float
    candidate_indices: np.ndarray
    used_indices: np.ndarray
    velocity_dispersion_mps: float
    accepted: bool
    rejection_reason: Optional[str]

    @property
    def n_candidates(self):
        return len(self.candidate_indices)

    @property
    def n_used(self):
        return len(self.used_indices)


def _empty(candidate_indices=None, reason="no_candidate"):
    candidates = np.asarray(
        [] if candidate_indices is None else candidate_indices, dtype=int
    )
    return RadarVelocityObservation(
        velocity_xy=np.zeros(2, dtype=float),
        velocity_std_mps=float("inf"),
        candidate_indices=candidates,
        used_indices=np.empty(0, dtype=int),
        velocity_dispersion_mps=float("nan"),
        accepted=False,
        rejection_reason=reason,
    )


def _dispersion(velocity):
    if len(velocity) <= 1:
        return 0.0
    center = np.median(velocity, axis=0)
    return float(np.median(np.linalg.norm(velocity - center, axis=1)))


def _baseline(radar, candidates, cfg):
    velocity = np.asarray(radar["velocity"])[candidates]
    estimate = np.median(velocity, axis=0)
    std = max(
        cfg.velocity_std_floor_mps,
        cfg.baseline_std_mps / np.sqrt(len(candidates)),
    )
    return RadarVelocityObservation(
        velocity_xy=estimate,
        velocity_std_mps=float(std),
        candidate_indices=candidates,
        used_indices=candidates.copy(),
        velocity_dispersion_mps=_dispersion(velocity),
        accepted=True,
        rejection_reason=None,
    )


def _robust(radar, center_xy, candidates, cfg):
    xy = np.asarray(radar["xy"], dtype=float)
    velocity = np.asarray(radar["velocity"], dtype=float)
    distance = np.linalg.norm(xy[candidates] - center_xy, axis=1)
    nearest_limit = float(distance.min()) + cfg.nearest_spatial_band_m
    used = candidates[distance <= nearest_limit]

    quality = radar.get("quality_valid")
    if quality is not None:
        quality = np.asarray(quality).astype(bool)
        quality_used = used[quality[used]]
        if len(quality_used) > 0:
            used = quality_used

    plausible = np.linalg.norm(velocity[used], axis=1) <= cfg.max_plausible_speed_mps
    used = used[plausible]
    if len(used) == 0:
        return _empty(candidates, "no_plausible_velocity")

    if len(used) >= 3:
        median_velocity = np.median(velocity[used], axis=0)
        residual = np.linalg.norm(velocity[used] - median_velocity, axis=1)
        residual_median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_median)))
        threshold = max(
            cfg.velocity_residual_floor_mps,
            residual_median + cfg.velocity_mad_scale * max(mad, 0.1),
        )
        trimmed = used[residual <= threshold]
        if len(trimmed) > 0:
            used = trimmed

    selected_velocity = velocity[used]
    estimate = np.median(selected_velocity, axis=0)
    dispersion = _dispersion(selected_velocity)
    if len(used) == 1:
        std = cfg.single_point_std_mps
    else:
        std = max(
            cfg.velocity_std_floor_mps,
            cfg.baseline_std_mps / np.sqrt(len(used)),
            cfg.dispersion_std_scale * dispersion,
        )
    std *= cfg.robust_std_multiplier
    if std > cfg.max_accepted_std_mps:
        return RadarVelocityObservation(
            velocity_xy=estimate,
            velocity_std_mps=float(std),
            candidate_indices=candidates,
            used_indices=used,
            velocity_dispersion_mps=dispersion,
            accepted=False,
            rejection_reason="velocity_dispersion_too_large",
        )
    return RadarVelocityObservation(
        velocity_xy=estimate,
        velocity_std_mps=float(std),
        candidate_indices=candidates,
        used_indices=used,
        velocity_dispersion_mps=dispersion,
        accepted=True,
        rejection_reason=None,
    )


def estimate_radar_velocity(radar, center_xy, extent_xy, cfg):
    """计算一个检测目标的Radar速度观测；无可靠观测时返回accepted=False。"""

    center_xy = np.asarray(center_xy, dtype=float)
    extent_xy = np.asarray(extent_xy, dtype=float)
    radar_xy = np.asarray(radar["xy"], dtype=float)
    if len(radar_xy) == 0:
        return _empty()
    gate = max(cfg.base_gate_m, cfg.gate_scale * float(np.linalg.norm(extent_xy)) + 1.0)
    distance = np.linalg.norm(radar_xy - center_xy, axis=1)
    candidates = np.flatnonzero(distance <= gate)
    if len(candidates) == 0:
        return _empty()
    if cfg.mode == "baseline":
        return _baseline(radar, candidates, cfg)
    return _robust(radar, center_xy, candidates, cfg)
