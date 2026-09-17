"""把 global 轨迹转换为当前 ego 风险与可靠度输出。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fusion_tracking_core.margin_safety import (
    MarginSafetyConfig,
    predict_margin_safety,
    rotate_cv_covariance,
)

from .geometry import global_to_ego_xy, global_vector_to_ego
from .types import TrackSnapshot


@dataclass(frozen=True)
class TrackSafety:
    track_id: int
    position_ego: np.ndarray
    velocity_ego: np.ndarray
    future_positions_ego: np.ndarray
    future_std_m: np.ndarray
    risk_score: float
    risk_level: str
    time_to_margin_s: float | None
    approx_ttc_s: float | None
    min_clearance_m: float
    evidence_reliability: float
    measurement_age_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "position_ego_m": self.position_ego.tolist(),
            "velocity_ego_mps": self.velocity_ego.tolist(),
            "future_positions_ego_m": self.future_positions_ego.tolist(),
            "future_position_std_m": self.future_std_m.tolist(),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "time_to_margin_s": self.time_to_margin_s,
            "approx_ttc_s": self.approx_ttc_s,
            "min_clearance_m": self.min_clearance_m,
            "evidence_reliability": self.evidence_reliability,
            "measurement_age_s": self.measurement_age_s,
        }


def _approx_ttc(position_ego: np.ndarray, relative_velocity_ego: np.ndarray) -> float | None:
    """点模型最近接近时间，仅作仪表盘提示，不等价于安全裕度 TTC。"""

    speed2 = float(relative_velocity_ego @ relative_velocity_ego)
    if speed2 < 1e-4 or float(position_ego @ relative_velocity_ego) >= 0.0:
        return None
    time_s = -float(position_ego @ relative_velocity_ego) / speed2
    closest = position_ego + time_s * relative_velocity_ego
    return time_s if 0.0 < time_s <= 10.0 and np.linalg.norm(closest) < 4.0 else None


def _reliability(track: TrackSnapshot) -> float:
    """工程启发式证据分，不声称为训练标定概率。"""

    score_term = np.clip(track.score, 0.0, 1.0)
    lidar_term = 1.0 - np.exp(-track.lidar_support / 8.0)
    radar_term = 1.0 - np.exp(-track.radar_support / 2.0)
    position_std = np.sqrt(max(np.linalg.eigvalsh(track.covariance_global[:2, :2]).max(), 1e-9))
    uncertainty_term = np.exp(-position_std / 5.0)
    value = 0.35 * score_term + 0.30 * lidar_term + 0.20 * radar_term + 0.15 * uncertainty_term
    if track.lifecycle == "tentative":
        value *= 0.75
    elif track.lifecycle == "lost":
        value *= 0.55
    return float(np.clip(value, 0.0, 1.0))


def assess_tracks(
    tracks: list[TrackSnapshot],
    ego_pose_xyyaw: np.ndarray,
    ego_velocity_global: np.ndarray,
    config: MarginSafetyConfig,
) -> list[TrackSafety]:
    """计算周围目标相对 ego 的未来 0.5~3 s 位置、安全裕度和可靠度。"""

    yaw = float(ego_pose_xyyaw[2])
    ego_velocity_ego = global_vector_to_ego(ego_velocity_global[:2], yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, s], [-s, c]])
    output = []
    for track in tracks:
        # 单次命中的 tentative 轨迹仍保留在 Track 输出和 BEV 中供审计，
        # 但不能进入安全决策。否则一个低持续性误检就能直接产生
        # critical risk；这也与 oracle 评分默认排除 tentative 的口径保持一致。
        if track.lifecycle == "tentative":
            continue
        position_ego = global_to_ego_xy(track.state_global[:2], ego_pose_xyyaw)
        velocity_ego = global_vector_to_ego(track.state_global[2:], yaw)
        covariance_ego = rotate_cv_covariance(track.covariance_global, rotation)
        safety = predict_margin_safety(
            position_ego=position_ego,
            velocity_ego=velocity_ego,
            covariance=covariance_ego,
            extent_xy=track.extent_xy,
            ego_velocity_ego=ego_velocity_ego,
            config=config,
        )
        relative_velocity = velocity_ego - ego_velocity_ego
        output.append(TrackSafety(
            track_id=track.track_id,
            position_ego=position_ego,
            velocity_ego=velocity_ego,
            future_positions_ego=np.asarray(safety["future_positions_ego"]),
            future_std_m=np.asarray(safety["future_position_std_m"]),
            risk_score=float(safety["risk_score"]),
            risk_level=str(safety["risk_level"]),
            time_to_margin_s=safety["time_to_margin_s"],
            approx_ttc_s=_approx_ttc(position_ego, relative_velocity),
            min_clearance_m=float(safety["min_clearance_m"]),
            evidence_reliability=_reliability(track),
            measurement_age_s=(
                max(float(track.timestamp_s - track.measurement_timestamp_s), 0.0)
                if track.measurement_timestamp_s is not None else 0.0
            ),
        ))
    return sorted(output, key=lambda item: item.risk_score, reverse=True)
