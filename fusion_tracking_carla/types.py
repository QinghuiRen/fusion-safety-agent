"""CARLA 在线模块的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RadarReturn:
    """单个 Radar 回波，已转换到右手 global 坐标。

    为兼容既有JSON契约保留字段名 ``velocity_towards_mps``。CARLA 原始数值在
    实际符号上是视线距离变化率：靠近为负、远离为正。它不是完整二维速度，
    必须结合 ego 速度通过径向观测模型使用。
    """

    position_global: np.ndarray
    sensor_origin_global: np.ndarray
    radial_unit_global: np.ndarray  # 从传感器指向回波的二维单位向量。
    velocity_towards_mps: float
    sensor_name: str


@dataclass
class FusionDetection:
    """统一3D目标观测；可来自手工LiDAR抬升或神经BEV检测前端。"""

    timestamp_s: float
    position_global: np.ndarray
    extent_xy: np.ndarray
    class_name: str
    family: str
    score: float
    position_std_m: float
    # 可选Cartesian速度先验；不是GT。Radar仍以独立的一维径向观测更新KF。
    velocity_global: np.ndarray | None = None
    velocity_std_mps: float | None = None
    velocity_source: str | None = None
    camera_channels: tuple[str, ...] = ()
    lidar_support: int = 0
    radar_returns: list[RadarReturn] = field(default_factory=list)
    source_boxes: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TrackSnapshot:
    """对外结构化轨迹输出；状态和协方差均在右手 global 坐标。"""

    track_id: int
    class_name: str
    family: str
    lifecycle: str
    timestamp_s: float
    state_global: np.ndarray  # [px, py, vx, vy]
    covariance_global: np.ndarray
    extent_xy: np.ndarray
    score: float
    age_updates: int
    missed_updates: int
    lidar_support: int
    radar_support: int
    camera_channels: tuple[str, ...]
    radial_velocity_std_mps: float | None
    # ``timestamp_s`` is the query/prediction epoch.  This separate field
    # identifies the last spatial measurement that actually corrected the KF.
    # Older callers may omit it; in that case the snapshot is treated as fresh.
    measurement_timestamp_s: float | None = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "family": self.family,
            "lifecycle": self.lifecycle,
            "timestamp_s": self.timestamp_s,
            "measurement_timestamp_s": self.measurement_timestamp_s,
            "measurement_age_s": max(
                float(self.timestamp_s - self.measurement_timestamp_s), 0.0
            ) if self.measurement_timestamp_s is not None else 0.0,
            "position_global_m": self.state_global[:2].tolist(),
            "velocity_global_mps": self.state_global[2:].tolist(),
            "extent_xy_m": self.extent_xy.tolist(),
            "covariance_global": self.covariance_global.tolist(),
            "score": self.score,
            "age_updates": self.age_updates,
            "missed_updates": self.missed_updates,
            "lidar_support": self.lidar_support,
            "radar_support": self.radar_support,
            "camera_channels": list(self.camera_channels),
            "radial_velocity_std_mps": self.radial_velocity_std_mps,
        }
