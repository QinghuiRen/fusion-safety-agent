"""与仿真器无关的多传感器融合跟踪与安全裕度核心模块。

该包逐步承接原型脚本中的可复用算法。顶层 ``stage*.py`` 保留为实验入口，
核心算法放在这里，便于单元测试、替换实现和后续接入 ROS2/CARLA。
"""

from .lidar_center import LidarCenterConfig, LidarObservation, estimate_lidar_observation
from .multicamera_fusion import (
    MultiCameraFusionConfig,
    MultiCameraFusionStats,
    fuse_multicamera_detections,
)
from .radar_velocity import (
    RadarVelocityConfig,
    RadarVelocityObservation,
    estimate_radar_velocity,
)
from .association import AssociationConfig, AssociationResult, associate_tracks
from .margin_safety import (
    MarginSafetyConfig,
    assess_kinematic_update,
    predict_margin_safety,
    rotate_cv_covariance,
)

__all__ = [
    "LidarCenterConfig",
    "LidarObservation",
    "estimate_lidar_observation",
    "MultiCameraFusionConfig",
    "MultiCameraFusionStats",
    "fuse_multicamera_detections",
    "RadarVelocityConfig",
    "RadarVelocityObservation",
    "estimate_radar_velocity",
    "AssociationConfig",
    "AssociationResult",
    "associate_tracks",
    "MarginSafetyConfig",
    "assess_kinematic_update",
    "predict_margin_safety",
    "rotate_cv_covariance",
]
