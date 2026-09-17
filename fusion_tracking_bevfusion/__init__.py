"""MMDetection3D BEVFusion 与现有融合跟踪模块之间的隔离适配层。

本包本身不依赖 torch/mmcv。重型神经网络运行在独立环境中，只通过
``BEVFusionFrameResult`` JSON 契约向当前工程返回 3D 检测结果。
"""

from .adapter import BEVFusionAdapterConfig, detections_from_bevfusion
from .carla_online import CarlaBEVFusionConfig, LatestOnlyBEVFusionWorker
from .schema import BEVFusionBox, BEVFusionFrameResult, BEVFusionSequenceResult

__all__ = [
    "BEVFusionAdapterConfig",
    "BEVFusionBox",
    "CarlaBEVFusionConfig",
    "BEVFusionFrameResult",
    "BEVFusionSequenceResult",
    "LatestOnlyBEVFusionWorker",
    "detections_from_bevfusion",
]
