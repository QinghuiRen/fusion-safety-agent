"""CARLA 在线传感器、融合跟踪、风险评估与本地仿真运行组件。

产品公共入口为项目根目录的 ``fusion_safety_agent.py``。包内统一使用右手坐标：
x 向前、y 向左、z 向上；位置和速度保存在 global 坐标中，显示和风险计算时再
转换到当前 ego 坐标。部分历史类名仅为兼容既有实验记录而保留。
"""

from .config import Stage61Config, Stage62Config, Stage63Config
from .session import CarlaFusionSession
from .tracker import StreamingFusionTracker
from .types import FusionDetection, RadarReturn, TrackSnapshot

__all__ = [
    "FusionDetection",
    "RadarReturn",
    "Stage61Config",
    "Stage62Config",
    "Stage63Config",
    "CarlaFusionSession",
    "StreamingFusionTracker",
    "TrackSnapshot",
]
