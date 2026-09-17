"""多传感器融合安全 Agent 的稳定公共接口。

该包只暴露必要入口：把 CARLA Leaderboard 数据包接入 BEVFusion、Radar/KF、
轨迹预测和安全监督主链；研究脚本不进入公共导入面。
"""

from .config import FusionSafetyAgentConfig, Stage620AgentConfig
from .leaderboard_agent import FusionSafetyLeaderboardAgent, get_entry_point

__all__ = [
    "FusionSafetyAgentConfig",
    "FusionSafetyLeaderboardAgent",
    "Stage620AgentConfig",
    "get_entry_point",
]
