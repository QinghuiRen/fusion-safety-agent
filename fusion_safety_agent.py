"""Fusion Safety Agent 的稳定 CARLA Leaderboard 与命令行入口。

Leaderboard 从本文件加载 ``get_entry_point`` 和 ``FusionSafetyLeaderboardAgent``；
产品实现位于 ``fusion_tracking_agent``，本适配器不重复放置算法逻辑。
"""

from fusion_tracking_agent import FusionSafetyLeaderboardAgent, get_entry_point
from fusion_tracking_agent.cli import main

__all__ = ["FusionSafetyLeaderboardAgent", "get_entry_point"]


if __name__ == "__main__":
    raise SystemExit(main())
