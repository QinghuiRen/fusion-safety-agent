"""用于回归评测的确定性长尾传感器故障插件。

插件只变换公开传感器数据包，不能访问 CARLA actor 或场景真值，默认关闭。该隔离
保证故障标签不会成为在线感知、风险判断或控制的隐藏输入。
"""

from __future__ import annotations

import numpy as np


class SensorFaultPlugin:
    def __init__(self, config):
        self.config = config

    def _active(self, frame: int) -> bool:
        if self.config.fault_profile == "none":
            return False
        return frame % self.config.fault_period_frames < self.config.fault_duration_frames

    def apply(self, input_data: dict) -> tuple[dict, dict]:
        output = dict(input_data)
        frame = 0
        for value in input_data.values():
            if isinstance(value, tuple) and len(value) == 2:
                frame = int(value[0])
                break
        active = self._active(frame)
        affected = []
        if not active:
            return output, {"profile": self.config.fault_profile, "active": False, "affected": []}

        profile = self.config.fault_profile
        if profile in {"front_camera_dropout", "mixed"} and "CAM_FRONT" in output:
            value = output["CAM_FRONT"]
            sensor_frame, image = value if isinstance(value, tuple) else (frame, value)
            output["CAM_FRONT"] = (sensor_frame, np.zeros_like(image))
            affected.append("CAM_FRONT")
        if profile == "lidar_dropout" and "LIDAR_TOP" in output:
            value = output["LIDAR_TOP"]
            sensor_frame = value[0] if isinstance(value, tuple) else frame
            # 保持 Leaderboard 公开的 Nx4 LiDAR 数据契约。在线健康监控只能从
            # 空包推断故障，不能直接读取故障插件的标签。
            output["LIDAR_TOP"] = (
                sensor_frame,
                np.empty((0, 4), dtype=np.float32),
            )
            affected.append("LIDAR_TOP")
        if profile in {"radar_dropout", "mixed"}:
            for name in list(output):
                if not name.startswith("RADAR_"):
                    continue
                value = output[name]
                sensor_frame = value[0] if isinstance(value, tuple) else frame
                output[name] = (sensor_frame, np.empty((0, 4), dtype=np.float32))
                affected.append(name)
        return output, {"profile": profile, "active": True, "affected": affected}
