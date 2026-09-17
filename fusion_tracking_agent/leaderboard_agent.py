"""面向 CARLA Leaderboard 生命周期的产品 Agent 适配入口。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

try:  # Importable without Leaderboard so configuration and unit tests stay local.
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
except ImportError:  # pragma: no cover - exercised only outside Leaderboard.
    class AutonomousAgent:
        def set_global_plan(self, global_plan_gps, global_plan_world_coord):
            self._global_plan = global_plan_gps
            self._global_plan_world_coord = global_plan_world_coord

    class Track:
        SENSORS = "SENSORS"

from .config import FusionSafetyAgentConfig
from .fault_plugins import SensorFaultPlugin
from .pipeline import FusionSafetyPipeline
from .route_controller import RouteController
from .sensor_adapter import LeaderboardFrameAdapter, sensor_suite


def get_entry_point() -> str:
    return "FusionSafetyLeaderboardAgent"


class FusionSafetyLeaderboardAgent(AutonomousAgent):
    """Standard Agent lifecycle adapted from carla_garage/Leaderboard conventions.

    External code supplies sensor packets and a global plan. This class never
    reads CARLA actors, bounding boxes, or scenario truth for algorithm control.
    """

    track = Track.SENSORS

    def setup(self, path_to_conf_file: str) -> None:
        # setup 只完成配置与有状态模块装配，不读取 CARLA actor 或场景真值。
        self.config = FusionSafetyAgentConfig.from_json(path_to_conf_file)
        self.adapter = LeaderboardFrameAdapter(self.config)
        self.controller = RouteController(self.config)
        self.fault_plugin = SensorFaultPlugin(self.config)
        self.pipeline = FusionSafetyPipeline(self.config)
        if hasattr(self, "_global_plan"):
            self.controller.set_global_plan(
                getattr(self, "_dense_global_plan_gps", self._global_plan),
                getattr(
                    self,
                    "_dense_global_plan_world_coord",
                    getattr(self, "_global_plan_world_coord", None),
                ),
            )

    def sensors(self) -> list[dict]:
        return sensor_suite(self.config)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        # Keep the original dense plan for metric route following. The
        # Leaderboard base class intentionally downsamples its own copy to
        # roughly 200 m high-level command points, which is too sparse for a
        # low-level steering controller and causes corner cutting on Town10.
        dense_gps = list(global_plan_gps or ())
        dense_world = list(global_plan_world_coord or ())
        # The upstream evaluator invokes set_global_plan *before* setup creates
        # our RouteController. Preserve the dense inputs across that lifecycle
        # boundary instead of relying on the base class's downsampled fields.
        self._dense_global_plan_gps = dense_gps
        self._dense_global_plan_world_coord = dense_world
        try:
            super().set_global_plan(global_plan_gps, global_plan_world_coord)
        except AttributeError:
            self._global_plan = global_plan_gps
            self._global_plan_world_coord = global_plan_world_coord
        if hasattr(self, "controller"):
            self.controller.set_global_plan(dense_gps, dense_world)

    @staticmethod
    def _vehicle_control(values: dict):
        try:
            import carla

            control = carla.VehicleControl()
        except ImportError:  # Lets --self-check and tests run in the project venv.
            control = SimpleNamespace()
        control.throttle = float(values.get("throttle", 0.0))
        control.brake = float(values.get("brake", 0.0))
        control.steer = float(values.get("steer", 0.0))
        control.hand_brake = bool(values.get("hand_brake", False))
        control.reverse = bool(values.get("reverse", False))
        return control

    def run_step(self, input_data, timestamp):
        # 单帧主链：故障注入（仅评测）→公共传感器适配→基础路线控制→安全仲裁。
        effective_input, fault_status = self.fault_plugin.apply(input_data)
        adapted = self.adapter.convert(effective_input, float(timestamp))
        speed_mps = float((adapted.ego_velocity_global @ adapted.ego_velocity_global) ** 0.5)
        base = self.controller.control(adapted.ego_pose_xyyaw, speed_mps)
        applied, _ = self.pipeline.step(
            adapted, base, context={"fault_injection": fault_status}
        )
        return self._vehicle_control(applied)

    def __call__(self, sensors=None):
        """Execute one Leaderboard tick without redundant per-frame stdout.

        The upstream implementation prints a wall/simulation timing line on
        every tick.  Our pipeline records the same clocks plus stage latency in
        structured telemetry, so the default path keeps the upstream sensor
        fetch and control semantics while removing synchronous console I/O.
        Set ``quiet_leaderboard_timing=false`` to call the upstream method.
        """

        if not getattr(self.config, "quiet_leaderboard_timing", True):
            return super().__call__(sensors)
        from srunner.scenariomanager.timer import GameTime

        input_data = self.sensor_interface.get_data(GameTime.get_frame())
        control = self.run_step(input_data, GameTime.get_time())
        control.manual_gear_shift = False
        return control

    def destroy(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.close()
