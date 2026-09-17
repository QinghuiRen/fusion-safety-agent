"""基于定位与全局路线的轻量跟随控制器，为安全仲裁提供上游控制请求。"""

from __future__ import annotations

import math
import numpy as np


class RouteController:
    def __init__(self, config):
        self.config = config
        self._route_gps: list[dict] = []
        self._route_world_xy = np.empty((0, 2), dtype=float)
        self._origin: tuple[float, float] | None = None
        self._route_xy = np.empty((0, 2), dtype=float)
        self._progress_index = 0

    def set_global_plan(self, global_plan, global_plan_world_coord=None) -> None:
        self._route_gps = [dict(item[0]) for item in (global_plan or [])]
        world_points = []
        for item in global_plan_world_coord or ():
            transform = item[0]
            location = getattr(transform, "location", transform)
            world_points.append([float(location.x), float(location.y)])
        self._route_world_xy = np.asarray(world_points, dtype=float).reshape(-1, 2)
        self._route_xy = np.empty((0, 2), dtype=float)
        self._origin = None
        self._progress_index = 0

    def _prepare(self, current_xy: np.ndarray) -> None:
        if self._route_xy.size:
            return
        if len(self._route_world_xy):
            # Leaderboard supplies this static global route to every standard
            # Agent. Preserve its dense geometry before the upstream base class
            # downsamples it for high-level commands. CARLA world +y points
            # south, whereas the sensor adapter's local ENU +y points north.
            relative = self._route_world_xy - self._route_world_xy[0]
            relative[:, 1] *= -1.0
            self._route_xy = relative + current_xy
            return
        if not self._route_gps:
            return
        first = self._route_gps[0]
        self._origin = (float(first["lat"]), float(first["lon"]))
        lat0, lon0 = self._origin
        radius = 6_378_137.0
        points = []
        for gps in self._route_gps:
            north = radius * math.radians(float(gps["lat"]) - lat0)
            east = radius * math.cos(math.radians(lat0)) * math.radians(float(gps["lon"]) - lon0)
            points.append([east, north])
        route = np.asarray(points, dtype=float)
        # The adapter's ENU origin is its first observed GPS, not necessarily the
        # first route point. Align route translation using the current nearest start.
        route += current_xy - route[0]
        self._route_xy = route

    def control(self, pose_xyyaw: np.ndarray, speed_mps: float) -> dict:
        pose = np.asarray(pose_xyyaw, dtype=float)
        self._prepare(pose[:2])
        if len(self._route_xy) == 0:
            return {"throttle": 0.0, "brake": 1.0, "steer": 0.0, "hand_brake": False, "reverse": False}
        # Search near the last accepted progress point. A global nearest-point
        # query can jump to a spatially close, later segment on looped routes.
        start = max(0, self._progress_index - 5)
        stop = min(len(self._route_xy), self._progress_index + 101)
        local_distances = np.linalg.norm(self._route_xy[start:stop] - pose[:2], axis=1)
        nearest = start + int(np.argmin(local_distances))
        self._progress_index = max(self._progress_index, nearest)
        nearest = self._progress_index
        cumulative = 0.0
        target_index = nearest
        while target_index + 1 < len(self._route_xy) and cumulative < self.config.route_lookahead_m:
            cumulative += float(np.linalg.norm(self._route_xy[target_index + 1] - self._route_xy[target_index]))
            target_index += 1
        target = self._route_xy[target_index]
        desired_yaw = math.atan2(target[1] - pose[1], target[0] - pose[0])
        heading_error = math.atan2(math.sin(desired_yaw - pose[2]), math.cos(desired_yaw - pose[2]))
        # Mathematical/internal yaw is positive counter-clockwise, whereas
        # CARLA VehicleControl uses positive steer for a right turn.  The sign
        # must therefore be inverted.  Without this conversion a small lateral
        # error becomes positive feedback and quickly saturates the steering.
        steer = float(np.clip(-self.config.steering_gain * heading_error, -1.0, 1.0))
        target_speed = self.config.target_speed_kmh / 3.6
        error = target_speed - float(speed_mps)
        throttle = float(np.clip(self.config.speed_kp * error, 0.0, self.config.max_throttle))
        brake = float(np.clip(-self.config.speed_kp * error, 0.0, self.config.max_route_brake))
        if brake > 0.01:
            throttle = 0.0
        return {"throttle": throttle, "brake": brake, "steer": steer, "hand_brake": False, "reverse": False}
