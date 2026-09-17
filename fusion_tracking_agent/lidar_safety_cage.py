"""用于覆盖检测漏检的原始 LiDAR 快速安全笼。

BEVFusion 仍是主要目标感知来源。本模块是不依赖类别的传感器应急通道：在本车路径
走廊内搜索紧凑的离地点簇，并把净空距离与反应距离加制动距离比较。模块不读取
CARLA actor、场景标签、地图对象或碰撞真值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from fusion_tracking_carla.geometry import (
    global_to_ego_xy,
    global_vector_to_ego,
    transform_points,
)


@dataclass(frozen=True)
class LidarSafetyCageConfig:
    """Conservative longitudinal AEB parameters exposed for product tuning."""

    mode: str = "shadow"  # off | shadow | active
    ego_front_offset_m: float = 2.45
    corridor_half_width_m: float = 1.35
    crossing_monitor_half_width_m: float = 4.50
    max_path_distance_m: float = 35.0
    min_height_from_lidar_m: float = -1.65
    max_height_from_lidar_m: float = 0.80
    voxel_size_m: float = 0.35
    min_cluster_points: int = 3
    response_time_s: float = 0.75
    assumed_deceleration_mps2: float = 4.0
    longitudinal_margin_m: float = 1.0
    warning_buffer_m: float = 3.0
    min_ego_speed_mps: float = 1.0
    activation_frames: int = 2
    activation_window_s: float = 0.20
    crossing_min_inward_speed_mps: float = 0.75
    crossing_max_entry_time_s: float = 1.75
    crossing_match_memory_s: float = 0.20
    crossing_match_gate_m: float = 1.25
    crossing_max_cluster_span_m: float = 2.0
    crossing_max_object_speed_mps: float = 4.0
    crossing_velocity_consistency_mps: float = 1.5
    crossing_motion_observations: int = 2
    # Radar is only an early-motion cue: a candidate must also have nearby
    # above-ground LiDAR support.  No single Radar return can request control.
    radar_crossing_enabled: bool = True
    radar_cluster_radius_m: float = 0.9
    radar_match_gate_m: float = 1.5
    radar_lidar_support_radius_m: float = 1.35
    radar_min_lidar_support_points: int = 2
    radar_min_approach_rate_mps: float = 0.12
    radar_min_inward_speed_mps: float = 0.60
    radar_max_object_speed_mps: float = 5.0
    radar_velocity_consistency_mps: float = 2.0
    radar_measurement_consistency_mps: float = 1.5
    radar_motion_observations: int = 2
    radar_max_entry_time_s: float = 2.0
    collision_hold_s: float = 0.60
    emergency_brake: float = 0.65

    def validate(self) -> None:
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError("LiDAR safety cage mode must be off, shadow or active")
        positive = (
            self.ego_front_offset_m,
            self.corridor_half_width_m,
            self.crossing_monitor_half_width_m,
            self.max_path_distance_m,
            self.voxel_size_m,
            self.assumed_deceleration_mps2,
            self.warning_buffer_m,
            self.activation_window_s,
            self.crossing_max_entry_time_s,
            self.crossing_match_memory_s,
            self.crossing_match_gate_m,
            self.crossing_max_cluster_span_m,
            self.crossing_max_object_speed_mps,
            self.crossing_velocity_consistency_mps,
            self.radar_cluster_radius_m,
            self.radar_match_gate_m,
            self.radar_lidar_support_radius_m,
            self.radar_min_approach_rate_mps,
            self.radar_min_inward_speed_mps,
            self.radar_max_object_speed_mps,
            self.radar_velocity_consistency_mps,
            self.radar_measurement_consistency_mps,
            self.radar_max_entry_time_s,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("LiDAR safety cage geometry/dynamics must be positive")
        if self.min_height_from_lidar_m >= self.max_height_from_lidar_m:
            raise ValueError("LiDAR safety cage height interval is invalid")
        if self.crossing_monitor_half_width_m <= self.corridor_half_width_m:
            raise ValueError("crossing monitor must be wider than the path corridor")
        if (
            self.min_cluster_points < 1
            or self.activation_frames < 1
            or self.crossing_motion_observations < 1
            or self.radar_min_lidar_support_points < 1
            or self.radar_motion_observations < 1
        ):
            raise ValueError("LiDAR safety cage frame/cluster counts must be >=1")
        if self.response_time_s < 0.0 or self.longitudinal_margin_m < 0.0:
            raise ValueError("LiDAR safety cage time/margin cannot be negative")
        if self.collision_hold_s < 0.0:
            raise ValueError("LiDAR safety cage hold time cannot be negative")
        if not 0.0 < self.emergency_brake <= 1.0:
            raise ValueError("LiDAR safety cage brake must be in (0,1]")


@dataclass(frozen=True)
class LidarSafetyCageDecision:
    timestamp_s: float
    mode: str
    raw_triggered: bool
    triggered: bool
    held: bool
    new_trigger_event: bool
    reason: str
    eligible_point_count: int
    cluster_count: int
    path_cluster_count: int
    nearest_cluster_points: int | None
    nearest_forward_m: float | None
    nearest_lateral_m: float | None
    clearance_m: float | None
    required_stopping_distance_m: float
    stopping_margin_m: float | None
    hazard_mode: str | None
    crossing_inward_speed_mps: float | None
    crossing_entry_time_s: float | None
    crossing_predicted_clearance_m: float | None
    radar_candidate_count: int
    radar_return_count: int
    radar_longitudinal_range_count: int
    radar_front_monitor_count: int
    radar_unfiltered_forward_min_m: float | None
    radar_unfiltered_forward_max_m: float | None
    radar_unfiltered_min_abs_lateral_m: float | None
    radar_forward_min_m: float | None
    radar_forward_max_m: float | None
    radar_min_abs_lateral_m: float | None
    radar_lidar_supported_count: int
    radar_dynamic_candidate_count: int
    radar_crossing_inward_speed_mps: float | None
    radar_crossing_entry_time_s: float | None
    radar_crossing_predicted_clearance_m: float | None
    radar_motion_debug: tuple[dict, ...]
    risk_score: float
    recommended_throttle_cap: float
    recommended_brake: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _CageMetrics:
    frames: int = 0
    raw_trigger_frames: int = 0
    trigger_frames: int = 0
    trigger_events: int = 0
    applied_frames: int = 0
    applied_events: int = 0
    max_risk_score: float = 0.0
    min_clearance_m: float | None = None
    _last_applied: bool = field(default=False, repr=False)

    def observe(self, decision: LidarSafetyCageDecision) -> None:
        self.frames += 1
        self.raw_trigger_frames += int(decision.raw_triggered)
        self.trigger_frames += int(decision.triggered)
        self.trigger_events += int(decision.new_trigger_event)
        self.max_risk_score = max(self.max_risk_score, decision.risk_score)
        if decision.clearance_m is not None:
            self.min_clearance_m = (
                float(decision.clearance_m)
                if self.min_clearance_m is None
                else min(self.min_clearance_m, float(decision.clearance_m))
            )

    def observe_applied(self, applied: bool) -> None:
        applied = bool(applied)
        self.applied_frames += int(applied)
        self.applied_events += int(applied and not self._last_applied)
        self._last_applied = applied

    def finalize(self) -> dict:
        return {
            "frames": self.frames,
            "raw_trigger_frames": self.raw_trigger_frames,
            "trigger_frames": self.trigger_frames,
            "trigger_events": self.trigger_events,
            "applied_frames": self.applied_frames,
            "applied_events": self.applied_events,
            "max_risk_score": self.max_risk_score,
            "min_clearance_m": self.min_clearance_m,
        }


class LidarPathSafetyCage:
    """20 Hz point-path collision fallback independent of object detection."""

    def __init__(self, config: LidarSafetyCageConfig | None = None):
        self.config = config or LidarSafetyCageConfig()
        self.config.validate()
        self.metrics = _CageMetrics()
        self._activation_count = 0
        self._raw_hit_times: list[float] = []
        self._last_raw_trigger_s: float | None = None
        self._active = False
        self._previous_clusters: list[dict] = []
        self._previous_cluster_timestamp_s: float | None = None
        self._previous_radar_clusters: list[dict] = []
        self._previous_radar_timestamp_s: float | None = None

    @staticmethod
    def _local_points(lidar) -> np.ndarray:
        points = np.asarray(lidar.points_world_carla, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return np.empty((0, 3), dtype=float)
        sensor_to_world = lidar.sensor_to_world
        if sensor_to_world is None:
            return np.empty((0, 3), dtype=float)
        local = transform_points(
            points[:, :3], np.linalg.inv(np.asarray(sensor_to_world, dtype=float))
        )
        return local[np.all(np.isfinite(local), axis=1)]

    def _clusters(self, points: np.ndarray) -> list[dict]:
        """Connected occupied XY voxels; vertical returns share an XY cell."""

        if len(points) == 0:
            return []
        size = float(self.config.voxel_size_m)
        cells: dict[tuple[int, int], list[int]] = {}
        keys = np.floor(points[:, :2] / size).astype(int)
        for index, key in enumerate(keys):
            cells.setdefault((int(key[0]), int(key[1])), []).append(index)
        remaining = set(cells)
        components: list[dict] = []
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            indices: list[int] = []
            while stack:
                cell = stack.pop()
                indices.extend(cells[cell])
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbor = (cell[0] + dx, cell[1] + dy)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            stack.append(neighbor)
            if len(indices) < self.config.min_cluster_points:
                continue
            selected = points[np.asarray(indices, dtype=int)]
            lateral = selected[:, 1]
            centroid = np.median(selected[:, :3], axis=0)
            components.append({
                "points": int(len(selected)),
                "forward_m": float(np.percentile(selected[:, 0], 10.0)),
                "lateral_m": float(np.median(selected[:, 1])),
                "lateral_min_abs_m": float(np.min(np.abs(lateral))),
                "forward_span_m": float(np.ptp(selected[:, 0])),
                "lateral_span_m": float(np.ptp(lateral)),
                "local_centroid_xyz": centroid.tolist(),
            })
        components.sort(key=lambda item: item["forward_m"])
        return components

    def _crossing_candidate(
        self,
        clusters: list[dict],
        timestamp_s: float,
        required_distance_m: float,
        ego_speed_mps: float,
        world_to_sensor_rotation: np.ndarray,
    ) -> dict | None:
        """Estimate lateral entry from sensor-only cluster motion.

        The match is deliberately short-lived.  This lets a VRU approaching
        from behind an occluder trigger before it physically enters the ego
        footprint, while a steady adjacent vehicle has near-zero inward speed.
        """

        cfg = self.config
        previous_time = self._previous_cluster_timestamp_s
        if previous_time is None:
            return None
        dt_s = float(timestamp_s) - float(previous_time)
        if not 1e-3 <= dt_s <= cfg.crossing_match_memory_s:
            return None
        candidates = []
        for current in clusters:
            lateral = float(current["lateral_m"])
            if abs(lateral) <= cfg.corridor_half_width_m:
                continue
            if (
                current["forward_span_m"] > cfg.crossing_max_cluster_span_m
                or current["lateral_span_m"] > cfg.crossing_max_cluster_span_m
            ):
                continue
            best = None
            best_distance = float("inf")
            current_world = np.asarray(current["world_xy"], dtype=float)
            for previous in self._previous_clusters:
                previous_world = np.asarray(previous["world_xy"], dtype=float)
                distance = float(np.linalg.norm(current_world - previous_world))
                if distance < best_distance:
                    best_distance = distance
                    best = previous
            if best is None or best_distance > cfg.crossing_match_gate_m:
                continue
            previous_world = np.asarray(best["world_xy"], dtype=float)
            velocity_world = (current_world - previous_world) / dt_s
            object_speed = float(np.linalg.norm(velocity_world))
            if object_speed > cfg.crossing_max_object_speed_mps:
                continue
            previous_velocity = best.get("velocity_world_xy")
            if previous_velocity is None:
                motion_observations = 1
            elif float(np.linalg.norm(
                velocity_world - np.asarray(previous_velocity, dtype=float)
            )) <= cfg.crossing_velocity_consistency_mps:
                motion_observations = int(best.get("motion_observations", 1)) + 1
            else:
                motion_observations = 1
            current["velocity_world_xy"] = velocity_world.tolist()
            current["motion_observations"] = motion_observations
            velocity_local = np.asarray(world_to_sensor_rotation, dtype=float) @ np.array([
                velocity_world[0], velocity_world[1], 0.0
            ])
            velocity_forward = float(velocity_local[0]) - float(ego_speed_mps)
            velocity_lateral = float(velocity_local[1])
            inward_speed = -float(np.sign(lateral)) * velocity_lateral
            if (
                inward_speed < cfg.crossing_min_inward_speed_mps
                or motion_observations < cfg.crossing_motion_observations
            ):
                continue
            entry_time = max(
                (abs(lateral) - cfg.corridor_half_width_m) / inward_speed, 0.0
            )
            predicted_forward = float(current["forward_m"]) + (
                velocity_forward * entry_time
            )
            predicted_clearance = predicted_forward - cfg.ego_front_offset_m
            if (
                entry_time <= cfg.crossing_max_entry_time_s
                and predicted_forward >= cfg.ego_front_offset_m
                and predicted_clearance <= required_distance_m
            ):
                candidates.append({
                    "cluster": current,
                    "inward_speed_mps": float(inward_speed),
                    "entry_time_s": float(entry_time),
                    "predicted_clearance_m": float(predicted_clearance),
                })
        if not candidates:
            return None
        candidates.sort(key=lambda item: (
            item["entry_time_s"], item["predicted_clearance_m"]
        ))
        return candidates[0]

    def _radar_clusters(
        self,
        radar_returns,
        ego_pose_xyyaw: np.ndarray,
        ego_velocity_global: np.ndarray,
    ) -> tuple[list[dict], dict]:
        """Cluster the current Radar snapshot in the ego-front monitor area.

        Radar positions use the project's right-handed internal global frame,
        while raw LiDAR local Y follows CARLA (right-positive).  Both forms are
        retained explicitly to avoid a silent handedness error.
        """

        cfg = self.config
        rows = []
        diagnostics = {
            "return_count": len(radar_returns or ()),
            "longitudinal_range_count": 0,
            "front_monitor_count": 0,
            "all_forward_values": [],
            "all_abs_lateral_values": [],
            "forward_values": [],
            "abs_lateral_values": [],
        }
        pose = np.asarray(ego_pose_xyyaw, dtype=float)
        ego_velocity = np.asarray(ego_velocity_global, dtype=float)[:2]
        for item in radar_returns or ():
            world_xy = np.asarray(item.position_global, dtype=float)[:2]
            if world_xy.shape != (2,) or not np.all(np.isfinite(world_xy)):
                continue
            local_internal = global_to_ego_xy(world_xy[None, :], pose)[0]
            local_carla = np.array([local_internal[0], -local_internal[1]], dtype=float)
            diagnostics["all_forward_values"].append(float(local_carla[0]))
            diagnostics["all_abs_lateral_values"].append(abs(float(local_carla[1])))
            if not (
                cfg.ego_front_offset_m <= local_carla[0] <= cfg.max_path_distance_m
            ):
                continue
            diagnostics["longitudinal_range_count"] += 1
            diagnostics["forward_values"].append(float(local_carla[0]))
            diagnostics["abs_lateral_values"].append(abs(float(local_carla[1])))
            if abs(local_carla[1]) > cfg.crossing_monitor_half_width_m:
                continue
            diagnostics["front_monitor_count"] += 1
            unit = np.asarray(item.radial_unit_global, dtype=float)[:2]
            norm = float(np.linalg.norm(unit))
            if norm < 1e-6 or not np.all(np.isfinite(unit)):
                continue
            unit = unit / norm
            absolute_radial = float(item.velocity_towards_mps) + float(unit @ ego_velocity)
            if not np.isfinite(absolute_radial):
                continue
            rows.append({
                "world_xy": world_xy,
                "local_carla_xy": local_carla,
                "unit_global_xy": unit,
                "absolute_radial_mps": absolute_radial,
            })
        if not rows:
            return [], diagnostics

        # The filtered front-sector return count is small; an explicit graph
        # keeps this dependency-free and deterministic.
        remaining = set(range(len(rows)))
        components: list[dict] = []
        while remaining:
            seed = remaining.pop()
            stack = [seed]
            selected = []
            while stack:
                index = stack.pop()
                selected.append(index)
                neighbours = [
                    other for other in remaining
                    if float(np.linalg.norm(
                        rows[index]["world_xy"] - rows[other]["world_xy"]
                    )) <= cfg.radar_cluster_radius_m
                ]
                for other in neighbours:
                    remaining.remove(other)
                    stack.append(other)
            world = np.asarray([rows[i]["world_xy"] for i in selected], dtype=float)
            local = np.asarray([rows[i]["local_carla_xy"] for i in selected], dtype=float)
            units = np.asarray([rows[i]["unit_global_xy"] for i in selected], dtype=float)
            radial = np.asarray(
                [rows[i]["absolute_radial_mps"] for i in selected], dtype=float
            )
            unit = np.median(units, axis=0)
            unit_norm = float(np.linalg.norm(unit))
            if unit_norm < 1e-6:
                continue
            components.append({
                "points": len(selected),
                "world_xy": np.median(world, axis=0).tolist(),
                "local_carla_xy": np.median(local, axis=0).tolist(),
                "unit_global_xy": (unit / unit_norm).tolist(),
                "absolute_radial_mps": float(np.median(radial)),
            })
        return components, diagnostics

    def _radar_crossing_candidate(
        self,
        *,
        radar_clusters: list[dict],
        eligible_lidar: np.ndarray,
        timestamp_s: float,
        required_distance_m: float,
        ego_pose_xyyaw: np.ndarray,
    ) -> tuple[dict | None, dict]:
        """Predict path entry from corroborated Radar/LiDAR motion evidence."""

        cfg = self.config
        diagnostics = {
            "candidate_count": len(radar_clusters),
            "lidar_supported_count": 0,
            "dynamic_candidate_count": 0,
            "debug_candidates": [],
        }
        previous_time = self._previous_radar_timestamp_s
        if not cfg.radar_crossing_enabled or previous_time is None:
            return None, diagnostics
        dt_s = float(timestamp_s) - float(previous_time)
        if not 1e-3 <= dt_s <= cfg.crossing_match_memory_s:
            return None, diagnostics

        candidates = []
        for current in radar_clusters:
            local_xy = np.asarray(current["local_carla_xy"], dtype=float)
            lateral = float(local_xy[1])
            if abs(lateral) <= cfg.corridor_half_width_m:
                continue
            if len(eligible_lidar):
                lidar_support = int(np.count_nonzero(
                    np.linalg.norm(eligible_lidar[:, :2] - local_xy[None, :], axis=1)
                    <= cfg.radar_lidar_support_radius_m
                ))
            else:
                lidar_support = 0
            current["lidar_support"] = lidar_support
            if lidar_support < cfg.radar_min_lidar_support_points:
                continue
            diagnostics["lidar_supported_count"] += 1
            debug = {
                "forward_m": float(local_xy[0]),
                "lateral_m": lateral,
                "lidar_support": lidar_support,
                "radar_points": int(current["points"]),
                "measured_radial_mps": float(current["absolute_radial_mps"]),
            }

            current_world = np.asarray(current["world_xy"], dtype=float)
            best = None
            best_distance = float("inf")
            for previous in self._previous_radar_clusters:
                distance = float(np.linalg.norm(
                    current_world - np.asarray(previous["world_xy"], dtype=float)
                ))
                if distance < best_distance:
                    best_distance = distance
                    best = previous
            if best is None or best_distance > cfg.radar_match_gate_m:
                debug["rejection"] = "no_temporal_match"
                debug["match_distance_m"] = None if best is None else best_distance
                diagnostics["debug_candidates"].append(debug)
                continue
            velocity_world = (
                current_world - np.asarray(best["world_xy"], dtype=float)
            ) / dt_s
            object_speed = float(np.linalg.norm(velocity_world))
            debug["match_distance_m"] = best_distance
            debug["object_speed_mps"] = object_speed
            if object_speed > cfg.radar_max_object_speed_mps:
                debug["rejection"] = "object_speed_gate"
                diagnostics["debug_candidates"].append(debug)
                continue
            previous_velocity = best.get("velocity_world_xy")
            if previous_velocity is None:
                motion_observations = 1
            elif float(np.linalg.norm(
                velocity_world - np.asarray(previous_velocity, dtype=float)
            )) <= cfg.radar_velocity_consistency_mps:
                motion_observations = int(best.get("motion_observations", 1)) + 1
            else:
                motion_observations = 1
            current["velocity_world_xy"] = velocity_world.tolist()
            current["motion_observations"] = motion_observations
            debug["motion_observations"] = motion_observations

            unit = np.asarray(current["unit_global_xy"], dtype=float)
            measured_radial = float(current["absolute_radial_mps"])
            predicted_radial = float(unit @ velocity_world)
            debug["predicted_radial_mps"] = predicted_radial
            debug["radial_residual_mps"] = abs(measured_radial - predicted_radial)
            if (
                measured_radial > -cfg.radar_min_approach_rate_mps
                or abs(measured_radial - predicted_radial)
                > cfg.radar_measurement_consistency_mps
            ):
                debug["rejection"] = "radial_measurement_gate"
                diagnostics["debug_candidates"].append(debug)
                continue
            velocity_internal = global_vector_to_ego(
                velocity_world, float(np.asarray(ego_pose_xyyaw, dtype=float)[2])
            )
            velocity_carla = np.array(
                [velocity_internal[0], -velocity_internal[1]], dtype=float
            )
            inward_speed = -float(np.sign(lateral)) * float(velocity_carla[1])
            debug["inward_speed_mps"] = inward_speed
            if (
                inward_speed < cfg.radar_min_inward_speed_mps
                or motion_observations < cfg.radar_motion_observations
            ):
                debug["rejection"] = "inward_or_history_gate"
                diagnostics["debug_candidates"].append(debug)
                continue
            diagnostics["dynamic_candidate_count"] += 1
            entry_time = max(
                (abs(lateral) - cfg.corridor_half_width_m) / inward_speed, 0.0
            )
            predicted_forward = float(local_xy[0]) + float(velocity_carla[0]) * entry_time
            predicted_clearance = predicted_forward - cfg.ego_front_offset_m
            debug["entry_time_s"] = entry_time
            debug["predicted_clearance_m"] = predicted_clearance
            if (
                entry_time <= cfg.radar_max_entry_time_s
                and predicted_forward >= cfg.ego_front_offset_m
                and predicted_clearance <= required_distance_m
            ):
                candidates.append({
                    "cluster": current,
                    "inward_speed_mps": float(inward_speed),
                    "entry_time_s": float(entry_time),
                    "predicted_clearance_m": float(predicted_clearance),
                })
                debug["rejection"] = None
            else:
                debug["rejection"] = "entry_geometry_gate"
            diagnostics["debug_candidates"].append(debug)
        diagnostics["debug_candidates"] = diagnostics["debug_candidates"][:5]
        if not candidates:
            return None, diagnostics
        candidates.sort(key=lambda item: (
            item["entry_time_s"], item["predicted_clearance_m"]
        ))
        return candidates[0], diagnostics

    def evaluate(
        self,
        *,
        timestamp_s: float,
        lidar,
        ego_speed_mps: float,
        input_healthy: bool,
        radar_returns=(),
        ego_pose_xyyaw: np.ndarray | None = None,
        ego_velocity_global: np.ndarray | None = None,
    ) -> LidarSafetyCageDecision:
        cfg = self.config
        speed = max(float(ego_speed_mps), 0.0)
        required = (
            speed * cfg.response_time_s
            + speed * speed / (2.0 * cfg.assumed_deceleration_mps2)
            + cfg.longitudinal_margin_m
        )
        reason = "no_path_cluster"
        eligible = np.empty((0, 3), dtype=float)
        clusters: list[dict] = []
        path_clusters: list[dict] = []
        nearest = None
        crossing = None
        radar_crossing = None
        radar_clusters: list[dict] = []
        radar_diagnostics = {
            "return_count": 0,
            "longitudinal_range_count": 0,
            "front_monitor_count": 0,
            "all_forward_values": [],
            "all_abs_lateral_values": [],
            "forward_values": [],
            "abs_lateral_values": [],
            "candidate_count": 0,
            "lidar_supported_count": 0,
            "dynamic_candidate_count": 0,
            "debug_candidates": [],
        }
        hazard_mode = None
        raw = False
        if cfg.mode == "off":
            reason = "disabled"
        elif not input_healthy:
            reason = "lidar_health_inhibit"
        else:
            local = self._local_points(lidar)
            mask = (
                (local[:, 0] >= cfg.ego_front_offset_m)
                & (local[:, 0] <= cfg.max_path_distance_m)
                & (np.abs(local[:, 1]) <= cfg.crossing_monitor_half_width_m)
                & (local[:, 2] >= cfg.min_height_from_lidar_m)
                & (local[:, 2] <= cfg.max_height_from_lidar_m)
            )
            eligible = local[mask]
            clusters = self._clusters(eligible)
            sensor_to_world = np.asarray(lidar.sensor_to_world, dtype=float)
            for cluster in clusters:
                world = transform_points(
                    np.asarray(cluster["local_centroid_xyz"], dtype=float)[None, :],
                    sensor_to_world,
                )[0]
                cluster["world_xy"] = world[:2].tolist()
            path_clusters = [
                item for item in clusters
                if item["lateral_min_abs_m"] <= cfg.corridor_half_width_m
            ]
            nearest = path_clusters[0] if path_clusters else None
            crossing = self._crossing_candidate(
                clusters,
                float(timestamp_s),
                float(required),
                speed,
                np.linalg.inv(sensor_to_world)[:3, :3],
            )
            if ego_pose_xyyaw is not None and ego_velocity_global is not None:
                radar_clusters, radar_geometry = self._radar_clusters(
                    radar_returns,
                    np.asarray(ego_pose_xyyaw, dtype=float),
                    np.asarray(ego_velocity_global, dtype=float),
                )
                radar_crossing, radar_diagnostics = self._radar_crossing_candidate(
                    radar_clusters=radar_clusters,
                    eligible_lidar=eligible,
                    timestamp_s=float(timestamp_s),
                    required_distance_m=float(required),
                    ego_pose_xyyaw=np.asarray(ego_pose_xyyaw, dtype=float),
                )
                radar_diagnostics.update(radar_geometry)
                radar_diagnostics["candidate_count"] = len(radar_clusters)
            if speed < cfg.min_ego_speed_mps:
                reason = "ego_below_minimum_speed"
            elif nearest is not None:
                clearance = max(
                    float(nearest["forward_m"]) - cfg.ego_front_offset_m, 0.0
                )
                raw = bool(clearance <= required)
                if raw:
                    hazard_mode = "path_occupancy"
                    reason = "braking_distance_violation"
                else:
                    reason = "stopping_margin_safe"
            if not raw and speed >= cfg.min_ego_speed_mps and crossing is not None:
                raw = True
                hazard_mode = "predicted_crossing"
                reason = "predicted_crossing_entry"
            if not raw and speed >= cfg.min_ego_speed_mps and radar_crossing is not None:
                raw = True
                hazard_mode = "radar_lidar_predicted_crossing"
                reason = "radar_lidar_predicted_crossing_entry"

        # Preserve only fresh broad clusters. Empty scans can be bridged for a
        # bounded interval, which is necessary for sparse 32-beam VRU returns.
        if clusters:
            self._previous_clusters = [dict(item) for item in clusters]
            self._previous_cluster_timestamp_s = float(timestamp_s)
        elif (
            self._previous_cluster_timestamp_s is not None
            and float(timestamp_s) - self._previous_cluster_timestamp_s
            > cfg.crossing_match_memory_s
        ):
            self._previous_clusters = []
            self._previous_cluster_timestamp_s = None
        if radar_clusters:
            self._previous_radar_clusters = [dict(item) for item in radar_clusters]
            self._previous_radar_timestamp_s = float(timestamp_s)
        elif (
            self._previous_radar_timestamp_s is not None
            and float(timestamp_s) - self._previous_radar_timestamp_s
            > cfg.crossing_match_memory_s
        ):
            self._previous_radar_clusters = []
            self._previous_radar_timestamp_s = None

        if raw:
            self._raw_hit_times.append(float(timestamp_s))
            self._last_raw_trigger_s = float(timestamp_s)
        self._raw_hit_times = [
            value for value in self._raw_hit_times
            if float(timestamp_s) - value <= cfg.activation_window_s
        ]
        self._activation_count = len(self._raw_hit_times)
        activated = bool(raw and self._activation_count >= cfg.activation_frames)
        held = bool(
            not raw
            and self._active
            and self._last_raw_trigger_s is not None
            and float(timestamp_s) - self._last_raw_trigger_s <= cfg.collision_hold_s
        )
        triggered = bool(activated or held)
        new_event = bool(triggered and not self._active)
        self._active = triggered
        clearance = (
            None
            if nearest is None
            else max(float(nearest["forward_m"]) - cfg.ego_front_offset_m, 0.0)
        )
        margin = None if clearance is None else float(clearance - required)
        risk = 0.0
        if margin is not None:
            risk = float(np.clip(
                100.0 * (cfg.warning_buffer_m - margin) / cfg.warning_buffer_m,
                0.0,
                100.0,
            ))
        if held:
            reason = "collision_hold"
        decision = LidarSafetyCageDecision(
            timestamp_s=float(timestamp_s),
            mode=cfg.mode,
            raw_triggered=raw,
            triggered=triggered,
            held=held,
            new_trigger_event=new_event,
            reason=reason,
            eligible_point_count=int(len(eligible)),
            cluster_count=int(len(clusters)),
            path_cluster_count=int(len(path_clusters)),
            nearest_cluster_points=None if nearest is None else int(nearest["points"]),
            nearest_forward_m=None if nearest is None else float(nearest["forward_m"]),
            nearest_lateral_m=None if nearest is None else float(nearest["lateral_m"]),
            clearance_m=clearance,
            required_stopping_distance_m=float(required),
            stopping_margin_m=margin,
            hazard_mode=hazard_mode,
            crossing_inward_speed_mps=(
                None if crossing is None else float(crossing["inward_speed_mps"])
            ),
            crossing_entry_time_s=(
                None if crossing is None else float(crossing["entry_time_s"])
            ),
            crossing_predicted_clearance_m=(
                None if crossing is None
                else float(crossing["predicted_clearance_m"])
            ),
            radar_candidate_count=int(radar_diagnostics["candidate_count"]),
            radar_return_count=int(radar_diagnostics["return_count"]),
            radar_longitudinal_range_count=int(
                radar_diagnostics["longitudinal_range_count"]
            ),
            radar_front_monitor_count=int(
                radar_diagnostics["front_monitor_count"]
            ),
            radar_unfiltered_forward_min_m=(
                None if not radar_diagnostics["all_forward_values"]
                else float(min(radar_diagnostics["all_forward_values"]))
            ),
            radar_unfiltered_forward_max_m=(
                None if not radar_diagnostics["all_forward_values"]
                else float(max(radar_diagnostics["all_forward_values"]))
            ),
            radar_unfiltered_min_abs_lateral_m=(
                None if not radar_diagnostics["all_abs_lateral_values"]
                else float(min(radar_diagnostics["all_abs_lateral_values"]))
            ),
            radar_forward_min_m=(
                None if not radar_diagnostics["forward_values"]
                else float(min(radar_diagnostics["forward_values"]))
            ),
            radar_forward_max_m=(
                None if not radar_diagnostics["forward_values"]
                else float(max(radar_diagnostics["forward_values"]))
            ),
            radar_min_abs_lateral_m=(
                None if not radar_diagnostics["abs_lateral_values"]
                else float(min(radar_diagnostics["abs_lateral_values"]))
            ),
            radar_lidar_supported_count=int(
                radar_diagnostics["lidar_supported_count"]
            ),
            radar_dynamic_candidate_count=int(
                radar_diagnostics["dynamic_candidate_count"]
            ),
            radar_crossing_inward_speed_mps=(
                None if radar_crossing is None
                else float(radar_crossing["inward_speed_mps"])
            ),
            radar_crossing_entry_time_s=(
                None if radar_crossing is None
                else float(radar_crossing["entry_time_s"])
            ),
            radar_crossing_predicted_clearance_m=(
                None if radar_crossing is None
                else float(radar_crossing["predicted_clearance_m"])
            ),
            radar_motion_debug=tuple(radar_diagnostics["debug_candidates"]),
            risk_score=risk,
            recommended_throttle_cap=0.0 if triggered else 1.0,
            recommended_brake=cfg.emergency_brake if triggered else 0.0,
        )
        self.metrics.observe(decision)
        return decision

    def arbitrate(
        self,
        control: dict,
        decision: LidarSafetyCageDecision,
        *,
        inhibit_reasons: tuple[str, ...] = (),
    ) -> tuple[dict, bool, str]:
        output = dict(control)
        allowed = bool(
            self.config.mode == "active"
            and decision.triggered
            and not inhibit_reasons
        )
        if allowed:
            output["throttle"] = min(
                float(output.get("throttle", 0.0)), decision.recommended_throttle_cap
            )
            output["brake"] = max(
                float(output.get("brake", 0.0)), decision.recommended_brake
            )
            if output["brake"] > 0.01:
                output["throttle"] = 0.0
        changed = bool(
            allowed
            and (
                abs(float(output.get("throttle", 0.0)) - float(control.get("throttle", 0.0)))
                > 1e-6
                or abs(float(output.get("brake", 0.0)) - float(control.get("brake", 0.0)))
                > 1e-6
            )
        )
        self.metrics.observe_applied(changed)
        if inhibit_reasons and decision.triggered:
            reason = "sensor_health_inhibit"
        elif self.config.mode != "active":
            reason = f"{self.config.mode}_mode"
        elif not decision.triggered:
            reason = "no_lidar_path_hazard"
        else:
            reason = "lidar_path_safety_cage"
        return output, changed, reason

    def summary(self) -> dict:
        return {
            "implementation": "class_agnostic_raw_lidar_path_safety_cage",
            "truth_input_used": False,
            "config": asdict(self.config),
            **self.metrics.finalize(),
        }
