"""CARLA 闭环路线、安全性、实时性与舒适性评测指标。

指标只读取 ego、感知和仿真事件，不修改控制量。这样同一场景可以公平比较
manual/autopilot/route_agent，也不会把风险模型与控制策略耦合在一起。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EvaluationConfig


@dataclass
class CollisionIncidentCounter:
    """把CARLA逐帧接触回调合并为独立事故，同时保留原始接触数。"""

    dedup_window_s: float
    contact_count: int = 0
    incident_count: int = 0
    _last_contact_by_actor: dict[int, float] = field(default_factory=dict)
    _incident_by_actor: dict[int, int] = field(default_factory=dict)

    def observe(self, actor_id: int, timestamp_s: float) -> tuple[bool, int]:
        self.contact_count += 1
        last = self._last_contact_by_actor.get(actor_id)
        is_new = last is None or timestamp_s - last > self.dedup_window_s
        if is_new:
            self.incident_count += 1
            self._incident_by_actor[actor_id] = self.incident_count
        self._last_contact_by_actor[actor_id] = float(timestamp_s)
        return is_new, self._incident_by_actor[actor_id]


def route_completion_percent(initial_waypoints: int, remaining_waypoints: int, done: bool) -> float:
    """用 BasicAgent 剩余路线点估计完成率；done 时严格返回 100%。"""

    if done:
        return 100.0
    if initial_waypoints <= 0:
        return 0.0
    completed = 1.0 - remaining_waypoints / initial_waypoints
    return float(np.clip(100.0 * completed, 0.0, 99.9))


def percentile_or_none(values: list[float], percentile: float) -> float | None:
    return None if not values else float(np.percentile(values, percentile))


@dataclass
class LatencyMetricsAccumulator:
    """收集墙钟分段耗时；每项均输出样本数、均值、P50和P95。"""

    samples: dict[str, list[float]] = field(default_factory=dict)

    def observe(self, name: str, value_s: float | None) -> None:
        if value_s is None or not np.isfinite(value_s) or value_s < 0:
            return
        self.samples.setdefault(str(name), []).append(float(value_s))

    def finalize(self) -> dict:
        output = {}
        for name, values in sorted(self.samples.items()):
            output[name] = {
                "samples": len(values),
                "mean_s": float(np.mean(values)),
                "p50_s": float(np.percentile(values, 50)),
                "p95_s": float(np.percentile(values, 95)),
            }
        return output


@dataclass
class ScenarioMetricsAccumulator:
    """逐帧累积驾驶、安全和运行性能指标。"""

    config: EvaluationConfig
    speed_kmh: list[float] = field(default_factory=list)
    acceleration_mps2: list[float] = field(default_factory=list)
    jerk_mps3: list[float] = field(default_factory=list)
    comfort_jerk_mps3: list[float] = field(default_factory=list)
    lane_offset_m: list[float] = field(default_factory=list)
    steering_abs: list[float] = field(default_factory=list)
    risk_scores: list[float] = field(default_factory=list)
    fusion_age_s: list[float] = field(default_factory=list)
    lane_departure_frames: int = 0
    off_driving_lane_frames: int = 0
    _previous_acceleration_xy: np.ndarray | None = None
    _filtered_longitudinal_acceleration_mps2: float | None = None

    def update(
        self,
        *,
        speed_kmh: float,
        acceleration_xy: np.ndarray,
        steering: float,
        lane_offset_m: float | None,
        highest_risk_score: float,
        fusion_age_s: float | None,
        dt_s: float,
        longitudinal_acceleration_mps2: float | None = None,
    ) -> None:
        acceleration_xy = np.asarray(acceleration_xy, dtype=float)
        self.speed_kmh.append(float(speed_kmh))
        self.acceleration_mps2.append(float(np.linalg.norm(acceleration_xy)))
        self.steering_abs.append(abs(float(steering)))
        self.risk_scores.append(float(highest_risk_score))
        if fusion_age_s is not None:
            self.fusion_age_s.append(float(fusion_age_s))
        if lane_offset_m is None:
            self.off_driving_lane_frames += 1
            self.lane_departure_frames += 1
        else:
            offset = abs(float(lane_offset_m))
            self.lane_offset_m.append(offset)
            if offset > self.config.lane_departure_offset_m:
                self.lane_departure_frames += 1
        if self._previous_acceleration_xy is not None and dt_s > 0:
            jerk = np.linalg.norm(acceleration_xy - self._previous_acceleration_xy) / dt_s
            self.jerk_mps3.append(float(jerk))
        self._previous_acceleration_xy = acceleration_xy.copy()
        if longitudinal_acceleration_mps2 is not None and dt_s > 0:
            sample = float(longitudinal_acceleration_mps2)
            if np.isfinite(sample):
                previous = self._filtered_longitudinal_acceleration_mps2
                tau_s = max(float(self.config.comfort_acceleration_filter_tau_s), 0.0)
                alpha = 1.0 if tau_s <= 0.0 else dt_s / (tau_s + dt_s)
                filtered = sample if previous is None else (
                    previous + alpha * (sample - previous)
                )
                if previous is not None:
                    self.comfort_jerk_mps3.append(abs(filtered - previous) / dt_s)
                self._filtered_longitudinal_acceleration_mps2 = float(filtered)

    def finalize(
        self,
        *,
        drive_mode: str,
        route_completion_pct: float | None,
        route_completed: bool,
        collision_count: int,
        sensor_timeout_frames: int,
        simulation_duration_s: float,
        wall_duration_s: float,
        distance_m: float,
        detector_error: str | None,
        termination_reason: str,
    ) -> dict:
        frames = len(self.speed_kmh)
        lane_departure_ratio = self.lane_departure_frames / max(frames, 1)
        critical_ratio = (
            sum(score >= self.config.critical_risk_threshold for score in self.risk_scores)
            / max(frames, 1)
        )
        p95_vector_jerk = percentile_or_none(self.jerk_mps3, 95)
        p95_comfort_jerk = percentile_or_none(self.comfort_jerk_mps3, 95)
        route_ok = (
            drive_mode != "route_agent"
            or (route_completed and (route_completion_pct or 0.0) >= self.config.min_route_completion_pct)
        )
        scenario_pass = bool(
            frames > 0
            and route_ok
            and collision_count <= self.config.max_collision_count
            and sensor_timeout_frames <= self.config.max_sensor_timeout_frames
            and detector_error is None
        )
        return {
            "drive_mode": drive_mode,
            "termination_reason": termination_reason,
            "scenario_pass": scenario_pass,
            "route_completed": bool(route_completed),
            "route_completion_pct": route_completion_pct,
            "distance_m": float(distance_m),
            "collision_count": int(collision_count),
            "collision_per_km": float(collision_count / max(distance_m / 1000.0, 1e-6)),
            "lane_departure_frame_ratio": float(lane_departure_ratio),
            "off_driving_lane_frames": int(self.off_driving_lane_frames),
            "mean_lane_offset_m": None if not self.lane_offset_m else float(np.mean(self.lane_offset_m)),
            "p95_lane_offset_m": percentile_or_none(self.lane_offset_m, 95),
            "mean_speed_kmh": None if not self.speed_kmh else float(np.mean(self.speed_kmh)),
            "max_speed_kmh": None if not self.speed_kmh else float(np.max(self.speed_kmh)),
            "mean_acceleration_mps2": None if not self.acceleration_mps2 else float(np.mean(self.acceleration_mps2)),
            "p95_acceleration_mps2": percentile_or_none(self.acceleration_mps2, 95),
            # 兼容旧字段：它仍是原始二维向量jerk，不改历史含义。
            "p95_jerk_mps3": p95_vector_jerk,
            "p95_vector_jerk_mps3": p95_vector_jerk,
            "p95_comfort_jerk_mps3": p95_comfort_jerk,
            "comfort_jerk_metric": (
                "body_longitudinal_acceleration_first_order_lowpass_"
                f"tau_{self.config.comfort_acceleration_filter_tau_s:.2f}s"
            ),
            "comfort_jerk_pass": (
                p95_comfort_jerk is None
                or p95_comfort_jerk <= self.config.max_p95_jerk_mps3
            ),
            "mean_abs_steering": None if not self.steering_abs else float(np.mean(self.steering_abs)),
            "critical_risk_frame_ratio": float(critical_ratio),
            "mean_fusion_age_s": None if not self.fusion_age_s else float(np.mean(self.fusion_age_s)),
            "p95_fusion_age_s": percentile_or_none(self.fusion_age_s, 95),
            "sensor_timeout_frames": int(sensor_timeout_frames),
            "simulation_duration_s": float(simulation_duration_s),
            "wall_duration_s": float(wall_duration_s),
            "real_time_factor": float(simulation_duration_s / max(wall_duration_s, 1e-6)),
        }
