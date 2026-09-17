"""Autoware AEB 思路的无控制权 Shadow 对照器。

本模块复用本项目输出的 ``TrackSnapshot``，不读取 CARLA actor 真值。实现范围是
Autoware AEB 文档中最适合当前接口的三部分：ego 预测足迹、路径碰撞目标筛选、
RSS/制动距离判定与触发保持。它只输出可审计判定，不生成或写入车辆控制。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .types import TrackSnapshot


@dataclass(frozen=True)
class AutowareAEBShadowConfig:
    """常调参数；默认值对齐 Autoware AEB 文档量级而非量产标定。"""

    # 30 km/h 下 1.5 s 只覆盖约 12.5 m，短于本项目的 RSS 制动距离，
    # 会在路径碰撞筛选阶段漏掉本应评估的前车。Stage6.15 官方场景验证后
    # 固定为 4 s / 35 m；计算仍只遍历融合轨迹，不读取场景真值。
    prediction_horizon_s: float = 4.0  # ego/目标预测时域。
    prediction_interval_s: float = 0.1  # 足迹采样周期。
    max_path_length_m: float = 35.0  # 兼容 30 km/h 及响应时间内的制动距离。
    ego_length_m: float = 4.8
    ego_width_m: float = 2.0
    expand_width_m: float = -0.2  # 与官方默认思想一致，负值略收窄误触发区域。
    longitudinal_offset_margin_m: float = 1.0
    response_time_s: float = 1.0
    ego_min_deceleration_mps2: float = 3.0  # 均使用正的减速度幅值。
    object_min_deceleration_mps2: float = 1.0
    collision_keep_s: float = 3.0
    min_ego_speed_mps: float = 0.1
    front_min_center_m: float = 0.5  # 当前产品只做前向纵向监督，不处理倒车AEB。
    min_track_score: float = 0.20
    max_track_missed_updates: int = 3
    max_position_std_m: float = 4.0
    require_confirmed_track: bool = True

    def validate(self) -> None:
        if self.prediction_horizon_s <= 0 or self.prediction_interval_s <= 0:
            raise ValueError("AEB预测时域和采样周期必须>0")
        if self.max_path_length_m <= 0:
            raise ValueError("AEB最大路径长度必须>0")
        if self.ego_length_m <= 0 or self.ego_width_m <= 0:
            raise ValueError("ego尺寸必须>0")
        if self.ego_width_m + 2.0 * self.expand_width_m <= 0:
            raise ValueError("expand_width_m不能令ego有效宽度<=0")
        if self.response_time_s < 0:
            raise ValueError("AEB响应时间不能<0")
        if self.ego_min_deceleration_mps2 <= 0 or self.object_min_deceleration_mps2 <= 0:
            raise ValueError("AEB减速度幅值必须>0")
        if self.collision_keep_s < 0 or self.front_min_center_m < 0:
            raise ValueError("AEB触发保持时间不能<0")
        if not 0 <= self.min_track_score <= 1:
            raise ValueError("AEB轨迹最低分必须在0~1")
        if self.max_track_missed_updates < 0 or self.max_position_std_m <= 0:
            raise ValueError("AEB轨迹失配上限需>=0，位置标准差上限需>0")


@dataclass(frozen=True)
class EgoPathPose:
    time_s: float
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class AutowareAEBShadowDecision:
    timestamp_s: float
    source: str
    control_authority: str
    triggered: bool
    raw_triggered: bool
    held: bool
    new_trigger_event: bool
    reason: str
    target_track_id: int | None
    path_collision: bool
    time_to_path_collision_s: float | None
    longitudinal_clearance_m: float | None
    rss_distance_m: float | None
    rss_margin_m: float | None
    ego_speed_mps: float
    object_longitudinal_speed_mps: float | None
    candidate_count: int
    path_collision_candidate_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _ShadowMetrics:
    frames: int = 0
    raw_trigger_frames: int = 0
    trigger_frames: int = 0
    held_frames: int = 0
    trigger_events: int = 0
    path_collision_frames: int = 0
    target_ids: set[int] = field(default_factory=set)
    max_rss_deficit_m: float = 0.0
    reference_frames: int = 0
    agreement_frames: int = 0
    both_trigger_frames: int = 0
    aeb_only_frames: int = 0
    project_only_frames: int = 0
    neither_frames: int = 0

    def observe(
        self,
        decision: AutowareAEBShadowDecision,
        reference_triggered: bool | None,
    ) -> None:
        self.frames += 1
        self.raw_trigger_frames += int(decision.raw_triggered)
        self.trigger_frames += int(decision.triggered)
        self.held_frames += int(decision.held)
        self.trigger_events += int(decision.new_trigger_event)
        self.path_collision_frames += int(decision.path_collision)
        if decision.target_track_id is not None:
            self.target_ids.add(int(decision.target_track_id))
        if decision.rss_margin_m is not None:
            self.max_rss_deficit_m = max(
                self.max_rss_deficit_m, max(-float(decision.rss_margin_m), 0.0)
            )
        if reference_triggered is None:
            return
        self.reference_frames += 1
        reference = bool(reference_triggered)
        shadow = bool(decision.triggered)
        self.agreement_frames += int(reference == shadow)
        self.both_trigger_frames += int(reference and shadow)
        self.aeb_only_frames += int(shadow and not reference)
        self.project_only_frames += int(reference and not shadow)
        self.neither_frames += int(not reference and not shadow)

    def finalize(self) -> dict:
        return {
            "frames": self.frames,
            "raw_trigger_frames": self.raw_trigger_frames,
            "trigger_frames": self.trigger_frames,
            "held_frames": self.held_frames,
            "trigger_events": self.trigger_events,
            "path_collision_frames": self.path_collision_frames,
            "target_track_ids": sorted(self.target_ids),
            "max_rss_deficit_m": self.max_rss_deficit_m,
            "reference_comparison": {
                "frames": self.reference_frames,
                "agreement_rate": (
                    self.agreement_frames / self.reference_frames
                    if self.reference_frames else None
                ),
                "both_trigger_frames": self.both_trigger_frames,
                "autoware_only_frames": self.aeb_only_frames,
                "project_only_frames": self.project_only_frames,
                "neither_frames": self.neither_frames,
            },
        }


class AutowareAEBShadowEvaluator:
    """以 20 Hz 轨迹快照运行的 AEB Shadow；永远不拥有控制权。"""

    SOURCE = "autoware_aeb_design_shadow"

    def __init__(self, config: AutowareAEBShadowConfig | None = None):
        self.config = config or AutowareAEBShadowConfig()
        self.config.validate()
        self.metrics = _ShadowMetrics()
        self._last_trigger_s: float | None = None
        self._active = False
        self._last_raw_target: dict | None = None

    def _predict_ego_path(
        self,
        ego_pose_xyyaw: np.ndarray,
        ego_velocity_global: np.ndarray,
        ego_yaw_rate_radps: float,
    ) -> list[EgoPathPose]:
        cfg = self.config
        pose = np.asarray(ego_pose_xyyaw, dtype=float)
        velocity = np.asarray(ego_velocity_global, dtype=float)
        forward = np.array([np.cos(pose[2]), np.sin(pose[2])], dtype=float)
        speed = max(float(velocity[:2] @ forward), 0.0)
        times = np.arange(
            0.0,
            cfg.prediction_horizon_s + 0.5 * cfg.prediction_interval_s,
            cfg.prediction_interval_s,
        )
        path: list[EgoPathPose] = []
        x_m, y_m, yaw_rad = map(float, pose)
        travelled = 0.0
        for index, time_s in enumerate(times):
            path.append(EgoPathPose(float(time_s), x_m, y_m, yaw_rad))
            if index == len(times) - 1 or travelled >= cfg.max_path_length_m:
                break
            step = min(
                speed * cfg.prediction_interval_s,
                max(cfg.max_path_length_m - travelled, 0.0),
            )
            if step <= 0.0:
                break
            mid_yaw = yaw_rad + 0.5 * ego_yaw_rate_radps * cfg.prediction_interval_s
            x_m += step * float(np.cos(mid_yaw))
            y_m += step * float(np.sin(mid_yaw))
            yaw_rad += ego_yaw_rate_radps * cfg.prediction_interval_s
            travelled += step
        return path

    def _eligible(self, track: TrackSnapshot) -> bool:
        cfg = self.config
        if cfg.require_confirmed_track and track.lifecycle != "confirmed":
            return False
        if track.lifecycle == "lost" or track.missed_updates > cfg.max_track_missed_updates:
            return False
        if track.score < cfg.min_track_score:
            return False
        covariance = np.asarray(track.covariance_global, dtype=float)
        position_cov = covariance[:2, :2]
        position_std = float(np.sqrt(max(np.linalg.eigvalsh(position_cov).max(), 0.0)))
        return bool(np.isfinite(position_std) and position_std <= cfg.max_position_std_m)

    def _collision_candidate(
        self,
        track: TrackSnapshot,
        path: list[EgoPathPose],
        ego_pose_xyyaw: np.ndarray,
        ego_speed_mps: float,
    ) -> dict | None:
        cfg = self.config
        track_position = np.asarray(track.state_global[:2], dtype=float)
        track_velocity = np.asarray(track.state_global[2:], dtype=float)
        extent = np.maximum(np.asarray(track.extent_xy, dtype=float), 0.2)
        ego_half_length = 0.5 * cfg.ego_length_m
        ego_half_width = 0.5 * cfg.ego_width_m + cfg.expand_width_m
        object_half_length = 0.5 * float(extent[0])
        object_half_width = 0.5 * float(extent[1])

        collision_time = None
        collision_longitudinal = None
        collision_lateral = None
        collision_yaw = float(ego_pose_xyyaw[2])
        for pose in path:
            object_xy = track_position + track_velocity * pose.time_s
            delta = object_xy - np.array([pose.x_m, pose.y_m])
            c, s = float(np.cos(pose.yaw_rad)), float(np.sin(pose.yaw_rad))
            longitudinal = c * float(delta[0]) + s * float(delta[1])
            lateral = -s * float(delta[0]) + c * float(delta[1])
            if (
                abs(longitudinal) <= ego_half_length + object_half_length
                and abs(lateral) <= ego_half_width + object_half_width
            ):
                collision_time = pose.time_s
                collision_longitudinal = longitudinal
                collision_lateral = lateral
                collision_yaw = pose.yaw_rad
                break
        if collision_time is None:
            return None

        current_delta = track_position - np.asarray(ego_pose_xyyaw[:2], dtype=float)
        current_yaw = float(ego_pose_xyyaw[2])
        current_forward = np.array([np.cos(current_yaw), np.sin(current_yaw)])
        center_longitudinal = float(current_delta @ current_forward)
        if center_longitudinal < cfg.front_min_center_m:
            return None
        clearance = center_longitudinal - ego_half_length - object_half_length
        path_forward = np.array([np.cos(collision_yaw), np.sin(collision_yaw)])
        object_speed = float(track_velocity @ path_forward)
        rss_distance = (
            ego_speed_mps * cfg.response_time_s
            + ego_speed_mps**2 / (2.0 * cfg.ego_min_deceleration_mps2)
            - np.sign(object_speed)
            * object_speed**2
            / (2.0 * cfg.object_min_deceleration_mps2)
            + cfg.longitudinal_offset_margin_m
        )
        rss_distance = max(float(rss_distance), 0.0)
        return {
            "track_id": int(track.track_id),
            "time_to_collision_s": float(collision_time),
            "path_longitudinal_m": float(collision_longitudinal),
            "path_lateral_m": float(collision_lateral),
            "clearance_m": float(clearance),
            "rss_distance_m": rss_distance,
            "rss_margin_m": float(clearance - rss_distance),
            "object_speed_mps": object_speed,
        }

    def evaluate(
        self,
        *,
        timestamp_s: float,
        tracks: list[TrackSnapshot],
        ego_pose_xyyaw: np.ndarray,
        ego_velocity_global: np.ndarray,
        ego_yaw_rate_radps: float = 0.0,
        reference_intervention_requested: bool | None = None,
    ) -> AutowareAEBShadowDecision:
        cfg = self.config
        ego_pose = np.asarray(ego_pose_xyyaw, dtype=float)
        ego_velocity = np.asarray(ego_velocity_global, dtype=float)
        ego_forward = np.array([np.cos(ego_pose[2]), np.sin(ego_pose[2])])
        ego_speed = max(float(ego_velocity[:2] @ ego_forward), 0.0)
        path = self._predict_ego_path(
            ego_pose_xyyaw, ego_velocity_global, float(ego_yaw_rate_radps)
        )
        eligible = [track for track in tracks if self._eligible(track)]
        candidates = [
            item
            for track in eligible
            if (item := self._collision_candidate(
                track, path, ego_pose_xyyaw, ego_speed
            )) is not None
        ]
        # 优先选最早进入ego足迹者；同一时刻取当前纵向间距更小者。
        candidates.sort(key=lambda item: (
            item["time_to_collision_s"], item["clearance_m"]
        ))
        target = candidates[0] if candidates else None
        raw_triggered = bool(
            ego_speed >= cfg.min_ego_speed_mps
            and target is not None
            and target["rss_margin_m"] <= 0.0
        )
        if raw_triggered:
            self._last_trigger_s = float(timestamp_s)
            self._last_raw_target = dict(target)
        held = bool(
            not raw_triggered
            and self._last_trigger_s is not None
            and float(timestamp_s) - self._last_trigger_s <= cfg.collision_keep_s
        )
        triggered = bool(raw_triggered or held)
        new_event = bool(triggered and not self._active)
        self._active = triggered
        selected = target if target is not None else self._last_raw_target if held else None
        if raw_triggered:
            reason = "rss_distance_violation"
        elif held:
            reason = "collision_keep_hold"
        elif target is None:
            reason = "no_predicted_path_collision"
        else:
            reason = "rss_margin_safe"
        decision = AutowareAEBShadowDecision(
            timestamp_s=float(timestamp_s),
            source=self.SOURCE,
            control_authority="none_shadow_only",
            triggered=triggered,
            raw_triggered=raw_triggered,
            held=held,
            new_trigger_event=new_event,
            reason=reason,
            target_track_id=None if selected is None else selected["track_id"],
            path_collision=target is not None,
            time_to_path_collision_s=(
                None if target is None else target["time_to_collision_s"]
            ),
            longitudinal_clearance_m=(
                None if selected is None else selected["clearance_m"]
            ),
            rss_distance_m=None if selected is None else selected["rss_distance_m"],
            rss_margin_m=None if selected is None else selected["rss_margin_m"],
            ego_speed_mps=ego_speed,
            object_longitudinal_speed_mps=(
                None if selected is None else selected["object_speed_mps"]
            ),
            candidate_count=len(eligible),
            path_collision_candidate_count=len(candidates),
        )
        self.metrics.observe(decision, reference_intervention_requested)
        return decision

    def summary(self) -> dict:
        return {
            "implementation": self.SOURCE,
            "control_authority": "none_shadow_only",
            "truth_input_used": False,
            "design_scope": [
                "ego_predicted_footprint_path",
                "track_path_collision_filter",
                "rss_braking_distance",
                "collision_trigger_hold",
            ],
            "config": asdict(self.config),
            **self.metrics.finalize(),
        }
