"""风险感知纵向安全监督器。

该模块不做路线规划和转向；它只把已有轨迹安全裕度转换为
“是否值得采取纵向动作”，并在显式开启时限制油门/制动。
默认 Shadow：计算建议、记录事件，但不修改车辆控制。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .safety import TrackSafety
from .types import TrackSnapshot


@dataclass(frozen=True)
class LongitudinalSupervisorConfig:
    """常调参数。首版数值是工程先验，不声称已经量产标定。"""

    active_intervention: bool = False  # 默认只记录建议；CLI显式开启才动车。
    warning_score: float = 35.0
    brake_score: float = 60.0
    emergency_score: float = 85.0
    release_score: float = 25.0
    min_reliability: float = 0.55
    max_fusion_age_s: float = 0.90
    # Global fusion freshness is insufficient when a different target keeps
    # the detector stream alive.  Gate each candidate by its own last spatial
    # measurement epoch as well.
    max_track_measurement_age_s: float = 1.10
    front_min_x_m: float = 0.5
    corridor_half_width_m: float = 3.5
    # 宽corridor先排除明显侧后方目标；真正控制准入使用更窄的ego中心线路径包络。
    # 当前位置是可见LiDAR点簇质心而非完整3D框中心，不能再扣“估计车身半宽”。
    path_half_width_m: float = 2.0
    stationary_speed_mps: float = 0.5
    min_closing_speed_mps: float = 0.25
    closing_full_scale_mps: float = 3.0
    stationary_risk_scale: float = 0.20
    # 已由本监督器刹停且近距目标仍在前方时，保持低制动力，避免“紧急制动不释放”
    # 或“完全松闸后再次起步”的振荡。hold_score必须位于release与warning之间。
    stationary_hold_distance_m: float = 12.0
    stationary_hold_score: float = 30.0
    stationary_hold_brake: float = 0.15
    # 已由本模块刹停后，允许用最近一次传感器证据跨越短时感知空档。20Hz下40帧=2s；
    # 只接受missing/低可靠/过期/tentative，不接受目标明确移出路径的情况。
    stationary_hold_dropout_frames: int = 0
    # 行驶中已确认的高风险证据可跨越少量低可靠/过期/缺帧。
    # 0表示关闭；20Hz下6帧=0.3s。明确移出ego路径时不会启用记忆。
    risk_dropout_bridge_frames: int = 0
    # BRAKING后的HOLD/RELEASE允许松开制动，但产品配置可禁止立刻恢复巡航油门。
    # 默认关闭以保持旧阶段可复现；Stage6.18及以后显式开启。
    coast_during_hold_and_release: bool = False
    activation_frames: int = 2  # 20Hz下连续0.1s，抑制单帧抖动。
    release_frames: int = 6  # 20Hz下连续0.3s低风险才释放。
    max_service_brake: float = 0.70
    emergency_brake: float = 0.90
    brake_ramp_per_s: float = 2.5
    brake_release_per_s: float = 1.5
    # 可选的闭环减速度调节：旧阶段默认关闭，Stage6.18显式启用。
    # 前馈和反馈增益的单位均为 brake_command/(m/s^2)。
    deceleration_governor_enabled: bool = False
    service_deceleration_mps2: float = 3.0
    emergency_deceleration_mps2: float = 5.0
    brake_feedforward_gain: float = 0.05
    brake_deceleration_feedback_gain: float = 0.02
    deceleration_filter_tau_s: float = 0.10
    deceleration_overshoot_margin_mps2: float = 1.0
    deceleration_overshoot_release_per_s: float = 4.0
    low_speed_brake_taper_start_mps: float = 8.0

    def validate(self) -> None:
        scores = (
            self.release_score,
            self.warning_score,
            self.brake_score,
            self.emergency_score,
        )
        if not (0 <= scores[0] < scores[1] < scores[2] < scores[3] <= 100):
            raise ValueError(
                "风险阈值需满足 release < warning < brake < emergency，且在0~100"
            )
        if not 0 <= self.min_reliability <= 1:
            raise ValueError("min_reliability必须在0~1")
        if (
            self.max_fusion_age_s <= 0
            or self.max_track_measurement_age_s <= 0
            or self.corridor_half_width_m <= 0
        ):
            raise ValueError("融合年龄和走廊宽度必须大于0")
        if self.path_half_width_m <= 0:
            raise ValueError("路径半宽必须大于0")
        if self.activation_frames < 1 or self.release_frames < 1:
            raise ValueError("连续帧数必须>=1")
        if self.stationary_hold_distance_m <= 0:
            raise ValueError("静止保持距离必须>0")
        if self.stationary_hold_dropout_frames < 0:
            raise ValueError("静止保持空档帧数不能为负")
        if self.risk_dropout_bridge_frames < 0:
            raise ValueError("行驶风险证据跨帧数不能为负")
        if not self.release_score < self.stationary_hold_score < self.warning_score:
            raise ValueError("静止保持风险需满足release < hold < warning")
        if not 0 < self.max_service_brake <= self.emergency_brake <= 1:
            raise ValueError("制动上限必须满足0 < service <= emergency <= 1")
        if not 0 < self.stationary_hold_brake <= self.max_service_brake:
            raise ValueError("静止保持制动力需满足0 < hold_brake <= service_brake")
        if self.brake_ramp_per_s <= 0 or self.brake_release_per_s <= 0:
            raise ValueError("制动爬升/释放率必须>0")
        governor_positive = (
            self.service_deceleration_mps2,
            self.emergency_deceleration_mps2,
            self.brake_feedforward_gain,
            self.brake_deceleration_feedback_gain,
            self.deceleration_overshoot_margin_mps2,
            self.deceleration_overshoot_release_per_s,
            self.low_speed_brake_taper_start_mps,
        )
        if any(value <= 0 for value in governor_positive):
            raise ValueError("减速度调节器参数必须>0")
        if self.deceleration_filter_tau_s < 0:
            raise ValueError("减速度滤波时常不能为负")
        if self.service_deceleration_mps2 > self.emergency_deceleration_mps2:
            raise ValueError("服务减速度不能大于紧急减速度")


@dataclass(frozen=True)
class SupervisorDecision:
    timestamp_s: float
    mode: str
    state: str
    raw_margin_risk: float
    actionable_collision_risk: float
    trigger_track_id: int | None
    trigger_reliability: float | None
    closing_speed_mps: float | None
    approx_ttc_s: float | None
    predicted_clearance_m: float | None
    recommended_throttle_cap: float
    recommended_brake: float
    intervention_requested: bool
    control_suppressed: bool
    suppression_reasons: tuple[str, ...]
    new_intervention_event: bool = False

    def to_dict(self) -> dict:
        result = asdict(self)
        result["suppression_reasons"] = list(self.suppression_reasons)
        return result


@dataclass(frozen=True)
class AppliedSupervisorControl:
    source_timestamp_s: float | None
    applied: bool
    base_control: dict
    applied_control: dict
    reason: str
    # External arbitration gates (for example sensor-health fail-silent) are
    # kept separate from the risk model's own suppression reasons.
    inhibit_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["inhibit_reasons"] = list(self.inhibit_reasons)
        return result


@dataclass
class SupervisorMetrics:
    frames: int = 0
    requested_frames: int = 0
    applied_frames: int = 0
    applied_intervention_events: int = 0
    intervention_events: int = 0
    max_raw_margin_risk: float = 0.0
    max_actionable_collision_risk: float = 0.0
    actionable_sum: float = 0.0
    state_frames: dict[str, int] = field(default_factory=dict)
    suppression_frames: dict[str, int] = field(default_factory=dict)
    _last_applied: bool = field(default=False, repr=False)

    def observe(self, decision: SupervisorDecision) -> None:
        self.frames += 1
        self.requested_frames += int(decision.intervention_requested)
        self.intervention_events += int(decision.new_intervention_event)
        self.max_raw_margin_risk = max(
            self.max_raw_margin_risk, decision.raw_margin_risk
        )
        self.max_actionable_collision_risk = max(
            self.max_actionable_collision_risk,
            decision.actionable_collision_risk,
        )
        self.actionable_sum += decision.actionable_collision_risk
        self.state_frames[decision.state] = self.state_frames.get(decision.state, 0) + 1
        for reason in decision.suppression_reasons:
            self.suppression_frames[reason] = self.suppression_frames.get(reason, 0) + 1

    def observe_applied(self, applied: bool) -> None:
        """Count actual control modifications separately from recommendations."""

        applied = bool(applied)
        self.applied_frames += int(applied)
        self.applied_intervention_events += int(applied and not self._last_applied)
        self._last_applied = applied

    def finalize(self) -> dict:
        return {
            "frames": self.frames,
            "requested_frames": self.requested_frames,
            "applied_frames": self.applied_frames,
            "applied_intervention_events": self.applied_intervention_events,
            "intervention_events": self.intervention_events,
            "max_raw_margin_risk": self.max_raw_margin_risk,
            "max_actionable_collision_risk": self.max_actionable_collision_risk,
            "mean_actionable_collision_risk": (
                self.actionable_sum / self.frames if self.frames else 0.0
            ),
            "state_frames": dict(self.state_frames),
            "suppression_frames": dict(self.suppression_frames),
        }


class RiskAwareLongitudinalSupervisor:
    """只读取项目轨迹/风险，输出可审计的纵向控制建议。"""

    def __init__(self, config: LongitudinalSupervisorConfig):
        config.validate()
        self.config = config
        self.state = "NORMAL"
        self._activation_count = 0
        self._release_count = 0
        self._recommended_brake = 0.0
        self._pending: SupervisorDecision | None = None
        self._last_intervention_requested = False
        self._hold_dropout_count = 0
        self._last_hold_track_id: int | None = None
        self._risk_dropout_count = 0
        self._last_actionable_risk = 0.0
        self._last_actionable_safety: TrackSafety | None = None
        self._last_actionable_closing_speed_mps: float | None = None
        self._previous_ego_speed_mps: float | None = None
        self._filtered_longitudinal_acceleration_mps2: float | None = None
        self.metrics = SupervisorMetrics()

    @property
    def mode(self) -> str:
        return "active" if self.config.active_intervention else "shadow"

    def _candidate(
        self,
        safety: TrackSafety,
        track: TrackSnapshot,
        ego_velocity_ego: np.ndarray,
        fusion_age_s: float | None,
    ) -> tuple[float, float, tuple[str, ...]]:
        cfg = self.config
        reasons: list[str] = []
        position = np.asarray(safety.position_ego, dtype=float)
        target_velocity = np.asarray(safety.velocity_ego, dtype=float)
        relative_velocity = target_velocity - ego_velocity_ego
        distance = max(float(np.linalg.norm(position)), 1e-6)
        closing_speed = max(-float(position @ relative_velocity) / distance, 0.0)
        if track.lifecycle != "confirmed":
            reasons.append("track_not_confirmed")
        if safety.evidence_reliability < cfg.min_reliability:
            reasons.append("low_reliability")
        if fusion_age_s is None or fusion_age_s > cfg.max_fusion_age_s:
            reasons.append("stale_or_missing_perception")
        measurement_timestamp = track.measurement_timestamp_s
        measurement_age_s = (
            0.0 if measurement_timestamp is None
            else max(float(track.timestamp_s - measurement_timestamp), 0.0)
        )
        if measurement_age_s > cfg.max_track_measurement_age_s:
            reasons.append("stale_track_measurement")
        if position[0] < cfg.front_min_x_m:
            reasons.append("outside_front_sector")
        if abs(position[1]) > cfg.corridor_half_width_m:
            reasons.append("outside_longitudinal_corridor")
        if abs(float(position[1])) > cfg.path_half_width_m:
            reasons.append("outside_ego_path_overlap")
        if reasons:
            return 0.0, closing_speed, tuple(reasons)

        raw = float(np.clip(safety.risk_score, 0.0, 100.0))
        ego_speed = float(np.linalg.norm(ego_velocity_ego))
        if (
            ego_speed < cfg.stationary_speed_mps
            and closing_speed < cfg.min_closing_speed_mps
            and safety.approx_ttc_s is None
        ):
            reduced_risk = raw * cfg.stationary_risk_scale
            if (
                self.state in {"BRAKING", "HOLD", "RELEASE"}
                and position[0] <= cfg.stationary_hold_distance_m
            ):
                # 已安全刹停：落入HOLD并逐渐降到约0.15制动力。这里不标记为
                # control_suppressed，因为保持制动是有意执行的安全动作。
                return max(
                    reduced_risk, cfg.stationary_hold_score
                ), closing_speed, ()
            # 尚未由本监督器介入，或障碍已经足够远：保留裕度提示但不主动制动。
            return reduced_risk, closing_speed, ("stationary_non_closing",)

        closing_factor = float(np.clip(
            closing_speed / cfg.closing_full_scale_mps, 0.0, 1.0
        ))
        ttc_factor = 0.0
        if safety.approx_ttc_s is not None:
            ttc_factor = float(np.clip((5.0 - safety.approx_ttc_s) / 4.0, 0.0, 1.0))
        motion_factor = max(closing_factor, ttc_factor)
        actionable = raw * (0.25 + 0.75 * motion_factor)
        if safety.min_clearance_m <= 0.0:
            actionable = max(actionable, min(100.0, 80.0 - 10.0 * safety.min_clearance_m))
        return float(np.clip(actionable, 0.0, 100.0)), closing_speed, ()

    def _update_state(
        self,
        risk: float,
        dt_s: float,
        ego_speed_mps: float,
        measured_deceleration_mps2: float,
    ) -> tuple[bool, bool]:
        cfg = self.config
        was_requested = self.state in {"BRAKING", "HOLD", "RELEASE"}
        if risk >= cfg.brake_score:
            self._activation_count += 1
            self._release_count = 0
            if self._activation_count >= cfg.activation_frames:
                self.state = "BRAKING"
        elif risk >= cfg.warning_score:
            self._activation_count = 0
            self._release_count = 0
            if self.state not in {"BRAKING", "HOLD"}:
                self.state = "WARNING"
        else:
            self._activation_count = 0
            if self.state in {"BRAKING", "HOLD"}:
                if risk <= cfg.release_score:
                    self._release_count += 1
                    self.state = "HOLD"
                    if self._release_count >= cfg.release_frames:
                        self.state = "RELEASE"
                else:
                    self._release_count = 0
                    self.state = "HOLD"
            elif self.state == "RELEASE":
                if self._recommended_brake <= 1e-6:
                    self.state = "NORMAL"
            else:
                self.state = "NORMAL"

        target_brake = 0.0
        desired_deceleration = 0.0
        if self.state == "BRAKING":
            if risk >= cfg.emergency_score:
                target_brake = cfg.emergency_brake
                desired_deceleration = cfg.emergency_deceleration_mps2
            else:
                ratio = np.clip(
                    (risk - cfg.brake_score)
                    / (cfg.emergency_score - cfg.brake_score),
                    0.0,
                    1.0,
                )
                target_brake = 0.20 + ratio * (cfg.max_service_brake - 0.20)
                desired_deceleration = cfg.service_deceleration_mps2
        elif self.state == "HOLD":
            # HOLD包含两种情况：已刹停后的驻车保持，以及行驶中等待
            # release_frames的风险回滞。只有前者才应施加固定保持力；
            # 高速感知空窗若也强制到hold_brake，会造成不必要的减速脉冲。
            target_brake = (
                cfg.stationary_hold_brake
                if ego_speed_mps < cfg.stationary_speed_mps
                else 0.0
            )

        if cfg.deceleration_governor_enabled and self.state == "BRAKING":
            # 风险决定期望减速度，而非直接把风险分数当作刹车踏板。
            target_brake = (
                cfg.brake_feedforward_gain * desired_deceleration
                + cfg.brake_deceleration_feedback_gain
                * (desired_deceleration - measured_deceleration_mps2)
            )
            upper = (
                cfg.emergency_brake
                if risk >= cfg.emergency_score
                else cfg.max_service_brake
            )
            target_brake = float(np.clip(target_brake, 0.0, upper))
            # 低速制动扭矩对车速变化更敏感；线性退坡避免最后数米“点头”。
            if ego_speed_mps < cfg.low_speed_brake_taper_start_mps:
                fraction = float(np.clip(
                    ego_speed_mps / cfg.low_speed_brake_taper_start_mps,
                    0.0,
                    1.0,
                ))
                low_speed_cap = (
                    cfg.stationary_hold_brake
                    + fraction * (upper - cfg.stationary_hold_brake)
                )
                target_brake = min(target_brake, low_speed_cap)

        release_rate = cfg.brake_release_per_s
        if (
            cfg.deceleration_governor_enabled
            and desired_deceleration > 0.0
            and measured_deceleration_mps2
            > desired_deceleration + cfg.deceleration_overshoot_margin_mps2
        ):
            release_rate = max(
                release_rate, cfg.deceleration_overshoot_release_per_s
            )
        if target_brake > self._recommended_brake:
            self._recommended_brake = min(
                target_brake, self._recommended_brake + cfg.brake_ramp_per_s * dt_s
            )
        else:
            self._recommended_brake = max(
                target_brake, self._recommended_brake - release_rate * dt_s
            )
        if self.state == "RELEASE" and self._recommended_brake <= 1e-6:
            self.state = "NORMAL"
        requested = self.state in {"BRAKING", "HOLD", "RELEASE"} and (
            self._recommended_brake > 0.0
        )
        return bool(requested), bool(requested and not was_requested)

    def evaluate(
        self,
        *,
        timestamp_s: float,
        safety: list[TrackSafety],
        tracks: list[TrackSnapshot],
        ego_velocity_ego: np.ndarray,
        fusion_age_s: float | None,
        dt_s: float,
    ) -> SupervisorDecision:
        track_by_id = {track.track_id: track for track in tracks}
        candidates = []
        for item in safety:
            track = track_by_id.get(item.track_id)
            if track is None:
                continue
            actionable, closing_speed, reasons = self._candidate(
                item, track, ego_velocity_ego, fusion_age_s
            )
            candidates.append((actionable, item, closing_speed, reasons))
        candidates.sort(key=lambda value: value[0], reverse=True)
        best = candidates[0] if candidates else None
        # 抑制原因只描述最终控制候选，不能把同一帧其他弱轨迹的原因混入。
        # 否则会出现“触发ID可靠度0.95，但原因又写low_reliability”的歧义。
        suppressed = (
            ["no_confirmed_target"] if best is None else list(best[3])
        )
        actionable = 0.0 if best is None else best[0]
        trigger_track_id = (
            None if best is None or actionable <= 0 else best[1].track_id
        )
        selected_safety = None if best is None else best[1]
        selected_closing_speed = None if best is None else float(best[2])
        ego_speed = float(np.linalg.norm(ego_velocity_ego))
        dropout_only_reasons = {
            "no_confirmed_target",
            "low_reliability",
            "stale_or_missing_perception",
            "stale_track_measurement",
            "track_not_confirmed",
        }
        if actionable >= self.config.brake_score and best is not None:
            self._risk_dropout_count = 0
            self._last_actionable_risk = float(actionable)
            self._last_actionable_safety = best[1]
            self._last_actionable_closing_speed_mps = float(best[2])
        elif actionable > 0:
            # 新鲜的低风险证据比历史高风险更有效，不得继续沿用旧触发。
            self._risk_dropout_count = 0
            self._last_actionable_risk = 0.0
            self._last_actionable_safety = None
            self._last_actionable_closing_speed_mps = None
        elif (
            self.config.risk_dropout_bridge_frames > 0
            and ego_speed >= self.config.stationary_speed_mps
            and self._last_actionable_safety is not None
            and self._last_actionable_risk >= self.config.brake_score
            and set(suppressed).issubset(dropout_only_reasons)
            and self._risk_dropout_count < self.config.risk_dropout_bridge_frames
        ):
            self._risk_dropout_count += 1
            actionable = self._last_actionable_risk
            selected_safety = self._last_actionable_safety
            selected_closing_speed = self._last_actionable_closing_speed_mps
            trigger_track_id = selected_safety.track_id
            suppressed = ["risk_dropout_memory"]
        elif not set(suppressed).issubset(dropout_only_reasons):
            # 目标明确移出前向/路径门时立即清除，不允许历史风险穿透几何门。
            self._risk_dropout_count = 0
            self._last_actionable_risk = 0.0
            self._last_actionable_safety = None
            self._last_actionable_closing_speed_mps = None

        if actionable > 0:
            self._hold_dropout_count = 0
            if selected_safety is not None:
                position = np.asarray(selected_safety.position_ego, dtype=float)
                if (
                    position[0] <= self.config.stationary_hold_distance_m
                    and abs(position[1]) <= self.config.path_half_width_m
                ):
                    self._last_hold_track_id = selected_safety.track_id
        elif (
            self.config.stationary_hold_dropout_frames > 0
            and self.state in {"BRAKING", "HOLD", "RELEASE"}
            and ego_speed < self.config.stationary_speed_mps
            and self._last_hold_track_id is not None
            and set(suppressed).issubset(dropout_only_reasons)
            and self._hold_dropout_count
            < self.config.stationary_hold_dropout_frames
        ):
            self._hold_dropout_count += 1
            actionable = self.config.stationary_hold_score
            trigger_track_id = self._last_hold_track_id
            suppressed = ["stationary_hold_memory"]
        elif self.state == "NORMAL":
            self._hold_dropout_count = 0
            self._last_hold_track_id = None
        measured_deceleration = 0.0
        if dt_s > 0 and self._previous_ego_speed_mps is not None:
            raw_acceleration = (
                ego_speed - self._previous_ego_speed_mps
            ) / dt_s
            tau_s = self.config.deceleration_filter_tau_s
            alpha = 1.0 if tau_s <= 0.0 else dt_s / (tau_s + dt_s)
            previous = self._filtered_longitudinal_acceleration_mps2
            filtered = raw_acceleration if previous is None else (
                previous + alpha * (raw_acceleration - previous)
            )
            self._filtered_longitudinal_acceleration_mps2 = float(filtered)
            measured_deceleration = max(-float(filtered), 0.0)
        self._previous_ego_speed_mps = ego_speed
        brake_requested, _ = self._update_state(
            actionable, dt_s, ego_speed, measured_deceleration
        )
        raw_peak = max((item.risk_score for item in safety), default=0.0)
        throttle_cap = 1.0
        if (
            self.config.coast_during_hold_and_release
            and self.state in {"HOLD", "RELEASE"}
        ):
            # 状态机完整退出前只滑行，避免“松刹—给油—再次制动”的纵向冲击。
            throttle_cap = 0.0
        # 高风险仍需先满足activation_frames才进入BRAKING；普通WARNING只做温和限油。
        # NORMAL不直接按单帧尖峰切断油门。
        if self.state != "NORMAL" and actionable >= self.config.warning_score:
            throttle_cap = min(throttle_cap, float(np.clip(
                1.0 - (actionable - self.config.warning_score)
                / (self.config.brake_score - self.config.warning_score),
                0.0,
                1.0,
            )))
        intervention_requested = bool(brake_requested or throttle_cap < 1.0)
        new_event = bool(
            intervention_requested and not self._last_intervention_requested
        )
        self._last_intervention_requested = bool(intervention_requested)
        decision = SupervisorDecision(
            timestamp_s=float(timestamp_s),
            mode=self.mode,
            state=self.state,
            raw_margin_risk=float(raw_peak),
            actionable_collision_risk=float(actionable),
            trigger_track_id=trigger_track_id,
            trigger_reliability=(
                None if selected_safety is None or actionable <= 0
                else float(selected_safety.evidence_reliability)
            ),
            closing_speed_mps=selected_closing_speed,
            approx_ttc_s=(
                None if selected_safety is None else selected_safety.approx_ttc_s
            ),
            predicted_clearance_m=(
                None if selected_safety is None
                else float(selected_safety.min_clearance_m)
            ),
            recommended_throttle_cap=throttle_cap,
            recommended_brake=float(self._recommended_brake),
            intervention_requested=intervention_requested,
            control_suppressed=bool(suppressed and not intervention_requested),
            suppression_reasons=tuple(sorted(set(suppressed))),
            new_intervention_event=new_event,
        )
        self._pending = decision
        self.metrics.observe(decision)
        return decision

    def arbitrate(
        self,
        base_control: dict,
        drive_mode: str,
        inhibit_reasons: tuple[str, ...] = (),
    ) -> tuple[dict, AppliedSupervisorControl]:
        base = dict(base_control)
        applied = dict(base_control)
        pending = self._pending
        if pending is None:
            reason = "no_previous_decision"
            allowed = False
        elif not self.config.active_intervention:
            reason = "shadow_mode"
            allowed = False
        elif drive_mode == "autopilot":
            reason = "traffic_manager_control_not_owned"
            allowed = False
        elif inhibit_reasons:
            # Keep the recommendation in ``pending`` for diagnosis, but never
            # actuate it while critical sensor evidence is failed/recovering.
            reason = "sensor_health_inhibit"
            allowed = False
        elif not pending.intervention_requested and pending.recommended_throttle_cap >= 1.0:
            reason = "no_intervention_requested"
            allowed = False
        else:
            reason = "risk_supervisor"
            allowed = True
            applied["throttle"] = min(
                float(applied["throttle"]), pending.recommended_throttle_cap
            )
            applied["brake"] = max(
                float(applied["brake"]), pending.recommended_brake
            )
            if applied["brake"] > 0.01:
                applied["throttle"] = 0.0
        changed = bool(
            allowed
            and (
                abs(float(applied["throttle"]) - float(base["throttle"])) > 1e-6
                or abs(float(applied["brake"]) - float(base["brake"])) > 1e-6
            )
        )
        self.metrics.observe_applied(changed)
        return applied, AppliedSupervisorControl(
            source_timestamp_s=None if pending is None else pending.timestamp_s,
            applied=changed,
            base_control=base,
            applied_control=dict(applied),
            reason=reason,
            inhibit_reasons=tuple(inhibit_reasons),
        )

    def summary(self) -> dict:
        return {
            "config": asdict(self.config),
            "mode": self.mode,
            **self.metrics.finalize(),
        }
