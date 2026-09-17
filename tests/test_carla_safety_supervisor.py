import numpy as np

from fusion_tracking_carla.safety import TrackSafety
from fusion_tracking_carla.safety_supervisor import (
    LongitudinalSupervisorConfig,
    RiskAwareLongitudinalSupervisor,
)
from fusion_tracking_carla.types import TrackSnapshot


def _track(*, lifecycle="confirmed", reliability_inputs=True):
    return TrackSnapshot(
        track_id=7,
        class_name="car",
        family="vehicle",
        lifecycle=lifecycle,
        timestamp_s=1.0,
        state_global=np.array([15.0, 0.0, 0.0, 0.0]),
        covariance_global=np.diag([0.2, 0.2, 0.5, 0.5]),
        extent_xy=np.array([4.5, 2.0]),
        score=0.95 if reliability_inputs else 0.1,
        age_updates=5,
        missed_updates=0,
        lidar_support=20 if reliability_inputs else 0,
        radar_support=4 if reliability_inputs else 0,
        camera_channels=("CAM_FRONT",),
        radial_velocity_std_mps=0.5,
    )


def _safety(*, risk=90.0, reliability=0.9, velocity=(0.0, 0.0), ttc=1.5):
    return TrackSafety(
        track_id=7,
        position_ego=np.array([15.0, 0.0]),
        velocity_ego=np.asarray(velocity, dtype=float),
        future_positions_ego=np.array([[10.0, 0.0], [5.0, 0.0]]),
        future_std_m=np.array([0.5, 0.8]),
        risk_score=risk,
        risk_level="critical",
        time_to_margin_s=1.0,
        approx_ttc_s=ttc,
        min_clearance_m=1.0,
        evidence_reliability=reliability,
    )


def _evaluate(supervisor, safety, ego_speed=10.0, timestamp=1.0):
    return supervisor.evaluate(
        timestamp_s=timestamp,
        safety=[safety],
        tracks=[_track()],
        ego_velocity_ego=np.array([ego_speed, 0.0]),
        fusion_age_s=0.2,
        dt_s=0.05,
    )


def test_stationary_non_closing_keeps_raw_warning_but_suppresses_action():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())
    safety = _safety(risk=80.0, velocity=(0.0, 0.0), ttc=None)

    decision = _evaluate(supervisor, safety, ego_speed=0.0)

    assert decision.raw_margin_risk == 80.0
    assert decision.actionable_collision_risk == 16.0
    assert decision.state == "NORMAL"
    assert decision.intervention_requested is False
    assert decision.control_suppressed is True
    assert "stationary_non_closing" in decision.suppression_reasons


def test_near_stationary_obstacle_transitions_from_braking_to_low_brake_hold():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(active_intervention=True)
    )
    closing = _safety(risk=95.0, velocity=(0.0, 0.0), ttc=1.2)
    closing_decisions = [
        _evaluate(supervisor, closing, ego_speed=8.0, timestamp=1.00 + 0.05 * index)
        for index in range(9)
    ]
    assert supervisor.state == "BRAKING"

    # 让推荐制动力真实爬升到紧急档；这样能覆盖“高制动后进入HOLD”，防止
    # 只在尚未爬升的0.125轻制动状态下测试而漏掉释放固定点。
    assert closing_decisions[-1].recommended_brake == supervisor.config.emergency_brake

    stopped = TrackSafety(**{
        **closing.__dict__,
        "position_ego": np.array([8.0, 0.0]),
        "approx_ttc_s": None,
        "min_clearance_m": -3.0,
        "risk_score": 100.0,
    })
    decisions = [
        _evaluate(supervisor, stopped, ego_speed=0.0, timestamp=1.45 + 0.05 * index)
        for index in range(12)
    ]
    decision = decisions[-1]

    assert decision.state == "HOLD"
    assert decision.actionable_collision_risk == 30.0
    assert decision.intervention_requested is True
    assert decision.control_suppressed is False
    assert decision.suppression_reasons == ()
    assert all(
        later.recommended_brake <= earlier.recommended_brake
        for earlier, later in zip(decisions, decisions[1:])
    )
    assert decision.recommended_brake == supervisor.config.stationary_hold_brake


def test_low_reliability_target_cannot_request_control():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())

    decision = _evaluate(supervisor, _safety(risk=99.0, reliability=0.2))

    assert decision.actionable_collision_risk == 0.0
    assert decision.trigger_track_id is None
    assert decision.intervention_requested is False
    assert "low_reliability" in decision.suppression_reasons


def test_empty_track_set_reports_why_control_is_suppressed():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())

    decision = supervisor.evaluate(
        timestamp_s=1.0,
        safety=[],
        tracks=[],
        ego_velocity_ego=np.array([5.0, 0.0]),
        fusion_age_s=0.1,
        dt_s=0.05,
    )

    assert decision.actionable_collision_risk == 0.0
    assert decision.control_suppressed is True
    assert decision.suppression_reasons == ("no_confirmed_target",)


def test_stopped_active_supervisor_bridges_bounded_perception_dropout():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(
            active_intervention=True,
            stationary_hold_dropout_frames=3,
            release_frames=2,
        )
    )
    closing = TrackSafety(**{
        **_safety(risk=95.0, ttc=1.0).__dict__,
        "position_ego": np.array([8.0, 0.0]),
    })
    _evaluate(supervisor, closing, ego_speed=8.0, timestamp=1.00)
    _evaluate(supervisor, closing, ego_speed=8.0, timestamp=1.05)
    assert supervisor.state == "BRAKING"

    decisions = []
    for index in range(3):
        decisions.append(supervisor.evaluate(
            timestamp_s=1.10 + 0.05 * index,
            safety=[],
            tracks=[],
            ego_velocity_ego=np.zeros(2),
            fusion_age_s=None,
            dt_s=0.05,
        ))
    assert all(item.state == "HOLD" for item in decisions)
    assert all(item.intervention_requested for item in decisions)
    assert all(not item.control_suppressed for item in decisions)
    assert all(item.trigger_track_id == 7 for item in decisions)
    assert all(
        item.suppression_reasons == ("stationary_hold_memory",)
        for item in decisions
    )

    # 有界记忆耗尽后按原release_frames逐步退出，不会无限锁车。
    exhausted = [
        supervisor.evaluate(
            timestamp_s=1.25 + 0.05 * index,
            safety=[], tracks=[], ego_velocity_ego=np.zeros(2),
            fusion_age_s=None, dt_s=0.05,
        )
        for index in range(8)
    ]
    assert exhausted[-1].state == "NORMAL"
    assert exhausted[-1].intervention_requested is False


def test_moving_hold_releases_brake_instead_of_applying_stationary_hold_force():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(
            active_intervention=True,
            activation_frames=1,
            release_frames=3,
            brake_ramp_per_s=2.0,
            brake_release_per_s=1.0,
        )
    )
    closing = _safety(risk=100.0, ttc=1.0)
    braking = _evaluate(supervisor, closing, ego_speed=8.0, timestamp=1.00)
    assert braking.state == "BRAKING"
    assert braking.recommended_brake > 0.0

    dropout = supervisor.evaluate(
        timestamp_s=1.05,
        safety=[],
        tracks=[],
        ego_velocity_ego=np.array([8.0, 0.0]),
        fusion_age_s=None,
        dt_s=0.05,
    )

    assert dropout.state == "HOLD"
    assert dropout.recommended_brake < braking.recommended_brake
    assert dropout.recommended_brake != supervisor.config.stationary_hold_brake


def test_product_hold_coasts_without_restoring_route_throttle():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(
            active_intervention=True,
            activation_frames=1,
            release_frames=3,
            coast_during_hold_and_release=True,
        )
    )
    _evaluate(supervisor, _safety(risk=100.0, ttc=1.0), ego_speed=8.0)
    supervisor.evaluate(
        timestamp_s=1.05,
        safety=[],
        tracks=[],
        ego_velocity_ego=np.array([8.0, 0.0]),
        fusion_age_s=None,
        dt_s=0.05,
    )
    base = {
        "throttle": 0.65, "brake": 0.0, "steer": 0.1,
        "reverse": False, "hand_brake": False,
    }

    applied, record = supervisor.arbitrate(base, "route_agent")

    assert supervisor.state == "HOLD"
    assert record.applied is True
    assert applied["throttle"] == 0.0
    assert applied["steer"] == base["steer"]


def test_high_risk_evidence_bridges_only_bounded_transient_dropout():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(
            active_intervention=True,
            activation_frames=2,
            risk_dropout_bridge_frames=2,
        )
    )
    closing = _safety(risk=100.0, ttc=1.0)
    first = _evaluate(supervisor, closing, ego_speed=8.0, timestamp=1.00)
    assert first.state == "NORMAL"

    bridged = supervisor.evaluate(
        timestamp_s=1.05, safety=[], tracks=[],
        ego_velocity_ego=np.array([8.0, 0.0]), fusion_age_s=None, dt_s=0.05,
    )
    assert bridged.state == "BRAKING"
    assert bridged.trigger_track_id == 7
    assert bridged.suppression_reasons == ("risk_dropout_memory",)
    assert bridged.control_suppressed is False

    # 给出明确低风险新证据时必须立即清除历史高风险，不得继续触发。
    safe = _safety(risk=20.0, velocity=(8.0, 0.0), ttc=8.0)
    cleared = _evaluate(supervisor, safe, ego_speed=8.0, timestamp=1.10)
    assert cleared.actionable_collision_risk < supervisor.config.brake_score
    assert cleared.suppression_reasons == ()


def test_deceleration_governor_caps_command_and_releases_on_overshoot():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(
            active_intervention=True,
            activation_frames=1,
            max_service_brake=0.25,
            emergency_brake=0.30,
            brake_ramp_per_s=10.0,
            stationary_hold_brake=0.05,
            deceleration_governor_enabled=True,
        )
    )
    closing = _safety(risk=100.0, ttc=1.0)
    first = _evaluate(supervisor, closing, ego_speed=8.0, timestamp=1.00)
    assert first.recommended_brake <= 0.30

    # 0.5m/s / 0.05s = 10m/s^2减速，明显超过5m/s^2紧急目标；
    # 调节器应主动减小制动命令，而非继续向上爬升。
    overshoot = _evaluate(supervisor, closing, ego_speed=7.5, timestamp=1.05)
    assert overshoot.recommended_brake < first.recommended_brake
    assert overshoot.recommended_brake <= 0.30


def test_deceleration_governor_tapers_emergency_brake_at_low_speed():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(
            active_intervention=True,
            activation_frames=1,
            max_service_brake=0.25,
            emergency_brake=0.30,
            brake_ramp_per_s=10.0,
            stationary_hold_brake=0.05,
            low_speed_brake_taper_start_mps=8.0,
            deceleration_governor_enabled=True,
        )
    )
    decision = _evaluate(
        supervisor, _safety(risk=100.0, ttc=1.0),
        ego_speed=4.0, timestamp=1.00,
    )
    assert decision.recommended_brake <= 0.175 + 1e-9


def test_suppression_reasons_belong_only_to_selected_trigger_candidate():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())
    valid = _safety(risk=80.0, reliability=0.9, velocity=(0.0, 0.0), ttc=None)
    weak = TrackSafety(
        **{
            **valid.__dict__,
            "track_id": 8,
            "risk_score": 99.0,
            "evidence_reliability": 0.1,
        }
    )
    weak_track = TrackSnapshot(
        **{**_track().__dict__, "track_id": 8}
    )

    decision = supervisor.evaluate(
        timestamp_s=1.0,
        safety=[weak, valid],
        tracks=[weak_track, _track()],
        ego_velocity_ego=np.zeros(2),
        fusion_age_s=0.1,
        dt_s=0.05,
    )

    assert decision.trigger_track_id == 7
    assert decision.suppression_reasons == ("stationary_non_closing",)
    assert "low_reliability" not in decision.suppression_reasons


def test_adjacent_vehicle_inside_broad_gate_is_rejected_by_path_corridor():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())
    adjacent = TrackSafety(**{
        **_safety(risk=95.0).__dict__,
        "position_ego": np.array([18.0, 3.1]),
    })

    decision = _evaluate(supervisor, adjacent, ego_speed=8.0)

    assert decision.actionable_collision_risk == 0.0
    assert decision.intervention_requested is False
    assert "outside_ego_path_overlap" in decision.suppression_reasons


def test_vehicle_evidence_centroid_inside_path_corridor_remains_actionable():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())
    partial_overlap = TrackSafety(**{
        **_safety(risk=95.0).__dict__,
        "position_ego": np.array([18.0, 1.9]),
    })

    decision = _evaluate(supervisor, partial_overlap, ego_speed=8.0)

    assert decision.actionable_collision_risk > 0.0
    assert "outside_ego_path_overlap" not in decision.suppression_reasons


def test_closing_target_requires_two_frames_then_requests_brake():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())
    safety = _safety(risk=90.0)

    first = _evaluate(supervisor, safety, timestamp=1.00)
    second = _evaluate(supervisor, safety, timestamp=1.05)

    assert first.intervention_requested is False
    assert second.state == "BRAKING"
    assert second.intervention_requested is True
    assert second.new_intervention_event is True
    assert second.recommended_throttle_cap == 0.0
    assert second.recommended_brake > 0.0
    assert second.trigger_track_id == 7


def test_warning_throttle_limit_is_counted_as_an_intervention_event():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())

    decision = _evaluate(supervisor, _safety(risk=50.0, ttc=3.0))

    assert decision.state == "WARNING"
    assert 0.0 < decision.recommended_throttle_cap < 1.0
    assert decision.recommended_brake == 0.0
    assert decision.intervention_requested is True
    assert decision.new_intervention_event is True


def test_shadow_records_recommendation_without_modifying_control():
    supervisor = RiskAwareLongitudinalSupervisor(LongitudinalSupervisorConfig())
    _evaluate(supervisor, _safety(), timestamp=1.00)
    _evaluate(supervisor, _safety(), timestamp=1.05)
    base = {"throttle": 0.7, "brake": 0.0, "steer": 0.2,
            "reverse": False, "hand_brake": False}

    applied, record = supervisor.arbitrate(base, "route_agent")

    assert applied == base
    assert record.applied is False
    assert record.reason == "shadow_mode"


def test_active_supervisor_changes_only_longitudinal_control():
    config = LongitudinalSupervisorConfig(active_intervention=True)
    supervisor = RiskAwareLongitudinalSupervisor(config)
    _evaluate(supervisor, _safety(), timestamp=1.00)
    _evaluate(supervisor, _safety(), timestamp=1.05)
    base = {"throttle": 0.7, "brake": 0.0, "steer": -0.25,
            "reverse": False, "hand_brake": False}

    applied, record = supervisor.arbitrate(base, "route_agent")

    assert record.applied is True
    assert record.reason == "risk_supervisor"
    assert applied["throttle"] == 0.0
    assert applied["brake"] > 0.0
    assert applied["steer"] == base["steer"]
    assert applied["reverse"] == base["reverse"]


def test_active_supervisor_never_competes_with_traffic_manager_autopilot():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(active_intervention=True)
    )
    _evaluate(supervisor, _safety(), timestamp=1.00)
    _evaluate(supervisor, _safety(), timestamp=1.05)
    base = {"throttle": 0.7, "brake": 0.0, "steer": 0.1,
            "reverse": False, "hand_brake": False}

    applied, record = supervisor.arbitrate(base, "autopilot")

    assert applied == base
    assert record.applied is False
    assert record.reason == "traffic_manager_control_not_owned"


def test_fresh_stream_cannot_hide_an_individually_stale_target_measurement():
    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(max_track_measurement_age_s=1.0)
    )
    stale_track = TrackSnapshot(
        **{
            **_track().__dict__,
            "timestamp_s": 2.5,
            "measurement_timestamp_s": 1.0,
        }
    )

    decision = supervisor.evaluate(
        timestamp_s=2.5,
        safety=[_safety(risk=95.0)],
        tracks=[stale_track],
        ego_velocity_ego=np.array([8.0, 0.0]),
        # Another target could keep this global stream age fresh.
        fusion_age_s=0.1,
        dt_s=0.05,
    )

    assert decision.actionable_collision_risk == 0.0
    assert decision.intervention_requested is False
    assert "stale_track_measurement" in decision.suppression_reasons
