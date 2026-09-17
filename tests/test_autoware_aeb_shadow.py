import numpy as np

from fusion_tracking_carla.autoware_aeb_shadow import (
    AutowareAEBShadowConfig,
    AutowareAEBShadowEvaluator,
)
from fusion_tracking_carla.session import CarlaFusionSession
from fusion_tracking_carla.types import TrackSnapshot


def test_aeb_advisory_can_only_cap_throttle():
    decision = type("Decision", (), {
        "triggered": True,
        "timestamp_s": 12.5,
        "target_track_id": 42,
    })()
    original = {
        "throttle": 0.7,
        "brake": 0.12,
        "steer": -0.25,
        "hand_brake": False,
        "reverse": False,
        "manual_gear_shift": False,
    }
    output, audit = CarlaFusionSession._apply_aeb_throttle_advisory(
        original, decision, 0.0, active=True
    )
    assert output["throttle"] == 0.0
    assert output["brake"] == original["brake"]
    assert output["steer"] == original["steer"]
    assert original["throttle"] == 0.7
    assert audit["applied"] is True
    assert audit["target_track_id"] == 42


def _track(
    *,
    track_id=7,
    position=(14.5, 0.0),
    velocity=(0.0, 0.0),
    lifecycle="confirmed",
    missed=0,
):
    return TrackSnapshot(
        track_id=track_id,
        class_name="car",
        family="vehicle",
        lifecycle=lifecycle,
        timestamp_s=1.0,
        state_global=np.array([*position, *velocity], dtype=float),
        covariance_global=np.diag([0.2, 0.2, 0.5, 0.5]),
        extent_xy=np.array([4.4, 2.0]),
        score=0.95,
        age_updates=5,
        missed_updates=missed,
        lidar_support=20,
        radar_support=4,
        camera_channels=("CAM_FRONT",),
        radial_velocity_std_mps=0.5,
    )


def _evaluate(evaluator, tracks, *, timestamp=1.0, speed=10.0, reference=False):
    return evaluator.evaluate(
        timestamp_s=timestamp,
        tracks=tracks,
        ego_pose_xyyaw=np.array([0.0, 0.0, 0.0]),
        ego_velocity_global=np.array([speed, 0.0]),
        ego_yaw_rate_radps=0.0,
        reference_intervention_requested=reference,
    )


def test_stationary_lead_inside_predicted_footprint_triggers_rss_shadow():
    evaluator = AutowareAEBShadowEvaluator()

    decision = _evaluate(evaluator, [_track()])

    assert decision.path_collision is True
    assert decision.raw_triggered is True
    assert decision.triggered is True
    assert decision.new_trigger_event is True
    assert decision.target_track_id == 7
    assert decision.rss_margin_m < 0.0
    assert decision.control_authority == "none_shadow_only"


def test_adjacent_lane_target_is_visible_candidate_but_not_path_collision():
    evaluator = AutowareAEBShadowEvaluator()

    decision = _evaluate(evaluator, [_track(position=(14.5, 3.6))])

    assert decision.candidate_count == 1
    assert decision.path_collision_candidate_count == 0
    assert decision.path_collision is False
    assert decision.triggered is False


def test_same_speed_lead_does_not_enter_short_horizon_ego_footprint():
    evaluator = AutowareAEBShadowEvaluator()

    decision = _evaluate(
        evaluator,
        [_track(position=(15.0, 0.0), velocity=(10.0, 0.0))],
    )

    assert decision.path_collision is False
    assert decision.raw_triggered is False


def test_trigger_is_held_without_inventing_new_event_or_control():
    evaluator = AutowareAEBShadowEvaluator(
        AutowareAEBShadowConfig(collision_keep_s=3.0)
    )
    first = _evaluate(evaluator, [_track()], timestamp=1.0)
    held = _evaluate(evaluator, [], timestamp=2.0)
    expired = _evaluate(evaluator, [], timestamp=4.1)

    assert first.new_trigger_event is True
    assert held.triggered is True
    assert held.raw_triggered is False
    assert held.held is True
    assert held.new_trigger_event is False
    assert held.reason == "collision_keep_hold"
    assert expired.triggered is False


def test_tentative_or_stale_track_cannot_enter_aeb_decision():
    evaluator = AutowareAEBShadowEvaluator()

    decision = _evaluate(evaluator, [
        _track(track_id=1, lifecycle="tentative"),
        _track(track_id=2, missed=4),
    ])

    assert decision.candidate_count == 0
    assert decision.triggered is False


def test_rear_target_and_reverse_motion_do_not_trigger_front_aeb():
    evaluator = AutowareAEBShadowEvaluator()
    rear = _evaluate(evaluator, [_track(position=(-2.0, 0.0))])
    reverse = _evaluate(
        AutowareAEBShadowEvaluator(), [_track()], speed=-5.0
    )

    assert rear.path_collision is False
    assert rear.triggered is False
    assert reverse.ego_speed_mps == 0.0
    assert reverse.triggered is False


def test_summary_contains_reference_confusion_matrix_and_no_truth_contract():
    evaluator = AutowareAEBShadowEvaluator()
    _evaluate(evaluator, [_track()], reference=True)
    _evaluate(evaluator, [], timestamp=5.0, reference=False)

    summary = evaluator.summary()

    assert summary["truth_input_used"] is False
    assert summary["control_authority"] == "none_shadow_only"
    assert summary["reference_comparison"]["frames"] == 2
    assert summary["reference_comparison"]["agreement_rate"] == 1.0
