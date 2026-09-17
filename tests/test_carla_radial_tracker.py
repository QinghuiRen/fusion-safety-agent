from dataclasses import replace

import numpy as np

from fusion_tracking_core import AssociationConfig
from fusion_tracking_carla.config import TrackerConfig
from fusion_tracking_carla.tracker import StreamingFusionTracker, radial_velocity_measurement
from fusion_tracking_carla.types import FusionDetection, RadarReturn, TrackSnapshot


def _radar(unit=(1.0, 0.0), range_rate=5.0):
    return RadarReturn(
        position_global=np.array([10.0, 0.0, 0.0]),
        sensor_origin_global=np.zeros(3),
        radial_unit_global=np.asarray(unit, dtype=float),
        velocity_towards_mps=range_rate,
        sensor_name="RADAR_FRONT",
    )


def test_radial_measurement_compensates_ego_velocity_and_sign():
    # 静止目标、ego沿视线方向以3m/s靠近：CARLA回波为-3m/s，补偿后应为0。
    h, z = radial_velocity_measurement(
        _radar(range_rate=-3.0), np.array([3.0, 0.0])
    )
    assert np.allclose(h, [0.0, 0.0, 1.0, 0.0])
    assert np.isclose(z, 0.0)

    # 目标绝对速度5m/s、ego为2m/s：距离以+3m/s增加，补偿后恢复5m/s。
    _, z = radial_velocity_measurement(
        _radar(range_rate=3.0), np.array([2.0, 0.0])
    )
    assert np.isclose(z, 5.0)


def test_radial_update_constrains_only_observed_velocity_direction():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        radial_velocity_std_mps=0.2,
        radial_outlier_gate_sigma=20.0,
    ))
    detection = FusionDetection(
        timestamp_s=1.0,
        position_global=np.array([10.0, 0.0]),
        extent_xy=np.array([4.0, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.9,
        position_std_m=0.5,
        lidar_support=20,
        radar_returns=[_radar(range_rate=5.0)],
    )
    snapshot = tracker.update([detection], 1.0, np.zeros(2))[0]
    assert snapshot.state_global[2] > 4.9
    assert np.isclose(snapshot.state_global[3], 0.0)
    assert snapshot.radar_support == 1


def test_tracker_persists_id_across_position_updates():
    tracker = StreamingFusionTracker(TrackerConfig(min_confirmed_hits=2))
    base = dict(
        extent_xy=np.array([4.0, 2.0]), class_name="car", family="vehicle",
        score=0.9, position_std_m=0.5, lidar_support=10,
    )
    first = FusionDetection(timestamp_s=0.0, position_global=np.array([5.0, 0.0]), **base)
    second = FusionDetection(timestamp_s=0.5, position_global=np.array([6.0, 0.0]), **base)
    track1 = tracker.update([first], 0.0, np.zeros(2))[0]
    track2 = tracker.update([second], 0.5, np.zeros(2))[0]
    assert track1.track_id == track2.track_id
    assert track2.lifecycle == "confirmed"


def test_model_cartesian_velocity_initializes_track_and_is_diagnostic():
    tracker = StreamingFusionTracker(TrackerConfig(min_confirmed_hits=1))
    detection = FusionDetection(
        timestamp_s=1.0,
        position_global=np.array([10.0, 2.0]),
        extent_xy=np.array([4.0, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.9,
        position_std_m=0.5,
        velocity_global=np.array([3.0, -1.0]),
        velocity_std_mps=4.0,
        velocity_source="bevfusion_velocity_head",
    )
    snapshot = tracker.update([detection], 1.0, np.zeros(2))[0]
    np.testing.assert_allclose(snapshot.state_global[2:], [3.0, -1.0])
    assert tracker.diagnostics()["model_velocity_updates"] == 1


def test_radar_refines_model_velocity_after_cartesian_prior():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        radial_velocity_std_mps=0.2,
        radial_outlier_gate_sigma=20.0,
    ))
    detection = FusionDetection(
        timestamp_s=1.0,
        position_global=np.array([10.0, 0.0]),
        extent_xy=np.array([4.0, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.9,
        position_std_m=0.5,
        velocity_global=np.array([1.0, 2.0]),
        velocity_std_mps=4.0,
        velocity_source="bevfusion_velocity_head",
        radar_returns=[_radar(range_rate=5.0)],
    )
    snapshot = tracker.update([detection], 1.0, np.zeros(2))[0]
    assert snapshot.state_global[2] > 4.9
    assert np.isclose(snapshot.state_global[3], 2.0)
    assert snapshot.radar_support == 1


def test_low_score_detection_cannot_create_track_above_birth_threshold():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        new_track_score_threshold=0.25,
    ))
    low = FusionDetection(
        timestamp_s=0.0,
        position_global=np.array([5.0, 0.0]),
        extent_xy=np.array([4.0, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.20,
        position_std_m=0.5,
    )
    assert tracker.update([low], 0.0, np.zeros(2)) == []
    assert tracker.diagnostics()["new_tracks"] == 0
    assert tracker.diagnostics()["suppressed_low_score_births"] == 1


def test_low_score_detection_can_continue_existing_high_score_track():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        new_track_score_threshold=0.25,
    ))
    common = dict(
        extent_xy=np.array([4.0, 2.0]),
        class_name="car",
        family="vehicle",
        position_std_m=0.5,
    )
    high = FusionDetection(
        timestamp_s=0.0,
        position_global=np.array([5.0, 0.0]),
        score=0.90,
        **common,
    )
    low = FusionDetection(
        timestamp_s=0.5,
        position_global=np.array([5.5, 0.0]),
        score=0.20,
        **common,
    )
    first = tracker.update([high], 0.0, np.zeros(2))[0]
    second = tracker.update([low], 0.5, np.zeros(2))[0]
    assert second.track_id == first.track_id
    assert tracker.diagnostics()["new_tracks"] == 1
    assert tracker.diagnostics()["suppressed_low_score_births"] == 0


def test_track_measurement_age_is_distinct_from_prediction_epoch_and_prunes_stale():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        max_missed_updates=100,
        max_prediction_age_s=1.0,
    ))
    detection = FusionDetection(
        timestamp_s=0.0,
        position_global=np.array([5.0, 0.0]),
        extent_xy=np.array([4.0, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.9,
        position_std_m=0.5,
    )
    tracker.update([detection], 0.0, np.zeros(2))

    predicted = tracker.update([], 0.5, np.zeros(2))[0]
    assert predicted.timestamp_s == 0.5
    assert predicted.measurement_timestamp_s == 0.0
    assert predicted.to_dict()["measurement_age_s"] == 0.5

    assert tracker.update([], 1.1, np.zeros(2)) == []
    assert tracker.diagnostics()["stale_measurement_prunes"] == 1


def test_live_radar_refines_read_only_snapshot_without_mutating_base_track():
    tracker = StreamingFusionTracker(TrackerConfig(
        enable_live_radar_refinement=True,
        min_confirmed_hits=1,
        radial_velocity_std_mps=0.2,
        radial_outlier_gate_sigma=20.0,
        live_radar_association_gate_m=3.2,
    ))
    original = TrackSnapshot(
        track_id=7,
        class_name="car",
        family="vehicle",
        lifecycle="confirmed",
        timestamp_s=2.0,
        state_global=np.array([10.0, 0.0, 0.0, 0.0]),
        covariance_global=np.diag([1.0, 1.0, 25.0, 25.0]),
        extent_xy=np.array([4.0, 2.0]),
        score=0.9,
        age_updates=3,
        missed_updates=0,
        lidar_support=10,
        radar_support=0,
        camera_channels=("CAM_FRONT",),
        radial_velocity_std_mps=None,
    )
    refined, stats = tracker.refine_snapshots_with_radar(
        [original], [_radar(range_rate=5.0)], np.zeros(2)
    )
    assert original.state_global[2] == 0.0
    assert refined[0].state_global[2] > 4.9
    assert refined[0].radar_support == 1
    assert stats["accepted_returns"] == 1
    assert stats["refined_tracks"] == 1


def test_live_radar_memory_carries_fast_state_without_mutating_visual_base():
    tracker = StreamingFusionTracker(TrackerConfig(
        enable_live_radar_refinement=True,
        min_confirmed_hits=1,
        radial_velocity_std_mps=0.2,
        radial_outlier_gate_sigma=20.0,
        live_radar_memory_max_age_s=1.0,
    ))
    base = TrackSnapshot(
        track_id=9,
        class_name="car",
        family="vehicle",
        lifecycle="confirmed",
        timestamp_s=2.0,
        state_global=np.array([10.0, 0.0, 0.0, 0.0]),
        covariance_global=np.diag([1.0, 1.0, 25.0, 25.0]),
        extent_xy=np.array([4.0, 2.0]),
        score=0.9,
        age_updates=3,
        missed_updates=0,
        lidar_support=10,
        radar_support=0,
        camera_channels=("CAM_FRONT",),
        radial_velocity_std_mps=None,
    )
    first, _ = tracker.refine_snapshots_with_radar(
        [base], [_radar(range_rate=5.0)], np.zeros(2)
    )
    next_visual_base = replace(base, timestamp_s=2.05)
    carried, stats = tracker.refine_snapshots_with_radar(
        [next_visual_base], [], np.zeros(2)
    )
    assert first[0].state_global[2] > 4.9
    assert base.state_global[2] == 0.0
    assert next_visual_base.state_global[2] == 0.0
    assert carried[0].state_global[2] > 4.9
    assert carried[0].state_global[0] > first[0].state_global[0]
    assert stats["memory_seeded_tracks"] == 1
    assert stats["memory_carried_tracks"] == 1


def test_live_radar_memory_resets_when_new_visual_update_arrives():
    tracker = StreamingFusionTracker(TrackerConfig(
        enable_live_radar_refinement=True,
        radial_velocity_std_mps=0.2,
        radial_outlier_gate_sigma=20.0,
    ))
    base = TrackSnapshot(
        track_id=10, class_name="car", family="vehicle", lifecycle="confirmed",
        timestamp_s=2.0, state_global=np.array([10.0, 0.0, 0.0, 0.0]),
        covariance_global=np.diag([1.0, 1.0, 25.0, 25.0]),
        extent_xy=np.array([4.0, 2.0]), score=0.9, age_updates=3,
        missed_updates=0, lidar_support=4, radar_support=0,
        camera_channels=("CAM_FRONT",), radial_velocity_std_mps=None,
    )
    tracker.refine_snapshots_with_radar(
        [base], [_radar(range_rate=5.0)], np.zeros(2)
    )
    new_visual = replace(base, timestamp_s=2.05, age_updates=4)
    refined, stats = tracker.refine_snapshots_with_radar(
        [new_visual], [], np.zeros(2)
    )
    assert np.array_equal(refined[0].state_global, new_visual.state_global)
    assert stats["memory_resets"] == 1


def test_live_radar_rejects_return_outside_spatial_gate():
    tracker = StreamingFusionTracker(TrackerConfig(
        enable_live_radar_refinement=True,
        min_confirmed_hits=1,
        live_radar_association_gate_m=1.0,
    ))
    snapshot = TrackSnapshot(
        track_id=8, class_name="car", family="vehicle", lifecycle="confirmed",
        timestamp_s=2.0, state_global=np.zeros(4), covariance_global=np.eye(4),
        extent_xy=np.array([4.0, 2.0]), score=0.8, age_updates=2,
        missed_updates=0, lidar_support=4, radar_support=0,
        camera_channels=("CAM_FRONT",), radial_velocity_std_mps=None,
    )
    refined, stats = tracker.refine_snapshots_with_radar(
        [snapshot], [_radar(range_rate=5.0)], np.zeros(2)
    )
    assert np.array_equal(refined[0].state_global, snapshot.state_global)
    assert stats["spatially_assigned_returns"] == 0
    assert stats["refined_tracks"] == 0


def test_recovery_association_keeps_stable_id_after_center_jump():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        enable_recovery_association=True,
        recovery_vehicle_gate_m=9.0,
        association=AssociationConfig(
            mode="mahalanobis",
            vehicle_hard_gate_m=7.0,
            position_mahalanobis_gate=0.1,
        ),
    ))
    base = dict(
        extent_xy=np.array([4.0, 2.0]), class_name="car", family="vehicle",
        score=0.9, position_std_m=0.5, lidar_support=10,
        camera_channels=("CAM_FRONT",),
    )
    first = FusionDetection(timestamp_s=0.0, position_global=np.array([5.0, 0.0]), **base)
    jumped = FusionDetection(timestamp_s=0.5, position_global=np.array([9.0, 0.0]), **base)
    first_id = tracker.update([first], 0.0, np.zeros(2))[0].track_id
    second = tracker.update([jumped], 0.5, np.zeros(2))
    assert len(second) == 1
    assert second[0].track_id == first_id
    assert tracker.diagnostics()["recovery_matches"] == 1


def test_recovery_association_rejects_nonadjacent_camera_jump():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        enable_recovery_association=True,
        association=AssociationConfig(
            mode="mahalanobis",
            vehicle_hard_gate_m=7.0,
            position_mahalanobis_gate=0.1,
        ),
    ))
    common = dict(
        extent_xy=np.array([4.0, 2.0]), class_name="car", family="vehicle",
        score=0.9, position_std_m=0.5, lidar_support=10,
    )
    first = FusionDetection(
        timestamp_s=0.0, position_global=np.array([5.0, 0.0]),
        camera_channels=("CAM_FRONT",), **common,
    )
    opposite = FusionDetection(
        timestamp_s=0.5, position_global=np.array([9.0, 0.0]),
        camera_channels=("CAM_BACK",), **common,
    )
    tracker.update([first], 0.0, np.zeros(2))
    snapshots = tracker.update([opposite], 0.5, np.zeros(2))
    assert len(snapshots) == 2
    assert tracker.diagnostics()["recovery_matches"] == 0


def test_radar_only_track_survives_sparse_visual_updates_but_lidar_track_does_not():
    config = TrackerConfig(
        min_confirmed_hits=1,
        max_missed_updates=4,
        enable_radar_only_track_continuity=True,
        radar_only_max_missed_updates=8,
        radar_only_publish_max_missed_updates=3,
    )
    common = dict(
        timestamp_s=0.0,
        position_global=np.array([30.0, 0.0]),
        extent_xy=np.array([4.8, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.9,
        position_std_m=2.5,
        camera_channels=("CAM_FRONT",),
    )
    radar_tracker = StreamingFusionTracker(config)
    radar_tracker.update([
        FusionDetection(lidar_support=0, radar_returns=[_radar(range_rate=0.0)], **common)
    ], 0.0, np.zeros(2))
    radar_tracks = []
    for index in range(1, 7):
        radar_tracks = radar_tracker.update([], 0.25 * index, np.zeros(2))
    # 对外不再发布过期CV预测，但轨迹仍在内部等待原ID恢复。
    assert radar_tracks == []
    assert len(radar_tracker._tracks) == 1

    lidar_tracker = StreamingFusionTracker(config)
    lidar_tracker.update([
        FusionDetection(lidar_support=8, radar_returns=[], **common)
    ], 0.0, np.zeros(2))
    lidar_tracks = []
    for index in range(1, 7):
        lidar_tracks = lidar_tracker.update([], 0.25 * index, np.zeros(2))
    assert lidar_tracks == []


def test_unconfirmed_missed_track_remains_tentative_not_lost():
    tracker = StreamingFusionTracker(TrackerConfig(min_confirmed_hits=2))
    detection = FusionDetection(
        timestamp_s=0.0,
        position_global=np.array([20.0, 0.0]),
        extent_xy=np.array([4.8, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.9,
        position_std_m=1.0,
        lidar_support=4,
        radar_returns=[],
        camera_channels=("CAM_FRONT",),
    )
    tracker.update([detection], 0.0, np.zeros(2))
    snapshots = tracker.update([], 0.25, np.zeros(2))
    assert len(snapshots) == 1
    assert snapshots[0].lifecycle == "tentative"


def test_radar_only_track_reacquires_lidar_detection_without_new_id():
    tracker = StreamingFusionTracker(TrackerConfig(
        min_confirmed_hits=1,
        enable_recovery_association=True,
        enable_radar_only_track_continuity=True,
        recovery_vehicle_gate_m=9.0,
        radar_only_recovery_vehicle_gate_m=18.0,
        association=AssociationConfig(
            mode="mahalanobis",
            vehicle_hard_gate_m=7.0,
            position_mahalanobis_gate=0.1,
        ),
    ))
    common = dict(
        extent_xy=np.array([4.8, 2.0]),
        class_name="car",
        family="vehicle",
        score=0.9,
        camera_channels=("CAM_FRONT",),
    )
    radar_detection = FusionDetection(
        timestamp_s=0.0,
        position_global=np.array([10.0, 0.0]),
        position_std_m=2.5,
        lidar_support=0,
        radar_returns=[_radar(range_rate=0.0)],
        **common,
    )
    first_id = tracker.update([radar_detection], 0.0, np.zeros(2))[0].track_id
    lidar_detection = FusionDetection(
        timestamp_s=0.5,
        position_global=np.array([25.0, 0.0]),
        position_std_m=0.8,
        lidar_support=8,
        radar_returns=[_radar(range_rate=0.0)],
        **common,
    )
    snapshots = tracker.update([lidar_detection], 0.5, np.zeros(2))
    assert len(snapshots) == 1
    assert snapshots[0].track_id == first_id
    assert snapshots[0].lidar_support == 8
    diagnostics = tracker.diagnostics()
    assert diagnostics["recovery_matches"] == 1
    assert diagnostics["radar_only_continuity_matches"] == 1
    assert diagnostics["new_tracks"] == 1
