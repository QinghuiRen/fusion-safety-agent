# -*- coding: utf-8 -*-
import unittest

import numpy as np

from fusion_tracking_core.margin_safety import (
    MarginSafetyConfig,
    assess_kinematic_update,
    predict_margin_safety,
    rotate_cv_covariance,
)
from fusion_tracking_carla.safety import assess_tracks
from fusion_tracking_carla.types import TrackSnapshot


class TestMarginSafety(unittest.TestCase):
    def setUp(self):
        self.covariance = np.diag([0.2, 0.2, 0.5, 0.5])

    def test_far_stationary_target_is_low_risk(self):
        result = predict_margin_safety(
            [40.0, 15.0], [0.0, 0.0], self.covariance, [4.5, 2.0]
        )
        self.assertEqual(result["risk_level"], "low")
        self.assertFalse(result["predicted_margin_breach"])

    def test_approaching_target_breaches_margin(self):
        result = predict_margin_safety(
            [15.0, 0.0], [-8.0, 0.0], self.covariance, [4.5, 2.0]
        )
        self.assertTrue(result["predicted_margin_breach"])
        self.assertLessEqual(result["time_to_margin_s"], 1.5)
        self.assertEqual(result["risk_level"], "critical")

    def test_larger_covariance_is_more_conservative(self):
        small = predict_margin_safety(
            [18.0, 0.0], [-2.0, 0.0], self.covariance, [4.5, 2.0]
        )
        large = predict_margin_safety(
            [18.0, 0.0], [-2.0, 0.0], np.diag([9.0, 9.0, 9.0, 9.0]), [4.5, 2.0]
        )
        self.assertGreater(large["risk_score"], small["risk_score"])

    def test_kinematic_violation_is_reported(self):
        result = assess_kinematic_update(
            [0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [20.0, 0.0], 0.5, "car"
        )
        self.assertIn("acceleration_violation", result["flags"])
        self.assertLess(result["physics_consistency_score"], 100.0)

    def test_horizon_weight_length_is_validated(self):
        config = MarginSafetyConfig(horizon_weights=(1.0, 1.0))
        with self.assertRaises(ValueError):
            predict_margin_safety(
                [10.0, 0.0], [0.0, 0.0], self.covariance, [4.0, 2.0], config=config
            )

    def test_cv_covariance_rotates_position_and_velocity_blocks(self):
        covariance = np.diag([1.0, 4.0, 9.0, 16.0])
        rotation = np.array([[0.0, 1.0], [-1.0, 0.0]])
        rotated = rotate_cv_covariance(covariance, rotation)
        np.testing.assert_allclose(np.diag(rotated), [4.0, 1.0, 16.0, 9.0])

    def test_tentative_track_is_not_allowed_to_raise_safety_alert(self):
        common = dict(
            class_name="car",
            family="vehicle",
            timestamp_s=1.0,
            state_global=np.array([4.0, 0.0, 0.0, 0.0]),
            covariance_global=np.diag([0.2, 0.2, 0.5, 0.5]),
            extent_xy=np.array([4.5, 2.0]),
            score=0.99,
            age_updates=1,
            missed_updates=0,
            lidar_support=20,
            radar_support=5,
            camera_channels=("CAM_FRONT",),
            radial_velocity_std_mps=1.0,
        )
        tentative = TrackSnapshot(track_id=1, lifecycle="tentative", **common)
        confirmed = TrackSnapshot(
            track_id=2, lifecycle="confirmed", **{**common, "age_updates": 2}
        )

        tentative_safety = assess_tracks(
            [tentative], np.zeros(3), np.zeros(2), MarginSafetyConfig()
        )
        confirmed_safety = assess_tracks(
            [confirmed], np.zeros(3), np.zeros(2), MarginSafetyConfig()
        )

        self.assertEqual(tentative_safety, [])
        self.assertEqual(len(confirmed_safety), 1)
        self.assertGreater(confirmed_safety[0].risk_score, 0.0)


if __name__ == "__main__":
    unittest.main()
