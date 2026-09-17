"""Stage4.5轨迹关联模块单元测试。"""

import unittest

import numpy as np

from fusion_tracking_core.association import AssociationConfig, associate_tracks


def track(position, extent=(4.0, 2.0), covariance=1.0):
    return {
        "family": "vehicle",
        "position": np.asarray(position, dtype=float),
        "velocity": np.array([2.0, 0.0]),
        "position_covariance": np.eye(2) * covariance,
        "velocity_covariance": np.eye(2) * covariance,
        "extent_xy": np.asarray(extent, dtype=float),
    }


def detection(position, extent=(4.0, 2.0), velocity=(2.0, 0.0), has_velocity=True):
    return {
        "family": "vehicle",
        "position": np.asarray(position, dtype=float),
        "velocity": np.asarray(velocity, dtype=float),
        "position_std_m": 1.8,
        "velocity_std_mps": 2.0,
        "has_velocity": has_velocity,
        "extent_xy": np.asarray(extent, dtype=float),
    }


class TestAssociation(unittest.TestCase):
    def test_baseline_matches_nearest_detection(self):
        result = associate_tracks(
            [track((0.0, 0.0))],
            [detection((1.0, 0.0)), detection((4.0, 0.0))],
            AssociationConfig(mode="baseline"),
        )
        self.assertEqual(result.matches, [(0, 0)])

    def test_mahalanobis_allows_uncertain_track_reconnection(self):
        config = AssociationConfig(
            mode="mahalanobis", vehicle_hard_gate_m=6.0, position_mahalanobis_gate=3.0
        )
        low_uncertainty = associate_tracks(
            [track((0.0, 0.0), covariance=0.01)],
            [detection((5.5, 0.0))],
            config,
        )
        high_uncertainty = associate_tracks(
            [track((0.0, 0.0), covariance=9.0)],
            [detection((5.5, 0.0))],
            config,
        )
        self.assertEqual(low_uncertainty.matches, [])
        self.assertEqual(high_uncertainty.matches, [(0, 0)])

    def test_size_consistency_breaks_equal_position_tie(self):
        result = associate_tracks(
            [track((0.0, 0.0), extent=(4.0, 2.0))],
            [
                detection((1.0, 0.0), extent=(1.0, 0.5)),
                detection((1.0, 0.0), extent=(4.2, 2.0)),
            ],
            AssociationConfig(mode="mahalanobis", size_log_ratio_weight=1.0),
        )
        self.assertEqual(result.matches, [(0, 1)])

    def test_velocity_is_ignored_without_radar_observation(self):
        result = associate_tracks(
            [track((0.0, 0.0))],
            [detection((0.5, 0.0), velocity=(50.0, 0.0), has_velocity=False)],
            AssociationConfig(mode="mahalanobis"),
        )
        self.assertEqual(result.matches, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
