"""Stage4.4 Radar速度质量模块单元测试。"""

import unittest

import numpy as np

from fusion_tracking_core.radar_velocity import (
    RadarVelocityConfig,
    estimate_radar_velocity,
)


class TestRadarVelocity(unittest.TestCase):
    def test_baseline_uses_all_points_and_median(self):
        radar = {
            "xy": np.array([[0.0, 0.0], [0.2, 0.0], [0.3, 0.0]]),
            "velocity": np.array([[2.0, 0.0], [2.2, 0.0], [20.0, 0.0]]),
        }
        obs = estimate_radar_velocity(
            radar, [0.0, 0.0], [2.0, 1.0], RadarVelocityConfig(mode="baseline")
        )
        self.assertTrue(obs.accepted)
        self.assertEqual(obs.n_used, 3)
        self.assertAlmostEqual(obs.velocity_xy[0], 2.2)

    def test_robust_rejects_spatial_neighbor_and_velocity_outlier(self):
        radar = {
            "xy": np.array([
                [0.1, 0.0], [0.2, 0.1], [0.3, -0.1], [0.4, 0.0], [2.3, 0.0]
            ]),
            "velocity": np.array([
                [3.0, 0.0], [3.1, 0.1], [2.9, -0.1], [30.0, 20.0], [-5.0, 0.0]
            ]),
            "quality_valid": np.ones(5),
        }
        obs = estimate_radar_velocity(
            radar, [0.0, 0.0], [2.0, 1.0], RadarVelocityConfig(mode="robust")
        )
        self.assertTrue(obs.accepted)
        self.assertEqual(obs.n_candidates, 5)
        self.assertEqual(obs.n_used, 3)
        self.assertLess(np.linalg.norm(obs.velocity_xy - [3.0, 0.0]), 0.15)

    def test_single_point_is_kept_with_large_uncertainty(self):
        radar = {
            "xy": np.array([[1.0, 0.0]]),
            "velocity": np.array([[4.0, 0.0]]),
            "quality_valid": np.array([1]),
        }
        cfg = RadarVelocityConfig(mode="robust", single_point_std_mps=4.0)
        obs = estimate_radar_velocity(radar, [1.0, 0.0], [1.0, 1.0], cfg)
        self.assertTrue(obs.accepted)
        # 默认1.25保守系数作用后为4.0*1.25=5.0 m/s。
        self.assertEqual(obs.velocity_std_mps, 5.0)

    def test_empty_candidate_returns_no_observation(self):
        radar = {
            "xy": np.array([[20.0, 20.0]]),
            "velocity": np.array([[1.0, 0.0]]),
        }
        obs = estimate_radar_velocity(
            radar, [0.0, 0.0], [1.0, 1.0], RadarVelocityConfig(mode="robust")
        )
        self.assertFalse(obs.accepted)
        self.assertEqual(obs.n_used, 0)


if __name__ == "__main__":
    unittest.main()
