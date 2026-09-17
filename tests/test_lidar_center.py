"""Stage4.2 LiDAR中心模块的无数据集单元测试。"""

import unittest

import numpy as np

from fusion_tracking_core.lidar_center import (
    LidarCenterConfig,
    estimate_lidar_observation,
)


class TestLidarCenter(unittest.TestCase):
    def test_robust_rejects_ground_and_background_cluster(self):
        lidar = np.array([
            [8.0, 0.0, 0.00],   # 更近但属于地面
            [10.0, -0.2, 0.8],
            [10.2, 0.0, 0.9],
            [10.4, 0.2, 0.7],
            [20.0, -1.0, 1.0],  # 背景簇
            [20.2, 1.0, 1.1],
        ])
        depth = lidar[:, 0].copy()
        obs = estimate_lidar_observation(
            lidar,
            depth,
            np.arange(len(lidar)),
            LidarCenterConfig(mode="robust", min_points=2),
        )
        self.assertIsNotNone(obs)
        self.assertTrue(obs.used_ground_filter)
        self.assertLess(abs(obs.center_xy[0] - 10.2), 0.25)
        self.assertEqual(len(obs.foreground_indices), 3)

    def test_robust_falls_back_when_ground_filter_leaves_too_few_points(self):
        lidar = np.array([[5.0, 0.0, 0.0], [5.2, 0.1, 0.0]])
        obs = estimate_lidar_observation(
            lidar,
            lidar[:, 0],
            np.arange(2),
            LidarCenterConfig(mode="robust", min_points=2),
        )
        self.assertIsNotNone(obs)
        self.assertFalse(obs.used_ground_filter)
        self.assertEqual(len(obs.foreground_indices), 2)

    def test_too_few_points_returns_none(self):
        lidar = np.array([[5.0, 0.0, 1.0]])
        obs = estimate_lidar_observation(
            lidar,
            lidar[:, 0],
            np.array([0]),
            LidarCenterConfig(mode="robust", min_points=2),
        )
        self.assertIsNone(obs)

    def test_radar_guidance_selects_matching_far_depth_cluster(self):
        lidar = np.array([
            [8.0, -0.2, 1.0],
            [8.2, 0.2, 1.1],
            [20.0, -0.3, 1.0],
            [20.2, 0.0, 1.1],
            [20.4, 0.3, 0.9],
        ])
        obs = estimate_lidar_observation(
            lidar,
            lidar[:, 0],
            np.arange(len(lidar)),
            LidarCenterConfig(mode="robust", min_points=2),
            radar_reference_xy=np.array([[20.1, 0.1]]),
        )
        self.assertIsNotNone(obs)
        self.assertTrue(obs.used_radar_guidance)
        self.assertLess(abs(obs.center_xy[0] - 20.2), 0.25)
        self.assertEqual(len(obs.foreground_indices), 3)

    def test_unrelated_radar_falls_back_to_near_depth_cluster(self):
        lidar = np.array([
            [8.0, -0.2, 1.0],
            [8.2, 0.2, 1.1],
            [20.0, -0.3, 1.0],
            [20.2, 0.3, 1.1],
        ])
        obs = estimate_lidar_observation(
            lidar,
            lidar[:, 0],
            np.arange(len(lidar)),
            LidarCenterConfig(mode="robust", min_points=2),
            radar_reference_xy=np.array([[40.0, 15.0]]),
        )
        self.assertIsNotNone(obs)
        self.assertFalse(obs.used_radar_guidance)
        self.assertLess(obs.center_xy[0], 12.0)

    def test_robust_extent_does_not_mix_rejected_background(self):
        lidar = np.array([
            [10.0, -0.3, 1.0],
            [10.2, 0.0, 1.1],
            [10.4, 0.3, 0.9],
            [10.1, 30.0, 1.0],  # similar camera depth, unrelated lateral background
        ])
        obs = estimate_lidar_observation(
            lidar,
            lidar[:, 0],
            np.arange(len(lidar)),
            LidarCenterConfig(mode="robust", min_points=2, xy_mad_scale=1.0),
        )
        self.assertIsNotNone(obs)
        self.assertLess(obs.extent_xy[1], 2.0)

    def test_large_baseline_disagreement_uses_robust_center(self):
        lidar = np.array([
            [8.0, 25.0, 0.0],   # ground point contaminates baseline center
            [10.0, -0.2, 1.0],
            [10.2, 0.0, 1.1],
            [10.4, 0.2, 0.9],
        ])
        obs = estimate_lidar_observation(
            lidar,
            lidar[:, 0],
            np.arange(len(lidar)),
            LidarCenterConfig(
                mode="robust", min_points=2, robust_center_blend=0.1,
                max_center_blend_offset_m=2.0,
            ),
        )
        self.assertIsNotNone(obs)
        self.assertLess(abs(obs.center_xy[1]), 0.5)


if __name__ == "__main__":
    unittest.main()
