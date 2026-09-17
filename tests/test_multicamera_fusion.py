"""Stage4.3 多相机融合模块单元测试。"""

import unittest

import numpy as np

from fusion_tracking_core.multicamera_fusion import (
    MultiCameraFusionConfig,
    fuse_multicamera_detections,
)


def detection(channel, center, indices, score=0.9, extent=(2.0, 1.0)):
    return {
        "camera_channel": channel,
        "class_name": "car",
        "score": score,
        "center_ego": np.asarray(center, dtype=float),
        "extent_xy": np.asarray(extent, dtype=float),
        "n_lidar": len(indices),
        "lidar_indices": np.asarray(indices, dtype=int),
        "depth_m": float(center[0]),
        "lidar_depth_spread_m": 0.4,
        "lidar_ground_filtered": False,
        "lidar_xy_trimmed": False,
    }


class TestMultiCameraFusion(unittest.TestCase):
    def test_adjacent_views_with_shared_lidar_points_are_merged(self):
        detections = [
            detection("CAM_FRONT", (12.0, 2.0), [1, 2, 3, 4]),
            detection("CAM_FRONT_RIGHT", (12.2, 2.1), [2, 3, 4, 5], score=0.8),
        ]
        fused, stats = fuse_multicamera_detections(
            detections, MultiCameraFusionConfig(mode="overlap_aware")
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["n_camera_views"], 2)
        self.assertEqual(stats.merged_detections, 1)
        self.assertEqual(stats.lidar_overlap_links, 1)

    def test_non_adjacent_cameras_are_not_merged(self):
        detections = [
            detection("CAM_FRONT", (12.0, 2.0), [1, 2, 3]),
            detection("CAM_BACK", (12.1, 2.0), [1, 2, 3]),
        ]
        fused, _ = fuse_multicamera_detections(
            detections, MultiCameraFusionConfig(mode="overlap_aware")
        )
        self.assertEqual(len(fused), 2)

    def test_close_parallel_objects_without_overlap_are_kept(self):
        detections = [
            detection("CAM_FRONT", (15.0, 0.0), [1, 2, 3], extent=(1.0, 0.6)),
            detection("CAM_FRONT_RIGHT", (15.0, 1.5), [7, 8, 9], extent=(1.0, 0.6)),
        ]
        fused, _ = fuse_multicamera_detections(
            detections, MultiCameraFusionConfig(mode="overlap_aware")
        )
        self.assertEqual(len(fused), 2)

    def test_same_camera_family_duplicate_is_suppressed(self):
        detections = [
            detection("CAM_FRONT", (10.0, 0.0), [1, 2], score=0.9),
            detection("CAM_FRONT", (11.0, 0.0), [8, 9], score=0.8),
        ]
        fused, _ = fuse_multicamera_detections(
            detections, MultiCameraFusionConfig(mode="overlap_aware")
        )
        self.assertEqual(len(fused), 1)

    def test_cross_camera_uses_ego_range_not_camera_axis_depth(self):
        first = detection("CAM_FRONT", (12.0, 2.0), [1, 2, 3])
        second = detection("CAM_FRONT_RIGHT", (12.1, 2.1), [1, 2, 4])
        first["depth_m"], second["depth_m"] = 12.0, 7.0
        fused, _ = fuse_multicamera_detections(
            [first, second], MultiCameraFusionConfig(mode="overlap_aware")
        )
        self.assertEqual(len(fused), 1)

    def test_baseline_reproduces_fixed_radius_suppression(self):
        detections = [
            detection("CAM_FRONT", (10.0, 0.0), [1, 2], score=0.9),
            detection("CAM_BACK", (12.0, 0.0), [8, 9], score=0.8),
        ]
        fused, stats = fuse_multicamera_detections(
            detections, MultiCameraFusionConfig(mode="baseline")
        )
        self.assertEqual(len(fused), 1)
        self.assertEqual(stats.merged_detections, 1)


if __name__ == "__main__":
    unittest.main()
