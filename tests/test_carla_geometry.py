import numpy as np

from fusion_tracking_carla.geometry import (
    carla_xyz_to_internal,
    ego_to_global_xy,
    global_to_ego_xy,
    project_carla_world_to_camera,
)


def test_carla_internal_conversion_is_its_own_inverse():
    points = np.array([[2.0, 3.0, 4.0], [-1.0, -5.0, 0.2]])
    assert np.allclose(carla_xyz_to_internal(carla_xyz_to_internal(points)), points)
    assert np.allclose(carla_xyz_to_internal(points)[0], [2.0, -3.0, 4.0])


def test_ego_global_round_trip():
    ego = np.array([12.0, -7.0, np.deg2rad(37.0)])
    local = np.array([[5.0, 1.0], [-2.0, 3.0]])
    assert np.allclose(global_to_ego_xy(ego_to_global_xy(local, ego), ego), local)


def test_camera_projection_uses_carla_camera_axis_mapping():
    points_world = np.array([[10.0, 0.0, 0.0], [10.0, 2.0, 0.0]])
    intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    uv, depth, valid = project_carla_world_to_camera(points_world, np.eye(4), intrinsic)
    assert valid.tolist() == [True, True]
    assert np.allclose(depth, [10.0, 10.0])
    assert np.allclose(uv[0], [50.0, 50.0])
    assert np.allclose(uv[1], [70.0, 50.0])
