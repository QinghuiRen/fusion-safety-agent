"""CARLA 左手坐标与模块右手坐标之间的纯 NumPy 几何工具。"""

from __future__ import annotations

import numpy as np


_FLIP_Y_3 = np.diag([1.0, -1.0, 1.0])


def carla_xyz_to_internal(points_xyz) -> np.ndarray:
    """CARLA (x前,y右,z上) -> 内部 (x前,y左,z上)。"""

    points = np.asarray(points_xyz, dtype=float)
    if points.shape[-1] != 3:
        raise ValueError("points_xyz 最后一维必须为 3")
    return points @ _FLIP_Y_3


def internal_xyz_to_carla(points_xyz) -> np.ndarray:
    """内部右手系 -> CARLA 左手系；该变换自身即为逆。"""

    return carla_xyz_to_internal(points_xyz)


def transform_points(points_xyz, matrix_4x4) -> np.ndarray:
    """用齐次矩阵变换 Nx3 点，矩阵采用 CARLA/NumPy 列向量约定。"""

    points = np.asarray(points_xyz, dtype=float)
    matrix = np.asarray(matrix_4x4, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or matrix.shape != (4, 4):
        raise ValueError("points 必须为 Nx3，matrix 必须为 4x4")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :3]


def pose2d_matrix(x_m: float, y_m: float, yaw_rad: float) -> np.ndarray:
    """右手二维 pose 的 ego->global 3x3 齐次矩阵。"""

    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, x_m], [s, c, y_m], [0.0, 0.0, 1.0]])


def global_to_ego_xy(points_global, ego_pose_xyyaw) -> np.ndarray:
    """把 global XY 转到当前 ego，ego_pose=(x,y,yaw_rad)。"""

    points = np.asarray(points_global, dtype=float)
    x_m, y_m, yaw_rad = np.asarray(ego_pose_xyyaw, dtype=float)
    rotation = np.array([
        [np.cos(yaw_rad), np.sin(yaw_rad)],
        [-np.sin(yaw_rad), np.cos(yaw_rad)],
    ])
    return (points - np.array([x_m, y_m])) @ rotation.T


def ego_to_global_xy(points_ego, ego_pose_xyyaw) -> np.ndarray:
    """把当前 ego XY 转到 global。"""

    points = np.asarray(points_ego, dtype=float)
    x_m, y_m, yaw_rad = np.asarray(ego_pose_xyyaw, dtype=float)
    rotation = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad), np.cos(yaw_rad)],
    ])
    return points @ rotation.T + np.array([x_m, y_m])


def global_vector_to_ego(vectors_global, ego_yaw_rad: float) -> np.ndarray:
    """只旋转向量，不施加平移。"""

    vectors = np.asarray(vectors_global, dtype=float)
    c, s = np.cos(ego_yaw_rad), np.sin(ego_yaw_rad)
    rotation = np.array([[c, s], [-s, c]])
    return vectors @ rotation.T


def camera_intrinsic(width: int, height: int, horizontal_fov_deg: float) -> np.ndarray:
    """由水平 FOV 构造针孔相机内参。"""

    focal = width / (2.0 * np.tan(np.deg2rad(horizontal_fov_deg) / 2.0))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def project_carla_world_to_camera(points_world_carla, world_to_sensor, intrinsic):
    """投影 CARLA world 点，返回 uv、相机深度和有效深度掩码。

    CARLA/UE 相机局部轴为 x前、y右、z上；转换到 CV 相机轴时使用
    (x_cv, y_cv, z_cv)=(y_ue, -z_ue, x_ue)。
    """

    sensor_points = transform_points(points_world_carla, world_to_sensor)
    cv_points = np.column_stack([
        sensor_points[:, 1], -sensor_points[:, 2], sensor_points[:, 0]
    ])
    depth = cv_points[:, 2]
    valid = depth > 1e-3
    uvw = (np.asarray(intrinsic, dtype=float) @ cv_points.T).T
    uv = np.full((len(cv_points), 2), np.nan, dtype=float)
    uv[valid] = uvw[valid, :2] / uvw[valid, 2:3]
    return uv, depth, valid


def carla_transform_to_internal_pose(transform) -> np.ndarray:
    """从 carla.Transform 取右手 global (x,y,yaw_rad)，不在模块顶层依赖 carla。"""

    return np.array([
        float(transform.location.x),
        -float(transform.location.y),
        -np.deg2rad(float(transform.rotation.yaw)),
    ])
