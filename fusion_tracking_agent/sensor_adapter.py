"""声明 Leaderboard 传感器，并把原始数据包转换为统一内部坐标与时间契约。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from fusion_tracking_carla.geometry import carla_xyz_to_internal, transform_points
from fusion_tracking_carla.sensors import (
    CameraSample,
    LidarSample,
    RadarSample,
    SynchronizedSensorFrame,
)
from fusion_tracking_carla.types import RadarReturn


CAMERA_YAWS_DEG = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_RIGHT": 60.0,
    "CAM_BACK_RIGHT": 120.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": -120.0,
    "CAM_FRONT_LEFT": -60.0,
}
FULL_RADAR_YAWS_DEG = {
    "RADAR_FRONT": 0.0,
    "RADAR_FRONT_RIGHT": 65.0,
    "RADAR_BACK_RIGHT": 145.0,
    "RADAR_BACK_LEFT": -145.0,
    "RADAR_FRONT_LEFT": -65.0,
}
# Official Leaderboard permits at most four radars. Rear coverage is represented
# by one centred rear radar; native product_full retains the original five.
LEADERBOARD_RADAR_YAWS_DEG = {
    "RADAR_FRONT": 0.0,
    "RADAR_FRONT_RIGHT": 65.0,
    "RADAR_BACK": 180.0,
    "RADAR_FRONT_LEFT": -65.0,
}


def sensor_suite(config) -> list[dict]:
    sensors: list[dict] = []
    for channel, yaw in CAMERA_YAWS_DEG.items():
        sensors.append({
            "type": "sensor.camera.rgb",
            "x": 1.0,
            "y": 0.0,
            "z": 2.15,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": yaw,
            "width": int(config.camera_width),
            "height": int(config.camera_height),
            "fov": float(config.camera_fov_deg),
            "id": channel,
        })
    sensors.append({
        "type": "sensor.lidar.ray_cast",
        "x": 0.0,
        "y": 0.0,
        "z": 2.25,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "channels": 32,
        "range": float(config.lidar_range_m),
        "points_per_second": 160_000,
        "rotation_frequency": 20.0,
        "id": "LIDAR_TOP",
    })
    radars = (
        LEADERBOARD_RADAR_YAWS_DEG
        if config.sensor_profile == "leaderboard"
        else FULL_RADAR_YAWS_DEG
    )
    for channel, yaw in radars.items():
        sensors.append({
            "type": "sensor.other.radar",
            "x": 1.2,
            "y": 0.0,
            "z": 1.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": yaw,
            "horizontal_fov": 80.0,
            "vertical_fov": 20.0,
            "points_per_second": 1000,
            "range": float(config.radar_range_m),
            "id": channel,
        })
    sensors.extend([
        {"type": "sensor.other.gnss", "x": 0.0, "y": 0.0, "z": 0.0, "id": "GPS"},
        {
            "type": "sensor.other.imu",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            # carla_garage's real AgentWrapper requires an explicit rotation
            # for IMU sensors even though an identity transform is intended.
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "id": "IMU",
        },
        {"type": "sensor.speedometer", "reading_frequency": 20.0, "id": "SPEED"},
    ])
    return sensors


def _rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _transform(x: float, y: float, z: float, yaw_deg: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _rotation_matrix(0.0, 0.0, yaw_deg)
    matrix[:3, 3] = [x, y, z]
    return matrix


@dataclass(frozen=True)
class AdaptedAgentFrame:
    sensor_frame: SynchronizedSensorFrame
    ego_pose_xyyaw: np.ndarray
    ego_velocity_global: np.ndarray
    ego_yaw_rate_radps: float
    # Read-only provenance. Leaderboard exposes per-stream frame IDs but only a
    # shared Agent timestamp, so that distinction is kept explicit here.
    source_metadata: dict


class LeaderboardFrameAdapter:
    """Builds the exact data contract used by the native CARLA pipeline.

    GPS establishes only a local ENU origin. No CARLA actor/world truth enters
    detections, tracking, risk, or control.
    """

    EARTH_RADIUS_M = 6_378_137.0

    def __init__(self, config):
        self.config = config
        self.specs = {item["id"]: item for item in sensor_suite(config)}
        self._origin_latlon: tuple[float, float] | None = None

    @staticmethod
    def _payload(input_data: dict, name: str):
        value = input_data[name]
        return value[1] if isinstance(value, tuple) and len(value) == 2 else value

    @staticmethod
    def _frame_id(input_data: dict) -> int:
        frames = LeaderboardFrameAdapter._source_frames(input_data)
        return max(frames.values(), default=0)

    @staticmethod
    def _source_frames(input_data: dict) -> dict[str, int]:
        return {
            str(name): int(value[0])
            for name, value in input_data.items()
            if isinstance(value, tuple) and len(value) == 2
        }

    def _decode_radar_packet(self, raw: np.ndarray) -> tuple[np.ndarray, str]:
        """Normalize known Leaderboard Radar layouts to [v, alt, azi, depth].

        CARLA 0.9.15 exposes [velocity, altitude, azimuth, depth].  Some
        Leaderboard forks still apply an old ``np.flip`` callback and therefore
        deliver [depth, azimuth, altitude, velocity].  Depth is physically
        non-negative and, over a packet, normally has a much larger magnitude
        than radial velocity; that makes the distinction observable without
        simulator truth or a fork-specific flag.
        """

        values = np.asarray(raw, dtype=float).reshape(-1, 4)
        if not len(values):
            return values, "empty"
        endpoint_scale = np.median(np.abs(values[:, [0, 3]]), axis=0)
        first_is_depth = bool(
            endpoint_scale[0] > max(2.0, 1.5 * endpoint_scale[1])
            and np.mean(values[:, 0] >= 0.0) >= 0.95
        )
        if first_is_depth:
            # [depth, azimuth, altitude, velocity] -> canonical
            canonical = values[:, [3, 2, 1, 0]]
            layout = "leaderboard_flipped_depth_azimuth_altitude_velocity"
        else:
            canonical = values
            layout = "canonical_velocity_altitude_azimuth_depth"
        return canonical, layout

    def _pose(self, input_data: dict) -> tuple[np.ndarray, float, float]:
        gps = np.asarray(self._payload(input_data, "GPS"), dtype=float).reshape(-1)
        lat, lon = float(gps[0]), float(gps[1])
        if self._origin_latlon is None:
            self._origin_latlon = (lat, lon)
        lat0, lon0 = self._origin_latlon
        north = self.EARTH_RADIUS_M * math.radians(lat - lat0)
        east = self.EARTH_RADIUS_M * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
        imu = np.asarray(self._payload(input_data, "IMU"), dtype=float).reshape(-1)
        compass = float(imu[-1]) if len(imu) else 0.0
        if not np.isfinite(compass):
            compass = 0.0
        # Leaderboard compass: 0=north, pi/2=east. Internal yaw: 0=east, +CCW=north.
        yaw_internal = float(np.pi / 2.0 - compass)
        yaw_rate = float(imu[5]) if len(imu) > 5 and np.isfinite(imu[5]) else 0.0
        return np.array([east, north, yaw_internal], dtype=float), yaw_rate, compass

    def convert(self, input_data: dict, timestamp_s: float) -> AdaptedAgentFrame:
        required = set(self.specs)
        missing = sorted(required - set(input_data))
        if missing:
            raise KeyError(f"Leaderboard输入缺少传感器: {missing}")
        source_frames = self._source_frames(input_data)
        frame = max(source_frames.values(), default=0)
        pose, yaw_rate, _ = self._pose(input_data)
        speed_value = self._payload(input_data, "SPEED")
        speed_mps = float(speed_value.get("speed", 0.0) if isinstance(speed_value, dict) else speed_value)
        ego_velocity = speed_mps * np.array([np.cos(pose[2]), np.sin(pose[2])], dtype=float)

        # Pseudo CARLA-world pose is sufficient because every sensor is transformed
        # consistently; x=east and CARLA y=-internal north.
        ego_to_world = _transform(float(pose[0]), -float(pose[1]), 0.0, -np.rad2deg(pose[2]))
        cameras = {}
        for channel in CAMERA_YAWS_DEG:
            image = np.asarray(self._payload(input_data, channel))
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(f"{channel}图像必须为HxWx3/4")
            # Leaderboard camera packets are BGRA/BGR, matching CARLA raw order.
            image_rgb = image[:, :, :3][:, :, ::-1].copy()
            local = _transform(1.0, 0.0, 2.15, CAMERA_YAWS_DEG[channel])
            sensor_to_world = ego_to_world @ local
            cameras[channel] = CameraSample(
                frame=source_frames.get(channel, frame),
                timestamp_s=float(timestamp_s),
                image_rgb=image_rgb,
                world_to_sensor=np.linalg.inv(sensor_to_world),
            )

        lidar_raw = np.asarray(self._payload(input_data, "LIDAR_TOP"), dtype=float)
        if lidar_raw.ndim != 2 or lidar_raw.shape[1] < 3:
            raise ValueError("LIDAR_TOP必须为Nx3/Nx4")
        lidar_to_world = ego_to_world @ _transform(0.0, 0.0, 2.25)
        points_world = transform_points(lidar_raw[:, :3], lidar_to_world)
        intensity = (
            lidar_raw[:, 3].copy() if lidar_raw.shape[1] > 3 else np.ones(len(lidar_raw))
        )
        lidar = LidarSample(
            source_frames.get("LIDAR_TOP", frame),
            float(timestamp_s), points_world, intensity, lidar_to_world
        )

        radars = {}
        radar_packet_diagnostics = {}
        radar_names = (
            LEADERBOARD_RADAR_YAWS_DEG
            if self.config.sensor_profile == "leaderboard"
            else FULL_RADAR_YAWS_DEG
        )
        for channel, yaw_deg in radar_names.items():
            raw = np.asarray(self._payload(input_data, channel), dtype=float)
            if raw.size == 0:
                raw = np.empty((0, 4), dtype=float)
            raw = raw.reshape(-1, 4)
            canonical, radar_layout = self._decode_radar_packet(raw)
            radar_packet_diagnostics[channel] = {
                "rows": int(len(raw)),
                "input_layout": radar_layout,
                "normalized_column_order": [
                    "velocity", "altitude", "azimuth", "depth"
                ],
                "column_median": (
                    None if not len(raw) else np.median(raw, axis=0).tolist()
                ),
                "column_min": (
                    None if not len(raw) else np.min(raw, axis=0).tolist()
                ),
                "column_max": (
                    None if not len(raw) else np.max(raw, axis=0).tolist()
                ),
            }
            radar_to_world = ego_to_world @ _transform(1.2, 0.0, 1.0, yaw_deg)
            origin_internal = carla_xyz_to_internal(radar_to_world[:3, 3])
            returns = []
            for velocity, altitude, azimuth, depth in canonical:
                cos_alt = np.cos(altitude)
                local = np.array([
                    depth * cos_alt * np.cos(azimuth),
                    depth * cos_alt * np.sin(azimuth),
                    depth * np.sin(altitude),
                ])
                point_world = transform_points(local[None, :], radar_to_world)[0]
                point_internal = carla_xyz_to_internal(point_world)
                ray = point_internal[:2] - origin_internal[:2]
                norm = float(np.linalg.norm(ray))
                if norm < 1e-6:
                    continue
                returns.append(RadarReturn(
                    position_global=point_internal,
                    sensor_origin_global=origin_internal,
                    radial_unit_global=ray / norm,
                    velocity_towards_mps=float(velocity),
                    sensor_name=channel,
                ))
            radars[channel] = RadarSample(
                source_frames.get(channel, frame), float(timestamp_s), tuple(returns)
            )

        sensor_frame = SynchronizedSensorFrame(
            frame=frame,
            timestamp_s=float(timestamp_s),
            cameras=cameras,
            lidar=lidar,
            radars=radars,
            collisions=(),
        )
        aligned = {
            name: source_frame
            for name, source_frame in sorted(source_frames.items())
            if name in required
        }
        metadata = {
            "reference_frame": frame,
            "source_frames": aligned,
            "source_frame_min": min(aligned.values(), default=frame),
            "source_frame_max": max(aligned.values(), default=frame),
            "source_frame_spread": (
                max(aligned.values()) - min(aligned.values()) if aligned else 0
            ),
            "misaligned_channels": [
                name for name, source_frame in aligned.items()
                if source_frame != frame
            ],
            "timestamp_s": float(timestamp_s),
            "timestamp_provenance": "leaderboard_shared_agent_timestamp",
            # Compact contract diagnostics make upstream format regressions
            # auditable without logging all Radar points.
            "radar_packet_diagnostics": radar_packet_diagnostics,
        }
        return AdaptedAgentFrame(
            sensor_frame, pose, ego_velocity, yaw_rate, metadata
        )
