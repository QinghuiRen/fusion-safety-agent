"""CARLA 传感器生成、帧对齐与数据解码。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import CAMERA_CHANNELS, SensorConfig
from .geometry import carla_xyz_to_internal, global_to_ego_xy, transform_points
from .types import RadarReturn


CAMERA_YAWS_DEG = {
    "CAM_FRONT": 0.0,
    "CAM_FRONT_RIGHT": 60.0,
    "CAM_BACK_RIGHT": 120.0,
    "CAM_BACK": 180.0,
    "CAM_BACK_LEFT": -120.0,
    "CAM_FRONT_LEFT": -60.0,
}


def _server_disconnected_during_spawn(exc: Exception) -> bool:
    """判断构造传感器时是否已失去 CARLA RPC 服务。

    CARLA 服务器崩溃后，继续对每个已创建 actor 调用 ``destroy`` 会各等待一次
    RPC timeout。这里仅识别 CARLA 客户端的断连/服务器超时措辞；普通首帧超时仍
    走完整清理，避免服务器健康时泄漏 actor。
    """

    message = str(exc).lower()
    return any(marker in message for marker in (
        "while waiting for the simulator",
        "connection refused",
        "connection failed",
        "actively refused",
        "目标计算机积极拒绝",
    ))

RADAR_YAWS_DEG = {
    "RADAR_FRONT": 0.0,
    "RADAR_FRONT_RIGHT": 65.0,
    "RADAR_BACK_RIGHT": 145.0,
    "RADAR_BACK_LEFT": -145.0,
    "RADAR_FRONT_LEFT": -65.0,
}


@dataclass(frozen=True)
class CameraSample:
    frame: int
    timestamp_s: float
    image_rgb: np.ndarray
    world_to_sensor: np.ndarray


@dataclass(frozen=True)
class LidarSample:
    frame: int
    timestamp_s: float
    points_world_carla: np.ndarray
    intensity: np.ndarray
    # CARLA LiDAR局部坐标到CARLA世界坐标的齐次变换。旧模块只需要世界点，
    # Stage6.6 BEVFusion还需恢复网络要求的LIDAR_TOP局部坐标。
    sensor_to_world: np.ndarray | None = None


@dataclass(frozen=True)
class RadarSample:
    frame: int
    timestamp_s: float
    returns: tuple[RadarReturn, ...]


@dataclass(frozen=True)
class CollisionEvent:
    frame: int
    timestamp_s: float
    other_actor_id: int
    other_actor: str
    impulse: float


@dataclass(frozen=True)
class SynchronizedSensorFrame:
    frame: int
    timestamp_s: float
    cameras: dict[str, CameraSample]
    lidar: LidarSample
    radars: dict[str, RadarSample]
    collisions: tuple[CollisionEvent, ...]


class _FrameBuffer:
    """仅保存各流最新样本；同步仿真下主线程不会制造无限积压。"""

    def __init__(self, required_names, max_stale_frames: dict[str, int] | None = None):
        self.required_names = tuple(required_names)
        self.max_stale_frames = dict(max_stale_frames or {})
        self.latest = {}
        self.errors = {}
        self.collisions: list[CollisionEvent] = []
        self.condition = threading.Condition()

    def put(self, name, sample) -> None:
        with self.condition:
            self.latest[name] = sample
            self.condition.notify_all()

    def put_collision(self, event: CollisionEvent) -> None:
        with self.condition:
            self.collisions.append(event)
            self.condition.notify_all()

    def put_error(self, name: str, exc: Exception) -> None:
        """保留回调异常；CARLA 的后台线程异常否则很容易只表现为超时。"""

        with self.condition:
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            self.condition.notify_all()

    def wait(self, frame: int, timeout_s: float) -> tuple[dict, tuple[CollisionEvent, ...]]:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                ready = all(
                    name in self.latest
                    and self.latest[name].frame
                    >= frame - self.max_stale_frames.get(name, 0)
                    for name in self.required_names
                )
                if ready:
                    samples = {name: self.latest[name] for name in self.required_names}
                    events = tuple(event for event in self.collisions if event.frame <= frame)
                    self.collisions = [event for event in self.collisions if event.frame > frame]
                    return samples, events
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [
                        name for name in self.required_names
                        if name not in self.latest
                        or self.latest[name].frame
                        < frame - self.max_stale_frames.get(name, 0)
                    ]
                    latest_frames = {
                        name: getattr(self.latest.get(name), "frame", None)
                        for name in self.required_names
                    }
                    callback_errors = {
                        name: self.errors[name] for name in missing if name in self.errors
                    }
                    raise TimeoutError(
                        f"等待 CARLA frame={frame} 超时；缺少={missing}；"
                        f"各流最新帧={latest_frames}；回调异常={callback_errors or '无'}"
                    )
                self.condition.wait(remaining)

    def wait_for_names(self, names, frame: int, timeout_s: float) -> dict:
        """等待指定子集到达目标帧；用于分批相机冷启动，不改变正式同步契约。"""

        names = tuple(names)
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                ready = all(
                    name in self.latest and self.latest[name].frame >= frame
                    for name in names
                )
                if ready:
                    return {name: self.latest[name] for name in names}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [
                        name for name in names
                        if name not in self.latest or self.latest[name].frame < frame
                    ]
                    latest_frames = {
                        name: getattr(self.latest.get(name), "frame", None)
                        for name in names
                    }
                    callback_errors = {
                        name: self.errors[name] for name in missing if name in self.errors
                    }
                    raise TimeoutError(
                        f"分批启动相机等待 CARLA frame={frame} 超时；缺少={missing}；"
                        f"各流最新帧={latest_frames}；回调异常={callback_errors or '无'}"
                    )
                self.condition.wait(remaining)


class CarlaSensorRig:
    """在 ego 上安装六相机、LiDAR、五 Radar 和碰撞传感器。"""

    def __init__(self, world, ego_vehicle, carla_module, config: SensorConfig):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.carla = carla_module
        self.config = config
        required = [*CAMERA_CHANNELS, "LIDAR_TOP", *RADAR_YAWS_DEG]
        camera_tick = config.camera_sensor_tick_s or config.sensor_tick_s
        stale_budget = (
            int(np.ceil(camera_tick / config.sensor_tick_s)) + 1
            if camera_tick > config.sensor_tick_s else 0
        )
        self.buffer = _FrameBuffer(
            required,
            max_stale_frames={name: stale_budget for name in CAMERA_CHANNELS},
        )
        self.actors = []
        self._callbacks = []  # 显式保留 Python 回调，避免底层只持有弱引用。
        try:
            self._spawn_all()
        except Exception as exc:
            # 构造函数失败时对象不会返回给调用方，必须在这里回收已经创建的子集。
            if _server_disconnected_during_spawn(exc):
                # 服务器进程已经退出，actor 随进程消失；此时只能释放本地引用，
                # 不能再逐个发 RPC，否则 N 个 actor 会造成 N×timeout 的假死。
                print(
                    f"[传感器清理] CARLA已断开，跳过远端actor销毁（本地引用={len(self.actors)}）",
                    flush=True,
                )
                self.actors.clear()
                self._callbacks.clear()
            else:
                self.destroy()
            raise

    def _spawn(self, name, blueprint, transform, callback):
        actor = self.world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)
        def guarded_callback(data):
            try:
                callback(data)
            except Exception as exc:
                self.buffer.put_error(name, exc)

        self._callbacks.append(guarded_callback)
        actor.listen(guarded_callback)
        self.actors.append(actor)
        return actor

    def _spawn_all(self) -> None:
        library = self.world.get_blueprint_library()
        cfg, carla = self.config, self.carla

        startup_order = cfg.camera_startup_order or tuple(CAMERA_YAWS_DEG)
        camera_items = [(name, CAMERA_YAWS_DEG[name]) for name in startup_order]
        if cfg.camera_startup_order:
            print(f"[相机启动顺序] {','.join(startup_order)}", flush=True)
        batch_size = int(cfg.camera_startup_batch_size)
        if batch_size <= 0:
            batch_size = len(camera_items)
        for offset in range(0, len(camera_items), batch_size):
            batch = camera_items[offset:offset + batch_size]
            for channel, yaw_deg in batch:
                blueprint = library.find("sensor.camera.rgb")
                blueprint.set_attribute("image_size_x", str(cfg.camera_width))
                blueprint.set_attribute("image_size_y", str(cfg.camera_height))
                blueprint.set_attribute("fov", str(cfg.camera_fov_deg))
                blueprint.set_attribute(
                    "sensor_tick", str(cfg.camera_sensor_tick_s or cfg.sensor_tick_s)
                )
                transform = carla.Transform(
                    carla.Location(x=1.0, z=2.15), carla.Rotation(yaw=yaw_deg)
                )
                self._spawn(
                    channel, blueprint, transform, self._camera_callback(channel)
                )
            if cfg.camera_startup_batch_size > 0:
                # 会话仍是唯一 tick owner；这里只在主运行循环开始前推进冷启动帧。
                # 等到图像回调真正抵达后才创建下一批，避免六路渲染资源同帧初始化。
                startup_frame = int(self.world.tick())
                self.buffer.wait_for_names(
                    [channel for channel, _ in batch],
                    startup_frame,
                    cfg.camera_startup_batch_timeout_s,
                )
                print(
                    f"[相机分批启动] {offset + 1}-{offset + len(batch)}/"
                    f"{len(camera_items)} 就绪 | frame={startup_frame} | "
                    f"channels={','.join(channel for channel, _ in batch)}",
                    flush=True,
                )

        lidar_bp = library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(cfg.lidar_channels))
        lidar_bp.set_attribute("range", str(cfg.lidar_range_m))
        lidar_bp.set_attribute("points_per_second", str(cfg.lidar_points_per_second))
        lidar_bp.set_attribute("rotation_frequency", str(cfg.lidar_rotation_hz))
        lidar_bp.set_attribute("sensor_tick", str(cfg.sensor_tick_s))
        self._spawn(
            "LIDAR_TOP",
            lidar_bp,
            carla.Transform(carla.Location(z=2.25)),
            self._lidar_callback,
        )

        for channel, yaw_deg in RADAR_YAWS_DEG.items():
            radar_bp = library.find("sensor.other.radar")
            radar_bp.set_attribute("range", str(cfg.radar_range_m))
            radar_bp.set_attribute("horizontal_fov", str(cfg.radar_horizontal_fov_deg))
            radar_bp.set_attribute("vertical_fov", str(cfg.radar_vertical_fov_deg))
            radar_bp.set_attribute("points_per_second", str(cfg.radar_points_per_second))
            radar_bp.set_attribute("sensor_tick", str(cfg.sensor_tick_s))
            transform = carla.Transform(
                carla.Location(x=1.2, z=1.0), carla.Rotation(yaw=yaw_deg)
            )
            self._spawn(channel, radar_bp, transform, self._radar_callback(channel))

        collision_bp = library.find("sensor.other.collision")
        self._spawn("COLLISION", collision_bp, carla.Transform(), self._collision_callback)

    def _camera_callback(self, channel):
        def callback(image):
            bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4
            )
            rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])
            world_to_sensor = np.asarray(image.transform.get_inverse_matrix(), dtype=float)
            self.buffer.put(channel, CameraSample(
                frame=image.frame,
                timestamp_s=float(image.timestamp),
                image_rgb=rgb,
                world_to_sensor=world_to_sensor,
            ))
        return callback

    def _lidar_callback(self, measurement):
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
        sensor_xyz = raw[:, :3].astype(float, copy=False)
        sensor_to_world = np.asarray(measurement.transform.get_matrix(), dtype=float)
        world_xyz = transform_points(sensor_xyz, sensor_to_world)
        self.buffer.put("LIDAR_TOP", LidarSample(
            frame=measurement.frame,
            timestamp_s=float(measurement.timestamp),
            points_world_carla=world_xyz,
            intensity=raw[:, 3].copy(),
            sensor_to_world=sensor_to_world,
        ))

    def _radar_callback(self, channel):
        def callback(measurement):
            sensor_to_world = np.asarray(measurement.transform.get_matrix(), dtype=float)
            origin_carla = np.asarray([
                measurement.transform.location.x,
                measurement.transform.location.y,
                measurement.transform.location.z,
            ], dtype=float)
            origin_internal = carla_xyz_to_internal(origin_carla)
            returns = []
            for detection in measurement:
                cos_alt = np.cos(detection.altitude)
                local = np.array([
                    detection.depth * cos_alt * np.cos(detection.azimuth),
                    detection.depth * cos_alt * np.sin(detection.azimuth),
                    detection.depth * np.sin(detection.altitude),
                ])
                point_world = transform_points(local[None, :], sensor_to_world)[0]
                point_internal = carla_xyz_to_internal(point_world)
                ray_xy = point_internal[:2] - origin_internal[:2]
                norm = float(np.linalg.norm(ray_xy))
                if norm < 1e-6:
                    continue
                returns.append(RadarReturn(
                    position_global=point_internal,
                    sensor_origin_global=origin_internal,
                    radial_unit_global=ray_xy / norm,
                    velocity_towards_mps=float(detection.velocity),
                    sensor_name=channel,
                ))
            self.buffer.put(channel, RadarSample(
                frame=measurement.frame,
                timestamp_s=float(measurement.timestamp),
                returns=tuple(returns),
            ))
        return callback

    def _collision_callback(self, event):
        impulse = event.normal_impulse
        magnitude = float(np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2))
        self.buffer.put_collision(CollisionEvent(
            frame=event.frame,
            timestamp_s=float(event.timestamp),
            other_actor_id=int(event.other_actor.id),
            other_actor=event.other_actor.type_id,
            impulse=magnitude,
        ))

    def wait_for_frame(
        self, frame: int, timeout_s: float | None = None
    ) -> SynchronizedSensorFrame:
        samples, collisions = self.buffer.wait(
            frame,
            self.config.max_sensor_wait_s if timeout_s is None else timeout_s,
        )
        return SynchronizedSensorFrame(
            frame=frame,
            timestamp_s=float(samples["LIDAR_TOP"].timestamp_s),
            cameras={name: samples[name] for name in CAMERA_CHANNELS},
            lidar=samples["LIDAR_TOP"],
            radars={name: samples[name] for name in RADAR_YAWS_DEG},
            collisions=collisions,
        )

    @staticmethod
    def lidar_in_current_ego(frame: SynchronizedSensorFrame, ego_pose_xyyaw) -> np.ndarray:
        world_internal = carla_xyz_to_internal(frame.lidar.points_world_carla)
        xy_ego = global_to_ego_xy(world_internal[:, :2], ego_pose_xyyaw)
        ego_z = world_internal[:, 2]  # caller subtracts current ego z if required.
        return np.column_stack([xy_ego, ego_z])

    def destroy(self) -> None:
        """停止回调并销毁本模块创建的传感器。"""

        for actor in self.actors:
            try:
                actor.stop()
            except (RuntimeError, AttributeError):
                pass
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except RuntimeError:
                pass
        self.actors.clear()
        self._callbacks.clear()
