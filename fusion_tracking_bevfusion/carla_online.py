"""CARLA同步帧到常驻WSL BEVFusion的在线桥接。

重型MMDetection3D依赖只存在于WSL；Windows CARLA运行时通过文件化数据包和
stdin控制消息通信。工作器保持容量1的latest-only队列，慢神经网络不会阻塞
20 Hz车辆控制、Radar修正和仪表盘刷新。
"""

from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from fusion_tracking_carla.config import PerceptionConfig, SensorConfig, TrackerConfig
from fusion_tracking_carla.geometry import (
    camera_intrinsic,
    internal_xyz_to_carla,
    project_carla_world_to_camera,
    transform_points,
)
from fusion_tracking_carla.perception import PerceptionJob, PerceptionResult

from .adapter import (
    BEVFusionAdapterConfig,
    DEFAULT_TRACK_CLASSES,
    detections_from_bevfusion,
)
from .schema import BEVFusionFrameResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/bevfusion/fusion_safety_bevfusion_epoch_6.pth"
)
OFFICIAL_CHECKPOINT_SHA256 = (
    "E68C6C084D8971F4F879751B0ED3981F575C24EE3CF385A1E39197C26A9BB3AB"
)
BEVFUSION_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)
_FLIP_Y_4 = np.diag([1.0, -1.0, 1.0, 1.0])
_CARLA_CAMERA_TO_CV = np.array(
    [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0],
     [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)


@dataclass(frozen=True)
class CarlaBEVFusionConfig:
    """CARLA 在线 BEVFusion 的可调运行参数。

    ``backend_score_threshold`` 控制WSL返回的候选池；``score_threshold``
    控制已有轨迹续接。建轨阈值由CARLA ``TrackerConfig`` 单独控制，默认0.25。
    """

    wsl_distro: str = "Ubuntu-22.04"
    # 产品配置可使用 ``~/...``，工作器启动时会向目标 WSL 查询真实 HOME。
    wsl_runtime: str = os.environ.get(
        "FUSION_SAFETY_WSL_RUNTIME",
        "/home/fusion/fusion_tracking_runtime/bevfusion",
    )
    checkpoint: Path = DEFAULT_CHECKPOINT
    # 后端在构建模型前核验；任何替换权重都必须显式传入实际哈希，
    # 不能通过关闭校验来绕过官方权重的完整性保护。
    expected_checkpoint_sha256: str = OFFICIAL_CHECKPOINT_SHA256
    packet_root: Path = Path("out/product/runtime_packets")
    backend_score_threshold: float = 0.10
    score_threshold: float = 0.20
    radar_association_gate_m: float = 3.2
    use_model_velocity: bool = True
    use_amp_fp16: bool = False
    vehicle_nms_iou_threshold: float = 0.15
    vru_nms_iou_threshold: float = 0.10
    startup_timeout_s: float = 180.0
    frame_timeout_s: float = 12.0
    # The first real tensor shape triggers CUDA/cuDNN/MMCV kernel selection.
    # On the shared Windows/WSL laptop GPU this has measured up to 13.6 s even
    # though steady inference is below 0.7 s.  Keep a separate bounded cold
    # deadline so a legitimate first result cannot poison the streaming IPC.
    first_frame_timeout_s: float = 30.0
    # 官方测试增强最终只保留704x256；桥接层提前完成确定性resize+crop并同步内参，
    # WSL不再为每帧重复解码1600x900图像后缩放。
    backend_image_width: int = 704
    backend_image_height: int = 256
    jpeg_quality: int = 92
    lidar_intensity_scale: float = 255.0
    packet_transport: str = "wsl_native"
    # Six independent camera resize/JPEG/write tasks release the GIL inside
    # OpenCV.  Parallel packet preparation hides UNC/open latency without
    # changing the pixels or the BEVFusion input contract.
    packet_writer_workers: int = 6

    def validate(self) -> None:
        if not 0 <= self.backend_score_threshold <= self.score_threshold <= 1:
            raise ValueError("需满足 0 <= backend_score_threshold <= score_threshold <= 1")
        if (
            self.frame_timeout_s <= 0
            or self.first_frame_timeout_s < self.frame_timeout_s
            or self.startup_timeout_s <= 0
        ):
            raise ValueError("WSL启动/单帧超时必须大于0")
        if not 0 <= self.vehicle_nms_iou_threshold <= 1:
            raise ValueError("vehicle_nms_iou_threshold必须位于[0, 1]")
        if not 0 <= self.vru_nms_iou_threshold <= 1:
            raise ValueError("vru_nms_iou_threshold必须位于[0, 1]")
        if self.packet_transport not in {
            "windows_mount", "wsl_native", "wsl_inline_images",
            "wsl_stdin_packet",
        }:
            raise ValueError(
                "packet_transport must be windows_mount, wsl_native or "
                "wsl_inline_images/wsl_stdin_packet"
            )
        if not 1 <= int(self.packet_writer_workers) <= len(BEVFUSION_CAMERA_ORDER):
            raise ValueError("packet_writer_workers must be within [1, 6]")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"BEVFusion权重不存在: {self.checkpoint}")
        expected = self.expected_checkpoint_sha256.strip().upper()
        if len(expected) != 64 or any(ch not in "0123456789ABCDEF" for ch in expected):
            raise ValueError("expected_checkpoint_sha256必须是64位十六进制SHA256")


def windows_to_wsl(path: Path) -> str:
    """把本机绝对路径转换为WSL的/mnt/<drive>路径。"""

    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"不是Windows绝对路径: {resolved}")
    relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def wsl_native_windows_path(distro: str, linux_path: str) -> Path:
    """Return the Windows UNC view of one absolute path in a WSL distro."""

    pure = PurePosixPath(linux_path)
    if not pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"WSL native packet path must be absolute: {linux_path}")
    # ``\\server`` alone is not a complete UNC anchor; include the distro
    # share before constructing ``Path`` so pathlib does not collapse a slash.
    root = Path(f"\\\\wsl.localhost\\{distro}")
    return root / Path(*pure.parts[1:])


def _packet_roots(
    config: "CarlaBEVFusionConfig", session: str
) -> tuple[Path, str]:
    """Resolve isolated host/backend IPC roots for one worker session.

    Native and inline-image file transports intentionally store their control
    files inside WSL.  ``wsl_stdin_packet`` already carries the large request
    through the persistent process pipe, so keeping a UNC dependency for its
    small LiDAR/result files only exposes long runs to transient
    ``\\wsl.localhost`` disconnects.  Store those files on local NTFS and let
    Linux access them through ``/mnt/<drive>`` instead.
    """

    runtime = config.wsl_runtime.rstrip("/")
    if config.packet_transport in {"wsl_native", "wsl_inline_images"}:
        backend = f"{runtime}/carla_packet_ipc/{session}"
        return wsl_native_windows_path(config.wsl_distro, backend), backend
    if config.packet_transport == "wsl_stdin_packet":
        windows = Path(config.packet_root).resolve() / session
        return windows, windows_to_wsl(windows)
    windows = Path(config.packet_root).resolve()
    return windows, windows_to_wsl(windows)


def _check_wsl_runtime(distro: str, timeout_s: float = 15.0) -> float:
    """Fail fast when WSL cannot start a trivial process.

    A wedged WSL service otherwise looks identical to a very slow BEVFusion
    model load and consumes the full Agent setup deadline.  This probe does
    not import CUDA or mutate the distro; it only verifies process launch and
    stdin/stdout plumbing before CARLA commits to model initialization.
    """

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [
                "wsl.exe", "-d", distro, "--",
                "/bin/sh", "-lc", "printf fusion-safety-wsl-ready",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=max(1.0, float(timeout_s)),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"WSL distro {distro} did not answer within {timeout_s:.0f}s; "
            "restart WSL before launching the BEVFusion Agent"
        ) from exc
    if completed.returncode != 0 or "fusion-safety-wsl-ready" not in completed.stdout:
        detail = (completed.stderr or completed.stdout or "no output").strip()
        raise RuntimeError(
            f"WSL distro {distro} preflight failed, code={completed.returncode}: "
            f"{detail}"
        )
    return time.perf_counter() - started


def resolve_wsl_runtime(
    distro: str,
    configured_path: str,
    *,
    timeout_s: float = 15.0,
) -> str:
    """把 ``~``/``$HOME`` 路径解析为 WSL 内部绝对路径。

    解析发生在运行时，发布配置无需保存 Linux 用户名。绝对路径保持原样。
    """

    value = str(configured_path).strip().rstrip("/")
    if not value:
        raise ValueError("wsl_runtime must not be empty")
    if value == "~" or value.startswith("~/"):
        suffix = value[1:]
    elif value == "$HOME" or value.startswith("$HOME/"):
        suffix = value[len("$HOME"):]
    else:
        if not value.startswith("/") or ".." in PurePosixPath(value).parts:
            raise ValueError(f"wsl_runtime must be an absolute or HOME path: {value}")
        return value

    completed = subprocess.run(
        [
            "wsl.exe", "-d", distro, "--",
            "/bin/sh", "-lc", 'printf "%s" "$HOME"',
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=max(1.0, float(timeout_s)),
    )
    home = (completed.stdout or "").strip().rstrip("/")
    if completed.returncode != 0 or not home.startswith("/"):
        detail = (completed.stderr or completed.stdout or "no output").strip()
        raise RuntimeError(
            f"cannot resolve HOME in WSL distro {distro}: {detail}"
        )
    resolved = f"{home}{suffix}"
    if ".." in PurePosixPath(resolved).parts:
        raise ValueError(f"resolved WSL runtime contains parent traversal: {resolved}")
    return resolved


def lidar_geometry(sensor_frame) -> tuple[np.ndarray, np.ndarray]:
    """返回网络LiDAR局部点和内部右手global_from_lidar。"""

    sensor_to_world = sensor_frame.lidar.sensor_to_world
    if sensor_to_world is None:
        raise ValueError("LidarSample 缺少 sensor_to_world，无法构造三维标定变换")
    sensor_to_world = np.asarray(sensor_to_world, dtype=float)
    world_to_sensor = np.linalg.inv(sensor_to_world)
    local_carla = transform_points(
        sensor_frame.lidar.points_world_carla, world_to_sensor
    )
    local_internal = local_carla.copy()
    local_internal[:, 1] *= -1.0
    global_from_lidar = _FLIP_Y_4 @ sensor_to_world @ _FLIP_Y_4
    return local_internal, global_from_lidar


def canonical_bevfusion_image(
    image_rgb: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """复现官方nuScenes测试增强：1600x900→0.48倍→中心/底部裁剪704x256。

    对任意16:9 CARLA输入直接缩放到768x432再裁剪，数学上等价于先上采样
    到1600x900；返回同步更新后的有效相机内参。
    """

    height, width = image_rgb.shape[:2]
    if width * 9 != height * 16:
        raise ValueError("BEVFusion在线相机图像必须保持16:9")
    resized_width, resized_height = 768, 432
    crop_x, crop_y = 32, 176
    resized = cv2.resize(
        image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    cropped = np.ascontiguousarray(
        resized[crop_y:crop_y + 256, crop_x:crop_x + 704]
    )
    augmentation = np.array([
        [resized_width / width, 0.0, -crop_x],
        [0.0, resized_height / height, -crop_y],
        [0.0, 0.0, 1.0],
    ])
    return cropped, augmentation @ np.asarray(intrinsic, dtype=float)


def _box_corners_lidar(box) -> np.ndarray:
    length, width, height = box.size_lwh_m
    signs = np.array(
        [[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)],
        dtype=float,
    )
    corners = signs * np.array([length, width, height], dtype=float) / 2.0
    c, s = np.cos(box.yaw_lidar_rad), np.sin(box.yaw_lidar_rad)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return corners @ rotation.T + np.asarray(box.center_lidar_m, dtype=float)


def project_boxes_to_cameras(
    result: BEVFusionFrameResult,
    global_from_lidar: np.ndarray,
    sensor_frame,
    sensor_config: SensorConfig,
    score_threshold: float,
) -> dict[str, tuple[dict, ...]]:
    """把神经3D框投回六相机，仅用于仪表盘显示，不回流到算法。"""

    output: dict[str, tuple[dict, ...]] = {}
    for channel, sample in sensor_frame.cameras.items():
        height, width = sample.image_rgb.shape[:2]
        intrinsic = camera_intrinsic(width, height, sensor_config.camera_fov_deg)
        overlays = []
        for box in result.detections:
            if box.score < score_threshold or box.class_name not in DEFAULT_TRACK_CLASSES:
                continue
            corners_global_internal = transform_points(
                _box_corners_lidar(box), global_from_lidar
            )
            corners_global_carla = internal_xyz_to_carla(corners_global_internal)
            uv, _, valid_depth = project_carla_world_to_camera(
                corners_global_carla, sample.world_to_sensor, intrinsic
            )
            visible = valid_depth & np.isfinite(uv).all(axis=1)
            if np.count_nonzero(visible) < 2:
                continue
            points = uv[visible]
            x1, y1 = np.min(points, axis=0)
            x2, y2 = np.max(points, axis=0)
            x1, x2 = np.clip([x1, x2], 0, width - 1)
            y1, y2 = np.clip([y1, y2], 0, height - 1)
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            overlays.append({
                "box2d": [float(x1), float(y1), float(x2), float(y2)],
                "class_name": box.class_name,
                "score": float(box.score),
            })
        output[channel] = tuple(overlays)
    return output


class LatestOnlyBEVFusionWorker:
    """与LatestOnlyPerceptionWorker同接口的常驻BEVFusion工作器。"""

    def __init__(
        self,
        perception: PerceptionConfig,
        sensors: SensorConfig,
        tracker: TrackerConfig,
        config: CarlaBEVFusionConfig | None = None,
        *,
        inference_arbiter=None,
    ):
        self.perception_config = perception
        self.sensor_config = sensors
        self.tracker_config = tracker
        self.config = config or CarlaBEVFusionConfig()
        self.config.validate()
        # 仅在工作器真正启动时解析 WSL 用户目录，避免把用户名写入配置和发布包。
        self.config = replace(
            self.config,
            wsl_runtime=resolve_wsl_runtime(
                self.config.wsl_distro,
                self.config.wsl_runtime,
                timeout_s=min(15.0, self.config.startup_timeout_s),
            ),
        )
        if inference_arbiter is None:
            from fusion_tracking_carla.inference_arbiter import (
                GLOBAL_GPU_INFERENCE_ARBITER,
            )

            inference_arbiter = GLOBAL_GPU_INFERENCE_ARBITER
        self.inference_arbiter = inference_arbiter
        if self.config.packet_transport in {
            "wsl_native", "wsl_inline_images", "wsl_stdin_packet"
        }:
            session = (
                f"{Path(self.config.packet_root).parent.name}_"
                f"{os.getpid()}_{time.time_ns()}"
            )
            (
                self._packet_root_windows,
                self._packet_root_backend,
            ) = _packet_roots(
                self.config, session
            )
        else:
            self._packet_root_windows = Path(self.config.packet_root).resolve()
            self._packet_root_backend = windows_to_wsl(self._packet_root_windows)
        self.jobs: queue.Queue[PerceptionJob | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: PerceptionResult | None = None
        self._ready = threading.Event()
        self._initialization_error: str | None = None
        self.startup_diagnostics: dict[str, float | str | bool] = {}
        self._process: subprocess.Popen | None = None
        self._successful_backend_frames = 0
        self._packet_executor = ThreadPoolExecutor(
            max_workers=int(self.config.packet_writer_workers),
            thread_name_prefix="bevfusion_packet",
        )
        self._thread = threading.Thread(
            target=self._run, name="fusion_safety_bevfusion", daemon=True
        )
        self._thread.start()

    def _backend_packet_path(self, windows_path: Path) -> str:
        relative = windows_path.relative_to(self._packet_root_windows).as_posix()
        return f"{self._packet_root_backend.rstrip('/')}/{relative}"

    def wait_until_ready(self, timeout_s: float) -> None:
        timeout = min(float(timeout_s), self.config.startup_timeout_s)
        if not self._ready.wait(timeout):
            raise TimeoutError(f"BEVFusion初始化超过{timeout:.0f}s")
        if self._initialization_error is not None:
            raise RuntimeError(self._initialization_error)

    def submit(self, job: PerceptionJob) -> None:
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            self.jobs.put_nowait(job)

    def latest(self) -> PerceptionResult | None:
        with self._lock:
            return self._latest

    def warmup_with_frame(
        self,
        job: PerceptionJob,
        timeout_s: float,
        *,
        retain_result: bool = False,
    ) -> float:
        """在正式计时前跑一帧真实标定输入，消化CUDA kernel冷启动。"""

        started = time.perf_counter()
        self.submit(job)
        deadline = started + min(
            float(timeout_s), self.config.first_frame_timeout_s
        )
        while time.perf_counter() < deadline:
            result = self.latest()
            if result is not None and result.frame == job.sensor_frame.frame:
                if result.error is not None:
                    raise RuntimeError(result.error)
                if not retain_result:
                    with self._lock:
                        # Preserve a hypothetically newer publication if it
                        # arrived between latest() and lock acquisition.
                        if self._latest is result:
                            self._latest = None
                elapsed = time.perf_counter() - started
                self.startup_diagnostics["first_frame_warmup_s"] = elapsed
                self.startup_diagnostics["first_frame_result_retained"] = bool(
                    retain_result
                )
                return elapsed
            time.sleep(0.02)
        raise TimeoutError(
            f"BEVFusion真实帧预热超过{self.config.first_frame_timeout_s:.0f}s"
        )

    def _start_backend(self) -> None:
        cfg = self.config
        wsl_preflight_s = _check_wsl_runtime(
            cfg.wsl_distro, timeout_s=min(15.0, cfg.startup_timeout_s)
        )
        packet_root = self._packet_root_windows
        packet_root.mkdir(parents=True, exist_ok=True)
        ready = packet_root / "backend_ready.json"
        ready.unlink(missing_ok=True)
        project_wsl = windows_to_wsl(PROJECT_ROOT)
        checkpoint_wsl = windows_to_wsl(cfg.checkpoint)
        runtime = cfg.wsl_runtime.rstrip("/")
        repo_wsl = f"{runtime}/mmdetection3d"
        command = [
            "wsl.exe", "-d", cfg.wsl_distro, "--",
            f"{runtime}/.venv/bin/python",
            f"{project_wsl}/fusion_tracking_bevfusion/backend_mmdet3d.py",
            "serve-carla", "--repo-root", repo_wsl,
            "--config",
            f"{repo_wsl}/projects/BEVFusion/configs/"
            "bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py",
            "--checkpoint", checkpoint_wsl,
            "--expected-checkpoint-sha256", cfg.expected_checkpoint_sha256,
            "--ready-file", self._backend_packet_path(ready),
            "--score-threshold", str(cfg.backend_score_threshold),
        ]
        if cfg.use_amp_fp16:
            command.append("--amp-fp16")
        if cfg.packet_transport in {"wsl_inline_images", "wsl_stdin_packet"}:
            command.append("--inline-images")
        started = time.perf_counter()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        deadline = started + cfg.startup_timeout_s
        while time.perf_counter() < deadline:
            if ready.is_file():
                try:
                    metadata = json.loads(ready.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    # WSL/NTFS metadata publication can expose an empty file for
                    # one polling interval. Treat it as not-ready, not fatal.
                    time.sleep(0.05)
                    continue
                self.startup_diagnostics = {
                    "backend": "mmdetection3d.projects.BEVFusion",
                    "backend_persistent": True,
                    "backend_startup_wall_s": time.perf_counter() - started,
                    "wsl_preflight_s": wsl_preflight_s,
                    "backend_model_load_s": metadata.get("model_load_latency_s"),
                    "detector_warmup_inference_s": 0.0,
                    "backend_score_threshold": cfg.backend_score_threshold,
                    "packet_transport": cfg.packet_transport,
                    "packet_root_backend": self._packet_root_backend,
                    "effect_score_threshold": cfg.score_threshold,
                    "inference_precision_requested": (
                        "amp_fp16" if cfg.use_amp_fp16 else "fp32"
                    ),
                }
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"BEVFusion WSL服务提前退出，code={self._process.returncode}"
                )
            time.sleep(0.1)
        raise TimeoutError(f"BEVFusion WSL服务启动超过{cfg.startup_timeout_s:.0f}s")

    def _prepare_camera_packet(
        self,
        channel: str,
        sample,
        sensor_to_world: np.ndarray,
        slot_dir: Path,
    ) -> tuple[str, dict, dict[str, float]]:
        """Prepare one camera file and calibration without changing its data."""

        cfg = self.config
        native_height, native_width = sample.image_rgb.shape[:2]
        native_intrinsic = camera_intrinsic(
            native_width, native_height, self.sensor_config.camera_fov_deg
        )
        resize_started = time.perf_counter()
        backend_image, intrinsic = canonical_bevfusion_image(
            sample.image_rgb, native_intrinsic
        )
        resize_latency = time.perf_counter() - resize_started
        image_path = slot_dir / f"{channel}.jpg"
        encode_started = time.perf_counter()
        success, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(backend_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
        )
        encode_latency = time.perf_counter() - encode_started
        if not success:
            raise RuntimeError(f"OpenCV failed to encode {channel}")
        encoded_bytes = encoded.tobytes()
        write_latency = 0.0
        lidar2cam = (
            _CARLA_CAMERA_TO_CV
            @ np.asarray(sample.world_to_sensor, dtype=float)
            @ sensor_to_world
            @ _FLIP_Y_4
        )
        camera_payload = {
            "cam2img": intrinsic.tolist(),
            "lidar2cam": lidar2cam.tolist(),
        }
        if cfg.packet_transport in {"wsl_inline_images", "wsl_stdin_packet"}:
            # Preserve the exact JPEG byte stream used by the file transport,
            # but publish all six images inside one atomic request.  This
            # removes six high-latency Windows->WSL UNC file operations.
            camera_payload["img_bytes_b64"] = base64.b64encode(
                encoded_bytes
            ).decode("ascii")
        else:
            write_started = time.perf_counter()
            image_path.write_bytes(encoded_bytes)
            write_latency = time.perf_counter() - write_started
            camera_payload["img_path"] = self._backend_packet_path(image_path)
        return (
            channel,
            camera_payload,
            {
                "resize_s": resize_latency,
                "jpeg_encode_s": encode_latency,
                "file_write_s": write_latency,
            },
        )

    def _write_packet(
        self, job: PerceptionJob, slot: int
    ) -> tuple[Path, Path, dict, np.ndarray, np.ndarray, dict[str, float]]:
        packet_started = time.perf_counter()
        cfg = self.config
        slot_setup_started = time.perf_counter()
        slot_dir = self._packet_root_windows / f"slot_{slot}"
        slot_dir.mkdir(parents=True, exist_ok=True)
        slot_setup_latency = time.perf_counter() - slot_setup_started
        lidar_geometry_started = time.perf_counter()
        lidar_points, global_from_lidar = lidar_geometry(job.sensor_frame)
        intensity = np.asarray(job.sensor_frame.lidar.intensity, dtype=float)
        points5 = np.column_stack([
            lidar_points[:, :3],
            np.clip(intensity * cfg.lidar_intensity_scale, 0.0, 255.0),
            np.zeros(len(lidar_points)),
        ]).astype(np.float32)
        lidar_geometry_latency = time.perf_counter() - lidar_geometry_started
        lidar_path = slot_dir / "lidar.bin"
        lidar_write_started = time.perf_counter()
        points5.tofile(lidar_path)
        lidar_write_latency = time.perf_counter() - lidar_write_started

        sensor_to_world = np.asarray(job.sensor_frame.lidar.sensor_to_world, dtype=float)
        camera_batch_started = time.perf_counter()
        futures = [
            self._packet_executor.submit(
                self._prepare_camera_packet,
                channel,
                job.sensor_frame.cameras[channel],
                sensor_to_world,
                slot_dir,
            )
            for channel in BEVFUSION_CAMERA_ORDER
        ]
        camera_results = [future.result() for future in futures]
        camera_batch_latency = time.perf_counter() - camera_batch_started
        camera_by_channel = {channel: item for channel, item, _ in camera_results}
        # Preserve the official BEVFusion camera order even though the writes
        # completed concurrently.
        images = {
            channel: camera_by_channel[channel]
            for channel in BEVFUSION_CAMERA_ORDER
        }
        packet_stages = {
            "slot_setup_s": slot_setup_latency,
            "lidar_geometry_s": lidar_geometry_latency,
            "lidar_file_write_s": lidar_write_latency,
            "camera_parallel_wall_s": camera_batch_latency,
            "camera_resize_cpu_sum_s": sum(
                timing["resize_s"] for _, _, timing in camera_results
            ),
            "camera_jpeg_encode_cpu_sum_s": sum(
                timing["jpeg_encode_s"] for _, _, timing in camera_results
            ),
            "camera_file_write_wait_sum_s": sum(
                timing["file_write_s"] for _, _, timing in camera_results
            ),
        }

        token = f"carla_{job.sensor_frame.frame:09d}"
        request = {
            "schema_version": "fusion_tracking.bevfusion.carla.request.v1",
            "sample_token": token,
            "timestamp_s": float(job.sensor_frame.timestamp_s),
            "global_from_lidar_internal": global_from_lidar.tolist(),
            "data_info": {
                "sample_idx": token,
                "token": token,
                "timestamp": int(round(job.sensor_frame.timestamp_s * 1e6)),
                "ego2global": global_from_lidar.tolist(),
                "images": images,
                "lidar_points": {
                    "num_pts_feats": 5,
                    "lidar_path": self._backend_packet_path(lidar_path),
                    "lidar2ego": np.eye(4).tolist(),
                },
            },
        }
        # Per-frame control files prevent a late result from a timed-out CUDA
        # call being mistaken for the next request.  The heavy LiDAR/image
        # payload still uses two rotating slots to avoid unbounded disk I/O.
        token_suffix = f"{int(job.sensor_frame.frame):09d}"
        request_path = slot_dir / f"request_{token_suffix}.json"
        result_path = slot_dir / f"result_{token_suffix}.json"
        result_cleanup_started = time.perf_counter()
        result_path.unlink(missing_ok=True)
        result_cleanup_latency = time.perf_counter() - result_cleanup_started
        request_temporary = request_path.with_suffix(request_path.suffix + ".tmp")
        request_serialize_latency = 0.0
        request_write_latency = 0.0
        if cfg.packet_transport != "wsl_stdin_packet":
            request_serialize_started = time.perf_counter()
            request_payload = json.dumps(request, ensure_ascii=False)
            request_serialize_latency = (
                time.perf_counter() - request_serialize_started
            )
            request_write_started = time.perf_counter()
            request_temporary.write_text(request_payload, encoding="utf-8")
            request_temporary.replace(request_path)
            request_write_latency = time.perf_counter() - request_write_started
        packet_stages.update({
            "result_cleanup_s": result_cleanup_latency,
            "request_json_serialize_s": request_serialize_latency,
            "request_json_publish_s": request_write_latency,
            "host_packet_total_s": time.perf_counter() - packet_started,
        })
        return (
            request_path,
            result_path,
            request,
            lidar_points,
            global_from_lidar,
            packet_stages,
        )

    def _process_job(self, job: PerceptionJob, slot: int) -> PerceptionResult:
        started = time.perf_counter()
        (
            request_path,
            result_path,
            request,
            lidar_points,
            global_from_lidar,
            packet_stage_latencies,
        ) = self._write_packet(job, slot)
        preprocess_finished = time.perf_counter()
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("BEVFusion后端进程未启动")
        command = {"output_path": self._backend_packet_path(result_path)}
        if self.config.packet_transport == "wsl_stdin_packet":
            # The backend process is already persistent and waiting on stdin.
            # Carry the atomic JSON request over that pipe instead of writing
            # and reopening a large base64 control file through WSL UNC.  The
            # six JPEG byte streams and LiDAR tensor are unchanged.
            command["request"] = request
        else:
            command["request_path"] = self._backend_packet_path(request_path)
        command_serialize_started = time.perf_counter()
        command_payload = json.dumps(command, ensure_ascii=False)
        packet_stage_latencies["backend_command_serialize_s"] = (
            time.perf_counter() - command_serialize_started
        )
        arbitration_wait_started = time.perf_counter()
        with self.inference_arbiter.lease("high"):
            arbitration_acquired = time.perf_counter()
            backend_signal_started = arbitration_acquired
            self._process.stdin.write(command_payload + "\n")
            self._process.stdin.flush()
            packet_stage_latencies["backend_stdin_signal_s"] = (
                time.perf_counter() - backend_signal_started
            )
            timeout_s = (
                self.config.first_frame_timeout_s
                if self._successful_backend_frames == 0
                else self.config.frame_timeout_s
            )
            deadline = time.perf_counter() + timeout_s
            while not result_path.is_file():
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"BEVFusion后端退出，code={self._process.returncode}"
                    )
                if time.perf_counter() >= deadline:
                    raise TimeoutError(
                        f"BEVFusion帧{job.sensor_frame.frame}推理超过{timeout_s:.0f}s"
                    )
                time.sleep(0.01)
        packet_stage_latencies["gpu_arbiter_wait_s"] = (
            arbitration_acquired - arbitration_wait_started
        )
        packet_stage_latencies["gpu_arbiter_hold_s"] = (
            time.perf_counter() - arbitration_acquired
        )
        backend_finished = time.perf_counter()
        result_read_started = time.perf_counter()
        result = BEVFusionFrameResult.load(result_path)
        # The first real shape is a deliberate CUDA warm-up and must not arm
        # runtime protection.  Later long-tail model time temporarily
        # serialises this primary chain against the auxiliary VRU detector.
        if self._successful_backend_frames > 0:
            self.inference_arbiter.observe_primary_latency(
                float(result.inference_latency_s)
            )
        packet_stage_latencies["result_json_read_parse_s"] = (
            time.perf_counter() - result_read_started
        )
        if result.sample_token != f"carla_{job.sensor_frame.frame:09d}":
            raise ValueError("BEVFusion返回token与请求帧不一致")
        self._successful_backend_frames += 1
        radar_returns = [
            item for sample in job.sensor_frame.radars.values() for item in sample.returns
        ]
        adapter_config = BEVFusionAdapterConfig(
            score_threshold=self.config.score_threshold,
            radar_association_gate_m=self.config.radar_association_gate_m,
            use_model_velocity=self.config.use_model_velocity,
            vehicle_nms_iou_threshold=self.config.vehicle_nms_iou_threshold,
            vru_nms_iou_threshold=self.config.vru_nms_iou_threshold,
        )
        adapter_started = time.perf_counter()
        detections = detections_from_bevfusion(
            result,
            global_from_lidar,
            radar_returns=radar_returns,
            lidar_points_lidar=lidar_points,
            config=adapter_config,
        )
        packet_stage_latencies["detection_adapter_s"] = (
            time.perf_counter() - adapter_started
        )
        raw_trackable = sum(
            box.class_name in DEFAULT_TRACK_CLASSES for box in result.detections
        )
        class_counts: dict[str, int] = {}
        for box in result.detections:
            class_counts[box.class_name] = class_counts.get(box.class_name, 0) + 1
        print(
            f"[BEVFusion适配] frame={job.sensor_frame.frame} | "
            f"trackable_raw={raw_trackable} | accepted={len(detections)} | "
            f"classes={class_counts}",
            flush=True,
        )
        projection_started = time.perf_counter()
        camera_boxes = project_boxes_to_cameras(
            result, global_from_lidar, job.sensor_frame, self.sensor_config,
            self.config.score_threshold,
        )
        packet_stage_latencies["camera_projection_s"] = (
            time.perf_counter() - projection_started
        )
        finished = time.perf_counter()
        return PerceptionResult(
            frame=job.sensor_frame.frame,
            timestamp_s=job.sensor_frame.timestamp_s,
            detections=tuple(detections),
            camera_boxes=camera_boxes,
            latency_s=finished - started,
            ego_velocity_global=job.ego_velocity_global.copy(),
            queue_latency_s=max(started - job.submitted_wall_s, 0.0),
            preprocess_latency_s=preprocess_finished - started,
            detector_latency_s=backend_finished - preprocess_finished,
            fusion_latency_s=finished - backend_finished,
            diagnostics={
                "backend_raw_detections": len(result.detections),
                "backend_trackable_raw": raw_trackable,
                "accepted_detections": len(detections),
                "class_counts": class_counts,
                # The worker wall-clock stages above include filesystem/WSL
                # bridge time. These backend fields isolate model work.
                "backend_preprocess_latency_s": result.backend_preprocess_latency_s,
                "backend_inference_latency_s": result.inference_latency_s,
                "backend_inference_precision": result.inference_precision,
                "backend_amp_fallback_error": result.amp_fallback_error,
                "backend_stage_latencies_s": result.backend_stage_latencies_s,
                "host_packet_stage_latencies_s": packet_stage_latencies,
            },
        )

    def _run(self) -> None:
        try:
            self._start_backend()
        except Exception as exc:
            self._initialization_error = (
                f"BEVFusion加载失败: {type(exc).__name__}: {exc}"
            )
            self._ready.set()
            return
        self._ready.set()
        slot = 0
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                result = self._process_job(job, slot)
            except Exception as exc:
                result = PerceptionResult(
                    job.sensor_frame.frame, job.sensor_frame.timestamp_s,
                    (), {}, 0.0, job.ego_velocity_global.copy(),
                    error=f"BEVFusion帧失败: {type(exc).__name__}: {exc}",
                )
            finally:
                # Never overwrite a payload that a late backend call may still
                # be reading after an operational timeout.
                slot = 1 - slot
            with self._lock:
                self._latest = result

    def close(self) -> None:
        try:
            self.jobs.put_nowait(None)
        except queue.Full:
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                pass
            self.jobs.put_nowait(None)
        self._thread.join(timeout=max(
            self.config.frame_timeout_s,
            self.config.first_frame_timeout_s,
        ) + 2.0)
        process = self._process
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                try:
                    process.stdin.write('{"command":"shutdown"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.terminate()
        if self.config.packet_transport in {
            "wsl_native", "wsl_inline_images", "wsl_stdin_packet"
        }:
            if self.config.packet_transport == "wsl_stdin_packet":
                expected_parent = Path(self.config.packet_root).resolve()
            else:
                expected_parent = wsl_native_windows_path(
                    self.config.wsl_distro,
                    f"{self.config.wsl_runtime.rstrip('/')}/carla_packet_ipc",
                )
            try:
                # Safety guard: only the unique session directory created by
                # this worker may be recursively removed.
                self._packet_root_windows.relative_to(expected_parent)
                if self._packet_root_windows.exists():
                    shutil.rmtree(self._packet_root_windows)
            except (OSError, ValueError):
                # Transient diagnostics must never affect control shutdown.
                pass
        self._packet_executor.shutdown(wait=False, cancel_futures=True)
