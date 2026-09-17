"""Fusion Safety Agent 的可移植配置、默认值、路径解析与参数校验。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path


def _discover_product_root() -> Path:
    """Locate the external config/model bundle without hard-coding its drive."""

    override = os.environ.get("FUSION_SAFETY_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[1]
    working_root = Path.cwd().resolve()
    marker = Path("configs/product/fusion_safety_agent.json")
    if (working_root / marker).is_file():
        return working_root
    return source_root


PROJECT_ROOT = _discover_product_root()
PRODUCT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/bevfusion"
DEFAULT_CHECKPOINT = PRODUCT_ARTIFACT_ROOT / "fusion_safety_bevfusion_epoch_6.pth"
DEFAULT_CERTIFICATE = PRODUCT_ARTIFACT_ROOT / "active_runtime_certificate.json"


@dataclass(frozen=True)
class FusionSafetyAgentConfig:
    """传感器、感知、控制及评测共用的产品配置。

    ``leaderboard`` obeys the official four-Radar limit. ``product_full`` keeps
    the native five-Radar product contract for our extended local evaluator.
    """

    sensor_profile: str = "leaderboard"
    supervisor_mode: str = "shadow"
    frontend: str = "certified_bevfusion"
    target_speed_kmh: float = 30.0
    fixed_delta_s: float = 0.05
    # 20 Hz CARLA loop: 8 requests a 2.5 Hz BEV source cadence.  The
    # latest-only worker keeps control non-blocking when a frame runs long.
    detector_stride_frames: int = 8
    camera_width: int = 800
    camera_height: int = 450
    camera_fov_deg: float = 90.0
    lidar_range_m: float = 80.0
    radar_range_m: float = 70.0
    route_lookahead_m: float = 7.0
    steering_gain: float = 1.35
    speed_kp: float = 0.22
    max_throttle: float = 0.65
    max_route_brake: float = 0.35
    # Freshness gates used by the longitudinal safety supervisor.  The first
    # covers the detector stream; the second prevents another fresh target
    # from hiding one stale track.
    supervisor_max_fusion_age_s: float = 1.10
    supervisor_max_track_measurement_age_s: float = 1.10
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    certificate: str = str(DEFAULT_CERTIFICATE)
    # 必须与已签发的 BEVFusion Active 运行证书保持一致。
    inference_precision: str = "amp_fp16"
    wsl_distro: str = "Ubuntu-22.04"
    # ``~`` 由目标 WSL 在工作器启动时解析，不在配置中保存 Linux 用户名。
    wsl_runtime: str = os.environ.get(
        "FUSION_SAFETY_WSL_RUNTIME", "~/fusion_tracking_runtime/bevfusion"
    )
    # The certified transport publishes six encoded cameras and request
    # metadata through the resident backend's stdin. LiDAR and result files
    # remain on local NTFS and are accessed through /mnt/c, avoiding the less
    # reliable Windows view of the WSL filesystem.
    bevfusion_packet_transport: str = "wsl_stdin_packet"
    # Shared-GPU QoS: normal BEVFusion and VRU inference remains concurrent.
    # A true BEVFusion long-tail frame temporarily gives the primary 3D chain
    # exclusive priority for a few results, preventing repeated stale output
    # without permanently reducing auxiliary VRU throughput.
    gpu_arbiter_slow_threshold_s: float = 0.80
    gpu_arbiter_protection_results: int = 3
    startup_timeout_s: float = 240.0
    # The backend process is resident after setup, but its first real sensor
    # shape still triggers CUDA/MMCV kernel selection. Wait for that exact
    # six-camera + LiDAR result before releasing route control. Steady-state
    # inference remains latest-only and asynchronous after this one-time gate.
    first_perception_sync_enabled: bool = True
    first_perception_timeout_s: float = 30.0
    # Used only if the first input is unusable or synchronous inference fails.
    # Steering is preserved while longitudinal motion is held.
    first_perception_hold_brake: float = 0.30
    output_root: str = str(PROJECT_ROOT / "out/product")
    write_jsonl: bool = True
    # carla_garage's base Agent prints one timing line on every 20 Hz tick.
    # Product telemetry already records richer clock data, so suppressing that
    # synchronous console I/O improves runtime without changing any algorithm.
    quiet_leaderboard_timing: bool = True
    # Online sensor-health debouncing.  The monitor only inspects the public
    # sensor packet; evaluation fault labels never enter the algorithm.
    sensor_health_failure_frames: int = 3
    sensor_health_recovery_frames: int = 3
    sensor_health_min_lidar_points: int = 50
    sensor_health_max_frame_spread: int = 1
    # Detector-independent point-path AEB. Keep Shadow while commissioning a
    # new platform; promote to Active only after positive/negative regression.
    lidar_safety_cage_mode: str = "shadow"
    # Complementary fast VRU channel: one front camera is detected at a higher
    # rate than BEVFusion. Its calibrated image-ground estimate is checked and
    # weighted with real LiDAR/Radar evidence; no CARLA actor/scenario truth is
    # consumed. Shadow is the safe product default after positive/negative
    # regression; select Active explicitly before it may alter braking.
    vru_safety_monitor_mode: str = "shadow"
    vru_detector_model: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    vru_detector_score_threshold: float = 0.30
    vru_detector_stride_frames: int = 4
    vru_core_path_half_width_m: float = 1.8
    vru_monitor_half_width_m: float = 4.5
    vru_min_inward_speed_mps: float = 0.45
    # Fault profiles are evaluation inputs. Keep "none" for product acceptance.
    fault_profile: str = "none"
    fault_period_frames: int = 100
    fault_duration_frames: int = 10

    @classmethod
    def from_json(cls, path: str | Path | None) -> "FusionSafetyAgentConfig":
        if path is None or not str(path).strip():
            config = cls()
        else:
            source = Path(path).expanduser().resolve()
            payload = json.loads(source.read_text(encoding="utf-8"))
            known = {item.name for item in fields(cls)}
            unknown = sorted(set(payload) - known)
            if unknown:
                raise ValueError(f"Fusion Safety config contains unknown fields: {unknown}")
            # Product configurations are relocatable. Relative artifact and
            # output paths are resolved against the repository root instead
            # of the caller's current working directory.
            for key in ("checkpoint", "certificate", "output_root"):
                value = payload.get(key)
                if value and not Path(value).expanduser().is_absolute():
                    payload[key] = str((PROJECT_ROOT / value).resolve())
            wsl_override = os.environ.get("FUSION_SAFETY_WSL_RUNTIME", "").strip()
            if wsl_override:
                payload["wsl_runtime"] = wsl_override
            config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.sensor_profile not in {"leaderboard", "product_full"}:
            raise ValueError("sensor_profile must be leaderboard or product_full")
        if self.supervisor_mode not in {"shadow", "active"}:
            raise ValueError("supervisor_mode must be shadow or active")
        if self.frontend not in {"certified_bevfusion", "legacy"}:
            raise ValueError("frontend must be certified_bevfusion or legacy")
        if self.fault_profile not in {
            "none", "front_camera_dropout", "lidar_dropout",
            "radar_dropout", "mixed"
        }:
            raise ValueError("unknown fault_profile")
        if self.inference_precision not in {"fp32", "amp_fp16"}:
            raise ValueError("inference_precision must be fp32 or amp_fp16")
        if self.lidar_safety_cage_mode not in {"off", "shadow", "active"}:
            raise ValueError("lidar_safety_cage_mode must be off, shadow or active")
        if self.vru_safety_monitor_mode not in {"off", "shadow", "active"}:
            raise ValueError("vru_safety_monitor_mode must be off, shadow or active")
        if self.vru_detector_model not in {
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "ssdlite320_mobilenet_v3_large",
        }:
            raise ValueError("unsupported vru_detector_model")
        if not 0.0 < self.vru_detector_score_threshold <= 1.0:
            raise ValueError("vru_detector_score_threshold must be in (0,1]")
        if self.vru_detector_stride_frames < 1:
            raise ValueError("vru_detector_stride_frames must be >=1")
        if not (
            0.0 < self.vru_core_path_half_width_m
            < self.vru_monitor_half_width_m
        ):
            raise ValueError(
                "VRU path widths must satisfy 0 < core_path < monitor"
            )
        if self.vru_min_inward_speed_mps < 0:
            raise ValueError("vru_min_inward_speed_mps must be non-negative")
        if self.bevfusion_packet_transport not in {
            "windows_mount", "wsl_native", "wsl_inline_images",
            "wsl_stdin_packet",
        }:
            raise ValueError(
                "bevfusion_packet_transport must be windows_mount, wsl_native "
                "or wsl_inline_images/wsl_stdin_packet"
            )
        positive = (
            self.target_speed_kmh,
            self.fixed_delta_s,
            self.detector_stride_frames,
            self.camera_width,
            self.camera_height,
            self.camera_fov_deg,
            self.route_lookahead_m,
            self.startup_timeout_s,
            self.first_perception_timeout_s,
            self.supervisor_max_fusion_age_s,
            self.supervisor_max_track_measurement_age_s,
            self.gpu_arbiter_slow_threshold_s,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError(
                "rates, dimensions, speed, lookahead and timeout must be positive"
            )
        if not 0.0 < self.max_throttle <= 1.0:
            raise ValueError("max_throttle must be in (0, 1]")
        if not 0.0 <= self.max_route_brake <= 1.0:
            raise ValueError("max_route_brake must be in [0, 1]")
        if not isinstance(self.first_perception_sync_enabled, bool):
            raise ValueError("first_perception_sync_enabled must be boolean")
        if not 0.0 < self.first_perception_hold_brake <= 1.0:
            raise ValueError("first_perception_hold_brake must be in (0, 1]")
        if self.first_perception_timeout_s > self.startup_timeout_s:
            raise ValueError(
                "first_perception_timeout_s must not exceed startup_timeout_s"
            )
        if self.fault_period_frames < 1 or not (
            0 <= self.fault_duration_frames <= self.fault_period_frames
        ):
            raise ValueError(
                "fault period must be >=1 and duration must be within the period"
            )
        if self.sensor_health_failure_frames < 1:
            raise ValueError("sensor_health_failure_frames must be >=1")
        if self.sensor_health_recovery_frames < 1:
            raise ValueError("sensor_health_recovery_frames must be >=1")
        if self.sensor_health_min_lidar_points < 1:
            raise ValueError("sensor_health_min_lidar_points must be >=1")
        if self.sensor_health_max_frame_spread < 0:
            raise ValueError("sensor_health_max_frame_spread must be >=0")
        if self.gpu_arbiter_protection_results < 1:
            raise ValueError("gpu_arbiter_protection_results must be >=1")


# 保留旧类名只为读取历史实验代码；所有新集成统一使用产品名称。
Stage620AgentConfig = FusionSafetyAgentConfig
