"""产品 Agent 的在线传感器健康状态机。

监视器只使用已适配的公开传感器数据及帧来源，不接收故障注入标签或 CARLA actor
真值。因此同一健康判断接口可以迁移到真实传感器输入。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class SensorHealthConfig:
    """Tunable health thresholds expressed in 20 Hz control frames."""

    failure_frames: int = 3
    recovery_frames: int = 3
    min_lidar_points: int = 50
    max_lidar_invalid_fraction: float = 0.05
    max_source_frame_spread: int = 1
    min_usable_cameras: int = 3
    camera_black_peak: float = 1.0
    camera_black_nonzero_fraction: float = 0.001

    def validate(self) -> None:
        if self.failure_frames < 1 or self.recovery_frames < 1:
            raise ValueError("sensor health debounce frames must be >=1")
        if self.min_lidar_points < 1:
            raise ValueError("min_lidar_points must be >=1")
        if not 0.0 <= self.max_lidar_invalid_fraction < 1.0:
            raise ValueError("max_lidar_invalid_fraction must be in [0,1)")
        if self.max_source_frame_spread < 0:
            raise ValueError("max_source_frame_spread must be >=0")
        if self.min_usable_cameras < 1:
            raise ValueError("min_usable_cameras must be >=1")


@dataclass
class _DebouncedComponent:
    name: str
    failure_frames: int
    recovery_frames: int
    state: str = "healthy"
    bad_streak: int = 0
    good_streak: int = 0
    failed_latched: bool = False
    transitions: dict[str, int] = field(default_factory=dict)

    def update(self, raw_ok: bool) -> str:
        previous = self.state
        if raw_ok:
            self.bad_streak = 0
            if self.failed_latched:
                self.good_streak += 1
                if self.good_streak >= self.recovery_frames:
                    self.failed_latched = False
                    self.good_streak = 0
                    self.state = "healthy"
                else:
                    self.state = "recovering"
            else:
                self.good_streak = 0
                self.state = "healthy"
        else:
            self.good_streak = 0
            self.bad_streak += 1
            if self.failed_latched or self.bad_streak >= self.failure_frames:
                self.failed_latched = True
                self.state = "failed"
            else:
                self.state = "suspect"
        if self.state != previous:
            key = f"{previous}->{self.state}"
            self.transitions[key] = self.transitions.get(key, 0) + 1
        return self.state


@dataclass(frozen=True)
class SensorHealthSnapshot:
    frame: int
    timestamp_s: float
    overall_state: str
    quality: float
    active_control_permitted: bool
    control_inhibit_reasons: tuple[str, ...]
    perception_input_usable: bool
    components: dict
    raw: dict

    def to_dict(self) -> dict:
        result = asdict(self)
        result["control_inhibit_reasons"] = list(self.control_inhibit_reasons)
        return result


class OnlineSensorHealthMonitor:
    """Debounce sensor faults and expose a fail-silent control gate.

    A single front-camera or all-Radar loss is a degraded condition because
    LiDAR and the remaining cameras still provide obstacle evidence.  LiDAR
    loss, loss of most cameras, or severe source-frame misalignment is treated
    as unsafe after debouncing and prevents new Active control application.
    """

    _WEIGHTS = {
        "camera_front": 0.10,
        "camera_array": 0.25,
        "lidar": 0.40,
        "radar": 0.15,
        "alignment": 0.10,
    }
    _STATE_SCORE = {
        "healthy": 1.0,
        "suspect": 0.65,
        "recovering": 0.75,
        "failed": 0.0,
    }
    _CRITICAL = ("camera_array", "lidar", "alignment")

    def __init__(self, config: SensorHealthConfig):
        config.validate()
        self.config = config
        self._components = {
            name: _DebouncedComponent(
                name, config.failure_frames, config.recovery_frames
            )
            for name in self._WEIGHTS
        }
        self._frames = 0
        self._overall_frames: dict[str, int] = {}
        self._inhibited_frames = 0
        self._unusable_perception_frames = 0
        self._raw_bad_frames = {name: 0 for name in self._WEIGHTS}

    def _camera_metrics(self, sensor_frame) -> tuple[dict, int]:
        metrics = {}
        usable = 0
        for name, camera in sorted(sensor_frame.cameras.items()):
            image = np.asarray(camera.image_rgb)
            valid_shape = bool(image.ndim == 3 and image.shape[2] >= 3 and image.size)
            if valid_shape:
                # Sampling keeps this 20 Hz monitor cheap while exact injected
                # blackouts remain unambiguous.
                sample = np.asarray(image[::16, ::16, :3], dtype=float)
                finite = np.isfinite(sample)
                finite_fraction = float(np.mean(finite)) if sample.size else 0.0
                values = sample[finite]
                peak = float(np.max(values)) if values.size else 0.0
                nonzero_fraction = (
                    float(np.mean(np.abs(values) > self.config.camera_black_peak))
                    if values.size else 0.0
                )
            else:
                finite_fraction = 0.0
                peak = 0.0
                nonzero_fraction = 0.0
            blackout = bool(
                not valid_shape
                or finite_fraction < 0.99
                or (
                    peak <= self.config.camera_black_peak
                    and nonzero_fraction <= self.config.camera_black_nonzero_fraction
                )
            )
            channel_ok = not blackout
            usable += int(channel_ok)
            metrics[name] = {
                "ok": channel_ok,
                "shape": list(image.shape),
                "sample_finite_fraction": finite_fraction,
                "sample_peak": peak,
                "sample_nonzero_fraction": nonzero_fraction,
                "blackout": blackout,
            }
        return metrics, usable

    def evaluate(self, adapted) -> SensorHealthSnapshot:
        sensor_frame = adapted.sensor_frame
        camera_metrics, usable_cameras = self._camera_metrics(sensor_frame)
        front_ok = bool(camera_metrics.get("CAM_FRONT", {}).get("ok", False))
        camera_array_ok = usable_cameras >= min(
            self.config.min_usable_cameras, max(len(camera_metrics), 1)
        )

        points = np.asarray(sensor_frame.lidar.points_world_carla)
        point_count = int(len(points)) if points.ndim == 2 else 0
        if point_count and points.shape[1] >= 3:
            finite_rows = np.all(np.isfinite(points[:, :3]), axis=1)
            lidar_invalid_fraction = 1.0 - float(np.mean(finite_rows))
        else:
            lidar_invalid_fraction = 1.0
        lidar_ok = bool(
            point_count >= self.config.min_lidar_points
            and lidar_invalid_fraction <= self.config.max_lidar_invalid_fraction
        )

        radar_counts = {
            name: len(sample.returns)
            for name, sample in sorted(sensor_frame.radars.items())
        }
        radar_total = int(sum(radar_counts.values()))
        radar_ok = radar_total > 0

        metadata = dict(adapted.source_metadata)
        source_spread = int(metadata.get("source_frame_spread", 0))
        alignment_ok = source_spread <= self.config.max_source_frame_spread
        raw_ok = {
            "camera_front": front_ok,
            "camera_array": camera_array_ok,
            "lidar": lidar_ok,
            "radar": radar_ok,
            "alignment": alignment_ok,
        }
        states = {
            name: self._components[name].update(ok)
            for name, ok in raw_ok.items()
        }
        for name, ok in raw_ok.items():
            self._raw_bad_frames[name] += int(not ok)

        inhibit = tuple(
            f"sensor_health_{name}_{states[name]}"
            for name in self._CRITICAL
            if states[name] in {"failed", "recovering"}
        )
        active_permitted = not inhibit
        if not active_permitted:
            overall = "unsafe"
        elif any(state != "healthy" for state in states.values()):
            overall = "degraded"
        else:
            overall = "healthy"
        quality = float(sum(
            self._WEIGHTS[name] * self._STATE_SCORE[states[name]]
            for name in self._WEIGHTS
        ))
        # BEVFusion requires valid LiDAR and multi-camera content on the exact
        # submitted frame.  Skip unusable jobs immediately; the debounced gate
        # separately decides whether prior fresh evidence may still actuate.
        perception_usable = bool(lidar_ok and camera_array_ok and alignment_ok)

        self._frames += 1
        self._overall_frames[overall] = self._overall_frames.get(overall, 0) + 1
        self._inhibited_frames += int(not active_permitted)
        self._unusable_perception_frames += int(not perception_usable)
        return SensorHealthSnapshot(
            frame=int(sensor_frame.frame),
            timestamp_s=float(sensor_frame.timestamp_s),
            overall_state=overall,
            quality=quality,
            active_control_permitted=active_permitted,
            control_inhibit_reasons=inhibit,
            perception_input_usable=perception_usable,
            components={
                name: {
                    "state": states[name],
                    "raw_ok": raw_ok[name],
                    "bad_streak": self._components[name].bad_streak,
                    "good_streak": self._components[name].good_streak,
                }
                for name in self._WEIGHTS
            },
            raw={
                "cameras": camera_metrics,
                "usable_camera_count": usable_cameras,
                "required_usable_camera_count": self.config.min_usable_cameras,
                "lidar_point_count": point_count,
                "lidar_invalid_fraction": lidar_invalid_fraction,
                "radar_return_counts": radar_counts,
                "radar_return_total": radar_total,
                "source_frame_spread": source_spread,
            },
        )

    def summary(self) -> dict:
        return {
            "schema": "fusion_tracking.sensor_health_summary.v1",
            "config": asdict(self.config),
            "frames": self._frames,
            "overall_state_frames": dict(self._overall_frames),
            "control_inhibited_frames": self._inhibited_frames,
            "perception_input_unusable_frames": self._unusable_perception_frames,
            "raw_bad_frames": dict(self._raw_bad_frames),
            "component_transitions": {
                name: dict(component.transitions)
                for name, component in self._components.items()
            },
            "final_component_states": {
                name: component.state
                for name, component in self._components.items()
            },
        }
