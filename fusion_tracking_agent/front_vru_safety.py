"""面向行人和骑行者的独立前视快速安全通道。

经认证的 BEVFusion 仍是主三维感知前端。本模块只在 ``CAM_FRONT`` 上运行更快的
道路参与者检测器，并用同一公开 RGB/LiDAR/Radar 数据建立独立时间线，用于覆盖短时
遮挡窗口；不会注入 CARLA actor、地图对象或场景标签。

The monitor is intentionally commissioned in ``shadow`` first.  ``active``
only changes longitudinal throttle/brake after two independent observations,
sensor-health checks, and a path-crossing gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from fusion_tracking_carla.geometry import global_vector_to_ego
from fusion_tracking_carla.inference_arbiter import GLOBAL_GPU_INFERENCE_ARBITER
from fusion_tracking_carla.perception import LatestOnlyPerceptionWorker, PerceptionJob
from fusion_tracking_carla.safety import TrackSafety, assess_tracks
from fusion_tracking_carla.safety_supervisor import (
    LongitudinalSupervisorConfig,
    RiskAwareLongitudinalSupervisor,
)
from fusion_tracking_carla.tracker import StreamingFusionTracker


@dataclass(frozen=True)
class FrontVRUSafetyConfig:
    """Tunable contract for the complementary vulnerable-road-user channel."""

    mode: str = "off"
    detector_model: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    detector_score_threshold: float = 0.30
    detector_stride_frames: int = 4
    startup_timeout_s: float = 180.0
    # Tracks already overlapping the ego path can enter the risk model even
    # when their lateral velocity is small (for example a stopped pedestrian).
    core_path_half_width_m: float = 1.8
    # Outside the core path, a target must be moving inward and its predicted
    # trajectory must intersect the core path.  This prevents sidewalk/static
    # side objects from becoming false braking requests.
    monitor_half_width_m: float = 4.5
    min_inward_speed_mps: float = 0.45
    max_fusion_age_s: float = 0.85

    def validate(self) -> None:
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError("VRU safety mode must be off, shadow or active")
        if self.detector_model not in {
            "fasterrcnn_mobilenet_v3_large_320_fpn",
            "ssdlite320_mobilenet_v3_large",
        }:
            raise ValueError("unsupported front VRU detector model")
        if not 0.0 < self.detector_score_threshold <= 1.0:
            raise ValueError("VRU detector score threshold must be in (0,1]")
        if self.detector_stride_frames < 1:
            raise ValueError("VRU detector stride must be >=1")
        if self.startup_timeout_s <= 0 or self.max_fusion_age_s <= 0:
            raise ValueError("VRU startup/freshness timeouts must be positive")
        if not 0.0 < self.core_path_half_width_m < self.monitor_half_width_m:
            raise ValueError("VRU path widths must satisfy 0 < core < monitor")
        if self.min_inward_speed_mps < 0:
            raise ValueError("VRU inward-speed threshold must be non-negative")


def select_actionable_vru_safety(
    safety: list[TrackSafety],
    config: FrontVRUSafetyConfig,
) -> tuple[list[TrackSafety], list[dict]]:
    """Keep in-path VRUs and genuinely inward predicted crossings.

    The gate is deliberately applied before the generic longitudinal
    supervisor.  It uses only tracked position/velocity and predicted points;
    static pedestrians on a sidewalk remain visible in telemetry but cannot
    request braking merely because the ego vehicle is approaching them.
    """

    selected: list[TrackSafety] = []
    evidence: list[dict] = []
    for item in safety:
        x_m, y_m = np.asarray(item.position_ego, dtype=float)
        vy_mps = float(np.asarray(item.velocity_ego, dtype=float)[1])
        future = np.asarray(item.future_positions_ego, dtype=float).reshape(-1, 2)
        min_future_abs_y = (
            float(np.min(np.abs(future[:, 1]))) if len(future) else abs(float(y_m))
        )
        in_front_monitor = bool(
            x_m > 0.0 and abs(y_m) <= config.monitor_half_width_m
        )
        in_core_path = bool(abs(y_m) <= config.core_path_half_width_m)
        inward_speed = (
            max(-float(np.sign(y_m)) * vy_mps, 0.0)
            if abs(y_m) > 1e-6 else 0.0
        )
        predicted_crossing = bool(
            min_future_abs_y <= config.core_path_half_width_m
            and inward_speed >= config.min_inward_speed_mps
        )
        accepted = bool(in_front_monitor and (in_core_path or predicted_crossing))
        if accepted:
            selected.append(item)
        evidence.append({
            "track_id": int(item.track_id),
            "accepted": accepted,
            "in_core_path": in_core_path,
            "predicted_crossing": predicted_crossing,
            "inward_speed_mps": inward_speed,
            "min_future_abs_lateral_m": min_future_abs_y,
            "reason": (
                "in_core_path" if accepted and in_core_path
                else "predicted_crossing" if accepted
                else "outside_front_monitor" if not in_front_monitor
                else "static_or_not_crossing"
            ),
        })
    return selected, evidence


class FrontVRUSafetyMonitor:
    """Latest-only front VRU detector, tracker, predictor and supervisor."""

    def __init__(
        self,
        runtime_config,
        config: FrontVRUSafetyConfig,
        worker=None,
        inference_arbiter=None,
    ):
        config.validate()
        if config.mode == "off":
            raise ValueError("FrontVRUSafetyMonitor must not be built in off mode")
        self.config = config
        perception = replace(
            runtime_config.perception,
            detector_model=config.detector_model,
            detector_score_threshold=float(config.detector_score_threshold),
            detector_stride_frames=int(config.detector_stride_frames),
            detector_camera_channels=("CAM_FRONT",),
            detector_precision="fp32",
            detector_batch_size=1,
            detector_max_objects_per_camera=15,
            # For VRUs the calibrated contact point is the position anchor.
            # Projected LiDAR remains evidence/support, but cannot drag the
            # centre onto the wall/prop that originally occluded the person.
            # The fallback is never applied to ordinary vehicles.
            radar_only_lift_enabled=False,
            ground_plane_vru_lift_enabled=True,
            ground_plane_vru_prefer_over_lidar=True,
            ground_plane_vru_min_visual_score=max(
                0.70, float(config.detector_score_threshold)
            ),
            ground_plane_vru_max_lateral_m=float(config.monitor_half_width_m),
        )
        tracker_config = replace(
            runtime_config.tracker,
            min_confirmed_hits=2,
            new_track_score_threshold=float(config.detector_score_threshold),
            max_missed_updates=5,
            max_prediction_age_s=float(config.max_fusion_age_s),
            enable_recovery_association=True,
            enable_live_radar_refinement=True,
            process_accel_std_mps2=4.0,
            position_observation_std_m=0.9,
        )
        self.worker = worker or LatestOnlyPerceptionWorker(
            perception,
            runtime_config.sensors,
            tracker_config,
            inference_arbiter=(
                inference_arbiter or GLOBAL_GPU_INFERENCE_ARBITER
            ),
            inference_priority="normal",
        )
        if worker is None:
            self.worker.wait_until_ready(float(config.startup_timeout_s))
        self.tracker = StreamingFusionTracker(tracker_config)
        self.supervisor = RiskAwareLongitudinalSupervisor(
            LongitudinalSupervisorConfig(
                active_intervention=config.mode == "active",
                warning_score=35.0,
                brake_score=52.0,
                emergency_score=82.0,
                release_score=22.0,
                min_reliability=0.42,
                max_fusion_age_s=float(config.max_fusion_age_s),
                max_track_measurement_age_s=float(config.max_fusion_age_s),
                corridor_half_width_m=float(config.monitor_half_width_m),
                # The stricter conditional crossing gate above controls entry;
                # the supervisor must not reject an already approved crossing.
                path_half_width_m=float(config.monitor_half_width_m),
                activation_frames=2,
                release_frames=6,
                max_service_brake=0.45,
                emergency_brake=0.70,
                brake_ramp_per_s=4.0,
                brake_release_per_s=2.0,
                risk_dropout_bridge_frames=3,
                coast_during_hold_and_release=True,
            )
        )
        self._last_result_frame = -1
        self._last_fusion_timestamp_s: float | None = None
        self._frames = 0
        self._perception_results = 0
        self._perception_errors = 0
        self._accepted_detections = 0
        self._selected_safety_frames = 0
        self._last_record: dict[str, Any] = {}

    @staticmethod
    def _local_inhibit_reasons(health) -> tuple[str, ...]:
        reasons = list(health.control_inhibit_reasons)
        for name in ("camera_front", "lidar", "alignment"):
            state = str(health.components.get(name, {}).get("state", "failed"))
            if state in {"failed", "recovering"}:
                reason = f"vru_sensor_health_{name}_{state}"
                if reason not in reasons:
                    reasons.append(reason)
        return tuple(reasons)

    @staticmethod
    def _input_usable(health) -> bool:
        raw = health.raw
        return bool(
            raw.get("cameras", {}).get("CAM_FRONT", {}).get("ok", False)
            and raw.get("lidar_point_count", 0) > 0
            and raw.get("source_frame_spread", 999) <= 1
        )

    def step(
        self,
        *,
        job: PerceptionJob,
        health,
        radar_returns: list,
        dt_s: float,
        base_control: dict,
        safety_config,
    ) -> tuple[dict, dict]:
        """Advance the non-blocking channel and optionally arbitrate control."""

        timestamp_s = float(job.sensor_frame.timestamp_s)
        input_usable = self._input_usable(health)
        submitted = bool(
            input_usable
            and int(job.sensor_frame.frame) % self.config.detector_stride_frames == 0
        )
        if submitted:
            self.worker.submit(job)

        update: dict | None = None
        result = self.worker.latest()
        if result is not None and int(result.frame) > self._last_result_frame:
            self._last_result_frame = int(result.frame)
            self._perception_results += 1
            detections = tuple(
                item for item in result.detections if item.family == "vru"
            )
            if result.error is None:
                self.tracker.update(
                    list(detections),
                    float(result.timestamp_s),
                    np.asarray(result.ego_velocity_global, dtype=float),
                )
                self._last_fusion_timestamp_s = float(result.timestamp_s)
                self._accepted_detections += len(detections)
            else:
                self._perception_errors += 1
            update = {
                "frame": int(result.frame),
                "timestamp_s": float(result.timestamp_s),
                "latency_s": float(result.latency_s),
                "queue_latency_s": float(result.queue_latency_s),
                "detector_latency_s": float(result.detector_latency_s),
                "fusion_latency_s": float(result.fusion_latency_s),
                "all_road_user_boxes": int(sum(
                    len(items) for items in result.camera_boxes.values()
                )),
                "accepted_vru_detections": len(detections),
                "camera_boxes": {
                    channel: [
                        {
                            "class_name": str(item["class_name"]),
                            "score": float(item["score"]),
                            "box2d": np.asarray(
                                item["box2d"], dtype=float
                            ).tolist(),
                        }
                        for item in items
                    ]
                    for channel, items in result.camera_boxes.items()
                },
                "lifting_attempts": list(
                    (result.diagnostics or {}).get("lifting_attempts", [])
                ),
                "error": result.error,
            }

        tracks = self.tracker.snapshots(timestamp_s)
        tracks, radar_stats = self.tracker.refine_snapshots_with_radar(
            tracks, radar_returns, np.asarray(job.ego_velocity_global, dtype=float)
        )
        all_safety = assess_tracks(
            tracks,
            np.asarray(job.ego_pose_xyyaw, dtype=float),
            np.asarray(job.ego_velocity_global, dtype=float),
            safety_config,
        )
        selected_safety, crossing_evidence = select_actionable_vru_safety(
            all_safety, self.config
        )
        self._selected_safety_frames += int(bool(selected_safety))
        fusion_age_s = (
            None if self._last_fusion_timestamp_s is None
            else max(timestamp_s - self._last_fusion_timestamp_s, 0.0)
        )
        ego_velocity_ego = global_vector_to_ego(
            job.ego_velocity_global, float(job.ego_pose_xyyaw[2])
        )
        decision = self.supervisor.evaluate(
            timestamp_s=timestamp_s,
            safety=selected_safety,
            tracks=tracks,
            ego_velocity_ego=ego_velocity_ego,
            fusion_age_s=fusion_age_s,
            dt_s=float(dt_s),
        )
        inhibit_reasons = self._local_inhibit_reasons(health)
        control, applied = self.supervisor.arbitrate(
            base_control,
            "route_agent",
            inhibit_reasons=inhibit_reasons,
        )
        self._frames += 1
        record = {
            "mode": self.config.mode,
            "input_usable": input_usable,
            "job_submitted": submitted,
            "fusion_age_s": fusion_age_s,
            "perception_update": update,
            "tracks": [item.to_dict() for item in tracks],
            "safety": [item.to_dict() for item in all_safety],
            "crossing_gate": crossing_evidence,
            "selected_track_ids": [item.track_id for item in selected_safety],
            "radar_refinement": radar_stats,
            "decision": decision.to_dict(),
            "application": applied.to_dict(),
        }
        self._last_record = record
        return control, record

    def summary(self) -> dict:
        diagnostics = getattr(self.tracker, "diagnostics", None)
        return {
            "schema": "fusion_tracking.front_vru_safety_summary.v1",
            "config": asdict(self.config),
            "frames": self._frames,
            "perception_results": self._perception_results,
            "perception_errors": self._perception_errors,
            "accepted_vru_detections": self._accepted_detections,
            "selected_safety_frames": self._selected_safety_frames,
            "supervisor": self.supervisor.summary(),
            "tracker": diagnostics() if callable(diagnostics) else {},
            "worker_class": type(self.worker).__name__,
            "worker_startup": dict(getattr(self.worker, "startup_diagnostics", {})),
        }

    def close(self) -> None:
        self.worker.close()
