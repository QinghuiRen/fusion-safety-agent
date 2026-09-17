"""Fusion Safety Agent 从多传感器感知到纵向控制仲裁的有状态主流水线。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from fusion_tracking_carla.autoware_aeb_shadow import (
    AutowareAEBShadowConfig,
    AutowareAEBShadowEvaluator,
)
from fusion_tracking_carla.config import Stage63Config
from fusion_tracking_carla.geometry import global_vector_to_ego
from fusion_tracking_carla.inference_arbiter import PriorityInferenceArbiter
from fusion_tracking_carla.perception import LatestOnlyPerceptionWorker, PerceptionJob
from fusion_tracking_carla.safety import assess_tracks
from fusion_tracking_carla.safety_supervisor import (
    LongitudinalSupervisorConfig,
    RiskAwareLongitudinalSupervisor,
)
from fusion_tracking_carla.tracker import StreamingFusionTracker

from .config import PROJECT_ROOT
from .telemetry import OnlineTimingTelemetry
from .sensor_health import SensorHealthConfig, OnlineSensorHealthMonitor
from .lidar_safety_cage import LidarPathSafetyCage, LidarSafetyCageConfig
from .front_vru_safety import FrontVRUSafetyConfig, FrontVRUSafetyMonitor
from .output_contract import (
    FRAME_LOG_FILENAME,
    FRAME_SCHEMA,
    PRODUCT_ID,
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA,
)
from .privacy import sanitize_for_export


class AgentTelemetryWriter:
    """Append-only product output; failures here never take control authority."""

    def __init__(self, config):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = Path(config.output_root).expanduser().resolve() / f"agent_run_{stamp}"
        self.enabled = bool(config.write_jsonl)
        self._handle = None
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._handle = (self.run_dir / FRAME_LOG_FILENAME).open(
                "a", encoding="utf-8"
            )

    def write(self, record: dict) -> dict | None:
        if self._handle is None:
            return None
        started = time.perf_counter()
        # 运行时可以使用绝对路径，但产品日志不得固化用户名、主机名等部署信息。
        exported = sanitize_for_export(record, product_root=PROJECT_ROOT)
        line = json.dumps(exported, ensure_ascii=False) + "\n"
        serialized = time.perf_counter()
        self._handle.write(line)
        self._handle.flush()
        finished = time.perf_counter()
        return {
            "json_serialization_s": serialized - started,
            "write_flush_s": finished - serialized,
            "total_s": finished - started,
            "bytes": len(line.encode("utf-8")),
        }

    def close(self, summary: dict) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            exported = sanitize_for_export(
                summary, product_root=PROJECT_ROOT
            )
            (self.run_dir / SUMMARY_FILENAME).write_text(
                json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8"
            )


class FusionSafetyPipeline:
    """管理感知、跟踪、预测、风险、健康门控和控制状态。

    本类只依赖已经适配的传感器帧，不依赖 Leaderboard 类，也不读取仿真真值。
    """

    def __init__(self, config, worker=None, vru_worker=None):
        self.product_config = config
        self.runtime_config, contract = self._build_runtime_config(config)
        self.writer = AgentTelemetryWriter(config)
        self.inference_arbiter = PriorityInferenceArbiter(
            primary_slow_threshold_s=float(
                config.gpu_arbiter_slow_threshold_s
            ),
            protection_primary_observations=int(
                config.gpu_arbiter_protection_results
            ),
        )
        self.worker = worker if worker is not None else self._build_worker(config, contract)
        if worker is None:
            self.worker.wait_until_ready(float(config.startup_timeout_s))
        self.tracker = StreamingFusionTracker(self.runtime_config.tracker)
        self.supervisor = RiskAwareLongitudinalSupervisor(
            LongitudinalSupervisorConfig(
                active_intervention=config.supervisor_mode == "active",
                max_fusion_age_s=float(config.supervisor_max_fusion_age_s),
                max_track_measurement_age_s=float(
                    config.supervisor_max_track_measurement_age_s
                ),
                release_frames=6,
                max_service_brake=0.25,
                emergency_brake=0.30,
                brake_ramp_per_s=0.50,
                stationary_hold_brake=0.05,
                stationary_hold_distance_m=16.0,
                stationary_hold_dropout_frames=40,
                risk_dropout_bridge_frames=6,
                deceleration_governor_enabled=True,
                coast_during_hold_and_release=True,
            )
        )
        self.aeb_shadow = AutowareAEBShadowEvaluator(AutowareAEBShadowConfig())
        self.lidar_safety_cage = LidarPathSafetyCage(LidarSafetyCageConfig(
            mode=str(config.lidar_safety_cage_mode),
        ))
        self.vru_safety_monitor = None
        if config.vru_safety_monitor_mode != "off":
            self.vru_safety_monitor = FrontVRUSafetyMonitor(
                self.runtime_config,
                FrontVRUSafetyConfig(
                    mode=str(config.vru_safety_monitor_mode),
                    detector_model=str(config.vru_detector_model),
                    detector_score_threshold=float(
                        config.vru_detector_score_threshold
                    ),
                    detector_stride_frames=int(config.vru_detector_stride_frames),
                    startup_timeout_s=float(config.startup_timeout_s),
                    core_path_half_width_m=float(
                        config.vru_core_path_half_width_m
                    ),
                    monitor_half_width_m=float(config.vru_monitor_half_width_m),
                    min_inward_speed_mps=float(config.vru_min_inward_speed_mps),
                ),
                worker=vru_worker,
                inference_arbiter=self.inference_arbiter,
            )
        self.sensor_health = OnlineSensorHealthMonitor(SensorHealthConfig(
            failure_frames=int(config.sensor_health_failure_frames),
            recovery_frames=int(config.sensor_health_recovery_frames),
            min_lidar_points=int(config.sensor_health_min_lidar_points),
            max_source_frame_spread=int(config.sensor_health_max_frame_spread),
        ))
        self.timing = OnlineTimingTelemetry(
            stale_threshold_s=float(self.supervisor.config.max_fusion_age_s)
        )
        self._last_result_frame = -1
        self._last_fusion_timestamp_s: float | None = None
        self._last_step_timestamp_s: float | None = None
        self._frames = 0
        self._errors = 0
        self._fault_active_frames = 0
        self._fault_affected_counts: dict[str, int] = {}
        self._combined_applied_frames = 0
        self._combined_applied_events = 0
        self._last_combined_applied = False
        startup_gate_enabled = bool(
            config.first_perception_sync_enabled
            and config.frontend == "certified_bevfusion"
        )
        self._startup_gate = {
            "enabled": startup_gate_enabled,
            "ready": not startup_gate_enabled,
            "status": "pending" if startup_gate_enabled else "disabled",
            "attempts": 0,
            "failure_count": 0,
            "source_frame": None,
            "wait_wall_s": None,
            "error": None,
            "recovered_after_failure": False,
            "applied_hold_frames": 0,
        }
        self._startup_failure_result_frame: int | None = None

    @staticmethod
    def _certificate_contract(config) -> dict:
        if config.frontend != "certified_bevfusion":
            return {}
        path = Path(config.certificate).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload["runtime_contract"])

    @classmethod
    def _build_runtime_config(cls, config):
        base = Stage63Config()
        contract = cls._certificate_contract(config)
        perception = replace(
            base.perception,
            detector_model=(
                "mmdetection3d_bevfusion_lidar_camera"
                if config.frontend == "certified_bevfusion"
                else "fasterrcnn_resnet50_fpn_v2"
            ),
            detector_stride_frames=int(config.detector_stride_frames),
            detector_score_threshold=float(contract.get("score_threshold", 0.55)),
            detector_startup_timeout_s=float(config.startup_timeout_s),
        )
        tracker = replace(
            base.tracker,
            new_track_score_threshold=float(contract.get("new_track_score_threshold", 0.25)),
            min_confirmed_hits=int(contract.get("min_confirmed_hits", 2)),
            radar_association_gate_m=float(contract.get("radar_association_gate_m", 3.2)),
            live_radar_association_gate_m=float(contract.get("live_radar_association_gate_m", 3.2)),
            enable_live_radar_refinement=bool(contract.get("enable_live_radar_refinement", True)),
            enable_recovery_association=bool(contract.get("enable_recovery_association", True)),
        )
        sensors = replace(
            base.sensors,
            camera_width=int(config.camera_width),
            camera_height=int(config.camera_height),
            camera_fov_deg=float(config.camera_fov_deg),
            lidar_range_m=float(config.lidar_range_m),
            radar_range_m=float(config.radar_range_m),
        )
        result = replace(base, perception=perception, tracker=tracker, sensors=sensors)
        result.validate()
        return result, contract

    def _build_worker(self, config, contract):
        if config.frontend == "legacy":
            return LatestOnlyPerceptionWorker(
                self.runtime_config.perception,
                self.runtime_config.sensors,
                self.runtime_config.tracker,
            )
        from fusion_tracking_bevfusion.active_carla import (
            CertifiedActiveConfig,
            CertifiedBEVFusionActiveWorker,
        )
        from fusion_tracking_bevfusion.carla_online import CarlaBEVFusionConfig
        from fusion_tracking_bevfusion.quality_gate import sha256_file

        checkpoint = Path(config.checkpoint).expanduser().resolve()
        bridge = CarlaBEVFusionConfig(
            wsl_distro=config.wsl_distro,
            wsl_runtime=config.wsl_runtime,
            checkpoint=checkpoint,
            expected_checkpoint_sha256=sha256_file(checkpoint),
            packet_root=self.writer.run_dir / "runtime_packets",
            backend_score_threshold=float(contract["backend_score_threshold"]),
            score_threshold=float(contract["score_threshold"]),
            use_model_velocity=bool(contract["use_model_velocity"]),
            use_amp_fp16=config.inference_precision == "amp_fp16",
            packet_transport=config.bevfusion_packet_transport,
            vehicle_nms_iou_threshold=float(contract["vehicle_nms_iou_threshold"]),
            vru_nms_iou_threshold=float(contract["vru_nms_iou_threshold"]),
            frame_timeout_s=12.0,
            first_frame_timeout_s=float(config.first_perception_timeout_s),
        )
        active = CertifiedActiveConfig(
            certificate_path=str(Path(config.certificate).expanduser().resolve()),
            checkpoint=str(checkpoint),
            inference_precision=config.inference_precision,
            enable_legacy_fallback=False,
        )
        return CertifiedBEVFusionActiveWorker(
            self.runtime_config.perception,
            self.runtime_config.sensors,
            self.runtime_config.tracker,
            bridge,
            active,
            inference_arbiter=self.inference_arbiter,
        )

    @staticmethod
    def _radar_returns(sensor_frame) -> list:
        return [item for sample in sensor_frame.radars.values() for item in sample.returns]

    def _prime_first_perception(self, job, health) -> bool:
        """Synchronize exactly one usable real frame before route release.

        Returns ``True`` when this call submitted the job, allowing ``step`` to
        avoid immediately submitting the same frame a second time.  A bounded
        failure is recorded and control remains held; later successful async
        results may recover the gate without hiding the original error.
        """

        gate = self._startup_gate
        if not gate["enabled"] or gate["ready"] or gate["attempts"]:
            return False
        if not health.perception_input_usable:
            gate["status"] = "waiting_for_usable_input"
            return False
        warmup = getattr(self.worker, "warmup_with_frame", None)
        gate["attempts"] += 1
        gate["source_frame"] = int(job.sensor_frame.frame)
        if not callable(warmup):
            gate["status"] = "failed_closed"
            gate["failure_count"] += 1
            gate["error"] = "worker_missing_retainable_warmup_contract"
            self._startup_failure_result_frame = int(job.sensor_frame.frame)
            self._errors += 1
            return False
        started = time.perf_counter()
        try:
            elapsed = warmup(
                job,
                float(self.product_config.first_perception_timeout_s),
                retain_result=True,
            )
        except Exception as exc:
            gate["wait_wall_s"] = time.perf_counter() - started
            gate["status"] = "failed_closed"
            gate["failure_count"] += 1
            gate["error"] = f"{type(exc).__name__}: {exc}"
            self._startup_failure_result_frame = int(job.sensor_frame.frame)
            self._errors += 1
            return True
        gate["wait_wall_s"] = float(elapsed)
        gate["status"] = "ready"
        gate["ready"] = True
        gate["error"] = None
        return True

    def step(
        self, adapted, base_control: dict, context: dict | None = None
    ) -> tuple[dict, dict]:
        started = time.perf_counter()
        timestamp_s = float(adapted.sensor_frame.timestamp_s)
        dt_s = (
            float(self.product_config.fixed_delta_s)
            if self._last_step_timestamp_s is None
            else max(timestamp_s - self._last_step_timestamp_s, 1e-3)
        )
        self._last_step_timestamp_s = timestamp_s
        health = self.sensor_health.evaluate(adapted)
        job = PerceptionJob(
            sensor_frame=adapted.sensor_frame,
            ego_pose_xyyaw=adapted.ego_pose_xyyaw,
            ego_z_m=0.0,
            ego_velocity_global=adapted.ego_velocity_global,
            submitted_wall_s=started,
        )
        startup_submitted = self._prime_first_perception(job, health)
        if (
            health.perception_input_usable
            and adapted.sensor_frame.frame
            % int(self.product_config.detector_stride_frames) == 0
            and not startup_submitted
        ):
            self.worker.submit(job)

        result = self.worker.latest()
        perception_error = None
        perception_update = None
        if result is not None and result.frame > self._last_result_frame:
            self._last_result_frame = int(result.frame)
            perception_error = result.error
            perception_update = self.timing.observe_perception(
                result, time.perf_counter()
            )
            timestamp_regressed = bool(
                self._last_fusion_timestamp_s is not None
                and float(result.timestamp_s) + 1e-6
                < self._last_fusion_timestamp_s
            )
            if timestamp_regressed:
                perception_error = (
                    "out_of_sequence_perception_timestamp: "
                    f"{float(result.timestamp_s):.6f} < "
                    f"{self._last_fusion_timestamp_s:.6f}"
                )
                self._errors += 1
            elif result.error is None:
                self.tracker.update(
                    list(result.detections),
                    float(result.timestamp_s),
                    np.asarray(result.ego_velocity_global, dtype=float),
                )
                self._last_fusion_timestamp_s = float(result.timestamp_s)
                if self._startup_gate["enabled"] and not self._startup_gate["ready"]:
                    self._startup_gate["ready"] = True
                    self._startup_gate["status"] = "ready"
                    self._startup_gate["recovered_after_failure"] = bool(
                        self._startup_gate["failure_count"]
                    )
            else:
                if int(result.frame) != self._startup_failure_result_frame:
                    self._errors += 1

        tracks = self.tracker.snapshots(timestamp_s)
        tracks, radar_stats = self.tracker.refine_snapshots_with_radar(
            tracks,
            self._radar_returns(adapted.sensor_frame),
            adapted.ego_velocity_global,
        )
        safety = assess_tracks(
            tracks,
            adapted.ego_pose_xyyaw,
            adapted.ego_velocity_global,
            self.runtime_config.safety,
        )
        fusion_age_s = (
            None
            if self._last_fusion_timestamp_s is None
            else max(timestamp_s - self._last_fusion_timestamp_s, 0.0)
        )
        ego_velocity_ego = global_vector_to_ego(
            adapted.ego_velocity_global, float(adapted.ego_pose_xyyaw[2])
        )
        decision = self.supervisor.evaluate(
            timestamp_s=timestamp_s,
            safety=safety,
            tracks=tracks,
            ego_velocity_ego=ego_velocity_ego,
            fusion_age_s=fusion_age_s,
            dt_s=dt_s,
        )
        track_control, track_applied = self.supervisor.arbitrate(
            base_control,
            "route_agent",
            inhibit_reasons=health.control_inhibit_reasons,
        )
        vru_record = None
        vru_control = track_control
        vru_applied = False
        if self.vru_safety_monitor is not None:
            vru_control, vru_record = self.vru_safety_monitor.step(
                job=job,
                health=health,
                radar_returns=self._radar_returns(adapted.sensor_frame),
                dt_s=dt_s,
                base_control=track_control,
                safety_config=self.runtime_config.safety,
            )
            vru_applied = bool(vru_record["application"]["applied"])
        lidar_component = health.components.get("lidar", {})
        alignment_component = health.components.get("alignment", {})
        cage_decision = self.lidar_safety_cage.evaluate(
            timestamp_s=timestamp_s,
            lidar=adapted.sensor_frame.lidar,
            ego_speed_mps=float(np.linalg.norm(ego_velocity_ego)),
            radar_returns=self._radar_returns(adapted.sensor_frame),
            ego_pose_xyyaw=adapted.ego_pose_xyyaw,
            ego_velocity_global=adapted.ego_velocity_global,
            input_healthy=bool(
                lidar_component.get("state") == "healthy"
                and alignment_component.get("state") == "healthy"
            ),
        )
        applied_control, cage_applied, cage_apply_reason = (
            self.lidar_safety_cage.arbitrate(
                vru_control,
                cage_decision,
                inhibit_reasons=health.control_inhibit_reasons,
            )
        )
        startup_gate_applied = bool(
            self._startup_gate["enabled"] and not self._startup_gate["ready"]
        )
        startup_gate_reason = self._startup_gate["status"]
        if startup_gate_applied:
            applied_control = dict(applied_control)
            applied_control["throttle"] = 0.0
            applied_control["brake"] = max(
                float(applied_control.get("brake", 0.0)),
                float(self.product_config.first_perception_hold_brake),
            )
            self._startup_gate["applied_hold_frames"] += 1
        startup_gate_record = {
            **dict(self._startup_gate),
            "applied": startup_gate_applied,
            "apply_reason": startup_gate_reason,
            "hold_brake": float(self.product_config.first_perception_hold_brake),
        }
        combined_applied = bool(
            track_applied.applied
            or vru_applied
            or cage_applied
            or startup_gate_applied
        )
        self._combined_applied_frames += int(combined_applied)
        self._combined_applied_events += int(
            combined_applied and not self._last_combined_applied
        )
        self._last_combined_applied = combined_applied
        # Preserve the stable applied_control schema while exposing each
        # arbitration layer for audit and replay.
        applied = {
            "source_timestamp_s": timestamp_s,
            "applied": combined_applied,
            "base_control": dict(base_control),
            "applied_control": dict(applied_control),
            "reason": (
                "perception_startup_gate" if startup_gate_applied
                else "lidar_path_safety_cage" if cage_applied
                else "front_vru_safety_monitor" if vru_applied
                else track_applied.reason
            ),
            "inhibit_reasons": list(health.control_inhibit_reasons),
            "layers": {
                "track_supervisor": track_applied.to_dict(),
                "front_vru_safety_monitor": (
                    None if vru_record is None else vru_record["application"]
                ),
                "lidar_safety_cage": {
                    "applied": cage_applied,
                    "reason": cage_apply_reason,
                },
                "perception_startup_gate": startup_gate_record,
            },
        }
        aeb = self.aeb_shadow.evaluate(
            timestamp_s=timestamp_s,
            tracks=tracks,
            ego_pose_xyyaw=adapted.ego_pose_xyyaw,
            ego_velocity_global=adapted.ego_velocity_global,
            ego_yaw_rate_radps=adapted.ego_yaw_rate_radps,
            reference_intervention_requested=decision.intervention_requested,
        )
        self._frames += 1
        context_dict = dict(context or {})
        fault_status = context_dict.get("fault_injection", {})
        if fault_status.get("active", False):
            self._fault_active_frames += 1
            for channel in fault_status.get("affected", []):
                self._fault_affected_counts[channel] = (
                    self._fault_affected_counts.get(channel, 0) + 1
                )
        algorithm_step_latency_s = time.perf_counter() - started
        self.timing.observe_control(
            simulation_timestamp_s=timestamp_s,
            wall_timestamp_s=time.perf_counter(),
            algorithm_step_latency_s=algorithm_step_latency_s,
            fusion_age_s=fusion_age_s,
        )
        record = {
            "schema": FRAME_SCHEMA,
            "product": PRODUCT_ID,
            "frame": int(adapted.sensor_frame.frame),
            "timestamp_s": timestamp_s,
            "sensor_profile": self.product_config.sensor_profile,
            "frontend": self.product_config.frontend,
            "supervisor_mode": self.product_config.supervisor_mode,
            # Ego state is reconstructed only from the public Agent packet
            # (GNSS, IMU compass/yaw-rate and speedometer).  Keeping it next to
            # tracks/control makes every risk and intervention frame replayable
            # without granting the algorithm access to CARLA actor truth.
            "ego_state": {
                "pose_global_xyyaw": [
                    float(value) for value in adapted.ego_pose_xyyaw[:3]
                ],
                "velocity_global_mps": [
                    float(value) for value in adapted.ego_velocity_global[:2]
                ],
                "speed_mps": float(np.linalg.norm(adapted.ego_velocity_global[:2])),
                "yaw_rate_radps": float(adapted.ego_yaw_rate_radps),
                "source": "GNSS+IMU+speedometer",
                "oracle_truth_used": False,
            },
            "sensor_input_timing": dict(adapted.source_metadata),
            "sensor_health": health.to_dict(),
            "fusion_age_s": fusion_age_s,
            # Legacy alias retained for existing analysis scripts. This excludes
            # JSON serialization and disk I/O; see timing.clock semantics in
            # the episode summary for the precise definition.
            "pipeline_wall_latency_s": algorithm_step_latency_s,
            "algorithm_step_wall_latency_s": algorithm_step_latency_s,
            "perception_error": perception_error,
            "perception_update": perception_update,
            "perception_startup_gate": startup_gate_record,
            "radar_refinement": radar_stats,
            "tracks": [track.to_dict() for track in tracks],
            "safety": [item.to_dict() for item in safety],
            "supervisor": decision.to_dict(),
            "front_vru_safety_monitor": vru_record,
            "lidar_safety_cage": cage_decision.to_dict(),
            "applied_control": applied,
            "autoware_aeb_shadow": aeb.to_dict(),
            "context": context_dict,
        }
        self.timing.observe_output(self.writer.write(record))
        return applied_control, record

    def close(self) -> None:
        diagnostics_method = getattr(self.worker, "diagnostics_snapshot", None)
        if callable(diagnostics_method):
            perception_runtime = diagnostics_method()
        else:
            perception_runtime = {
                "worker_class": type(self.worker).__name__,
                "startup": dict(getattr(self.worker, "startup_diagnostics", {})),
            }
        vru_summary = (
            None
            if self.vru_safety_monitor is None
            else self.vru_safety_monitor.summary()
        )
        try:
            self.worker.close()
        finally:
            try:
                if self.vru_safety_monitor is not None:
                    self.vru_safety_monitor.close()
            finally:
                self.writer.close({
                "schema": SUMMARY_SCHEMA,
                "product": PRODUCT_ID,
                "frames": self._frames,
                "perception_errors": self._errors,
                "supervisor": self.supervisor.metrics.finalize(),
                "autoware_aeb_shadow": self.aeb_shadow.metrics.finalize(),
                "lidar_safety_cage": self.lidar_safety_cage.summary(),
                "front_vru_safety_monitor": vru_summary,
                "combined_control": {
                    "applied_frames": self._combined_applied_frames,
                    "applied_intervention_events": self._combined_applied_events,
                },
                "sensor_profile": self.product_config.sensor_profile,
                "frontend": self.product_config.frontend,
                "perception_runtime": perception_runtime,
                "gpu_inference_arbiter": self.inference_arbiter.snapshot(),
                "perception_startup_gate": dict(self._startup_gate),
                "online_timing": self.timing.finalize(),
                "sensor_health": self.sensor_health.summary(),
                "supervisor_mode": self.product_config.supervisor_mode,
                "vru_safety_monitor_mode": (
                    self.product_config.vru_safety_monitor_mode
                ),
                "fault_injection": {
                    "profile": self.product_config.fault_profile,
                    "active_frames": self._fault_active_frames,
                    "active_fraction": (
                        self._fault_active_frames / self._frames if self._frames else 0.0
                    ),
                    "affected_channel_frames": self._fault_affected_counts,
                },
                })
