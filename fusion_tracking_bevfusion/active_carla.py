"""受运行证书约束的 CARLA 在线 BEVFusion 主感知工作器。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from fusion_tracking_carla.perception import LatestOnlyPerceptionWorker, PerceptionResult

from .carla_online import CarlaBEVFusionConfig, LatestOnlyBEVFusionWorker
from .quality_gate import validate_active_certificate


@dataclass(frozen=True)
class CertifiedActiveConfig:
    certificate_path: str
    checkpoint: str
    inference_precision: str = "fp32"
    # 8 GB GPUs cannot safely keep the legacy torchvision detector and
    # BEVFusion resident at the same time.  False means a pure certified
    # BEVFusion path; stale perception is then handled by tracker ageing,
    # 20 Hz Radar refinement and the supervisor's stale-data inhibit.
    enable_legacy_fallback: bool = False
    max_bevfusion_lag_frames: int = 12
    max_consecutive_backend_errors: int = 2
    reject_nontrackable_ratio: float = 0.95


class CertifiedBEVFusionActiveWorker:
    """BEVFusion 主前端，并可按显式配置启用传统检测器安全回退。"""

    def __init__(
        self,
        perception_config,
        sensor_config,
        tracker_config,
        bevfusion_config: CarlaBEVFusionConfig,
        active_config: CertifiedActiveConfig,
        *,
        inference_arbiter=None,
    ):
        certificate = validate_active_certificate(
            active_config.certificate_path,
            active_config.checkpoint,
            inference_precision=active_config.inference_precision,
            runtime_contract={
                "inference_precision": active_config.inference_precision,
                "backend_score_threshold": float(
                    bevfusion_config.backend_score_threshold
                ),
                "score_threshold": float(bevfusion_config.score_threshold),
                "vehicle_nms_iou_threshold": float(
                    bevfusion_config.vehicle_nms_iou_threshold
                ),
                "vru_nms_iou_threshold": float(
                    bevfusion_config.vru_nms_iou_threshold
                ),
                "use_model_velocity": bool(bevfusion_config.use_model_velocity),
                "new_track_score_threshold": float(
                    tracker_config.new_track_score_threshold
                ),
                "min_confirmed_hits": int(tracker_config.min_confirmed_hits),
                "radar_association_gate_m": float(
                    tracker_config.radar_association_gate_m
                ),
                "live_radar_association_gate_m": float(
                    tracker_config.live_radar_association_gate_m
                ),
                "enable_live_radar_refinement": bool(
                    tracker_config.enable_live_radar_refinement
                ),
                "enable_recovery_association": bool(
                    tracker_config.enable_recovery_association
                ),
            },
        )
        self.certificate = certificate
        self.active_config = active_config
        self.primary = (
            LatestOnlyPerceptionWorker(perception_config, sensor_config, tracker_config)
            if active_config.enable_legacy_fallback
            else None
        )
        self.bevfusion = None
        self._bevfusion_args = (
            perception_config,
            sensor_config,
            tracker_config,
            replace(bevfusion_config, checkpoint=type(bevfusion_config.checkpoint)(active_config.checkpoint)),
        )
        self._inference_arbiter = inference_arbiter
        self.startup_diagnostics = {}
        self._latest_bevfusion = None
        self._consecutive_errors = 0
        self._fallback_frames = 0
        self._bevfusion_frames = 0
        self._bevfusion_unique_frames = 0
        self._fallback_unique_frames = 0
        self._last_bevfusion_frame = -1
        self._last_fallback_frame = -1
        self._last_active_frontend = "not_ready"
        self._last_gate_reason = "not_ready"

    def wait_until_ready(self, timeout_s: float) -> None:
        # 顺序加载可降低 8 GB 显存冷启动峰值；默认产品配置不加载传统前端。
        if self.primary is not None:
            self.primary.wait_until_ready(timeout_s)
        worker_kwargs = {}
        inference_arbiter = getattr(self, "_inference_arbiter", None)
        if inference_arbiter is not None:
            worker_kwargs["inference_arbiter"] = inference_arbiter
        self.bevfusion = LatestOnlyBEVFusionWorker(
            *self._bevfusion_args,
            **worker_kwargs,
        )
        self.bevfusion.wait_until_ready(timeout_s)
        self.startup_diagnostics = {
            "frontend_mode": (
                "certified_bevfusion_active_with_legacy_fallback"
                if self.primary is not None
                else "certified_bevfusion_active"
            ),
            "certificate_sha256": self.certificate["checkpoint_sha256"],
            "certificate_source_run": self.certificate["source_run"],
            "primary": (
                dict(self.primary.startup_diagnostics)
                if self.primary is not None else None
            ),
            "bevfusion": dict(self.bevfusion.startup_diagnostics),
            # Session沿用统一字段名展示预热耗时。纯BEV模式没有legacy worker，
            # 此处必须保持0而不能为了诊断字段偷偷加载第二套模型。
            "detector_warmup_inference_s": (
                self.primary.startup_diagnostics.get(
                    "detector_warmup_inference_s", 0.0
                )
                if self.primary is not None else 0.0
            ),
        }

    def warmup_with_frame(
        self, job, timeout_s: float, *, retain_result: bool = False
    ) -> float:
        if self.bevfusion is None:
            raise RuntimeError("BEVFusion 尚未初始化")
        elapsed = self.bevfusion.warmup_with_frame(
            job, timeout_s, retain_result=retain_result
        )
        self.startup_diagnostics["bevfusion"] = dict(
            self.bevfusion.startup_diagnostics
        )
        return elapsed

    def submit(self, job) -> None:
        if self.primary is not None:
            self.primary.submit(job)
        if self.bevfusion is not None:
            self.bevfusion.submit(job)

    def _bevfusion_health(self, result: PerceptionResult, primary) -> tuple[bool, str]:
        if result.error:
            self._consecutive_errors += 1
            return False, "backend_error"
        self._consecutive_errors = 0
        if (
            primary is not None
            and result.frame
            < primary.frame - self.active_config.max_bevfusion_lag_frames
        ):
            return False, "bevfusion_stale"
        diag = result.diagnostics or {}
        raw = int(diag.get("backend_raw_detections") or 0)
        trackable = int(diag.get("backend_trackable_raw") or 0)
        if raw > 0 and (raw - trackable) / raw >= self.active_config.reject_nontrackable_ratio:
            return False, "runtime_domain_regression"
        if self._consecutive_errors > self.active_config.max_consecutive_backend_errors:
            return False, "backend_error_limit"
        return True, "certified_bevfusion"

    def latest(self):
        primary = None if self.primary is None else self.primary.latest()
        candidate = None if self.bevfusion is None else self.bevfusion.latest()
        if candidate is not None and (
            self._latest_bevfusion is None or candidate.frame > self._latest_bevfusion.frame
        ):
            self._latest_bevfusion = candidate
        bev = self._latest_bevfusion
        healthy, reason = (
            (False, "bevfusion_not_ready")
            if bev is None
            else self._bevfusion_health(bev, primary)
        )
        self._last_gate_reason = reason
        if healthy:
            self._bevfusion_frames += 1
            if int(bev.frame) != self._last_bevfusion_frame:
                self._bevfusion_unique_frames += 1
                self._last_bevfusion_frame = int(bev.frame)
            self._last_active_frontend = "bevfusion"
            diagnostics = {
                **(bev.diagnostics or {}),
                "active_frontend": "bevfusion",
                "frontend_gate_reason": reason,
                "legacy_fallback_available": primary is not None,
                "certificate_checkpoint_sha256": self.certificate["checkpoint_sha256"],
                "bevfusion_selected_frames": self._bevfusion_frames,
                "fallback_selected_frames": self._fallback_frames,
            }
            return replace(bev, diagnostics=diagnostics)
        if primary is None:
            self._last_active_frontend = "no_healthy_result"
            return bev
        self._fallback_frames += 1
        if int(primary.frame) != self._last_fallback_frame:
            self._fallback_unique_frames += 1
            self._last_fallback_frame = int(primary.frame)
        self._last_active_frontend = "legacy_fallback"
        diagnostics = {
            **(primary.diagnostics or {}),
            "active_frontend": "legacy_fallback",
            "frontend_gate_reason": reason,
            "certificate_checkpoint_sha256": self.certificate["checkpoint_sha256"],
            "bevfusion_selected_frames": self._bevfusion_frames,
            "fallback_selected_frames": self._fallback_frames,
        }
        return replace(primary, diagnostics=diagnostics)

    def diagnostics_snapshot(self) -> dict:
        """Return JSON-safe evidence of which frontend actually drove tracking."""

        return {
            "worker_class": type(self).__name__,
            "frontend_mode": self.startup_diagnostics.get("frontend_mode"),
            "startup": dict(self.startup_diagnostics),
            "last_active_frontend": self._last_active_frontend,
            "last_gate_reason": self._last_gate_reason,
            "bevfusion_selected_control_frames": self._bevfusion_frames,
            "bevfusion_unique_perception_frames": self._bevfusion_unique_frames,
            "legacy_fallback_control_frames": self._fallback_frames,
            "legacy_fallback_unique_perception_frames": self._fallback_unique_frames,
            "legacy_fallback_available": self.primary is not None,
            "backend_errors_at_close": self._consecutive_errors,
        }

    def close(self) -> None:
        if self.primary is not None:
            self.primary.close()
        if self.bevfusion is not None:
            self.bevfusion.close()
