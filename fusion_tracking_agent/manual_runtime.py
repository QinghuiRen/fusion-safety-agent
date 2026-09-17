"""使用正式产品组件组装的本地 CARLA 手动驾驶与仪表盘运行入口。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from fusion_tracking_bevfusion.quality_gate import (
    sha256_file,
    validate_active_certificate,
)

from .config import PROJECT_ROOT, FusionSafetyAgentConfig


@dataclass(frozen=True)
class ManualRunOptions:
    """本地 CARLA 可视化会话中允许操作者调整的参数。

    ``manual`` 用于人在环驾驶，``route_agent`` 用于自动路线演示。两种模式
    共用同一套传感器、BEVFusion、跟踪、风险和纵向安全监督链路，避免演示入口
    与正式产品能力分叉。
    """

    town: str = "Town01"
    weather: str = "ClearNoon"
    duration_s: float = 0.0
    npc_vehicles: int = 8
    target_speed_kmh: float = 30.0
    supervisor_mode: str = "shadow"
    drive_mode: str = "manual"
    record_video: bool = True
    reuse_server: bool = False

    def validate(self) -> None:
        if self.duration_s < 0:
            raise ValueError("duration_s must be non-negative; zero means run until Esc")
        if self.npc_vehicles < 0:
            raise ValueError("npc_vehicles must be non-negative")
        if self.target_speed_kmh <= 0:
            raise ValueError("target_speed_kmh must be positive")
        if self.supervisor_mode not in {"shadow", "active"}:
            raise ValueError("supervisor_mode must be shadow or active")
        if self.drive_mode not in {"manual", "route_agent", "autopilot"}:
            raise ValueError("drive_mode must be manual, route_agent, or autopilot")


def _session_config(
    product: FusionSafetyAgentConfig,
    options: ManualRunOptions,
    runtime_contract: dict,
):
    """把稳定产品契约映射到兼容的 CARLA 会话配置。"""

    from fusion_tracking_carla.config import Stage63Config
    from fusion_tracking_carla.runtime_launcher import DEFAULT_CARLA_AGENTS_ROOT

    base = Stage63Config()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_kind = "manual" if options.drive_mode == "manual" else "demo"
    output_root = Path(product.output_root).resolve() / session_kind
    config = replace(
        base,
        runtime=replace(
            base.runtime,
            stage_id="fusion_safety_product",
            output_root=output_root,
            run_name=f"{session_kind}_{stamp}",
            scenario_name=f"{session_kind}_fusion_safety",
            duration_s=options.duration_s,
            town=options.town,
            weather=options.weather,
            spawn_npc_vehicles=options.npc_vehicles,
            render_dashboard=True,
            record_video=options.record_video,
        ),
        driving=replace(
            base.driving,
            mode=options.drive_mode,
            wait_for_enter=options.drive_mode == "manual",
            target_speed_kmh=options.target_speed_kmh,
            route_target_distance_m=120.0,
            route_min_distance_m=60.0,
            carla_agents_root=DEFAULT_CARLA_AGENTS_ROOT,
        ),
        sensors=replace(
            base.sensors,
            camera_width=product.camera_width,
            camera_height=product.camera_height,
            camera_sensor_tick_s=0.5,
            camera_startup_batch_size=0,
        ),
        perception=replace(
            base.perception,
            detector_model="mmdetection3d_bevfusion_lidar_camera",
            detector_score_threshold=float(runtime_contract["score_threshold"]),
            detector_stride_frames=product.detector_stride_frames,
            detector_startup_timeout_s=product.startup_timeout_s,
        ),
        tracker=replace(
            base.tracker,
            new_track_score_threshold=float(
                runtime_contract["new_track_score_threshold"]
            ),
            min_confirmed_hits=int(runtime_contract["min_confirmed_hits"]),
            radar_association_gate_m=float(
                runtime_contract["radar_association_gate_m"]
            ),
            live_radar_association_gate_m=float(
                runtime_contract["live_radar_association_gate_m"]
            ),
            enable_live_radar_refinement=bool(
                runtime_contract["enable_live_radar_refinement"]
            ),
            enable_recovery_association=bool(
                runtime_contract["enable_recovery_association"]
            ),
        ),
    )
    config.validate()
    return config


def run_manual(
    product: FusionSafetyAgentConfig,
    options: ManualRunOptions,
) -> Path:
    """启动六相机仪表盘，并返回本次运行的独立输出目录。"""

    options.validate()
    from fusion_tracking_carla.runtime_launcher import (
        launch_server_and_wait,
        stop_existing_carla_server,
    )

    certificate = validate_active_certificate(
        product.certificate,
        product.checkpoint,
        inference_precision=product.inference_precision,
    )
    if not options.reuse_server:
        stop_existing_carla_server()
    launch_server_and_wait(
        startup_map=options.town,
        restart_if_map_mismatch=True,
        extra_args=("-ExecCmds=r.SkeletalMeshLODBias 0",),
    )

    from fusion_tracking_bevfusion.active_carla import (
        CertifiedActiveConfig,
        CertifiedBEVFusionActiveWorker,
    )
    from fusion_tracking_bevfusion.carla_online import CarlaBEVFusionConfig
    from fusion_tracking_carla.autoware_aeb_shadow import (
        AutowareAEBShadowConfig,
        AutowareAEBShadowEvaluator,
    )
    from fusion_tracking_carla.safety_supervisor import (
        LongitudinalSupervisorConfig,
        RiskAwareLongitudinalSupervisor,
    )
    from fusion_tracking_carla.session import CarlaFusionSession

    contract = certificate["runtime_contract"]
    session_config = _session_config(product, options, contract)
    packet_root = Path(product.output_root).resolve() / "runtime_packets/manual"
    bridge = CarlaBEVFusionConfig(
        wsl_distro=product.wsl_distro,
        wsl_runtime=product.wsl_runtime,
        checkpoint=Path(product.checkpoint),
        expected_checkpoint_sha256=sha256_file(product.checkpoint),
        packet_root=packet_root,
        packet_transport=product.bevfusion_packet_transport,
        backend_score_threshold=float(contract["backend_score_threshold"]),
        score_threshold=float(contract["score_threshold"]),
        use_model_velocity=bool(contract["use_model_velocity"]),
        use_amp_fp16=product.inference_precision == "amp_fp16",
        vehicle_nms_iou_threshold=float(contract["vehicle_nms_iou_threshold"]),
        vru_nms_iou_threshold=float(contract["vru_nms_iou_threshold"]),
        startup_timeout_s=product.startup_timeout_s,
        first_frame_timeout_s=product.first_perception_timeout_s,
        frame_timeout_s=12.0,
    )
    active = CertifiedActiveConfig(
        certificate_path=str(Path(product.certificate).resolve()),
        checkpoint=str(Path(product.checkpoint).resolve()),
        inference_precision=product.inference_precision,
        enable_legacy_fallback=False,
    )

    def worker_factory(perception, sensors, tracker):
        return CertifiedBEVFusionActiveWorker(
            perception,
            sensors,
            tracker,
            bridge,
            active,
        )

    supervisor = RiskAwareLongitudinalSupervisor(
        LongitudinalSupervisorConfig(
            active_intervention=options.supervisor_mode == "active",
            max_fusion_age_s=product.supervisor_max_fusion_age_s,
            max_track_measurement_age_s=(
                product.supervisor_max_track_measurement_age_s
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
    aeb_shadow = AutowareAEBShadowEvaluator(AutowareAEBShadowConfig())
    if options.drive_mode == "manual":
        print(
            "[Manual driving] Click the dashboard and press Enter. "
            "W/S: throttle/brake, A/D: steer, Q: reverse, Space: handbrake, "
            f"Esc: stop. Safety supervisor={options.supervisor_mode.upper()}",
            flush=True,
        )
    else:
        print(
            f"[Automatic demo] drive_mode={options.drive_mode} | "
            f"supervisor={options.supervisor_mode.upper()} | Esc: stop",
            flush=True,
        )
    run_dir = CarlaFusionSession(
        session_config,
        perception_worker_factory=worker_factory,
        longitudinal_supervisor=supervisor,
        aeb_shadow_evaluator=aeb_shadow,
        aeb_advisory_throttle_cap=(
            0.0 if options.supervisor_mode == "active" else None
        ),
    ).run()
    return Path(run_dir)
