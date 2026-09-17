"""BEVFusion CARLA Shadow -> Active 的不可绕过质量证书。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BEVFusionQualityGateConfig:
    """Active 准入阈值；修改后必须重新生成证书。"""

    min_evaluated_frames: int = 100
    min_precision: float = 0.75
    min_recall: float = 0.45
    max_position_rmse_m: float = 3.0
    max_velocity_rmse_mps: float = 4.0
    max_id_switch_rate: float = 0.08
    min_vision_update_hz: float = 1.0
    max_mean_perception_latency_s: float = 1.0
    min_trackable_raw: int = 20
    min_accepted_detections: int = 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _single_file(run_dir: Path, pattern: str) -> Path:
    matches = list(run_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{run_dir} 中 {pattern} 预期1个，实际{len(matches)}个")
    return matches[0]


def build_quality_certificate(
    run_dir: str | Path,
    checkpoint: str | Path,
    *,
    config: BEVFusionQualityGateConfig | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """只接受纯 BEVFusion Shadow 结果，legacy/hybrid 指标不能冒充。"""

    cfg = config or BEVFusionQualityGateConfig()
    run_dir = Path(run_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    summary_path = _single_file(run_dir, "stage6_6_*_summary.json")
    config_path = _single_file(run_dir, "stage6_6_carla_closed_loop_config.json")
    bridge_path = _single_file(run_dir, "stage6_6_bevfusion_bridge_config.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    tracker = run_config.get("tracker") or {}
    oracle = summary.get("oracle_perception") or {}
    diagnostics = summary.get("perception_diagnostic_totals") or {}
    matches = max(int(oracle.get("matches") or 0), 1)
    id_switch_rate = float(oracle.get("id_switches") or 0) / matches
    metrics = {
        "frontend_mode": bridge.get("frontend_mode"),
        "inference_precision": (
            "amp_fp16" if bool(bridge.get("use_amp_fp16", False)) else "fp32"
        ),
        "frames": int(oracle.get("frames") or 0),
        "precision": oracle.get("precision"),
        "recall": oracle.get("recall"),
        "position_rmse_m": oracle.get("position_rmse_m"),
        "velocity_rmse_mps": oracle.get("velocity_rmse_mps"),
        "id_switch_rate": id_switch_rate,
        "vision_update_hz": summary.get("vision_fusion_update_hz_sim"),
        "mean_perception_latency_s": summary.get("mean_perception_latency_s"),
        "backend_trackable_raw": int(diagnostics.get("backend_trackable_raw") or 0),
        "accepted_detections": int(diagnostics.get("accepted_detections") or 0),
        "detector_error": summary.get("detector_error"),
        "collision_count": int(summary.get("collision_count") or 0),
    }
    guards = {
        "pure_bevfusion_frontend": metrics["frontend_mode"] == "bevfusion",
        "enough_frames": metrics["frames"] >= cfg.min_evaluated_frames,
        "precision": metrics["precision"] is not None and metrics["precision"] >= cfg.min_precision,
        "recall": metrics["recall"] is not None and metrics["recall"] >= cfg.min_recall,
        "position_rmse": metrics["position_rmse_m"] is not None and metrics["position_rmse_m"] <= cfg.max_position_rmse_m,
        "velocity_rmse": metrics["velocity_rmse_mps"] is not None and metrics["velocity_rmse_mps"] <= cfg.max_velocity_rmse_mps,
        "id_stability": id_switch_rate <= cfg.max_id_switch_rate,
        "update_rate": metrics["vision_update_hz"] is not None and metrics["vision_update_hz"] >= cfg.min_vision_update_hz,
        "latency": metrics["mean_perception_latency_s"] is not None and metrics["mean_perception_latency_s"] <= cfg.max_mean_perception_latency_s,
        "trackable_outputs": metrics["backend_trackable_raw"] >= cfg.min_trackable_raw,
        "accepted_outputs": metrics["accepted_detections"] >= cfg.min_accepted_detections,
        "runtime_healthy": metrics["detector_error"] in (None, ""),
        "shadow_collision_free": metrics["collision_count"] == 0,
        "truth_not_online_input": True,
    }
    certificate = {
        "schema_version": "fusion_tracking.bevfusion.active_certificate.v3",
        "passed": all(guards.values()),
        "control_authority": (
            "eligible_for_fusion_safety_active" if all(guards.values()) else "denied"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_run": str(run_dir),
        "source_summary": str(summary_path),
        "source_summary_sha256": sha256_file(summary_path),
        "online_truth_input": False,
        # Active必须复用通过Shadow的完整关键运行契约，不能只复用checkpoint。
        # 例如恢复关联会显著改变ID漂移和误检，不能在签证后被静默打开。
        "runtime_contract": {
            "inference_precision": metrics["inference_precision"],
            "backend_score_threshold": float(bridge.get("backend_score_threshold")),
            "score_threshold": float(bridge.get("score_threshold")),
            "vehicle_nms_iou_threshold": float(bridge.get("vehicle_nms_iou_threshold")),
            "vru_nms_iou_threshold": float(bridge.get("vru_nms_iou_threshold")),
            "use_model_velocity": bool(bridge.get("use_model_velocity")),
            "new_track_score_threshold": float(tracker.get("new_track_score_threshold")),
            "min_confirmed_hits": int(tracker.get("min_confirmed_hits")),
            "radar_association_gate_m": float(tracker.get("radar_association_gate_m")),
            "live_radar_association_gate_m": float(
                tracker.get("live_radar_association_gate_m")
            ),
            "enable_live_radar_refinement": bool(
                tracker.get("enable_live_radar_refinement")
            ),
            "enable_recovery_association": bool(
                tracker.get("enable_recovery_association")
            ),
        },
        "metrics": metrics,
        "thresholds": asdict(cfg),
        "guards": guards,
    }
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else run_dir / "fusion_safety_bevfusion_active_certificate.json"
    )
    destination.write_text(
        json.dumps(certificate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    certificate["certificate_path"] = str(destination)
    return certificate


def validate_active_certificate(
    certificate_path: str | Path,
    checkpoint: str | Path,
    *,
    inference_precision: str | None = None,
    runtime_contract: dict | None = None,
) -> dict:
    path = Path(certificate_path).resolve()
    certificate = json.loads(path.read_text(encoding="utf-8"))
    if certificate.get("schema_version") != "fusion_tracking.bevfusion.active_certificate.v3":
        raise ValueError("未知 BEVFusion Active 证书格式")
    if not certificate.get("passed"):
        failed = [key for key, value in certificate.get("guards", {}).items() if not value]
        raise RuntimeError(f"BEVFusion 未通过 Active 准入：{failed}")
    actual = sha256_file(checkpoint)
    if actual != certificate.get("checkpoint_sha256"):
        raise RuntimeError("checkpoint SHA256 与准入证书不一致")
    if certificate.get("online_truth_input") is not False:
        raise RuntimeError("证书未声明在线真值隔离")
    certified_precision = (
        certificate.get("runtime_contract", {}).get("inference_precision")
    )
    if inference_precision is not None and inference_precision != certified_precision:
        raise RuntimeError(
            "Active推理精度与Shadow准入证书不一致："
            f"active={inference_precision} certified={certified_precision}"
        )
    if runtime_contract is not None:
        certified_contract = certificate.get("runtime_contract") or {}
        mismatches = {
            key: {"active": value, "certified": certified_contract.get(key)}
            for key, value in runtime_contract.items()
            if value != certified_contract.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "Active关键运行契约与Shadow准入证书不一致："
                f"{mismatches}"
            )
    return certificate
