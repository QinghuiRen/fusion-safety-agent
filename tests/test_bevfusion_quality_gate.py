import json

import pytest

from fusion_tracking_bevfusion.quality_gate import (
    build_quality_certificate,
    validate_active_certificate,
)


def _write_run(run_dir, frontend="bevfusion"):
    run_dir.mkdir()
    summary = {
        "oracle_perception": {
            "frames": 200,
            "matches": 160,
            "precision": 0.9,
            "recall": 0.7,
            "position_rmse_m": 1.2,
            "velocity_rmse_mps": 1.5,
            "id_switches": 2,
        },
        "perception_diagnostic_totals": {
            "backend_trackable_raw": 300,
            "accepted_detections": 220,
        },
        "vision_fusion_update_hz_sim": 1.8,
        "mean_perception_latency_s": 0.45,
        "detector_error": None,
        "collision_count": 0,
    }
    (run_dir / "stage6_6_carla_closed_loop_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (run_dir / "stage6_6_bevfusion_bridge_config.json").write_text(
        json.dumps({
            "frontend_mode": frontend,
            "backend_score_threshold": 0.10,
            "score_threshold": 0.20,
            "vehicle_nms_iou_threshold": 0.15,
            "vru_nms_iou_threshold": 0.10,
            "use_model_velocity": False,
        }), encoding="utf-8"
    )
    (run_dir / "stage6_6_carla_closed_loop_config.json").write_text(
        json.dumps({"tracker": {
            "new_track_score_threshold": 0.25,
            "min_confirmed_hits": 3,
            "radar_association_gate_m": 3.2,
            "live_radar_association_gate_m": 3.2,
            "enable_live_radar_refinement": True,
            "enable_recovery_association": False,
        }}), encoding="utf-8"
    )


def test_certificate_passes_only_pure_bevfusion_and_binds_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"real checkpoint bytes")
    certificate = build_quality_certificate(run_dir, checkpoint)
    assert certificate["passed"] is True
    loaded = validate_active_certificate(
        certificate["certificate_path"], checkpoint
    )
    assert loaded["control_authority"] == "eligible_for_fusion_safety_active"
    assert loaded["runtime_contract"]["inference_precision"] == "fp32"
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA256"):
        validate_active_certificate(certificate["certificate_path"], checkpoint)


def test_certificate_binds_inference_precision(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    certificate = build_quality_certificate(run_dir, checkpoint)
    validate_active_certificate(
        certificate["certificate_path"], checkpoint, inference_precision="fp32"
    )
    with pytest.raises(RuntimeError, match="推理精度"):
        validate_active_certificate(
            certificate["certificate_path"],
            checkpoint,
            inference_precision="amp_fp16",
        )


def test_certificate_binds_tracker_and_adapter_contract(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    certificate = build_quality_certificate(run_dir, checkpoint)
    contract = dict(certificate["runtime_contract"])
    validate_active_certificate(
        certificate["certificate_path"], checkpoint, runtime_contract=contract
    )
    contract["enable_recovery_association"] = True
    with pytest.raises(RuntimeError, match="运行契约"):
        validate_active_certificate(
            certificate["certificate_path"], checkpoint,
            runtime_contract=contract,
        )


def test_hybrid_metrics_cannot_be_used_as_active_certificate(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, frontend="hybrid")
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"checkpoint")
    certificate = build_quality_certificate(run_dir, checkpoint)
    assert certificate["passed"] is False
    assert certificate["guards"]["pure_bevfusion_frontend"] is False
    with pytest.raises(RuntimeError, match="未通过"):
        validate_active_certificate(certificate["certificate_path"], checkpoint)


def test_pure_active_worker_startup_does_not_require_legacy(monkeypatch):
    import fusion_tracking_bevfusion.active_carla as active_module

    class FakeBEVWorker:
        def __init__(self, *args):
            self.startup_diagnostics = {"backend": "ready"}

        def wait_until_ready(self, timeout_s):
            assert timeout_s == 30.0

    monkeypatch.setattr(
        active_module, "LatestOnlyBEVFusionWorker", FakeBEVWorker
    )
    worker = object.__new__(active_module.CertifiedBEVFusionActiveWorker)
    worker.primary = None
    worker.bevfusion = None
    worker.certificate = {
        "checkpoint_sha256": "TEST",
        "source_run": "shadow-run",
    }
    worker._bevfusion_args = (object(), object(), object(), object())
    worker.startup_diagnostics = {}
    worker.wait_until_ready(30.0)
    assert worker.startup_diagnostics["frontend_mode"] == "certified_bevfusion_active"
    assert worker.startup_diagnostics["primary"] is None
    assert worker.startup_diagnostics["detector_warmup_inference_s"] == 0.0
