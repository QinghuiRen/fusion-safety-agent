"""产品安装、模型证书与外部运行环境的只读诊断。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fusion_tracking_bevfusion.quality_gate import validate_active_certificate

from .config import PROJECT_ROOT, FusionSafetyAgentConfig
from .leaderboard_agent import get_entry_point
from .privacy import sanitize_for_export
from .sensor_adapter import sensor_suite


def build_diagnostic_report(
    config: FusionSafetyAgentConfig,
    *,
    check_wsl: bool = False,
) -> dict[str, Any]:
    """在不启动 CARLA 的情况下返回机器可读诊断报告。

    深度检查会校验模型运行契约，并执行有超时上限的 WSL 进程握手；它不会加载
    神经网络，也不会修改 CARLA 世界状态。
    """

    sensors = sensor_suite(config)
    radar_count = sum(item["type"] == "sensor.other.radar" for item in sensors)
    checkpoint = Path(config.checkpoint)
    certificate = Path(config.certificate)
    report: dict[str, Any] = {
        "schema": "fusion_tracking.product_diagnostic.v1",
        "entry_point": get_entry_point(),
        "sensor_profile": config.sensor_profile,
        "sensor_count": {
            "camera": sum(item["type"] == "sensor.camera.rgb" for item in sensors),
            "lidar": sum(item["type"] == "sensor.lidar.ray_cast" for item in sensors),
            "radar": radar_count,
        },
        "official_leaderboard_radar_limit_satisfied": (
            config.sensor_profile != "leaderboard" or radar_count <= 4
        ),
        "frontend": config.frontend,
        "packet_transport": config.bevfusion_packet_transport,
        "supervisor_mode": config.supervisor_mode,
        "vru_safety_monitor_mode": config.vru_safety_monitor_mode,
        "lidar_safety_cage_mode": config.lidar_safety_cage_mode,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_exists": checkpoint.is_file(),
        "certificate": str(certificate.resolve()),
        "certificate_exists": certificate.is_file(),
        "output_root": str(Path(config.output_root).resolve()),
    }

    if check_wsl:
        from fusion_tracking_carla.runtime_launcher import (
            DEFAULT_CARLA_PYTHON,
            DEFAULT_CARLA_SERVER,
        )

        report["carla_python_exists"] = DEFAULT_CARLA_PYTHON.is_file()
        report["carla_server_exists"] = DEFAULT_CARLA_SERVER.is_file()
        try:
            certificate_payload = validate_active_certificate(
                certificate,
                checkpoint,
                inference_precision=config.inference_precision,
            )
            report["runtime_certificate"] = {
                "valid": True,
                "schema": certificate_payload["schema_version"],
                "checkpoint_sha256": certificate_payload["checkpoint_sha256"],
                "error": None,
            }
        except Exception as exc:  # 诊断报告需保留根因，便于部署排错。
            report["runtime_certificate"] = {
                "valid": False,
                "schema": None,
                "checkpoint_sha256": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            from fusion_tracking_bevfusion.carla_online import _check_wsl_runtime

            elapsed_s = _check_wsl_runtime(config.wsl_distro, timeout_s=15.0)
            report["wsl_runtime"] = {
                "responsive": True,
                "distro": config.wsl_distro,
                "handshake_s": elapsed_s,
                "error": None,
            }
        except Exception as exc:
            report["wsl_runtime"] = {
                "responsive": False,
                "distro": config.wsl_distro,
                "handshake_s": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    # 诊断结果可直接随交付物保存，因此默认隐藏部署机器的个人目录和主机名。
    return sanitize_for_export(report, product_root=PROJECT_ROOT)


def diagnostic_passed(report: dict[str, Any], *, check_wsl: bool = False) -> bool:
    required = [
        report["official_leaderboard_radar_limit_satisfied"],
        report["checkpoint_exists"],
        report["certificate_exists"],
    ]
    if check_wsl:
        required.extend(
            [
                report["carla_python_exists"],
                report["carla_server_exists"],
                report["runtime_certificate"]["valid"],
                report["wsl_runtime"]["responsive"],
            ]
        )
    return all(required)
