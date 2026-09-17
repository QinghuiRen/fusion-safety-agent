"""用于部署检查、配置查看和本地驾驶的产品命令行接口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT, FusionSafetyAgentConfig
from .diagnostics import build_diagnostic_report, diagnostic_passed
from .privacy import sanitize_for_export


DEFAULT_CONFIG = PROJECT_ROOT / "configs/product/fusion_safety_agent.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusion-safety",
        description="Multi-sensor trajectory-risk and longitudinal-safety Agent",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("doctor", "self-check", "show-config", "manual", "demo"),
        default="doctor",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--town", default="Town01")
    parser.add_argument("--weather", default="ClearNoon")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Driving duration in simulation seconds; zero runs until Esc.",
    )
    parser.add_argument("--npc", type=int, default=8)
    parser.add_argument("--target-speed", type=float, default=30.0)
    parser.add_argument(
        "--manual-supervisor",
        "--supervisor-mode",
        dest="supervisor_mode",
        choices=("shadow", "active"),
        default="shadow",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--reuse-server", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = FusionSafetyAgentConfig.from_json(args.config)
    if args.command in {"manual", "demo"}:
        # 主项目虚拟环境有意不安装 CARLA/Torch；本地驾驶和自动演示均转接到隔离
        # 运行环境，避免污染项目依赖，同时保持统一入口和产品配置。
        from fusion_tracking_carla.runtime_launcher import handoff_if_needed

        handoff_if_needed(PROJECT_ROOT / "fusion_safety_agent.py", PROJECT_ROOT)
        from .manual_runtime import ManualRunOptions, run_manual

        output = run_manual(
            config,
            ManualRunOptions(
                town=args.town,
                weather=args.weather,
                duration_s=args.duration,
                npc_vehicles=args.npc,
                target_speed_kmh=args.target_speed,
                supervisor_mode=args.supervisor_mode,
                drive_mode="manual" if args.command == "manual" else "route_agent",
                record_video=not args.no_video,
                reuse_server=args.reuse_server,
            ),
        )
        print(f"[Fusion Safety Agent output] {output}")
        return 0
    if args.command == "show-config":
        print(
            json.dumps(
                sanitize_for_export(config.__dict__, product_root=PROJECT_ROOT),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    deep = args.command == "doctor"
    report = build_diagnostic_report(config, check_wsl=deep)
    print("[Fusion Safety Agent diagnostics]")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if diagnostic_passed(report, check_wsl=deep) else 2
