"""Load Fusion Safety Agent through the real carla_garage contract.

This is a non-simulation validation: it checks dynamic entry-point loading,
inheritance, SensorInterface construction and the official sensor limits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_tracking_carla.runtime_launcher import DEFAULT_CARLA_AGENTS_ROOT

DEFAULT_GARAGE = Path(
    os.environ.get("FUSION_SAFETY_GARAGE_ROOT", "D:/fusion_tracking_runtime/carla_garage")
).expanduser()
DEFAULT_CONFIG = PROJECT_ROOT / "configs/product/fusion_safety_agent.json"
DEFAULT_AGENT = PROJECT_ROOT / "fusion_safety_agent.py"
DEFAULT_CARLA_AGENTS = DEFAULT_CARLA_AGENTS_ROOT


def _prepend_external_paths(garage: Path, carla_agents: Path) -> None:
    paths = (
        PROJECT_ROOT,
        garage / "leaderboard",
        garage / "scenario_runner",
        carla_agents,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 Leaderboard 路径: {missing}")
    for path in reversed(paths):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def validate(
    agent_path: Path,
    config_path: Path,
    garage: Path,
    carla_agents: Path,
    *,
    load_backend: bool = False,
) -> dict:
    _prepend_external_paths(garage, carla_agents)
    from leaderboard.autoagents.autonomous_agent import AutonomousAgent
    from leaderboard.autoagents.agent_wrapper import validate_sensor_configuration
    from fusion_tracking_agent.config import Stage620AgentConfig

    spec = importlib.util.spec_from_file_location("fusion_safety_external_agent", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 Agent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry_name = module.get_entry_point()
    agent_class = getattr(module, entry_name)
    if not issubclass(agent_class, AutonomousAgent):
        raise TypeError("Fusion Safety Agent 未继承当前 carla_garage AutonomousAgent")

    # Deliberately skip setup(): this validator must not load CUDA/WSL models.
    agent = agent_class("127.0.0.1", 2000, False)
    agent.config = Stage620AgentConfig.from_json(config_path)
    sensors = agent.sensors()
    validate_sensor_configuration(sensors, agent.track, "SENSORS")
    counts: dict[str, int] = {}
    for sensor in sensors:
        counts[sensor["type"]] = counts.get(sensor["type"], 0) + 1
    result = {
        "entry_point": entry_name,
        "inherits_official_autonomous_agent": True,
        "sensor_configuration_valid": True,
        "sensor_counts": counts,
        "leaderboard_root": str(garage.resolve()),
        "agent": str(agent_path.resolve()),
        "config": str(config_path.resolve()),
        "carla_agents": str(carla_agents.resolve()),
        "backend_loaded": False,
    }
    if load_backend:
        try:
            agent.setup(str(config_path))
            result["backend_loaded"] = True
            result["backend_diagnostics"] = dict(
                getattr(agent.pipeline.worker, "startup_diagnostics", {})
            )
        finally:
            agent.destroy()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-root", type=Path, default=DEFAULT_GARAGE)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--carla-agents-root", type=Path, default=DEFAULT_CARLA_AGENTS)
    parser.add_argument(
        "--load-backend",
        action="store_true",
        help="同时执行 Agent.setup()/destroy() 并启动真实 WSL BEVFusion 后端",
    )
    args = parser.parse_args(argv)
    result = validate(
        args.agent.resolve(),
        args.config.resolve(),
        args.leaderboard_root.resolve(),
        args.carla_agents_root.resolve(),
        load_backend=args.load_backend,
    )
    print("[Fusion Safety carla_garage contract validation passed]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
