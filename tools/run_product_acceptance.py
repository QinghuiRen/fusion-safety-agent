"""Run Fusion Safety Agent through the real carla_garage evaluator.

The Windows CARLA 0.9.15 shipping build used by this project can terminate on
``client.load_world()`` even when the requested map is already loaded.  The
official evaluator unconditionally reloads the map for every route.  This
launcher keeps the evaluator unchanged and replaces only that map-loading hook:
an already matching world is reused, while a mismatching world is rejected.

All route parsing, sensor spawning, Agent lifecycle, ScenarioManager execution
and Leaderboard statistics remain the upstream carla_garage implementation.
"""

from __future__ import annotations

import argparse
import _thread
from datetime import datetime
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_tracking_carla.runtime_launcher import (
    DEFAULT_CARLA_AGENTS_ROOT,
    _discover_carla_root,
)

# 外部 CARLA / carla_garage 不属于本仓库；部署时可通过环境变量覆盖。
CARLA_ROOT = _discover_carla_root()
CARLA_AGENTS_ROOT = DEFAULT_CARLA_AGENTS_ROOT
GARAGE_ROOT = Path(
    os.environ.get("FUSION_SAFETY_GARAGE_ROOT", "D:/fusion_tracking_runtime/carla_garage")
).expanduser()
DEFAULT_AGENT = PROJECT_ROOT / "fusion_safety_agent.py"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/product/fusion_safety_agent.json"
DEFAULT_ROUTE = PROJECT_ROOT / "configs/evaluation/routes/town01_nominal.xml"
EVALUATION_ROOT = PROJECT_ROOT / "out/evaluation/product"
EFFECTIVE_CONFIG_FILENAME = "fusion_safety_effective_config.json"
ACCEPTANCE_SUMMARY_FILENAME = "fusion_safety_acceptance_summary.json"
SCENARIO_EVIDENCE_FILENAME = "scenario_evidence.jsonl"


def _monitor_launched_server(
    process,
    stop_event: threading.Event,
    failure: dict,
    *,
    poll_interval_s: float = 0.25,
    interrupt_main=_thread.interrupt_main,
) -> None:
    """Interrupt a blocked evaluator when its launcher-owned UE4 exits.

    CARLA's streaming client otherwise keeps retrying a dead port and may hold
    the regression process until its outer wall timeout. The monitor never
    probes or terminates a user-owned CARLA server; it only observes the exact
    ``Popen`` returned by this invocation's launcher.
    """

    while not stop_event.wait(max(0.01, float(poll_interval_s))):
        return_code = process.poll()
        if return_code is None:
            continue
        failure.update(
            {
                "kind": "carla_server_process_exit",
                "return_code": int(return_code),
            }
        )
        interrupt_main()
        return


def _write_infrastructure_failure(output_dir: Path, failure: dict) -> Path:
    destination = output_dir / "infrastructure_failure.json"
    destination.write_text(
        json.dumps(
            {
                "schema": "fusion_tracking.infrastructure_failure.v1",
                **failure,
                "classification": "simulator_infrastructure",
                "algorithm_acceptance": "not_evaluated",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def _is_successful_route_status(status: object) -> bool:
    """Match the two terminal-success labels emitted by Leaderboard."""

    return status in {"Perfect", "Completed"}


def _route_elapsed_s(game_time_s: float, start_game_time_s: float) -> float:
    """Return route-relative simulation time, never CARLA server uptime."""

    return max(0.0, float(game_time_s) - float(start_game_time_s))


def _add_ignored_timeout_compatibility(scenario_class: type) -> bool:
    """Adapt an older ScenarioRunner constructor to newer Leaderboard calls.

    The bundled Leaderboard passes ``timeout=`` to every route scenario, while
    a few classes in the bundled ScenarioRunner predate that contract.  The
    timeout is already owned by RouteScenario, so ignoring only this unsupported
    keyword preserves the scenario's behavior and avoids editing third-party
    source files.  Return ``True`` only when an adapter was installed.
    """

    if getattr(scenario_class, "_stage620_timeout_compatible", False):
        return False
    original = scenario_class.__init__
    if "timeout" in inspect.signature(original).parameters:
        return False

    def compatible_init(self, *args, timeout=None, **kwargs):
        del timeout
        return original(self, *args, **kwargs)

    compatible_init.__name__ = original.__name__
    compatible_init.__doc__ = original.__doc__
    scenario_class.__init__ = compatible_init
    scenario_class._stage620_timeout_compatible = True
    return True


def _install_scenario_constructor_compatibility() -> tuple[str, ...]:
    """Install narrow in-memory adapters for known upstream API mismatches."""

    adapted = []
    # carla_garage's class discovery imports each scenario file by its bare
    # module name (``follow_leading_vehicle``), not by the package-qualified
    # name.  Patch that exact module object so discovery receives the adapter.
    scenario_dir = GARAGE_ROOT / "scenario_runner" / "srunner" / "scenarios"
    scenario_dir_value = str(scenario_dir.resolve())
    if scenario_dir_value not in sys.path:
        sys.path.insert(0, scenario_dir_value)
    module = importlib.import_module("follow_leading_vehicle")
    scenario_class = module.FollowLeadingVehicleWithObstacle
    if _add_ignored_timeout_compatibility(scenario_class):
        adapted.append("FollowLeadingVehicleWithObstacle:ignore_timeout_keyword")
    return tuple(adapted)


def _route_town(route_path: Path) -> str:
    routes = ET.parse(route_path.expanduser().resolve()).getroot().findall("route")
    towns = {str(route.attrib.get("town", "")).strip() for route in routes}
    towns.discard("")
    if len(towns) != 1:
        raise ValueError(
            "Fusion Safety runner requires exactly one town per route file: "
            f"{sorted(towns)}"
        )
    return towns.pop()


def _prepend_runtime_paths() -> None:
    paths = (
        PROJECT_ROOT,
        GARAGE_ROOT / "leaderboard",
        GARAGE_ROOT / "scenario_runner",
        CARLA_AGENTS_ROOT,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Fusion Safety runtime paths: {missing}")
    for path in reversed(paths):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    # RouteScenario discovers scenario classes via this upstream environment
    # contract rather than sys.path alone. Set it to the isolated checkout;
    # never overwrite files in the user's installed CARLA runtime.
    os.environ["LEADERBOARD_ROOT"] = str((GARAGE_ROOT / "leaderboard").resolve())
    os.environ["SCENARIO_RUNNER_ROOT"] = str(
        (GARAGE_ROOT / "scenario_runner").resolve()
    )
    os.environ["CARLA_ROOT"] = str(CARLA_ROOT.resolve())


def _install_same_map_reuse(evaluator_module) -> None:
    data_provider = evaluator_module.CarlaDataProvider

    def reuse_matching_world(self, args, town):
        world = self.client.get_world()
        current_town = world.get_map().name.split("/")[-1]
        if current_town != town:
            raise RuntimeError(
                f"CARLA is on {current_town}, but route requires {town}. "
                "Restart the server directly on the requested map; hot map "
                "switching is disabled for this Windows runtime."
            )
        print(f"[Fusion Safety map guard] reuse current world: {current_town}", flush=True)
        self.world = world
        settings = world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        world.apply_settings(settings)
        world.reset_all_traffic_lights()
        data_provider.set_client(self.client)
        data_provider.set_traffic_manager_port(args.traffic_manager_port)
        data_provider.set_world(world)
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)
        world.tick()

    evaluator_module.LeaderboardEvaluator._load_and_wait_for_world = reuse_matching_world


def _install_scenario_observability(
    evaluator_module,
    recorder,
    max_sim_duration_s: float,
    background_traffic: str = "standard",
) -> None:
    """Instrument the evaluator side without exposing truth to the Agent.

    The upstream Leaderboard route timeout is intentionally very large.  A
    malformed custom regression route can consequently run for hours.  The
    bounded manager below records the official behavior-tree state and stops
    the evaluator after the configured *simulation* time.  A capped run remains
    an acceptance failure; the cap is operational protection, not a pass rule.
    """

    base_route_scenario = evaluator_module.RouteScenario
    base_manager = evaluator_module.ScenarioManager
    # ``world.get_snapshot().timestamp.elapsed_seconds`` is absolute since the
    # CARLA server started.  It is therefore wrong for a reused server and can
    # make a new route look minutes old on its very first tick.  ScenarioRunner
    # owns a resettable simulation clock specifically for per-route timing.
    from srunner.scenariomanager.timer import GameTime

    if background_traffic == "disabled":
        # RouteScenario normally injects BackgroundActivity in addition to the
        # scenario actors declared by the route.  That is desirable for broad
        # nominal tests, but makes an isolated positive AEB test ambiguous: a
        # random vehicle can be spawned between ego and FollowLeadingVehicle.
        # Replace only the evaluator-side background behavior with a forever-
        # running no-op.  The official scenario tree, criteria, sensors and
        # Agent stay unchanged; no actor truth enters run_step or control.
        import importlib
        from srunner.scenariomanager.scenarioatomics.atomic_behaviors import WaitForever

        route_scenario_module = importlib.import_module(base_route_scenario.__module__)

        def no_background_behavior(_ego_actor, _route, debug=False, name="BackgroundActivity"):
            del debug
            return WaitForever(name=name)

        route_scenario_module.BackgroundBehavior = no_background_behavior

    class ObservableRouteScenario(base_route_scenario):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorder.register_route(self)

    class BoundedObservableScenarioManager(base_manager):
        def _tick_scenario(self):
            super()._tick_scenario()
            if self.scenario is None:
                return
            elapsed_s = _route_elapsed_s(GameTime.get_time(), self.start_game_time)
            recorder.observe(self.scenario, elapsed_s)
            if self._running and elapsed_s >= max_sim_duration_s:
                recorder.mark_termination("max_sim_duration", elapsed_s)
                print(
                    "[Fusion Safety runtime guard] "
                    f"sim={elapsed_s:.1f}s reached max={max_sim_duration_s:.1f}s; "
                    "stopping as failed/incomplete evidence",
                    flush=True,
                )
                self._running = False

        def stop_scenario(self):
            result = super().stop_scenario()
            if self.scenario is not None and recorder.summary()["termination_reason"] is None:
                recorder.mark_termination(
                    "official_tree_terminal",
                    max(0.0, float(self.scenario_duration_game)),
                )
            return result

    evaluator_module.RouteScenario = ObservableRouteScenario
    evaluator_module.ScenarioManager = BoundedObservableScenarioManager


def _evaluator_arguments(args, output_dir: Path, config_path: Path) -> list[str]:
    return [
        "leaderboard_evaluator.py",
        "--host", args.host,
        "--port", str(args.port),
        "--traffic-manager-port", str(args.traffic_manager_port),
        "--traffic-manager-seed", str(args.seed),
        "--debug", str(args.debug),
        "--timeout", str(args.timeout),
        "--routes", str(args.routes.resolve()),
        "--repetitions", "1",
        "--agent", str(args.agent.resolve()),
        "--agent-config", str(config_path.resolve()),
        "--track", "SENSORS",
        "--checkpoint", str(output_dir / "leaderboard_results.json"),
        "--debug-checkpoint", str(output_dir / "leaderboard_live.json"),
    ]


def _write_effective_config(args, output_dir: Path) -> tuple[Path, dict]:
    """Create an immutable per-run config without changing the product default."""

    source = args.config.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    overrides = {
        "fault_profile": args.fault_profile,
        "supervisor_mode": args.supervisor_mode,
        "detector_stride_frames": args.detector_stride_frames,
        "fault_period_frames": args.fault_period_frames,
        "fault_duration_frames": args.fault_duration_frames,
        "bevfusion_packet_transport": args.bevfusion_packet_transport,
        "lidar_safety_cage_mode": args.lidar_safety_cage_mode,
        "vru_safety_monitor_mode": args.vru_safety_monitor_mode,
        "vru_detector_stride_frames": args.vru_detector_stride_frames,
        "vru_detector_score_threshold": args.vru_detector_score_threshold,
    }
    applied = {key: value for key, value in overrides.items() if value is not None}
    payload.update(applied)
    from fusion_tracking_agent.config import Stage620AgentConfig

    # Validate before CARLA spends time spawning sensors.
    Stage620AgentConfig(**payload).validate()
    destination = output_dir / EFFECTIVE_CONFIG_FILENAME
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "source_config": str(source),
        "effective_config": str(destination),
        "overrides": applied,
        "fault_profile": payload.get("fault_profile", "none"),
        "supervisor_mode": payload.get("supervisor_mode", "shadow"),
        "agent_output_root": str(
            (
                Path(payload.get("output_root", "out/product")).expanduser()
                if Path(payload.get("output_root", "out/product")).expanduser().is_absolute()
                else PROJECT_ROOT / payload.get("output_root", "out/product")
            ).resolve()
        ),
        "detector_stride_frames": payload.get("detector_stride_frames"),
        "vru_safety_monitor_mode": payload.get(
            "vru_safety_monitor_mode", "off"
        ),
    }
    return destination, metadata


def _write_acceptance_summary(
    output_dir: Path,
    agent_runs_before: set[Path],
    run_config: dict,
    scenario_evidence: dict,
) -> bool:
    result_path = output_dir / "leaderboard_results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    records = payload.get("_checkpoint", {}).get("records", [])
    record = records[0] if records else {}
    infractions = record.get("infractions", {})
    scores = record.get("scores", {})
    agent_output_root = Path(run_config["agent_output_root"]).resolve()
    current_runs = set(agent_output_root.glob("agent_run_*"))
    created_runs = sorted(current_runs - agent_runs_before, key=lambda path: path.stat().st_mtime)
    agent_run = created_runs[-1] if created_runs else None
    agent_summary = {}
    if agent_run is not None:
        from fusion_tracking_agent.output_contract import resolve_summary

        summary_path = resolve_summary(agent_run)
        if summary_path is not None:
            agent_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    collisions = sum(
        len(infractions.get(name, []))
        for name in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle")
    )
    route_completion = float(scores.get("score_route", 0.0))
    supervisor = agent_summary.get("supervisor", {}) or {}
    combined_control = agent_summary.get("combined_control", {}) or {}
    startup_gate = agent_summary.get("perception_startup_gate", {}) or {}
    startup_gate_pass = bool(
        not startup_gate.get("enabled", False)
        or (
            startup_gate.get("ready") is True
            and int(startup_gate.get("failure_count", 0)) == 0
        )
    )
    applied_frames = int(combined_control.get(
        "applied_frames", supervisor.get("applied_frames", 0)
    ))
    requested_events = int(supervisor.get("intervention_events", 0))
    applied_events = int(combined_control.get(
        "applied_intervention_events",
        supervisor.get("applied_intervention_events", 0),
    ))
    # Upstream Leaderboard reports an infraction-free completed route as
    # ``Perfect`` and a completed route with non-fatal infractions as
    # ``Completed``.  Both are terminal successes; safety gates below still
    # reject collisions and route-lane violations explicitly.
    route_status_success = _is_successful_route_status(record.get("status"))
    common_gates = (
        payload.get("entry_status") == "Finished",
        route_status_success,
        route_completion >= 99.0,
        collisions == 0,
        not infractions.get("outside_route_lanes", []),
        int(agent_summary.get("frames", 0)) > 0,
        int(agent_summary.get("perception_errors", 1)) == 0,
        int((agent_summary.get("front_vru_safety_monitor") or {}).get(
            "perception_errors", 0
        )) == 0,
        startup_gate_pass,
    )
    contract = run_config["acceptance_contract"]
    if contract == "active_hazard":
        contract_gates = (
            applied_frames > 0
            and applied_events > 0
            and scenario_evidence.get("expected_scenario_initialized") is True
        )
    elif contract == "normal_negative":
        contract_gates = applied_frames == 0 and applied_events == 0
    else:
        contract_gates = True
    acceptance = {
        "schema": "fusion_tracking.safety_agent.acceptance.v1",
        "product": "fusion_safety_agent",
        "contract": f"carla_garage_leaderboard_{contract}",
        "pass": all(common_gates) and contract_gates,
        "entry_status": payload.get("entry_status"),
        "route_status": record.get("status"),
        "route_completion_percent": route_completion,
        "driving_score": float(scores.get("score_composed", 0.0)),
        "infraction_penalty": float(scores.get("score_penalty", 0.0)),
        "collisions": collisions,
        "outside_route_lane_events": len(infractions.get("outside_route_lanes", [])),
        # Min-speed is intentionally reported, not hidden.  It is a score
        # penalty in this smoke contract, while completion/collision are gates.
        "min_speed_infractions": list(infractions.get("min_speed_infractions", [])),
        "agent_run": str(agent_run) if agent_run is not None else None,
        "agent_frames": int(agent_summary.get("frames", 0)),
        "perception_errors": int(agent_summary.get("perception_errors", 0)),
        "frontend": agent_summary.get("frontend"),
        "perception_runtime": agent_summary.get("perception_runtime", {}),
        "gpu_inference_arbiter": agent_summary.get(
            "gpu_inference_arbiter", {}
        ),
        "perception_startup_gate": startup_gate,
        "sensor_profile": agent_summary.get("sensor_profile"),
        "run_config": run_config,
        "fault_injection": agent_summary.get("fault_injection", {}),
        "supervisor_mode": agent_summary.get("supervisor_mode"),
        "supervisor_applied_frames": applied_frames,
        "supervisor_requested_intervention_events": requested_events,
        "supervisor_applied_intervention_events": applied_events,
        "combined_control": combined_control,
        "lidar_safety_cage": agent_summary.get("lidar_safety_cage", {}),
        "front_vru_safety_monitor": agent_summary.get(
            "front_vru_safety_monitor"
        ),
        "expected_intervention": run_config.get("expected_intervention"),
        "online_timing": agent_summary.get("online_timing", {}),
        "sensor_health": agent_summary.get("sensor_health", {}),
        "scenario_evidence": scenario_evidence,
    }
    summary_path = output_dir / ACCEPTANCE_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "[Fusion Safety acceptance] "
        f"pass={acceptance['pass']} | route={route_completion:.1f}% | "
        f"score={acceptance['driving_score']:.2f} | collision={collisions} | "
        f"agent_frames={acceptance['agent_frames']} | "
        f"perception_errors={acceptance['perception_errors']}",
        flush=True,
    )
    print(f"[acceptance summary] {summary_path}", flush=True)
    return bool(acceptance["pass"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fusion Safety real carla_garage acceptance evaluation"
    )
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument(
        "--restart-server-if-map-mismatch",
        action="store_true",
        help=(
            "For unattended multi-map regression only: stop the exact running "
            "CARLA process tree when its map differs, then cold-start the route map."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTE)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fault-profile",
        choices=(
            "none", "front_camera_dropout", "lidar_dropout",
            "radar_dropout", "mixed",
        ),
        default=None,
        help="Per-run deterministic sensor fault; the base config is not modified.",
    )
    parser.add_argument(
        "--supervisor-mode", choices=("shadow", "active"), default=None
    )
    parser.add_argument("--detector-stride-frames", type=int, default=None)
    parser.add_argument("--fault-period-frames", type=int, default=None)
    parser.add_argument("--fault-duration-frames", type=int, default=None)
    parser.add_argument(
        "--bevfusion-packet-transport",
        choices=(
            "windows_mount", "wsl_native", "wsl_inline_images",
            "wsl_stdin_packet",
        ),
        default=None,
        help=(
            "Transient BEVFusion packet path. wsl_stdin_packet is the "
            "certified product default; wsl_inline_images and wsl_native "
            "remain compatibility transports."
        ),
    )
    parser.add_argument(
        "--lidar-safety-cage-mode",
        choices=("off", "shadow", "active"),
        default=None,
        help=(
            "Per-run detector-independent raw-LiDAR path safety cage mode; "
            "use shadow before promoting a new scenario/platform to active."
        ),
    )
    parser.add_argument(
        "--vru-safety-monitor-mode",
        choices=("off", "shadow", "active"),
        default=None,
        help=(
            "Independent CAM_FRONT+LiDAR/Radar VRU safety channel. Start in "
            "shadow; use active only after controlled positive/negative tests."
        ),
    )
    parser.add_argument(
        "--vru-detector-stride-frames",
        type=int,
        default=None,
        help="VRU detector period in 20 Hz control frames (4 means about 5 Hz).",
    )
    parser.add_argument(
        "--vru-detector-score-threshold",
        type=float,
        default=None,
        help="COCO person/bicycle/motorcycle score gate for the VRU channel.",
    )
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument(
        "--max-sim-duration-s",
        type=float,
        default=120.0,
        help=(
            "Hard simulation-time guard for malformed routes. Reaching it is "
            "recorded as incomplete and can never make acceptance pass."
        ),
    )
    parser.add_argument(
        "--expected-scenario-type",
        default=None,
        help=(
            "Evaluator-side scenario class required by a positive regression; "
            "never included in the Agent configuration."
        ),
    )
    parser.add_argument(
        "--background-traffic",
        choices=("standard", "disabled"),
        default="standard",
        help=(
            "Evaluator traffic policy. 'disabled' removes only random "
            "RouteScenario BackgroundActivity actors; official scenario "
            "actors and all Agent inputs remain unchanged."
        ),
    )
    parser.add_argument(
        "--acceptance-contract",
        choices=("smoke", "active_hazard", "normal_negative"),
        default="smoke",
        help=(
            "active_hazard requires an applied intervention; normal_negative "
            "requires zero applied interventions."
        ),
    )
    parser.add_argument(
        "--expected-intervention",
        choices=("unknown", "yes", "no"),
        default="unknown",
        help="Evaluator-side label only; it is never passed to Agent control.",
    )
    args = parser.parse_args(argv)
    if args.max_sim_duration_s <= 0:
        parser.error("--max-sim-duration-s must be positive")

    # A directly executed file exposes ``tools/`` as sys.path[0].  Add the
    # repository root before importing the local runtime launcher.
    project_value = str(PROJECT_ROOT)
    if project_value not in sys.path:
        sys.path.insert(0, project_value)

    # Do the environment handoff before importing CARLA/Leaderboard modules.
    from fusion_tracking_carla.runtime_launcher import (
        _terminate_launched_server,
        handoff_if_needed,
        launch_server_and_wait,
    )

    handoff_if_needed(Path(__file__), PROJECT_ROOT)
    _prepend_runtime_paths()
    scenario_constructor_adapters = _install_scenario_constructor_compatibility()
    from leaderboard import leaderboard_evaluator

    _install_same_map_reuse(leaderboard_evaluator)
    launched_process = None
    if args.launch_server:
        launched_process = launch_server_and_wait(
            max_wait_s=150.0,
            host=args.host,
            port=args.port,
            startup_map=_route_town(args.routes),
            extra_args=("-ExecCmds=r.SkeletalMeshLODBias 0",),
            restart_if_map_mismatch=args.restart_server_if_map_mismatch,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = EVALUATION_ROOT / f"acceptance_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    server_watch_stop = threading.Event()
    server_failure: dict = {}
    server_watch = None
    if launched_process is not None:
        server_watch = threading.Thread(
            target=_monitor_launched_server,
            args=(launched_process, server_watch_stop, server_failure),
            name="fusion-safety-carla-server-watchdog",
            daemon=True,
        )
        server_watch.start()
    from fusion_tracking_agent.scenario_evidence import ScenarioLifecycleRecorder

    scenario_recorder = ScenarioLifecycleRecorder(
        output_dir / SCENARIO_EVIDENCE_FILENAME,
        expected_scenario_type=args.expected_scenario_type,
    )
    _install_scenario_observability(
        leaderboard_evaluator,
        scenario_recorder,
        max_sim_duration_s=float(args.max_sim_duration_s),
        background_traffic=args.background_traffic,
    )
    effective_config, run_config = _write_effective_config(args, output_dir)
    run_config["acceptance_contract"] = args.acceptance_contract
    run_config["expected_intervention"] = {
        "unknown": None,
        "yes": True,
        "no": False,
    }[args.expected_intervention]
    run_config["expected_scenario_type"] = args.expected_scenario_type
    run_config["max_sim_duration_s"] = float(args.max_sim_duration_s)
    run_config["background_traffic"] = args.background_traffic
    run_config["scenario_constructor_adapters"] = list(scenario_constructor_adapters)
    agent_runs_before = set(
        Path(run_config["agent_output_root"]).resolve().glob("agent_run_*")
    )
    print(
        "[Fusion Safety acceptance] upstream evaluator + standard Agent + "
        f"{run_config['supervisor_mode'].upper()} | "
        f"VRU={run_config['vru_safety_monitor_mode'].upper()} | "
        f"fault={run_config['fault_profile']}",
        flush=True,
    )
    print(f"[output] {output_dir}", flush=True)
    old_argv = sys.argv
    sys.argv = _evaluator_arguments(args, output_dir, effective_config)
    exit_code = 1
    try:
        try:
            leaderboard_evaluator.main()
        except SystemExit as exc:
            exit_code = int(exc.code or 0)
        except KeyboardInterrupt:
            if not server_failure:
                raise
            failure_path = _write_infrastructure_failure(output_dir, server_failure)
            print(
                "[Fusion Safety infrastructure failure] launcher-owned CARLA "
                f"exited; diagnostic={failure_path}",
                flush=True,
            )
            exit_code = 86
    finally:
        sys.argv = old_argv
        server_watch_stop.set()
        if server_watch is not None:
            server_watch.join(timeout=1.0)
        if launched_process is not None and not args.keep_server:
            _terminate_launched_server(launched_process)
            print("[CARLA server] stopped launcher-owned process", flush=True)
    if exit_code != 0:
        return exit_code
    return 0 if _write_acceptance_summary(
        output_dir, agent_runs_before, run_config, scenario_recorder.summary()
    ) else 3


if __name__ == "__main__":
    raise SystemExit(main())
