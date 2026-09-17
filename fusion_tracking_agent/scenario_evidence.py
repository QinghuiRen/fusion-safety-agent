"""Evaluator-side lifecycle evidence for CARLA RouteScenario regressions.

This module deliberately observes only the evaluator's scenario tree.  No
scenario actor, expected label, or behavior status is exposed to the driving
Agent.  The resulting JSONL is therefore suitable for diagnosing a regression
without creating a hidden ground-truth control path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


def _status_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _actor_state(actor: Any) -> dict[str, Any]:
    if actor is None:
        return {"available": False}
    try:
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        location = transform.location
        attributes = getattr(actor, "attributes", {}) or {}
        bounding_box = getattr(actor, "bounding_box", None)
        extent = getattr(bounding_box, "extent", None)
        return {
            "available": True,
            "actor_id": int(actor.id),
            "type_id": str(actor.type_id),
            "role_name": str(attributes.get("role_name", "")),
            "location_m": [float(location.x), float(location.y), float(location.z)],
            "speed_mps": float(
                math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z)
            ),
            "bounding_extent_m": (
                [float(extent.x), float(extent.y), float(extent.z)]
                if extent is not None
                else None
            ),
        }
    except Exception as exc:  # Actor may be destroyed between tree ticks.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _tree_snapshot(node: Any, depth: int = 0, max_depth: int = 8) -> dict[str, Any]:
    payload = {
        "name": str(getattr(node, "name", type(node).__name__)),
        "type": type(node).__name__,
        "status": _status_name(getattr(node, "status", None)),
    }
    children = list(getattr(node, "children", ()) or ())
    if children and depth < max_depth:
        payload["children"] = [
            _tree_snapshot(child, depth + 1, max_depth=max_depth) for child in children
        ]
    return payload


def _find_tree_node(node: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    """Return the first named behavior-tree node from a serialized snapshot."""

    if not isinstance(node, dict):
        return None
    if str(node.get("name", "")) == name:
        return node
    for child in node.get("children", ()) or ():
        found = _find_tree_node(child, name)
        if found is not None:
            return found
    return None


def audit_scenario_evidence(
    path: str | Path,
    *,
    expected_scenario_type: str | None = None,
    minimum_stuck_s: float = 8.0,
    maximum_actor_speed_mps: float = 0.05,
) -> dict[str, Any]:
    """Classify a narrowly evidenced upstream simulator/scenario deadlock.

    This audit is evaluator-side only.  It never changes the CARLA world and
    none of its actor truth is exposed to the Agent.  The current rule targets
    a reproducible CARLA 0.9.15 failure in ``FollowLeadingVehicleWithObstacle``:
    its lead vehicle can overshoot into the bicycle, leaving both physics
    actors interlocked while the official ``Obstalce clearing road`` behavior
    remains RUNNING indefinitely.

    A run is called invalid only when all of the following are observed over a
    sustained window: runtime-guard termination, the exact upstream scenario,
    the exact stuck behavior, two nearly stationary actors, and overlapping
    bounding-envelope evidence.  Other incomplete runs remain valid algorithm
    failures and are never retried under this rule.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return {
            "status": "unknown",
            "classification": None,
            "reason": "scenario_evidence_missing",
            "evidence_path": str(source),
        }
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            return {
                "status": "unknown",
                "classification": None,
                "reason": "scenario_evidence_malformed",
                "evidence_path": str(source),
            }
    termination = next(
        (row for row in reversed(rows) if row.get("event") == "termination"),
        {},
    )
    if termination.get("reason") != "max_sim_duration":
        return {
            "status": "valid",
            "classification": None,
            "reason": "no_known_simulator_deadlock",
            "evidence_path": str(source),
        }
    if expected_scenario_type != "FollowLeadingVehicleWithObstacle":
        return {
            "status": "valid",
            "classification": None,
            "reason": "no_known_simulator_deadlock",
            "evidence_path": str(source),
        }

    matching = []
    for row in rows:
        if row.get("event") != "tick":
            continue
        elapsed_s = _finite(row.get("elapsed_s"))
        for scenario in row.get("scenarios", ()) or ():
            if scenario.get("class") != expected_scenario_type:
                continue
            branch = _find_tree_node(
                scenario.get("behavior"), "Obstalce clearing road"
            )
            actors = [
                actor for actor in scenario.get("actors", ()) or ()
                if actor.get("available") is True
            ]
            if branch is None or branch.get("status") != "RUNNING" or len(actors) < 2:
                continue
            lead, obstacle = actors[:2]
            lead_speed = _finite(lead.get("speed_mps"))
            obstacle_speed = _finite(obstacle.get("speed_mps"))
            lead_location = lead.get("location_m")
            obstacle_location = obstacle.get("location_m")
            lead_extent = lead.get("bounding_extent_m")
            obstacle_extent = obstacle.get("bounding_extent_m")
            if any(
                value is None
                for value in (elapsed_s, lead_speed, obstacle_speed)
            ):
                continue
            if lead_speed > maximum_actor_speed_mps or obstacle_speed > maximum_actor_speed_mps:
                continue
            if not all(
                isinstance(value, list) and len(value) >= 2
                for value in (lead_location, obstacle_location, lead_extent, obstacle_extent)
            ):
                continue
            center_distance_xy = math.hypot(
                float(lead_location[0]) - float(obstacle_location[0]),
                float(lead_location[1]) - float(obstacle_location[1]),
            )
            # This envelope is intentionally conservative because the actor
            # yaw is not needed for the diagnostic: a center separation below
            # the sum of both longitudinal half-extents is unambiguously too
            # small for the official lead/obstacle pair to clear one another.
            overlap_limit_m = float(lead_extent[0]) + float(obstacle_extent[0])
            if center_distance_xy <= overlap_limit_m:
                matching.append(
                    {
                        "elapsed_s": elapsed_s,
                        "center_distance_xy_m": center_distance_xy,
                        "overlap_limit_m": overlap_limit_m,
                        "lead_speed_mps": lead_speed,
                        "obstacle_speed_mps": obstacle_speed,
                        "actor_ids": [lead.get("actor_id"), obstacle.get("actor_id")],
                    }
                )
    if matching:
        span_s = float(matching[-1]["elapsed_s"] - matching[0]["elapsed_s"])
        if span_s >= float(minimum_stuck_s):
            return {
                "status": "invalid",
                "classification": "invalid_simulator_scenario",
                "reason": "upstream_lead_obstacle_actor_interlock",
                "evidence_path": str(source),
                "matching_samples": len(matching),
                "stuck_span_s": span_s,
                "last_actor_ids": matching[-1]["actor_ids"],
                "last_center_distance_xy_m": matching[-1]["center_distance_xy_m"],
                "overlap_limit_m": matching[-1]["overlap_limit_m"],
                "maximum_actor_speed_mps": maximum_actor_speed_mps,
                "agent_truth_exposure": False,
            }
    return {
        "status": "valid",
        "classification": None,
        "reason": "incomplete_without_known_simulator_deadlock_evidence",
        "evidence_path": str(source),
    }


@dataclass
class ScenarioLifecycleRecorder:
    """Persist low-rate RouteScenario evidence without touching Agent inputs."""

    path: Path
    expected_scenario_type: str | None = None
    sample_period_s: float = 1.0
    _last_sample_s: float = field(default=-math.inf, init=False)
    _route_registered: bool = field(default=False, init=False)
    _initialized_types: set[str] = field(default_factory=set, init=False)
    _termination_reason: str | None = field(default=None, init=False)
    _samples: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def register_route(self, route_scenario: Any) -> None:
        configurations = []
        for config in list(getattr(route_scenario, "scenario_configurations", ()) or ()):
            configurations.append(
                {
                    "name": str(getattr(config, "name", "")),
                    "type": str(getattr(config, "type", "")),
                    "route_var_name": getattr(config, "route_var_name", None),
                }
            )
        initialized = list(getattr(route_scenario, "list_scenarios", ()) or ())
        self._initialized_types.update(type(item).__name__ for item in initialized)
        self._route_registered = True
        self._append(
            {
                "event": "route_registered",
                "configurations": configurations,
                "initialized_scenarios": [type(item).__name__ for item in initialized],
                "missing_scenarios": [
                    str(getattr(item, "type", ""))
                    for item in list(
                        getattr(route_scenario, "missing_scenario_configurations", ()) or ()
                    )
                ],
            }
        )

    def observe(self, route_scenario: Any, elapsed_s: float) -> None:
        if elapsed_s - self._last_sample_s < self.sample_period_s:
            return
        self._last_sample_s = elapsed_s
        initialized = list(getattr(route_scenario, "list_scenarios", ()) or ())
        self._initialized_types.update(type(item).__name__ for item in initialized)
        ego_vehicles = list(getattr(route_scenario, "ego_vehicles", ()) or ())
        ego_actor = ego_vehicles[0] if ego_vehicles else None
        scenario_actor_ids: set[int] = set()
        scenarios = []
        for scenario in initialized:
            actors = list(getattr(scenario, "other_actors", ()) or ())
            scenario_actor_ids.update(
                int(actor.id) for actor in actors if getattr(actor, "id", None) is not None
            )
            scenarios.append(
                {
                    "class": type(scenario).__name__,
                    "name": str(getattr(scenario, "name", "")),
                    "behavior": _tree_snapshot(getattr(scenario, "behavior_tree", None)),
                    "actors": [_actor_state(actor) for actor in actors],
                }
            )
        nearby_vehicles = []
        # This inventory is evaluator-only evidence.  It is intentionally not
        # returned by the Agent or included in any run_step input/control path.
        # Recording all nearby vehicles distinguishes an official scenario
        # actor from random Leaderboard BackgroundActivity traffic.
        try:
            world = getattr(route_scenario, "world", None)
            actors = world.get_actors().filter("vehicle.*") if world is not None else ()
            ego_location = ego_actor.get_location() if ego_actor is not None else None
            ego_id = int(ego_actor.id) if ego_actor is not None else None
            for actor in actors:
                actor_location = actor.get_location()
                distance = (
                    float(actor_location.distance(ego_location))
                    if ego_location is not None
                    else None
                )
                if distance is not None and distance > 100.0:
                    continue
                state = _actor_state(actor)
                actor_id = int(actor.id)
                state.update(
                    {
                        "distance_to_ego_m": distance,
                        "is_ego": actor_id == ego_id,
                        "is_scenario_actor": actor_id in scenario_actor_ids,
                    }
                )
                nearby_vehicles.append(state)
            nearby_vehicles.sort(
                key=lambda item: (
                    item.get("distance_to_ego_m") is None,
                    item.get("distance_to_ego_m") or 0.0,
                )
            )
        except Exception as exc:
            nearby_vehicles = [
                {"available": False, "error": f"{type(exc).__name__}: {exc}"}
            ]
        self._append(
            {
                "event": "tick",
                "elapsed_s": _finite(elapsed_s),
                "route_tree_status": _status_name(
                    getattr(getattr(route_scenario, "scenario_tree", None), "status", None)
                ),
                "ego": _actor_state(ego_actor),
                "scenarios": scenarios,
                "nearby_vehicles": nearby_vehicles,
                "missing_scenarios": [
                    str(getattr(item, "type", ""))
                    for item in list(
                        getattr(route_scenario, "missing_scenario_configurations", ()) or ()
                    )
                ],
            }
        )
        self._samples += 1

    def mark_termination(self, reason: str, elapsed_s: float | None = None) -> None:
        if self._termination_reason is not None:
            return
        self._termination_reason = str(reason)
        self._append(
            {
                "event": "termination",
                "reason": self._termination_reason,
                "elapsed_s": _finite(elapsed_s),
            }
        )

    def summary(self) -> dict[str, Any]:
        expected = self.expected_scenario_type
        return {
            "evidence_path": str(self.path),
            "route_registered": self._route_registered,
            "expected_scenario_type": expected,
            "initialized_scenario_types": sorted(self._initialized_types),
            "expected_scenario_initialized": (
                None if not expected else expected in self._initialized_types
            ),
            "samples": self._samples,
            "termination_reason": self._termination_reason,
            "validity": audit_scenario_evidence(
                self.path, expected_scenario_type=expected
            ),
        }
