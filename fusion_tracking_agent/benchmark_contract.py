"""Unified Leaderboard/SafeBench-style episode metrics.

The product runtime does not import either benchmark. This module translates
their JSON outputs into a small stable contract so normal, fault-injection and
Active-control runs can be compared with the same definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeOutcome:
    scenario_id: str
    collision_count: int
    route_completion_pct: float
    offroad_fraction: float
    mean_actionable_risk: float = 0.0
    intervention_events: int = 0
    # ``None`` means the episode has no positive/negative evaluation label;
    # it must not silently improve the false-intervention metric.
    false_intervention_events: int | None = None
    driving_score: float = 0.0
    duration_game_s: float = 0.0
    duration_system_s: float = 0.0
    applied_intervention_frames: int = 0
    applied_intervention_events: int | None = None
    false_applied_intervention_events: int | None = None
    expected_intervention: bool | None = None


COLLISION_KEYS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
)


def _event_count(value: Any) -> int:
    """Count Leaderboard per-route event lists without assuming global format."""

    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    return int(bool(value))


def episode_from_leaderboard(
    result_path: str | Path,
    agent_summary_path: str | Path | None = None,
    *,
    expected_intervention: bool | None = None,
) -> tuple[EpisodeOutcome, dict]:
    """Translate one official Leaderboard record and our Agent summary.

    ``expected_intervention`` is scenario metadata used only for evaluation.
    It is never passed to the online algorithm. On a negative scenario every
    intervention event is counted as a false positive; on a positive scenario
    false-positive attribution is left at zero.
    """

    result_source = Path(result_path).expanduser().resolve()
    payload = json.loads(result_source.read_text(encoding="utf-8"))
    records = payload.get("_checkpoint", {}).get("records", [])
    if not records:
        raise ValueError(f"Leaderboard result contains no route records: {result_source}")
    if len(records) != 1:
        raise ValueError(
            "episode_from_leaderboard expects exactly one route record; "
            f"received {len(records)}"
        )
    record = records[0]
    infractions = record.get("infractions", {}) or {}
    scores = record.get("scores", {}) or {}
    meta = record.get("meta", {}) or {}

    agent_summary: dict = {}
    summary_source = None
    if agent_summary_path is not None:
        summary_source = Path(agent_summary_path).expanduser().resolve()
        agent_summary = json.loads(summary_source.read_text(encoding="utf-8"))
    supervisor = agent_summary.get("supervisor", {}) or {}
    combined_control = agent_summary.get("combined_control", {}) or {}
    intervention_events = int(supervisor.get("intervention_events", 0))
    applied_frames = int(combined_control.get(
        "applied_frames", supervisor.get("applied_frames", 0)
    ))
    applied_events_raw = combined_control.get(
        "applied_intervention_events",
        supervisor.get("applied_intervention_events"),
    )
    applied_events = (
        None if applied_events_raw is None else int(applied_events_raw)
    )
    collision_count = sum(_event_count(infractions.get(key, [])) for key in COLLISION_KEYS)
    outside_events = _event_count(infractions.get("outside_route_lanes", []))
    route_completion = float(scores.get("score_route", 0.0))
    outcome = EpisodeOutcome(
        scenario_id=str(record.get("route_id", result_source.stem)),
        collision_count=collision_count,
        route_completion_pct=route_completion,
        # SafeBench exposes distance outside the road. Leaderboard's route JSON
        # exposes event text, so record episode occurrence (0/1), not fake metres.
        offroad_fraction=float(outside_events > 0),
        mean_actionable_risk=float(supervisor.get("mean_actionable_collision_risk", 0.0)),
        intervention_events=intervention_events,
        false_intervention_events=(
            intervention_events
            if expected_intervention is False
            else (0 if expected_intervention is True else None)
        ),
        driving_score=float(scores.get("score_composed", 0.0)),
        duration_game_s=float(meta.get("duration_game", 0.0)),
        duration_system_s=float(meta.get("duration_system", 0.0)),
        applied_intervention_frames=applied_frames,
        applied_intervention_events=applied_events,
        false_applied_intervention_events=(
            applied_events
            if expected_intervention is False
            else (0 if expected_intervention is True else None)
        ),
        expected_intervention=expected_intervention,
    )
    details = {
        "entry_status": payload.get("entry_status"),
        "route_status": record.get("status"),
        "result_path": str(result_source),
        "agent_summary_path": str(summary_source) if summary_source else None,
        "expected_intervention": expected_intervention,
        "requested_intervention_events": intervention_events,
        "applied_intervention_frames": applied_frames,
        "applied_intervention_events": applied_events,
        "collision_events": collision_count,
        "outside_route_lane_events": outside_events,
        "min_speed_infractions": _event_count(infractions.get("min_speed_infractions", [])),
        "infraction_penalty": float(scores.get("score_penalty", 0.0)),
        "real_time_factor": (
            float(meta.get("duration_game", 0.0)) / float(meta.get("duration_system", 1.0))
            if float(meta.get("duration_system", 0.0)) > 0.0
            else 0.0
        ),
        "fault_injection": agent_summary.get("fault_injection", {}),
        "frontend": agent_summary.get("frontend"),
        "perception_runtime": agent_summary.get("perception_runtime", {}),
        "gpu_inference_arbiter": agent_summary.get(
            "gpu_inference_arbiter", {}
        ),
        "sensor_profile": agent_summary.get("sensor_profile"),
    }
    return outcome, details


def aggregate_safebench_metrics(outcomes: list[EpisodeOutcome]) -> dict:
    """Aggregate SafeBench-compatible metrics plus project safety fields."""

    if not outcomes:
        raise ValueError("At least one episode is required")
    collision_rate = float(np.mean([item.collision_count > 0 for item in outcomes]))
    route_completion = float(np.mean([
        np.clip(item.route_completion_pct, 0.0, 100.0) / 100.0 for item in outcomes
    ]))
    offroad_rate = float(np.mean([
        np.clip(item.offroad_fraction, 0.0, 1.0) for item in outcomes
    ]))
    intervention_total = sum(item.intervention_events for item in outcomes)
    labeled_requested = [
        item for item in outcomes if item.false_intervention_events is not None
    ]
    false_total = sum(
        int(item.false_intervention_events or 0) for item in labeled_requested
    )
    labeled_requested_total = sum(
        item.intervention_events for item in labeled_requested
    )
    applied_frames_total = sum(item.applied_intervention_frames for item in outcomes)
    applied_event_items = [
        item for item in outcomes if item.applied_intervention_events is not None
    ]
    applied_events_total = sum(
        int(item.applied_intervention_events or 0) for item in applied_event_items
    )
    false_applied_items = [
        item for item in outcomes
        if item.false_applied_intervention_events is not None
    ]
    false_applied_total = sum(
        int(item.false_applied_intervention_events or 0)
        for item in false_applied_items
    )
    labeled_applied_total = sum(
        int(item.applied_intervention_events or 0) for item in false_applied_items
    )
    return {
        "episodes": len(outcomes),
        "collision_rate": collision_rate,
        "route_completion": route_completion,
        "incomplete_route": 1.0 - route_completion,
        "offroad_rate": offroad_rate,
        "success_rate": float(np.mean([
            item.collision_count == 0
            and item.route_completion_pct >= 95.0
            and item.offroad_fraction <= 0.0
            for item in outcomes
        ])),
        "mean_actionable_risk": float(np.mean([
            item.mean_actionable_risk for item in outcomes
        ])),
        "intervention_events": int(intervention_total),
        "requested_intervention_events": int(intervention_total),
        "applied_intervention_frames": int(applied_frames_total),
        "applied_intervention_events": (
            int(applied_events_total) if applied_event_items else None
        ),
        "false_intervention_events": (
            int(false_total) if labeled_requested else None
        ),
        "false_intervention_labeled_episodes": len(labeled_requested),
        "false_intervention_ratio": (
            float(false_total / labeled_requested_total)
            if labeled_requested_total
            else (0.0 if labeled_requested else None)
        ),
        "false_applied_intervention_events": (
            int(false_applied_total) if false_applied_items else None
        ),
        "false_applied_intervention_ratio": (
            float(false_applied_total / labeled_applied_total)
            if labeled_applied_total
            else (0.0 if false_applied_items else None)
        ),
        "mean_driving_score": float(np.mean([item.driving_score for item in outcomes])),
        "mean_real_time_factor": float(np.mean([
            item.duration_game_s / item.duration_system_s
            if item.duration_system_s > 0.0 else 0.0
            for item in outcomes
        ])),
    }
