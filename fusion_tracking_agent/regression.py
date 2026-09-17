"""Declarative regression matrix for the deployable Fusion Safety Agent.

Scenario labels belong to evaluation metadata and are deliberately kept out of
the Agent configuration passed to ``run_step``.  This prevents ground-truth
leakage while still making false interventions measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluate_freshness_gate(
    summary: dict,
    *,
    max_uninhibited_stale_frames: int = 0,
    max_uninhibited_missing_frames: int = 0,
) -> dict:
    """Evaluate whether perception freshness degraded outside a safety inhibit.

    LiDAR loss deliberately inhibits control, so stale results during those
    protected frames are not treated as an unsafe realtime regression.  A
    camera/Radar fault, nominal run, or any stale span longer than the inhibit
    window remains a hard regression failure.
    """

    freshness = (
        summary.get("online_timing", {}).get("freshness", {}) or {}
    )
    health = summary.get("sensor_health", {}) or {}
    inhibited_frames = max(0, int(health.get("control_inhibited_frames", 0)))
    stale_frames = max(0, int(freshness.get("stale_age_frames", 0)))
    missing_frames = max(0, int(freshness.get("missing_age_frames", 0)))
    stale_budget = inhibited_frames + max(
        0, int(max_uninhibited_stale_frames)
    )
    missing_budget = inhibited_frames + max(
        0, int(max_uninhibited_missing_frames)
    )
    return {
        "pass": stale_frames <= stale_budget and missing_frames <= missing_budget,
        "stale_frames": stale_frames,
        "missing_frames": missing_frames,
        "protected_inhibit_frames": inhibited_frames,
        "stale_budget_frames": stale_budget,
        "missing_budget_frames": missing_budget,
    }


@dataclass(frozen=True)
class RegressionCase:
    case_id: str
    description: str
    runner: str
    routes: str
    supervisor_mode: str
    fault_profile: str
    acceptance_contract: str
    expected_intervention: bool | None
    expected_health_inhibit: bool | None = None
    # Product regression is realtime-aware: nominal/camera/Radar cases permit
    # no stale or missing control frames.  LiDAR-fault staleness is permitted
    # only while the sensor-health gate is actively inhibiting control.
    max_uninhibited_stale_frames: int = 0
    max_uninhibited_missing_frames: int = 0
    # Freeze the promoted product cadence in every declarative case unless a
    # research matrix explicitly overrides it.
    detector_stride_frames: int = 8
    # Match the deployable product default: collect/calibrate VRU evidence in
    # Shadow without changing longitudinal control.  Positive VRU acceptance
    # cases must still opt in to Active explicitly in the matrix.
    vru_safety_monitor_mode: str = "shadow"
    vru_detector_stride_frames: int = 4
    vru_detector_score_threshold: float = 0.30
    packet_transport: str = "wsl_stdin_packet"
    expected_scenario_type: str | None = None
    background_traffic: str = "standard"
    launch_server: bool = True
    restart_server_if_map_mismatch: bool = True
    max_sim_duration_s: float = 45.0
    wall_timeout_s: float = 180.0
    max_invalid_scenario_retries: int = 0
    required: bool = True

    @classmethod
    def from_dict(cls, payload: dict) -> "RegressionCase":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown regression case fields: {unknown}")
        case = cls(**payload)
        case.validate()
        return case

    def validate(self) -> None:
        if not self.case_id or any(ch.isspace() for ch in self.case_id):
            raise ValueError("regression case_id must be non-empty without whitespace")
        if self.supervisor_mode not in {"shadow", "active"}:
            raise ValueError(f"{self.case_id}: invalid supervisor_mode")
        if self.fault_profile not in {
            "none", "front_camera_dropout", "lidar_dropout",
            "radar_dropout", "mixed"
        }:
            raise ValueError(f"{self.case_id}: invalid fault_profile")
        if self.acceptance_contract not in {
            "smoke", "active_hazard", "normal_negative"
        }:
            raise ValueError(f"{self.case_id}: invalid acceptance_contract")
        if self.acceptance_contract == "active_hazard" and self.expected_intervention is not True:
            raise ValueError(f"{self.case_id}: active_hazard must have a positive label")
        if self.acceptance_contract == "normal_negative" and self.expected_intervention is not False:
            raise ValueError(f"{self.case_id}: normal_negative must have a negative label")
        if self.packet_transport not in {
            "windows_mount", "wsl_native", "wsl_inline_images",
            "wsl_stdin_packet",
        }:
            raise ValueError(f"{self.case_id}: invalid packet_transport")
        if int(self.detector_stride_frames) < 1:
            raise ValueError(f"{self.case_id}: detector_stride_frames must be positive")
        if int(self.max_uninhibited_stale_frames) < 0:
            raise ValueError(
                f"{self.case_id}: max_uninhibited_stale_frames must be non-negative"
            )
        if int(self.max_uninhibited_missing_frames) < 0:
            raise ValueError(
                f"{self.case_id}: max_uninhibited_missing_frames must be non-negative"
            )
        if self.vru_safety_monitor_mode not in {"off", "shadow", "active"}:
            raise ValueError(f"{self.case_id}: invalid vru_safety_monitor_mode")
        if int(self.vru_detector_stride_frames) < 1:
            raise ValueError(
                f"{self.case_id}: vru_detector_stride_frames must be positive"
            )
        if not 0.0 < float(self.vru_detector_score_threshold) <= 1.0:
            raise ValueError(
                f"{self.case_id}: vru_detector_score_threshold must be in (0,1]"
            )
        if float(self.max_sim_duration_s) <= 0:
            raise ValueError(f"{self.case_id}: max_sim_duration_s must be positive")
        if float(self.wall_timeout_s) <= float(self.max_sim_duration_s):
            raise ValueError(
                f"{self.case_id}: wall_timeout_s must exceed max_sim_duration_s"
            )
        if not 0 <= int(self.max_invalid_scenario_retries) <= 3:
            raise ValueError(
                f"{self.case_id}: max_invalid_scenario_retries must be in [0,3]"
            )
        if self.expected_scenario_type and self.expected_intervention is not True:
            raise ValueError(
                f"{self.case_id}: expected_scenario_type is only valid for a positive label"
            )
        if self.background_traffic not in {"standard", "disabled"}:
            raise ValueError(f"{self.case_id}: invalid background_traffic")
        if self.background_traffic == "disabled" and self.acceptance_contract not in {
            "active_hazard",
            "normal_negative",
        }:
            raise ValueError(
                f"{self.case_id}: background isolation is reserved for labeled controlled tests"
            )
        for field_name in ("runner", "routes"):
            source = (PROJECT_ROOT / getattr(self, field_name)).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"{self.case_id}: missing {field_name}: {source}")

    def command(self, python_executable: str) -> list[str]:
        expected = (
            "unknown" if self.expected_intervention is None
            else ("yes" if self.expected_intervention else "no")
        )
        command = [
            python_executable,
            str((PROJECT_ROOT / self.runner).resolve()),
            "--routes", str((PROJECT_ROOT / self.routes).resolve()),
            "--supervisor-mode", self.supervisor_mode,
            "--fault-profile", self.fault_profile,
            "--acceptance-contract", self.acceptance_contract,
            "--expected-intervention", expected,
            "--detector-stride-frames", str(self.detector_stride_frames),
            "--vru-safety-monitor-mode", self.vru_safety_monitor_mode,
            "--vru-detector-stride-frames", str(self.vru_detector_stride_frames),
            "--vru-detector-score-threshold", str(
                self.vru_detector_score_threshold
            ),
            "--bevfusion-packet-transport", self.packet_transport,
            "--max-sim-duration-s", str(self.max_sim_duration_s),
            "--background-traffic", self.background_traffic,
        ]
        if self.launch_server:
            command.append("--launch-server")
            if self.restart_server_if_map_mismatch:
                command.append("--restart-server-if-map-mismatch")
        if self.expected_scenario_type:
            command.extend(["--expected-scenario-type", self.expected_scenario_type])
        return command


@dataclass(frozen=True)
class RegressionMatrix:
    schema: str
    name: str
    cases: tuple[RegressionCase, ...]

    @classmethod
    def load(cls, path: str | Path) -> "RegressionMatrix":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        schema = str(payload.get("schema", ""))
        if schema != "fusion_tracking.regression_matrix.v1":
            raise ValueError(f"unsupported regression matrix schema: {schema}")
        cases = tuple(RegressionCase.from_dict(item) for item in payload.get("cases", []))
        if not cases:
            raise ValueError("regression matrix must contain at least one case")
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("regression case_id values must be unique")
        return cls(schema=schema, name=str(payload.get("name", source.stem)), cases=cases)

    def select(self, case_ids: Iterable[str] | None = None) -> tuple[RegressionCase, ...]:
        requested = set(case_ids or ())
        if not requested:
            return self.cases
        available = {case.case_id for case in self.cases}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"unknown regression cases: {unknown}")
        return tuple(case for case in self.cases if case.case_id in requested)
