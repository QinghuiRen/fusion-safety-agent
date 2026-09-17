"""Execute the declarative Fusion Safety product regression matrix."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = PROJECT_ROOT / "configs/evaluation/product_regression_matrix.json"
EVALUATION_ROOT = PROJECT_ROOT / "out/evaluation/product"
LEGACY_EVALUATION_ROOT = PROJECT_ROOT / "out/stage6_20"
ACCEPTANCE_SUMMARY_FILENAME = "fusion_safety_acceptance_summary.json"
REGRESSION_REPORT_FILENAME = "fusion_safety_regression_report.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    """Python 3.8-compatible containment check for resume-path validation."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _implementation_fingerprint() -> str:
    """Hash product code/configs so resume cannot cross a code revision."""

    sources = set()
    for package in (
        "fusion_tracking_agent",
        "fusion_tracking_carla",
        "fusion_tracking_bevfusion",
        "fusion_tracking_core",
    ):
        sources.update((PROJECT_ROOT / package).rglob("*.py"))
    sources.update((PROJECT_ROOT / "configs").rglob("*.json"))
    sources.update((PROJECT_ROOT / "configs").rglob("*.xml"))
    sources.update({
        PROJECT_ROOT / "fusion_safety_agent.py",
        PROJECT_ROOT / "tools/run_product_acceptance.py",
        Path(__file__).resolve(),
    })
    digest = hashlib.sha256()
    for source in sorted(sources, key=lambda path: path.as_posix().lower()):
        if not source.is_file():
            continue
        relative = source.resolve().relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


def _new_output(before: set[Path]) -> Path | None:
    after = set(EVALUATION_ROOT.glob("acceptance_*"))
    created = sorted(after - before, key=lambda item: item.stat().st_mtime)
    return created[-1] if created else None


def _run_case(command: list[str], timeout_s: float) -> tuple[int, bool]:
    """Run one evaluator and terminate only its process tree on timeout."""

    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, creationflags=creationflags)
    try:
        return int(process.wait(timeout=float(timeout_s))), False
    except subprocess.TimeoutExpired:
        print(
            f"[回归超时] pid={process.pid} 超过 {timeout_s:.1f}s；"
            "终止本次 evaluator 进程树",
            flush=True,
        )
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10.0,
                )
            except subprocess.TimeoutExpired:
                # A wedged UE process can block taskkill itself.  Preserve the
                # matrix deadline and fall through to the direct child kill;
                # the next product doctor/cold launch then reports or removes
                # any detached simulator process explicitly.
                print(
                    "[回归清理警告] evaluator 进程树终止超过10s，"
                    "改为结束直接子进程",
                    flush=True,
                )
                process.kill()
        else:
            process.terminate()
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
        code = process.returncode if process.returncode is not None else -9
        return int(code), True


def _case_contract(case) -> dict:
    """Freeze every declarative field that makes a result reusable."""

    return {name: getattr(case, name) for name in case.__dataclass_fields__}


def _resume_record_matches_case(
    record: dict, case, implementation_fingerprint: str
) -> bool:
    """Reject a passed checkpoint when its case definition has changed."""

    if record.get("implementation_fingerprint") != implementation_fingerprint:
        return False
    stored = record.get("case_contract")
    if stored is not None:
        return stored == _case_contract(case)

    # Compatibility for rollout-window checkpoints that already carry the
    # source fingerprint but predate strict case_contract serialization.
    # Older unfingerprinted files are deliberately rejected above.
    run_config = (record.get("acceptance", {}).get("run_config", {}) or {})
    overrides = run_config.get("overrides", {}) or {}
    expected_overrides = {
        "supervisor_mode": case.supervisor_mode,
        "fault_profile": case.fault_profile,
        "detector_stride_frames": case.detector_stride_frames,
        "vru_safety_monitor_mode": case.vru_safety_monitor_mode,
        "vru_detector_stride_frames": case.vru_detector_stride_frames,
        "vru_detector_score_threshold": case.vru_detector_score_threshold,
        "bevfusion_packet_transport": case.packet_transport,
    }
    if any(
        overrides.get(key) != value
        for key, value in expected_overrides.items()
    ):
        return False
    return (
        run_config.get("acceptance_contract") == case.acceptance_contract
        and run_config.get("expected_intervention") == case.expected_intervention
        and run_config.get("expected_scenario_type") == case.expected_scenario_type
        and float(run_config.get("max_sim_duration_s", -1.0))
        == float(case.max_sim_duration_s)
        and run_config.get("background_traffic") == case.background_traffic
    )


def _load_resume_records(
    progress_path: Path,
    cases,
    implementation_fingerprint: str,
) -> tuple[Path, list[dict]]:
    """Load matching passed records from an interrupted matrix directory."""

    source = progress_path.expanduser().resolve()
    allowed_roots = (EVALUATION_ROOT.resolve(), LEGACY_EVALUATION_ROOT.resolve())
    if source.name != "regression_progress.json" or not source.is_file():
        raise ValueError(
            "--resume-progress must name an existing regression_progress.json"
        )
    if not any(_is_relative_to(source, root) for root in allowed_roots):
        raise ValueError(
            "resume progress must stay inside out/evaluation/product "
            "or the historical out/stage6_20 directory"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("resume progress must contain a JSON record list")
    case_by_id = {case.case_id: case for case in cases}
    loaded = []
    seen = set()
    for record in payload:
        case_id = record.get("case_id")
        if case_id in seen:
            raise ValueError(f"duplicate resume case_id: {case_id}")
        seen.add(case_id)
        case = case_by_id.get(case_id)
        if case is None or not bool(record.get("pass", False)):
            continue
        if not _resume_record_matches_case(
            record, case, implementation_fingerprint
        ):
            raise ValueError(
                f"resume code/case contract changed for {case_id}; "
                "start a new matrix run"
            )
        loaded.append(record)
    order = {case.case_id: index for index, case in enumerate(cases)}
    loaded.sort(key=lambda record: order[record["case_id"]])
    return source.parent, loaded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fusion Safety automated regression matrix")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--packet-transport",
        choices=(
            "windows_mount", "wsl_native", "wsl_inline_images",
            "wsl_stdin_packet",
        ),
        default=None,
        help=(
            "Override only the BEVFusion transport for controlled A/B runs; "
            "the source matrix is never modified."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument(
        "--resume-progress",
        type=Path,
        default=None,
        help=(
            "Resume an interrupted matrix from regression_progress.json; "
            "only passed records with an identical case contract are reused."
        ),
    )
    args = parser.parse_args(argv)

    project_value = str(PROJECT_ROOT)
    if project_value not in sys.path:
        sys.path.insert(0, project_value)
    from fusion_tracking_carla.runtime_launcher import handoff_if_needed

    handoff_if_needed(Path(__file__), PROJECT_ROOT)
    from fusion_tracking_agent.regression import (
        RegressionMatrix,
        evaluate_freshness_gate,
    )

    matrix = RegressionMatrix.load(args.matrix)
    cases = matrix.select(args.cases)
    if args.packet_transport is not None:
        cases = tuple(
            replace(case, packet_transport=args.packet_transport)
            for case in cases
        )
    implementation_fingerprint = _implementation_fingerprint()
    print(f"[回归矩阵] {matrix.name} | cases={len(cases)}", flush=True)
    for case in cases:
        print(
            f"  {case.case_id}: mode={case.supervisor_mode} "
            f"fault={case.fault_profile} label={case.expected_intervention} "
            f"transport={case.packet_transport}",
            flush=True,
        )
    if args.dry_run:
        for case in cases:
            print("DRY-RUN " + subprocess.list2cmdline(case.command(sys.executable)))
        return 0

    if args.resume_progress is not None:
        output_dir, records = _load_resume_records(
            args.resume_progress, cases, implementation_fingerprint
        )
        print(
            f"[回归断点续跑] reused={len(records)}/{len(cases)} | "
            f"output={output_dir}",
            flush=True,
        )
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = EVALUATION_ROOT / f"regression_{stamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        records = []
    resumed_case_ids = {record["case_id"] for record in records}
    for index, case in enumerate(cases, start=1):
        if case.case_id in resumed_case_ids:
            print(
                f"\n[回归 {index}/{len(cases)}] {case.case_id} | "
                "已通过，断点复用",
                flush=True,
            )
            continue
        print(f"\n[回归 {index}/{len(cases)}] {case.case_id}", flush=True)
        started = time.perf_counter()
        attempts = []
        maximum_attempts = 1 + int(case.max_invalid_scenario_retries)
        return_code = -1
        timed_out = False
        source = None
        summary = {}
        health_inhibit_observed = False
        health_gate = False
        freshness_gate = {}
        attempt_pass = False
        for attempt_index in range(1, maximum_attempts + 1):
            if attempt_index > 1:
                print(
                    f"[场景有效性重试 {attempt_index}/{maximum_attempts}] "
                    f"{case.case_id} | 保留上一无效尝试，重新冷启动评测",
                    flush=True,
                )
            before = set(EVALUATION_ROOT.glob("acceptance_*"))
            attempt_started = time.perf_counter()
            return_code, timed_out = _run_case(
                case.command(sys.executable), timeout_s=case.wall_timeout_s
            )
            source = _new_output(before)
            summary = {}
            if source is not None:
                summary_path = source / ACCEPTANCE_SUMMARY_FILENAME
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
            inhibited_frames = int(
                summary.get("sensor_health", {}).get("control_inhibited_frames", 0)
            )
            health_inhibit_observed = inhibited_frames > 0
            health_gate = (
                case.expected_health_inhibit is None
                or health_inhibit_observed is case.expected_health_inhibit
            )
            freshness_gate = evaluate_freshness_gate(
                summary,
                max_uninhibited_stale_frames=(
                    case.max_uninhibited_stale_frames
                ),
                max_uninhibited_missing_frames=(
                    case.max_uninhibited_missing_frames
                ),
            )
            attempt_pass = (
                bool(summary.get("pass", False))
                and return_code == 0
                and not timed_out
                and health_gate
                and freshness_gate["pass"]
            )
            validity = (
                summary.get("scenario_evidence", {}).get("validity", {}) or {}
            )
            # Never retry a possible algorithm failure.  The retry gate needs
            # explicit evaluator-side interlock evidence plus zero collision
            # and zero perception exceptions from the Agent run.
            invalid_scenario = (
                validity.get("status") == "invalid"
                and validity.get("classification")
                == "invalid_simulator_scenario"
                and int(summary.get("collisions", 1)) == 0
                and int(summary.get("perception_errors", 1)) == 0
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "return_code": return_code,
                    "timed_out": timed_out,
                    "pass": attempt_pass,
                    "freshness_gate": freshness_gate,
                    "invalid_simulator_scenario": invalid_scenario,
                    "scenario_validity": validity,
                    "elapsed_wall_s": time.perf_counter() - attempt_started,
                    "source_output": str(source) if source else None,
                }
            )
            if invalid_scenario and attempt_index < maximum_attempts:
                print(
                    "[场景有效性] 本次不计入算法成败 | "
                    f"reason={validity.get('reason')} | "
                    f"stuck={validity.get('stuck_span_s', 0):.1f}s | "
                    "即将有限重试",
                    flush=True,
                )
                continue
            break
        inhibited_frames = int(
            summary.get("sensor_health", {}).get("control_inhibited_frames", 0)
        )
        record = {
            "case_id": case.case_id,
            "case_contract": _case_contract(case),
            "implementation_fingerprint": implementation_fingerprint,
            "description": case.description,
            "required": case.required,
            "return_code": return_code,
            "timed_out": timed_out,
            "wall_timeout_s": case.wall_timeout_s,
            "pass": attempt_pass,
            "max_invalid_scenario_retries": case.max_invalid_scenario_retries,
            "attempt_count": len(attempts),
            "invalid_scenario_attempts": sum(
                attempt["invalid_simulator_scenario"] for attempt in attempts
            ),
            "attempts": attempts,
            "health_inhibit_expected": case.expected_health_inhibit,
            "health_inhibit_observed": health_inhibit_observed,
            "health_inhibited_frames": inhibited_frames,
            "health_gate_pass": health_gate,
            "freshness_gate": freshness_gate,
            "elapsed_wall_s": time.perf_counter() - started,
            "source_output": str(source) if source else None,
            "acceptance": summary,
        }
        records.append(record)
        (output_dir / "regression_progress.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[回归结果] {case.case_id} | pass={record['pass']} | "
            f"code={return_code} | timed_out={timed_out} | "
            f"attempts={len(attempts)} | "
            f"invalid={record['invalid_scenario_attempts']}",
            flush=True,
        )
        if args.stop_on_failure and not record["pass"]:
            break

    required = [record for record in records if record["required"]]
    report = {
        "schema": "fusion_tracking.regression_report.v1",
        "matrix": matrix.name,
        "selected_cases": [case.case_id for case in cases],
        "executed_cases": len(records),
        "required_passed": sum(record["pass"] for record in required),
        "required_total": len(required),
        "pass": len(records) == len(cases) and all(record["pass"] for record in required),
        "resumed_cases": sorted(resumed_case_ids),
        "resumed_from": (
            str(args.resume_progress.expanduser().resolve())
            if args.resume_progress is not None else None
        ),
        "records": records,
    }
    destination = output_dir / REGRESSION_REPORT_FILENAME
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[回归完成] pass={report['pass']} | report={destination}", flush=True)
    return 0 if report["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
