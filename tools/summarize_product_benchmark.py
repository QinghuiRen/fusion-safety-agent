"""Aggregate Fusion Safety Leaderboard runs with SafeBench-style metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_tracking_agent.benchmark_contract import (  # noqa: E402
    aggregate_safebench_metrics,
    episode_from_leaderboard,
)
from fusion_tracking_agent.output_contract import resolve_summary  # noqa: E402


EVALUATION_ROOT = PROJECT_ROOT / "out/evaluation/product"
ACCEPTANCE_SUMMARY_FILENAMES = (
    "fusion_safety_acceptance_summary.json",
    "stage6_20_leaderboard_smoke_summary.json",
)


def _resolve_source(source: Path) -> tuple[Path, Path | None]:
    source = source.expanduser().resolve()
    result = source if source.is_file() else source / "leaderboard_results.json"
    if not result.exists():
        raise FileNotFoundError(f"No leaderboard_results.json under {source}")

    summary = None
    if source.is_dir():
        acceptance = next(
            (
                source / name
                for name in ACCEPTANCE_SUMMARY_FILENAMES
                if (source / name).is_file()
            ),
            None,
        )
        if acceptance is not None:
            payload = json.loads(acceptance.read_text(encoding="utf-8"))
            agent_run = payload.get("agent_run")
            if agent_run:
                summary = resolve_summary(agent_run)
    return result, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fusion Safety Leaderboard + SafeBench-style aggregate report"
    )
    parser.add_argument("sources", nargs="+", type=Path, help="Leaderboard run directories")
    parser.add_argument(
        "--expected-intervention",
        choices=("unknown", "yes", "no"),
        default="unknown",
        help="Evaluation label only; never enters the online algorithm.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    expected = {"unknown": None, "yes": True, "no": False}[
        args.expected_intervention
    ]
    outcomes = []
    details = []
    for source in args.sources:
        result, summary = _resolve_source(source)
        outcome, episode_details = episode_from_leaderboard(
            result, summary, expected_intervention=expected
        )
        outcomes.append(outcome)
        details.append(episode_details)

    payload = {
        "contract": "fusion_safety_leaderboard_safebench_adapter_v3",
        "metric_provenance": {
            "collision_rate": "SafeBench episode-level collision semantics",
            "route_completion": "Leaderboard score_route normalized to [0,1]",
            "offroad_rate": "Leaderboard episode-level outside-route occurrence",
            "driving_score": "Leaderboard score_composed",
            "false_intervention": (
                "requested supervisor events on explicitly labeled negative scenarios; "
                "unknown labels remain null"
            ),
            "false_applied_intervention": (
                "actual control-modification events on explicitly labeled negative scenarios"
            ),
        },
        "aggregate": aggregate_safebench_metrics(outcomes),
        "episodes": [
            {"outcome": asdict(outcome), "details": detail}
            for outcome, detail in zip(outcomes, details)
        ],
    }
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = EVALUATION_ROOT / f"benchmark_summary_{stamp}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    aggregate = payload["aggregate"]
    print(
        "[Fusion Safety benchmark] "
        f"episodes={aggregate['episodes']} | success={aggregate['success_rate']:.1%} | "
        f"collision={aggregate['collision_rate']:.1%} | "
        f"route={aggregate['route_completion']:.1%} | "
        f"driving_score={aggregate['mean_driving_score']:.2f} | "
        f"RTF={aggregate['mean_real_time_factor']:.3f}",
        flush=True,
    )
    print(f"[output] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
