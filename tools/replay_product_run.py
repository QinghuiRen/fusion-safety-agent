"""Render a Fusion Safety Agent run without work in the live control loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion_tracking_agent.replay import write_replay


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline Fusion Safety BEV replay")
    parser.add_argument("source", type=Path, help="Agent run directory or frame JSONL")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    summary = write_replay(
        args.source,
        args.output_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        write_video=not args.no_video,
    )
    print("[Fusion Safety offline replay complete]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
