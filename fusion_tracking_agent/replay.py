"""根据结构化日志离线生成 BEV 轨迹、风险与控制回放，不占用在线闭环算力。"""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from .output_contract import (
    REPLAY_PREVIEW_FILENAME,
    REPLAY_SUMMARY_FILENAME,
    REPLAY_VIDEO_FILENAME,
    resolve_frame_log,
)


CANVAS_W = 1280
CANVAS_H = 720
BEV_W = 820
BEV_TOP = 50
BEV_BOTTOM = 570
FORWARD_M = 65.0
BACKWARD_M = 15.0
SIDE_M = 32.0


def load_records(source: str | Path, max_frames: int | None = None) -> list[dict]:
    path = resolve_frame_log(source)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_frames is not None and len(records) >= max_frames:
                break
    if not records:
        raise ValueError(f"Fusion Safety frame log is empty: {path}")
    return records


def _point(position_ego) -> tuple[int, int]:
    x, y = (float(value) for value in position_ego[:2])
    px = int(round(BEV_W / 2.0 - y / (2.0 * SIDE_M) * BEV_W))
    py = int(round(
        BEV_TOP + (FORWARD_M - x) / (FORWARD_M + BACKWARD_M)
        * (BEV_BOTTOM - BEV_TOP)
    ))
    return px, py


def _risk_color(risk: float) -> tuple[int, int, int]:
    if risk >= 80.0:
        return 40, 40, 245
    if risk >= 60.0:
        return 20, 130, 255
    if risk >= 30.0:
        return 20, 215, 235
    return 70, 220, 90


def _text(image, value, position, *, color=(220, 225, 230), scale=0.48, thick=1):
    cv2.putText(
        image, str(value), position, cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thick, cv2.LINE_AA,
    )


def _draw_bev(canvas: np.ndarray, record: dict) -> None:
    cv2.rectangle(canvas, (0, BEV_TOP), (BEV_W, BEV_BOTTOM), (22, 27, 33), -1)
    for x in range(-10, 70, 10):
        _, py = _point((x, 0))
        cv2.line(canvas, (0, py), (BEV_W, py), (49, 55, 63), 1)
        _text(canvas, f"{x}m", (8, py - 4), color=(125, 135, 145), scale=0.38)
    for y in range(-30, 31, 10):
        px, _ = _point((0, y))
        cv2.line(canvas, (px, BEV_TOP), (px, BEV_BOTTOM), (49, 55, 63), 1)
    center = _point((0, 0))
    cv2.rectangle(
        canvas, (center[0] - 10, center[1] - 20),
        (center[0] + 10, center[1] + 20), (255, 190, 30), -1,
    )
    cv2.arrowedLine(
        canvas, center, (center[0], center[1] - 34),
        (255, 230, 100), 2, tipLength=0.25,
    )

    for item in record.get("safety", []):
        position = np.asarray(item.get("position_ego_m", (0.0, 0.0)), dtype=float)
        velocity = np.asarray(item.get("velocity_ego_mps", (0.0, 0.0)), dtype=float)
        risk = float(item.get("risk_score", 0.0))
        point = _point(position)
        if not (0 <= point[0] <= BEV_W and BEV_TOP <= point[1] <= BEV_BOTTOM):
            continue
        color = _risk_color(risk)
        cv2.circle(canvas, point, 8, color, 2)
        _text(
            canvas,
            f"ID {item.get('track_id')}  R {risk:.0f}",
            (point[0] + 10, point[1] - 8),
            color=color,
            scale=0.42,
        )
        # This solid line is target self-motion, not relative ego collision
        # motion.  Static detections remain visible but intentionally get no
        # misleading trajectory line.
        if float(np.linalg.norm(velocity)) >= 0.75:
            times = np.arange(0.5, 3.01, 0.5)
            future = position[None, :] + times[:, None] * velocity[None, :]
            points = np.asarray([_point(value) for value in future], dtype=np.int32)
            visible = [
                value for value in points
                if 0 <= value[0] <= BEV_W and BEV_TOP <= value[1] <= BEV_BOTTOM
            ]
            if visible:
                polyline = np.asarray([point, *visible], dtype=np.int32)
                cv2.polylines(canvas, [polyline], False, color, 2, cv2.LINE_AA)

    _text(canvas, "BEV: solid path = moving target self-motion", (14, 30), scale=0.58)


def _draw_status(canvas: np.ndarray, record: dict) -> None:
    x = BEV_W + 20
    supervisor = record.get("supervisor", {})
    applied = record.get("applied_control", {})
    health = record.get("sensor_health", {})
    aeb = record.get("autoware_aeb_shadow", {})
    overall = str(health.get("overall_state", "unknown"))
    health_color = {
        "healthy": (70, 220, 90),
        "degraded": (20, 210, 235),
        "unsafe": (30, 50, 245),
    }.get(overall, (180, 180, 180))
    _text(canvas, "FUSION SAFETY AGENT REPLAY", (x, 35), color=(75, 220, 235), scale=0.60)
    _text(canvas, f"frame      {record.get('frame')}", (x, 75))
    _text(canvas, f"sim time   {float(record.get('timestamp_s', 0.0)):.2f} s", (x, 98))
    _text(canvas, f"frontend   {record.get('frontend', '-')}", (x, 121), scale=0.42)
    _text(canvas, f"mode       {record.get('supervisor_mode', '-').upper()}", (x, 144))
    _text(canvas, f"health     {overall.upper()}  q={float(health.get('quality', 0.0)):.2f}",
          (x, 176), color=health_color, thick=2)
    _text(canvas, f"control ok {health.get('active_control_permitted', False)}", (x, 199))
    components = health.get("components", {})
    y = 226
    for name in ("camera_front", "camera_array", "lidar", "radar", "alignment"):
        state = components.get(name, {}).get("state", "-")
        _text(canvas, f"{name:14s} {state}", (x, y), scale=0.40)
        y += 20
    _text(canvas, f"raw risk       {float(supervisor.get('raw_margin_risk', 0.0)):.1f}", (x, 345))
    _text(canvas, f"actionable     {float(supervisor.get('actionable_collision_risk', 0.0)):.1f}", (x, 368))
    _text(canvas, f"state          {supervisor.get('state', '-')}", (x, 391))
    _text(canvas, f"trigger ID     {supervisor.get('trigger_track_id')}", (x, 414))
    _text(canvas, f"recommended B  {float(supervisor.get('recommended_brake', 0.0)):.2f}", (x, 437))
    _text(canvas, f"applied        {applied.get('applied', False)}", (x, 460))
    _text(canvas, f"apply reason   {applied.get('reason', '-')}", (x, 483), scale=0.40)
    _text(canvas, f"Autoware AEB   {aeb.get('triggered', False)}", (x, 512))
    _text(canvas, f"fusion age     {record.get('fusion_age_s')} s", (x, 535), scale=0.40)
    reasons = applied.get("inhibit_reasons") or supervisor.get("suppression_reasons") or []
    _text(canvas, "reason: " + ",".join(str(item) for item in reasons)[:43], (x, 560), scale=0.36)


def _draw_history(canvas: np.ndarray, history: dict[str, deque]) -> None:
    top, bottom = 590, 705
    left, right = 40, CANVAS_W - 25
    cv2.rectangle(canvas, (left, top), (right, bottom), (20, 24, 29), -1)
    cv2.line(canvas, (left, bottom), (right, bottom), (75, 80, 88), 1)
    palette = {
        "risk": (30, 70, 245),
        "brake": (30, 190, 245),
        "health": (70, 220, 90),
    }
    for key, values in history.items():
        if len(values) < 2:
            continue
        points = []
        for index, value in enumerate(values):
            px = int(left + index / max(len(values) - 1, 1) * (right - left))
            py = int(bottom - np.clip(value, 0.0, 1.0) * (bottom - top - 8))
            points.append((px, py))
        cv2.polylines(
            canvas, [np.asarray(points, dtype=np.int32)], False,
            palette[key], 2, cv2.LINE_AA,
        )
    _text(canvas, "history: risk(red)  applied brake(orange)  sensor quality(green)",
          (left + 8, top + 19), scale=0.42)


def summarize_records(records: list[dict], source: Path) -> dict:
    states: dict[str, int] = {}
    max_tracks = 0
    max_raw_risk = 0.0
    max_actionable = 0.0
    applied_frames = 0
    applied_events = 0
    previous_applied = False
    for record in records:
        state = record.get("sensor_health", {}).get("overall_state", "unknown")
        states[state] = states.get(state, 0) + 1
        max_tracks = max(max_tracks, len(record.get("tracks", [])))
        supervisor = record.get("supervisor", {})
        max_raw_risk = max(max_raw_risk, float(supervisor.get("raw_margin_risk", 0.0)))
        max_actionable = max(
            max_actionable,
            float(supervisor.get("actionable_collision_risk", 0.0)),
        )
        applied = bool(record.get("applied_control", {}).get("applied", False))
        applied_frames += int(applied)
        applied_events += int(applied and not previous_applied)
        previous_applied = applied
    first_t = float(records[0].get("timestamp_s", 0.0))
    last_t = float(records[-1].get("timestamp_s", first_t))
    return {
        "schema": "fusion_tracking.agent_replay_summary.v1",
        "source": str(source),
        "frames": len(records),
        "simulation_duration_s": max(last_t - first_t, 0.0),
        "sensor_health_state_frames": states,
        "max_concurrent_tracks": max_tracks,
        "max_raw_margin_risk": max_raw_risk,
        "max_actionable_collision_risk": max_actionable,
        "applied_intervention_frames": applied_frames,
        "applied_intervention_events": applied_events,
    }


def write_replay(
    source: str | Path,
    output_dir: str | Path | None = None,
    *,
    fps: float = 20.0,
    max_frames: int | None = None,
    write_video: bool = True,
) -> dict:
    source_path = resolve_frame_log(source)
    records = load_records(source_path, max_frames=max_frames)
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = source_path.parent / f"replay_{stamp}"
    else:
        destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    video_path = destination / REPLAY_VIDEO_FILENAME
    writer = None
    if write_video:
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps), (CANVAS_W, CANVAS_H),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the MP4 replay writer")
    history = {
        "risk": deque(maxlen=240),
        "brake": deque(maxlen=240),
        "health": deque(maxlen=240),
    }
    preview_path = destination / REPLAY_PREVIEW_FILENAME
    preview_canvas = None
    preview_score = -1.0
    try:
        for index, record in enumerate(records):
            supervisor = record.get("supervisor", {})
            applied = record.get("applied_control", {}).get("applied_control", {})
            history["risk"].append(
                float(supervisor.get("actionable_collision_risk", 0.0)) / 100.0
            )
            history["brake"].append(float(applied.get("brake", 0.0)))
            history["health"].append(
                float(record.get("sensor_health", {}).get("quality", 0.0))
            )
            canvas = np.full((CANVAS_H, CANVAS_W, 3), (15, 18, 22), dtype=np.uint8)
            _draw_bev(canvas, record)
            _draw_status(canvas, record)
            _draw_history(canvas, history)
            informative_score = (
                float(supervisor.get("actionable_collision_risk", 0.0))
                + 8.0 * len(record.get("tracks", []))
                + (40.0 if record.get("sensor_health", {}).get("overall_state") == "unsafe" else 0.0)
            )
            if informative_score > preview_score:
                preview_score = informative_score
                preview_canvas = canvas.copy()
            if writer is not None:
                writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
    if preview_canvas is not None:
        cv2.imwrite(str(preview_path), preview_canvas)
    summary = summarize_records(records, source_path)
    summary.update({
        "output_dir": str(destination),
        "preview": str(preview_path),
        "video": str(video_path) if write_video else None,
        "video_fps": float(fps) if write_video else None,
    })
    (destination / REPLAY_SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
