"""六相机、BEV、路线、风险与控制状态的固定分区实时仪表盘。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import CAMERA_CHANNELS, DashboardConfig
from .geometry import carla_xyz_to_internal, global_to_ego_xy
from .safety import TrackSafety
from .sensors import SynchronizedSensorFrame
from .types import TrackSnapshot


CLASS_COLORS = {
    "car": (45, 190, 255), "truck": (255, 120, 70), "bus": (255, 190, 60),
    "pedestrian": (240, 80, 210), "bicycle": (80, 220, 90),
    "motorcycle": (210, 100, 255),
}
RISK_COLORS = {"low": (60, 200, 80), "medium": (255, 190, 40), "high": (255, 100, 30), "critical": (255, 40, 40)}


def target_motion_prediction_points(position_ego, velocity_ego, config):
    """生成仅供BEV显示的目标自身轨迹；静态目标返回空数组。

    安全计算仍会使用相对速度判断ego是否会撞上静态目标，两者不能混成同一条线。
    """

    position = np.asarray(position_ego, dtype=float)
    velocity = np.asarray(velocity_ego, dtype=float)
    if float(np.linalg.norm(velocity)) < config.trajectory_display_min_speed_mps:
        return np.empty((0, 2), dtype=float)
    times = np.arange(
        config.trajectory_display_step_s,
        config.trajectory_display_horizon_s + 1e-9,
        config.trajectory_display_step_s,
        dtype=float,
    )
    return position[None, :] + times[:, None] * velocity[None, :]


@dataclass(frozen=True)
class ControlState:
    throttle: float
    brake: float
    steer: float
    reverse: bool
    hand_brake: bool


@dataclass(frozen=True)
class DashboardState:
    sensor_frame: SynchronizedSensorFrame
    camera_boxes: dict[str, tuple[dict, ...]]
    tracks: list[TrackSnapshot]
    safety: list[TrackSafety]
    ego_pose_xyyaw: np.ndarray
    ego_velocity_global: np.ndarray
    ego_path_global: list[np.ndarray]
    speed_kmh: float
    control: ControlState
    scene_id: str
    elapsed_s: float
    fps: float
    fusion_result_age_s: float | None
    perception_latency_s: float | None
    detector_error: str | None
    collision_count: int
    stage_label: str = "STAGE6.1"
    drive_mode: str = "manual"
    control_reason: str | None = None
    supervisor: dict | None = None


class Stage61Dashboard:
    """输出 1600x900 RGB 画布；每个区域都在独立 ROI 内绘制。"""

    def __init__(self, config: DashboardConfig):
        self.cfg = config
        self.speed_history = deque(maxlen=600)
        self.ttc_history = deque(maxlen=600)

    @staticmethod
    def _text(image, text, xy, scale=0.48, color=(235, 235, 235), thickness=1):
        cv2.putText(image, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def render(self, state: DashboardState) -> np.ndarray:
        cfg = self.cfg
        canvas = np.full((cfg.height_px, cfg.width_px, 3), 17, dtype=np.uint8)
        left = canvas[:cfg.top_height_px, :cfg.left_width_px]
        center = canvas[:cfg.top_height_px, cfg.left_width_px:cfg.left_width_px + cfg.center_width_px]
        right = canvas[:cfg.top_height_px, cfg.left_width_px + cfg.center_width_px:]
        bottom = canvas[cfg.top_height_px:, :]
        self._draw_cameras(left, state)
        self._draw_bev(center, state)
        self._draw_status(right, state)
        self._draw_telemetry(bottom, state)
        cv2.line(canvas, (cfg.left_width_px, 0), (cfg.left_width_px, cfg.top_height_px), (80, 80, 80), 2)
        cv2.line(canvas, (cfg.left_width_px + cfg.center_width_px, 0), (cfg.left_width_px + cfg.center_width_px, cfg.top_height_px), (80, 80, 80), 2)
        cv2.line(canvas, (0, cfg.top_height_px), (cfg.width_px, cfg.top_height_px), (80, 80, 80), 2)
        return canvas

    def _draw_cameras(self, panel, state):
        cell_w, cell_h = panel.shape[1] // 2, panel.shape[0] // 3
        for index, channel in enumerate(CAMERA_CHANNELS):
            row, col = divmod(index, 2)
            x0, y0 = col * cell_w, row * cell_h
            image = state.sensor_frame.cameras[channel].image_rgb.copy()
            sx, sy = cell_w / image.shape[1], cell_h / image.shape[0]
            image = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            for det in state.camera_boxes.get(channel, ()):
                x1, y1, x2, y2 = det["box2d"]
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                color = CLASS_COLORS.get(det["class_name"], (220, 220, 220))
                cv2.rectangle(image, p1, p2, color, 2)
                self._text(image, f'{det["class_name"]} {det["score"]:.2f}', (p1[0], max(14, p1[1] - 4)), 0.42, color, 1)
            panel[y0:y0 + cell_h, x0:x0 + cell_w] = image
            cv2.rectangle(panel, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (90, 90, 90), 1)
            self._text(panel, channel, (x0 + 7, y0 + 18), 0.48, (255, 255, 255), 1)

    def _bev_pixel(self, point_ego, width, height):
        x, y = float(point_ego[0]), float(point_ego[1])
        u = int((self.cfg.bev_half_width_m - y) / (2 * self.cfg.bev_half_width_m) * width)
        v = int((self.cfg.bev_forward_m - x) / (self.cfg.bev_forward_m + self.cfg.bev_backward_m) * height)
        return u, v

    def _draw_bev(self, panel, state):
        h, w = panel.shape[:2]
        panel[:] = (24, 26, 30)
        for distance in range(-10, 61, 10):
            _, v = self._bev_pixel((distance, 0), w, h)
            if 0 <= v < h:
                cv2.line(panel, (0, v), (w, v), (48, 50, 54), 1)
                self._text(panel, f"{distance}m", (4, max(v - 3, 12)), 0.35, (125, 125, 125))
        for lateral in range(-30, 31, 10):
            u, _ = self._bev_pixel((0, lateral), w, h)
            if 0 <= u < w:
                cv2.line(panel, (u, 0), (u, h), (48, 50, 54), 1)

        lidar_world = carla_xyz_to_internal(state.sensor_frame.lidar.points_world_carla)
        lidar_ego = global_to_ego_xy(lidar_world[:, :2], state.ego_pose_xyyaw)
        stride = max(len(lidar_ego) // 8000, 1)
        sampled = lidar_ego[::stride]
        lidar_u = ((self.cfg.bev_half_width_m - sampled[:, 1]) / (2 * self.cfg.bev_half_width_m) * w).astype(int)
        lidar_v = ((self.cfg.bev_forward_m - sampled[:, 0]) / (self.cfg.bev_forward_m + self.cfg.bev_backward_m) * h).astype(int)
        visible = (lidar_u >= 0) & (lidar_u < w) & (lidar_v >= 0) & (lidar_v < h)
        panel[lidar_v[visible], lidar_u[visible]] = (105, 105, 105)

        radar_global = np.asarray([
            radar.position_global[:2]
            for sample in state.sensor_frame.radars.values()
            for radar in sample.returns
        ], dtype=float)
        if len(radar_global):
            radar_ego = global_to_ego_xy(radar_global, state.ego_pose_xyyaw)
            radar_u = ((self.cfg.bev_half_width_m - radar_ego[:, 1]) / (2 * self.cfg.bev_half_width_m) * w).astype(int)
            radar_v = ((self.cfg.bev_forward_m - radar_ego[:, 0]) / (self.cfg.bev_forward_m + self.cfg.bev_backward_m) * h).astype(int)
            visible = (radar_u >= 0) & (radar_u < w) & (radar_v >= 0) & (radar_v < h)
            # 邻域赋值比逐点调用 cv2.circle 更省 CPU，同时保留青色雷达点可见性。
            for du, dv in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                uu, vv = radar_u[visible] + du, radar_v[visible] + dv
                inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                panel[vv[inside], uu[inside]] = (30, 220, 240)

        safety_by_id = {item.track_id: item for item in state.safety}
        for track in state.tracks:
            risk = safety_by_id.get(track.track_id)
            if risk is None:
                continue
            color = RISK_COLORS.get(risk.risk_level, (220, 220, 220))
            u, v = self._bev_pixel(risk.position_ego, w, h)
            if not (0 <= u < w and 0 <= v < h):
                continue
            half_l = max(int(track.extent_xy[0] / 2 / (self.cfg.bev_forward_m + self.cfg.bev_backward_m) * h), 3)
            half_w = max(int(track.extent_xy[1] / 2 / (2 * self.cfg.bev_half_width_m) * w), 3)
            cv2.rectangle(panel, (u - half_w, v - half_l), (u + half_w, v + half_l), color, 2)
            # risk.future_positions_ego是相对ego的碰撞预测：ego前进时，静止障碍也会
            # 形成一条相对路径。仪表盘改画目标自身运动轨迹，使“静止框无预测线”。
            display_future = target_motion_prediction_points(
                risk.position_ego, risk.velocity_ego, self.cfg
            )
            future_pixels = [self._bev_pixel(p, w, h) for p in display_future]
            valid = [(px, py) for px, py in future_pixels if 0 <= px < w and 0 <= py < h]
            if len(valid) >= 2:
                cv2.polylines(panel, [np.asarray(valid, dtype=np.int32)], False, color, 2)
            self._text(panel, f"ID{track.track_id} {risk.risk_score:.0f}", (u + 5, v - 5), 0.42, color, 1)
        ego_u, ego_v = self._bev_pixel((0.0, 0.0), w, h)
        cv2.rectangle(panel, (ego_u - 8, ego_v - 16), (ego_u + 8, ego_v + 16), (40, 255, 90), -1)
        self._text(
            panel,
            "BEV: LiDAR(gray) Radar(cyan) line=target motion",
            (8, 20), 0.43, (240, 240, 240),
        )

    def _draw_status(self, panel, state):
        panel[:] = (21, 23, 28)
        mode_label = state.drive_mode.upper()
        supervisor = state.supervisor
        supervisor_mode = "SHADOW" if supervisor is None else supervisor["mode"].upper()
        self._text(panel, f"{state.stage_label} {mode_label} + {supervisor_mode}", (10, 24), 0.52, (90, 230, 255), 1)
        subtitle = (
            "Risk output does not override controller"
            if supervisor_mode == "SHADOW" else
            "Longitudinal safety supervisor may intervene"
        )
        self._text(panel, subtitle, (10, 46), 0.40, (170, 170, 170))
        path_h = 260
        cv2.rectangle(panel, (8, 60), (panel.shape[1] - 8, path_h), (55, 55, 60), 1)
        self._text(panel, "GLOBAL EGO PATH (DRIVEN)", (16, 80), 0.42, (220, 220, 220))
        if len(state.ego_path_global) >= 2:
            points = np.asarray(state.ego_path_global)
            lo, hi = points.min(axis=0), points.max(axis=0)
            span = np.maximum(hi - lo, 5.0)
            scale = min((panel.shape[1] - 36) / span[0], (path_h - 100) / span[1])
            pixels = np.column_stack([
                18 + (points[:, 0] - lo[0]) * scale,
                path_h - 16 - (points[:, 1] - lo[1]) * scale,
            ]).astype(np.int32)
            cv2.polylines(panel, [pixels], False, (80, 180, 255), 2)
            cv2.circle(panel, tuple(pixels[-1]), 5, (60, 255, 90), -1)

        peak = state.safety[0] if state.safety else None
        y = 290
        if peak:
            color = RISK_COLORS.get(peak.risk_level, (220, 220, 220))
            lines = [
                f"Highest risk: ID{peak.track_id}", f"score / level: {peak.risk_score:.1f} / {peak.risk_level}",
                f"reliability*: {peak.evidence_reliability:.2f}",
                f"time_to_margin: {peak.time_to_margin_s if peak.time_to_margin_s is not None else '--'} s",
                f"min clearance: {peak.min_clearance_m:.2f} m",
            ]
        else:
            color, lines = (150, 150, 150), ["Highest risk: no confirmed target"]
        for line in lines:
            self._text(panel, line, (12, y), 0.48, color)
            y += 28
        self._text(panel, "* heuristic evidence score", (12, y + 4), 0.36, (130, 130, 130))
        y += 45
        if supervisor is not None:
            application = supervisor.get("application") or {}
            trigger = supervisor.get("trigger_track_id")
            action_color = (
                (255, 100, 50)
                if supervisor.get("intervention_requested") else (100, 220, 130)
            )
            for line in [
                f"Action risk: {supervisor['actionable_collision_risk']:.1f}",
                f"state={supervisor['state']} trigger=ID{trigger if trigger is not None else '--'}",
                f"recommend: throttle<={supervisor['recommended_throttle_cap']:.2f} brake={supervisor['recommended_brake']:.2f}",
                f"applied={application.get('applied', False)} reason={application.get('reason', '--')}",
            ]:
                self._text(panel, line, (12, y), 0.40, action_color)
                y += 23
            if supervisor.get("suppression_reasons"):
                reasons = ",".join(supervisor["suppression_reasons"])
                self._text(panel, f"suppressed: {reasons}"[:47], (12, y), 0.35, (180, 180, 180))
                y += 22
        control = state.control
        fusion_age = "--" if state.fusion_result_age_s is None else f"{state.fusion_result_age_s:.2f}"
        perception_time = "--" if state.perception_latency_s is None else f"{state.perception_latency_s:.2f}"
        for line in [
            f"Control: {mode_label}  reverse={control.reverse}",
            f"throttle={control.throttle:.2f} brake={control.brake:.2f}",
            f"steer={control.steer:+.2f} handbrake={control.hand_brake}",
            f"control reason={state.control_reason or '--'}",
            f"fusion age={fusion_age} s",
            f"perception={perception_time} s",
            f"tracks={len(state.tracks)} fps={state.fps:.1f}",
        ]:
            self._text(panel, line, (12, y), 0.43, (220, 220, 220))
            y += 25
        if state.detector_error:
            self._text(panel, state.detector_error[:39], (12, min(y + 8, panel.shape[0] - 12)), 0.36, (255, 80, 80))

    def _draw_telemetry(self, panel, state):
        panel[:] = (16, 18, 22)
        peak = state.safety[0] if state.safety else None
        ttc = peak.approx_ttc_s if peak else None
        margin = peak.time_to_margin_s if peak else None
        clearance = peak.min_clearance_m if peak else None
        self.speed_history.append(state.speed_kmh)
        self.ttc_history.append(10.0 if ttc is None else min(ttc, 10.0))
        ttc_text = "--" if ttc is None else f"{ttc:.2f}"
        margin_text = "--" if margin is None else f"{margin:.2f}"
        clearance_text = "--" if clearance is None else f"{clearance:.2f}"
        action_text = (
            "--" if state.supervisor is None else
            f"{state.supervisor['actionable_collision_risk']:.0f}/{state.supervisor['state']}"
        )
        headline = (
            f"scene={state.scene_id}  t={state.elapsed_s:05.1f}s  speed={state.speed_kmh:05.1f} km/h  "
            f"throttle={state.control.throttle:.2f}  brake={state.control.brake:.2f}  steer={state.control.steer:+.2f}  "
            f"TTC~={ttc_text}s  margin_t={margin_text}s  "
            f"clearance={clearance_text}m  action={action_text}  collisions={state.collision_count}"
        )
        self._text(panel, headline, (12, 27), 0.51, (240, 240, 240), 1)
        graph_x0, graph_y0, graph_w, graph_h = 12, 45, panel.shape[1] - 24, panel.shape[0] - 55
        cv2.rectangle(panel, (graph_x0, graph_y0), (graph_x0 + graph_w, graph_y0 + graph_h), (55, 55, 60), 1)
        if len(self.speed_history) >= 2:
            speed = np.asarray(self.speed_history)
            xs = np.linspace(graph_x0, graph_x0 + graph_w, len(speed)).astype(int)
            ys = (graph_y0 + graph_h - np.clip(speed / 100.0, 0, 1) * graph_h).astype(int)
            cv2.polylines(panel, [np.column_stack([xs, ys]).astype(np.int32)], False, (70, 210, 255), 2)
        self._text(panel, "speed history (0-100 km/h)", (graph_x0 + 8, graph_y0 + 18), 0.39, (70, 210, 255))
