"""CARLA 手动驾驶会话：世界管理、在线融合、仪表盘和结构化落盘。"""

from __future__ import annotations

import csv
import json
import os
import queue
import random
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import CAMERA_CHANNELS, Stage61Config
from .dashboard import ControlState, DashboardState, Stage61Dashboard
from .evaluation import (
    CollisionIncidentCounter,
    LatencyMetricsAccumulator,
    ScenarioMetricsAccumulator,
    route_completion_percent,
)
from .geometry import carla_transform_to_internal_pose, global_vector_to_ego
from .oracle_evaluation import OracleTrackEvaluator, collect_vehicle_truth
from .perception import LatestOnlyPerceptionWorker, PerceptionJob
from .safety import assess_tracks
from .sensors import CarlaSensorRig
from .tracker import StreamingFusionTracker


def _json_default(value):
    """把数值计算产生的NumPy类型收敛到公开JSON数据契约。

    只接受NumPy标量/数组；其他未知对象继续抛错，避免静默字符串化掩盖接口问题。
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _json_dumps(value, **kwargs) -> str:
    return json.dumps(value, default=_json_default, **kwargs)


class _WindowsAsyncKeyboard:
    """Windows 原生全局按键状态；不要求 Pygame 窗口持有焦点。"""

    VK_RETURN = 0x0D
    VK_ESCAPE = 0x1B
    VK_SPACE = 0x20
    VK_LEFT = 0x25
    VK_UP = 0x26
    VK_RIGHT = 0x27
    VK_DOWN = 0x28
    VK_A, VK_D, VK_Q, VK_S, VK_W = 0x41, 0x44, 0x51, 0x53, 0x57

    def __init__(self):
        import ctypes

        self._get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        self._get_async_key_state.argtypes = [ctypes.c_int]
        self._get_async_key_state.restype = ctypes.c_short

    def down(self, virtual_key: int) -> bool:
        return bool(self._get_async_key_state(virtual_key) & 0x8000)


class ManualKeyboardController:
    """Pygame 键盘控制；Stage6.1 永不向控制量叠加自动制动。"""

    def __init__(
        self,
        pygame_module,
        use_global_keyboard: bool = True,
        async_key_reader=None,
    ):
        self.pygame = pygame_module
        self.reverse = False
        self.steer_cache = 0.0
        self._held_keys: set[int] = set()
        self.has_reported_input = False
        if async_key_reader is not None:
            self._async_keyboard = async_key_reader
        elif use_global_keyboard and os.name == "nt":
            self._async_keyboard = _WindowsAsyncKeyboard()
        else:
            self._async_keyboard = None
        self._q_was_down = False

    @property
    def backend_name(self) -> str:
        return (
            "Windows GetAsyncKeyState (global)"
            if self._async_keyboard is not None
            else "Pygame KEYDOWN/KEYUP (window focus)"
        )

    def _down(self, *keys: int) -> bool:
        return any(key in self._held_keys for key in keys)

    def _control_down(self, pygame_keys, virtual_keys) -> bool:
        event_down = self._down(*pygame_keys)
        global_down = (
            self._async_keyboard is not None
            and any(self._async_keyboard.down(key) for key in virtual_keys)
        )
        return bool(event_down or global_down)

    def read(self, dt_s: float):
        pygame = self.pygame
        quit_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_requested = True
            elif event.type == pygame.KEYDOWN:
                self._held_keys.add(event.key)
            elif event.type == pygame.KEYUP:
                self._held_keys.discard(event.key)
                if event.key == pygame.K_ESCAPE:
                    quit_requested = True
            elif event.type == getattr(pygame, "WINDOWFOCUSLOST", -1):
                # 窗口失焦时清空锁存，避免车辆保持上一次油门或转向。
                self._held_keys.clear()

        async_keys = _WindowsAsyncKeyboard
        # 首选Windows全局状态，同时保留Pygame事件锁存作为跨平台后备。
        throttle = 1.0 if self._control_down(
            (pygame.K_w, pygame.K_UP), (async_keys.VK_W, async_keys.VK_UP)
        ) else 0.0
        brake = 1.0 if self._control_down(
            (pygame.K_s, pygame.K_DOWN), (async_keys.VK_S, async_keys.VK_DOWN)
        ) else 0.0
        steer_target = float(self._control_down(
            (pygame.K_d, pygame.K_RIGHT), (async_keys.VK_D, async_keys.VK_RIGHT)
        )) - float(self._control_down(
            (pygame.K_a, pygame.K_LEFT), (async_keys.VK_A, async_keys.VK_LEFT)
        ))
        q_is_down = self._control_down((pygame.K_q,), (async_keys.VK_Q,))
        if q_is_down and not self._q_was_down:
            self.reverse = not self.reverse
        self._q_was_down = q_is_down
        if (
            self._async_keyboard is not None
            and self._async_keyboard.down(async_keys.VK_ESCAPE)
        ):
            quit_requested = True
        response = min(1.0, 5.0 * dt_s)
        self.steer_cache += response * (steer_target - self.steer_cache)
        if steer_target == 0.0:
            self.steer_cache *= max(0.0, 1.0 - 7.0 * dt_s)
        hand_brake = self._control_down(
            (pygame.K_SPACE,), (async_keys.VK_SPACE,)
        )
        control = {
            "throttle": throttle,
            "brake": brake,
            "steer": float(np.clip(self.steer_cache, -1.0, 1.0)),
            "reverse": self.reverse,
            "hand_brake": hand_brake,
        }
        return control, quit_requested

    def wait_for_start(self, screen, stage_label="Stage6.1", drive_mode="manual") -> bool:
        """等待用户聚焦仪表盘并按 Enter；Esc 返回 False。"""

        pygame = self.pygame
        screen.fill((16, 18, 22))
        title_font = pygame.font.Font(None, 52)
        body_font = pygame.font.Font(None, 34)
        lines = [
            (title_font, f"{stage_label} sensors and detector are ready", (70, 220, 255)),
            (body_font, "Click this dashboard, then press ENTER to start timing", (235, 235, 235)),
            (body_font, f"drive mode: {drive_mode}", (190, 190, 190)),
            (body_font, "W/S/A/D: manual control   ESC: stop", (190, 190, 190)),
        ]
        y = 300
        for font, text, color in lines:
            surface = font.render(text, True, color)
            screen.blit(surface, ((screen.get_width() - surface.get_width()) // 2, y))
            y += 62
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return True
                    if event.key == pygame.K_ESCAPE:
                        return False
            if self._async_keyboard is not None:
                if self._async_keyboard.down(_WindowsAsyncKeyboard.VK_ESCAPE):
                    return False
                if self._async_keyboard.down(_WindowsAsyncKeyboard.VK_RETURN):
                    # 等待Enter松开，避免它残留到正式控制循环。
                    while self._async_keyboard.down(_WindowsAsyncKeyboard.VK_RETURN):
                        pygame.time.wait(10)
                    return True
            pygame.time.wait(20)


class _AsyncVideoWriter:
    """在后台编码 MP4，主循环只入队；队列满时阻塞以保证不丢仿真帧。"""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], queue_size: int = 8):
        self._writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
        if not self._writer.isOpened():
            raise RuntimeError("OpenCV 无法创建 MP4，请检查编码器和输出路径")
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self._error: Exception | None = None
        self._closed = False
        self.frames_written = 0
        self._thread = threading.Thread(
            target=self._run, name="stage6_video_writer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            frame_rgb = self._queue.get()
            if frame_rgb is None:
                return
            if self._error is not None:
                continue
            try:
                self._writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                self.frames_written += 1
            except Exception as exc:  # pragma: no cover - codec failure depends on host.
                self._error = exc

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("视频写入器已经关闭")
        if self._error is not None:
            raise RuntimeError(f"后台视频编码失败: {self._error}") from self._error
        # render 每帧返回独立数组；队列持有引用直到编码完成，不需要再复制 4.3 MB。
        self._queue.put(frame_rgb)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=30.0)
        self._writer.release()
        if self._thread.is_alive():
            raise RuntimeError("后台视频编码线程在 30s 内未完成清空")
        if self._error is not None:
            raise RuntimeError(f"后台视频编码失败: {self._error}") from self._error


class Stage61OutputWriter:
    """每次运行写入独立时间戳目录，不覆盖历史产物。"""

    def __init__(self, config: Stage61Config):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.runtime.run_name or f"{config.driving.mode}_run_{timestamp}"
        self.run_dir = (config.runtime.output_root / run_name).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        stem = (
            "stage6_1_carla_manual_fusion"
            if config.runtime.stage_id == "stage6_1"
            else f"{config.runtime.stage_id}_carla_closed_loop"
        )
        self.video_path = self.run_dir / f"{stem}_dashboard.mp4"
        self.frames_path = self.run_dir / f"{stem}_frames.jsonl"
        self.summary_path = self.run_dir / f"{stem}_summary.json"
        self.timeline_path = self.run_dir / f"{stem}_risk_timeline.png"
        self.events_path = self.run_dir / f"{stem}_events.csv"
        self.config_path = self.run_dir / f"{stem}_config.json"
        self.metrics_path = self.run_dir / f"{stem}_metrics.json"
        self.oracle_metrics_path = self.run_dir / f"{stem}_oracle_metrics.json"
        self.latency_metrics_path = self.run_dir / f"{stem}_latency_metrics.json"
        self.supervisor_events_path = self.run_dir / f"{stem}_supervisor_events.jsonl"
        self.aeb_shadow_events_path = self.run_dir / f"{stem}_aeb_shadow_events.jsonl"
        self.frame_file = self.frames_path.open("w", encoding="utf-8")
        self.video = None
        if config.runtime.record_video:
            size = (config.dashboard.width_px, config.dashboard.height_px)
            self.video = _AsyncVideoWriter(
                self.video_path, config.runtime.video_fps, size,
            )
        self.config_path.write_text(
            _json_dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.risk_history: list[tuple[float, float]] = []
        self.actionable_risk_history: list[tuple[float, float]] = []
        self.events: list[dict] = []
        self.supervisor_events: list[dict] = []
        self.aeb_shadow_events: list[dict] = []

    def write_frame(self, rgb_frame: np.ndarray | None, record: dict) -> None:
        self.frame_file.write(_json_dumps(record, ensure_ascii=False) + "\n")
        self.frame_file.flush()
        if self.video is not None:
            if rgb_frame is None:
                raise RuntimeError("录制视频时仪表盘帧不能为空")
            self.video.write(rgb_frame)
        self.risk_history.append((record["elapsed_s"], record["highest_risk_score"]))
        supervisor = record.get("safety_supervisor")
        if supervisor is not None:
            self.actionable_risk_history.append((
                record["elapsed_s"],
                float(supervisor["actionable_collision_risk"]),
            ))

    def add_collision(self, event, *, is_new_incident: bool, incident_id: int) -> None:
        self.events.append({
            "frame": event.frame, "timestamp_s": event.timestamp_s,
            "event": "collision_contact",
            "incident_id": incident_id,
            "is_new_incident": is_new_incident,
            "other_actor_id": event.other_actor_id,
            "other_actor": event.other_actor,
            "impulse": event.impulse,
        })

    def add_supervisor_event(self, event: dict) -> None:
        """记录一次从未介入到请求介入的状态跃迁。"""

        self.supervisor_events.append(dict(event))

    def add_aeb_shadow_event(self, event: dict) -> None:
        """记录旁路AEB从未触发到触发的跃迁；该事件没有控制权。"""

        self.aeb_shadow_events.append(dict(event))

    def close(self, summary: dict) -> None:
        self.frame_file.close()
        if self.video is not None:
            self.video.close()
        self.summary_path.write_text(
            _json_dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if "evaluation" in summary:
            self.metrics_path.write_text(
                _json_dumps(summary["evaluation"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if summary.get("oracle_perception") is not None:
            self.oracle_metrics_path.write_text(
                _json_dumps(summary["oracle_perception"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if summary.get("latency_breakdown") is not None:
            self.latency_metrics_path.write_text(
                _json_dumps(summary["latency_breakdown"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        with self.events_path.open("w", encoding="utf-8-sig", newline="") as handle:
            fields = [
                "frame", "timestamp_s", "event", "incident_id", "is_new_incident",
                "other_actor_id", "other_actor", "impulse",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.events)
        if summary.get("safety_supervisor") is not None:
            with self.supervisor_events_path.open("w", encoding="utf-8") as handle:
                for event in self.supervisor_events:
                    handle.write(_json_dumps(event, ensure_ascii=False) + "\n")
        if summary.get("autoware_aeb_shadow") is not None:
            with self.aeb_shadow_events_path.open("w", encoding="utf-8") as handle:
                for event in self.aeb_shadow_events:
                    handle.write(_json_dumps(event, ensure_ascii=False) + "\n")
        self._write_timeline()

    def _write_timeline(self) -> None:
        image = np.full((420, 1200, 3), 248, dtype=np.uint8)
        cv2.rectangle(image, (70, 35), (1170, 360), (60, 60, 60), 2)
        title = (
            "Raw margin risk and actionable collision risk"
            if self.actionable_risk_history else "Highest predicted risk over time"
        )
        cv2.putText(image, title, (70, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
        if self.risk_history:
            values = np.asarray(self.risk_history, dtype=float)
            max_time = max(float(values[-1, 0]), 1.0)
            pixels = np.column_stack([
                70 + values[:, 0] / max_time * 1100,
                360 - np.clip(values[:, 1], 0, 100) / 100 * 325,
            ]).astype(np.int32)
            if len(pixels) >= 2:
                cv2.polylines(image, [pixels], False, (30, 60, 230), 2)
        if self.actionable_risk_history:
            values = np.asarray(self.actionable_risk_history, dtype=float)
            max_time = max(
                float(self.risk_history[-1][0]) if self.risk_history else 0.0,
                float(values[-1, 0]),
                1.0,
            )
            pixels = np.column_stack([
                70 + values[:, 0] / max_time * 1100,
                360 - np.clip(values[:, 1], 0, 100) / 100 * 325,
            ]).astype(np.int32)
            if len(pixels) >= 2:
                cv2.polylines(image, [pixels], False, (30, 160, 30), 2)
        if self.actionable_risk_history:
            cv2.line(image, (780, 20), (815, 20), (30, 60, 230), 2)
            cv2.putText(image, "raw margin", (825, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.line(image, (965, 20), (1000, 20), (30, 160, 30), 2)
            cv2.putText(image, "actionable", (1010, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(image, "0", (42, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.putText(image, "100", (28, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.imwrite(str(self.timeline_path), image)


class CarlaFusionSession:
    """Stage6.1/6.2 共用在线会话；实例化不连接 CARLA，调用 ``run`` 才改变世界。"""

    def __init__(
        self,
        config: Stage61Config,
        perception_worker_factory=None,
        longitudinal_supervisor=None,
        scenario_controller=None,
        aeb_shadow_evaluator=None,
        aeb_advisory_throttle_cap: float | None = None,
        dataset_recorder=None,
    ):
        config.validate()
        self.config = config
        # 默认仍是原来的torchvision前端；Stage6.6通过工厂注入BEVFusion，
        # 不复制CARLA世界/控制/跟踪/风险主循环，便于严格A/B对照。
        self.perception_worker_factory = perception_worker_factory
        # Stage6.7可选注入；None时Stage6.1~6.6的控制和输出行为不变。
        self.longitudinal_supervisor = longitudinal_supervisor
        # Stage6.8可选注入确定性场景；通过小型hook控制出生点/目标车辆，
        # 不把CARLA真值写入感知、跟踪、风险或监督器输入。
        self.scenario_controller = scenario_controller
        # Stage6.15旁路对照。它读取与主风险监督器相同的TrackSnapshot，但没有
        # arbitrate/apply接口，因此不能改写油门、制动或转向。
        self.aeb_shadow_evaluator = aeb_shadow_evaluator
        # Optional project-owned use of the AEB comparison result: the previous
        # frame may only cap throttle.  It never writes brake or steering, so the
        # project's risk supervisor remains the sole Active braking authority.
        if aeb_advisory_throttle_cap is not None and not (
            0.0 <= float(aeb_advisory_throttle_cap) <= 1.0
        ):
            raise ValueError("aeb_advisory_throttle_cap must be in [0, 1]")
        self.aeb_advisory_throttle_cap = (
            None
            if aeb_advisory_throttle_cap is None
            else float(aeb_advisory_throttle_cap)
        )
        # Stage6.16 可选离线数据记录器。它可以读取 actor 真值生成训练标签，
        # 但 record() 没有任何返回到感知/跟踪/控制链的接口。
        self.dataset_recorder = dataset_recorder
        self.created_actors = []
        self.npc_actors = []
        self._smoothed_route_throttle = 0.0
        self._smoothed_route_brake = 0.0
        self._route_governor_acceleration_mps2 = 0.0

    @staticmethod
    def _rate_limit(current: float, target: float, rise: float, fall: float, dt_s: float) -> float:
        if target >= current:
            return min(target, current + rise * dt_s)
        return max(target, current - fall * dt_s)

    @staticmethod
    def _apply_aeb_throttle_advisory(
        control_values: dict,
        prior_decision,
        throttle_cap: float | None,
        *,
        active: bool,
    ) -> tuple[dict, dict]:
        """Apply a one-frame-old AEB advisory without granting brake authority."""

        output = dict(control_values)
        application = {
            "enabled": throttle_cap is not None,
            "applied": False,
            "source_timestamp_s": None,
            "target_track_id": None,
            "throttle_before": float(output["throttle"]),
            "throttle_after": float(output["throttle"]),
            "brake_unchanged": float(output["brake"]),
            "reason": "no_prior_aeb_trigger",
        }
        if not (
            active
            and throttle_cap is not None
            and prior_decision is not None
            and prior_decision.triggered
        ):
            return output, application

        before_throttle = float(output["throttle"])
        output["throttle"] = min(before_throttle, float(throttle_cap))
        application.update({
            "applied": output["throttle"] < before_throttle,
            "source_timestamp_s": float(prior_decision.timestamp_s),
            "target_track_id": prior_decision.target_track_id,
            "throttle_before": before_throttle,
            "throttle_after": float(output["throttle"]),
            "reason": "prior_aeb_path_rss_trigger",
        })
        return output, application

    def _smooth_route_longitudinal_control(
        self, control: dict, current_speed_mps: float | None = None,
    ) -> dict:
        """Smooth only BasicAgent cruise PID; Active safety arbitration runs afterwards."""

        cfg = self.config.driving
        if cfg.route_speed_governor_enabled:
            if current_speed_mps is None:
                raise ValueError("route speed governor requires current_speed_mps")
            dt_s = float(self.config.runtime.fixed_delta_s)
            target_speed_mps = float(cfg.target_speed_kmh) / 3.6
            speed_error = target_speed_mps - float(current_speed_mps)
            desired_acceleration = float(np.clip(
                cfg.route_speed_governor_kp * speed_error,
                -cfg.route_speed_governor_max_decel_mps2,
                cfg.route_speed_governor_max_accel_mps2,
            ))
            jerk_step = cfg.route_speed_governor_jerk_mps3 * dt_s
            self._route_governor_acceleration_mps2 = float(np.clip(
                desired_acceleration,
                self._route_governor_acceleration_mps2 - jerk_step,
                self._route_governor_acceleration_mps2 + jerk_step,
            ))
            governed_acceleration = self._route_governor_acceleration_mps2
            throttle = float(np.clip(
                cfg.route_speed_governor_throttle_feedforward
                + cfg.route_speed_governor_throttle_per_accel * governed_acceleration,
                0.0,
                cfg.route_speed_governor_max_throttle,
            ))
            brake = float(np.clip(
                -governed_acceleration * cfg.route_speed_governor_brake_per_decel,
                0.0,
                cfg.route_speed_governor_max_brake,
            ))
            if brake > 0.0:
                throttle = 0.0
            # Preserve an explicit BasicAgent hazard/light stop in normal road use.
            # Deterministic safety tests disable those truth-based checks, so their
            # obstacle response still comes exclusively from project perception.
            raw_brake = float(control["brake"])
            if raw_brake >= 0.5 and not (
                cfg.route_ignore_vehicles and cfg.route_ignore_traffic_lights
            ):
                throttle = 0.0
                brake = max(brake, raw_brake)
            result = dict(control)
            result["throttle"] = throttle
            result["brake"] = brake
            return result
        if not cfg.smooth_route_longitudinal_control:
            return dict(control)
        dt_s = float(self.config.runtime.fixed_delta_s)
        raw_throttle = float(control["throttle"])
        raw_brake = float(control["brake"])
        if raw_brake > 0.01:
            self._smoothed_route_throttle = self._rate_limit(
                self._smoothed_route_throttle, 0.0,
                cfg.route_throttle_rise_per_s, cfg.route_throttle_fall_per_s, dt_s,
            )
            brake_target = raw_brake if self._smoothed_route_throttle <= 0.01 else 0.0
            self._smoothed_route_brake = self._rate_limit(
                self._smoothed_route_brake, brake_target,
                cfg.route_brake_rise_per_s, cfg.route_brake_fall_per_s, dt_s,
            )
        else:
            self._smoothed_route_brake = self._rate_limit(
                self._smoothed_route_brake, 0.0,
                cfg.route_brake_rise_per_s, cfg.route_brake_fall_per_s, dt_s,
            )
            throttle_target = (
                raw_throttle if self._smoothed_route_brake <= 0.01 else 0.0
            )
            self._smoothed_route_throttle = self._rate_limit(
                self._smoothed_route_throttle, throttle_target,
                cfg.route_throttle_rise_per_s, cfg.route_throttle_fall_per_s, dt_s,
            )
        result = dict(control)
        result["throttle"] = float(np.clip(self._smoothed_route_throttle, 0.0, 1.0))
        result["brake"] = float(np.clip(self._smoothed_route_brake, 0.0, 1.0))
        return result

    def _spawn_ego(self, world, carla):
        random.seed(self.config.runtime.seed)
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        requested_blueprint_id = None
        if self.scenario_controller is not None:
            select_blueprint = getattr(
                self.scenario_controller, "select_ego_blueprint_id", None
            )
            if callable(select_blueprint):
                requested_blueprint_id = select_blueprint()

        if requested_blueprint_id:
            matches = [bp for bp in blueprints if bp.id == requested_blueprint_id]
            if not matches:
                raise RuntimeError(
                    f"官方场景要求ego={requested_blueprint_id}，"
                    "但当前CARLA blueprint library中不存在该车型"
                )
            blueprint = matches[0]
            blueprint_source = "ScenarioRunner XML"
        else:
            preferred = [bp for bp in blueprints if bp.id in {
                "vehicle.tesla.model3", "vehicle.lincoln.mkz_2020", "vehicle.audi.tt"
            }]
            blueprint = random.choice(preferred or list(blueprints))
            blueprint_source = "project default"

        print(
            f"[ego车型] {blueprint.id} | source={blueprint_source}",
            flush=True,
        )
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", f"{self.config.runtime.stage_id}_ego")
        spawn_points = list(world.get_map().get_spawn_points())
        ordered_spawn_points = list(spawn_points)
        if self.scenario_controller is not None:
            preferred_transform = self.scenario_controller.select_ego_spawn(
                world, ordered_spawn_points
            )
            if preferred_transform is not None:
                ordered_spawn_points = [preferred_transform] + [
                    item for item in ordered_spawn_points if item is not preferred_transform
                ]
        else:
            random.shuffle(ordered_spawn_points)
        for transform in ordered_spawn_points:
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                # 明确声明ego由本进程手动控制并启用物理，避免继承旧世界控制状态。
                actor.set_autopilot(False)
                actor.set_simulate_physics(True)
                actor.apply_control(carla.VehicleControl(
                    throttle=0.0,
                    brake=0.0,
                    steer=0.0,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False,
                ))
                self.created_actors.append(actor)
                return actor, spawn_points
        raise RuntimeError("CARLA 地图没有可用 ego 出生点")

    def _spawn_npcs(self, client, world, carla, spawn_points, ego):
        count = self.config.runtime.spawn_npc_vehicles
        if count <= 0:
            return
        candidates = world.get_blueprint_library().filter("vehicle.*")
        candidates = [bp for bp in candidates if int(bp.get_attribute("number_of_wheels")) == 4]
        commands = []
        ego_location = ego.get_location()
        # Town10HD 很大；短演示若全图随机，车辆通常不在任何相机视野内。
        ordered_spawn_points = sorted(
            spawn_points, key=lambda item: item.location.distance(ego_location)
        )
        nearby = [
            item for item in ordered_spawn_points
            if self.config.runtime.npc_min_spawn_distance_m
            <= item.location.distance(ego_location)
            <= self.config.runtime.npc_spawn_radius_m
        ]
        fallback = [item for item in ordered_spawn_points if item not in nearby]
        for transform in [*nearby, *fallback]:
            if len(commands) >= count:
                break
            if transform.location.distance(ego_location) < self.config.runtime.npc_min_spawn_distance_m:
                continue
            blueprint = random.choice(candidates)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", f"{self.config.runtime.stage_id}_npc")
            commands.append(
                (
                    carla.command.SpawnActor(blueprint, transform).then(
                        carla.command.SetAutopilot(
                            carla.command.FutureActor,
                            True,
                            self.config.runtime.traffic_manager_port,
                        )
                    )
                    if self.config.runtime.npc_autopilot_enabled
                    else carla.command.SpawnActor(blueprint, transform)
                )
            )
        responses = client.apply_batch_sync(commands, True)
        for response in responses:
            if not response.error:
                actor = world.get_actor(response.actor_id)
                if actor is not None:
                    self.npc_actors.append(actor)
        distances = sorted(
            actor.get_location().distance(ego_location) for actor in self.npc_actors
        )
        nearest = ", ".join(f"{value:.0f}m" for value in distances[:5]) or "无"
        print(f"[交通流] 成功生成 {len(self.npc_actors)}/{count} 辆 | 最近距离：{nearest}")

    @staticmethod
    def _pygame_show(pygame, screen, rgb_frame):
        surface = pygame.surfarray.make_surface(np.swapaxes(rgb_frame, 0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    @staticmethod
    def _control_to_dict(control) -> dict:
        """把 CARLA VehicleControl 统一成日志/仪表盘使用的五字段。"""

        return {
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "steer": float(control.steer),
            "reverse": bool(control.reverse),
            "hand_brake": bool(control.hand_brake),
        }

    def _prepare_automatic_drive(self, world, ego, spawn_points, traffic_manager):
        """配置 autopilot 或 BasicAgent；manual 返回空控制器。"""

        cfg = self.config
        mode = cfg.driving.mode
        if mode == "manual":
            return None, 0, None
        if mode == "autopilot":
            ego.set_autopilot(True, cfg.runtime.traffic_manager_port)
            traffic_manager.distance_to_leading_vehicle(
                ego, cfg.driving.traffic_manager_follow_distance_m
            )
            speed_limit = max(float(ego.get_speed_limit()), 1.0)
            target_difference = 100.0 * (
                speed_limit - cfg.driving.target_speed_kmh
            ) / speed_limit
            traffic_manager.vehicle_percentage_speed_difference(
                ego,
                target_difference + cfg.driving.traffic_manager_speed_difference_pct,
            )
            return None, 0, None

        agents_root = cfg.driving.carla_agents_root
        if agents_root is None or not (Path(agents_root) / "agents").exists():
            raise FileNotFoundError(
                "route_agent 需要 CARLA PythonAPI/carla 目录；请通过 --carla-agents-root 指定"
            )
        root_text = str(Path(agents_root).resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from agents.navigation.basic_agent import BasicAgent

        ego_location = ego.get_location()
        candidates = [
            transform for transform in spawn_points
            if transform.location.distance(ego_location) >= cfg.driving.route_min_distance_m
        ]
        if not candidates:
            raise RuntimeError("当前地图没有满足 route_min_distance_m 的路线终点")
        destination = min(
            candidates,
            key=lambda transform: abs(
                transform.location.distance(ego_location) - cfg.driving.route_target_distance_m
            ),
        )
        agent = BasicAgent(
            ego,
            target_speed=cfg.driving.target_speed_kmh,
            opt_dict={
                "base_vehicle_threshold": cfg.driving.route_vehicle_base_threshold_m,
                "detection_speed_ratio": cfg.driving.route_detection_speed_ratio_s,
                "max_brake": cfg.driving.route_max_brake,
                "ignore_vehicles": cfg.driving.route_ignore_vehicles,
                "ignore_traffic_lights": cfg.driving.route_ignore_traffic_lights,
            },
            map_inst=world.get_map(),
        )
        agent.set_destination(destination.location)
        initial_waypoints = len(agent.get_local_planner().get_plan())
        if initial_waypoints < 2:
            raise RuntimeError("BasicAgent 未生成有效路线；请更换 seed/地图或增大 route-distance")
        destination_xyz = [
            float(destination.location.x),
            float(destination.location.y),
            float(destination.location.z),
        ]
        print(
            f"[路线] BasicAgent 路线点={initial_waypoints} | "
            f"直线距离={destination.location.distance(ego_location):.1f}m | "
            f"目标速度={cfg.driving.target_speed_kmh:.1f}km/h | "
            f"车辆真值避障={'关' if cfg.driving.route_ignore_vehicles else '开'} | "
            f"交通灯真值={'关' if cfg.driving.route_ignore_traffic_lights else '开'}",
            flush=True,
        )
        return agent, initial_waypoints, destination_xyz

    @staticmethod
    def _lane_offset_m(world_map, location, carla):
        """返回相对 driving lane 中线的有符号横向偏移；不在车道时返回 None。"""

        exact = world_map.get_waypoint(
            location, project_to_road=False, lane_type=carla.LaneType.Driving
        )
        if exact is None:
            return None
        center = world_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if center is None:
            return None
        yaw = np.deg2rad(float(center.transform.rotation.yaw))
        right_xy = np.array([-np.sin(yaw), np.cos(yaw)], dtype=float)
        delta_xy = np.array([
            location.x - center.transform.location.x,
            location.y - center.transform.location.y,
        ], dtype=float)
        return float(delta_xy @ right_xy)

    @staticmethod
    def _route_control_context(route_agent, ego, world) -> dict:
        """读取BasicAgent当前阻塞证据，仅用于诊断，不改变其控制决策。"""

        light = getattr(route_agent, "_last_traffic_light", None)
        light_state = None if light is None else str(light.state).split(".")[-1]
        light_id = None if light is None else int(light.id)
        ego_transform = ego.get_transform()
        ego_location = ego_transform.location
        forward = ego_transform.get_forward_vector()
        nearest_front = None
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id == ego.id:
                continue
            location = actor.get_location()
            dx, dy = location.x - ego_location.x, location.y - ego_location.y
            longitudinal = dx * forward.x + dy * forward.y
            lateral = abs(-dx * forward.y + dy * forward.x)
            if longitudinal <= 0.0 or lateral > 4.0:
                continue
            distance = float(np.hypot(dx, dy))
            if nearest_front is None or distance < nearest_front["distance_m"]:
                nearest_front = {
                    "actor_id": int(actor.id),
                    "type_id": actor.type_id,
                    "distance_m": distance,
                    "longitudinal_m": float(longitudinal),
                    "lateral_m": float(lateral),
                }
        # 调用本机 BasicAgent 使用的同一障碍检测函数，记录真正触发制动的 actor。
        # 这是只读诊断，不修改规划器状态，也不改变控制输出。
        velocity = ego.get_velocity()
        ego_speed_mps = float(np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2))
        max_vehicle_distance = (
            float(route_agent._base_vehicle_threshold)
            + float(route_agent._speed_ratio) * ego_speed_mps
        )
        affected, hazard_actor, hazard_distance = route_agent._vehicle_obstacle_detected(
            world.get_actors().filter("vehicle.*"),
            max_vehicle_distance,
        )
        vehicle_hazard = None
        if affected and hazard_actor is not None:
            hazard_transform = hazard_actor.get_transform()
            hazard_location = hazard_transform.location
            dx = hazard_location.x - ego_location.x
            dy = hazard_location.y - ego_location.y
            hazard_velocity = hazard_actor.get_velocity()
            hazard_speed_kmh = 3.6 * float(np.sqrt(
                hazard_velocity.x ** 2 + hazard_velocity.y ** 2 + hazard_velocity.z ** 2
            ))
            hazard_waypoint = world.get_map().get_waypoint(hazard_location)
            hazard_light = hazard_actor.get_traffic_light()
            vehicle_hazard = {
                "actor_id": int(hazard_actor.id),
                "type_id": hazard_actor.type_id,
                "distance_m": float(hazard_distance),
                "longitudinal_m": float(dx * forward.x + dy * forward.y),
                "lateral_m": float(abs(-dx * forward.y + dy * forward.x)),
                "speed_kmh": hazard_speed_kmh,
                "road_id": int(hazard_waypoint.road_id),
                "lane_id": int(hazard_waypoint.lane_id),
                "is_junction": bool(hazard_waypoint.is_junction),
                "traffic_light_id": None if hazard_light is None else int(hazard_light.id),
                "traffic_light_state": (
                    None if hazard_light is None else str(hazard_light.state).split(".")[-1]
                ),
                "detection_gate_m": max_vehicle_distance,
            }
        remaining = len(route_agent.get_local_planner().get_plan())
        return {
            "active_traffic_light_id": light_id,
            "active_traffic_light_state": light_state,
            "nearest_front_vehicle": nearest_front,
            "basic_agent_vehicle_hazard": vehicle_hazard,
            "remaining_route_waypoints": remaining,
            "vehicle_truth_avoidance_enabled": not bool(
                getattr(route_agent, "_ignore_vehicles", False)
            ),
            "traffic_light_truth_control_enabled": not bool(
                getattr(route_agent, "_ignore_traffic_lights", False)
            ),
        }

    def run(self) -> Path:
        import carla
        import pygame

        cfg = self.config
        # Stage6.7可选依赖必须绑定在整个在线run作用域内；后续运行提示、
        # 每帧控制仲裁、结构化输出和summary均读取同一个监督器实例。
        supervisor = self.longitudinal_supervisor
        aeb_shadow = self.aeb_shadow_evaluator
        dataset_recorder = self.dataset_recorder
        client = carla.Client(cfg.runtime.host, cfg.runtime.port)
        client.set_timeout(cfg.runtime.timeout_s)
        world = client.get_world()
        if cfg.runtime.town and cfg.runtime.town not in world.get_map().name:
            print(
                f"[CARLA] 显式切换地图：{world.get_map().name} -> {cfg.runtime.town}，"
                f"最多等待 {cfg.runtime.map_load_timeout_s:.0f}s ...",
                flush=True,
            )
            # load_world 会重启仿真世界；30s 对 Windows 首次载图经常不够。
            client.set_timeout(cfg.runtime.map_load_timeout_s)
            world = client.load_world(cfg.runtime.town)
            client.set_timeout(cfg.runtime.timeout_s)
        else:
            print(f"[CARLA] 沿用当前地图：{world.get_map().name}（未执行切图）", flush=True)
        original_weather = world.get_weather()
        if not hasattr(carla.WeatherParameters, cfg.runtime.weather):
            raise ValueError(f"未知 CARLA weather: {cfg.runtime.weather}")
        world.set_weather(getattr(carla.WeatherParameters, cfg.runtime.weather))
        original_settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(cfg.runtime.traffic_manager_port)
        sensor_rig = worker = writer = None
        pygame.init()
        screen = None
        if cfg.runtime.render_dashboard:
            screen = pygame.display.set_mode(
                (cfg.dashboard.width_px, cfg.dashboard.height_px)
            )
        else:
            # 自动评测仍保留Pygame事件泵与Windows全局Esc，但不创建大窗口、
            # 不合成1600x900仪表盘。CARLA传感器本身不受此开关影响。
            hidden_flag = getattr(pygame, "HIDDEN", 0)
            screen = pygame.display.set_mode((1, 1), flags=hidden_flag)
        stage_label = cfg.runtime.stage_id.replace("_", ".").upper()
        pygame.display.set_caption(
            f"{stage_label} CARLA Fusion Dashboard"
            if cfg.runtime.render_dashboard else f"{stage_label} headless evaluation"
        )
        clock = pygame.time.Clock()
        try:
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = cfg.runtime.fixed_delta_s
            world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(cfg.runtime.seed)
            traffic_manager.global_percentage_speed_difference(
                cfg.driving.traffic_manager_speed_difference_pct
            )

            if cfg.runtime.world_startup_warmup_ticks > 0:
                print(
                    f"[地图预热] 创建车辆/传感器前推进 "
                    f"{cfg.runtime.world_startup_warmup_ticks} 个同步帧 ...",
                    flush=True,
                )
                last_warmup_frame = None
                for _ in range(cfg.runtime.world_startup_warmup_ticks):
                    last_warmup_frame = world.tick()
                print(
                    f"[地图预热] 完成 | frame={last_warmup_frame}",
                    flush=True,
                )

            ego, spawn_points = self._spawn_ego(world, carla)
            if self.scenario_controller is not None:
                bind_runtime = getattr(self.scenario_controller, "bind_runtime", None)
                if bind_runtime is not None:
                    bind_runtime(
                        client=client,
                        traffic_manager_port=cfg.runtime.traffic_manager_port,
                    )
                # 同步模式下try_spawn_actor返回后，actor位姿可能仍是(0,0,0)，直到
                # 下一次tick才绑定请求的出生位姿。场景必须基于初始化后的ego布置目标。
                world.tick()
                scenario_actors = self.scenario_controller.setup(world, ego, carla)
                self.created_actors.extend(list(scenario_actors or []))
            else:
                # try_spawn_actor 返回时同步世界尚未提交出生位姿；若立刻读取
                # ego.get_location()，部分地图会得到(0,0,0)，进而把“近场NPC”
                # 全部生成在地图原点。先推进一帧再按真实ego位置选择出生点。
                world.tick()
            self._spawn_npcs(client, world, carla, spawn_points, ego)
            sensor_rig = CarlaSensorRig(world, ego, carla, cfg.sensors)

            # 先完成传感器首帧，再加载 GPU 检测器，避免两者同时冷启动争抢资源。
            print(
                f"[传感器预热] 等待六相机 + LiDAR + 五Radar，"
                f"首帧最多 {cfg.sensors.startup_sensor_wait_s:.0f}s ...",
                flush=True,
            )
            warmup_frame = None
            for attempt in range(1, cfg.sensors.sensor_warmup_retries + 1):
                candidate_frame = world.tick()
                try:
                    warmup_frame = sensor_rig.wait_for_frame(
                        candidate_frame, cfg.sensors.startup_sensor_wait_s
                    )
                    break
                except TimeoutError as exc:
                    print(
                        f"[传感器预热] 第 {attempt}/{cfg.sensors.sensor_warmup_retries} 次未就绪：{exc}",
                        flush=True,
                    )
                    if attempt == cfg.sensors.sensor_warmup_retries:
                        raise
            print(
                f"[传感器预热] 完成 | frame={warmup_frame.frame} | "
                f"LiDAR={len(warmup_frame.lidar.points_world_carla)} | "
                f"Radar={sum(len(x.returns) for x in warmup_frame.radars.values())}",
                flush=True,
            )
            if cfg.runtime.enable_perception:
                worker_factory = (
                    self.perception_worker_factory or LatestOnlyPerceptionWorker
                )
                worker = worker_factory(cfg.perception, cfg.sensors, cfg.tracker)
                print(
                    f"[检测器预热] 加载 {cfg.perception.detector_model}，"
                    f"最多等待 {cfg.perception.detector_startup_timeout_s:.0f}s ...",
                    flush=True,
                )
                worker.wait_until_ready(cfg.perception.detector_startup_timeout_s)
                warmup_hook = getattr(worker, "warmup_with_frame", None)
                if callable(warmup_hook):
                    warmup_pose = carla_transform_to_internal_pose(ego.get_transform())
                    warmup_velocity_carla = ego.get_velocity()
                    warmup_velocity = np.array([
                        warmup_velocity_carla.x, -warmup_velocity_carla.y
                    ], dtype=float)
                    warmup_elapsed = warmup_hook(
                        PerceptionJob(
                            sensor_frame=warmup_frame,
                            ego_pose_xyyaw=warmup_pose,
                            ego_z_m=float(ego.get_transform().location.z),
                            ego_velocity_global=warmup_velocity,
                            submitted_wall_s=time.perf_counter(),
                        ),
                        cfg.perception.detector_startup_timeout_s,
                    )
                    print(
                        f"[检测器真实帧预热] 完成 | latency={warmup_elapsed:.2f}s",
                        flush=True,
                    )
                warmup_s = worker.startup_diagnostics.get(
                    "detector_warmup_inference_s", 0.0
                )
                print(f"[检测器预热] 完成 | 实际前向预热={warmup_s:.2f}s", flush=True)
            else:
                print("[控制测试] 已跳过检测器与跟踪，只验证驾驶控制、车辆物理与评测链", flush=True)
            tracker = StreamingFusionTracker(cfg.tracker)
            dashboard = (
                Stage61Dashboard(cfg.dashboard)
                if cfg.runtime.render_dashboard else None
            )
            controller = ManualKeyboardController(
                pygame, use_global_keyboard=cfg.runtime.global_keyboard_input
            )
            print(f"[键盘后端] {controller.backend_name}", flush=True)
            if cfg.driving.wait_for_enter:
                print("[等待运行] 点击仪表盘窗口并按 Enter，之后才开始计时", flush=True)
                if not controller.wait_for_start(screen, stage_label, cfg.driving.mode):
                    print("[结束] 用户在计时前取消")
                    return cfg.runtime.output_root.resolve()
            route_agent, initial_route_waypoints, route_destination = (
                self._prepare_automatic_drive(world, ego, spawn_points, traffic_manager)
            )
            writer = Stage61OutputWriter(cfg)
            scene_id = writer.run_dir.name
            control_contract = (
                "BASELINE"
                if supervisor is None and cfg.runtime.stage_id in {
                    "stage6_7", "stage6_8", "stage6_9", "stage6_10", "stage6_11",
                    "stage6_12", "stage6_13", "stage6_14", "stage6_15",
                }
                else ("SHADOW" if supervisor is None else supervisor.mode.upper())
            )
            if cfg.driving.mode == "manual":
                print(
                    f"[运行] MANUAL + {control_contract} | W/S/A/D 驾驶，"
                    "Q 倒挡，Space 手刹，Esc 结束"
                )
            else:
                print(
                    f"[运行] {cfg.driving.mode.upper()} + {control_contract} | "
                    + (
                        "安全监督器可限制油门/制动，不接管转向，Esc 可提前结束"
                        if supervisor is not None and supervisor.mode == "active"
                        else "风险只评估、不接管控制，Esc 可提前结束"
                    )
                )
            print(f"[输出] {writer.run_dir}")
            print(
                "[显示] 仪表盘=开"
                if cfg.runtime.render_dashboard
                else "[显示] 仪表盘=关（传感器/感知/控制/结构化输出保持开启）"
            )
            print("[说明] 检测首次加载可能下载/载入权重；在线推理由独立工作线程执行。")

            start_sim = None
            start_wall = time.perf_counter()
            last_perception_frame = -1
            last_submitted_camera_generation = None
            path = deque(maxlen=cfg.dashboard.path_history_length)
            collision_counter = CollisionIncidentCounter(
                cfg.evaluation.collision_dedup_window_s
            )
            collision_count = 0
            max_risk = 0.0
            min_ttc = float("inf")
            min_clearance = float("inf")
            distance_m = 0.0
            previous_pose = None
            fusion_updates = 0
            latency_values = []
            frame_count = 0
            speed_values = []
            detector_error = None
            sensor_timeouts = 0
            consecutive_sensor_timeouts = 0
            route_completion_pct = None
            route_completed = False
            termination_reason = "unknown"
            metrics = ScenarioMetricsAccumulator(cfg.evaluation)
            latency_metrics = LatencyMetricsAccumulator()
            oracle_evaluator = (
                OracleTrackEvaluator(cfg.evaluation)
                if cfg.evaluation.enable_oracle_perception_metrics else None
            )
            radar_refinement_frames = 0
            radar_refinement_accepted = 0
            world_map = world.get_map()
            last_progress_print_s = -5.0
            control_reason_frames: dict[str, int] = {}
            perception_diagnostic_totals = {
                "backend_frames": 0,
                "backend_raw_detections": 0,
                "backend_trackable_raw": 0,
                "accepted_detections": 0,
                "lidar_lifted_candidates": 0,
                "radar_only_lifted_candidates": 0,
                "radar_only_lift_attempts": 0,
                "radar_only_lift_decisions": {},
                "class_counts": {},
                "bevfusion_shadow": {
                    "frames": 0,
                    "backend_raw_detections": 0,
                    "backend_trackable_raw": 0,
                    "accepted_detections": 0,
                    "class_counts": {},
                },
            }
            last_bevfusion_diagnostic_frame = -1
            scenario_elapsed_s = 0.0
            latest_aeb_shadow_decision = None
            aeb_advisory_frames = 0
            aeb_advisory_target_ids: set[int] = set()

            while True:
                frame_wall_started = time.perf_counter()
                observed_manual, quit_requested = controller.read(cfg.runtime.fixed_delta_s)
                if self.scenario_controller is not None:
                    self.scenario_controller.before_tick(scenario_elapsed_s, carla)
                control_context = {}
                if cfg.driving.mode == "manual":
                    control_values = observed_manual
                elif cfg.driving.mode == "route_agent":
                    route_control = route_agent.run_step()
                    control_values = self._control_to_dict(route_control)
                    control_context = self._route_control_context(route_agent, ego, world)
                else:
                    # Traffic Manager 在 world.tick 内更新控制；此处先记录上帧回读值。
                    control_values = self._control_to_dict(ego.get_control())
                base_control_values = dict(control_values)
                if cfg.driving.mode == "route_agent":
                    ego_velocity_for_governor = ego.get_velocity()
                    current_speed_mps = float(np.linalg.norm([
                        ego_velocity_for_governor.x,
                        ego_velocity_for_governor.y,
                    ]))
                    base_control_values = self._smooth_route_longitudinal_control(
                        base_control_values, current_speed_mps=current_speed_mps
                    )
                    control_values = dict(base_control_values)
                supervisor_application = None
                if supervisor is not None:
                    control_values, supervisor_application = supervisor.arbitrate(
                        base_control_values, cfg.driving.mode
                    )
                control_values, aeb_advisory_application = (
                    self._apply_aeb_throttle_advisory(
                        control_values,
                        latest_aeb_shadow_decision,
                        self.aeb_advisory_throttle_cap,
                        active=(
                            supervisor is not None
                            and supervisor.mode == "active"
                            and cfg.driving.mode != "autopilot"
                        ),
                    )
                )
                if aeb_advisory_application["applied"]:
                    aeb_advisory_frames += 1
                    target_id = aeb_advisory_application["target_track_id"]
                    if target_id is not None:
                        aeb_advisory_target_ids.add(int(target_id))
                first_active_input = False
                if (
                    cfg.driving.mode == "manual"
                    and not controller.has_reported_input
                    and (control_values["throttle"] > 0.0
                         or control_values["brake"] > 0.0
                         or abs(control_values["steer"]) > 0.01)
                ):
                    print(
                        f"[键盘] 已收到控制：throttle={control_values['throttle']:.1f} "
                        f"brake={control_values['brake']:.1f} "
                        f"steer={control_values['steer']:+.2f}",
                        flush=True,
                    )
                    controller.has_reported_input = True
                    first_active_input = True
                if cfg.driving.mode != "autopilot":
                    ego.apply_control(carla.VehicleControl(**control_values))
                sensor_wait_started = time.perf_counter()
                frame_id = world.tick()
                if cfg.driving.mode == "autopilot":
                    control_values = self._control_to_dict(ego.get_control())
                    base_control_values = dict(control_values)
                    if supervisor is not None:
                        # Traffic Manager在tick内持有控制权；当前只记录抑制原因，
                        # 不把两个控制器同时写入车辆。
                        _, supervisor_application = supervisor.arbitrate(
                            base_control_values, cfg.driving.mode
                        )
                if first_active_input:
                    # CARLA在world.tick后才把本轮命令应用到物理帧，此时回读才有意义。
                    applied = ego.get_control()
                    print(
                        f"[车辆控制回读] throttle={applied.throttle:.1f} "
                        f"brake={applied.brake:.1f} steer={applied.steer:+.2f} "
                        f"handbrake={applied.hand_brake} manual_gear={applied.manual_gear_shift}",
                        flush=True,
                    )
                try:
                    sensor_frame = sensor_rig.wait_for_frame(frame_id)
                    latency_metrics.observe(
                        "sensor_tick_and_wait", time.perf_counter() - sensor_wait_started
                    )
                    consecutive_sensor_timeouts = 0
                except TimeoutError as exc:
                    sensor_timeouts += 1
                    consecutive_sensor_timeouts += 1
                    if frame_count == 0 and start_sim is None:
                        # CARLA传感器在长时间同步暂停（等待Enter/模型加载）后的首tick
                        # 通常只恢复时钟、不产出measurement；第二tick起正常，不算故障。
                        print(
                            f"[传感器恢复] 已跳过计时后的首个空帧 frame={frame_id}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[传感器警告] frame={frame_id} 暂时未齐，"
                            f"连续 {consecutive_sensor_timeouts}/3 次：{exc}",
                            flush=True,
                        )
                    if consecutive_sensor_timeouts >= 3:
                        raise RuntimeError("传感器连续3帧超时，停止运行以保留诊断现场") from exc
                    clock.tick(cfg.runtime.video_fps)
                    continue
                sim_time = sensor_frame.timestamp_s
                if start_sim is None:
                    start_sim = sim_time
                elapsed_s = sim_time - start_sim
                scenario_elapsed_s = elapsed_s
                if self.scenario_controller is not None:
                    self.scenario_controller.observe(elapsed_s)
                transform = ego.get_transform()
                ego_pose = carla_transform_to_internal_pose(transform)
                ego_z_m = float(transform.location.z)
                velocity = ego.get_velocity()
                ego_velocity = np.array([velocity.x, -velocity.y], dtype=float)
                angular_velocity = ego.get_angular_velocity()
                # 内部坐标yaw=-CARLA yaw，因此z角速度也必须反号并由deg/s转rad/s。
                ego_yaw_rate_radps = -np.deg2rad(float(angular_velocity.z))
                speed_kmh = float(np.linalg.norm(ego_velocity) * 3.6)
                if dataset_recorder is not None:
                    recorder_started = time.perf_counter()
                    dataset_recorder.record(
                        sensor_frame, world=world, ego_vehicle=ego
                    )
                    latency_metrics.observe(
                        "offline_dataset_recording",
                        time.perf_counter() - recorder_started,
                    )
                control_reason = None
                if cfg.driving.mode == "route_agent":
                    light_state = control_context.get("active_traffic_light_state")
                    front_vehicle = control_context.get("basic_agent_vehicle_hazard")
                    if supervisor_application is not None and supervisor_application.applied:
                        control_reason = "risk_supervisor"
                    elif control_values["brake"] >= cfg.driving.route_max_brake - 0.05:
                        if light_state == "Red":
                            control_reason = "red_light"
                        elif front_vehicle is not None:
                            control_reason = "front_vehicle"
                        else:
                            control_reason = "emergency_unknown"
                    elif speed_kmh < 1.0 and control_values["throttle"] > 0.1:
                        control_reason = "stalled_under_throttle"
                    else:
                        control_reason = "route_following"
                    control_context["reason"] = control_reason
                    control_reason_frames[control_reason] = (
                        control_reason_frames.get(control_reason, 0) + 1
                    )
                speed_values.append(speed_kmh)
                path.append(ego_pose[:2].copy())
                if previous_pose is not None:
                    distance_m += float(np.linalg.norm(ego_pose[:2] - previous_pose[:2]))
                previous_pose = ego_pose.copy()
                for event in sensor_frame.collisions:
                    is_new_incident, incident_id = collision_counter.observe(
                        event.other_actor_id, event.timestamp_s
                    )
                    collision_count = collision_counter.incident_count
                    writer.add_collision(
                        event,
                        is_new_incident=is_new_incident,
                        incident_id=incident_id,
                    )

                camera_generation = tuple(
                    sensor_frame.cameras[name].frame for name in CAMERA_CHANNELS
                )
                multirate_camera = (
                    cfg.sensors.camera_sensor_tick_s is not None
                    and cfg.sensors.camera_sensor_tick_s > cfg.sensors.sensor_tick_s
                )
                submit_perception = (
                    worker is not None
                    and (
                        (
                            multirate_camera
                            and len(set(camera_generation)) == 1
                            and camera_generation != last_submitted_camera_generation
                        )
                        or (
                            not multirate_camera
                            and frame_count % cfg.perception.detector_stride_frames == 0
                        )
                    )
                )
                if submit_perception:
                    worker.submit(PerceptionJob(
                        sensor_frame=sensor_frame,
                        ego_pose_xyyaw=ego_pose.copy(),
                        ego_z_m=ego_z_m,
                        ego_velocity_global=ego_velocity.copy(),
                        submitted_wall_s=time.perf_counter(),
                    ))
                    last_submitted_camera_generation = camera_generation
                perception = worker.latest() if worker is not None else None
                camera_boxes = {}
                perception_latency = None
                result_age = None
                perception_diagnostics = {}
                if perception is not None:
                    detector_error = perception.error
                    camera_boxes = perception.camera_boxes
                    perception_latency = perception.latency_s
                    result_age = max(sim_time - perception.timestamp_s, 0.0)
                    perception_diagnostics = dict(perception.diagnostics or {})
                    if result_age > cfg.tracker.max_prediction_age_s:
                        camera_boxes = {}
                    if perception.frame > last_perception_frame and perception.error is None:
                        tracker.update(
                            list(perception.detections), perception.timestamp_s,
                            perception.ego_velocity_global,
                        )
                        last_perception_frame = perception.frame
                        fusion_updates += 1
                        if perception_diagnostics:
                            perception_diagnostic_totals["backend_frames"] += 1
                            for key in (
                                "backend_raw_detections",
                                "backend_trackable_raw",
                                "accepted_detections",
                                "lidar_lifted_candidates",
                                "radar_only_lifted_candidates",
                            ):
                                perception_diagnostic_totals[key] += int(
                                    perception_diagnostics.get(key, 0)
                                )
                            radar_evidence = perception_diagnostics.get(
                                "radar_only_lift_evidence", []
                            )
                            perception_diagnostic_totals[
                                "radar_only_lift_attempts"
                            ] += len(radar_evidence)
                            decision_totals = perception_diagnostic_totals[
                                "radar_only_lift_decisions"
                            ]
                            for evidence in radar_evidence:
                                decision = str(evidence.get("decision", "unknown"))
                                decision_totals[decision] = (
                                    decision_totals.get(decision, 0) + 1
                                )
                            totals = perception_diagnostic_totals["class_counts"]
                            for class_name, count in perception_diagnostics.get(
                                "class_counts", {}
                            ).items():
                                totals[class_name] = totals.get(class_name, 0) + int(count)
                            bev_diag = perception_diagnostics.get("bevfusion_shadow")
                            if isinstance(bev_diag, dict):
                                bev_frame = int(bev_diag.get("frame", -1))
                                if bev_frame > last_bevfusion_diagnostic_frame:
                                    bev_totals = perception_diagnostic_totals[
                                        "bevfusion_shadow"
                                    ]
                                    bev_totals["frames"] += 1
                                    for key in (
                                        "backend_raw_detections",
                                        "backend_trackable_raw",
                                        "accepted_detections",
                                    ):
                                        bev_totals[key] += int(bev_diag.get(key, 0))
                                    for class_name, count in bev_diag.get(
                                        "class_counts", {}
                                    ).items():
                                        bev_classes = bev_totals["class_counts"]
                                        bev_classes[class_name] = (
                                            bev_classes.get(class_name, 0) + int(count)
                                        )
                                    last_bevfusion_diagnostic_frame = bev_frame
                        latency_values.append(perception.latency_s)
                        latency_metrics.observe("perception_queue", perception.queue_latency_s)
                        latency_metrics.observe(
                            "perception_preprocess", perception.preprocess_latency_s
                        )
                        latency_metrics.observe("perception_detector", perception.detector_latency_s)
                        latency_metrics.observe("perception_fusion", perception.fusion_latency_s)
                        latency_metrics.observe("perception_total", perception.latency_s)

                tracking_started = time.perf_counter()
                tracks = tracker.snapshots(sim_time)
                all_radar_returns = [
                    radar
                    for sample in sensor_frame.radars.values()
                    for radar in sample.returns
                ]
                radar_started = time.perf_counter()
                tracks, radar_refinement = tracker.refine_snapshots_with_radar(
                    tracks, all_radar_returns, ego_velocity
                )
                latency_metrics.observe(
                    "live_radar_refinement", time.perf_counter() - radar_started
                )
                if radar_refinement["refined_tracks"] > 0:
                    radar_refinement_frames += 1
                    radar_refinement_accepted += radar_refinement["accepted_returns"]
                safety = assess_tracks(tracks, ego_pose, ego_velocity, cfg.safety)
                supervisor_decision = None
                if supervisor is not None:
                    supervisor_decision = supervisor.evaluate(
                        timestamp_s=sim_time,
                        safety=safety,
                        tracks=tracks,
                        ego_velocity_ego=global_vector_to_ego(
                            ego_velocity, float(ego_pose[2])
                        ),
                        fusion_age_s=result_age,
                        dt_s=cfg.runtime.fixed_delta_s,
                    )
                    if supervisor_decision.new_intervention_event:
                        writer.add_supervisor_event({
                            "frame": frame_id,
                            "timestamp_s": sim_time,
                            "elapsed_s": elapsed_s,
                            **supervisor_decision.to_dict(),
                        })
                aeb_shadow_decision = None
                if aeb_shadow is not None:
                    aeb_started = time.perf_counter()
                    aeb_shadow_decision = aeb_shadow.evaluate(
                        timestamp_s=sim_time,
                        tracks=tracks,
                        ego_pose_xyyaw=ego_pose,
                        ego_velocity_global=ego_velocity,
                        ego_yaw_rate_radps=ego_yaw_rate_radps,
                        reference_intervention_requested=(
                            None if supervisor_decision is None
                            else supervisor_decision.intervention_requested
                        ),
                    )
                    latency_metrics.observe(
                        "autoware_aeb_shadow", time.perf_counter() - aeb_started
                    )
                    if aeb_shadow_decision.new_trigger_event:
                        writer.add_aeb_shadow_event({
                            "frame": frame_id,
                            "timestamp_s": sim_time,
                            "elapsed_s": elapsed_s,
                            **aeb_shadow_decision.to_dict(),
                        })
                    latest_aeb_shadow_decision = aeb_shadow_decision
                latency_metrics.observe(
                    "tracking_prediction_and_risk", time.perf_counter() - tracking_started
                )
                peak = safety[0] if safety else None
                highest_risk = peak.risk_score if peak else 0.0
                max_risk = max(max_risk, highest_risk)
                if peak and peak.approx_ttc_s is not None:
                    min_ttc = min(min_ttc, peak.approx_ttc_s)
                if peak:
                    min_clearance = min(min_clearance, peak.min_clearance_m)
                oracle_frame = None
                if oracle_evaluator is not None:
                    oracle_started = time.perf_counter()
                    truth_objects = collect_vehicle_truth(
                        world,
                        ego,
                        sensor_frame.lidar.points_world_carla,
                    )
                    oracle_frame = oracle_evaluator.update(
                        timestamp_s=sim_time,
                        tracks=tracks,
                        truth_objects=truth_objects,
                        ego_position_global=ego_pose[:2],
                    )
                    latency_metrics.observe(
                        "oracle_evaluation_only", time.perf_counter() - oracle_started
                    )
                if route_agent is not None:
                    route_completed = bool(route_agent.done())
                    remaining_waypoints = len(route_agent.get_local_planner().get_plan())
                    route_completion_pct = route_completion_percent(
                        initial_route_waypoints, remaining_waypoints, route_completed
                    )
                acceleration = ego.get_acceleration()
                acceleration_xy = np.array([acceleration.x, -acceleration.y], dtype=float)
                ego_forward_xy = np.array(
                    [np.cos(float(ego_pose[2])), np.sin(float(ego_pose[2]))],
                    dtype=float,
                )
                longitudinal_acceleration_mps2 = float(
                    np.dot(acceleration_xy, ego_forward_xy)
                )
                lane_offset = self._lane_offset_m(world_map, ego.get_location(), carla)
                metrics.update(
                    speed_kmh=speed_kmh,
                    acceleration_xy=acceleration_xy,
                    steering=control_values["steer"],
                    lane_offset_m=lane_offset,
                    highest_risk_score=highest_risk,
                    fusion_age_s=result_age,
                    dt_s=cfg.runtime.fixed_delta_s,
                    longitudinal_acceleration_mps2=longitudinal_acceleration_mps2,
                )
                if (
                    cfg.runtime.stage_id != "stage6_1"
                    and elapsed_s - last_progress_print_s >= 5.0
                ):
                    completion_text = (
                        "--" if route_completion_pct is None else f"{route_completion_pct:.1f}%"
                    )
                    supervisor_progress = (
                        "" if supervisor_decision is None else
                        f"action={supervisor_decision.actionable_collision_risk:.1f} "
                        f"state={supervisor_decision.state} | "
                    )
                    print(
                        f"[进度] t={elapsed_s:.1f}s | speed={speed_kmh:.1f}km/h | "
                        f"distance={distance_m:.1f}m | completion={completion_text} | "
                        f"reason={control_reason or '--'} | tracks={len(tracks)} | "
                        f"risk={highest_risk:.1f} | {supervisor_progress}"
                        f"collision={collision_count}",
                        flush=True,
                    )
                    last_progress_print_s = elapsed_s
                rgb_frame = None
                if dashboard is not None:
                    fps = clock.get_fps()
                    control_state = ControlState(**control_values)
                    dashboard_state = DashboardState(
                        sensor_frame=sensor_frame,
                        camera_boxes=camera_boxes,
                        tracks=tracks,
                        safety=safety,
                        ego_pose_xyyaw=ego_pose,
                        ego_velocity_global=ego_velocity,
                        ego_path_global=list(path),
                        speed_kmh=speed_kmh,
                        control=control_state,
                        scene_id=scene_id,
                        elapsed_s=elapsed_s,
                        fps=fps,
                        fusion_result_age_s=result_age,
                        perception_latency_s=perception_latency,
                        detector_error=detector_error,
                        collision_count=collision_count,
                        stage_label=stage_label,
                        drive_mode=cfg.driving.mode,
                        control_reason=control_reason,
                        supervisor=(
                            None if supervisor_decision is None else {
                                **supervisor_decision.to_dict(),
                                "application": (
                                    None if supervisor_application is None
                                    else supervisor_application.to_dict()
                                ),
                            }
                        ),
                    )
                    dashboard_started = time.perf_counter()
                    rgb_frame = dashboard.render(dashboard_state)
                    self._pygame_show(pygame, screen, rgb_frame)
                    latency_metrics.observe(
                        "dashboard_render_and_show",
                        time.perf_counter() - dashboard_started,
                    )
                record = {
                    "schema_version": f"{cfg.runtime.stage_id.replace('_', '.')}-v1",
                    "frame": frame_id,
                    "timestamp_s": sim_time,
                    "elapsed_s": elapsed_s,
                    "scene_id": scene_id,
                    "mode": (
                        f"{cfg.driving.mode}_baseline"
                        if supervisor is None and cfg.runtime.stage_id in {
                            "stage6_7", "stage6_8", "stage6_9", "stage6_10", "stage6_11",
                            "stage6_12", "stage6_13",
                            "stage6_14",
                            "stage6_15",
                        }
                        else f"{cfg.driving.mode}_shadow"
                        if supervisor is None
                        else f"{cfg.driving.mode}_{supervisor.mode}"
                    ),
                    "scenario_name": cfg.runtime.scenario_name,
                    "deterministic_scenario": (
                        None if self.scenario_controller is None
                        else self.scenario_controller.frame_state()
                    ),
                    "ego": {
                        "pose_global_xyyaw": ego_pose.tolist(),
                        "velocity_global_mps": ego_velocity.tolist(),
                        "speed_kmh": speed_kmh,
                    },
                    "base_control": base_control_values,
                    "control": control_values,
                    "control_context": control_context,
                    "safety_supervisor": (
                        None if supervisor_decision is None else {
                            **supervisor_decision.to_dict(),
                            "application": (
                                None if supervisor_application is None
                                else supervisor_application.to_dict()
                            ),
                        }
                    ),
                    "autoware_aeb_shadow": (
                        None if aeb_shadow_decision is None
                        else aeb_shadow_decision.to_dict()
                    ),
                    "autoware_aeb_advisory": aeb_advisory_application,
                    "sensor_counts": {
                        "cameras": len(sensor_frame.cameras),
                        "lidar": len(sensor_frame.lidar.points_world_carla),
                        "radar": sum(len(sample.returns) for sample in sensor_frame.radars.values()),
                    },
                    "fusion_result_age_s": result_age,
                    "perception_latency_s": perception_latency,
                    "perception_diagnostics": perception_diagnostics,
                    "latency_s": {
                        "frame_before_output": time.perf_counter() - frame_wall_started,
                        "perception_queue": (
                            None if perception is None else perception.queue_latency_s
                        ),
                        "perception_preprocess": (
                            None if perception is None else perception.preprocess_latency_s
                        ),
                        "perception_detector": (
                            None if perception is None else perception.detector_latency_s
                        ),
                        "perception_fusion": (
                            None if perception is None else perception.fusion_latency_s
                        ),
                    },
                    "live_radar_refinement": radar_refinement,
                    "tracks": [track.to_dict() for track in tracks],
                    "safety": [item.to_dict() for item in safety],
                    "oracle_evaluation": oracle_frame,
                    "highest_risk_score": highest_risk,
                    "collision_count": collision_count,
                    "collision_contact_count": collision_counter.contact_count,
                    "detector_error": detector_error,
                    "evaluation": {
                        "route_completion_pct": route_completion_pct,
                        "route_completed": route_completed,
                        "lane_offset_m": lane_offset,
                        "acceleration_mps2": float(np.linalg.norm(acceleration_xy)),
                    },
                }
                output_started = time.perf_counter()
                writer.write_frame(rgb_frame, record)
                latency_metrics.observe(
                    "structured_output_and_video", time.perf_counter() - output_started
                )
                latency_metrics.observe(
                    "frame_processing_before_rate_limit",
                    time.perf_counter() - frame_wall_started,
                )
                frame_count += 1
                clock.tick(cfg.runtime.video_fps)
                latency_metrics.observe(
                    "frame_loop_with_rate_limit", time.perf_counter() - frame_wall_started
                )
                if quit_requested:
                    termination_reason = "user_escape"
                    break
                # Only controllers that explicitly expose ``terminal_status``
                # may terminate the shared session. Stage6.8 deterministic
                # scenarios therefore keep their fixed-duration contract.
                if self.scenario_controller is not None:
                    terminal_status_fn = getattr(
                        self.scenario_controller, "terminal_status", None
                    )
                    official_terminal_status = (
                        terminal_status_fn()
                        if callable(terminal_status_fn)
                        else None
                    )
                    if official_terminal_status is not None:
                        termination_reason = (
                            f"scenario_tree_{official_terminal_status.lower()}"
                        )
                        print(
                            "[官方场景终止] "
                            f"tree={official_terminal_status} | sim={elapsed_s:.1f}s；"
                            "停止继续累计无关路线帧",
                            flush=True,
                        )
                        break
                if (
                    route_completed
                    and cfg.driving.mode == "route_agent"
                    and cfg.driving.stop_on_route_completion
                ):
                    termination_reason = "route_completed"
                    break
                if cfg.runtime.duration_s > 0 and elapsed_s >= cfg.runtime.duration_s:
                    termination_reason = "duration_reached"
                    break

            wall_duration = time.perf_counter() - start_wall
            evaluation = metrics.finalize(
                drive_mode=cfg.driving.mode,
                route_completion_pct=route_completion_pct,
                route_completed=route_completed,
                collision_count=collision_count,
                sensor_timeout_frames=sensor_timeouts,
                simulation_duration_s=elapsed_s,
                wall_duration_s=wall_duration,
                distance_m=distance_m,
                detector_error=detector_error,
                termination_reason=termination_reason,
            )
            oracle_summary = (
                None if oracle_evaluator is None else oracle_evaluator.finalize()
            )
            latency_summary = latency_metrics.finalize()
            dataset_summary = (
                None if dataset_recorder is None else dataset_recorder.finalize()
            )
            summary = {
                "schema_version": f"{cfg.runtime.stage_id.replace('_', '.')}-v1",
                "scene_id": scene_id,
                "scenario_name": cfg.runtime.scenario_name,
                "seed": cfg.runtime.seed,
                "mode": (
                    f"{cfg.driving.mode}_baseline"
                    if supervisor is None and cfg.runtime.stage_id in {
                        "stage6_7", "stage6_8", "stage6_9", "stage6_10", "stage6_11",
                        "stage6_12", "stage6_13", "stage6_14", "stage6_15",
                    }
                    else f"{cfg.driving.mode}_shadow"
                    if supervisor is None
                    else f"{cfg.driving.mode}_{supervisor.mode}"
                ),
                "simulation_duration_s": elapsed_s,
                "wall_duration_s": wall_duration,
                "frames": frame_count,
                "simulation_update_hz": (
                    frame_count / elapsed_s if elapsed_s > 0 else None
                ),
                "wall_throughput_fps": (
                    frame_count / wall_duration if wall_duration > 0 else None
                ),
                "runtime_profile": {
                    "dashboard_rendered": cfg.runtime.render_dashboard,
                    "video_recorded": cfg.runtime.record_video,
                    "camera_streams_enabled": True,
                    "camera_stream_count": len(CAMERA_CHANNELS),
                    "detector_camera_channels": list(
                        cfg.perception.detector_camera_channels
                    ),
                    "detector_camera_count": len(
                        cfg.perception.detector_camera_channels
                    ),
                    "detector_precision_requested": (
                        cfg.perception.detector_precision
                    ),
                    "detector_model": cfg.perception.detector_model,
                    "detector_batch_size": cfg.perception.detector_batch_size,
                    "lidar_enabled": True,
                    "radar_streams_enabled": True,
                },
                "distance_m": distance_m,
                "mean_speed_kmh": float(np.mean(speed_values)) if speed_values else 0.0,
                "max_speed_kmh": float(np.max(speed_values)) if speed_values else 0.0,
                "collision_count": collision_count,
                "collision_contact_count": collision_counter.contact_count,
                "max_risk_score": max_risk,
                "min_approx_ttc_s": None if np.isinf(min_ttc) else min_ttc,
                "min_predicted_clearance_m": None if np.isinf(min_clearance) else min_clearance,
                "fusion_updates": fusion_updates,
                "vision_fusion_update_hz_sim": (
                    fusion_updates / elapsed_s if elapsed_s > 0 else None
                ),
                "live_radar_refinement_frames": radar_refinement_frames,
                "live_radar_refinement_hz_sim": (
                    radar_refinement_frames / elapsed_s if elapsed_s > 0 else None
                ),
                "live_radar_accepted_returns": radar_refinement_accepted,
                "tracker_diagnostics": tracker.diagnostics(),
                "sensor_timeout_frames": sensor_timeouts,
                "mean_perception_latency_s": float(np.mean(latency_values)) if latency_values else None,
                "latency_breakdown": latency_summary,
                "oracle_perception": oracle_summary,
                "detector_error": detector_error,
                "perception_startup": (
                    {} if worker is None else dict(worker.startup_diagnostics)
                ),
                "detector_runtime": {
                    "requested": cfg.perception.detector_precision,
                    "active": perception_diagnostics.get(
                        "detector_precision_active"
                    ),
                    "amp_fallback_count": perception_diagnostics.get(
                        "detector_amp_fallback_count", 0
                    ),
                    "amp_fallback_reason": perception_diagnostics.get(
                        "detector_amp_fallback_reason"
                    ),
                },
                "perception_diagnostic_totals": perception_diagnostic_totals,
                "route_destination_carla_xyz": route_destination,
                "control_reason_frames": control_reason_frames,
                "safety_supervisor": (
                    None if supervisor is None else supervisor.summary()
                ),
                "autoware_aeb_shadow": (
                    None if aeb_shadow is None else aeb_shadow.summary()
                ),
                "autoware_aeb_advisory": {
                    "enabled": self.aeb_advisory_throttle_cap is not None,
                    "control_authority": "throttle_cap_only",
                    "throttle_cap": self.aeb_advisory_throttle_cap,
                    "applied_frames": aeb_advisory_frames,
                    "target_track_ids": sorted(aeb_advisory_target_ids),
                    "brake_authority": False,
                    "steering_authority": False,
                },
                "bevfusion_dataset_recording": dataset_summary,
                "deterministic_scenario": (
                    None if self.scenario_controller is None
                    else self.scenario_controller.summary()
                ),
                "scenario_controller": (
                    None if self.scenario_controller is None
                    else self.scenario_controller.summary()
                ),
                "evaluation": evaluation,
                "output_files": {
                    "dashboard_video": str(writer.video_path) if cfg.runtime.record_video else None,
                    "frames_jsonl": str(writer.frames_path),
                    "risk_timeline": str(writer.timeline_path),
                    "events_csv": str(writer.events_path),
                    "supervisor_events_jsonl": (
                        str(writer.supervisor_events_path)
                        if supervisor is not None else None
                    ),
                    "aeb_shadow_events_jsonl": (
                        str(writer.aeb_shadow_events_path)
                        if aeb_shadow is not None else None
                    ),
                    "config_json": str(writer.config_path),
                    "metrics_json": str(writer.metrics_path),
                    "oracle_metrics_json": (
                        str(writer.oracle_metrics_path) if oracle_summary is not None else None
                    ),
                    "latency_metrics_json": str(writer.latency_metrics_path),
                },
            }
            writer.close(summary)
            writer = None
            print(
                f"[完成] sim={elapsed_s:.1f}s | distance={distance_m:.1f}m | "
                f"completion={evaluation['route_completion_pct']} | "
                f"collision={collision_count} | "
                f"route_metric_pass={evaluation['scenario_pass']}"
            )
            return Path(summary["output_files"]["frames_jsonl"]).parent
        finally:
            if writer is not None:
                try:
                    writer.close({"status": "interrupted_before_summary"})
                except Exception:
                    pass
            if worker is not None:
                worker.close()
            if dataset_recorder is not None and not dataset_recorder.finalized:
                try:
                    dataset_recorder.finalize()
                except Exception as exc:
                    print(
                        f"[数据集收口警告] {type(exc).__name__}: {exc}",
                        flush=True,
                    )

            # UE渲染线程原生崩溃后，逐个destroy会让每个actor都等待默认30s并持续刷
            # streaming connection failed。先用1s探活；服务端已消失时，其进程内actor
            # 也已全部消失，跳过无意义RPC清理，只回收本地worker/Pygame资源。
            server_responsive = True
            try:
                client.set_timeout(1.0)
                client.get_world()
                client.set_timeout(cfg.runtime.timeout_s)
            except RuntimeError:
                server_responsive = False
                print(
                    "[CARLA清理] 服务端已不可达，跳过传感器/actor RPC销毁；"
                    "服务端进程内资源已随崩溃释放。",
                    flush=True,
                )

            if server_responsive and sensor_rig is not None:
                sensor_rig.destroy()
            if server_responsive and self.scenario_controller is not None:
                cleanup = getattr(self.scenario_controller, "cleanup", None)
                if cleanup is not None:
                    try:
                        cleanup()
                    except RuntimeError:
                        pass
            if server_responsive:
                for actor in reversed(self.npc_actors):
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass
                for actor in reversed(self.created_actors):
                    try:
                        actor.destroy()
                    except RuntimeError:
                        pass
            if server_responsive and self.scenario_controller is not None:
                after_destroy = getattr(
                    self.scenario_controller, "after_actors_destroyed", None
                )
                if after_destroy is not None:
                    try:
                        after_destroy()
                    except RuntimeError:
                        pass
            if server_responsive:
                try:
                    traffic_manager.set_synchronous_mode(False)
                    world.apply_settings(original_settings)
                    world.set_weather(original_weather)
                except RuntimeError:
                    pass
            pygame.quit()


# 向后兼容 Stage6.1 已公开的类名；新代码优先使用更准确的 CarlaFusionSession。
CarlaManualFusionSession = CarlaFusionSession
