"""在线多目标 CV-KF，支持位置观测与 CARLA 径向 Radar 观测。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from fusion_tracking_core.association import associate_tracks

from .config import CAMERA_CHANNELS, TrackerConfig
from .types import FusionDetection, RadarReturn, TrackSnapshot


def cv_transition(dt_s: float) -> np.ndarray:
    return np.array([
        [1.0, 0.0, dt_s, 0.0],
        [0.0, 1.0, 0.0, dt_s],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def cv_process_noise(dt_s: float, accel_std_mps2: float) -> np.ndarray:
    """白噪声加速度模型 Q，状态顺序 [px,py,vx,vy]。"""

    q = accel_std_mps2**2
    t2, t3, t4 = dt_s**2, dt_s**3, dt_s**4
    return q * np.array([
        [t4 / 4, 0.0, t3 / 2, 0.0],
        [0.0, t4 / 4, 0.0, t3 / 2],
        [t3 / 2, 0.0, t2, 0.0],
        [0.0, t3 / 2, 0.0, t2],
    ])


def radial_velocity_measurement(
    radar: RadarReturn, ego_velocity_global: np.ndarray
) -> tuple[np.ndarray, float]:
    """构造 ``H @ target_state = z`` 的一维绝对速度观测。

    CARLA 文档把字段描述为 ``velocity towards the sensor``，但其官方
    ``manual_control.py`` 可视化与本机静止前车实测均表明数值符号按距离变化率使用：
    靠近传感器为负、远离为正。因此

    ``range_rate = u · (v_target - v_ego)``，
    ``u · v_target = range_rate + u · v_ego``。

    该公式避免把一个径向标量错误复制成 vx/vy，也避免把静止前车错误补偿成
    两倍 ego 速度。
    """

    unit = np.asarray(radar.radial_unit_global, dtype=float)[:2]
    norm = float(np.linalg.norm(unit))
    if norm < 1e-9:
        raise ValueError("radial_unit_global 不能为零向量")
    unit = unit / norm
    h = np.array([0.0, 0.0, unit[0], unit[1]])
    z = float(radar.velocity_towards_mps) + float(unit @ ego_velocity_global[:2])
    return h, z


def _stable_update(state, covariance, h, z, variance):
    h = np.atleast_2d(np.asarray(h, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    r = np.eye(len(z)) * float(variance) if np.isscalar(variance) else np.asarray(variance)
    innovation = z - h @ state
    innovation_cov = h @ covariance @ h.T + r
    gain = covariance @ h.T @ np.linalg.pinv(innovation_cov)
    updated_state = state + gain @ innovation
    identity = np.eye(len(state))
    # Joseph form keeps P symmetric positive semi-definite under finite precision.
    updated_covariance = (
        (identity - gain @ h) @ covariance @ (identity - gain @ h).T
        + gain @ r @ gain.T
    )
    return updated_state, 0.5 * (updated_covariance + updated_covariance.T), innovation, innovation_cov


@dataclass
class _Track:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    timestamp_s: float
    measurement_timestamp_s: float
    class_name: str
    family: str
    extent_xy: np.ndarray
    score: float
    hits: int = 1
    missed: int = 0
    lidar_support: int = 0
    radar_support: int = 0
    camera_channels: tuple[str, ...] = ()
    radial_velocity_std_mps: float | None = None


@dataclass
class _LiveRadarMemory:
    """与异步视觉KF隔离的短时20Hz Radar状态。"""

    state: np.ndarray
    covariance: np.ndarray
    timestamp_s: float
    measurement_timestamp_s: float
    age_updates: int


class StreamingFusionTracker:
    """有状态在线跟踪器；每次检测批次调用一次 ``update``。"""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._live_radar_memory: dict[int, _LiveRadarMemory] = {}
        self._next_track_id = 1
        self._diagnostics = {
            "updates": 0,
            "primary_matches": 0,
            "recovery_matches": 0,
            "radar_only_continuity_matches": 0,
            "new_tracks": 0,
            "suppressed_low_score_births": 0,
            "model_velocity_updates": 0,
            "stale_measurement_prunes": 0,
        }

    def _is_radar_only_track(self, track: _Track) -> bool:
        """True only for tracks whose latest spatial evidence came from Radar."""

        return bool(
            self.config.enable_radar_only_track_continuity
            and track.lidar_support == 0
            and track.radar_support > 0
        )

    @staticmethod
    def _camera_evidence_compatible(
        track_channels: tuple[str, ...], detection_channels: tuple[str, ...]
    ) -> bool:
        """相同或环形相邻相机才允许恢复匹配，防止跨车身误接。"""

        if not track_channels or not detection_channels:
            return False
        for first in track_channels:
            for second in detection_channels:
                if first == second:
                    return True
                if first not in CAMERA_CHANNELS or second not in CAMERA_CHANNELS:
                    continue
                distance = abs(CAMERA_CHANNELS.index(first) - CAMERA_CHANNELS.index(second))
                if distance == 1 or distance == len(CAMERA_CHANNELS) - 1:
                    return True
        return False

    def _recovery_matches(
        self,
        unmatched_tracks: list[int],
        unmatched_detections: list[int],
        detections: list[FusionDetection],
        ego_velocity_global: np.ndarray,
    ) -> list[tuple[int, int]]:
        """为第一阶段拒绝的稳定轨迹做保守恢复，不放宽主关联器。"""

        if (
            not self.config.enable_recovery_association
            or not unmatched_tracks
            or not unmatched_detections
        ):
            return []
        cost = np.full(
            (len(unmatched_tracks), len(unmatched_detections)), 1e6, dtype=float
        )
        for row, track_index in enumerate(unmatched_tracks):
            track = self._tracks[track_index]
            radar_only_track = self._is_radar_only_track(track)
            if track.hits < self.config.min_confirmed_hits and not radar_only_track:
                continue
            for column, detection_index in enumerate(unmatched_detections):
                detection = detections[detection_index]
                if track.family != detection.family:
                    continue
                if not self._camera_evidence_compatible(
                    track.camera_channels, detection.camera_channels
                ):
                    continue
                distance = float(np.linalg.norm(
                    track.state[:2] - detection.position_global[:2]
                ))
                gate = (
                    self.config.radar_only_recovery_vehicle_gate_m
                    if radar_only_track and detection.family == "vehicle"
                    else self.config.recovery_vehicle_gate_m
                    if detection.family == "vehicle"
                    else self.config.recovery_vru_gate_m
                )
                if distance > gate:
                    continue
                track_area = max(float(np.prod(track.extent_xy)), 1e-3)
                detection_area = max(float(np.prod(detection.extent_xy)), 1e-3)
                size_log_ratio = abs(float(np.log(detection_area / track_area)))
                if size_log_ratio > self.config.recovery_max_size_log_ratio:
                    continue
                if detection.radar_returns:
                    radial_innovations = []
                    for radar in detection.radar_returns:
                        h, z = radial_velocity_measurement(radar, ego_velocity_global)
                        radial_innovations.append(abs(float(z - h @ track.state)))
                    if (
                        float(np.median(radial_innovations))
                        > (
                            self.config.radar_only_recovery_radial_innovation_gate_mps
                            if radar_only_track and detection.family == "vehicle"
                            else self.config.recovery_radial_innovation_gate_mps
                        )
                    ):
                        continue
                # 距离主导，尺寸只作为弱排序项；仍由匈牙利完成一对一分配。
                cost[row, column] = distance + 0.5 * size_log_ratio
        rows, columns = linear_sum_assignment(cost)
        return [
            (unmatched_tracks[int(row)], unmatched_detections[int(column)])
            for row, column in zip(rows, columns)
            if cost[row, column] < 1e5
        ]

    def diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def _predict(self, track: _Track, timestamp_s: float) -> None:
        dt_s = max(float(timestamp_s - track.timestamp_s), 0.0)
        if dt_s <= 0.0:
            return
        transition = cv_transition(dt_s)
        track.state = transition @ track.state
        track.covariance = (
            transition @ track.covariance @ transition.T
            + cv_process_noise(dt_s, self.config.process_accel_std_mps2)
        )
        track.timestamp_s = float(timestamp_s)

    def _association_views(self, detections):
        tracks = [{
            "family": t.family,
            "position": t.state[:2],
            "velocity": t.state[2:],
            "position_covariance": t.covariance[:2, :2],
            "velocity_covariance": t.covariance[2:, 2:],
            "extent_xy": t.extent_xy,
        } for t in self._tracks]
        observations = []
        for detection in detections:
            velocity = self._cartesian_velocity(detection)
            observations.append({
                "family": detection.family,
                "position": detection.position_global[:2],
                "velocity": np.zeros(2) if velocity is None else velocity,
                "has_velocity": velocity is not None,
                "position_std_m": detection.position_std_m,
                "velocity_std_mps": (
                    self.config.initial_velocity_std_mps
                    if detection.velocity_std_mps is None
                    else max(float(detection.velocity_std_mps), 0.1)
                ),
                "extent_xy": detection.extent_xy,
            })
        return tracks, observations

    @staticmethod
    def _cartesian_velocity(detection: FusionDetection) -> np.ndarray | None:
        """校验可选二维速度先验；无效值按“无速度观测”处理。"""

        if detection.velocity_global is None:
            return None
        velocity = np.asarray(detection.velocity_global, dtype=float).reshape(-1)
        if velocity.size < 2 or not np.all(np.isfinite(velocity[:2])):
            return None
        return velocity[:2]

    def _update_position(self, track: _Track, detection: FusionDetection) -> None:
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        observation_std = max(float(detection.position_std_m), 0.1)
        track.state, track.covariance, _, _ = _stable_update(
            track.state, track.covariance, h, detection.position_global[:2],
            np.eye(2) * observation_std**2,
        )

    def _update_cartesian_velocity(
        self, track: _Track, detection: FusionDetection
    ) -> bool:
        """以弱二维先验更新速度；随后仍由更强的Radar径向证据修正。"""

        velocity = self._cartesian_velocity(detection)
        if velocity is None:
            return False
        observation_std = max(
            float(
                self.config.initial_velocity_std_mps
                if detection.velocity_std_mps is None
                else detection.velocity_std_mps
            ),
            0.1,
        )
        h = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        track.state, track.covariance, _, _ = _stable_update(
            track.state,
            track.covariance,
            h,
            velocity,
            np.eye(2) * observation_std**2,
        )
        return True

    def _update_radial(
        self, track: _Track, returns: list[RadarReturn], ego_velocity_global: np.ndarray
    ) -> int:
        accepted = 0
        for radar in returns:
            h, z = radial_velocity_measurement(radar, ego_velocity_global)
            variance = self.config.radial_velocity_std_mps**2
            innovation = float(z - h @ track.state)
            innovation_variance = float(h @ track.covariance @ h + variance)
            gate = self.config.radial_outlier_gate_sigma * np.sqrt(max(innovation_variance, 1e-9))
            if abs(innovation) > gate:
                continue
            track.state, track.covariance, _, _ = _stable_update(
                track.state, track.covariance, h, z, variance
            )
            accepted += 1
        return accepted

    def refine_snapshots_with_radar(
        self,
        snapshots: list[TrackSnapshot],
        returns: list[RadarReturn],
        ego_velocity_global: np.ndarray,
    ) -> tuple[list[TrackSnapshot], dict]:
        """用当前帧 Radar 修正只读预测快照，不改写异步视觉轨迹时间线。

        每个回波只分配给最近且通过空间门限的轨迹，再用径向创新门限剔除
        护栏/邻车异常值。下一视觉批次到达时仍按其原始时间戳更新基础轨迹，
        因而不会产生 out-of-sequence measurement 问题。
        """

        stats = {
            "input_returns": len(returns),
            "spatially_assigned_returns": 0,
            "accepted_returns": 0,
            "refined_tracks": 0,
            "memory_seeded_tracks": 0,
            "memory_carried_tracks": 0,
            "memory_resets": 0,
        }
        if not self.config.enable_live_radar_refinement or not snapshots:
            if not snapshots:
                self._live_radar_memory.clear()
            return list(snapshots), stats

        active_ids = {item.track_id for item in snapshots}
        for track_id in tuple(self._live_radar_memory):
            if track_id not in active_ids:
                self._live_radar_memory.pop(track_id, None)

        # 先用上一个Radar快状态预测到当前20Hz时刻；仅当视觉命中数没有变化且
        # 最近Radar测量未过期时使用。基础视觉KF完全不被改写。
        seeded: list[tuple[np.ndarray, np.ndarray, bool, float | None]] = []
        for snapshot in snapshots:
            memory = self._live_radar_memory.get(snapshot.track_id)
            memory_valid = bool(
                memory is not None
                and memory.age_updates == snapshot.age_updates
                and snapshot.timestamp_s + 1e-9 >= memory.timestamp_s
                and snapshot.timestamp_s - memory.measurement_timestamp_s
                <= self.config.live_radar_memory_max_age_s
            )
            if memory_valid and memory is not None:
                dt_s = max(snapshot.timestamp_s - memory.timestamp_s, 0.0)
                transition = cv_transition(dt_s)
                state = transition @ memory.state
                covariance = (
                    transition @ memory.covariance @ transition.T
                    + cv_process_noise(dt_s, self.config.process_accel_std_mps2)
                )
                stats["memory_seeded_tracks"] += 1
                seeded.append((
                    state,
                    covariance,
                    True,
                    memory.measurement_timestamp_s,
                ))
            else:
                if memory is not None:
                    self._live_radar_memory.pop(snapshot.track_id, None)
                    stats["memory_resets"] += 1
                seeded.append((
                    snapshot.state_global.copy(),
                    snapshot.covariance_global.copy(),
                    False,
                    None,
                ))

        assigned: list[list[RadarReturn]] = [[] for _ in snapshots]
        positions = np.asarray([item[0][:2] for item in seeded], dtype=float)
        gate = float(self.config.live_radar_association_gate_m)
        for radar in returns:
            distances = np.linalg.norm(positions - radar.position_global[:2], axis=1)
            track_index = int(np.argmin(distances))
            if float(distances[track_index]) <= gate:
                assigned[track_index].append(radar)
                stats["spatially_assigned_returns"] += 1

        refined = []
        for snapshot, candidates, seed in zip(snapshots, assigned, seeded):
            state, covariance, memory_used, measurement_timestamp_s = seed
            usable = []
            for radar in candidates:
                h, z = radial_velocity_measurement(radar, ego_velocity_global)
                variance = self.config.radial_velocity_std_mps**2
                innovation = float(z - h @ state)
                innovation_variance = float(h @ covariance @ h + variance)
                innovation_gate = (
                    self.config.radial_outlier_gate_sigma
                    * np.sqrt(max(innovation_variance, 1e-9))
                )
                if abs(innovation) <= innovation_gate:
                    usable.append((h, z, variance))
            if len(usable) < self.config.radar_min_support:
                if memory_used:
                    # 没有新测量时也保持上一次Radar快状态的20Hz预测，避免速度弹回
                    # 旧视觉值；测量年龄超过上限后会自动回退基础视觉KF。
                    stats["memory_carried_tracks"] += 1
                    self._live_radar_memory[snapshot.track_id] = _LiveRadarMemory(
                        state=state.copy(),
                        covariance=covariance.copy(),
                        timestamp_s=float(snapshot.timestamp_s),
                        measurement_timestamp_s=float(measurement_timestamp_s),
                        age_updates=snapshot.age_updates,
                    )
                    refined.append(replace(
                        snapshot,
                        state_global=state,
                        covariance_global=covariance,
                    ))
                else:
                    refined.append(snapshot)
                continue
            for h, z, variance in usable:
                state, covariance, _, _ = _stable_update(
                    state, covariance, h, z, variance
                )
            accepted = len(usable)
            stats["accepted_returns"] += accepted
            stats["refined_tracks"] += 1
            self._live_radar_memory[snapshot.track_id] = _LiveRadarMemory(
                state=state.copy(),
                covariance=covariance.copy(),
                timestamp_s=float(snapshot.timestamp_s),
                measurement_timestamp_s=float(snapshot.timestamp_s),
                age_updates=snapshot.age_updates,
            )
            refined.append(replace(
                snapshot,
                state_global=state,
                covariance_global=covariance,
                radar_support=accepted,
                radial_velocity_std_mps=(
                    self.config.radial_velocity_std_mps / np.sqrt(accepted)
                ),
            ))
        return refined, stats

    def _new_track(self, detection: FusionDetection) -> _Track:
        cfg = self.config
        model_velocity = self._cartesian_velocity(detection)
        initial_velocity = (
            np.zeros(2, dtype=float)
            if model_velocity is None
            else model_velocity
        )
        velocity_std = (
            cfg.initial_velocity_std_mps
            if model_velocity is None or detection.velocity_std_mps is None
            else max(float(detection.velocity_std_mps), 0.1)
        )
        covariance = np.diag([
            cfg.initial_position_std_m**2, cfg.initial_position_std_m**2,
            velocity_std**2, velocity_std**2,
        ])
        track = _Track(
            track_id=self._next_track_id,
            state=np.r_[detection.position_global[:2], initial_velocity],
            covariance=covariance,
            timestamp_s=float(detection.timestamp_s),
            measurement_timestamp_s=float(detection.timestamp_s),
            class_name=detection.class_name,
            family=detection.family,
            extent_xy=np.asarray(detection.extent_xy, dtype=float),
            score=float(detection.score),
            lidar_support=detection.lidar_support,
            camera_channels=detection.camera_channels,
        )
        self._next_track_id += 1
        return track

    def update(
        self,
        detections: list[FusionDetection],
        timestamp_s: float,
        ego_velocity_global: np.ndarray,
    ) -> list[TrackSnapshot]:
        """推进到时间戳、关联检测、依次更新位置和径向速度。"""

        for track in self._tracks:
            self._predict(track, timestamp_s)
        track_views, detection_views = self._association_views(detections)
        assignment = associate_tracks(track_views, detection_views, self.config.association)
        recovery_matches = self._recovery_matches(
            assignment.unmatched_tracks,
            assignment.unmatched_detections,
            detections,
            ego_velocity_global,
        )
        matches = list(assignment.matches) + recovery_matches
        radar_only_continuity_matches = sum(
            self._is_radar_only_track(self._tracks[track_index])
            for track_index, _ in recovery_matches
        )
        recovered_tracks = {item[0] for item in recovery_matches}
        recovered_detections = {item[1] for item in recovery_matches}
        unmatched_tracks = [
            index for index in assignment.unmatched_tracks
            if index not in recovered_tracks
        ]
        unmatched_detections = [
            index for index in assignment.unmatched_detections
            if index not in recovered_detections
        ]
        self._diagnostics["updates"] += 1
        self._diagnostics["primary_matches"] += len(assignment.matches)
        self._diagnostics["recovery_matches"] += len(recovery_matches)
        self._diagnostics["radar_only_continuity_matches"] += (
            radar_only_continuity_matches
        )
        birth_detections = [
            index
            for index in unmatched_detections
            if detections[index].score + 1e-9
            >= self.config.new_track_score_threshold
        ]
        self._diagnostics["new_tracks"] += len(birth_detections)
        self._diagnostics["suppressed_low_score_births"] += (
            len(unmatched_detections) - len(birth_detections)
        )

        for track_index, detection_index in matches:
            track, detection = self._tracks[track_index], detections[detection_index]
            self._update_position(track, detection)
            if self._update_cartesian_velocity(track, detection):
                self._diagnostics["model_velocity_updates"] += 1
            radar_count = self._update_radial(track, detection.radar_returns, ego_velocity_global)
            track.class_name = detection.class_name
            track.family = detection.family
            track.extent_xy = 0.75 * track.extent_xy + 0.25 * detection.extent_xy
            track.score = 0.7 * track.score + 0.3 * detection.score
            track.hits += 1
            track.missed = 0
            track.measurement_timestamp_s = float(detection.timestamp_s)
            track.lidar_support = detection.lidar_support
            track.radar_support = radar_count
            track.camera_channels = detection.camera_channels
            track.radial_velocity_std_mps = (
                self.config.radial_velocity_std_mps / np.sqrt(radar_count)
                if radar_count else None
            )

        for track_index in unmatched_tracks:
            self._tracks[track_index].missed += 1

        for detection_index in birth_detections:
            detection = detections[detection_index]
            track = self._new_track(detection)
            if self._cartesian_velocity(detection) is not None:
                self._diagnostics["model_velocity_updates"] += 1
            radar_count = self._update_radial(
                track, detection.radar_returns, ego_velocity_global
            )
            track.radar_support = radar_count
            track.radial_velocity_std_mps = (
                self.config.radial_velocity_std_mps / np.sqrt(radar_count)
                if radar_count else None
            )
            self._tracks.append(track)

        retained = []
        for track in self._tracks:
            missed_ok = track.missed <= (
                self.config.radar_only_max_missed_updates
                if self._is_radar_only_track(track)
                else self.config.max_missed_updates
            )
            measurement_age_s = max(
                float(timestamp_s - track.measurement_timestamp_s), 0.0
            )
            age_ok = measurement_age_s <= self.config.max_prediction_age_s
            if missed_ok and age_ok:
                retained.append(track)
            elif not age_ok:
                self._diagnostics["stale_measurement_prunes"] += 1
        self._tracks = retained
        return self.snapshots(timestamp_s)

    def snapshots(self, timestamp_s: float) -> list[TrackSnapshot]:
        """返回指定时刻的只读预测，不改变内部 KF。"""

        snapshots = []
        for track in self._tracks:
            # Radar-only轨迹分离“内部ID连续性”和“外部可发布性”：长寿命只用于
            # 后续恢复同一ID，超过发布窗口的CV外推不进入预测、风险与评分链。
            if (
                self._is_radar_only_track(track)
                and track.missed > self.config.radar_only_publish_max_missed_updates
            ):
                continue
            dt_s = max(float(timestamp_s - track.timestamp_s), 0.0)
            # 感知线程若异常停止，不让旧轨迹无限外推并制造虚假高风险。
            if dt_s > self.config.max_prediction_age_s:
                continue
            measurement_age_s = max(
                float(timestamp_s - track.measurement_timestamp_s), 0.0
            )
            if measurement_age_s > self.config.max_prediction_age_s:
                continue
            transition = cv_transition(dt_s)
            state = transition @ track.state
            covariance = transition @ track.covariance @ transition.T + cv_process_noise(
                dt_s, self.config.process_accel_std_mps2
            )
            # 未达到确认命中数的轨迹，即使已经失配也仍是tentative；旧顺序会把
            # 单次假检在首个miss后错误升级为lost，使评分器和安全链误认为正式轨迹。
            lifecycle = (
                "tentative" if track.hits < self.config.min_confirmed_hits else
                "lost" if track.missed > 0 else
                "confirmed"
            )
            snapshots.append(TrackSnapshot(
                track_id=track.track_id,
                class_name=track.class_name,
                family=track.family,
                lifecycle=lifecycle,
                timestamp_s=float(timestamp_s),
                state_global=state.copy(),
                covariance_global=covariance.copy(),
                extent_xy=track.extent_xy.copy(),
                score=track.score,
                age_updates=track.hits,
                missed_updates=track.missed,
                lidar_support=track.lidar_support,
                radar_support=track.radar_support,
                camera_channels=track.camera_channels,
                radial_velocity_std_mps=track.radial_velocity_std_mps,
                measurement_timestamp_s=float(track.measurement_timestamp_s),
            ))
        return snapshots
