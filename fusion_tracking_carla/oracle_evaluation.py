"""CARLA 真值只读评估器。

该模块只比较跟踪输出与仿真真值，结果不会回流到检测、跟踪、风险或控制。
这样既能量化感知/预测精度，又不会把 CARLA actor 真值变成算法输入。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import EvaluationConfig
from .geometry import carla_xyz_to_internal
from .types import TrackSnapshot


@dataclass(frozen=True)
class OracleObject:
    actor_id: int
    family: str
    position_global: np.ndarray
    velocity_global: np.ndarray
    source_type: str = "dynamic_actor"
    observable: bool = True


@dataclass(frozen=True)
class _StaticTruthGeometry:
    truth: OracleObject
    center_world_carla: np.ndarray
    extent_xyz: np.ndarray
    yaw_world_deg: float


_STATIC_VEHICLE_CACHE: dict[str, tuple[_StaticTruthGeometry, ...]] = {}


def _lidar_observable(
    points_world_carla: np.ndarray | None,
    center_world_carla: np.ndarray,
    extent_xyz: np.ndarray,
    yaw_world_deg: float,
) -> bool:
    """以当前LiDAR是否命中3D框定义可观测性，避免把建筑后目标计入召回。"""

    if points_world_carla is None:
        return True
    points = np.asarray(points_world_carla, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        return False
    delta = points[:, :3] - np.asarray(center_world_carla, dtype=float)
    yaw = np.radians(float(yaw_world_deg))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    extent = np.asarray(extent_xyz, dtype=float) + 0.20
    inside = (
        (np.abs(local_x) <= extent[0])
        & (np.abs(local_y) <= extent[1])
        & (np.abs(delta[:, 2]) <= extent[2])
    )
    return bool(np.any(inside))


def _static_vehicle_truth(world) -> tuple[_StaticTruthGeometry, ...]:
    """读取地图烘焙车辆网格，补齐CARLA actor列表不覆盖的静态真值。"""

    map_name = str(world.get_map().name)
    cached = _STATIC_VEHICLE_CACHE.get(map_name)
    if cached is not None:
        return cached
    import carla

    labels = (
        carla.CityObjectLabel.Car,
        carla.CityObjectLabel.Truck,
        carla.CityObjectLabel.Bus,
        carla.CityObjectLabel.Motorcycle,
        carla.CityObjectLabel.Bicycle,
    )
    objects = []
    seen_ids: set[int] = set()
    for label in labels:
        for item in world.get_environment_objects(label):
            object_id = int(item.id)
            if object_id in seen_ids:
                continue
            seen_ids.add(object_id)
            # EnvironmentObject.bounding_box 已位于世界坐标，不再重复乘transform。
            location = item.bounding_box.location
            position = carla_xyz_to_internal(np.array([
                location.x, location.y, location.z,
            ]))
            objects.append(_StaticTruthGeometry(
                truth=OracleObject(
                    actor_id=object_id,
                    family="vehicle",
                    position_global=np.asarray(position[:2], dtype=float),
                    velocity_global=np.zeros(2, dtype=float),
                    source_type="static_environment_object",
                ),
                center_world_carla=np.array([
                    location.x, location.y, location.z,
                ], dtype=float),
                extent_xyz=np.array([
                    item.bounding_box.extent.x,
                    item.bounding_box.extent.y,
                    item.bounding_box.extent.z,
                ], dtype=float),
                yaw_world_deg=float(item.bounding_box.rotation.yaw),
            ))
    cached = tuple(objects)
    _STATIC_VEHICLE_CACHE[map_name] = cached
    return cached


def collect_vehicle_truth(
    world,
    ego_vehicle,
    lidar_points_world_carla: np.ndarray | None = None,
) -> list[OracleObject]:
    """读取动态actor与地图静态车辆真值；返回值只能交给评估器。"""

    output = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_vehicle.id:
            continue
        # 3D 检测器、训练标注和 TrackSnapshot 表示的是 bounding-box 中心，
        # 不能拿车辆 actor 原点（车型不同，可能位于轴间/底盘处）直接评分。
        # 这里只修正只读 Oracle 的比较基准，不把该中心返回在线算法。
        transform = np.asarray(actor.get_transform().get_matrix(), dtype=float)
        bbox_location = actor.bounding_box.location
        bbox_local = np.array([
            float(bbox_location.x), float(bbox_location.y),
            float(bbox_location.z), 1.0,
        ])
        center_world = transform @ bbox_local
        velocity = actor.get_velocity()
        position = carla_xyz_to_internal(center_world[:3])
        velocity_internal = carla_xyz_to_internal(np.array([
            velocity.x, velocity.y, velocity.z,
        ]))
        output.append(OracleObject(
            actor_id=int(actor.id),
            family="vehicle",
            position_global=np.asarray(position[:2], dtype=float),
            velocity_global=np.asarray(velocity_internal[:2], dtype=float),
            observable=_lidar_observable(
                lidar_points_world_carla,
                center_world[:3],
                np.array([
                    actor.bounding_box.extent.x,
                    actor.bounding_box.extent.y,
                    actor.bounding_box.extent.z,
                ]),
                float(actor.get_transform().rotation.yaw + actor.bounding_box.rotation.yaw),
            ),
        ))
    for item in _static_vehicle_truth(world):
        output.append(OracleObject(
            actor_id=item.truth.actor_id,
            family=item.truth.family,
            position_global=item.truth.position_global,
            velocity_global=item.truth.velocity_global,
            source_type=item.truth.source_type,
            observable=_lidar_observable(
                lidar_points_world_carla,
                item.center_world_carla,
                item.extent_xyz,
                item.yaw_world_deg,
            ),
        ))
    return output


@dataclass(frozen=True)
class _PendingPrediction:
    due_timestamp_s: float
    actor_id: int
    horizon_s: float
    predicted_position_global: np.ndarray


@dataclass
class OracleTrackEvaluator:
    config: EvaluationConfig
    frames: int = 0
    total_truth: int = 0
    total_tracks: int = 0
    total_matches: int = 0
    observable_matches: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    # CARLA的world actor列表不包含地图烘焙的静态车辆网格。场景级precision会把
    # 这些真实可见物体计作FP，因此另报“存在动态actor标注帧”的条件指标；两者都保留。
    actor_labeled_frames: int = 0
    actor_labeled_tracks: int = 0
    actor_labeled_truth: int = 0
    actor_labeled_matches: int = 0
    actor_labeled_false_positives: int = 0
    actor_labeled_false_negatives: int = 0
    id_switches: int = 0
    position_errors_m: list[float] = field(default_factory=list)
    velocity_errors_mps: list[float] = field(default_factory=list)
    prediction_errors_m: dict[float, list[float]] = field(default_factory=dict)
    _last_track_by_actor: dict[int, tuple[int, float]] = field(default_factory=dict)
    _pending: list[_PendingPrediction] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.prediction_errors_m = {
            float(horizon): [] for horizon in self.config.oracle_prediction_horizons_s
        }

    def _eligible_tracks(self, tracks: list[TrackSnapshot]) -> list[TrackSnapshot]:
        if not self.config.oracle_exclude_tentative_tracks:
            return list(tracks)
        return [track for track in tracks if track.lifecycle != "tentative"]

    def _resolve_predictions(
        self,
        timestamp_s: float,
        truth_by_id: dict[int, OracleObject],
    ) -> None:
        remaining = []
        tolerance = 0.5 * self.config.oracle_frame_tolerance_s
        for item in self._pending:
            if item.due_timestamp_s > timestamp_s + tolerance:
                remaining.append(item)
                continue
            # 若传感器超时导致目标时刻已经错过，不拿更晚真值冒充目标时刻真值。
            if item.due_timestamp_s < timestamp_s - tolerance:
                continue
            truth = truth_by_id.get(item.actor_id)
            if truth is not None:
                error = float(np.linalg.norm(
                    item.predicted_position_global - truth.position_global
                ))
                self.prediction_errors_m[item.horizon_s].append(error)
        self._pending = remaining

    def update(
        self,
        *,
        timestamp_s: float,
        tracks: list[TrackSnapshot],
        truth_objects: list[OracleObject],
        ego_position_global: np.ndarray,
    ) -> dict:
        """完成一帧一对一匹配，并登记未来位置预测供后续真值兑现。"""

        truth_by_id = {item.actor_id: item for item in truth_objects}
        self._resolve_predictions(timestamp_s, truth_by_id)
        ego_position = np.asarray(ego_position_global, dtype=float)[:2]
        truths = [
            item for item in truth_objects
            if np.linalg.norm(item.position_global - ego_position)
            <= self.config.oracle_max_range_m
        ]
        # 轨迹和真值必须使用同一个距离 ROI；否则 ROI 外轨迹会被误记成假阳性。
        # 这是“距离条件下的场景指标”，不声称等价于带遮挡标注的检测 benchmark。
        eligible_tracks = [
            track for track in self._eligible_tracks(tracks)
            if np.linalg.norm(track.state_global[:2] - ego_position)
            <= self.config.oracle_max_range_m
        ]
        self.frames += 1
        observable_truth_indices = {
            index for index, item in enumerate(truths) if item.observable
        }
        self.total_truth += len(observable_truth_indices)
        self.total_tracks += len(eligible_tracks)

        matches: list[tuple[int, int]] = []
        if truths and eligible_tracks:
            cost = np.full((len(eligible_tracks), len(truths)), 1e6, dtype=float)
            for track_index, track in enumerate(eligible_tracks):
                for truth_index, truth in enumerate(truths):
                    if track.family != truth.family:
                        continue
                    distance = float(np.linalg.norm(
                        track.state_global[:2] - truth.position_global
                    ))
                    if distance <= self.config.oracle_match_gate_m:
                        cost[track_index, truth_index] = distance
            rows, cols = linear_sum_assignment(cost)
            matches = [
                (int(row), int(col)) for row, col in zip(rows, cols)
                if cost[row, col] < 1e5
            ]

        self.total_matches += len(matches)
        frame_observable_matches = sum(
            truth_index in observable_truth_indices for _, truth_index in matches
        )
        self.observable_matches += frame_observable_matches
        frame_false_positive = len(eligible_tracks) - len(matches)
        frame_false_negative = len(observable_truth_indices) - frame_observable_matches
        self.false_positives += frame_false_positive
        self.false_negatives += frame_false_negative
        if observable_truth_indices:
            self.actor_labeled_frames += 1
            self.actor_labeled_tracks += len(eligible_tracks)
            self.actor_labeled_truth += len(observable_truth_indices)
            self.actor_labeled_matches += frame_observable_matches
            self.actor_labeled_false_positives += frame_false_positive
            self.actor_labeled_false_negatives += frame_false_negative
        frame_matches = []
        for track_index, truth_index in matches:
            track = eligible_tracks[track_index]
            truth = truths[truth_index]
            position_error = float(np.linalg.norm(
                track.state_global[:2] - truth.position_global
            ))
            velocity_error = float(np.linalg.norm(
                track.state_global[2:] - truth.velocity_global
            ))
            self.position_errors_m.append(position_error)
            self.velocity_errors_mps.append(velocity_error)
            previous = self._last_track_by_actor.get(truth.actor_id)
            if (
                previous is not None
                and previous[0] != track.track_id
                and timestamp_s - previous[1] <= self.config.oracle_id_switch_gap_s
            ):
                self.id_switches += 1
            self._last_track_by_actor[truth.actor_id] = (track.track_id, timestamp_s)
            for horizon in self.config.oracle_prediction_horizons_s:
                horizon = float(horizon)
                self._pending.append(_PendingPrediction(
                    due_timestamp_s=timestamp_s + horizon,
                    actor_id=truth.actor_id,
                    horizon_s=horizon,
                    predicted_position_global=(
                        track.state_global[:2] + horizon * track.state_global[2:]
                    ).copy(),
                ))
            frame_matches.append({
                "track_id": int(track.track_id),
                "actor_id": int(truth.actor_id),
                "position_error_m": position_error,
                "velocity_error_mps": velocity_error,
                # 仅写入评测产物，便于定位坐标/标定偏差；不参与后续算法。
                "position_error_xy_m": (
                    track.state_global[:2] - truth.position_global
                ).astype(float).tolist(),
                "velocity_error_xy_mps": (
                    track.state_global[2:] - truth.velocity_global
                ).astype(float).tolist(),
            })
        return {
            "truth_count": len(observable_truth_indices),
            "physical_truth_count": len(truths),
            "evaluated_track_count": len(eligible_tracks),
            "match_count": len(matches),
            "observable_match_count": frame_observable_matches,
            "false_positive_count": frame_false_positive,
            "false_negative_count": frame_false_negative,
            "matches": frame_matches,
        }

    @staticmethod
    def _rmse(values: list[float]) -> float | None:
        return None if not values else float(np.sqrt(np.mean(np.square(values))))

    @staticmethod
    def _percentile(values: list[float], q: float) -> float | None:
        return None if not values else float(np.percentile(values, q))

    def finalize(self) -> dict:
        precision_denominator = self.total_matches + self.false_positives
        recall_denominator = self.observable_matches + self.false_negatives
        actor_precision_denominator = (
            self.actor_labeled_matches + self.actor_labeled_false_positives
        )
        actor_recall_denominator = (
            self.actor_labeled_matches + self.actor_labeled_false_negatives
        )
        prediction = {}
        for horizon, values in sorted(self.prediction_errors_m.items()):
            prediction[f"{horizon:g}s"] = {
                "samples": len(values),
                "rmse_m": self._rmse(values),
                "median_m": self._percentile(values, 50),
                "p95_m": self._percentile(values, 95),
            }
        return {
            "truth_source": (
                "CARLA dynamic actor state + static environment vehicle objects; "
                "evaluation-only, never fed to algorithm"
            ),
            "metric_scope": (
                "range-conditioned vehicle bbox centers; precision matches all physical "
                "dynamic/static vehicles, recall denominator requires current LiDAR support"
            ),
            "max_range_m": float(self.config.oracle_max_range_m),
            "match_gate_m": float(self.config.oracle_match_gate_m),
            "frames": self.frames,
            "total_truth_instances": self.total_truth,
            "total_evaluated_tracks": self.total_tracks,
            "matches": self.total_matches,
            "observable_matches": self.observable_matches,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": (
                None if precision_denominator == 0
                else float(self.total_matches / precision_denominator)
            ),
            "recall": (
                None if recall_denominator == 0
                else float(self.observable_matches / recall_denominator)
            ),
            "actor_labeled_frames": self.actor_labeled_frames,
            "actor_conditioned_tracks": self.actor_labeled_tracks,
            "actor_conditioned_truth": self.actor_labeled_truth,
            "actor_conditioned_matches": self.actor_labeled_matches,
            "actor_conditioned_false_positives": self.actor_labeled_false_positives,
            "actor_conditioned_false_negatives": self.actor_labeled_false_negatives,
            "actor_conditioned_precision": (
                None if actor_precision_denominator == 0
                else float(self.actor_labeled_matches / actor_precision_denominator)
            ),
            "actor_conditioned_recall": (
                None if actor_recall_denominator == 0
                else float(self.actor_labeled_matches / actor_recall_denominator)
            ),
            "position_rmse_m": self._rmse(self.position_errors_m),
            "position_median_m": self._percentile(self.position_errors_m, 50),
            "position_p95_m": self._percentile(self.position_errors_m, 95),
            "velocity_rmse_mps": self._rmse(self.velocity_errors_mps),
            "velocity_median_mps": self._percentile(self.velocity_errors_mps, 50),
            "velocity_p95_mps": self._percentile(self.velocity_errors_mps, 95),
            "id_switches": self.id_switches,
            "future_prediction": prediction,
        }
