"""BEVFusion 独立进程的稳定 JSON 输入/输出契约。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "fusion_tracking.bevfusion.v1"
SEQUENCE_SCHEMA_VERSION = "fusion_tracking.bevfusion.sequence.v1"
NUSCENES_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def _float_tuple(value: Any, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} 必须是长度为 {length} 的数组")
    result = tuple(float(item) for item in value)
    if not all(item == item and abs(item) != float("inf") for item in result):
        raise ValueError(f"{field_name} 含 NaN/Inf")
    return result


@dataclass(frozen=True)
class BEVFusionBox:
    """一个位于 LIDAR_TOP 坐标系的神经网络 3D 检测框。

    ``size_lwh_m`` 显式规定为 length/width/height，避免不同仓库的
    wlh/dx-dy-dz 命名混淆。``velocity_lidar_mps`` 是可选模型输出，首版
    跟踪仍以 Radar 更新为准，不把它无条件当作真值。
    """

    center_lidar_m: tuple[float, float, float]
    size_lwh_m: tuple[float, float, float]
    yaw_lidar_rad: float
    class_name: str
    score: float
    velocity_lidar_mps: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BEVFusionBox":
        class_name = str(payload["class_name"])
        if class_name not in NUSCENES_CLASSES:
            raise ValueError(f"未知 nuScenes 类别: {class_name}")
        score = float(payload["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("score 必须位于 [0, 1]")
        velocity = payload.get("velocity_lidar_mps")
        return cls(
            center_lidar_m=_float_tuple(payload["center_lidar_m"], 3, "center_lidar_m"),
            size_lwh_m=_float_tuple(payload["size_lwh_m"], 3, "size_lwh_m"),
            yaw_lidar_rad=float(payload["yaw_lidar_rad"]),
            class_name=class_name,
            score=score,
            velocity_lidar_mps=(
                None
                if velocity is None
                else _float_tuple(velocity, 2, "velocity_lidar_mps")
            ),
        )


@dataclass(frozen=True)
class BEVFusionFrameResult:
    """BEVFusion 后端对一个同步传感器帧的完整输出。"""

    sample_token: str
    timestamp_s: float
    detections: tuple[BEVFusionBox, ...]
    backend: str = "mmdetection3d.projects.BEVFusion"
    checkpoint: str = ""
    inference_latency_s: float | None = None
    backend_preprocess_latency_s: float | None = None
    inference_precision: str | None = None
    amp_fallback_error: str | None = None
    backend_stage_latencies_s: dict[str, float] | None = None
    schema_version: str = SCHEMA_VERSION
    coordinate_frame: str = "LIDAR_TOP"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BEVFusionFrameResult":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"不兼容的 schema_version={payload.get('schema_version')!r}；"
                f"期望 {SCHEMA_VERSION!r}"
            )
        if payload.get("coordinate_frame") != "LIDAR_TOP":
            raise ValueError("首版桥接只接受 LIDAR_TOP 坐标输出")
        latency = payload.get("inference_latency_s")
        preprocess_latency = payload.get("backend_preprocess_latency_s")
        return cls(
            sample_token=str(payload["sample_token"]),
            timestamp_s=float(payload["timestamp_s"]),
            detections=tuple(
                BEVFusionBox.from_dict(item) for item in payload.get("detections", [])
            ),
            backend=str(payload.get("backend", "mmdetection3d.projects.BEVFusion")),
            checkpoint=str(payload.get("checkpoint", "")),
            inference_latency_s=None if latency is None else float(latency),
            backend_preprocess_latency_s=(
                None if preprocess_latency is None else float(preprocess_latency)
            ),
            inference_precision=(
                None
                if payload.get("inference_precision") is None
                else str(payload.get("inference_precision"))
            ),
            amp_fallback_error=(
                None
                if payload.get("amp_fallback_error") is None
                else str(payload.get("amp_fallback_error"))
            ),
            backend_stage_latencies_s=(
                None
                if payload.get("backend_stage_latencies_s") is None
                else {
                    str(name): float(value)
                    for name, value in dict(
                        payload.get("backend_stage_latencies_s")
                    ).items()
                }
            ),
            schema_version=str(payload["schema_version"]),
            coordinate_frame=str(payload["coordinate_frame"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "BEVFusionFrameResult":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def dump(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output


@dataclass(frozen=True)
class BEVFusionSequenceResult:
    """One model load followed by an ordered sequence of frame results."""

    scene_name: str
    frames: tuple[BEVFusionFrameResult, ...]
    checkpoint: str = ""
    model_load_latency_s: float | None = None
    sparseconv_backend: str = ""
    sparseconv_layout_converted_count: int = 0
    checkpoint_strict_load: bool = False
    repo_commit: str = ""
    # 后端写入JSON前使用的最低分数。None表示旧结果未记录该元数据。
    backend_score_threshold: float | None = None
    schema_version: str = SEQUENCE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BEVFusionSequenceResult":
        if payload.get("schema_version") != SEQUENCE_SCHEMA_VERSION:
            raise ValueError(
                f"不兼容的 sequence schema_version={payload.get('schema_version')!r}；"
                f"期望 {SEQUENCE_SCHEMA_VERSION!r}"
            )
        frames = tuple(
            BEVFusionFrameResult.from_dict(item) for item in payload.get("frames", [])
        )
        if not frames:
            raise ValueError("BEVFusion序列结果不能为空")
        tokens = [item.sample_token for item in frames]
        if len(tokens) != len(set(tokens)):
            raise ValueError("BEVFusion序列存在重复sample_token")
        timestamps = [item.timestamp_s for item in frames]
        if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
            raise ValueError("BEVFusion序列时间戳必须严格递增")
        backend_threshold = payload.get("backend_score_threshold")
        if backend_threshold is not None:
            backend_threshold = float(backend_threshold)
            if not 0.0 <= backend_threshold <= 1.0:
                raise ValueError("backend_score_threshold必须位于[0,1]")
        return cls(
            scene_name=str(payload["scene_name"]),
            frames=frames,
            checkpoint=str(payload.get("checkpoint", "")),
            model_load_latency_s=(
                None
                if payload.get("model_load_latency_s") is None
                else float(payload["model_load_latency_s"])
            ),
            sparseconv_backend=str(payload.get("sparseconv_backend", "")),
            sparseconv_layout_converted_count=int(
                payload.get("sparseconv_layout_converted_count", 0)
            ),
            checkpoint_strict_load=bool(payload.get("checkpoint_strict_load", False)),
            repo_commit=str(payload.get("repo_commit", "")),
            backend_score_threshold=backend_threshold,
            schema_version=str(payload["schema_version"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scene_name": self.scene_name,
            "checkpoint": self.checkpoint,
            "model_load_latency_s": self.model_load_latency_s,
            "sparseconv_backend": self.sparseconv_backend,
            "sparseconv_layout_converted_count": (
                self.sparseconv_layout_converted_count
            ),
            "checkpoint_strict_load": self.checkpoint_strict_load,
            "repo_commit": self.repo_commit,
            "backend_score_threshold": self.backend_score_threshold,
            "frames": [item.to_dict() for item in self.frames],
        }

    @classmethod
    def load(cls, path: str | Path) -> "BEVFusionSequenceResult":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def dump(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output
