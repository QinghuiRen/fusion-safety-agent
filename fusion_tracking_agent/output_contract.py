"""可部署产品输出的稳定文件名与数据模式。

读取端保留历史文件名兼容，以便审计已验证实验；所有新运行只生成产品命名文件。
"""

from __future__ import annotations

from pathlib import Path


PRODUCT_ID = "fusion_safety_agent"
FRAME_SCHEMA = "fusion_tracking.safety_agent.frame.v1"
SUMMARY_SCHEMA = "fusion_tracking.safety_agent.summary.v1"

FRAME_LOG_FILENAME = "fusion_safety_frames.jsonl"
SUMMARY_FILENAME = "fusion_safety_summary.json"
REPLAY_VIDEO_FILENAME = "fusion_safety_replay.mp4"
REPLAY_PREVIEW_FILENAME = "fusion_safety_replay_preview.png"
REPLAY_SUMMARY_FILENAME = "fusion_safety_replay_summary.json"

LEGACY_FRAME_LOG_FILENAMES = ("stage6_20_agent_frames.jsonl",)
LEGACY_SUMMARY_FILENAMES = ("stage6_20_agent_summary.json",)


def first_existing(directory: str | Path, names: tuple[str, ...]) -> Path | None:
    """按约定顺序返回第一个存在的普通文件。"""

    root = Path(directory).expanduser().resolve()
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def resolve_frame_log(source: str | Path) -> Path:
    """解析产品逐帧日志，同时允许读取历史运行目录。"""

    path = Path(source).expanduser().resolve()
    if path.is_dir():
        candidate = first_existing(
            path, (FRAME_LOG_FILENAME, *LEGACY_FRAME_LOG_FILENAMES)
        )
        if candidate is None:
            expected = ", ".join(
                (FRAME_LOG_FILENAME, *LEGACY_FRAME_LOG_FILENAMES)
            )
            raise FileNotFoundError(
                f"No Fusion Safety frame log under {path}; expected one of: {expected}"
            )
        return candidate
    if not path.is_file():
        raise FileNotFoundError(f"Fusion Safety frame log not found: {path}")
    return path


def resolve_summary(directory: str | Path) -> Path | None:
    """从运行目录解析当前或历史 Agent 汇总文件。"""

    return first_existing(
        directory, (SUMMARY_FILENAME, *LEGACY_SUMMARY_FILENAMES)
    )
