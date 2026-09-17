"""产品结构化输出的隐私脱敏工具。

算法内部仍使用真实绝对路径访问模型与运行时；只有写入日志、诊断报告或交付证据时
才替换机器相关信息，因此不会影响文件访问、感知结果或车辆控制。
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
from typing import Any


_WINDOWS_USER_RE = re.compile(
    r"[A-Z]:[\\/]Users[\\/][^\\/\s\"']+", re.IGNORECASE
)
_WSL_WINDOWS_USER_RE = re.compile(
    r"/mnt/[a-z]/Users/[^/\s\"']+", re.IGNORECASE
)
_WSL_HOME_RE = re.compile(r"/home/[^/\s\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
)
_GENERIC_HOSTNAME_RE = re.compile(
    r"\b(?:DESKTOP|LAPTOP)-[A-Z0-9-]+\b", re.IGNORECASE
)
_LOCAL_HOSTNAME = os.environ.get("COMPUTERNAME", "").strip()
_LOCAL_HOSTNAME_RE = (
    re.compile(re.escape(_LOCAL_HOSTNAME), re.IGNORECASE)
    if _LOCAL_HOSTNAME
    else None
)


@lru_cache(maxsize=16)
def _path_variants(path: Path) -> tuple[str, ...]:
    """缓存 Windows 路径及其 WSL ``/mnt`` 表示，避免逐字段访问文件系统。"""

    resolved = path.expanduser().resolve()
    variants = {str(resolved), resolved.as_posix()}
    drive = resolved.drive.rstrip(":").lower()
    if drive:
        relative = resolved.as_posix().split(":", 1)[1].lstrip("/")
        variants.add(f"/mnt/{drive}/{relative}")
    return tuple(sorted(variants, key=len, reverse=True))


def sanitize_text_for_export(
    value: str,
    *,
    product_root: str | Path | None = None,
) -> str:
    """移除字符串中的用户目录、WSL 主目录和本机主机名。"""

    text = str(value)
    if product_root is not None:
        for variant in _path_variants(Path(product_root)):
            text = text.replace(variant, "${PRODUCT_ROOT}")

    user_home = Path.home()
    for variant in _path_variants(user_home):
        text = text.replace(variant, "${USERPROFILE}")

    # 处理来自其他机器的证据文件，不能只依赖当前登录用户目录。
    text = _WINDOWS_USER_RE.sub("${USERPROFILE}", text)
    text = _WSL_WINDOWS_USER_RE.sub("${USERPROFILE_WSL}", text)
    text = _WSL_HOME_RE.sub("${WSL_HOME}", text)
    # 运行异常可能包含代理账号或主机名；它们对复现算法没有价值，导出时统一隐藏。
    text = _EMAIL_RE.sub("${EMAIL}", text)
    text = _GENERIC_HOSTNAME_RE.sub("${HOSTNAME}", text)
    if _LOCAL_HOSTNAME_RE is not None:
        text = _LOCAL_HOSTNAME_RE.sub("${HOSTNAME}", text)
    return text


def sanitize_for_export(
    value: Any,
    *,
    product_root: str | Path | None = None,
) -> Any:
    """递归脱敏 JSON 兼容结构，同时保持数值和布尔类型不变。"""

    if isinstance(value, str):
        return sanitize_text_for_export(value, product_root=product_root)
    if isinstance(value, dict):
        return {
            key: sanitize_for_export(item, product_root=product_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize_for_export(item, product_root=product_root) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            sanitize_for_export(item, product_root=product_root) for item in value
        )
    return value
