"""CARLA 运行时发现、解释器转接、服务端启动和环境检查。"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


REQUIRED_RUNTIME_MODULES = ("carla", "pygame", "torch", "torchvision", "cv2")
_MAP_BACKUP_SUFFIX = ".fusion_tracking_backup"


def _path_from_environment(*names: str) -> Path | None:
    """读取显式部署路径；环境变量优先于自动发现。"""

    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return Path(value).expanduser()
    return None


def _discover_carla_root() -> Path:
    """在不固化用户名或组织目录名的前提下定位 CARLA 0.9.15。"""

    explicit = _path_from_environment("FUSION_SAFETY_CARLA_ROOT", "CARLA_ROOT")
    if explicit is not None:
        return explicit

    home = Path.home()
    candidates = [
        home / "CARLA_0.9.15",
        home / "Desktop" / "CARLA_0.9.15",
        Path("C:/CARLA_0.9.15"),
        Path("D:/CARLA_0.9.15"),
    ]
    desktop = home / "Desktop"
    if desktop.is_dir():
        # 支持“桌面/任意项目目录/03_tools/CARLA_0.9.15”这类本地布局；
        # 任意项目目录不会写入源码或发布清单。
        candidates.extend(desktop.glob("*/03_tools/CARLA_0.9.15"))
    for candidate in candidates:
        if (candidate / "CarlaUE4.exe").is_file():
            return candidate
    return home / "fusion_safety_runtime" / "CARLA_0.9.15"


def _discover_carla_python(carla_root: Path) -> Path:
    """定位带 CARLA、PyTorch 与 Pygame 的隔离 Python 解释器。"""

    explicit = _path_from_environment(
        "FUSION_SAFETY_CARLA_PYTHON", "CARLA_PYTHON"
    )
    if explicit is not None:
        return explicit
    candidates = [
        carla_root.parent.parent
        / "04_code/carla_local/.venv/Scripts/python.exe",
        carla_root / ".venv/Scripts/python.exe",
        Path.home() / "fusion_safety_runtime/carla/.venv/Scripts/python.exe",
    ]
    desktop = Path.home() / "Desktop"
    if desktop.is_dir():
        candidates.extend(
            desktop.glob("*/04_code/carla_local/.venv/Scripts/python.exe")
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


_DEFAULT_CARLA_ROOT = _discover_carla_root()
DEFAULT_CARLA_SERVER = _path_from_environment("FUSION_SAFETY_CARLA_SERVER") or (
    _DEFAULT_CARLA_ROOT / "CarlaUE4.exe"
)
DEFAULT_CARLA_AGENTS_ROOT = _path_from_environment(
    "FUSION_SAFETY_CARLA_AGENTS_ROOT"
) or (_DEFAULT_CARLA_ROOT / "PythonAPI/carla")
DEFAULT_CARLA_PYTHON = _discover_carla_python(_DEFAULT_CARLA_ROOT)


def _default_engine_ini(server_path: Path) -> Path:
    return server_path.parent / "CarlaUE4" / "Config" / "DefaultEngine.ini"


def _restore_stale_map_override(server_path: Path) -> None:
    """恢复上次被强制中止时可能遗留的临时默认地图覆盖。"""

    ini_path = _default_engine_ini(server_path)
    backup_path = ini_path.with_name(ini_path.name + _MAP_BACKUP_SUFFIX)
    if backup_path.exists():
        ini_path.write_bytes(backup_path.read_bytes())
        backup_path.unlink()
        print("[CARLA server] 已恢复上次中断遗留的DefaultEngine.ini备份", flush=True)


@contextmanager
def _temporary_default_map(server_path: Path, map_name: str):
    """事务式覆盖CARLA Shipping包默认地图，退出作用域后逐字节恢复。"""

    if not map_name:
        yield
        return
    ini_path = _default_engine_ini(server_path)
    if not ini_path.exists():
        raise FileNotFoundError(f"找不到CARLA默认地图配置：{ini_path}")
    _restore_stale_map_override(server_path)
    original = ini_path.read_bytes()
    text = original.decode("utf-8")
    asset = f"/Game/Carla/Maps/{map_name}.{map_name}"
    keys = ("EditorStartupMap", "GameDefaultMap", "ServerDefaultMap", "TransitionMap")
    for key in keys:
        text, count = re.subn(
            rf"(?m)^{key}=.*$", f"{key}={asset}", text, count=1
        )
        if count != 1:
            raise RuntimeError(f"DefaultEngine.ini缺少唯一字段：{key}")
    backup_path = ini_path.with_name(ini_path.name + _MAP_BACKUP_SUFFIX)
    backup_path.write_bytes(original)
    ini_path.write_text(text, encoding="utf-8")
    print(f"[CARLA server] 临时默认地图覆盖：{asset}", flush=True)
    try:
        yield
    finally:
        if backup_path.exists():
            ini_path.write_bytes(backup_path.read_bytes())
            backup_path.unlink()
            print("[CARLA server] DefaultEngine.ini已恢复", flush=True)


def _terminate_launched_server(process) -> None:
    """仅结束本次启动器创建的精确Windows进程树。"""

    pid = getattr(process, "pid", None)
    if not pid:
        return
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=8.0,
        )
    except subprocess.TimeoutExpired:
        pass
    # The Shipping executable can detach from the short-lived CarlaUE4.exe
    # launcher.  A PID-tree kill then reports success while the API port and
    # render process remain alive.  This function is called only for a server
    # created by this invocation, so the narrow image fallback is in scope.
    _force_stop_carla_images()


def _force_stop_carla_images() -> None:
    """Force-stop both executable layers of the Windows CARLA package."""

    command = (
        "Get-Process -Name 'CarlaUE4','CarlaUE4-Win64-Shipping' "
        "-ErrorAction SilentlyContinue | Stop-Process -Force"
    )
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=15.0,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def stop_existing_carla_server() -> None:
    """Stop only the known CARLA executable tree for an explicit map restart.

    Normal launch paths never call this implicitly. Automated multi-map
    regression opts in through ``restart_if_map_mismatch``.
    """

    if os.name != "nt":
        raise RuntimeError("automatic CARLA process restart is implemented for Windows only")
    _force_stop_carla_images()
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if server_world(timeout_s=0.5) is None:
            return
        time.sleep(0.25)
    raise RuntimeError("CARLA process was stopped but the API port is still responding")


def missing_modules() -> list[str]:
    return [name for name in REQUIRED_RUNTIME_MODULES if importlib.util.find_spec(name) is None]


def handoff_if_needed(script_path: Path, project_root: Path) -> None:
    """当前解释器缺依赖时，保持所有 CLI 参数转交给独立 CARLA 环境。"""

    missing = missing_modules()
    if not missing:
        return
    if not DEFAULT_CARLA_PYTHON.exists():
        raise RuntimeError(f"当前解释器缺少 {missing}，且 CARLA 环境不存在：{DEFAULT_CARLA_PYTHON}")
    print(f"[环境转接] 当前解释器缺少 {missing}，自动改用：{DEFAULT_CARLA_PYTHON}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    command = [str(DEFAULT_CARLA_PYTHON), str(script_path.resolve()), *sys.argv[1:]]
    raise SystemExit(
        subprocess.run(command, cwd=project_root, env=environment, check=False).returncode
    )


def server_world(host="127.0.0.1", port=2000, timeout_s=2.0):
    import carla

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout_s)
        return client.get_world()
    except RuntimeError:
        return None


def launch_server_and_wait(
    max_wait_s=120.0,
    host="127.0.0.1",
    port=2000,
    *,
    extra_args: tuple[str, ...] = (),
    startup_map: str = "",
    restart_if_map_mismatch: bool = False,
) -> subprocess.Popen | None:
    """启动CARLA，并可让UE直接冷启动到指定地图。

    ``startup_map``为空时保持旧阶段行为。非空时把官方UE地图路径作为第一个位置
    参数传给``CarlaUE4.exe``，避免先进入默认Town10再调用``client.load_world``。
    """

    startup_map_name = startup_map.rsplit("/", 1)[-1] if startup_map else ""

    world = server_world(host, port)
    if world is not None:
        current_map = world.get_map().name
        print(f"[CARLA server] 已在运行 | map={current_map}")
        if startup_map_name and startup_map_name not in current_map:
            if not restart_if_map_mismatch:
                raise RuntimeError(
                    f"已有CARLA地图是{current_map}，但本次要求{startup_map_name}；"
                    "为避免不稳定的运行中热切图，请先关闭旧CARLA再重试"
                )
            print(
                "[CARLA server] regression map transition: "
                f"{current_map} -> {startup_map_name}; restarting exact CARLA tree",
                flush=True,
            )
            stop_existing_carla_server()
            world = None
        if world is not None and extra_args:
            print(
                "[CARLA server警告] 服务端不是本次新建，无法确认渲染兼容参数是否生效："
                + " ".join(extra_args),
                flush=True,
            )
        if world is not None:
            return None
    if not DEFAULT_CARLA_SERVER.exists():
        raise FileNotFoundError(f"找不到 CARLA 服务端：{DEFAULT_CARLA_SERVER}")
    _restore_stale_map_override(DEFAULT_CARLA_SERVER)
    command = [str(DEFAULT_CARLA_SERVER)]
    command.extend([
        "-windowed", "-ResX=1280", "-ResY=720",
        "-quality-level=Low", f"-carla-port={port}",
    ])
    command.extend(extra_args)
    print(f"[CARLA server] 正在启动：{DEFAULT_CARLA_SERVER}", flush=True)
    if extra_args:
        print(f"[CARLA server] 附加渲染参数：{' '.join(extra_args)}", flush=True)
    process = None
    launch_succeeded = False
    with _temporary_default_map(DEFAULT_CARLA_SERVER, startup_map_name):
        process = subprocess.Popen(
            command,
            cwd=DEFAULT_CARLA_SERVER.parent,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        try:
            deadline = time.monotonic() + max_wait_s
            next_message = 0.0
            wrong_map_since = None
            while time.monotonic() < deadline:
                # A UE4 render/shader crash closes the launcher-owned process
                # immediately. Waiting until the network deadline in that
                # state hides the real infrastructure failure behind a generic
                # timeout and can waste several minutes per regression case.
                if process is not None:
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            "CARLA server exited during startup "
                            f"(return_code={return_code}); inspect the latest "
                            "CarlaUE4 crash report instead of retrying the Agent"
                        )
                world = server_world(host, port, timeout_s=3.0)
                if world is not None:
                    current_map = world.get_map().name
                    if not startup_map_name or startup_map_name in current_map:
                        print(f"[CARLA server] 启动完成 | map={current_map}", flush=True)
                        launch_succeeded = True
                        return process
                    if wrong_map_since is None:
                        wrong_map_since = time.monotonic()
                        print(
                            f"[CARLA server] 已连接但地图仍为{current_map}，等待启动覆盖生效 ...",
                            flush=True,
                        )
                    elif time.monotonic() - wrong_map_since >= 15.0:
                        raise RuntimeError(
                            f"CARLA已启动但地图覆盖未生效：current={current_map}, "
                            f"expected={startup_map_name}；已提前停止等待"
                        )
                if time.monotonic() >= next_message:
                    remaining = max(int(deadline - time.monotonic()), 0)
                    print(
                        f"[CARLA server] 地图载入中，剩余等待上限 {remaining}s ...",
                        flush=True,
                    )
                    next_message = time.monotonic() + 10.0
                time.sleep(2.0)
            raise RuntimeError("CARLA 启动超过等待上限；请查看 CarlaUE4 窗口是否报错")
        finally:
            if not launch_succeeded and process is not None:
                _terminate_launched_server(process)


def check_environment(host="127.0.0.1", port=2000) -> int:
    import carla
    import cv2
    import pygame
    import torch
    import torchvision

    print(f"[环境] Python: {sys.executable}")
    print(f"[环境] CARLA API: {getattr(carla, '__version__', '0.9.x')} | pygame={pygame.version.ver}")
    print(f"[环境] torch={torch.__version__} | torchvision={torchvision.__version__}")
    print(f"[环境] CUDA={torch.cuda.is_available()} | OpenCV={cv2.__version__}")
    world = server_world(host, port, timeout_s=3.0)
    if world is None:
        print(f"[CARLA server] 未连接：{host}:{port}")
        return 2
    print(f"[CARLA server] 已连接 | map={world.get_map().name}")
    return 0
