"""同一进程内共享 GPU 推理任务的优先级仲裁。

CARLA 渲染、WSL BEVFusion 后端和 Windows 前视 VRU 检测器共用一块 GPU。
优先级租约可防止跨运行环境推理重叠，同时不改变传感器输入或关闭任一安全通道。
"""

from __future__ import annotations

from contextlib import contextmanager
import threading
import time


class PriorityInferenceArbiter:
    """Adaptive high/normal-priority inference lease with timing telemetry.

    Healthy inference stays concurrent for maximum throughput.  A slow primary
    BEVFusion observation activates a bounded exclusive window; during that
    window normal-priority VRU work yields to the primary chain.  Stable
    primary observations automatically retire the protection window.
    """

    def __init__(
        self,
        *,
        primary_slow_threshold_s: float = 0.80,
        protection_primary_observations: int = 3,
    ) -> None:
        if primary_slow_threshold_s <= 0:
            raise ValueError("primary slow threshold must be positive")
        if protection_primary_observations < 1:
            raise ValueError("protection observations must be >=1")
        self._condition = threading.Condition()
        self.primary_slow_threshold_s = float(primary_slow_threshold_s)
        self.protection_primary_observations = int(
            protection_primary_observations
        )
        self._high_inflight = 0
        self._normal_inflight = 0
        self._high_waiters = 0
        self._protected_remaining = 0
        self._protection_activations = 0
        self._stats = {
            "high": {
                "leases": 0, "exclusive_leases": 0,
                "wait_s": 0.0, "hold_s": 0.0,
            },
            "normal": {
                "leases": 0, "exclusive_leases": 0,
                "wait_s": 0.0, "hold_s": 0.0,
            },
        }

    @contextmanager
    def lease(self, priority: str):
        """Acquire one window; queued high-priority work precedes normal work."""

        if priority not in self._stats:
            raise ValueError("inference priority must be high or normal")
        wait_started = time.perf_counter()
        with self._condition:
            exclusive = self._protected_remaining > 0
            if exclusive and priority == "high":
                self._high_waiters += 1
            try:
                while exclusive and (
                    self._high_inflight > 0
                    or self._normal_inflight > 0
                    or (priority == "normal" and self._high_waiters > 0)
                ):
                    self._condition.wait()
                if priority == "high":
                    self._high_inflight += 1
                else:
                    self._normal_inflight += 1
            finally:
                if exclusive and priority == "high":
                    self._high_waiters -= 1
        acquired = time.perf_counter()
        try:
            yield
        finally:
            released = time.perf_counter()
            with self._condition:
                stats = self._stats[priority]
                stats["leases"] += 1
                stats["exclusive_leases"] += int(exclusive)
                stats["wait_s"] += acquired - wait_started
                stats["hold_s"] += released - acquired
                if priority == "high":
                    self._high_inflight -= 1
                else:
                    self._normal_inflight -= 1
                self._condition.notify_all()

    def observe_primary_latency(self, latency_s: float) -> None:
        """Activate or retire protection using measured BEVFusion model time."""

        latency_s = float(latency_s)
        with self._condition:
            if latency_s > self.primary_slow_threshold_s:
                if self._protected_remaining == 0:
                    self._protection_activations += 1
                self._protected_remaining = self.protection_primary_observations
            elif self._protected_remaining > 0:
                self._protected_remaining -= 1
            self._condition.notify_all()

    def snapshot(self) -> dict:
        """Return stable aggregate metrics suitable for JSON telemetry."""

        with self._condition:
            output = {
                "policy": "adaptive_exclusive_high_priority_first",
                "primary_slow_threshold_s": self.primary_slow_threshold_s,
                "protection_primary_observations": (
                    self.protection_primary_observations
                ),
                "protection_activations": int(self._protection_activations),
                "protected_remaining": int(self._protected_remaining),
                "high_inflight": int(self._high_inflight),
                "normal_inflight": int(self._normal_inflight),
                "high_waiters": int(self._high_waiters),
                "priorities": {},
            }
            for priority, values in self._stats.items():
                leases = int(values["leases"])
                output["priorities"][priority] = {
                    "leases": leases,
                    "exclusive_leases": int(values["exclusive_leases"]),
                    "total_wait_s": float(values["wait_s"]),
                    "mean_wait_s": (
                        float(values["wait_s"]) / leases if leases else 0.0
                    ),
                    "total_hold_s": float(values["hold_s"]),
                    "mean_hold_s": (
                        float(values["hold_s"]) / leases if leases else 0.0
                    ),
                }
            return output


GLOBAL_GPU_INFERENCE_ARBITER = PriorityInferenceArbiter()
