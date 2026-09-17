"""在线融合与控制流水线的低开销时延统计。

各时钟口径严格分离：

* 仿真时间衡量控制周期和传感器源更新率；
* ``perf_counter`` 衡量主机墙钟吞吐与端到端延迟；
* 后端诊断单独衡量神经网络推理时间。

Keeping those domains separate prevents a slow CARLA wall-clock run from being
reported as a fast perception loop (or the reverse).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _finite_nonnegative(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


@dataclass
class _SecondsSeries:
    values: list[float] = field(default_factory=list)

    def add(self, value: Any) -> None:
        number = _finite_nonnegative(value)
        if number is not None:
            self.values.append(number)

    def summary(self) -> dict[str, int | float | None]:
        return self._summary_values(self.values)

    @staticmethod
    def _summary_values(values: list[float]) -> dict[str, int | float | None]:
        if not values:
            return {
                "samples": 0,
                "mean_s": None,
                "p50_s": None,
                "p95_s": None,
                "max_s": None,
            }
        samples = np.asarray(values, dtype=float)
        return {
            "samples": int(samples.size),
            "mean_s": float(np.mean(samples)),
            "p50_s": float(np.percentile(samples, 50)),
            "p95_s": float(np.percentile(samples, 95)),
            "max_s": float(np.max(samples)),
        }

    def cold_and_steady_summary(self) -> dict:
        """Expose first-call kernel warm-up without contaminating steady state."""

        output = self.summary()
        output["cold_first_s"] = self.values[0] if self.values else None
        output["steady_state"] = self._summary_values(self.values[1:])
        return output


def _span(first: float | None, last: float | None) -> float | None:
    if first is None or last is None:
        return None
    return max(float(last - first), 0.0)


def _interval_rate(events: int, span_s: float | None) -> float | None:
    """Return interval rate; two samples create one measurable interval."""

    if events < 2 or span_s is None or span_s <= 0.0:
        return None
    return float((events - 1) / span_s)


class OnlineTimingTelemetry:
    """Accumulate one episode of timing data without touching control inputs."""

    def __init__(self, stale_threshold_s: float):
        self.stale_threshold_s = float(stale_threshold_s)
        self.control_steps = 0
        self._first_control_sim_s: float | None = None
        self._last_control_sim_s: float | None = None
        self._first_control_wall_s: float | None = None
        self._last_control_wall_s: float | None = None
        self.perception_results = 0
        self.perception_successes = 0
        self.perception_errors = 0
        self._first_source_sim_s: float | None = None
        self._last_source_sim_s: float | None = None
        self._first_delivery_wall_s: float | None = None
        self._last_delivery_wall_s: float | None = None
        self.missing_fusion_age_frames = 0
        self.stale_fusion_age_frames = 0
        self.algorithm_step = _SecondsSeries()
        self.control_simulation_gap = _SecondsSeries()
        self.control_wall_gap = _SecondsSeries()
        self.fusion_age = _SecondsSeries()
        self.worker_queue = _SecondsSeries()
        self.windows_packet_preprocess = _SecondsSeries()
        self.backend_roundtrip_wait = _SecondsSeries()
        self.result_adapter_fusion = _SecondsSeries()
        self.worker_process = _SecondsSeries()
        self.submit_to_delivery = _SecondsSeries()
        self.backend_preprocess = _SecondsSeries()
        self.backend_inference = _SecondsSeries()
        self.perception_source_gap = _SecondsSeries()
        self.perception_delivery_gap = _SecondsSeries()
        self.output_json_serialization = _SecondsSeries()
        self.output_write_flush = _SecondsSeries()
        self.output_total = _SecondsSeries()
        self.output_bytes = 0
        self.backend_stages: dict[str, _SecondsSeries] = {}
        self.host_packet_stages: dict[str, _SecondsSeries] = {}

    def observe_perception(self, result: Any, delivery_wall_s: float) -> dict:
        """Record a *new* worker result and return its JSON-safe timing snapshot."""

        self.perception_results += 1
        self.perception_errors += int(result.error is not None)
        self.perception_successes += int(result.error is None)
        source_sim_s = float(result.timestamp_s)
        delivery_wall_s = float(delivery_wall_s)
        if self._last_source_sim_s is not None:
            self.perception_source_gap.add(source_sim_s - self._last_source_sim_s)
        if self._last_delivery_wall_s is not None:
            self.perception_delivery_gap.add(
                delivery_wall_s - self._last_delivery_wall_s
            )
        if self._first_source_sim_s is None:
            self._first_source_sim_s = source_sim_s
            self._first_delivery_wall_s = delivery_wall_s
        self._last_source_sim_s = source_sim_s
        self._last_delivery_wall_s = delivery_wall_s

        self.worker_queue.add(result.queue_latency_s)
        self.windows_packet_preprocess.add(result.preprocess_latency_s)
        self.backend_roundtrip_wait.add(result.detector_latency_s)
        self.result_adapter_fusion.add(result.fusion_latency_s)
        self.worker_process.add(result.latency_s)
        submit_to_delivery_s = _finite_nonnegative(result.queue_latency_s)
        process_s = _finite_nonnegative(result.latency_s)
        if submit_to_delivery_s is not None and process_s is not None:
            submit_to_delivery_s += process_s
            self.submit_to_delivery.add(submit_to_delivery_s)

        diagnostics = dict(result.diagnostics or {})
        self.backend_preprocess.add(diagnostics.get("backend_preprocess_latency_s"))
        self.backend_inference.add(diagnostics.get("backend_inference_latency_s"))
        for name, value in dict(
            diagnostics.get("backend_stage_latencies_s") or {}
        ).items():
            self.backend_stages.setdefault(str(name), _SecondsSeries()).add(value)
        for name, value in dict(
            diagnostics.get("host_packet_stage_latencies_s") or {}
        ).items():
            self.host_packet_stages.setdefault(str(name), _SecondsSeries()).add(value)
        return {
            "source_frame": int(result.frame),
            "source_timestamp_s": source_sim_s,
            "error": result.error,
            "queue_latency_s": _finite_nonnegative(result.queue_latency_s),
            "worker_process_latency_s": _finite_nonnegative(result.latency_s),
            "submit_to_delivery_latency_s": submit_to_delivery_s,
            "windows_packet_preprocess_latency_s": _finite_nonnegative(
                result.preprocess_latency_s
            ),
            "backend_roundtrip_wait_latency_s": _finite_nonnegative(
                result.detector_latency_s
            ),
            "result_adapter_fusion_latency_s": _finite_nonnegative(
                result.fusion_latency_s
            ),
            "backend_preprocess_latency_s": _finite_nonnegative(
                diagnostics.get("backend_preprocess_latency_s")
            ),
            "backend_inference_latency_s": _finite_nonnegative(
                diagnostics.get("backend_inference_latency_s")
            ),
            "backend_inference_precision": diagnostics.get(
                "backend_inference_precision"
            ),
        }

    def observe_control(
        self,
        *,
        simulation_timestamp_s: float,
        wall_timestamp_s: float,
        algorithm_step_latency_s: float,
        fusion_age_s: float | None,
    ) -> None:
        self.control_steps += 1
        simulation_timestamp_s = float(simulation_timestamp_s)
        wall_timestamp_s = float(wall_timestamp_s)
        if self._last_control_sim_s is not None:
            self.control_simulation_gap.add(
                simulation_timestamp_s - self._last_control_sim_s
            )
        if self._last_control_wall_s is not None:
            self.control_wall_gap.add(wall_timestamp_s - self._last_control_wall_s)
        if self._first_control_sim_s is None:
            self._first_control_sim_s = simulation_timestamp_s
            self._first_control_wall_s = wall_timestamp_s
        self._last_control_sim_s = simulation_timestamp_s
        self._last_control_wall_s = wall_timestamp_s
        self.algorithm_step.add(algorithm_step_latency_s)
        age = _finite_nonnegative(fusion_age_s)
        if age is None:
            self.missing_fusion_age_frames += 1
        else:
            self.fusion_age.add(age)
            self.stale_fusion_age_frames += int(age > self.stale_threshold_s)

    def observe_output(self, timing: dict | None) -> None:
        if not timing:
            return
        self.output_json_serialization.add(timing.get("json_serialization_s"))
        self.output_write_flush.add(timing.get("write_flush_s"))
        self.output_total.add(timing.get("total_s"))
        self.output_bytes += int(timing.get("bytes", 0) or 0)

    def finalize(self) -> dict:
        control_sim_span = _span(
            self._first_control_sim_s, self._last_control_sim_s
        )
        control_wall_span = _span(
            self._first_control_wall_s, self._last_control_wall_s
        )
        source_sim_span = _span(self._first_source_sim_s, self._last_source_sim_s)
        delivery_wall_span = _span(
            self._first_delivery_wall_s, self._last_delivery_wall_s
        )
        return {
            "schema": "fusion_tracking.online_timing.v1",
            "clock_semantics": {
                "simulation": "CARLA/game timestamps; source and control cadence",
                "wall": "time.perf_counter; host delivery cadence and latency",
                "algorithm_step": "step computation before JSON serialization/I/O",
                "backend_inference": "CUDA-synchronized neural-network forward when exposed",
            },
            "control": {
                "steps": self.control_steps,
                "simulation_span_s": control_sim_span,
                "wall_span_s": control_wall_span,
                "simulation_hz": _interval_rate(self.control_steps, control_sim_span),
                "wall_hz": _interval_rate(self.control_steps, control_wall_span),
                "real_time_factor": (
                    control_sim_span / control_wall_span
                    if control_sim_span is not None
                    and control_wall_span is not None
                    and control_wall_span > 0.0
                    else None
                ),
                "algorithm_step_latency": self.algorithm_step.summary(),
                "simulation_frame_gap": self.control_simulation_gap.summary(),
                "wall_frame_gap": self.control_wall_gap.summary(),
            },
            "perception": {
                "unique_results": self.perception_results,
                "successful_results": self.perception_successes,
                "error_results": self.perception_errors,
                "source_simulation_span_s": source_sim_span,
                "delivery_wall_span_s": delivery_wall_span,
                "source_hz": _interval_rate(self.perception_results, source_sim_span),
                "delivery_wall_hz": _interval_rate(
                    self.perception_results, delivery_wall_span
                ),
                "source_result_gap": self.perception_source_gap.summary(),
                "delivery_result_gap": self.perception_delivery_gap.summary(),
                "worker_queue_latency": self.worker_queue.summary(),
                "windows_packet_preprocess_latency": (
                    self.windows_packet_preprocess.summary()
                ),
                "backend_roundtrip_wait_latency": (
                    self.backend_roundtrip_wait.cold_and_steady_summary()
                ),
                "backend_preprocess_latency": (
                    self.backend_preprocess.cold_and_steady_summary()
                ),
                "backend_inference_latency": (
                    self.backend_inference.cold_and_steady_summary()
                ),
                "backend_stage_latencies": {
                    name: series.cold_and_steady_summary()
                    for name, series in sorted(self.backend_stages.items())
                },
                "host_packet_stage_latencies": {
                    name: series.cold_and_steady_summary()
                    for name, series in sorted(self.host_packet_stages.items())
                },
                "result_adapter_fusion_latency": (
                    self.result_adapter_fusion.summary()
                ),
                "worker_process_latency": self.worker_process.summary(),
                "submit_to_delivery_latency": self.submit_to_delivery.summary(),
            },
            "output": {
                "bytes": self.output_bytes,
                "json_serialization_latency": (
                    self.output_json_serialization.summary()
                ),
                "write_flush_latency": self.output_write_flush.summary(),
                "total_latency": self.output_total.summary(),
            },
            "freshness": {
                "stale_threshold_s": self.stale_threshold_s,
                "missing_age_frames": self.missing_fusion_age_frames,
                "missing_age_fraction": (
                    self.missing_fusion_age_frames / self.control_steps
                    if self.control_steps
                    else 0.0
                ),
                "stale_age_frames": self.stale_fusion_age_frames,
                "stale_age_fraction": (
                    self.stale_fusion_age_frames / self.control_steps
                    if self.control_steps
                    else 0.0
                ),
                "fusion_age": self.fusion_age.summary(),
            },
        }
