"""Lightweight performance monitor and optional torch.profiler integration.

Base 层只关心通用训练阶段（DataLoader 等待 / H2D / forward / backward / optimizer），
不接触 sequence、文件、缓存或 task 专属的 Dataset schema。
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any, ContextManager

import torch


class PerformanceMonitor:
    """Shared lightweight timing and optional torch.profiler integration.

    支持三种模式：
    - ``off``：完全关闭，不做任何统计也不启动 profiler。
    - ``light``：长期轻量统计，输出 ``perf/*`` 指标。
    - ``profile``：轻量统计 + 短时间内 ``torch.profiler`` 采集并写 TensorBoard trace。
    """

    VALID_MODES = {"off", "light", "profile"}

    def __init__(
        self,
        *,
        mode: str,
        device: torch.device,
        output_dir: str | Path,
        total_steps: int,
        log_every_steps: int,
        warmup_steps: int = 20,
        profile_wait_steps: int = 5,
        profile_warmup_steps: int = 5,
        profile_active_steps: int = 10,
        is_primary: bool = True,
    ) -> None:
        mode = str(mode).lower().strip()
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"performance.mode must be one of "
                f"{sorted(self.VALID_MODES)}, got {mode!r}"
            )

        self.mode = mode
        self.device = device
        self.output_dir = Path(output_dir)
        self.total_steps = int(total_steps)
        self.log_every_steps = max(1, int(log_every_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.is_primary = bool(is_primary)

        self.profiler: Any | None = self._build_profiler(
            wait_steps=profile_wait_steps,
            warmup_steps=profile_warmup_steps,
            active_steps=profile_active_steps,
        )

        self._profiler_started = False
        self._reset_window()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def _build_profiler(
        self,
        *,
        wait_steps: int,
        warmup_steps: int,
        active_steps: int,
    ) -> Any:
        """构造 ``torch.profiler``。仅 profile 模式 + primary 进程会真正创建。"""
        if self.mode != "profile" or not self.is_primary:
            return None

        activities = [
            torch.profiler.ProfilerActivity.CPU,
        ]

        if self.device.type == "cuda":
            activities.append(
                torch.profiler.ProfilerActivity.CUDA
            )

        trace_dir = self.output_dir / "profiler"
        trace_dir.mkdir(parents=True, exist_ok=True)

        return torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=max(0, int(wait_steps)),
                warmup=max(0, int(warmup_steps)),
                active=max(1, int(active_steps)),
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(trace_dir)
            ),
            # 默认关闭高开销选项。轻量模式长期运行不追踪 shape/stack/memory。
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )

    def start(self) -> None:
        """启动 profiler。仅 profile 模式有效果。"""
        if self.profiler is None or self._profiler_started:
            return

        self.profiler.start()
        self._profiler_started = True

    def stop(self) -> None:
        """停止 profiler。建议在 ``learn()`` 的 ``finally`` 分支调用。"""
        if self.profiler is None or not self._profiler_started:
            return

        self.profiler.stop()
        self._profiler_started = False

    def section(
        self,
        name: str,
    ) -> ContextManager:
        """Create a named range visible in profile mode.

        轻量 / 关闭模式下退化为 ``nullcontext()``，对正常训练无开销。
        """
        if self.profiler is None:
            return nullcontext()

        return torch.profiler.record_function(name)

    def observe_step(
        self,
        *,
        global_step: int,
        batch_size: int,
        data_wait_seconds: float,
    ) -> dict[str, float] | None:
        """Update the lightweight window and optionally return log metrics.

        每个 ``global_step`` 调用一次：
        - 总是推进 ``profiler.step()``（如果启用）；
        - 在 ``warmup_steps`` 之前不计入窗口；
        - 每 ``log_every_steps`` 返回一次 ``perf/*`` 指标 dict。
        """
        if self.profiler is not None:
            self.profiler.step()

        if not self.enabled:
            return None

        global_step = int(global_step)

        # CUDA 初始化、worker 启动和文件冷缓存不用于稳定吞吐估计。
        if global_step <= self.warmup_steps:
            if global_step == self.warmup_steps:
                # warmup 结束，重置窗口让吞吐 / ETA 从此刻起算
                self._reset_window()
            return None

        self._window_steps += 1
        self._window_samples += int(batch_size)
        self._window_data_wait_seconds += float(
            data_wait_seconds
        )

        if global_step % self.log_every_steps != 0:
            return None

        # 每个日志窗口只同步一次，使窗口总时间包含真正的 CUDA 执行。
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed_seconds = max(
            perf_counter() - self._window_start,
            1e-8,
        )

        num_steps = max(self._window_steps, 1)
        step_seconds = elapsed_seconds / num_steps
        data_wait_seconds = (
            self._window_data_wait_seconds / num_steps
        )

        remaining_steps = max(
            self.total_steps - global_step,
            0,
        )

        metrics = {
            "perf/step_ms": 1000.0 * step_seconds,
            "perf/data_wait_ms": (
                1000.0 * data_wait_seconds
            ),
            "perf/data_wait_ratio": (
                self._window_data_wait_seconds
                / elapsed_seconds
            ),
            "perf/samples_per_s": (
                self._window_samples
                / elapsed_seconds
            ),
            "perf/eta_hours": (
                remaining_steps
                * step_seconds
                / 3600.0
            ),
        }

        self._reset_window()
        return metrics

    def _reset_window(self) -> None:
        """重置当前日志窗口的累加器。"""
        self._window_start = perf_counter()
        self._window_steps = 0
        self._window_samples = 0
        self._window_data_wait_seconds = 0.0
