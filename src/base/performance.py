"""Lightweight performance monitoring and optional profiler support.

Base only observes generic training phases and does not depend on
task-specific data semantics.
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

    窗口统计采用**逐步累加**策略：调用方每步传入 ``step_start_seconds``，
    本类只累加 ``step_seconds`` 与 ``data_wait_seconds``，避免把验证、checkpoint
    保存、W&B 写日志等"非训练"时长混入吞吐估计。
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
        world_size: int = 1,
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
        self.world_size = max(1, int(world_size))
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
        """进入训练循环前调用。

        - **总是**重置轻量统计窗口，避免 ``_setup_train`` 末尾的
          ``build_model`` / ``load_state_dict``（resume 场景）污染第一个窗口；
        - 仅 profile 模式才真正 ``profiler.start()``。
        """
        self._reset_window()

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
        step_start_seconds: float,
    ) -> dict[str, float] | None:
        """累加当前 step 并在窗口末尾返回 ``perf/*`` 指标。

        - **总是**推进 ``profiler.step()``（如果启用）；
        - 在 ``warmup_steps`` 之前不计入窗口；
        - 每 ``log_every_steps`` 返回一次 ``perf/*`` 指标 dict。

        窗口耗时由调用方传入的 ``step_start_seconds`` 累加得到，**不**依赖墙上
        时间差，所以验证、checkpoint 保存、W&B 写日志等不会污染吞吐 / ETA。
        ``samples_per_s`` 已经乘以 ``world_size``，输出全局训练吞吐量。
        """
        if self.profiler is not None:
            self.profiler.step()

        if not self.enabled:
            return None

        global_step = int(global_step)
        step_start_seconds = float(step_start_seconds)
        data_wait_seconds = float(data_wait_seconds)

        # CUDA 初始化、worker 启动和文件冷缓存不用于稳定吞吐估计。
        if global_step <= self.warmup_steps:
            if global_step == self.warmup_steps:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                # warmup 结束，重置窗口让吞吐 / ETA 从此刻起算
                self._reset_window()
            return None

        step_seconds = max(perf_counter() - step_start_seconds, 1e-8)
        self._window_steps += 1
        self._window_samples += int(batch_size)
        self._window_data_wait_seconds += data_wait_seconds
        self._window_step_seconds += step_seconds

        should_log = global_step % self.log_every_steps == 0
        if not should_log:
            return None

        # 每个日志窗口只同步一次，使窗口内累计的 step_seconds 包含真正完成的 CUDA 执行。
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        elapsed_seconds = max(self._window_step_seconds, 1e-8)
        num_steps = max(self._window_steps, 1)
        remaining_steps = max(self.total_steps - global_step, 0)

        metrics = {
            "perf/step_ms": (
                1000.0 * elapsed_seconds / num_steps
            ),
            "perf/data_wait_ms": (
                1000.0
                * self._window_data_wait_seconds
                / num_steps
            ),
            "perf/data_wait_ratio": (
                self._window_data_wait_seconds
                / elapsed_seconds
            ),
            "perf/samples_per_s": (
                self._window_samples
                * self.world_size
                / elapsed_seconds
            ),
            "perf/eta_hours": (
                remaining_steps
                * elapsed_seconds
                / num_steps
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
        self._window_step_seconds = 0.0
