from __future__ import annotations

import tempfile
import time
import unittest
from time import perf_counter
from unittest import mock

import torch

from src.base import performance as performance_module
from src.base.performance import PerformanceMonitor


class PerformanceMonitorTests(unittest.TestCase):
    """覆盖 docs/指导.md 列出的 4 个回归点 + 3 个最小测试。

    1. off 模式不返回指标
    2. warmup 后第一个 log 窗口返回 5 个 ``perf/*`` 指标
    3. ``samples_per_s`` 正确乘 ``world_size``
    """

    def _make_monitor(
        self,
        *,
        mode: str = "light",
        world_size: int = 1,
        log_every_steps: int = 4,
        warmup_steps: int = 2,
    ) -> PerformanceMonitor:
        tmp = tempfile.mkdtemp(prefix="perf_test_")
        return PerformanceMonitor(
            mode=mode,
            device=torch.device("cpu"),
            output_dir=tmp,
            total_steps=1_000_000,
            log_every_steps=log_every_steps,
            warmup_steps=warmup_steps,
            world_size=world_size,
            is_primary=True,
        )

    # 1. off 模式：任何时候 observe_step 都返回 None
    def test_off_mode_never_returns_metrics(self) -> None:
        monitor = self._make_monitor(mode="off")
        self.assertFalse(monitor.enabled)
        for step in range(1, 50):
            result = monitor.observe_step(
                global_step=step,
                batch_size=4,
                data_wait_seconds=0.01,
                step_start_seconds=perf_counter() - 0.01,
            )
            self.assertIsNone(
                result,
                f"off mode should not return metrics at step {step}",
            )

    # 2. light 模式：warmup 期间不返回；第一个 log 窗口返回 5 个 perf/* 指标
    def test_light_mode_returns_five_metrics_after_warmup(self) -> None:
        monitor = self._make_monitor(
            mode="light", world_size=1, log_every_steps=4, warmup_steps=2,
        )
        self.assertTrue(monitor.enabled)

        # 2 步 warmup；前 2 步 observe_step 返回 None
        for step in (1, 2):
            result = monitor.observe_step(
                global_step=step,
                batch_size=4,
                data_wait_seconds=0.005,
                step_start_seconds=perf_counter() - 0.005,
            )
            self.assertIsNone(
                result,
                f"warmup step {step} should not return metrics",
            )

        # 窗口前 3 步（step 3）不触发 log；窗口内已经有 step 3 一次非 log 步 + step 4 一次 log 步，
        # 所以当 step 4 返回指标时，窗口内累计 2 步（_window_steps == 2）。
        result = monitor.observe_step(
            global_step=3,
            batch_size=4,
            data_wait_seconds=0.005,
            step_start_seconds=perf_counter() - 0.005,
        )
        self.assertIsNone(result, "non-log step should return None")

        # step=4 是第一个 log 步
        result = monitor.observe_step(
            global_step=4,
            batch_size=4,
            data_wait_seconds=0.005,
            step_start_seconds=perf_counter() - 0.005,
        )
        self.assertIsNotNone(result, "log step should return metrics")
        self.assertEqual(
            set(result.keys()),
            {
                "perf/step_ms",
                "perf/data_wait_ms",
                "perf/data_wait_ratio",
                "perf/samples_per_s",
                "perf/eta_hours",
            },
        )
        # 窗口内 2 步（step 3 + step 4），所以 num_steps=2，data_wait_ms ≈ 5ms
        self.assertGreater(result["perf/step_ms"], 0.0)
        self.assertAlmostEqual(result["perf/data_wait_ms"], 5.0, delta=4.0)
        self.assertGreaterEqual(result["perf/data_wait_ratio"], 0.0)
        self.assertLessEqual(result["perf/data_wait_ratio"], 1.0)
        # 窗口内 8 样本（4 + 4）累加，world_size=1
        self.assertGreater(result["perf/samples_per_s"], 0.0)
        # eta_hours 应当非负
        self.assertGreaterEqual(result["perf/eta_hours"], 0.0)

    # 3. DDP：samples_per_s 必须乘以 world_size
    def test_samples_per_s_scales_with_world_size(self) -> None:
        # 跑两份 monitor，唯一的差异是 world_size
        monitor_ws1 = self._make_monitor(
            mode="light", world_size=1, log_every_steps=2, warmup_steps=1,
        )
        monitor_ws4 = self._make_monitor(
            mode="light", world_size=4, log_every_steps=2, warmup_steps=1,
        )

        # 同样的 step / batch_size / 真实 sleep；只让 world_size 不同
        batch_size = 8
        step_dt = 0.05        # 50ms / step
        data_wait_dt = 0.01   # 10ms / step

        # warmup 步：让两个 monitor 都跳过第 1 步
        result_ws1 = result_ws4 = None
        for step in (1, 2):
            t0 = perf_counter()
            # 真实 sleep，模拟"这个 step 确实耗时 50ms"
            time.sleep(step_dt)
            t_step = t0
            result_ws1 = monitor_ws1.observe_step(
                global_step=step,
                batch_size=batch_size,
                data_wait_seconds=data_wait_dt,
                step_start_seconds=t_step,
            )
            result_ws4 = monitor_ws4.observe_step(
                global_step=step,
                batch_size=batch_size,
                data_wait_seconds=data_wait_dt,
                step_start_seconds=t_step,
            )

        # warmup=1，所以第 2 步触发 log
        self.assertIsNotNone(result_ws1)
        self.assertIsNotNone(result_ws4)

        # 关键断言：world_size=4 的 samples_per_s 应当接近 4 倍于 world_size=1
        # 给一个 20% 的相对误差容忍（处理两次 perf_counter() 调用的开销差）。
        self.assertAlmostEqual(
            result_ws4["perf/samples_per_s"],
            result_ws1["perf/samples_per_s"] * 4.0,
            delta=result_ws1["perf/samples_per_s"] * 4.0 * 0.20 + 1.0,
        )

    # 4. start() 总是重置窗口：避免 _setup_train 末尾的 build_model / load_state_dict
    #    污染第一个窗口（resume 场景）。
    def test_start_resets_window_even_in_light_mode(self) -> None:
        monitor = self._make_monitor(mode="light", log_every_steps=2, warmup_steps=0)
        # 模拟"_setup_train 期间的累加"
        monitor._window_step_seconds += 999.0
        monitor._window_data_wait_seconds += 999.0
        monitor._window_samples += 999
        monitor._window_steps += 999
        # start() 必须把窗口清零
        monitor.start()
        self.assertEqual(monitor._window_step_seconds, 0.0)
        self.assertEqual(monitor._window_data_wait_seconds, 0.0)
        self.assertEqual(monitor._window_samples, 0)
        self.assertEqual(monitor._window_steps, 0)
        monitor.stop()

    # 5. 非法 mode 抛 ValueError
    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._make_monitor(mode="invalid-mode")

    # 6. world_size 必须 ≥ 1（即使传 0 也要归一化为 1，避免除零）
    def test_world_size_normalized_to_at_least_one(self) -> None:
        monitor = self._make_monitor(mode="light", world_size=0)
        self.assertEqual(monitor.world_size, 1)

    # 7. CUDA 计时顺序：窗口最后一步必须先 cuda.synchronize 再算 step_seconds，
    #    这样尾部尚未结算的 GPU 工作（loss.backward / scaler.step / optimizer.step）
    #    才会进入 perf/*；否则既拖慢训练又不在指标里出现。
    def test_cuda_synchronize_runs_before_step_seconds_calc(self) -> None:
        # 用 mock 包装 device 让 device.type == "cuda"，并记录 synchronize / perf_counter
        # 调用顺序；这样不依赖真实 GPU 也能验证调用时序。
        call_log: list[str] = []
        mock_device = mock.MagicMock()
        mock_device.type = "cuda"

        monitor = self._make_monitor(
            mode="light", world_size=1, log_every_steps=1, warmup_steps=0,
        )
        monitor.device = mock_device

        # 替换 synchronize 和 perf_counter，记录调用时间点
        with mock.patch.object(
            torch.cuda, "synchronize", side_effect=lambda *a, **k: call_log.append("cuda.synchronize"),
        ), mock.patch.object(
            performance_module, "perf_counter", side_effect=lambda: call_log.append("perf_counter") or 0.0,
        ):
            monitor.observe_step(
                global_step=1,
                batch_size=4,
                data_wait_seconds=0.001,
                step_start_seconds=0.0,
            )

        # 关键断言：synchronize 必须在所有 perf_counter 之前
        # 找到所有 cuda.synchronize / perf_counter 的位置
        sync_indices = [
            i for i, name in enumerate(call_log) if name == "cuda.synchronize"
        ]
        perf_indices = [
            i for i, name in enumerate(call_log) if name == "perf_counter"
        ]
        self.assertGreater(
            len(sync_indices), 0, "expected at least one cuda.synchronize call"
        )
        self.assertGreater(
            len(perf_indices), 0, "expected at least one perf_counter call"
        )
        # 第一个 sync 必须早于第一个 perf_counter
        self.assertLess(
            sync_indices[0], perf_indices[0],
            f"cuda.synchronize must run before perf_counter; got order: {call_log}",
        )


if __name__ == "__main__":
    unittest.main()
