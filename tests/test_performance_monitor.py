from __future__ import annotations

import tempfile
import time
import unittest
from time import perf_counter

import torch

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

        # 窗口前 3 步（step 3, 5, 6）也不触发 log，step 4 才触发
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
        # 窗口内 1 步，所以 num_steps=1，data_wait_ms ≈ 5ms
        self.assertGreater(result["perf/step_ms"], 0.0)
        self.assertAlmostEqual(result["perf/data_wait_ms"], 5.0, delta=4.0)
        self.assertGreaterEqual(result["perf/data_wait_ratio"], 0.0)
        self.assertLessEqual(result["perf/data_wait_ratio"], 1.0)
        # 4 样本 / 5ms ≈ 800 samples/s，world_size=1
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


if __name__ == "__main__":
    unittest.main()
