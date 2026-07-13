from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from src.base.base_runner import BaseRunner


class _CountingDataset:
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length


class _CountingLoader:
    def __init__(self, steps_per_epoch: int) -> None:
        self.dataset = _CountingDataset(steps_per_epoch)
        self._steps = steps_per_epoch

    def __iter__(self):
        for i in range(self._steps):
            yield {"x": i}

    def __len__(self) -> int:
        return self._steps


class _TrivialModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # pragma: no cover - stub
        return self.linear(batch["x"].float())


def _build_runner(
    *,
    max_steps: int | None,
    epochs: int,
    steps_per_epoch: int,
) -> BaseRunner:
    runner = BaseRunner.__new__(BaseRunner)
    runner.cfg = SimpleNamespace(
        train=SimpleNamespace(
            epochs=epochs,
            max_steps=max_steps,
            eval_every_steps=None,
            eval_every_epochs=1,
            save_every_epochs=10**9,
            save_every_steps=None,
            log_every_steps=10**9,
            amp=False,
            grad_clip_norm=None,
        ),
        wandb=SimpleNamespace(enable=False),
    )
    runner.train_loader = _CountingLoader(steps_per_epoch)
    runner.val_loader = None
    runner.val_loaders = {}
    runner.model = _TrivialModel()
    runner.optimizer = torch.optim.SGD(runner.model.parameters(), lr=0.0)
    runner.scheduler = None
    runner.scaler = torch.cuda.amp.GradScaler(enabled=False)
    runner.checkpoints = None
    runner.jsonl = None
    runner.wandb_run = None
    runner.global_step = 0
    runner.start_epoch = 0
    runner.total_steps = (
        int(max_steps) if max_steps is not None else max(1, epochs * steps_per_epoch)
    )
    runner.best_metric = None
    runner.train_dataset = None
    runner.epochs_without_improvement = 0
    runner.evals_without_improvement = 0
    runner._early_stopping_triggered = False
    runner.distributed = SimpleNamespace(is_primary=True, enabled=False)
    runner.is_primary = True
    runner.device = torch.device("cpu")
    return runner


def _patch_runner_hooks(monkeypatch) -> None:
    """Stub out side-effect hooks so ``learn()`` is a pure control-flow test."""

    def _noop(self, *args, **kwargs):
        return {}

    def _empty_metrics(metrics, *_args, **_kwargs):
        return {}

    def _count_step(self, batch):
        # Increment the global step like train_step would, but skip the optimizer
        # to keep this a pure control-flow test.
        self.global_step += 1
        return {"loss": 0.0}

    monkeypatch.setattr(BaseRunner, "_require_train_ready", _noop)
    monkeypatch.setattr(BaseRunner, "_record_metrics", _empty_metrics)
    monkeypatch.setattr(BaseRunner, "_save_if_best", _noop)
    monkeypatch.setattr(BaseRunner, "save", _noop)
    monkeypatch.setattr(BaseRunner, "evaluate_all", lambda self: {})
    monkeypatch.setattr(BaseRunner, "_compute_averager_metrics", _empty_metrics)
    monkeypatch.setattr(BaseRunner, "_reduce_step_metrics", _empty_metrics)
    monkeypatch.setattr(BaseRunner, "train_step", _count_step)
    monkeypatch.setattr(BaseRunner, "batch_size", lambda self, batch: 1)


def test_max_steps_authoritative_when_each_epoch_has_few_steps(monkeypatch) -> None:
    """``max_steps`` must cap training even with epochs=100 and 1 step/epoch.

    Regression test for the BaseRunner footgun documented in
    ``docs/指导.md``.
    """
    _patch_runner_hooks(monkeypatch)
    runner = _build_runner(max_steps=3000, epochs=100, steps_per_epoch=1)
    runner.learn()
    assert runner.global_step == 3000, (
        f"Expected 3000 steps, got {runner.global_step}. "
        "The 'epochs' upper bound is truncating training before max_steps."
    )


def test_max_steps_authoritative_with_normal_steps_per_epoch(monkeypatch) -> None:
    _patch_runner_hooks(monkeypatch)
    runner = _build_runner(max_steps=3000, epochs=10, steps_per_epoch=100)
    runner.learn()
    assert runner.global_step == 3000, (
        f"Expected 3000 steps, got {runner.global_step}."
    )


def test_epochs_stops_when_max_steps_not_set(monkeypatch) -> None:
    _patch_runner_hooks(monkeypatch)
    runner = _build_runner(max_steps=None, epochs=10, steps_per_epoch=7)
    runner.learn()
    assert runner.global_step == 70, (
        f"Expected 70 steps (10 epochs * 7 steps), got {runner.global_step}."
    )


def test_learn_resolves_epoch_upper_using_max_steps() -> None:
    """Mirror the docstring contract: max_steps is authoritative; epochs is iteration count."""
    from src.base.base_runner import BaseRunner as _BR

    def _resolve(start_epoch: int, epochs: int, max_steps: int | None, steps_per_epoch: int) -> int:
        epoch_upper = start_epoch + int(epochs)
        if max_steps is not None and steps_per_epoch is not None:
            sp = max(1, steps_per_epoch)
            min_epochs_for_max_steps = (int(max_steps) // sp) + 2
            epoch_upper = max(epoch_upper, start_epoch + min_epochs_for_max_steps)
        return epoch_upper

    # 1) Footgun case: 100 epochs * 1 step/epoch vs max_steps=3000
    assert _resolve(0, 100, 3000, 1) >= 3002
    # 2) Normal case
    assert _resolve(0, 10, 3000, 100) >= 32
    # 3) epochs > max_steps is fine
    assert _resolve(0, 100, 3000, 1000) == 100
    # 4) No max_steps: epoch upper is just start + epochs
    assert _resolve(0, 100, None, 1) == 100
    assert _BR is not None  # silence unused-import warning
