from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.base.base_runner import BaseRunner, CosineRestartScheduler


def _optimizer(lr: float = 5e-5) -> torch.optim.Optimizer:
    return torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=lr)


def test_cosine_restart_scheduler_starts_at_base_and_reaches_min_lr() -> None:
    optimizer = _optimizer()
    scheduler = CosineRestartScheduler(optimizer, total_steps=4, min_lr=5e-6)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6 + (5e-5 - 5e-6) * 0.5 * (1.0 + 2**0.5 / 2.0))
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.75e-5)
    scheduler.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)


def test_cosine_restart_scheduler_state_resumes_without_restarting() -> None:
    optimizer = _optimizer()
    scheduler = CosineRestartScheduler(optimizer, total_steps=10, min_lr=5e-6)
    for _ in range(3):
        scheduler.step()
    state = scheduler.state_dict()
    expected_lr = optimizer.param_groups[0]["lr"]

    resumed_optimizer = _optimizer()
    resumed = CosineRestartScheduler(resumed_optimizer, total_steps=10, min_lr=5e-6)
    assert resumed.is_compatible_state_dict(state)
    resumed.load_state_dict(state)

    assert resumed.completed_steps == 3
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    resumed.step()
    scheduler.step()
    assert resumed_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_cosine_restart_resume_from_old_scheduler_keeps_optimizer_state_but_resets_lr() -> None:
    model = torch.nn.Linear(1, 1)
    source_optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    source_optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 1)).sum().backward()
    source_optimizer.step()

    runner = BaseRunner.__new__(BaseRunner)
    runner.cfg = SimpleNamespace(train=SimpleNamespace(lr=5e-5))
    runner.model = torch.nn.Linear(1, 1)
    runner.optimizer = torch.optim.AdamW(runner.model.parameters(), lr=5e-5)
    runner.scheduler = CosineRestartScheduler(runner.optimizer, total_steps=107000, min_lr=5e-6)
    runner.scaler = None
    runner.is_primary = False
    runner.total_steps = 154670
    runner.epochs_without_improvement = 0
    runner.evals_without_improvement = 0
    runner._early_stopping_triggered = False

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "scheduler": {"last_epoch": 45400, "base_lrs": [1.5e-4]},
        "step": 47670,
        "epoch": 21,
        "best_metric": 0.1,
        "runner_state": {},
    }
    runner._load_checkpoint_payload(checkpoint, load_optimizer=True)

    assert runner.global_step == 47670
    assert runner.start_epoch == 21
    assert runner.scheduler.completed_steps == 0
    assert runner.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
    # Adam state was still restored even though the old scheduler was discarded.
    assert runner.optimizer.state_dict()["state"]


def test_cosine_restart_from_old_scheduler_requires_exact_absolute_stop_step() -> None:
    model = torch.nn.Linear(1, 1)
    runner = BaseRunner.__new__(BaseRunner)
    runner.cfg = SimpleNamespace(train=SimpleNamespace(lr=5e-5))
    runner.model = torch.nn.Linear(1, 1)
    runner.optimizer = torch.optim.AdamW(runner.model.parameters(), lr=5e-5)
    runner.scheduler = CosineRestartScheduler(runner.optimizer, total_steps=107000, min_lr=5e-6)
    runner.scaler = None
    runner.is_primary = False
    runner.total_steps = 155000
    runner.epochs_without_improvement = 0
    runner.evals_without_improvement = 0
    runner._early_stopping_triggered = False

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1.5e-4).state_dict(),
        "scheduler": {"last_epoch": 45400, "base_lrs": [1.5e-4]},
        "step": 47670,
        "epoch": 21,
        "best_metric": 0.1,
        "runner_state": {},
    }
    with pytest.raises(ValueError, match=r"must equal checkpoint_step \+ train.finetune_steps"):
        runner._load_checkpoint_payload(checkpoint, load_optimizer=True)


def test_cosine_restart_requires_explicit_no_warmup() -> None:
    runner = BaseRunner.__new__(BaseRunner)
    runner.cfg = SimpleNamespace(
        train=SimpleNamespace(
            scheduler="cosine_restart",
            warmup_ratio=0.03,
            warmup_steps=0,
        )
    )
    with pytest.raises(ValueError, match="has no warmup"):
        runner._resolve_warmup_steps(154670)
