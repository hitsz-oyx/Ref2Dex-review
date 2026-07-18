from __future__ import annotations

import math
from typing import Any

import torch


def cosine_schedule(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_schedule(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)


class CosineRestartScheduler:
    """Checkpointable cosine fine-tuning phase independent of global step.

    ``BaseRunner.global_step`` stays an absolute run counter. This scheduler
    owns a phase-local counter, so a checkpoint from a long pre-training
    schedule can begin a short fine-tune at the configured base LR and later
    resume that same phase without restarting its decay.
    """

    STATE_TYPE = "cosine_restart"
    STATE_VERSION = 1

    def __init__(self, optimizer: torch.optim.Optimizer, *, total_steps: int, min_lr: float) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.min_lrs = [float(min_lr) for _ in optimizer.param_groups]
        if any(min_lr > base_lr for base_lr in self.base_lrs):
            raise ValueError(f"min_lr={min_lr:g} exceeds an optimizer base LR {self.base_lrs}.")
        self.completed_steps = 0
        self.restart()

    def restart(self) -> None:
        self.completed_steps = 0
        self._apply_lr()

    @property
    def progress(self) -> float:
        return min(1.0, max(0.0, float(self.completed_steps) / float(self.total_steps)))

    def _apply_lr(self) -> None:
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * self.progress))
        for group, base_lr, min_lr in zip(self.optimizer.param_groups, self.base_lrs, self.min_lrs):
            group["lr"] = min_lr + (base_lr - min_lr) * cosine_factor

    def step(self) -> None:
        self.completed_steps = min(self.total_steps, self.completed_steps + 1)
        self._apply_lr()

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_type": self.STATE_TYPE,
            "scheduler_version": self.STATE_VERSION,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "base_lrs": list(self.base_lrs),
            "min_lrs": list(self.min_lrs),
        }

    def is_compatible_state_dict(self, state_dict: Any) -> bool:
        if not isinstance(state_dict, dict) or state_dict.get("scheduler_type") != self.STATE_TYPE:
            return False
        if int(state_dict.get("scheduler_version", -1)) != self.STATE_VERSION:
            return False
        if int(state_dict.get("total_steps", -1)) != self.total_steps:
            return False
        state_base_lrs = [float(value) for value in state_dict.get("base_lrs", [])]
        state_min_lrs = [float(value) for value in state_dict.get("min_lrs", [])]
        if len(state_base_lrs) != len(self.base_lrs) or len(state_min_lrs) != len(self.min_lrs):
            return False
        return all(
            math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-15)
            for old, new in zip(state_base_lrs, self.base_lrs)
        ) and all(
            math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-15)
            for old, new in zip(state_min_lrs, self.min_lrs)
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not self.is_compatible_state_dict(state_dict):
            raise ValueError("Incompatible cosine_restart scheduler state.")
        self.completed_steps = min(self.total_steps, max(0, int(state_dict["completed_steps"])))
        self._apply_lr()
