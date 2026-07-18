"""Legacy checkpoint path compatibility owned by correspondence_ptv3.

The current shared checkpoint layout is handled by ``src.base.checkpoint``.
This module is deliberately task-local because the historical ``checkpoint.pt``
and numeric/epoch directory layouts predate the current correspondence task
format and should not affect Cm or correspondence_ptv3_v2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.base.checkpoint import load_checkpoint, resolve_checkpoint_file


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve current paths first, then correspondence_ptv3 legacy layouts."""
    path = Path(path)
    current = resolve_checkpoint_file(path)
    if current.exists() and current.is_file():
        return current
    if not path.is_dir():
        return path

    direct_legacy = path / "checkpoint.pt"
    if direct_legacy.is_file():
        return direct_legacy

    legacy_last = path / "last"
    if legacy_last.exists():
        return resolve_checkpoint_path(legacy_last)

    candidates = sorted(
        (candidate for candidate in path.iterdir() if _legacy_checkpoint_sort_key(candidate) is not None),
        key=lambda candidate: _legacy_checkpoint_sort_key(candidate),
    )
    if candidates:
        return resolve_checkpoint_path(candidates[-1])
    return path


def load_checkpoint_compat(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return load_checkpoint(resolve_checkpoint_path(path), map_location=map_location)


def adapt_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Upgrade the runner state written before early-stop state was explicit."""
    state = checkpoint.get("runner_state")
    if isinstance(state, dict) and "early_stopping_metric" in state:
        return checkpoint

    upgraded = dict(checkpoint)
    upgraded_state = dict(state or {})
    best_metric = checkpoint.get("best_metric")
    if best_metric is not None:
        upgraded_state["early_stopping_metric"] = float(best_metric)
    upgraded["runner_state"] = upgraded_state
    return upgraded


def build_runner_from_checkpoint(
    checkpoint: str | Path,
    **kwargs: Any,
):
    """Task-local wrapper for tools that need config-driven runner discovery."""
    from src.base import build_runner_from_checkpoint as build_current_runner

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    return build_current_runner(checkpoint_path, **kwargs)


def _legacy_checkpoint_sort_key(path: Path) -> tuple[int, int] | None:
    stem = path.stem if path.is_file() else path.name
    if stem.startswith("epoch_") and stem[len("epoch_") :].isdigit():
        return (1, int(stem[len("epoch_") :]))
    if stem.isdigit():
        return (0, int(stem))
    return None
