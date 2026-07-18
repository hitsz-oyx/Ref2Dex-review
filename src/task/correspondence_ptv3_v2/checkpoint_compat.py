"""Checkpoint migrations owned by correspondence_ptv3_v2."""
from __future__ import annotations

from typing import Any


def adapt_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Add v2 runner state that did not exist in older v2 checkpoints."""
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
