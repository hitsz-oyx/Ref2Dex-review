"""Reusable components from the rejected hand-prior experiment.

These are intentionally not imported by the production task. They preserve
the tested mechanisms for a future candidate-gated follow-up.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def make_edge_global_residual_head(token_dim: int, hand_context_dim: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(token_dim // 2 + 64 + hand_context_dim, token_dim // 2),
        nn.GELU(),
        nn.Linear(token_dim // 2, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def edge_global_residual(
    head: nn.Module,
    edge_shared: torch.Tensor,
    hand_local_context: torch.Tensor,
    hand_context: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if edge_shared.shape[:-1] != hand_local_context.shape[:-1]:
        raise ValueError("edge_shared and hand_local_context must have the same edge shape.")
    if hand_context.ndim != 2 or hand_context.shape[0] != edge_shared.shape[0]:
        raise ValueError("hand_context must have shape [batch, context_dim].")
    if mode == "zero":
        hand_context = torch.zeros_like(hand_context)
    elif mode == "batch_roll" and hand_context.shape[0] > 1:
        hand_context = hand_context.roll(1, dims=0)
    elif mode != "matched":
        raise ValueError(f"Unknown global-context control: {mode!r}.")
    expanded_context = hand_context.view(
        hand_context.shape[0],
        *([1] * (edge_shared.ndim - 2)),
        hand_context.shape[-1],
    ).expand(*edge_shared.shape[:-1], hand_context.shape[-1])
    return head(torch.cat([edge_shared, hand_local_context, expanded_context], dim=-1)).squeeze(-1)


def load_shared_model_tensors(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    allowed_missing_prefixes: tuple[str, ...] = (
        "hand_global_encoder.",
        "edge_global_residual_head.",
    ),
) -> dict[str, int]:
    """Load model weights without restoring optimizer/scheduler/RNG/step."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {name for name in model.state_dict() if name.startswith(allowed_missing_prefixes)}
    if missing != allowed_missing or unexpected:
        raise RuntimeError(
            f"Incompatible warm start: missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )
    return {"step": int(checkpoint.get("step", 0)), "epoch": int(checkpoint.get("epoch", 0))}
