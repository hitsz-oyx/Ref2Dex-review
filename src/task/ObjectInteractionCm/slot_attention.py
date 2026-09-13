"""Object-side masked Slot Attention for ObjectInteractionCm V1.2.1."""
from __future__ import annotations

import torch
from torch import nn


class SlotAttention(nn.Module):
    """Slot Attention with hard object-pool mask and finite all-empty fallback."""

    def __init__(self, *, dim: int = 128, num_slots: int = 16, num_iterations: int = 3) -> None:
        super().__init__()
        if dim <= 0 or num_slots <= 0 or num_iterations <= 0:
            raise ValueError("dim, num_slots and num_iterations must be positive")
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.slot_init = nn.Parameter(torch.empty(1, self.num_slots, self.dim))
        nn.init.normal_(self.slot_init, std=self.dim ** -0.5)
        self.norm_inputs = nn.LayerNorm(self.dim)
        self.norm_slots = nn.LayerNorm(self.dim)
        self.norm_mlp = nn.LayerNorm(self.dim)
        self.to_key = nn.Linear(self.dim, self.dim, bias=False)
        self.to_value = nn.Linear(self.dim, self.dim, bias=False)
        self.to_query = nn.Linear(self.dim, self.dim, bias=False)
        self.gru = nn.GRUCell(self.dim, self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2), nn.GELU(), nn.Linear(self.dim * 2, self.dim)
        )
        self.scale = self.dim ** -0.5

    def forward(self, inputs: torch.Tensor, input_mask: torch.Tensor | None = None):
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"Expected [B,N,{self.dim}], got {tuple(inputs.shape)}")
        batch_size, num_points, _ = inputs.shape
        if input_mask is None:
            input_mask = torch.ones((batch_size, num_points), dtype=torch.bool, device=inputs.device)
        if input_mask.shape != (batch_size, num_points):
            raise ValueError(f"input_mask must be [B,N], got {tuple(input_mask.shape)}")
        input_mask = input_mask.bool()
        empty_sample = ~input_mask.any(dim=1)
        safe_mask = input_mask.clone()
        safe_mask[empty_sample, 0] = True

        safe_inputs = inputs.clone()
        safe_inputs[empty_sample, 0] = 0.0
        encoded = self.norm_inputs(safe_inputs)
        keys = self.to_key(encoded)
        values = self.to_value(encoded)
        slots = self.slot_init.expand(batch_size, -1, -1)
        assignment = None
        slot_weights = None
        for _ in range(self.num_iterations):
            previous = slots
            queries = self.to_query(self.norm_slots(slots))
            logits = torch.einsum("bsd,bnd->bsn", queries, keys) * self.scale
            assignment = torch.softmax(logits, dim=1)
            assignment = assignment * safe_mask[:, None, :].to(assignment.dtype)
            slot_weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bsn,bnd->bsd", slot_weights, values)
            slots = self.gru(
                updates.reshape(-1, self.dim), previous.reshape(-1, self.dim)
            ).reshape(batch_size, self.num_slots, self.dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        assert assignment is not None and slot_weights is not None
        # The numerical dummy is never exposed as an active object-pool point.
        slot_weights = slot_weights * input_mask[:, None, :].to(slot_weights.dtype)
        assignment = assignment * input_mask[:, None, :].to(assignment.dtype)
        return slots, assignment, slot_weights, empty_sample
