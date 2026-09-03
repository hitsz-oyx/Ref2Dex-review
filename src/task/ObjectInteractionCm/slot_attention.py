"""Object-side Slot Attention for ObjectInteractionCm V1.1."""
from __future__ import annotations

import torch
from torch import nn


class SlotAttention(nn.Module):
    """Deterministic Slot Attention returning object-point assignments."""

    def __init__(self, *, dim: int = 32, num_slots: int = 16, num_iterations: int = 3) -> None:
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
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.scale = self.dim ** -0.5

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``slots``, point-to-slot assignment and slot-normalized weights.

        Args:
            inputs: ``[B,N,D]`` object point tokens.
        Returns:
            slots: ``[B,S,D]``;
            assignment: ``[B,S,N]`` (softmax over slots for each point);
            slot_weights: ``[B,S,N]`` (normalized over points for each slot).
        """
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(f"Expected [B,N,{self.dim}], got {tuple(inputs.shape)}")
        encoded = self.norm_inputs(inputs)
        keys = self.to_key(encoded)
        values = self.to_value(encoded)
        batch_size = inputs.shape[0]
        slots = self.slot_init.expand(batch_size, -1, -1)
        assignment = None
        slot_weights = None
        for _ in range(self.num_iterations):
            previous = slots
            queries = self.to_query(self.norm_slots(slots))
            logits = torch.einsum("bsd,bnd->bsn", queries, keys) * self.scale
            assignment = torch.softmax(logits, dim=1)
            slot_weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bsn,bnd->bsd", slot_weights, values)
            slots = self.gru(
                updates.reshape(-1, self.dim),
                previous.reshape(-1, self.dim),
            ).reshape(batch_size, self.num_slots, self.dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        assert assignment is not None and slot_weights is not None
        return slots, assignment, slot_weights
