from __future__ import annotations

from typing import Protocol

import torch

from src.task.correspondence_ptv3.contracts import ContactLossResult


class ContactSupervision(Protocol):
    name: str

    @property
    def output_dim(self) -> int:
        ...

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        ...

    def export_compat(
        self,
        logits: torch.Tensor,
        probability: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def legacy_output_aliases(
        self,
        *,
        pred_obj_contact_logits: torch.Tensor,
        pred_cross_contact_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ...

    def compute_object_loss(
        self,
        *,
        logits: torch.Tensor,
        target_probability: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> ContactLossResult:
        ...

    def compute_edge_loss(
        self,
        *,
        logits: torch.Tensor,
        target_probability: torch.Tensor,
        edge_weight: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> ContactLossResult:
        ...
