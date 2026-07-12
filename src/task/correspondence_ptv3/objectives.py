from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CrossEdgeRankKObjective:
    k: int
    label_gap: float
    margin: float

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = pred.sum() * 0.0
        if int(self.k) <= 0:
            return zero, pred.new_tensor(0.0)

        num_slots = int(pred.shape[-1])
        topk = min(int(self.k), num_slots)
        if topk <= 0:
            return zero, pred.new_tensor(0.0)

        masked_target = target.masked_fill(~valid_mask, float("-inf"))
        anchor_idx = torch.topk(masked_target, k=topk, dim=-1).indices
        valid_count = valid_mask.sum(dim=-1)
        rank_range = torch.arange(topk, device=pred.device).view(1, 1, topk)
        anchor_valid = rank_range < valid_count.unsqueeze(-1)

        anchor_target = target.gather(dim=-1, index=anchor_idx)
        anchor_pred = pred.gather(dim=-1, index=anchor_idx)
        label_gap = anchor_target.unsqueeze(-1) - target.unsqueeze(-2)
        score_gap = anchor_pred.unsqueeze(-1) - pred.unsqueeze(-2)

        slot_idx = torch.arange(num_slots, device=pred.device).view(1, 1, 1, num_slots)
        pair_valid = (
            anchor_valid.unsqueeze(-1)
            & valid_mask.unsqueeze(-2)
            & (slot_idx != anchor_idx.unsqueeze(-1))
            & (label_gap >= float(self.label_gap))
        )
        pair_weight = label_gap.clamp(min=0.0)
        pair_loss = F.relu(float(self.margin) - score_gap) * pair_weight
        pair_valid_f = pair_valid.float()
        pair_count = pair_valid_f.sum()
        loss = (pair_loss * pair_valid_f).sum() / pair_count.clamp(min=1.0)
        return loss, pair_count


__all__ = ["CrossEdgeRankKObjective"]
