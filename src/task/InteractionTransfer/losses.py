from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1e-3) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, beta=beta)
