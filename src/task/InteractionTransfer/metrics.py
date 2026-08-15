from __future__ import annotations

import torch


@torch.no_grad()
def flow_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    return {"point_error_m": float(error.mean()),
            "translation_error_m": float(torch.linalg.vector_norm(pred.mean(1) - target.mean(1), dim=-1).mean())}
