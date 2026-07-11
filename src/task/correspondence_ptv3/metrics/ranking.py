from __future__ import annotations

import torch

from src.base import MetricStat


def batched_cross_edge_rank_at_k_stat(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    k: int,
) -> MetricStat:
    per_sample: list[torch.Tensor] = []
    for batch_idx in range(pred.shape[0]):
        per_object_hits: list[torch.Tensor] = []
        sample_valid_mask = valid_mask[batch_idx]
        sample_pred = pred[batch_idx]
        sample_target = target[batch_idx]
        valid_objects = torch.nonzero(
            sample_valid_mask.any(dim=-1),
            as_tuple=False,
        ).squeeze(-1)
        for obj_idx in valid_objects.tolist():
            edge_valid = sample_valid_mask[obj_idx]
            valid_idx = torch.nonzero(edge_valid, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            kk = min(int(k), int(valid_idx.numel()))
            gt_scores = sample_target[obj_idx, valid_idx]
            if float(gt_scores.max().item()) <= 0.0:
                continue
            pred_scores = sample_pred[obj_idx, valid_idx]
            gt_topk = torch.topk(gt_scores, k=kk, dim=-1).indices
            pred_topk = torch.topk(pred_scores, k=kk, dim=-1).indices
            hit = (pred_topk.unsqueeze(-1) == gt_topk.unsqueeze(-2)).any().float()
            per_object_hits.append(hit)
        if per_object_hits:
            per_sample.append(torch.stack(per_object_hits).mean())
    if not per_sample:
        return MetricStat.invalid(expose_validity=True)
    total = float(torch.stack(per_sample).sum().detach().cpu())
    return MetricStat(total=total, count=float(len(per_sample)), expose_validity=True)
