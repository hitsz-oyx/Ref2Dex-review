from __future__ import annotations

import torch

from src.base import MetricStat


def binary_auprc(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores.new_tensor(0.0)
    labels = labels.bool()
    num_pos = int(labels.sum().item())
    if num_pos <= 0:
        return scores.new_tensor(0.0)
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    cumsum_pos = torch.cumsum(sorted_labels, dim=0)
    rank = torch.arange(
        1,
        sorted_labels.numel() + 1,
        device=scores.device,
        dtype=scores.dtype,
    )
    precision = cumsum_pos / rank
    return (precision * sorted_labels).sum() / float(num_pos)


def batched_binary_auprc_stat(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> MetricStat:
    total = 0.0
    count = 0.0
    for batch_idx in range(scores.shape[0]):
        sample_valid = valid_mask[batch_idx].reshape(-1)
        if not bool(sample_valid.any()):
            continue
        sample_scores = scores[batch_idx].reshape(-1)[sample_valid]
        sample_labels = labels[batch_idx].reshape(-1)[sample_valid]
        if int(sample_labels.bool().sum().item()) <= 0:
            continue
        total += float(binary_auprc(sample_scores, sample_labels).detach().cpu())
        count += 1.0
    if count <= 0.0:
        return MetricStat.invalid(expose_validity=True)
    return MetricStat(total=total, count=count, expose_validity=True)
