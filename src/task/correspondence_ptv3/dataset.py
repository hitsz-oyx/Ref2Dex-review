"""Legacy import compatibility for correspondence data APIs.

New code should import from:
    src.task.correspondence_ptv3.data

Do not add new dataset or sampling implementations here.
"""

from __future__ import annotations

# DEPRECATED: compatibility only.

from src.task.correspondence_ptv3.data import (
    CorrStaticDataset,
    SequenceLocalitySampler,
    Stage3Store,
    _compute_dense_logit_neighbors,
    _compute_input_knn,
    _compute_runtime_context_neighbors,
    _compute_runtime_logit_neighbors,
    _compute_stratified_logit_neighbors,
    make_dataloaders,
    sequence_group_key,
)

__all__ = [
    "CorrStaticDataset",
    "SequenceLocalitySampler",
    "Stage3Store",
    "_compute_dense_logit_neighbors",
    "_compute_input_knn",
    "_compute_runtime_context_neighbors",
    "_compute_runtime_logit_neighbors",
    "_compute_stratified_logit_neighbors",
    "make_dataloaders",
    "sequence_group_key",
]
