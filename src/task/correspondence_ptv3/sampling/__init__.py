from __future__ import annotations

from typing import Any

from .balanced import BalancedEdgeSampler
from .base import EdgeSampler
from .dense import DenseEdgeSampler
from .stratified import (
    StratifiedEdgeSampler,
    stratified_distance_bucket_candidates,
    validate_stratified_edge_sampler_config,
)
from .utils import (
    AugmentedGeometry,
    augment_geometry,
    sample_object_indices,
    stable_frame_seed,
)


def build_edge_sampler(config: Any) -> EdgeSampler:
    name = str(getattr(config, "name", "")).lower()
    params = dict(getattr(config, "params", {}) or {})
    if name == "balanced":
        return BalancedEdgeSampler(**params)
    if name == "stratified":
        return StratifiedEdgeSampler(**params)
    if name == "dense":
        return DenseEdgeSampler(**params)
    raise ValueError(f"Unsupported edge sampler: {name!r}.")


__all__ = [
    "AugmentedGeometry",
    "BalancedEdgeSampler",
    "DenseEdgeSampler",
    "EdgeSampler",
    "StratifiedEdgeSampler",
    "augment_geometry",
    "build_edge_sampler",
    "sample_object_indices",
    "stable_frame_seed",
    "stratified_distance_bucket_candidates",
    "validate_stratified_edge_sampler_config",
]
