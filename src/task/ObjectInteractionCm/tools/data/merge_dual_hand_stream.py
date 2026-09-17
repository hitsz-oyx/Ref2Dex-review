"""Utilities for the V1.4 side-free bilateral hand stream contract.

The returned stream is always ``left followed by right``.  KNN indices must be
computed after this merge; this module deliberately does not try to offset or
concatenate per-side KNN results because cross-side neighbours can change.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def merge_bilateral_hand_arrays(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str = "hand",
    dtype: Any = np.float32,
) -> np.ndarray:
    """Concatenate two ``[T,H,3]`` streams with strict shape/finite checks."""

    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != 3 or right.ndim != 3 or left.shape[-1] != 3 or right.shape[-1] != 3:
        raise ValueError(f"{name}: expected left/right [T,H,3], got {left.shape} and {right.shape}")
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"{name}: left/right frame counts differ: {left.shape[0]} != {right.shape[0]}")
    merged = np.concatenate((left, right), axis=1).astype(dtype, copy=False)
    if not np.isfinite(merged).all():
        raise ValueError(f"{name}: merged stream contains non-finite values")
    return np.ascontiguousarray(merged)


def merge_bilateral_hand_stream(
    left_points: np.ndarray,
    right_points: np.ndarray,
    left_normals: np.ndarray,
    right_normals: np.ndarray,
    left_future: np.ndarray,
    right_future: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the canonical side-free bilateral point/normal/flow inputs."""

    points = merge_bilateral_hand_arrays(left_points, right_points, name="hand_points")
    normals = merge_bilateral_hand_arrays(left_normals, right_normals, name="hand_normals")
    future = merge_bilateral_hand_arrays(left_future, right_future, name="hand_future")
    if normals.shape != points.shape or future.shape != points.shape:
        raise ValueError("Bilateral hand points, normals and future shapes do not match")
    return {
        "hand_points": points,
        "hand_normals": normals,
        "hand_future": future,
        "hand_flow": np.ascontiguousarray(future - points),
    }


__all__ = ["merge_bilateral_hand_arrays", "merge_bilateral_hand_stream"]
