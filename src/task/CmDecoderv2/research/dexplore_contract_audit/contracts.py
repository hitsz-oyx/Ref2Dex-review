"""Pure checks for the native state versus decoder state/action contract."""
from __future__ import annotations

import numpy as np

from ...kinematics import (
    INDEPENDENT_FINGER_NATIVE_INDICES,
    InspireKinematics,
    expand_finger_q,
)

MIMIC_TARGET = np.array([7, 9, 11, 13, 16, 17])
MIMIC_SOURCE = np.array([6, 8, 10, 12, 15, 15])
MIMIC_SCALE = np.array([1.05, 1.05, 1.05, 1.05, 0.6, 0.8])
UNMODIFIED_COLUMNS = np.r_[0:198, 205:373, 391:598]


def native_array(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 18 or len(q) == 0:
        raise ValueError(f"Expected nonempty native q [T,18], got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("Native q contains nonfinite values")
    return q


def mimic_residual(q: np.ndarray) -> np.ndarray:
    q = native_array(q)
    return q[:, MIMIC_TARGET] - q[:, MIMIC_SOURCE] * MIMIC_SCALE


def reconstruct_native(q: np.ndarray, kinematics: InspireKinematics) -> np.ndarray:
    """Exactly reproduce existing decoder state extraction, clamp and expansion."""
    q = native_array(q)
    finger = kinematics.clamp_finger_q(q[:, INDEPENDENT_FINGER_NATIVE_INDICES])
    out = np.stack([expand_finger_q(row) for row in finger])
    out[:, :6] = q[:, :6]
    return out


def joint_least_squares(q: np.ndarray, kinematics: InspireKinematics) -> np.ndarray:
    """Joint-space optimum on the bounded linear coupling manifold, not a FK optimum."""
    q = native_array(q)
    basis = np.stack([expand_finger_q(row)[6:] for row in np.eye(6)], axis=1)
    finger = (q[:, 6:] @ basis) / np.square(basis).sum(axis=0)
    finger = kinematics.clamp_finger_q(finger)
    out = np.stack([expand_finger_q(row) for row in finger])
    out[:, :6] = q[:, :6]
    return out


def sampled_frames(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("Frame length/count must be positive")
    return np.unique(np.linspace(0, length - 1, min(length, count), dtype=np.int64))


def validate_pair(geometric: np.ndarray, actual: np.ndarray) -> None:
    for value in (geometric, actual):
        if value.ndim != 2 or value.shape[1] != 598 or len(value) < 5:
            raise ValueError(f"Expected native tensor [T>=5,598], got {value.shape}")
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Native tensor must be finite float32")
    if geometric.shape != actual.shape:
        raise ValueError("Paired tensors have different frame counts")
    # Bytewise comparison also detects a sign-bit change in an unchanged zero.
    if geometric[:, UNMODIFIED_COLUMNS].tobytes() != actual[:, UNMODIFIED_COLUMNS].tobytes():
        raise ValueError("Columns outside object pose and q changed")
    for value in (geometric, actual):
        if not np.allclose(np.linalg.norm(value[:, 201:205], axis=-1), 1.0, atol=1e-4):
            raise ValueError("Object quaternion is not unit length")


def existing_splits(index: dict) -> dict[str, str]:
    result = {}
    for split in ("train", "val", "test"):
        for entry in index["sequences"][split]:
            name = entry["parent_seq_id"].replace("/", "_", 1)
            if name in result:
                raise ValueError(f"Duplicate parent identity: {name}")
            result[name] = split
    return result


def statistics(value: np.ndarray) -> dict:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(value) or not np.isfinite(value).all():
        raise ValueError("Statistics require finite nonempty data")
    return {"count": len(value), "mean": float(value.mean()),
            "p50": float(np.median(value)), "p95": float(np.quantile(value, .95)),
            "max": float(value.max())}
