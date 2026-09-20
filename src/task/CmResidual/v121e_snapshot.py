"""Contracts and metrics for the V1.21e snapshot-restore parity test."""
from __future__ import annotations

from typing import Mapping, Any
import numpy as np

PHASE_QUOTAS = {0: 6, 1: 6, 2: 4}
POSITION_CEILING_M = 5e-4
ROTATION_CEILING_RAD = 5e-3
FLOORS = {
    "object_position": 1e-5,
    "object_rotation": 1e-4,
    "object_linear_velocity": 1e-3,
    "object_angular_velocity": 1e-3,
    "dof_q": 1e-4,
    "dof_dq": 1e-2,
    "ig": 1e-4,
    "physics_score": 1e-6,
}
PRODUCER = "v121e_snapshot_parity_physx_producer.v1"


def quaternion_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))


def _arm_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    if str(record.get("score_producer")) != PRODUCER:
        raise ValueError("physics score did not come from the V1.21e producer")
    if record.get("score_baseline") != "shared_canonical_collected_pre_action":
        raise ValueError("physics score baseline is not the collected canonical state")
    roots = np.asarray(record["post_actor_root_state"], dtype=np.float64)
    dof = np.asarray(record["post_dof_state"], dtype=np.float64)
    ig = np.asarray(record["post_ig"], dtype=np.float64)
    score = np.asarray(record["physics_score"], dtype=np.float64)
    if roots.shape[0] != 2 or dof.shape[0] != 2 or ig.shape != (2, 18) or score.shape != (2,):
        raise ValueError("each arm must contain exactly two duplicate outcomes")
    if not all(np.isfinite(x).all() for x in (roots, dof, ig, score)):
        raise ValueError("producer outcome contains non-finite values")
    obj = int(record["object_actor_index"])
    a, b = roots[:, obj]
    return {
        "object_position": float(np.linalg.norm(a[:3] - b[:3])),
        "object_rotation": quaternion_geodesic(a[3:7], b[3:7]),
        "object_linear_velocity": float(np.max(np.abs(a[7:10] - b[7:10]))),
        "object_angular_velocity": float(np.max(np.abs(a[10:13] - b[10:13]))),
        "dof_q": float(np.max(np.abs(dof[0, :, 0] - dof[1, :, 0]))),
        "dof_dq": float(np.max(np.abs(dof[0, :, 1] - dof[1, :, 1]))),
        "ig": float(np.max(np.abs(ig[0] - ig[1]))),
        "physics_score": float(abs(score[0] - score[1])),
    }


def evaluate_state(prefix: Mapping[str, Any], restore: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("state_id", "candidate_action_sha256", "task_indices_before"):
        if prefix[name] != restore[name]:
            raise ValueError(f"prefix/restore {name} mismatch")
    a_noise, b_noise = _arm_metrics(prefix), _arm_metrics(restore)
    roots_a = np.asarray(prefix["post_actor_root_state"], dtype=np.float64)
    roots_b = np.asarray(restore["post_actor_root_state"], dtype=np.float64)
    dof_a = np.asarray(prefix["post_dof_state"], dtype=np.float64)
    dof_b = np.asarray(restore["post_dof_state"], dtype=np.float64)
    ig_a = np.asarray(prefix["post_ig"], dtype=np.float64)
    ig_b = np.asarray(restore["post_ig"], dtype=np.float64)
    score_a = np.asarray(prefix["physics_score"], dtype=np.float64)
    score_b = np.asarray(restore["physics_score"], dtype=np.float64)
    obj = int(prefix["object_actor_index"])
    root_a, root_b = roots_a[:, obj].mean(0), roots_b[:, obj].mean(0)
    cross = {
        "object_position": float(np.linalg.norm(root_a[:3] - root_b[:3])),
        "object_rotation": quaternion_geodesic(root_a[3:7], root_b[3:7]),
        "object_linear_velocity": float(np.max(np.abs(root_a[7:10] - root_b[7:10]))),
        "object_angular_velocity": float(np.max(np.abs(root_a[10:13] - root_b[10:13]))),
        "dof_q": float(np.max(np.abs(dof_a[:, :, 0].mean(0) - dof_b[:, :, 0].mean(0)))),
        "dof_dq": float(np.max(np.abs(dof_a[:, :, 1].mean(0) - dof_b[:, :, 1].mean(0)))),
        "ig": float(np.max(np.abs(ig_a.mean(0) - ig_b.mean(0)))),
        "physics_score": float(abs(score_a.mean() - score_b.mean())),
    }
    limits = {name: max(5.0 * max(a_noise[name], b_noise[name]), FLOORS[name]) for name in FLOORS}
    gates = {name: cross[name] <= limits[name] for name in FLOORS}
    gates.update({
        "prefix_duplicate_object_ceiling": a_noise["object_position"] <= POSITION_CEILING_M and a_noise["object_rotation"] <= ROTATION_CEILING_RAD,
        "restore_duplicate_object_ceiling": b_noise["object_position"] <= POSITION_CEILING_M and b_noise["object_rotation"] <= ROTATION_CEILING_RAD,
        "cross_object_ceiling": cross["object_position"] <= POSITION_CEILING_M and cross["object_rotation"] <= ROTATION_CEILING_RAD,
    })
    return {"state_id": prefix["state_id"], "prefix_duplicate": a_noise,
            "restore_duplicate": b_noise, "cross_method": cross, "limits": limits,
            "gates": gates, "valid": bool(all(gates.values()))}
