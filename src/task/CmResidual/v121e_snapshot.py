"""Contracts and metrics for the V1.21e snapshot-restore parity test."""
from __future__ import annotations

from typing import Mapping, Any
import numpy as np

PHASE_QUOTAS = {0: 2, 1: 2, 2: 2}
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
PRODUCER = "v121e_snapshot_parity_physx_producer.v2"


def quaternion_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(left, right)), 0.0, 1.0)))


def _outcome(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if str(record.get("score_producer")) != PRODUCER:
        raise ValueError("physics score did not come from the V1.21e producer")
    if record.get("score_baseline") != "shared_canonical_collected_pre_action":
        raise ValueError("physics score baseline is not the collected canonical state")
    if record.get("num_envs") != 1:
        raise ValueError("each V1.21e.1 producer must own exactly one environment")
    errors = record.get("setter_input_max_abs_error", {})
    if set(errors) != {"dof", "root"} or any(float(value) != 0.0 for value in errors.values()):
        raise ValueError("setter input must exactly equal its recorded source")
    statuses = record.get("setter_return_status", {})
    if set(statuses) != {"dof", "root"} or any(value is False for value in statuses.values()):
        raise ValueError("Isaac Gym setter reported failure")
    roots = np.asarray(record["post_actor_root_state"], dtype=np.float64)
    dof = np.asarray(record["post_dof_state"], dtype=np.float64)
    ig = np.asarray(record["post_ig"], dtype=np.float64)
    score = float(record["physics_score"])
    if roots.ndim != 2 or roots.shape[-1] != 13 or dof.ndim != 2 or dof.shape[-1] != 2 or ig.shape != (18,):
        raise ValueError("single-environment producer outcome has invalid shape")
    if not all(np.isfinite(x).all() for x in (roots, dof, ig)) or not np.isfinite(score):
        raise ValueError("producer outcome contains non-finite values")
    return roots, dof, ig, score


def _arm_metrics(records: tuple[Mapping[str, Any], Mapping[str, Any]]) -> dict[str, float]:
    first, second = (_outcome(record) for record in records)
    roots = np.stack((first[0], second[0]))
    dof = np.stack((first[1], second[1]))
    ig = np.stack((first[2], second[2]))
    score = np.asarray((first[3], second[3]), dtype=np.float64)
    obj = int(records[0]["object_actor_index"])
    if int(records[1]["object_actor_index"]) != obj:
        raise ValueError("duplicate object actor index mismatch")
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


def evaluate_state(
    prefix: tuple[Mapping[str, Any], Mapping[str, Any]] | list[Mapping[str, Any]],
    restore: tuple[Mapping[str, Any], Mapping[str, Any]] | list[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(prefix) != 2 or len(restore) != 2:
        raise ValueError("V1.21e.1 requires A1/A2 and B1/B2")
    records = tuple(prefix) + tuple(restore)
    for name in ("state_id", "candidate_action_sha256", "task_indices_before", "canonical_snapshot_sha256"):
        if any(record[name] != records[0][name] for record in records[1:]):
            raise ValueError(f"four-arm {name} mismatch")
    for pair in (prefix, restore):
        for name in ("setter_input_dof_sha256", "setter_input_root_sha256"):
            if pair[0][name] != pair[1][name]:
                raise ValueError(f"duplicate {name} mismatch")
    a_noise, b_noise = _arm_metrics(tuple(prefix)), _arm_metrics(tuple(restore))
    prefix_values = [_outcome(record) for record in prefix]
    restore_values = [_outcome(record) for record in restore]
    roots_a = np.stack([value[0] for value in prefix_values])
    roots_b = np.stack([value[0] for value in restore_values])
    dof_a = np.stack([value[1] for value in prefix_values])
    dof_b = np.stack([value[1] for value in restore_values])
    ig_a = np.stack([value[2] for value in prefix_values])
    ig_b = np.stack([value[2] for value in restore_values])
    score_a = np.asarray([value[3] for value in prefix_values])
    score_b = np.asarray([value[3] for value in restore_values])
    obj = int(records[0]["object_actor_index"])
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
    return {"state_id": records[0]["state_id"], "prefix_duplicate": a_noise,
            "restore_duplicate": b_noise, "cross_method": cross, "limits": limits,
            "gates": gates, "valid": bool(all(gates.values()))}
