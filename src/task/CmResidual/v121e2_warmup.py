"""Pure contracts and metrics for the V1.21e.2 warm-up replay sweep."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence
import numpy as np

from src.task.CmResidual.v121e_snapshot import (
    FLOORS, POSITION_CEILING_M, ROTATION_CEILING_RAD, quaternion_geodesic,
)

WINDOWS = (0, 1, 2, 4, 8)
METHODS = tuple(str(value) for value in WINDOWS) + ("full",)
PRODUCER = "v121e2_warmup_physx_producer.v1"
STAGE_METRICS = (
    "object_position", "object_rotation", "object_linear_velocity",
    "object_angular_velocity", "dof_q", "dof_dq", "ig",
)


def warmup_action_slice(actions: np.ndarray, frame: int, window: int) -> np.ndarray:
    """Return exactly actions[t-L:t], with explicit protocol bounds."""
    values = np.asarray(actions)
    if window not in WINDOWS or frame < window or frame > len(values):
        raise ValueError("invalid warm-up action slice")
    return values[frame - window:frame]


def canonical_snapshot_sha256(state_id: str, window: int, dof: np.ndarray,
                              roots: np.ndarray, indices: np.ndarray, reset: int,
                              terminate: int, contact: np.ndarray, ig: np.ndarray) -> str:
    digest = hashlib.sha256(f"{state_id}:{window}".encode())
    for value, dtype in ((dof, "<f4"), (roots, "<f4"), (indices, "<i8"),
                         (np.asarray([reset, terminate]), "<i8"),
                         (contact, "<f4"), (ig, "<f4")):
        digest.update(np.ascontiguousarray(np.asarray(value, dtype=dtype)).tobytes())
    return digest.hexdigest()


def _outcome(record: Mapping[str, Any], stage: str):
    if record.get("score_producer") != PRODUCER or record.get("num_envs") != 1:
        raise ValueError("invalid V1.21e.2 producer identity")
    errors = record.get("setter_input_max_abs_error", {})
    if set(errors) != {"dof", "root"} or any(float(value) != 0.0 for value in errors.values()):
        raise ValueError("setter input must exactly equal its frozen source")
    statuses = record.get("setter_return_status", {})
    if set(statuses) != {"dof", "root"} or any(value is False for value in statuses.values()):
        raise ValueError("Isaac Gym setter reported failure")
    roots = np.asarray(record[f"{stage}_actor_root_state"], dtype=np.float64)
    dof = np.asarray(record[f"{stage}_dof_state"], dtype=np.float64)
    ig = np.asarray(record[f"{stage}_ig"], dtype=np.float64)
    score = None if stage == "pre" else float(record["physics_score"])
    if roots.ndim != 2 or roots.shape[-1] != 13 or dof.ndim != 2 or dof.shape[-1] != 2 or ig.shape != (18,):
        raise ValueError(f"invalid {stage} outcome shape")
    values = (roots, dof, ig) if score is None else (roots, dof, ig, np.asarray(score))
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError(f"non-finite {stage} outcome")
    return roots, dof, ig, score


def _centroid(left, right):
    roots = (left[0] + right[0]) * 0.5
    # Actor-root quaternions are xyzw.  Sign-align before normalized averaging.
    for actor in range(roots.shape[0]):
        lq, rq = left[0][actor, 3:7], right[0][actor, 3:7]
        if np.dot(lq, rq) < 0:
            rq = -rq
        mean = lq + rq
        norm = np.linalg.norm(mean)
        roots[actor, 3:7] = mean / norm if norm > 0 else lq
    score = None if left[3] is None else (left[3] + right[3]) * 0.5
    return roots, (left[1] + right[1]) * 0.5, (left[2] + right[2]) * 0.5, score


def _difference(left, right, object_actor_index: int, *, include_score: bool) -> dict[str, float]:
    lroot, rroot = left[0][object_actor_index], right[0][object_actor_index]
    result = {
        "object_position": float(np.linalg.norm(lroot[:3] - rroot[:3])),
        "object_rotation": quaternion_geodesic(lroot[3:7], rroot[3:7]),
        "object_linear_velocity": float(np.max(np.abs(lroot[7:10] - rroot[7:10]))),
        "object_angular_velocity": float(np.max(np.abs(lroot[10:13] - rroot[10:13]))),
        "dof_q": float(np.max(np.abs(left[1][:, 0] - right[1][:, 0]))),
        "dof_dq": float(np.max(np.abs(left[1][:, 1] - right[1][:, 1]))),
        "ig": float(np.max(np.abs(left[2] - right[2]))),
    }
    if include_score:
        result["physics_score"] = float(abs(left[3] - right[3]))
    return result


def _pair(pair: Sequence[Mapping[str, Any]], stage: str):
    if len(pair) != 2:
        raise ValueError("each method requires two independent arms")
    outcomes = (_outcome(pair[0], stage), _outcome(pair[1], stage))
    obj = int(pair[0]["object_actor_index"])
    if int(pair[1]["object_actor_index"]) != obj:
        raise ValueError("duplicate object actor index mismatch")
    return outcomes, _centroid(*outcomes), _difference(*outcomes, obj, include_score=stage == "post")


def evaluate_state(method_records: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if set(method_records) != set(METHODS):
        raise ValueError("warm-up methods must be L=0/1/2/4/8 plus full")
    records = tuple(record for method in METHODS for record in method_records[method])
    identity_fields = ("state_id", "candidate_action_sha256", "task_indices_target", "source_episode_sha256")
    for name in identity_fields:
        if any(record[name] != records[0][name] for record in records[1:]):
            raise ValueError(f"72-arm {name} mismatch")
    for method in METHODS:
        pair = method_records[method]
        if any(str(record["method"]) != method for record in pair):
            raise ValueError(f"{method} arm method identity mismatch")
        for name in ("method", "source_snapshot_sha256"):
            if pair[0][name] != pair[1][name]:
                raise ValueError(f"{method} duplicate {name} mismatch")

    stages = {}
    for method in METHODS:
        stages[method] = {stage: _pair(method_records[method], stage) for stage in ("pre", "post")}
    obj = int(records[0]["object_actor_index"])
    full_valid = all(
        stages["full"][stage][2]["object_position"] <= POSITION_CEILING_M
        and stages["full"][stage][2]["object_rotation"] <= ROTATION_CEILING_RAD
        for stage in ("pre", "post")
    )
    methods = {}
    implementation_valid = full_valid
    for method in (str(value) for value in WINDOWS):
        method_result = {}
        method_duplicate_valid = all(
            stages[method][stage][2]["object_position"] <= POSITION_CEILING_M
            and stages[method][stage][2]["object_rotation"] <= ROTATION_CEILING_RAD
            for stage in ("pre", "post")
        )
        implementation_valid = implementation_valid and method_duplicate_valid
        all_gates = []
        for stage in ("pre", "post"):
            duplicate = stages[method][stage][2]
            full_duplicate = stages["full"][stage][2]
            cross = _difference(stages[method][stage][1], stages["full"][stage][1], obj,
                                include_score=stage == "post")
            limits = {name: max(5.0 * max(duplicate[name], full_duplicate[name]), FLOORS[name])
                      for name in cross}
            gates = {name: cross[name] <= limits[name] for name in cross}
            gates["object_ceiling"] = (
                cross["object_position"] <= POSITION_CEILING_M
                and cross["object_rotation"] <= ROTATION_CEILING_RAD
            )
            all_gates.extend(gates.values())
            method_result[stage] = {
                "duplicate": duplicate, "full_duplicate": full_duplicate,
                "cross_full": cross, "limits": limits, "gates": gates,
            }
        method_result["duplicate_valid"] = method_duplicate_valid
        method_result["pass"] = bool(method_duplicate_valid and all(all_gates))
        methods[method] = method_result
    return {
        "state_id": records[0]["state_id"],
        "full_repeatability_valid": full_valid,
        "implementation_valid": bool(implementation_valid),
        "methods": methods,
    }


def classify_sweep(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(results) != 6:
        raise ValueError("V1.21e.2 requires exactly six states")
    passes = {value: all(result["methods"][str(value)]["pass"] for result in results) for value in WINDOWS}
    if not all(result["implementation_valid"] for result in results):
        return {"conclusion": "INVALID_IMPLEMENTATION", "minimal_supported_L": None,
                "method_pass": passes, "conclusion_scope": "protocol invalid"}
    flags = [passes[value] for value in WINDOWS]
    if not any(flags):
        return {"conclusion": "REFUTED", "minimal_supported_L": None,
                "method_pass": passes, "conclusion_scope": "warmup<=8 insufficient"}
    first = flags.index(True)
    if all(flags[first:]):
        return {"conclusion": "SUPPORTED", "minimal_supported_L": WINDOWS[first],
                "method_pass": passes, "conclusion_scope": "stable tested suffix"}
    return {"conclusion": "INCONCLUSIVE", "minimal_supported_L": None,
            "method_pass": passes, "conclusion_scope": "non-monotonic valid outcomes"}
