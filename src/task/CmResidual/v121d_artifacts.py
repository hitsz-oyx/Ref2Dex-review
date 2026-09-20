"""Versioned artifact validation for the V1.21d fresh-simulator backend."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.task.CmResidual.v121c_artifacts import (
    build_manifest,
    validate_physics_records,
)
from src.task.CmResidual.v121d_fresh_sim import (
    ALL_DUPLICATE_PAIR_COUNT,
    BACKEND_VERSION,
    BRANCH_BACKEND,
)


PHYSICS_SCHEMA = "cmv2_native_action_physics_v121d"
MANIFEST_SCHEMA = "ref2dex.cmresidual.v121d.run.v1"
PARITY_FIELD_COUNT = 5
DIVERGENCE_FIELD_COUNT = 4


def _rows(name: str, value: Any, count: int, shape: tuple[int, ...] = ()) -> np.ndarray:
    array = np.asarray(value)
    expected = (count, *shape)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
    return array


def validate_fresh_sim_physics_records(records: Mapping[str, Any], count: int | None = None) -> int:
    """Reject legacy tensor-restore records and validate V1.21d provenance fields."""
    row_count = validate_physics_records(records, count)
    required = {
        "branch_backend",
        "backend_version",
        "sim_instance_id",
        "native_initial_parity",
        "native_initial_max_abs_error_by_field",
        "pre_candidate_parity",
        "pre_candidate_max_abs_error_by_field",
        "prefix_length",
        "task_indices_exact",
        "sim_create_ok",
        "sim_destroy_ok",
        "duplicate_replica_count",
        "duplicate_pair_count",
        "duplicate_pairwise_divergence",
        "state_max_abs_pairwise_score_divergence",
    }
    missing = required - set(records)
    if missing:
        raise ValueError(f"V1.21d physics records missing fields: {sorted(missing)}")
    backends = _rows("branch_backend", records["branch_backend"], row_count).astype(str)
    versions = _rows("backend_version", records["backend_version"], row_count).astype(str)
    if not np.all(backends == BRANCH_BACKEND) or not np.all(versions == BACKEND_VERSION):
        raise ValueError("legacy or unknown branch backend cannot be V1.21d evidence")
    instance_ids = _rows("sim_instance_id", records["sim_instance_id"], row_count)
    if instance_ids.dtype.kind not in "iu" or len(set(instance_ids.tolist())) != row_count:
        raise ValueError("sim_instance_id must be unique integers per state")
    for name in (
        "native_initial_parity",
        "pre_candidate_parity",
        "task_indices_exact",
        "sim_create_ok",
        "sim_destroy_ok",
    ):
        if _rows(name, records[name], row_count).dtype != np.bool_:
            raise ValueError(f"{name} must be boolean")
    if not np.asarray(records["sim_create_ok"]).all() or not np.asarray(records["sim_destroy_ok"]).all():
        raise ValueError("every persisted V1.21d state must have a complete simulator lifecycle")
    initial_error = _rows(
        "native_initial_max_abs_error_by_field",
        records["native_initial_max_abs_error_by_field"],
        row_count,
        (PARITY_FIELD_COUNT,),
    )
    if not np.isfinite(initial_error).all():
        raise ValueError("native initial errors must be finite")
    pre_error = _rows(
        "pre_candidate_max_abs_error_by_field",
        records["pre_candidate_max_abs_error_by_field"],
        row_count,
        (PARITY_FIELD_COUNT,),
    )
    native_valid = np.asarray(records["native_initial_parity"], dtype=bool)
    if not np.isfinite(pre_error[native_valid]).all() or not np.isnan(pre_error[~native_valid]).all():
        raise ValueError("pre-candidate errors require native parity; unavailable rows must be NaN")
    prefix = _rows("prefix_length", records["prefix_length"], row_count)
    if prefix.dtype.kind not in "iu" or (prefix <= 0).any():
        raise ValueError("prefix_length must be positive integer")
    replica_count = _rows("duplicate_replica_count", records["duplicate_replica_count"], row_count)
    pair_count = _rows("duplicate_pair_count", records["duplicate_pair_count"], row_count)
    if replica_count.dtype.kind not in "iu" or not np.isin(replica_count, (0, 2, 9)).all():
        raise ValueError("duplicate_replica_count must be 0, 2, or 9")
    expected_pairs = np.where(
        replica_count == 9,
        ALL_DUPLICATE_PAIR_COUNT,
        np.where(replica_count == 2, 1, 0),
    )
    if pair_count.dtype.kind not in "iu" or not np.array_equal(pair_count, expected_pairs):
        raise ValueError("duplicate_pair_count does not match replica count")
    divergence = _rows(
        "duplicate_pairwise_divergence",
        records["duplicate_pairwise_divergence"],
        row_count,
        (ALL_DUPLICATE_PAIR_COUNT, DIVERGENCE_FIELD_COUNT),
    )
    for row, pairs in enumerate(pair_count.tolist()):
        if not np.isfinite(divergence[row, :pairs]).all():
            raise ValueError("active duplicate pairwise divergence must be finite")
        if not np.isnan(divergence[row, pairs:]).all():
            raise ValueError("unused duplicate pair slots must be NaN")
    state_max = _rows(
        "state_max_abs_pairwise_score_divergence",
        records["state_max_abs_pairwise_score_divergence"],
        row_count,
    )
    for row, pairs in enumerate(pair_count.tolist()):
        if pairs:
            expected_max = float(np.max(divergence[row, :pairs, 3]))
            if not np.isfinite(state_max[row]) or not np.isclose(
                state_max[row], expected_max, rtol=0, atol=1e-12
            ):
                raise ValueError("state max score divergence does not match pairwise records")
        elif not np.isnan(state_max[row]):
            raise ValueError("state max score divergence must be NaN when candidate was not run")
    pre_valid = np.asarray(records["pre_candidate_parity"], dtype=bool)
    task_indices_exact = np.asarray(records["task_indices_exact"], dtype=bool)
    if (pre_valid & ~native_valid).any() or ((native_valid | pre_valid) & ~task_indices_exact).any():
        raise ValueError("parity success requires its preceding gate and exact task indices")
    if not np.array_equal(replica_count > 0, pre_valid):
        raise ValueError("duplicate outcomes may exist only after pre-candidate parity")
    return row_count


def build_v121d_manifest(**kwargs: Any) -> dict[str, Any]:
    """Build the inherited manifest while making the new backend/schema explicit."""
    manifest = build_manifest(**kwargs)
    manifest["manifest_schema"] = MANIFEST_SCHEMA
    manifest["schema"]["physics_records"] = PHYSICS_SCHEMA
    manifest["protocol"]["branch_backend"] = BRANCH_BACKEND
    manifest["protocol"]["backend_version"] = BACKEND_VERSION
    return manifest
