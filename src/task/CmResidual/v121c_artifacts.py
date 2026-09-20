"""Replay-record and run-manifest helpers for the V1.21c ranking gate.

This file only validates/serializes Task-local artifacts.  It does not resolve
simulator paths or silently copy data into the Task tree; a caller must pass the
resolved external paths explicitly in the manifest.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


STATE_SCHEMA = "cmv2_native_action_state_v121c"
PHYSICS_SCHEMA = "cmv2_native_action_physics_v121c"
EPISODE_SCHEMA = "cmv2_native_action_episode_v121c"
MANIFEST_SCHEMA = "ref2dex.cmresidual.v121c.run.v1"

_STATE_SHAPES = {
    "raw_obs": (1442,),
    "native_q": (18,),
    "native_dq": (18,),
    "object_pose": (7,),
    "object_twist": (6,),
    "contact_force": (5, 3),
    "policy_mu": (18,),
    "policy_sigma": (18,),
    "candidate_actions": (8, 18),
    "candidate_valid": (8,),
    "cm_delta_xi": (8, 6),
    "cm_valid": (8,),
    "cm_token_mask": (8, 16),
    "cm_token_mass": (8, 16),
    "cm_score": (8,),
}
_PHYSICS_SHAPES = {
    "env_candidate_ids": (9,),
    "physics_candidate_valid": (8,),
    "physics_score": (8,),
    "physics_next_object_pose": (8, 7),
    "physics_next_IG": (8, 18),
}


def sha256_file(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_array(name: str, value: Any, shape: tuple[int, ...], count: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    expected = shape if count is None else (count, *shape)
    if tuple(array.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(array.shape)}")
    return array


def validate_state_records(records: Mapping[str, Any]) -> int:
    """Validate a batched ``state_records`` mapping and return its row count."""
    missing = set(_STATE_SHAPES) - set(records)
    required_metadata = {"state_id", "collection_batch_id", "episode_id", "seed", "frame_id",
                         "reference_index", "checkpoint_sha256", "motion_source_sha",
                         "obs_rms_snapshot_sha256", "active_reason_mask", "phase_id",
                         "candidate_seed"}
    missing |= required_metadata - set(records)
    if missing:
        raise ValueError(f"state_records missing fields: {sorted(missing)}")
    state_ids = np.asarray(records["state_id"]).reshape(-1)
    if state_ids.size == 0 or len(set(map(str, state_ids.tolist()))) != state_ids.size:
        raise ValueError("state_id must be non-empty and unique")
    count = int(state_ids.size)
    for name, shape in _STATE_SHAPES.items():
        array = _as_array(name, records[name], shape, count)
        if name not in {"candidate_valid", "cm_valid", "cm_token_mask", "candidate_actions",
                        "cm_delta_xi", "cm_token_mass", "cm_score"} and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("candidate_valid", "cm_valid", "cm_token_mask"):
        array = np.asarray(records[name])
        if array.dtype != np.bool_:
            raise ValueError(f"{name} must be boolean")
    candidate_valid = np.asarray(records["candidate_valid"], dtype=bool)
    cm_valid = np.asarray(records["cm_valid"], dtype=bool)
    for name, validity in (("candidate_actions", candidate_valid), ("cm_delta_xi", cm_valid),
                           ("cm_token_mass", cm_valid), ("cm_score", cm_valid)):
        array = np.asarray(records[name])
        if not np.isfinite(array[validity]).all():
            raise ValueError(f"finite {name} required for valid candidates")
    for name in ("episode_id", "seed", "frame_id", "reference_index", "active_reason_mask",
                 "phase_id", "candidate_seed"):
        _as_array(name, records[name], (), count)
    for name in ("collection_batch_id", "checkpoint_sha256", "motion_source_sha",
                 "obs_rms_snapshot_sha256"):
        values = np.asarray(records[name]).reshape(-1)
        if values.size != count or any(not str(value) for value in values.tolist()):
            raise ValueError(f"{name} must contain one non-empty value per state")
    return count


def validate_physics_records(records: Mapping[str, Any], count: int | None = None) -> int:
    """Validate a batched ``physics_records`` mapping and return its row count."""
    missing = set(_PHYSICS_SHAPES) - set(records)
    missing |= ({"physics_clone_valid", "duplicate_delta_object_pose", "duplicate_delta_IG",
                 "duplicate_delta_score"} - set(records))
    if missing:
        raise ValueError(f"physics_records missing fields: {sorted(missing)}")
    clone_valid = np.asarray(records["physics_clone_valid"]).reshape(-1)
    row_count = int(clone_valid.size)
    if count is not None and row_count != count:
        raise ValueError("state/physics record count mismatch")
    if clone_valid.dtype != np.bool_:
        raise ValueError("physics_clone_valid must be boolean")
    for name, shape in _PHYSICS_SHAPES.items():
        array = _as_array(name, records[name], shape, row_count)
        # A crashed candidate may be non-finite, but only a valid candidate must be finite.
        if name == "physics_score":
            valid = np.asarray(records["physics_candidate_valid"], dtype=bool)
            if not np.isfinite(array[valid]).all():
                raise ValueError("finite physics_score required for valid candidates")
        elif name == "physics_next_object_pose":
            valid = np.asarray(records["physics_candidate_valid"], dtype=bool)
            if not np.isfinite(array[valid]).all():
                raise ValueError("finite physics_next_object_pose required for valid candidates")
        elif name == "physics_next_IG":
            valid = np.asarray(records["physics_candidate_valid"], dtype=bool)
            if not np.isfinite(array[valid]).all():
                raise ValueError("finite physics_next_IG required for valid candidates")
        elif name != "physics_candidate_valid" and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
    assignments = np.asarray(records["env_candidate_ids"])
    if assignments.dtype.kind not in "iu":
        raise ValueError("env_candidate_ids must be integer")
    expected_assignment = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7])
    if not np.all(np.sort(assignments, axis=1) == expected_assignment[None]):
        raise ValueError("each env_candidate_ids row must contain [0,0,1,2,3,4,5,6,7]")
    if np.asarray(records["physics_candidate_valid"]).dtype != np.bool_:
        raise ValueError("physics_candidate_valid must be boolean")
    for name in ("duplicate_delta_object_pose", "duplicate_delta_IG", "duplicate_delta_score"):
        array = np.asarray(records[name])
        if array.shape[0] != row_count or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite with {row_count} rows")
    return row_count


def validate_episode_replay(record: Mapping[str, Any]) -> int:
    """Validate one source-episode prefix-replay artifact and return its horizon."""
    required = {"episode_id", "seed", "initial_dof_state", "initial_actor_root_state",
                "initial_task_indices", "executed_action_history", "done_history",
                "reference_index", "data_id", "start_time", "progress_history",
                "resolved_sim_config_sha256"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"episode replay missing fields: {sorted(missing)}")
    actions = np.asarray(record["executed_action_history"])
    if actions.ndim != 2 or actions.shape[1:] != (18,) or not np.isfinite(actions).all():
        raise ValueError("executed_action_history must be finite [T,18]")
    if (actions < -1).any() or (actions > 1).any():
        raise ValueError("executed_action_history must remain in native [-1,1]")
    horizon = int(actions.shape[0])
    done = np.asarray(record["done_history"])
    reference = np.asarray(record["reference_index"])
    progress = np.asarray(record["progress_history"])
    for name, value in (("done_history", done), ("reference_index", reference),
                        ("progress_history", progress)):
        if value.ndim == 0 or value.shape[0] != horizon:
            raise ValueError(f"{name} must have first dimension T={horizon}")
    if done.dtype != np.bool_:
        raise ValueError("done_history must be boolean")
    for name in ("initial_dof_state", "initial_actor_root_state", "initial_task_indices"):
        value = np.asarray(record[name])
        if value.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{name} must be non-empty")
    for name in ("episode_id", "seed", "data_id", "start_time", "resolved_sim_config_sha256"):
        value = np.asarray(record[name])
        if value.size != 1 or not str(value.reshape(-1)[0]):
            raise ValueError(f"{name} must be a non-empty scalar")
    if len(str(np.asarray(record["resolved_sim_config_sha256"]).reshape(-1)[0])) != 64:
        raise ValueError("resolved_sim_config_sha256 must be a SHA256 hex string")
    return horizon


def write_json(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def build_manifest(
    *,
    run_id: str,
    output_dir: str | Path,
    git_commit: str,
    guidance_sha256: Mapping[str, str],
    plan_sha256: str,
    inputs: Mapping[str, Mapping[str, str]],
    protocol: Mapping[str, Any],
    command: str,
) -> dict[str, Any]:
    """Build a manifest with the fixed V1.21c identity fields."""
    if not run_id or not git_commit:
        raise ValueError("run_id and git_commit are required")
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "CmResidual",
        "work_version": "V1.21",
        "run_id": run_id,
        "run_status": "STARTED",
        "output_dir": str(Path(output_dir).resolve()),
        "git_commit": git_commit,
        "guidance_sha256": dict(guidance_sha256),
        "plan_sha256": plan_sha256,
        "resolved_inputs": dict(inputs),
        "protocol": dict(protocol),
        "command": command,
        "schema": {"state_records": STATE_SCHEMA, "physics_records": PHYSICS_SCHEMA},
        "conclusion": "INCONCLUSIVE",
    }
