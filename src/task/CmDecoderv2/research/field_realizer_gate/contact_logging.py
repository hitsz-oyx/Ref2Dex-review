"""Read-only Isaac Gym hand↔object contact extraction for the V1.1.16 evaluator."""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Iterable
from pathlib import Path

import numpy as np


def contact_api_capability() -> dict[str, Any]:
    """Report API availability without creating a simulator or changing physics."""
    names = ("get_env_rigid_contacts", "get_env_rigid_contact_forces")
    # Isaac Gym requires its bindings to be imported before torch.  The audit
    # itself may already have imported torch, so probe in a clean subprocess.
    code = "from isaacgym import gymapi; print(int(all(hasattr(gymapi.Gym, n) for n in %r)))" % (names,)
    env = dict(os.environ)
    isaac_python = Path("/home2/wyy/isaac-gym/isaacgym/python")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(isaac_python) + (os.pathsep + current if current else "")
    try:
        value = subprocess.check_output([sys.executable, "-c", code], env=env, stderr=subprocess.STDOUT, text=True)
        return {"available": value.strip().splitlines()[-1] == "1", "methods": names, "probe": "clean_subprocess"}
    except Exception as error:  # pragma: no cover - depends on local Isaac Gym
        return {"available": False, "methods": names, "reason": f"{type(error).__name__}: {error}"}


def _field_name(dtype: np.dtype, candidates: Iterable[str]) -> str:
    names = set(dtype.names or ())
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"Rigid contact records do not expose any of {tuple(candidates)}; fields={sorted(names)}")


def extract_hand_object_contacts(
    records: Any,
    *,
    hand_body_ids: Iterable[int],
    object_body_ids: Iterable[int],
) -> dict[str, np.ndarray]:
    """Filter one environment's rigid contacts by body identity.

    The function accepts the structured array returned by
    ``Gym.get_env_rigid_contacts`` and never treats net contact force as a
    pairwise label.  Normal-force extraction is best effort because Isaac Gym
    versions expose different force field names; missing force is explicit
    ``NaN`` rather than a fabricated zero.
    """
    array = np.asarray(records)
    if array.dtype.names is None:
        raise TypeError("get_env_rigid_contacts must return a structured array")
    body0 = _field_name(array.dtype, ("body0", "body0_id"))
    body1 = _field_name(array.dtype, ("body1", "body1_id"))
    hand = np.asarray(tuple(int(x) for x in hand_body_ids), dtype=np.int64)
    obj = np.asarray(tuple(int(x) for x in object_body_ids), dtype=np.int64)
    h0 = np.isin(array[body0], hand)
    h1 = np.isin(array[body1], hand)
    o0 = np.isin(array[body0], obj)
    o1 = np.isin(array[body1], obj)
    mask = (h0 & o1) | (h1 & o0)
    selected = array[mask]
    force_name = next((name for name in ("force", "normal_force", "lambda") if name in (array.dtype.names or ())), None)
    if force_name is None:
        normal_force = np.full(len(selected), np.nan, dtype=np.float32)
    else:
        force = np.asarray(selected[force_name])
        normal_force = np.asarray(force, dtype=np.float32).reshape(len(selected), -1)
        normal_force = np.linalg.norm(normal_force, axis=1)
    return {"records": selected, "mask": mask, "normal_force": normal_force.astype(np.float32, copy=False)}


class PairwiseContactTracker:
    """Evaluator-only tracker for one rollout step across all environments.

    Isaac Gym reports body ids in each environment's local domain.  The
    current asset creation order is hand, then the one-body airplane, then
    table; the caller supplies the hand body ids and the object body id after
    validating that ordering from the loaded asset.
    """

    def __init__(self, *, hand_body_ids: Iterable[int], object_body_ids: Iterable[int]) -> None:
        self.hand_body_ids = tuple(int(x) for x in hand_body_ids)
        self.object_body_ids = tuple(int(x) for x in object_body_ids)
        if not self.hand_body_ids or not self.object_body_ids:
            raise ValueError("contact tracker requires non-empty hand and object body ids")

    def capture(self, gym: Any, envs: Iterable[Any]) -> dict[str, np.ndarray]:
        occupancy = []
        counts = []
        normal_force = []
        for env in envs:
            contacts = extract_hand_object_contacts(
                gym.get_env_rigid_contacts(env),
                hand_body_ids=self.hand_body_ids,
                object_body_ids=self.object_body_ids,
            )
            occupancy.append(bool(len(contacts["records"])))
            counts.append(len(contacts["records"]))
            finite_force = contacts["normal_force"][np.isfinite(contacts["normal_force"])]
            normal_force.append(float(finite_force.sum()) if len(finite_force) else np.nan)
        return {
            "occupancy": np.asarray(occupancy, dtype=bool),
            "contact_count": np.asarray(counts, dtype=np.int32),
            "normal_force": np.asarray(normal_force, dtype=np.float32),
        }
