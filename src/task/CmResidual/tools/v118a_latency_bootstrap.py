"""Intercept the first active V1.18 planner state, profile it, then exit."""
from __future__ import annotations

import os
from pathlib import Path
import runpy

from isaacgym import gymapi  # noqa: F401  # Isaac Gym must precede torch.
import numpy as np

from src.task.CmResidual.v118_latency_profile import profile_first_active_state, write_profile
from src.task.CmResidual.v118_planner import FrozenCmv2Planner


ORIGINAL_TEACHER = FrozenCmv2Planner.teacher
CALLS = 0


def _intercept(self, *args, **kwargs):
    global CALLS
    CALLS += 1
    if args:
        raise RuntimeError("V1.18 task must call planner teacher with keyword inputs")
    try:
        # Restore the production method before profiling so all repeated calls
        # execute exactly the real planner path without re-entering this hook.
        object_points, _ = self.geometry.object(kwargs["object_pose"])
        active = self._activation(
            kwargs["current_links"], object_points,
            kwargs["reference_transport"], kwargs["desired_delta_xi"],
        )
        if not bool(active.any()):
            return ORIGINAL_TEACHER(self, **kwargs)
        FrozenCmv2Planner.teacher = ORIGINAL_TEACHER
        payload = profile_first_active_state(
            self, kwargs,
            warmup=int(os.environ["REF2DEX_V118A_WARMUP"]),
            repeats=int(os.environ["REF2DEX_V118A_REPEATS"]),
        )
        payload["planner_call_index"] = CALLS
        write_profile(os.environ["REF2DEX_V118A_PROFILE"], payload)
        print(f"V1.18a profile completed at planner call {CALLS}", flush=True)
        os._exit(0)
    except BaseException:
        FrozenCmv2Planner.teacher = ORIGINAL_TEACHER
        raise


FrozenCmv2Planner.teacher = _intercept
np.float = float
runpy.run_path(os.environ["REF2DEX_V118A_VENDOR_TRAIN"], run_name="__main__")
