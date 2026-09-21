"""Intercept the first active V1.18 planner state and profile V1.18b variants."""
from __future__ import annotations

import os
import runpy

from isaacgym import gymapi  # noqa: F401  # Isaac Gym must precede torch.
import numpy as np

from src.task.CmResidual.v118_latency_profile import profile_interaction_variants, write_profile
from src.task.CmResidual.v118_planner import FrozenCmv2Planner


ORIGINAL_TEACHER = FrozenCmv2Planner.teacher
CALLS = 0


def _intercept(self, *args, **kwargs):
    global CALLS
    CALLS += 1
    if args:
        raise RuntimeError("V1.18 task must call planner teacher with keyword inputs")
    object_points, _ = self.geometry.object(kwargs["object_pose"])
    active = self._activation(
        kwargs["current_links"], object_points,
        kwargs["reference_transport"], kwargs["desired_delta_xi"],
    )
    if not bool(active.any()):
        return ORIGINAL_TEACHER(self, **kwargs)
    FrozenCmv2Planner.teacher = ORIGINAL_TEACHER
    try:
        payload = profile_interaction_variants(
            self, kwargs,
            warmup=int(os.environ["REF2DEX_V118B_WARMUP"]),
            repeats=int(os.environ["REF2DEX_V118B_REPEATS"]),
        )
        payload["planner_call_index"] = CALLS
        write_profile(os.environ["REF2DEX_V118B_PROFILE"], payload)
        print(f"V1.18b profile completed at planner call {CALLS}", flush=True)
        os._exit(0)
    except BaseException:
        FrozenCmv2Planner.teacher = ORIGINAL_TEACHER
        raise


FrozenCmv2Planner.teacher = _intercept
np.float = float
runpy.run_path(os.environ["REF2DEX_V118B_VENDOR_TRAIN"], run_name="__main__")
