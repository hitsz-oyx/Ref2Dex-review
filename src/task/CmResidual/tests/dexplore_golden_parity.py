"""Isolated source-level parity check; Isaac Gym must be imported before torch."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


for name, value in (("float", float), ("int", int), ("bool", bool)):
    if name not in np.__dict__:
        setattr(np, name, value)

ROOT = Path(__file__).resolve().parents[4]
ISAAC_GYM = Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python"))
DEXPLORE = Path(os.environ.get("DEXPLORE_SOURCE_ROOT", "/home2/wyy/oyx_ws/dexplore/dexplore"))
for path in (ISAAC_GYM, ROOT / "third_party/IsaacGymEnvs", DEXPLORE):
    sys.path.insert(0, str(path))

import isaacgym  # noqa: F401,E402
import torch  # noqa: E402
from env.tasks import base_dexplore_task as original  # noqa: E402


def main() -> int:
    module_path = ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/dexplore_observation.py"
    spec = importlib.util.spec_from_file_location("cmresidual_v14_observation", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(123)
    batch, bodies = 2, 25
    body_pos = torch.randn(batch, bodies, 3)
    body_rot = torch.nn.functional.normalize(torch.randn(batch, bodies, 4), dim=-1)
    body_vel = torch.randn(batch, bodies, 3)
    body_ang = torch.randn(batch, bodies, 3)
    contact = torch.randn(batch, bodies, 3)
    key_ids = torch.tensor([7, 12, 13, 21, 14, 15, 22, 18, 19, 24, 16, 17, 23, 8, 10, 20])
    contact_ids = torch.tensor([13, 15, 19, 17, 11])
    reference = torch.randn(batch, 428)
    for section in (slice(0, 4), slice(109, 113)):
        reference[:, section] = torch.nn.functional.normalize(reference[:, section], dim=-1)
    reference[:, 232:296] = torch.nn.functional.normalize(
        reference[:, 232:296].view(batch, 16, 4), dim=-1).view(batch, -1)
    reference[:, 168:184] = (torch.rand(batch, 16) > 0.5).float()
    object_state = torch.randn(batch, 13)
    object_state[:, 3:7] = torch.nn.functional.normalize(object_state[:, 3:7], dim=-1)
    object_points = torch.randn(256, 3)

    body_expected = original.compute_humanoid_observations_max(
        body_pos, body_rot, body_vel, body_ang, False, False,
        contact, contact_ids, reference, key_ids)
    object_expected, world_points = original.compute_obj_observations(
        body_pos, body_rot, object_state,
        object_points.unsqueeze(0).repeat(batch, 1, 1), reference)
    actual_key = body_pos[:, key_ids[[0, 3, 6, 9, 12, 15]]]
    interaction = original.compute_sdf(actual_key, world_points).view(-1, 3)
    heading = original.torch_utils.calc_heading_quat_inv(body_rot[:, 7])
    interaction = original.quat_rotate(
        heading[:, None].repeat(1, 6, 1).view(-1, 4), interaction).view(batch, -1)
    ref_interaction = reference[:, 184:232].view(batch, 16, 3)
    ref_interaction = ref_interaction[:, [0, 3, 6, 9, 12, 15]].reshape(batch, -1)
    expected = torch.cat((
        body_expected, object_expected, interaction, ref_interaction - interaction), dim=-1)

    actual = module.build_dexplore_observation(
        body_pos, body_rot, body_vel, body_ang, contact, object_state,
        reference, object_points, key_ids, contact_ids)
    error = float((actual - expected).abs().max())
    print(json.dumps({"shape": list(actual.shape), "max_abs_error": error}))
    return 0 if tuple(actual.shape) == (batch, 721) and error <= 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
