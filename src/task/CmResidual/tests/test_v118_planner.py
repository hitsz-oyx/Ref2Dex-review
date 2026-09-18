from pathlib import Path

import numpy as np
import torch

from src.task.CmDecoderv2.kinematics import InspireKinematics
from src.task.CmResidual.v118_planner import ACTION_DIM, PlannerConfig, QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
URDF = ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"


def test_v118_gpu_fk_matches_pinned_cpu_urdf_at_tolerance():
    torch.manual_seed(17)
    native = torch.randn(2, 3, ACTION_DIM) * 0.1
    actual = TorchInspireKinematics(URDF, "cpu").forward(native)
    legacy = InspireKinematics(URDF)
    expected = np.asarray([[
        [legacy.link_transforms_native(native[batch, candidate].numpy())[name] for name in QUERY_LINKS]
        for candidate in range(native.shape[1])
    ] for batch in range(native.shape[0])])
    assert actual.shape == (2, 3, len(QUERY_LINKS), 4, 4)
    assert np.max(np.abs(actual.numpy() - expected)) < 1e-5


def test_v118_planner_contract_is_fixed_to_eight_candidates():
    assert PlannerConfig().candidates == 8
    assert PlannerConfig().translation_scale_m == 0.02
    assert PlannerConfig().rotation_scale_rad == 0.05
    assert PlannerConfig().max_active_envs == 16
