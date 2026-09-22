from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch

from src.task.ObjectInteractionCmv2.articulated import (
    ARTICULATED_VERSION,
    ArticulatedObjectInteractionCmv2V15Model,
    ArticulatedTransitions,
    analytic_fk_flow,
    articulated_v15_loss,
    collate_articulated,
)
from src.task.ObjectInteractionCmv2.tests.test_v1_4_three_domain import _make_domain


def _zero_root(batch: int) -> torch.Tensor:
    return torch.zeros((batch, 6), dtype=torch.float32)


def test_analytic_fk_reduces_to_rigid_root_se3():
    points = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    delta_xi = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.0, math.pi / 2]])
    flow = analytic_fk_flow(
        points, torch.zeros((1, 2), dtype=torch.long), torch.ones((1, 1), dtype=torch.bool),
        torch.empty((1, 0), dtype=torch.long), torch.empty((1, 0), dtype=torch.long),
        torch.empty((1, 0, 3)), torch.empty((1, 0, 3)), torch.empty((1, 0), dtype=torch.bool),
        delta_xi, torch.empty((1, 0)),
    )
    expected_future = torch.tensor([[[0.1, 0.8, 0.3], [-1.9, -0.2, 0.3]]])
    assert torch.allclose(points + flow, expected_future, atol=1e-5)


def test_analytic_fk_rotates_only_child_link():
    points = torch.tensor([[[0.2, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    flow = analytic_fk_flow(
        points, torch.tensor([[0, 1]]), torch.ones((1, 2), dtype=torch.bool),
        torch.tensor([[0]]), torch.tensor([[1]]), torch.tensor([[[0.0, 0.0, 1.0]]]),
        torch.zeros((1, 1, 3)), torch.ones((1, 1), dtype=torch.bool), _zero_root(1),
        torch.tensor([[math.pi / 2]]),
    )
    assert torch.allclose(points[0, 0] + flow[0, 0], points[0, 0], atol=1e-6)
    assert torch.allclose(points[0, 1] + flow[0, 1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def _sample(domain: str, hand_points: int, joints: int) -> dict:
    object_points = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
    result = {
        "obj_points": object_points,
        "obj_normals": torch.nn.functional.normalize(object_points + 0.1, dim=-1),
        "hand_points": torch.zeros((hand_points, 3)), "hand_normals": torch.tensor([0.0, 0.0, 1.0]).expand(hand_points, 3).clone(),
        "hand_flow": torch.zeros((hand_points, 3)), "obj_flow_gt": torch.zeros((4, 3)),
        "obj_link_id": torch.tensor([0, 0, 1, 1] if joints else [0, 0, 0, 0]),
        "link_valid_mask": torch.ones((2 if joints else 1,), dtype=torch.bool),
        "joint_parent": torch.tensor([0] if joints else [], dtype=torch.long),
        "joint_child": torch.tensor([1] if joints else [], dtype=torch.long),
        "joint_axis_root": torch.tensor([[0.0, 0.0, 1.0]] if joints else [], dtype=torch.float32).reshape(joints, 3),
        "joint_origin_root": torch.zeros((joints, 3)), "joint_q_t": torch.zeros((joints,)),
        "joint_valid_mask": torch.ones((joints,), dtype=torch.bool), "delta_q_gt": torch.zeros((joints,)),
        "delta_xi_root_gt": torch.zeros((6,)), "delta_time_s": torch.tensor(1 / 30),
        "hand_valid_mask": torch.ones((hand_points,), dtype=torch.bool), "source": domain, "sequence_id": domain,
    }
    return result


def test_mixed_batch_masks_and_v15_forward_backward_are_finite():
    torch.manual_seed(4)
    batch = collate_articulated([_sample("grab", 8, 0), _sample("arctic", 11, 1)])
    assert batch["hand_points"].shape == (2, 11, 3)
    assert batch["link_valid_mask"].tolist() == [[True, False], [True, True]]
    assert batch["joint_valid_mask"].tolist() == [[False], [True]]
    model = ArticulatedObjectInteractionCmv2V15Model(SimpleNamespace(
        hidden_width=16, knn_k=4, interaction_radius_m=0.02, feature_scale_m=0.02, frame_dt_s=1 / 30,
    ))
    assert model.architecture_version == ARTICULATED_VERSION
    output = model(batch)
    loss = articulated_v15_loss(output, batch)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_arctic_loader_uses_root_pose_and_current_q(tmp_path):
    sequence, _, _ = _make_domain(tmp_path, "arctic", "mano", frames=4)
    geometry = sequence / "geometry"
    root = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    root[:, 1, 3] = np.arange(4, dtype=np.float32) * 0.01
    np.save(geometry / "obj_root_pose_world.npy", root)
    np.save(geometry / "obj_part_id.npy", np.r_[np.zeros(2048, dtype=np.int64), np.ones(2048, dtype=np.int64)])
    np.save(geometry / "obj_articulation.npy", np.arange(4, dtype=np.float32)[:, None] * 0.1)
    dataset = ArticulatedTransitions([{
        "name": "arctic", "path": str(sequence), "hand_variant": "mano",
        "articulation": {"num_links": 2, "joints": [{"parent": 0, "child": 1, "type": "revolute",
                                                            "axis_root": [0, 0, 1], "origin_root": [0, 0, 0]}]},
    }], "train", num_obj_points=16, fixed_stride=1, base_seed=9)
    sample = dataset[0]
    assert sample["obj_link_id"].min() >= 0 and sample["obj_link_id"].max() < 2
    assert torch.allclose(sample["delta_xi_root_gt"][:3], torch.tensor([0.0, 0.01, 0.0]), atol=1e-6)
    assert torch.allclose(sample["joint_q_t"], torch.tensor([0.0]))
    assert torch.allclose(sample["delta_q_gt"], torch.tensor([0.1]))
