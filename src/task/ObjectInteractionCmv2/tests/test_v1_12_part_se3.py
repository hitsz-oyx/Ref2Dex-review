from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.task.ObjectInteractionCmv2.part_se3 import (
    PART_SE3_VERSION,
    PartSE3ObjectInteractionCmv2V112Model,
    collate_part_se3,
    endpoint_union_topk,
    part_se3_v112_loss,
    transform_points_by_part,
)
from src.task.ObjectInteractionCmv2.multi_domain import ThreeDomainTransitions
from src.task.ObjectInteractionCmv2.part_se3_data import (
    OakInkWholePartTransitions,
    RigidArticulatedPartTransitions,
    stratified_part_indices,
)
from src.task.ObjectInteractionCmv2.part_se3_training import (
    CHECKPOINT_SCHEMA,
    load_part_se3_config,
    checkpoint_payload,
    initialize_random_model,
    restore_checkpoint,
    load_part_se3_training_config,
)
from src.task.ObjectInteractionCmv2.tests.test_mixed_training_v1_11g import make_part_adapter
from src.task.ObjectInteractionCmv2.tests.test_v1_4_three_domain import _make_domain
from src.task.ObjectInteractionCmv2.train_part_se3_ddp import validation_indices


def _model() -> PartSE3ObjectInteractionCmv2V112Model:
    return PartSE3ObjectInteractionCmv2V112Model(SimpleNamespace(
        hidden_width=16, num_tokens=16, knn_k=32, interaction_radius_m=0.02,
        feature_scale_m=0.02, frame_dt_s=1 / 30,
    ))


def _sample(parts: int, *, interacting_parts: tuple[int, ...] = (0,)) -> dict:
    generator = torch.Generator().manual_seed(100 + parts)
    count, hand_count = 64, 40
    part_ids = torch.arange(count) % parts
    points = torch.randn((count, 3), generator=generator) * 0.003
    points[:, 0] += part_ids * 0.08
    hand = torch.full((hand_count, 3), 0.5)
    for part in interacting_parts:
        chosen = torch.nonzero(part_ids == part).flatten()[0]
        hand[part] = points[chosen] + torch.tensor([0.001, 0.0, 0.0])
    normals = torch.nn.functional.normalize(torch.randn((count, 3), generator=generator), dim=-1)
    hand_normals = torch.nn.functional.normalize(
        torch.randn((hand_count, 3), generator=generator), dim=-1)
    return {
        "obj_points": points,
        "obj_normals": normals,
        "obj_part_id": part_ids,
        "part_valid_mask": torch.ones((parts,), dtype=torch.bool),
        "hand_points": hand,
        "hand_normals": hand_normals,
        "hand_flow": torch.zeros_like(hand),
        "hand_valid_mask": torch.ones((hand_count,), dtype=torch.bool),
        "delta_translation_part_gt": torch.zeros((parts, 3)),
        "delta_rotation_part_gt": torch.eye(3).expand(parts, 3, 3).clone(),
        "obj_flow_gt": torch.zeros_like(points),
        "delta_time_s": torch.tensor(1 / 30),
        "source": "synthetic",
        "sequence_id": f"synthetic-{parts}",
    }


def _brute_endpoint(object_points, hand_points, hand_flow, k):
    start = torch.cdist(object_points, hand_points)
    end = torch.cdist(object_points, hand_points + hand_flow)
    rows = []
    for obj in range(object_points.shape[1]):
        start_ids = sorted(range(hand_points.shape[1]), key=lambda i: (float(start[0, obj, i]), i))[:k]
        end_ids = sorted(range(hand_points.shape[1]), key=lambda i: (float(end[0, obj, i]), i))[:k]
        union = set(start_ids) | set(end_ids)
        rows.append(sorted(union, key=lambda i: (float(min(start[0, obj, i], end[0, obj, i])), i))[:k])
    return torch.tensor(rows).unsqueeze(0)


def test_endpoint_knn_unions_then_reranks_to_32_with_stable_ties():
    hand = torch.zeros((1, 40, 3))
    hand[0, :, 0] = torch.arange(40) * 0.001
    flow = torch.zeros_like(hand)
    flow[0, 32:, 0] = -0.04
    objects = torch.tensor([[[0.0, 0.0, 0.0], [0.0045, 0.0, 0.0]]])
    distance, indices, start, end = endpoint_union_topk(
        objects, hand, flow, k=32, object_chunk=1, hand_chunk=7)
    assert indices.shape == distance.shape == start.shape == end.shape == (1, 2, 32)
    assert torch.equal(indices, _brute_endpoint(objects, hand, flow, 32))
    assert 39 in indices[0, 0].tolist()
    assert indices[0, 1, 0].item() == 4


def test_endpoint_knn_rejects_fewer_than_32_valid_points():
    points = torch.zeros((1, 2, 3))
    hands = torch.zeros((1, 40, 3))
    valid = torch.arange(40)[None] < 31
    with pytest.raises(ValueError, match="at least 32"):
        endpoint_union_topk(points, hands, torch.zeros_like(hands), valid)


def test_stratified_sampling_is_deterministic_no_replace_and_has_floor():
    ids = np.repeat(np.arange(3), [2000, 1500, 596])
    first = stratified_part_indices(ids, 1024, seed=9)
    second = stratified_part_indices(ids, 1024, seed=9)
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 1024
    counts = np.bincount(ids[first], minlength=3)
    assert np.all(counts >= 64)
    assert counts.sum() == 1024


def test_direct_part_se3_replays_each_point_from_stable_identity():
    points = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 0.0, 0.0]]])
    ids = torch.tensor([[0, 0, 1]])
    motion = torch.tensor([[[0.1, 0.2, 0.3, 0.0, 0.0, math.pi / 2],
                            [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    future = transform_points_by_part(points, motion, ids)
    expected = torch.tensor([[[0.1, 1.2, 0.3], [-1.9, 0.2, 0.3], [1.9, 0.0, 0.0]]])
    assert torch.allclose(future, expected, atol=1e-5)


def test_grab_and_arctic_whole_object_adapters_replay_direct_parts(tmp_path):
    grab, _, _ = _make_domain(tmp_path, "grab", "mano", frames=4)
    grab_data = RigidArticulatedPartTransitions([{
        "name": "grab", "path": str(grab), "hand_variant": "mano",
        "articulation": {"num_links": 1, "joints": []},
    }], "train", stride_values=(1,), base_seed=7)
    grab_sample = grab_data[0]
    assert torch.bincount(grab_sample["obj_part_id"]).tolist() == [1024]
    assert grab_sample["pose_flow_residual_max_m"] < 2e-4

    arctic, _, _ = _make_domain(tmp_path, "arctic", "mano", frames=4)
    geometry = arctic / "geometry"
    root = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    root[:, 1, 3] = np.arange(4, dtype=np.float32) * 0.01
    part_ids = np.r_[np.zeros(2048, dtype=np.int64), np.ones(2048, dtype=np.int64)]
    q = np.arange(4, dtype=np.float32) * 0.1
    local = np.load(geometry / "obj_points_pool_world.npy")[0]
    points = np.empty((4, 4096, 3), dtype=np.float32)
    for frame in range(4):
        cosine, sine = np.cos(q[frame]), np.sin(q[frame])
        rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)
        root_points = local.copy()
        root_points[part_ids == 1] = root_points[part_ids == 1] @ rotation.T
        points[frame] = root_points @ root[frame, :3, :3].T + root[frame, :3, 3]
    np.save(geometry / "obj_points_pool_world.npy", points)
    np.save(geometry / "obj_root_pose_world.npy", root)
    np.save(geometry / "obj_part_id.npy", part_ids)
    np.save(geometry / "obj_articulation.npy", q[:, None])
    arctic_data = RigidArticulatedPartTransitions([{
        "name": "arctic", "path": str(arctic), "hand_variant": "mano",
        "articulation": {"num_links": 2, "joints": [{
            "parent": 0, "child": 1, "type": "revolute",
            "axis_root": [0, 0, 1], "origin_root": [0, 0, 0],
        }]},
    }], "train", stride_values=(1,), base_seed=7)
    arctic_sample = arctic_data[0]
    assert torch.all(torch.bincount(arctic_sample["obj_part_id"], minlength=2) >= 64)
    assert arctic_sample["pose_flow_residual_max_m"] < 2e-4


def test_oakink_whole_object_adapter_keeps_both_parts_and_same_point_ids(tmp_path):
    sequence, index, manifest = _make_domain(tmp_path, "oakink2", "mano", frames=4)
    geometry = sequence / "geometry"
    base_points = np.load(geometry / "obj_points_pool_world.npy")[0]
    poses = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    combined = np.empty((4, 4096, 3), dtype=np.float32)
    for frame in range(4):
        combined[frame, :2048] = base_points[:2048] + [0.01 * frame, 0, 0]
        combined[frame, 2048:] = base_points[2048:] + [0, 0.02 * frame, 0]
    np.save(geometry / "obj_points_pool_world.npy", combined)
    np.save(geometry / "obj_pose_world.npy", poses)
    np.save(geometry / "obj_point_id.npy", np.r_[np.arange(2048), 4096 + np.arange(2048)])
    adapter = make_part_adapter(tmp_path, sequence, "oakink2/demo", parts=2)
    base = ThreeDomainTransitions(
        [{"name": "oakink2", "hand_variant": "mano", "index": str(index),
          "manifest": str(manifest)}], "train", fixed_stride=1, active_only=False)
    dataset = OakInkWholePartTransitions(base, adapter)
    sample = dataset[0]
    counts = torch.bincount(sample["obj_part_id"], minlength=2)
    assert sample["obj_points"].shape == sample["obj_flow_gt"].shape == (1024, 3)
    assert torch.all(counts >= 64) and counts.sum() == 1024
    assert sample["pose_flow_residual_max_m"] < 2e-4


def test_model_routes_surface_only_to_directly_interacting_part_then_propagates():
    torch.manual_seed(4)
    batch = collate_part_se3([_sample(2, interacting_parts=(0,))])
    model = _model()
    assert model.architecture_version == PART_SE3_VERSION
    output = model(batch)
    assert output["part_has_interaction"].tolist() == [[True, False]]
    assert torch.count_nonzero(output["part_surface_features"][0, 1]).item() == 0
    assert torch.allclose(output["part_token_rho"].sum(1)[output["token_mask"]],
                          torch.ones_like(output["token_mass"][output["token_mask"]]), atol=1e-5)
    loss = part_se3_v112_loss(output, batch)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_point_and_part_permutations_are_equivariant():
    torch.manual_seed(8)
    model = _model().eval()
    sample = _sample(3, interacting_parts=(0, 2))
    batch = collate_part_se3([sample])
    with torch.no_grad():
        reference = model(batch)

    point_order = torch.randperm(64, generator=torch.Generator().manual_seed(3))
    point_sample = dict(sample)
    for key in ("obj_points", "obj_normals", "obj_part_id", "obj_flow_gt"):
        point_sample[key] = sample[key][point_order]
    with torch.no_grad():
        point_output = model(collate_part_se3([point_sample]))
    inverse = torch.argsort(point_order)
    assert torch.allclose(point_output["obj_flow_pred"][:, inverse], reference["obj_flow_pred"], atol=2e-6)

    permutation = torch.tensor([2, 0, 1])
    inverse_part = torch.empty_like(permutation)
    inverse_part[permutation] = torch.arange(3)
    part_sample = dict(sample)
    part_sample["obj_part_id"] = inverse_part[sample["obj_part_id"]]
    for key in ("part_valid_mask", "delta_translation_part_gt", "delta_rotation_part_gt"):
        part_sample[key] = sample[key][permutation]
    with torch.no_grad():
        part_output = model(collate_part_se3([part_sample]))
    assert torch.allclose(part_output["delta_xi_part"][:, inverse_part],
                          reference["delta_xi_part"], atol=2e-6)


@pytest.mark.parametrize("angle", [1e-7, math.pi - 1e-6])
def test_so3_geodesic_loss_is_finite_near_zero_and_pi(angle):
    prediction = torch.zeros((1, 1, 6), requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, 0.0],
                             [0.0, math.cos(angle), -math.sin(angle)],
                             [0.0, math.sin(angle), math.cos(angle)]]]])
    output = {"delta_xi_part": prediction, "obj_flow_pred": torch.zeros((1, 2, 3), requires_grad=True)}
    batch = {"part_valid_mask": torch.ones((1, 1), dtype=torch.bool),
             "delta_translation_part_gt": torch.zeros((1, 1, 3)),
             "delta_rotation_part_gt": target, "obj_flow_gt": torch.zeros((1, 2, 3))}
    loss = part_se3_v112_loss(output, batch)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_v112_config_is_random_only_and_checkpoint_rejects_v111(tmp_path):
    config_path = Path(__file__).parents[1] / "configs/active/mixed_part_se3_v1_12.yaml"
    config = load_part_se3_config(config_path)
    first = initialize_random_model(config)
    torch.manual_seed(123)
    second = initialize_random_model(config)
    assert first.architecture_version == second.architecture_version == PART_SE3_VERSION
    assert any(not torch.equal(first.state_dict()[key], second.state_dict()[key])
               for key in first.state_dict())

    optimizer = torch.optim.Adam(second.parameters(), lr=1e-3)
    payload = checkpoint_payload(second, optimizer, epoch=2, step=7, best_metric=0.5)
    assert payload["schema_name"] == CHECKPOINT_SCHEMA
    valid_path = tmp_path / "v112.pt"
    torch.save(payload, valid_path)
    assert restore_checkpoint(valid_path, first) == {"epoch": 2, "step": 7, "best_metric": 0.5}

    payload["architecture_version"] = "v1_5_articulated_fk"
    old_path = tmp_path / "v111.pt"
    torch.save(payload, old_path)
    with pytest.raises(ValueError, match="not a V1.12"):
        restore_checkpoint(old_path, first)


def test_v112_formal_config_freezes_previous_training_budget_on_gpu_zero_two():
    path = Path(__file__).parents[1] / "configs/active/mixed_part_se3_v1_12_ddp.yaml"
    config = load_part_se3_training_config(path)
    assert config["training"]["batch_size_per_rank"] == 64
    assert config["training"]["epochs"] == 16
    assert config["resources"]["physical_gpus"] == [0, 2]
    assert config["initialization"] == "random"
    assert validation_indices(7, 0, 2) == [0, 2, 4, 6]
    assert validation_indices(7, 1, 2) == [1, 3, 5]
