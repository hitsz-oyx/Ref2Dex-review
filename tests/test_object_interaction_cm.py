from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.task.ObjectInteractionCm.dataset import ObjectInteractionCmDataset
from src.task.ObjectInteractionCm.model import LocalHandInteraction, ObjectInteractionCmModel
from src.task.ObjectInteractionCm.runner import masked_vector_huber


def _write_dual_hand_npz(root: Path) -> Path:
    sequence = root / "grab" / "s1" / "toy"
    sequence.mkdir(parents=True)
    frames, pool, hand = 6, 4096, 8
    obj = np.zeros((frames, pool, 3), np.float32)
    obj[:, :, 0] = np.linspace(0.0, 0.02, pool, dtype=np.float32)[None, :]
    obj_future = obj + np.array([0.001, 0.0, 0.0], np.float32)
    normals = np.zeros_like(obj)
    normals[..., 2] = 1.0
    pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    np.savez(sequence / "shared.npz", schema_name=np.asarray("test"), dataset_name=np.asarray("grab"),
             raw_frame_id=np.arange(frames, dtype=np.int32), ds_rate=np.asarray(1), source_fps=np.asarray(30.0),
             obj_points_world=obj, obj_normals_world=normals, obj_root_pose_world=pose,
             obj_future_for_test=obj_future)
    # The Dataset reads the future frame from obj_points_world, so encode the
    # motion directly in the later frames.
    with np.load(sequence / "shared.npz") as data:
        values = {key: data[key] for key in data.files if key != "obj_future_for_test"}
    values["obj_points_world"] = obj + np.arange(frames, dtype=np.float32)[:, None, None] * np.array([0.001, 0, 0], np.float32)
    np.savez(sequence / "shared.npz", **values)
    hand_points = np.zeros((frames, hand, 3), np.float32)
    hand_points[0, :, 2] = 0.10  # inactive at frame 0
    hand_points[1:, :, 2] = 0.01  # active at all later frames
    hand_normals = np.zeros_like(hand_points)
    hand_normals[..., 2] = -1.0
    active = np.zeros((frames, pool), bool)
    active[1:, :32] = True
    for side in ("left", "right"):
        np.savez(sequence / f"{side}.npz", side=np.asarray(side), hand_points_world=hand_points,
                 hand_normals_world=hand_normals, obj_candidate_mask_5cm=active)
    return root


def test_dual_hand_dataset_filters_current_frames_and_samples_1024(tmp_path: Path) -> None:
    root = _write_dual_hand_npz(tmp_path)
    dataset = ObjectInteractionCmDataset(root, num_obj_points=1024, num_hand_points=8, min_stride=1, max_stride=1)
    assert len(dataset) == 4  # frame 0 is >5cm; last frame has no future frame
    sample = dataset[0]
    assert sample["obj_points"].shape == (1024, 3)
    assert sample["hand_points"].shape == (16, 3)
    assert sample["hand_valid_mask"].shape == (16,)
    assert sample["hand_supervision_mask"].shape == (16,)
    assert sample["hand_valid_mask"].all()
    assert sample["hand_supervision_mask"].all()
    assert sample["obj_valid_mask"].all()


def test_local_interaction_cross_attention_masks_edges_and_model_shapes() -> None:
    torch.manual_seed(5)
    module = LocalHandInteraction(dim=32, knn_k=8, radius_m=0.05)
    object_points = torch.zeros(1, 2, 3)
    object_normals = torch.tensor([[[0.0, 0.0, 1.0]] * 2])
    hand_points = torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, 0.02], [1.0, 1.0, 1.0]]])
    hand_normals = torch.tensor([[[0.0, 0.0, -1.0]] * 3])
    hand_flow = torch.zeros_like(hand_points)
    object_features = torch.randn(1, 2, 32)
    interaction, diagnostics = module(object_features, object_points, object_normals, hand_points, hand_normals, hand_flow)
    assert interaction.shape == (1, 2, 32)
    assert diagnostics["edge_valid_mask"].shape == (1, 2, 3)
    assert torch.isfinite(interaction).all()
    far_hand = torch.full((1, 3, 3), 1.0)
    far_interaction, far_diag = module(
        object_features, object_points, object_normals, far_hand, hand_normals, hand_flow
    )
    assert not far_diag["edge_valid_mask"].any()
    assert torch.equal(far_interaction, torch.zeros_like(far_interaction))
    model_cfg = SimpleNamespace(meta=SimpleNamespace(feature_dim=32, num_cm_tokens=16, slot_iters=3, knn_k=8, interaction_radius_m=0.05))
    model = ObjectInteractionCmModel(model_cfg)
    batch = {
        "obj_points": torch.randn(1, 4, 3),
        "obj_normals": torch.randn(1, 4, 3),
        "obj_valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "hand_points": torch.randn(1, 6, 3),
        "hand_normals": torch.randn(1, 6, 3),
        "hand_flow": torch.randn(1, 6, 3),
        "hand_valid_mask": torch.ones(1, 6, dtype=torch.bool),
    }
    output = model(batch)
    assert output["cm_tokens"].shape == (1, 16, 32)
    assert output["cm_assignment"].shape == (1, 16, 4)
    assert output["cm_anchor_pos"].shape == (1, 16, 3)
    assert output["pred_obj_flow"].shape == (1, 4, 3)
    assert output["pred_hand_flow"].shape == (1, 6, 3)
    assert torch.isfinite(output["pred_obj_flow"]).all()


def test_empty_hand_mask_has_zero_finite_loss() -> None:
    prediction = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    loss = masked_vector_huber(prediction, target, mask)
    assert float(loss) == 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))
