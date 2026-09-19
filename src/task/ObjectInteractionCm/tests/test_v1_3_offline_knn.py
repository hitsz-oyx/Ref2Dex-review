from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.task.ObjectInteractionCm.dataset import (
    ObjectInteractionCmDataset,
    _collate_object_interaction_cm,
    _resolve_index_entries,
)
from src.task.ObjectInteractionCm.model import LocalHandInteraction, ObjectInteractionCmModel
from src.task.ObjectInteractionCm.tools.data.calibrate_v1_3_scales import calibrate


def _write_v1_3_sequence(root: Path) -> tuple[Path, Path]:
    sequence = root / "sequences" / "train" / "grab" / "s1_toy"
    geometry = sequence / "geometry"
    geometry.mkdir(parents=True)
    frames, object_points, decoder_points, knn_points, knn_k = 3, 4096, 1538, 5, 32
    obj = np.zeros((frames, object_points, 3), dtype=np.float32)
    obj[1:, :, 0] = 0.002
    obj_normals = np.zeros_like(obj)
    obj_normals[..., 2] = 1.0
    pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    raw = np.arange(frames, dtype=np.int32)
    frame_time = np.arange(frames, dtype=np.float32) / 30.0
    hand = np.zeros((frames, decoder_points, 3), dtype=np.float32)
    hand[..., 2] = 0.01
    hand[1:, ..., 0] = 0.001
    hand_normals = np.zeros_like(hand)
    hand_normals[..., 2] = -1.0
    highres_hand = np.zeros((frames, knn_points, 3), dtype=np.float32)
    highres_hand[..., 2] = 0.01
    highres_hand[1:, ..., 0] = 0.001
    highres_normals = np.zeros_like(highres_hand)
    highres_normals[..., 2] = -1.0
    indices = np.tile(np.arange(knn_k, dtype=np.uint16) % knn_points, (frames, object_points, 1))
    candidate = np.zeros((frames, object_points), dtype=bool)
    candidate[:, :16] = True
    supervision = np.ones((frames, decoder_points), dtype=bool)
    min_distance = np.full((frames,), 0.01, dtype=np.float32)
    for name, value in {
        "obj_points_pool_world.npy": obj,
        "obj_normals_pool_world.npy": obj_normals,
        "obj_pose_world.npy": pose,
        "source_frame_id.npy": raw,
        "frame_time.npy": frame_time,
        "hand_points_world.npy": hand,
        "hand_normals_world.npy": hand_normals,
        "knn_hand_points_world.npy": highres_hand,
        "knn_hand_normals_world.npy": highres_normals,
        "obj_knn_indices.npy": indices,
        "obj_candidate_mask_2cm.npy": candidate,
        "hand_supervision_mask_2cm.npy": supervision,
        "hand_min_object_distance_m.npy": min_distance,
    }.items():
        np.save(geometry / name, value)
    (geometry / "manifest.json").write_text(
        json.dumps(
            {
                "schema_name": "ref2dex_object_interaction_cm_dexplore_rl_v1_3",
                "source": "grab",
                "effective_fps": 30.0,
                "knn_k": knn_k,
                "knn_hand_points": knn_points,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index = root / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
                "knn_k": knn_k,
                "sequences": {
                    "train": [
                        {
                            "id": "s1/toy",
                            "path": "sequences/train/grab/s1_toy",
                            "source": "grab",
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return sequence, index


def test_v1_3_dataset_selects_unique_valid_knn_hand_points(tmp_path: Path) -> None:
    sequence, index = _write_v1_3_sequence(tmp_path)
    entries = _resolve_index_entries(index, "train")
    assert entries == [{"path": str(sequence.resolve()), "source": "grab"}]
    dataset = ObjectInteractionCmDataset(
        Path("."),
        sequence_entries=entries,
        num_obj_points=4,
        num_hand_points=1538,
        max_hand_points=1538,
        max_knn_hand_points=5,
        hand_stream_mode="unique_knn_edges",
        fixed_stride=1,
        active_only=False,
    )
    sample = dataset[0]
    assert sample["hand_points"].shape == (5, 3)
    assert sample["hand_valid_points"].item() == 5
    assert sample["hand_valid_mask"].all()
    assert sample["hand_supervision_mask"].all()
    assert torch.unique(sample["hand_point_ids"]).numel() == 5
    assert sample["knn_edge_indices"].shape == (4, 32)
    assert sample["knn_edge_valid_mask"].shape == (4, 32)
    assert sample["knn_edge_valid_mask"].all()
    assert sample["knn_edge_indices"].dtype == torch.int64
    assert sample["knn_hand_valid_points"].item() == 5
    assert sample["min_hand_object_distance_mm"].item() == 10.0
    assert sample["full_active_count"].item() == 16


def test_offline_knn_path_does_not_construct_cdist(monkeypatch) -> None:
    torch.manual_seed(3)
    module = LocalHandInteraction(dim=16, knn_k=4, radius_m=0.02, geometry_scale=0.02)
    object_points = torch.zeros((1, 2, 3))
    object_normals = torch.zeros((1, 2, 3))
    object_normals[..., 2] = 1.0
    hand_points = torch.tensor(
        [[[0.0, 0.0, 0.010], [0.0, 0.0, 0.015], [0.0, 0.0, 0.030], [0.0, 0.0, 0.040], [0.0, 0.0, 0.050]]]
    )
    hand_normals = torch.zeros_like(hand_points)
    hand_normals[..., 2] = -1.0
    hand_flow = torch.ones_like(hand_points) * 0.001
    hand_valid = torch.ones((1, 5), dtype=torch.bool)
    indices = torch.tensor([[[0, 1, 2, 3], [1, 2, 3, 4]]], dtype=torch.int64)
    object_features = torch.randn((1, 2, 16))

    def fail_cdist(*args, **kwargs):
        raise AssertionError("offline KNN path must not call torch.cdist")

    monkeypatch.setattr(torch, "cdist", fail_cdist)
    interaction, diagnostics = module(
        object_features,
        object_points,
        object_normals,
        hand_points,
        hand_normals,
        hand_flow,
        hand_valid,
        indices,
    )
    expected_distances = torch.linalg.vector_norm(
        torch.gather(
            hand_points[:, None].expand(-1, 2, -1, -1),
            2,
            indices[..., None].expand(-1, -1, -1, 3),
        )
        - object_points[:, :, None, :],
        dim=-1,
    )
    assert interaction.shape == (1, 2, 16)
    assert torch.equal(diagnostics["edge_indices"], indices)
    assert torch.allclose(diagnostics["edge_distances"], expected_distances)
    assert torch.equal(
        diagnostics["edge_valid_mask"],
        expected_distances <= 0.02,
    )
    assert torch.isfinite(interaction).all()
    edge_mask = torch.ones_like(indices, dtype=torch.bool)
    edge_mask[0, 0] = False
    _, masked_diagnostics = module(
        object_features,
        object_points,
        object_normals,
        hand_points,
        hand_normals,
        hand_flow,
        hand_valid,
        indices,
        edge_mask,
    )
    assert not masked_diagnostics["edge_valid_mask"][0, 0, 0]


def test_v1_3_collate_pads_only_to_batch_max(tmp_path: Path) -> None:
    sequence, index = _write_v1_3_sequence(tmp_path)
    entries = _resolve_index_entries(index, "train")
    dataset = ObjectInteractionCmDataset(
        Path("."),
        sequence_entries=entries,
        num_obj_points=4,
        num_hand_points=1538,
        max_hand_points=1538,
        max_knn_hand_points=5,
        hand_stream_mode="unique_knn_edges",
        fixed_stride=1,
        active_only=False,
    )
    first = dataset[0]
    second = dict(first)
    second["hand_points"] = first["hand_points"][:2]
    second["hand_normals"] = first["hand_normals"][:2]
    second["hand_flow"] = first["hand_flow"][:2]
    second["hand_valid_mask"] = first["hand_valid_mask"][:2]
    second["hand_supervision_mask"] = first["hand_supervision_mask"][:2]
    second["hand_point_ids"] = first["hand_point_ids"][:2]
    second["hand_valid_points"] = torch.tensor(2, dtype=torch.int64)
    batch = _collate_object_interaction_cm([first, second])
    assert batch["hand_points"].shape == (2, 5, 3)
    assert batch["hand_valid_points"].tolist() == [5, 2]
    assert batch["hand_valid_mask"][1].tolist() == [True, True, False, False, False]
    assert batch["hand_point_ids"][1].tolist() == [0, 1, -1, -1, -1]


def test_v1_3_scales_keep_decoder_and_knn_flow_separate(tmp_path: Path) -> None:
    sequence, index = _write_v1_3_sequence(tmp_path)
    output = tmp_path / "scales.json"
    payload = calibrate(
        index,
        output,
        max_sequences_per_source=0,
        frames_per_sequence_stride=2,
        radius_m=0.02,
        knn_k=32,
        seed=42,
        grab_strides=(1,),
        inspire_strides=(1,),
    )
    assert output.is_file()
    assert payload["scales"]["s_hand_flow"] > 0.0
    assert payload["scales"]["s_knn_hand_flow"] > 0.0
    assert payload["scales"]["s_obj_flow"] > 0.0
    assert payload["scales"]["s_geo"] > 0.0
    meta = SimpleNamespace(
        work_version="V1.3",
        processing_dim=16,
        cm_dim=8,
        num_cm_tokens=2,
        slot_iters=1,
        knn_k=32,
        interaction_radius_m=0.02,
        geometry_scale_m=0.02,
        hand_flow_input_scale=2.0,
        knn_hand_flow_input_scale=7.0,
        object_flow_target_scale=3.0,
    )
    model = ObjectInteractionCmModel(SimpleNamespace(meta=meta))
    assert model.scale_values["s_hand_flow"] == 2.0
    assert model.scale_values["s_knn_hand_flow"] == 7.0
    assert model.local_interaction.hand_flow_scale == 7.0
    assert sequence.is_dir()
