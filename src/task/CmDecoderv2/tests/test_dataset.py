from __future__ import annotations

import json

import numpy as np
import torch

from src.task.CmDecoderv2.dataset import CmDecoderV2Dataset, _collate_cm_decoder


def test_pilot_view_window_contract_and_shapes() -> None:
    view = json.loads(open("data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/index.json", encoding="utf-8").read())
    assert view["split_contract"] == {"train": "inspire_rl", "val": "inspire_rl", "test": "mano_qualitative_only"}
    dataset = CmDecoderV2Dataset(
        view["sequences"]["train"],
        urdf_path="src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf",
        window_size=4,
        num_obj_points=64,
        num_hand_points=1538,
        perturb=False,
    )
    sequence = dataset.sequences[0]
    assert len(dataset) == sequence.frame_count - 4
    sample = dataset[0]
    assert sample["obj_points"].shape == (4, 64, 3)
    assert sample["hand_flow"].shape == (4, 1538, 3)
    assert sample["target_q_delta"].shape == (4, 6)
    assert sample["target_wrist_translation"].shape == (4, 3)
    assert sample["target_wrist_rotation"].shape == (4, 3, 3)
    assert sample["target_hand_points_object"].shape == (4, 1538, 3)
    assert sample["current_wrist_pose_world"].shape == (4, 4)
    assert sample["object_pose_world"].shape == (4, 4)
    assert sample["active_mask"].shape == (4,)
    assert sample["active_mask"].dtype == torch.bool
    assert np.all(np.diff(np.load(view["sequences"]["train"][0]["window_source_frame_ids"])[0]) == 4)


def test_v13_unique_knn_edges_dynamic_padding_and_full_target() -> None:
    view = json.loads(
        open(
            "data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135_pilot/index.json",
            encoding="utf-8",
        ).read()
    )
    assert view["source_index_schema"] == "ref2dex_object_interaction_cm_index_v1_2"
    assert view["source_cache_schema"] == "ref2dex_object_interaction_cm_dexplore_rl_v1_3"
    assert view["cm_input_contract"] == {
        "hand_stream_mode": "unique_knn_edges",
        "knn_k": 32,
        "interaction_radius_m": 0.02,
        "hand_supervision_radius_m": 0.02,
        "distance_storage": "runtime_recompute_from_cached_32_indices",
    }
    assert view["point_flow_supervision"] == {
        "target_file": "knn_hand_points_world.npy",
        "hand_points": 10135,
        "all_points": True,
        "mask": "none",
        "coordinate_frame": "object_pose_t",
    }
    dataset = CmDecoderV2Dataset(
        view["sequences"]["train"],
        urdf_path="src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf",
        window_size=4,
        num_obj_points=64,
        num_hand_points=10135,
        hand_stream_mode="unique_knn_edges",
        knn_k=32,
        hand_supervision_radius_m=0.02,
        perturb=False,
        active_only=True,
    )
    first, second = dataset[0], dataset[1]
    assert first["target_hand_points_object"].shape == (4, 10135, 3)
    assert second["target_hand_points_object"].shape == (4, 10135, 3)
    for sample in (first, second):
        assert sample["hand_points"].shape[0] == 4
        assert sample["hand_points"].shape[2] == 3
        assert sample["hand_normals"].shape == sample["hand_points"].shape
        assert sample["hand_flow"].shape == sample["hand_points"].shape
        assert sample["hand_valid_mask"].shape[:1] == (4,)
        assert sample["knn_edge_indices"].shape == (4, 64, 32)
        assert sample["knn_edge_valid_mask"].shape == (4, 64, 32)
        assert sample["knn_edge_indices"].dtype == torch.int64
        assert sample["cm_hand_valid_points"].shape == (4,)
        assert torch.equal(
            sample["hand_valid_mask"].sum(dim=1),
            sample["cm_hand_valid_points"],
        )
        gathered_valid = torch.gather(
            sample["hand_valid_mask"][:, None].expand(-1, 64, -1),
            2,
            sample["knn_edge_indices"],
        )
        assert gathered_valid[sample["knn_edge_valid_mask"]].all()
        assert sample["knn_edge_indices"].max().item() < sample["hand_points"].shape[1]

    assert first["hand_points"].shape[1] < second["hand_points"].shape[1]
    batch = _collate_cm_decoder([first, second])
    assert batch["hand_points"].shape == (2, 4, second["hand_points"].shape[1], 3)
    assert batch["target_hand_points_object"].shape == (2, 4, 10135, 3)
    assert torch.equal(
        batch["hand_valid_mask"].sum(dim=2),
        torch.stack([first["cm_hand_valid_points"], second["cm_hand_valid_points"]]),
    )
