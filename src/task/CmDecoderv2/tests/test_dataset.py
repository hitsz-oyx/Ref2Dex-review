from __future__ import annotations

import json

import numpy as np

from src.task.CmDecoderv2.dataset import CmDecoderV2Dataset


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
    assert np.all(np.diff(np.load(view["sequences"]["train"][0]["window_source_frame_ids"])[0]) == 4)
