from __future__ import annotations

import torch
import numpy as np

from src.task.CmDecoderv2.field_realizer import (
    DirectManoHRealizer,
    InspireFieldRealizer,
    MatchedTemporalD2Core,
    build_f7_field,
    spatial_shuffle_f7,
    zero_f7,
)
from src.task.CmDecoderv2.research.field_realizer_gate.contact_logging import extract_hand_object_contacts
from src.task.CmDecoderv2.field_dataset import DirectManoHDataset, FieldRealizerDataset
from src.task.InteractionDynamics.field_state_v20 import build_causal_field


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(11)
    f7 = torch.randn(batch, 4, 128, 7, generator=generator)
    pos = torch.randn(batch, 4, 128, 3, generator=generator)
    normal = torch.randn(batch, 4, 128, 3, generator=generator)
    state = torch.randn(batch, 15, generator=generator)
    links = torch.randn(batch, 18, 10, generator=generator)
    return f7, pos, normal, state, links


def test_f7_matches_v20_without_penetration_channel() -> None:
    generator = torch.Generator().manual_seed(4)
    points = torch.randn(6, 17, 3, generator=generator)
    anchors = torch.randn(128, 3, generator=generator)
    penetration = torch.randn(6, 17, generator=generator)
    expected = build_causal_field(points, anchors, penetration)[..., :7]
    actual = build_f7_field(points, anchors)
    torch.testing.assert_close(actual, expected)


def test_interventions_preserve_contract() -> None:
    f7, *_ = _inputs(batch=2)
    assert torch.count_nonzero(zero_f7(f7)) == 0
    permutation = torch.stack([torch.randperm(128, generator=torch.Generator().manual_seed(i)) for i in (1, 2)])
    shuffled = spatial_shuffle_f7(f7, permutation)
    for batch in range(2):
        torch.testing.assert_close(shuffled[batch], f7[batch, :, permutation[batch]])


def test_matched_core_and_realizers_have_identical_action_shapes() -> None:
    values = _inputs()
    field = InspireFieldRealizer(dropout=0.0)
    field_output = field(*values)
    direct = DirectManoHRealizer(dropout=0.0)
    h = torch.randn(2, 4, 43, generator=torch.Generator().manual_seed(8))
    direct_output = direct(h, values[-2], values[-1])
    for output in (field_output, direct_output):
        assert output["pred_q_delta"].shape == (2, 4, 6)
        assert output["pred_wrist_translation"].shape == (2, 4, 3)
        assert output["pred_wrist_rotvec"].shape == (2, 4, 3)
        assert torch.isfinite(output["pred_q_delta"]).all()


def test_realizer_heads_start_at_identity_action() -> None:
    values = _inputs(batch=1)
    model = InspireFieldRealizer(dropout=0.0)
    output = model(*values)
    assert torch.count_nonzero(output["pred_q_delta"]) == 0
    assert torch.count_nonzero(output["pred_wrist_translation"]) == 0
    assert torch.count_nonzero(output["pred_wrist_rotvec"]) == 0


def test_shape_guard_rejects_wrong_f7_anchor_count() -> None:
    values = _inputs(batch=1)
    model = InspireFieldRealizer(dropout=0.0)
    try:
        model(values[0][:, :, :127], values[1][:, :, :127], values[2][:, :, :127], values[3], values[4])
    except ValueError as error:
        assert "Expected source shape" in str(error)
    else:
        raise AssertionError("wrong anchor count must be rejected")


def test_pairwise_contact_filter_does_not_use_net_force() -> None:
    dtype = np.dtype([("body0", "i4"), ("body1", "i4"), ("force", "f4", (3,))])
    records = np.array([(1, 9, (0.0, 0.0, 2.0)), (2, 3, (0.0, 0.0, 7.0))], dtype=dtype)
    result = extract_hand_object_contacts(records, hand_body_ids=(1, 2), object_body_ids=(9,))
    assert result["records"].shape == (1,)
    assert result["mask"].tolist() == [True, False]
    assert result["normal_force"].tolist() == [2.0]


def test_parent_f7_dataset_contract() -> None:
    import json
    index = json.loads(open("data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/index.json", encoding="utf-8").read())
    dataset = FieldRealizerDataset(
        index["sequences"]["train"][:1],
        field_root="data/processed_data/cm_decoder_v2/field_f7_parent_v1_1_16_all",
        urdf_path="src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf",
    )
    sample = dataset[0]
    assert sample["f7"].shape == (4, 128, 7)
    assert sample["anchor_pos"].shape == (4, 128, 3)
    assert sample["current_link_features"].shape == (18, 10)
    assert sample["target_q_delta"].shape == (4, 6)
    assert sample["active_mask"].shape == (4,)
    assert all(torch.isfinite(value).all() for value in sample.values() if torch.is_floating_point(value))


def test_parent_mano_h_dataset_contract() -> None:
    import json
    index = json.loads(open("data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/index.json", encoding="utf-8").read())
    dataset = DirectManoHDataset(
        index["sequences"]["train"][:3],
        field_root="data/processed_data/cm_decoder_v2/field_f7_parent_v1_1_16_all",
        mano_h_root="data/processed_data/cm_decoder_v2/field_mano_h_parent_smoke_v116_20260914",
        urdf_path="src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf",
    )
    sample = dataset[0]
    assert sample["mano_h"].shape == (4, 43)
    assert torch.isfinite(sample["mano_h"]).all()
