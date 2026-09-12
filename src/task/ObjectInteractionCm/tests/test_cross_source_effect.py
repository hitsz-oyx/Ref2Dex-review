from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel
from src.task.ObjectInteractionCm.research.cross_source_effect.run import (
    METRIC_KEYS, cluster_estimates, decode, match_rows, masked_rms, shuffled_flow, vector_errors,
)


def test_vector_errors_use_vector_norm_and_mask():
    prediction = torch.tensor([[[.003, .004, 0.], [10., 10., 10.]]])
    valid = torch.tensor([[True, False]])
    epe, mse = vector_errors(prediction, torch.zeros_like(prediction), valid)
    assert epe.item() == pytest.approx(5)
    assert mse.item() == pytest.approx(25)
    assert masked_rms(prediction, valid).item() == pytest.approx(5)
    assert masked_rms(prediction, torch.zeros_like(valid)).item() == 0
    with pytest.raises(ValueError):
        vector_errors(prediction, prediction, torch.zeros_like(valid))


def test_spatial_shuffle_preserves_vectors_and_invalid_padding():
    flow = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    valid = torch.tensor([[True, False, True, True, False], [False] * 5])
    before = flow.clone()
    changed = shuffled_flow(flow, valid, [42, 7])
    assert torch.equal(flow, before)
    assert torch.equal(changed[~valid], flow[~valid])
    assert torch.equal(changed, shuffled_flow(flow, valid, [42, 7]))
    assert sorted(map(tuple, changed[0, valid[0]].tolist())) == sorted(map(tuple, flow[0, valid[0]].tolist()))


def test_match_balances_each_stratum_and_excludes_invalid_or_unshared():
    rows = []
    for source, count in (("grab", 5), ("inspire_f1", 3)):
        for _ in range(count):
            rows.append({"source": source, "sample_index": len(rows), "primary_valid": True,
                         "object_name": "cup", "hand_rms_mm": 2., "object_rms_mm": 1., "active_fraction": .1})
    rows += [dict(rows[0], sample_index=8, primary_valid=False),
             dict(rows[0], sample_index=9, object_name="unique")]
    selection, strata = match_rows(rows)
    assert len(selection) == 6 and len(strata) == 1
    assert strata[0]["stratum"] == ["cup", 2, 1, 3]
    assert strata[0]["per_source_count"] == 3
    assert 8 not in selection and 9 not in selection
    assert (selection, strata) == match_rows(rows)
    assert match_rows([]) == (set(), [])


def test_cluster_bootstrap_resamples_sequences_not_frames():
    rows = []
    for name, count, value in (("a", 9, 0.), ("b", 1, 10.)):
        rows.extend(dict({key: value for key in METRIC_KEYS}, sequence_id=name) for _ in range(count))
    stats, boot = cluster_estimates(rows, repeats=2000)
    assert stats["frame_micro"]["epe_mm"] == 1
    assert stats["sequence_macro"]["epe_mm"] == 5
    assert set(boot[:, 0]) == {0., 1., 10.}
    assert stats["frame_micro_cluster_ci95"]["epe_mm"] == [0., 10.]
    assert stats["frame_micro"]["rmse_mm"] == 1
    assert cluster_estimates([]) == (None, None)


def test_frozen_decoder_replay_and_zero_token_ablation():
    torch.manual_seed(42)
    model = ObjectInteractionCmModel(SimpleNamespace(meta=SimpleNamespace(
        processing_dim=16, cm_dim=8, num_cm_tokens=3, slot_iters=2, knn_k=2,
        interaction_radius_m=.02, object_flow_target_scale=.01)))
    model.requires_grad_(False).eval()
    batch = {"obj_points": torch.randn(2, 5, 3) * .001, "obj_normals": torch.randn(2, 5, 3),
             "obj_valid_mask": torch.ones(2, 5, dtype=torch.bool),
             "hand_points": torch.randn(2, 3, 3) * .001, "hand_normals": torch.randn(2, 3, 3),
             "hand_flow": torch.randn(2, 3, 3) * .002,
             "hand_valid_mask": torch.tensor([[True] * 3, [False] * 3]),
             "knn_edge_indices": torch.zeros(2, 5, 2, dtype=torch.long),
             "knn_edge_valid_mask": torch.tensor([[[True] * 2] * 5, [[False] * 2] * 5])}
    state = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        result = model(batch)
        torch.testing.assert_close(decode(model, batch, result), result["pred_obj_flow"])
        anchors = result["cm_anchor_pos"].clone()
        zero = decode(model, batch, result, zero_tokens=True)
        assert torch.equal(result["cm_anchor_pos"], anchors)
        assert torch.isfinite(zero).all()
        assert result["sample_valid"].tolist() == [True, False]
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
