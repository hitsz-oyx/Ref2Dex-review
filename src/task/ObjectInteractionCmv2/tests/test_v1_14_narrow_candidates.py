from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.task.ObjectInteractionCmv2.compact_endpoint import pack_compact_endpoint, unpack_compact_endpoint
from src.task.ObjectInteractionCmv2.part_se3 import collate_part_se3, part_se3_v112_loss
from src.task.ObjectInteractionCmv2.part_se3_v114 import (
    PART_SE3_V114_VERSION,
    PartSE3ObjectInteractionCmv2V114Model,
)
from src.task.ObjectInteractionCmv2.part_se3_v114_training import (
    CHECKPOINT_SCHEMA,
    checkpoint_payload_v114,
    initialize_random_v114_model,
    load_v114_config,
    restore_checkpoint_v114,
)
from src.task.ObjectInteractionCmv2.tests.test_v1_12_part_se3 import _sample


def _model(interaction_dim: int = 8) -> PartSE3ObjectInteractionCmv2V114Model:
    return PartSE3ObjectInteractionCmv2V114Model(SimpleNamespace(
        hidden_width=16,
        interaction_dim=interaction_dim,
        num_tokens=16,
        knn_k=32,
        interaction_radius_m=0.02,
        feature_scale_m=0.02,
        frame_dt_s=1 / 30,
    ))


def _candidate_batch(batch: dict, candidates: int) -> dict[str, torch.Tensor]:
    keys = ("hand_points", "hand_normals", "hand_flow", "hand_valid_mask", "delta_time_s")
    result = {key: batch[key][:, None].expand(
        batch[key].shape[0], candidates, *batch[key].shape[1:]).clone() for key in keys}
    for candidate in range(candidates):
        result["hand_flow"][:, candidate, :, 0] += candidate * 0.0002
    return result


def test_k1_wrapper_matches_candidate_api_and_backward_is_finite():
    torch.manual_seed(14)
    batch = collate_part_se3([_sample(2, interacting_parts=(0,))])
    model = _model()
    context = model.encode_object(batch)
    candidate = _candidate_batch(batch, 1)
    direct = model(batch)
    shared = model.forward_candidates(context, candidate)
    for key in ("delta_xi_part", "obj_flow_pred", "contact_features", "token_assignment"):
        assert torch.allclose(direct[key], shared[key][:, 0], atol=1e-6, rtol=1e-6)
    loss = part_se3_v112_loss(direct, batch)["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_candidate_axis_matches_independent_calls_and_is_permutation_equivariant():
    torch.manual_seed(15)
    batch = collate_part_se3([
        _sample(2, interacting_parts=(0,)),
        _sample(3, interacting_parts=(0, 2)),
    ])
    model = _model().eval()
    calls = []
    hook = model.geometry_encoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        context = model.encode_object(batch)
        candidate = _candidate_batch(batch, 3)
        together = model.forward_candidates(context, candidate)
        assert len(calls) == 1
        individual = []
        for index in range(3):
            one = {key: value[:, index:index + 1] for key, value in candidate.items()}
            individual.append(model.forward_candidates(context, one)["delta_xi_part"])
        stacked = torch.cat(individual, dim=1)
        assert torch.allclose(together["delta_xi_part"], stacked, atol=2e-6, rtol=2e-6)

        order = torch.tensor([2, 0, 1])
        permuted = {key: value[:, order] for key, value in candidate.items()}
        reordered = model.forward_candidates(context, permuted)
        assert torch.allclose(
            reordered["delta_xi_part"], together["delta_xi_part"][:, order], atol=2e-6, rtol=2e-6)
    hook.remove()


def test_reference_and_v113_compact_inputs_match_with_narrow_encoder():
    torch.manual_seed(16)
    sample = _sample(2, interacting_parts=(0,))
    sample.update({
        "obj_point_id": torch.arange(sample["obj_points"].shape[0], dtype=torch.int64),
        "stride": torch.tensor(1, dtype=torch.int64),
        "source_frame_id": torch.tensor(100, dtype=torch.int64),
        "next_source_frame_id": torch.tensor(101, dtype=torch.int64),
    })
    compact = unpack_compact_endpoint(pack_compact_endpoint(sample))
    model = _model().eval()
    with torch.no_grad():
        reference = model(collate_part_se3([sample]))
        accelerated = model(collate_part_se3([compact]))
    for key in ("edge_indices", "edge_valid_mask"):
        assert torch.equal(reference[key], accelerated[key])
    for key in ("edge_distances", "contact_features", "delta_xi_part", "obj_flow_pred"):
        assert torch.allclose(reference[key], accelerated[key], atol=1e-6, rtol=1e-6)


def test_v114_config_and_checkpoint_are_random_only_and_reject_v112(tmp_path):
    config_path = Path(__file__).parents[1] / "configs/active/mixed_part_se3_v1_14.yaml"
    config = load_v114_config(config_path)
    assert config["data_backend"] == "reference"
    model = initialize_random_v114_model(config)
    assert model.architecture_version == PART_SE3_V114_VERSION
    assert model.interaction_width == 32

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    payload = checkpoint_payload_v114(model, optimizer, epoch=1, step=3, best_metric=0.4)
    assert payload["schema_name"] == CHECKPOINT_SCHEMA
    valid = tmp_path / "v114.pt"
    torch.save(payload, valid)
    assert restore_checkpoint_v114(valid, model) == {"epoch": 1, "step": 3, "best_metric": 0.4}

    payload["architecture_version"] = "v1_12_endpoint_part_se3"
    old = tmp_path / "v112.pt"
    torch.save(payload, old)
    with pytest.raises(ValueError, match="not a V1.14"):
        restore_checkpoint_v114(old, model)


def test_v114_plan_and_pointer_are_declared():
    repo = Path(__file__).parents[4]
    assert (repo / "src/task/ObjectInteractionCmv2/docs/plan/V1.14.md").is_file()
    assert "work_version: V1.14" in (repo / "docs/current_versions.yaml").read_text()
