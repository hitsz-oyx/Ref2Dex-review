"""V1.12 static Cmv2 adapter contract, independent of the real checkpoint."""
import hashlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.task.CmResidual.cm_v2_adapter import (CONTEXT_DIM, MODEL_CONFIG,
                                                SCHEMA, FrozenCmv2Adapter,
                                                encode_context)
from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2V13Model


class _RecordingProfiler:
    def __init__(self):
        self.names = []

    @contextmanager
    def stage(self, name):
        self.names.append(name)
        yield


def _object_points():
    x = torch.linspace(-0.05, 0.05, 1024)
    return torch.stack((x, x.square(), x * 0), -1)[None]


def test_context_layout_mask_and_scale():
    obj = _object_points()
    center = obj.mean(1)[0]
    output = {"cm_tokens": torch.ones(1, 16, 32),
              "token_anchors": center.repeat(16, 1)[None],
              "token_normals": torch.tensor([0., 0., 2.]).repeat(1, 16, 1),
              "token_mass": torch.full((1, 16), 1024.),
              "token_mask": torch.tensor([[True] + [False] * 15])}
    output["cm_tokens"][0, 1:] = float("nan")
    context = encode_context(output, obj)
    assert context.shape == (1, CONTEXT_DIM)
    torch.testing.assert_close(context[0, :40], torch.tensor([1.] * 32 + [0.] * 3 + [0., 0., 1., 1., 1.]))
    assert torch.equal(context[0, 40:], torch.zeros(CONTEXT_DIM - 40))
    output["token_mass"][0, 0] = -1
    with pytest.raises(ValueError, match="Negative"):
        encode_context(output, obj)


def test_checkpoint_requires_sha_architecture_and_strict_state(tmp_path: Path):
    model = ObjectInteractionCmv2V13Model(SimpleNamespace(**MODEL_CONFIG))
    checkpoint = tmp_path / "synthetic.pt"

    def write(payload):
        torch.save(payload, checkpoint)
        return hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    sha = write({"architecture_version": "v1_3_rigid_only", "model": model.state_dict()})
    adapter = FrozenCmv2Adapter(checkpoint, sha, "cpu")
    assert not adapter.model.training
    assert not any(p.requires_grad for p in adapter.model.parameters())
    object_points = _object_points()
    object_normals = torch.zeros_like(object_points)
    object_normals[..., 2] = 1.0
    hand_points = torch.zeros(1, 1538, 3)
    hand_normals = torch.zeros_like(hand_points)
    hand_normals[..., 2] = 1.0
    output = adapter.predict(object_points, object_normals, hand_points, hand_normals,
                             torch.zeros_like(hand_points), 1 / 30)
    assert output["delta_xi_root"].shape == (1, 6)
    assert output["obj_flow_pred"].shape == (1, 1024, 3)
    assert output["cm_context"].shape == (1, CONTEXT_DIM)
    assert torch.isfinite(output["obj_flow_pred"]).all()
    actor_output = adapter.predict(object_points, object_normals, hand_points, hand_normals,
                                   torch.zeros_like(hand_points), 1 / 30,
                                   include_context=False)
    assert "cm_context" not in actor_output
    assert actor_output["cm_tokens"].shape == (1, 16, 32)
    effect = adapter.predict_effect_only(object_points, object_normals, hand_points, hand_normals,
                                         torch.zeros_like(hand_points), 1 / 30,
                                         interaction_object_chunk=32)
    assert set(effect) == {"delta_xi_root", "token_mask", "token_mass"}
    torch.testing.assert_close(effect["delta_xi_root"], actor_output["delta_xi_root"], atol=1e-6, rtol=1e-6)
    assert torch.equal(effect["token_mask"], actor_output["token_mask"])
    torch.testing.assert_close(effect["token_mass"], actor_output["token_mass"], atol=1e-6, rtol=1e-6)
    profiler = _RecordingProfiler()
    profiled = adapter.predict_effect_only(
        object_points, object_normals, hand_points, hand_normals,
        torch.zeros_like(hand_points), 1 / 30,
        interaction_object_chunk=32, latency_profiler=profiler)
    assert profiler.names == ["geometry_encoder", "swept_topk", "edge_contact", "token_attention_effect"]
    for name in effect:
        torch.testing.assert_close(profiled[name], effect[name])
    with pytest.raises(ValueError, match="SHA256"):
        FrozenCmv2Adapter(checkpoint, "0" * 64, "cpu")
    sha = write({"architecture_version": "v1_2", "model": model.state_dict()})
    with pytest.raises(ValueError, match="architecture_version"):
        FrozenCmv2Adapter(checkpoint, sha, "cpu")
    sha = write({"architecture_version": "v1_3_rigid_only", "model": {}})
    with pytest.raises(RuntimeError):
        FrozenCmv2Adapter(checkpoint, sha, "cpu")


def test_action_eval_config_exposes_frozen_cmv2_contract():
    root = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg")
    task = yaml.safe_load((root / "task/CmResidualDexYCBCmv2ActionEval.yaml").read_text())
    assert task["env"]["numObservations"] == 1442
    assert task["basePolicy"]["useCmv2Context"] is False
    assert task["basePolicy"]["useCmv2ActionEvaluator"] is True
    assert task["basePolicy"]["cmv2Schema"] == SCHEMA
    assert task["basePolicy"]["cmv2ModelConfig"] == MODEL_CONFIG
    assert task["reference"]["allowIneligibleFor"] == "diagnostic"
