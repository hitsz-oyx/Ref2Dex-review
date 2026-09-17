import torch
from types import SimpleNamespace

from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2Model, swept_topk
from src.task.ObjectInteractionCmv2.synthetic import make_synthetic_batch


def swept_model():
    return ObjectInteractionCmv2Model(SimpleNamespace(interaction_mode="swept"))


def brute_swept(object_points, hand_points, hand_flow):
    rel = hand_points[:, None] - object_points[:, :, None]
    alpha = (-(rel * hand_flow[:, None]).sum(-1) /
             hand_flow.square().sum(-1)[:, None].clamp_min(1e-12)).clamp(0, 1)
    return torch.linalg.vector_norm(rel + alpha[..., None] * hand_flow[:, None], dim=-1)


def test_middle_of_sweep_beats_current_knn_and_zero_flow():
    obj = torch.zeros(1, 1, 3)
    hand = torch.tensor([[[0.1, 0., 0.], [0.01, 0., 0.]]])
    flow = torch.tensor([[[-0.2, 0., 0.], [0., 0., 0.]]])
    distance, index = swept_topk(obj, hand, flow, k=2, object_chunk=1, hand_chunk=1)
    assert index[0, 0].tolist() == [0, 1]
    assert torch.allclose(distance[0, 0], torch.tensor([0., 0.01]), atol=1e-6)


def test_blocked_topk_matches_brute_with_ties_and_mask():
    torch.manual_seed(2)
    obj = torch.rand(2, 7, 3)
    hand = torch.rand(2, 19, 3)
    flow = torch.rand(2, 19, 3) * 0.1
    hand[:, 1] = hand[:, 0]
    flow[:, 1] = flow[:, 0]
    mask = torch.ones(2, 19, dtype=torch.bool)
    mask[:, -2:] = False
    distances, indices = swept_topk(obj, hand, flow, mask, k=5, object_chunk=3, hand_chunk=4)
    brute = brute_swept(obj, hand, flow).masked_fill(~mask[:, None], float("inf"))
    expected = torch.argsort(brute, dim=-1, stable=True)[..., :5]
    assert torch.equal(indices, expected)
    assert torch.allclose(distances, torch.gather(brute, -1, expected), atol=1e-6)


def test_hard_radius_and_empty_contact_have_finite_gradients():
    batch = make_synthetic_batch(batch_size=1, num_object=6, num_hand=8)
    batch["hand_points"] = batch["hand_points"] * 0 + torch.tensor([0.02, 0., 0.])
    batch["hand_flow"] = torch.zeros_like(batch["hand_flow"])
    batch["obj_points"] = torch.zeros_like(batch["obj_points"])
    batch["delta_time_s"] = torch.tensor([1 / 30])
    model = swept_model()
    output = model(batch)
    assert not output["contact_active"].any()
    assert torch.all(output["contact_features"] == 0)
    output["obj_flow_pred"].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    batch["hand_points"][:, 0, 0] = 0.019
    assert model(batch)["contact_active"].all()


def test_future_object_target_is_not_a_model_input():
    batch = make_synthetic_batch(batch_size=1, num_object=6, num_hand=8)
    batch["delta_time_s"] = torch.tensor([1 / 30])
    model = swept_model().eval()
    with torch.no_grad():
        first = model(batch)["obj_flow_pred"]
        batch["obj_flow_gt"] += 100
        second = model(batch)["obj_flow_pred"]
    assert torch.equal(first, second)


def test_all_padded_hands_remain_finite():
    batch = make_synthetic_batch(batch_size=1, num_object=6, num_hand=8)
    batch["hand_valid_mask"][:] = False
    output = swept_model()(batch)
    assert not output["contact_active"].any()
    assert torch.isfinite(output["obj_flow_pred"]).all()
    assert torch.all(output["contact_features"] == 0)
