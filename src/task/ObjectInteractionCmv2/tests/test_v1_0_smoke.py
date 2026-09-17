import torch

from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2Model, transform_points, object_interaction_loss
from src.task.ObjectInteractionCmv2.synthetic import make_synthetic_batch


def test_rigid_transform_identity_and_translation():
    points = torch.zeros(2, 3, 3)
    xi = torch.zeros(2, 6)
    xi[:, 0] = 0.1
    assert torch.allclose(transform_points(points, xi)[..., 0], torch.full((2, 3), 0.1), atol=1e-6)


def test_v1_model_forward_and_backward():
    batch = make_synthetic_batch(batch_size=2, num_object=24, num_hand=32)
    model = ObjectInteractionCmv2Model()
    output = model(batch)
    assert output["tokens"].shape == (2, 16, 128)
    assert output["obj_flow_pred"].shape == (2, 24, 3)
    losses = object_interaction_loss(output, batch)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_no_contact_is_explicitly_zeroed():
    batch = make_synthetic_batch(batch_size=1, num_object=12, num_hand=8)
    batch["hand_points"] = batch["hand_points"] + 10.0
    model = ObjectInteractionCmv2Model()
    output = model(batch)
    assert not output["contact_active"].any()
    assert torch.allclose(output["contact_features"], torch.zeros_like(output["contact_features"]))
    assert output["num_links"].shape == (1,) if "num_links" in output else True
