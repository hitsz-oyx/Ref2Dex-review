import torch

from src.task.InteractionDynamics.interaction_autoencoder import InteractionAutoencoder


def test_interaction_autoencoder_shape_and_input_gradient():
    model = InteractionAutoencoder(slots=4, dim=32, heads=4)
    model.requires_grad_(False)
    anchors = torch.randn(2, 7, 3, requires_grad=True)
    r = torch.randn(2, 9, 7, 3, requires_grad=True)
    d = torch.rand(2, 9, 7, requires_grad=True)
    u = torch.randn(2, 8, 7, 3, requires_grad=True)
    output = model(anchors, r, d, u)
    assert output["latent"].shape == (2, 4, 32)
    assert output["relative_geometry"].shape == r.shape
    assert output["relative_distance"].shape == d.shape
    assert output["relative_motion"].shape == u.shape
    sum(value.square().mean() for key, value in output.items() if key != "latent").backward()
    assert r.grad is not None and r.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
