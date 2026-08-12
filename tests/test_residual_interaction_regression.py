import torch

from src.task.InteractionDynamics.residual_interaction_regression import (
    ResidualInteractionRegressor, persistence_future, residual_target)
from src.task.InteractionDynamics.train_residual_interaction_regression import intervene_goal


def test_residual_reconstructs_future():
    state = torch.randn(2, 128, 4)
    future = torch.randn(2, 128, 28)
    persistence = persistence_future(state, 4)
    assert torch.allclose(persistence + residual_target(state, future, 4), future, atol=1e-6)


def test_shape_and_anchor_permutation_equivariance():
    torch.manual_seed(0)
    model = ResidualInteractionRegressor(4, 4, 32, 4, 2).eval()
    state, anchors = torch.randn(2, 128, 4), torch.randn(2, 128, 3)
    patches, goal = torch.randn(2, 128, 32, 6), torch.randn(2, 25)
    permutation = torch.randperm(128)
    prediction = model(state, anchors, patches, goal)
    permuted = model(state[:, permutation], anchors[:, permutation], patches[:, permutation], goal)
    assert prediction.shape == (2, 128, 28)
    assert torch.allclose(permuted, prediction[:, permutation], atol=1e-5)


def test_goal_interventions_are_in_physical_space():
    goal = torch.randn(3, 25)
    goal[:, 0] = torch.tensor([1., 1., 0.])
    zero = intervene_goal(goal, "motion_zero")
    reverse = intervene_goal(goal, "motion_reverse")
    active_zero = intervene_goal(goal, "active_zero")
    assert torch.equal(zero[:2, 1:], torch.zeros_like(zero[:2, 1:]))
    assert torch.equal(reverse[:2, 1:], -goal[:2, 1:])
    assert torch.equal(active_zero[:2], torch.zeros_like(active_zero[:2]))
    assert torch.equal(active_zero[2], goal[2])
