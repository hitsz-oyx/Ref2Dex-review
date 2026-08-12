import numpy as np
import torch

from src.task.InteractionDynamics.goal_interaction_diffusion import (
    GoalInteractionDiffusion, extract_goal, meaningful_motion_mask,
    sample_goal_ddim, stratified_frames)


def poses(translations):
    result = torch.eye(4).repeat(len(translations), 1, 1)
    result[:, 0, 3] = torch.tensor(translations)
    return result


def test_goal_extraction_finds_next_meaningful_motion():
    trajectory = poses([0, 0, 0, 0, .004, .008, .012, .016, .020])
    assert meaningful_motion_mask(trajectory, 3, .2, 1.).tolist()[:2] == [False, True]
    goal, onset = extract_goal(trajectory, current=0, active=True, segment=4)
    assert onset == 1 and goal.shape == (25,) and goal[0] == 1
    inactive, onset = extract_goal(trajectory, current=0, active=False, segment=4)
    assert onset == -1 and torch.count_nonzero(inactive) == 0


def test_stratified_frames_cover_active_and_inactive():
    selected = stratified_frames(100, 20, 75, horizon=4, count=32)
    labels = np.asarray([(20 <= value <= 75) for value in selected])
    assert len(selected) == 32 and labels.any() and (~labels).any()


def test_goal_diffusion_permutation_equivariance_and_sampling():
    torch.manual_seed(4)
    model = GoalInteractionDiffusion(2, 4, 32, 4, 2).eval()
    noisy, state = torch.randn(2, 7, 14), torch.randn(2, 7, 4)
    anchors, patches = torch.randn(2, 7, 3), torch.randn(2, 7, 5, 6)
    goal, timestep = torch.randn(2, 25), torch.tensor([2, 7])
    order = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    expected = model(noisy, state, anchors, patches, goal, timestep)[:, order]
    actual = model(noisy[:, order], state[:, order], anchors[:, order], patches[:, order], goal, timestep)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    sampled = sample_goal_ddim(model, state[:1], anchors[:1], patches[:1], goal[:1], 4,
                               initial_noise=torch.randn(1, 7, 14))
    assert torch.isfinite(sampled).all() and sampled.abs().max() <= 5
