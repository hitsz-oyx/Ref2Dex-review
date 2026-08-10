import torch

from src.task.InteractionDynamics.correspondence_diagnostics import _gather_time_patches


def test_gather_time_patches_uses_step_specific_indices():
    values = torch.tensor([[[[1.], [2.]], [[3.], [4.]]]])
    index = torch.tensor([[[0, 1], [1, 0]]])
    gathered = _gather_time_patches(values, index)
    assert gathered.shape == (1, 2, 2, 1)
    assert torch.equal(gathered[..., 0], torch.tensor([[[1., 2.], [4., 3.]]]))
