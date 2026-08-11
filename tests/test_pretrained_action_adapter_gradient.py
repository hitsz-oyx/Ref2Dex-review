import torch
from torch import nn

from src.task.InteractionDynamics.model import PretrainedActionAdapter


class _FrozenFlowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 5)
        self.requires_grad_(False)

    def encode(self, patch_flow):
        return self.projection(patch_flow.mean(-2))


def test_frozen_flow_adapter_preserves_input_gradient():
    adapter = PretrainedActionAdapter.__new__(PretrainedActionAdapter)
    nn.Module.__init__(adapter)
    adapter.flow_v2 = True
    adapter.dynamic_action = _FrozenFlowEncoder()
    hand = torch.randn(1, 3, 8, 3, requires_grad=True)
    patch_index = torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    output = adapter(hand, torch.empty(1, 0, 3), patch_index)
    output.square().mean().backward()
    assert hand.grad is not None and hand.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in adapter.parameters())
