import torch

from src.task.InteractionDynamics.direct_h_dynamics_v20_3 import DirectHDynamicsV20_3
from src.task.InteractionDynamics.train_direct_h_v20_3 import apply_direct_ablation


def test_direct_h_shape_zero_init_and_gradient() -> None:
    model=DirectHDynamicsV20_3(dim=32,heads=4,temporal_layers=1)
    inputs=(torch.randn(2,16,8),torch.randn(2,16,3),torch.randn(2,16,4,6),torch.randn(2,33))
    output=model(*inputs);assert output.shape==(2,8,30);torch.testing.assert_close(output,torch.zeros_like(output))
    output.sum().backward();assert model.head.weight.grad is not None


def test_direct_h_anchor_permutation_invariance() -> None:
    torch.manual_seed(1);model=DirectHDynamicsV20_3(dim=32,heads=4,temporal_layers=1).eval()
    inputs=(torch.randn(2,16,8),torch.randn(2,16,3),torch.randn(2,16,4,6),torch.randn(2,33));order=torch.randperm(16)
    with torch.no_grad():normal=model(*inputs);permuted=model(inputs[0][:,order],inputs[1][:,order],inputs[2][:,order],inputs[3])
    torch.testing.assert_close(normal,permuted,atol=1e-6,rtol=1e-6)


def test_direct_h_ablation_changes_only_y() -> None:
    inputs=(torch.randn(3,8,8),torch.randn(3,8,3),torch.randn(3,8,4,6),torch.randn(3,33))
    shuffled=apply_direct_ablation(*inputs,"shuffle_y");torch.testing.assert_close(shuffled[0],inputs[0].roll(1,0))
    for got,expected in zip(shuffled[1:],inputs[1:]):torch.testing.assert_close(got,expected)
