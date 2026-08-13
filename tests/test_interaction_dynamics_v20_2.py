import torch

from src.task.InteractionDynamics.field_h_realizer_v20_2 import FieldHRealizerV20_2
from src.task.InteractionDynamics.train_h_realizer_v20_2 import apply_ablation, delta_h_std


def test_realizer_shape_zero_initialization_and_gradient() -> None:
    torch.manual_seed(0); model = FieldHRealizerV20_2(dim=32, heads=4, temporal_layers=1)
    inputs = (torch.randn(2, 16, 8, 8), torch.randn(2, 16, 3),
              torch.randn(2, 16, 4, 6), torch.randn(2, 33))
    output = model(*inputs)
    assert output.shape == (2, 8, 30); torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward(); assert model.head.weight.grad is not None


def test_realizer_does_not_read_p_channel() -> None:
    model = FieldHRealizerV20_2(dim=32, heads=4, temporal_layers=1).eval()
    y = torch.randn(2, 8, 8, 8); args = (torch.randn(2, 8, 3), torch.randn(2, 8, 4, 6), torch.randn(2, 33))
    with torch.no_grad():
        first = model(y, *args); y[..., 7] += 1000; second = model(y, *args)
    torch.testing.assert_close(first, second)


def test_ablation_changes_only_requested_input() -> None:
    values = (torch.randn(3, 4, 8, 8), torch.randn(3, 4, 3),
              torch.randn(3, 4, 2, 6), torch.randn(3, 33))
    shuffled = apply_ablation(*values, "shuffle_y")
    torch.testing.assert_close(shuffled[0], values[0].roll(1, 0))
    for got, expected in zip(shuffled[1:], values[1:]): torch.testing.assert_close(got, expected)


def test_delta_h_std_has_no_mean_and_is_nonzero() -> None:
    dataset = [{"future_delta_h": torch.zeros(8, 30)}, {"future_delta_h": torch.ones(8, 30)}]
    std = delta_h_std(dataset)
    assert std.shape == (1, 1, 30); assert bool((std > 0).all())
