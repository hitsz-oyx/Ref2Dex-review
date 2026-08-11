import torch

from src.task.InteractionDynamics.eval_interaction_compression import pack_y, unpack_y


def test_pack_unpack_y_teacher_v1():
    y = {
        "relative_geometry": torch.randn(9, 128, 3),
        "relative_distance": torch.rand(9, 128),
        "relative_motion": torch.randn(8, 128, 3),
    }
    restored = unpack_y(pack_y(y))
    for name in y:
        torch.testing.assert_close(restored[name], 100 * y[name])
    assert pack_y(y).shape == (7680,)
