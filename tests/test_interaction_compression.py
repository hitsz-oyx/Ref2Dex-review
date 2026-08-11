import torch

from src.task.InteractionDynamics.eval_interaction_compression import (
    _round_robin, _uniform_limit, pack_y, unpack_y)


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


def test_sequence_balancing_is_uniform_and_round_robin():
    assert _uniform_limit(list(range(10)), 3) == [0, 4, 9]
    assert _round_robin([[0, 1, 2], [10, 11], [20]], 6) == [0, 10, 20, 1, 11, 2]
