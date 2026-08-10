from types import SimpleNamespace

import torch

from src.task.Actiontoken.generator import sample_pose_pairs, sample_smooth_pose_trajectory
from src.task.Actiontoken.model import DynamicActionEncoder, FlowActionEncoder
from src.task.Actiontoken.probes import OddProbe


def test_smooth_trajectory_is_deterministic_and_temporally_correlated():
    trajectory = sample_smooth_pose_trajectory(4, 24, 9, 7, .9, .035)
    repeated = sample_smooth_pose_trajectory(4, 24, 9, 7, .9, .035)
    torch.testing.assert_close(trajectory, repeated)
    velocity = trajectory[:, 1:] - trajectory[:, :-1]
    adjacent_cosine = torch.nn.functional.cosine_similarity(velocity[:, 1:], velocity[:, :-1], dim=-1)
    assert adjacent_cosine.mean() > .5


def test_dynamic_action_shapes_and_frozen_pose_independent_interface():
    encoder = DynamicActionEncoder(dim=24, global_dim=8, heads=4, layers=1, patch_size=5)
    output = encoder(torch.randn(2, 9, 3, 24), torch.randn(2, 9, 8))
    assert output["action_tokens"].shape == (2, 8, 3, 24)
    assert output["pred_dense_flow_internal"].shape == (2, 8, 3, 5, 3)


def test_static_pair_is_zero_and_swapped_pair_reverses_flow():
    encoder = DynamicActionEncoder(dim=24, global_dim=8, heads=4, layers=1, patch_size=5).eval()
    first, second = torch.randn(2, 1, 3, 24), torch.randn(2, 1, 3, 24)
    global_first, global_second = torch.randn(2, 1, 8), torch.randn(2, 1, 8)
    static = encoder(torch.cat([first, first], 1), torch.cat([global_first, global_first], 1))
    forward = encoder(torch.cat([first, second], 1), torch.cat([global_first, global_second], 1))
    reverse = encoder(torch.cat([second, first], 1), torch.cat([global_second, global_first], 1))
    torch.testing.assert_close(static["pred_dense_flow_internal"],
                               torch.zeros_like(static["pred_dense_flow_internal"]))
    torch.testing.assert_close(forward["pred_dense_flow_internal"],
                               -reverse["pred_dense_flow_internal"])


def test_probe_is_analytically_odd():
    probe = OddProbe(12, (16, 8), 6).eval()
    first, second = torch.randn(4, 3, 12), torch.randn(4, 3, 12)
    torch.testing.assert_close(probe(first, first), torch.zeros(4, 3, 6))
    torch.testing.assert_close(probe(first, second), -probe(second, first))


def test_v2_pose_pair_sampling_is_deterministic_and_contains_sparse_motion():
    first = sample_pose_pairs(32, 24, 9, (.02, .06, .12), (.4, .4, .2), .5)
    second = sample_pose_pairs(32, 24, 9, (.02, .06, .12), (.4, .4, .2), .5)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert (first[1].ne(0).sum(-1) <= 5).any()


def test_v2_flow_encoder_shape_static_and_reverse_contract():
    encoder = FlowActionEncoder(dim=24, patch_size=5).eval()
    flow = torch.randn(2, 7, 5, 3)
    forward = encoder(flow)
    reverse = encoder(-flow)
    static = encoder(torch.zeros_like(flow))
    assert forward["action_tokens"].shape == (2, 7, 24)
    assert forward["pred_dense_flow_internal"].shape == flow.shape
    torch.testing.assert_close(reverse["action_tokens"], -forward["action_tokens"])
    torch.testing.assert_close(reverse["pred_dense_flow_internal"],
                               -forward["pred_dense_flow_internal"])
    torch.testing.assert_close(static["action_tokens"],
                               torch.zeros_like(static["action_tokens"]))
    torch.testing.assert_close(static["pred_dense_flow_internal"],
                               torch.zeros_like(static["pred_dense_flow_internal"]))
