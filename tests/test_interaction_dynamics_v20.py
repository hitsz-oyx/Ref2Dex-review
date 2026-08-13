import numpy as np
import torch

from src.task.InteractionDynamics.field_dynamics_v20 import FieldDynamicsTransition
from src.task.InteractionDynamics.field_state_v20 import build_causal_field
from src.task.InteractionDynamics.viewer_v2.stable import CausalStableDetector
from src.task.InteractionDynamics.train_field_v20 import channel_statistics
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import build_causal_rdv_batched
from src.task.InteractionDynamics.research.v20.build_cache import mano_parameters


def test_causal_velocity_uses_current_minus_previous() -> None:
    hand = torch.tensor([[[0., 0, 0]], [[.01, 0, 0]], [[.03, 0, 0]]])
    field = build_causal_field(hand, torch.zeros(1, 3), torch.zeros(3, 1))
    np.testing.assert_allclose(field[:, 0, 4].numpy(), [.01, .02], atol=1e-7)
    assert field.shape == (2, 1, 8)


def test_field_model_shape_and_permutation_equivariance() -> None:
    torch.manual_seed(0); model = FieldDynamicsTransition(dim=32, heads=4, layers=2).eval()
    current = torch.randn(2, 16, 8); anchors = torch.randn(2, 16, 3)
    patches = torch.randn(2, 16, 4, 6); permutation = torch.randperm(16)
    with torch.no_grad():
        normal = model(current, anchors, patches)
        permuted = model(current[:, permutation], anchors[:, permutation], patches[:, permutation])
    assert normal.shape == (2, 16, 8, 8)
    torch.testing.assert_close(normal, permuted[:, torch.argsort(permutation)], atol=1e-6, rtol=1e-6)


def test_mean_intervention_preserves_anchor_axis_without_pooling_output() -> None:
    model = FieldDynamicsTransition(dim=32, heads=4, layers=1).eval()
    args = (torch.randn(1, 8, 8), torch.randn(1, 8, 3), torch.randn(1, 8, 4, 6))
    with torch.no_grad(): output = model(*args, intervention="mean")
    assert output.shape == (1, 8, 8, 8)
    torch.testing.assert_close(output[:, :1].expand_as(output), output)


def test_causal_stable_observation_uses_v_channel() -> None:
    detector = CausalStableDetector(consecutive_frames=1)
    state = detector.observe(np.ones(8), np.full((8, 3), .1))
    assert state.latched and state.u_rms_cm < .3


def test_invalid_p_does_not_affect_channel_statistics() -> None:
    valid = {"delta_y": torch.zeros(2, 2, 8), "p_valid": torch.tensor(True)}
    valid["delta_y"][..., 7] = 1
    invalid = {"delta_y": torch.zeros(2, 2, 8), "p_valid": torch.tensor(False)}
    invalid["delta_y"][..., 7] = 1000
    mean, std = channel_statistics([valid, invalid])
    assert float(mean[..., 7]) == 1
    assert abs(float(std[..., 7]) - .01) < 1e-7


def test_batched_causal_field_matches_single_teacher() -> None:
    torch.manual_seed(1);surface=torch.randn(1,4,7,3)*.01;anchors=torch.randn(1,5,3)*.01
    batched=build_causal_rdv_batched(surface,anchors)
    single=build_causal_field(surface[0],anchors[0],torch.zeros(4,7))[...,:7].transpose(0,1)*100
    torch.testing.assert_close(batched[0],single,atol=1e-5,rtol=1e-5)


def test_v20_cache_reads_mano_parameters_from_hand() -> None:
    class Sequence:
        def get_hand_params(self, side):
            assert side == "right"
            return {"hand_pose": "mano"}
        def get_object_params(self):
            raise AssertionError("MANO 参数不能从 object 字典读取")
    assert mano_parameters(Sequence(), "right")["hand_pose"] == "mano"
