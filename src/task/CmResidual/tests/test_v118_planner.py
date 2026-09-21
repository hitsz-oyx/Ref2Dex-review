from pathlib import Path
import sys
import types

import numpy as np
import torch
import yaml

from src.task.CmDecoderv2.kinematics import InspireKinematics
from src.task.CmResidual.v118_latency_profile import STAGES, summarize
from src.task.CmResidual.v118_planner import ACTION_DIM, PlannerConfig, QUERY_LINKS, TorchInspireKinematics


ROOT = Path(__file__).resolve().parents[4]
URDF = ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"


def test_v118_gpu_fk_matches_pinned_cpu_urdf_at_tolerance():
    torch.manual_seed(17)
    native = torch.randn(2, 3, ACTION_DIM) * 0.1
    actual = TorchInspireKinematics(URDF, "cpu").forward(native)
    legacy = InspireKinematics(URDF)
    expected = np.asarray([[
        [legacy.link_transforms_native(native[batch, candidate].numpy())[name] for name in QUERY_LINKS]
        for candidate in range(native.shape[1])
    ] for batch in range(native.shape[0])])
    assert actual.shape == (2, 3, len(QUERY_LINKS), 4, 4)
    assert np.max(np.abs(actual.numpy() - expected)) < 1e-5


def test_v118_planner_contract_is_fixed_to_eight_candidates():
    assert PlannerConfig().candidates == 8
    assert PlannerConfig().translation_scale_m == 0.02
    assert PlannerConfig().rotation_scale_rad == 0.05
    assert PlannerConfig().planner_env_microbatch == 16
    assert PlannerConfig().interaction_object_chunk == 32


def test_v118a_latency_summary_reports_percentiles_and_accounting():
    samples = [
        {name: float(stage_index + sample_index + 1) for stage_index, name in enumerate(STAGES)}
        for sample_index in range(2)
    ]
    summary = summarize(samples)
    assert summary["candidate_generation"]["mean_ms"] == 1.5
    assert summary["end_to_end"]["median_ms"] == 10.5
    assert summary["end_to_end"]["p90_ms"] == 10.9
    assert summary["accounting"]["stages_mean_ms"] == 49.5


def test_v118a_runner_pins_gpu6_real_phaseb_state_and_cuda_events():
    runner = (ROOT / "src/task/CmResidual/tools/run_v118a_latency_profile.py").read_text()
    bootstrap = (ROOT / "src/task/CmResidual/tools/v118a_latency_bootstrap.py").read_text()
    profiler = (ROOT / "src/task/CmResidual/v118_latency_profile.py").read_text()
    assert "default=6" in runner
    assert '"num_envs=128"' in runner and '"seed=42"' in runner
    assert '"train.params.config.cm_distill_coef=0.1"' in runner
    assert "first planner-active pre-action state" in runner
    assert "torch.cuda.Event(enable_timing=True)" in profiler
    assert "FrozenCmv2Planner.teacher = _intercept" in bootstrap


def test_v119_config_uses_streaming_memory_contract():
    task = yaml.safe_load((ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceV118.yaml").read_text())
    planner = task["cmPlanner"]
    assert planner["plannerEnvMicrobatch"] == 16
    assert planner["interactionObjectChunk"] == 32
    assert "maxActiveEnvs" not in planner


def test_v119_planner_microbatch_covers_all_active_envs_in_order(monkeypatch):
    action_mapping = types.ModuleType("action_mapping")

    def compose_reference_residual(base, actions, lower, upper, **kwargs):
        assert torch.equal(lower, torch.full_like(lower, -1.0))
        assert torch.equal(upper, torch.full_like(upper, 1.0))
        return base, {"applied_delta": actions}

    action_mapping.compose_reference_residual = compose_reference_residual
    dexplore_observation = types.ModuleType("dexplore_observation")
    dexplore_observation.calc_heading_quat_inv = lambda quat: quat
    dexplore_observation.quat_rotate = lambda quat, vector: vector
    monkeypatch.setitem(sys.modules, "isaacgymenvs", types.ModuleType("isaacgymenvs"))
    monkeypatch.setitem(sys.modules, "isaacgymenvs.tasks", types.ModuleType("tasks"))
    monkeypatch.setitem(sys.modules, "isaacgymenvs.tasks.cm_residual", types.ModuleType("cm_residual"))
    monkeypatch.setitem(sys.modules, "isaacgymenvs.tasks.cm_residual.action_mapping", action_mapping)
    monkeypatch.setitem(sys.modules, "isaacgymenvs.tasks.cm_residual.dexplore_observation", dexplore_observation)

    class FakeAdapter:
        def __init__(self):
            self.calls = []

        def predict_effect_only(self, object_points, object_normals, hand_points, hand_normals,
                                hand_flow, delta_time_s, hand_valid_mask=None,
                                interaction_object_chunk=None):
            self.calls.append((object_points.shape[0], interaction_object_chunk))
            batch = object_points.shape[0]
            return {"delta_xi_root": torch.zeros(batch, 6),
                    "token_mask": torch.ones(batch, 16, dtype=torch.bool),
                    "token_mass": torch.ones(batch, 16)}

    class FakeGeometry:
        object_local = torch.zeros(4, 3)

        def object(self, object_pose):
            batch = object_pose.shape[0]
            points = torch.zeros(batch, 4, 3)
            normals = torch.zeros_like(points)
            normals[..., 2] = 1.0
            return points, normals

        def hand(self, links):
            batch = links.shape[0]
            points = torch.zeros(batch, 5, 3)
            normals = torch.zeros_like(points)
            normals[..., 2] = 1.0
            return points, normals

    class FakeKinematics:
        device = torch.device("cpu")

        def forward(self, targets):
            batch, candidates, _ = targets.shape
            return torch.eye(4).expand(batch, candidates, len(QUERY_LINKS), 4, 4).clone()

    adapter = FakeAdapter()
    planner = __import__("src.task.CmResidual.v118_planner", fromlist=["FrozenCmv2Planner"]).FrozenCmv2Planner(
        adapter, FakeGeometry(), FakeKinematics(),
        PlannerConfig(planner_env_microbatch=7, interaction_object_chunk=11))
    batch = 20
    links = torch.eye(4).expand(batch, len(QUERY_LINKS), 4, 4).clone()
    result = planner.teacher(
        mu=torch.zeros(batch, ACTION_DIM),
        current_native=torch.zeros(batch, ACTION_DIM),
        base_target=torch.zeros(batch, ACTION_DIM),
        native_lower=torch.full((ACTION_DIM,), -1.0),
        native_upper=torch.full((ACTION_DIM,), 1.0),
        mimic_scales=tuple(1.0 for _ in range(ACTION_DIM)),
        current_links=links,
        object_pose=torch.eye(4).expand(batch, 4, 4).clone(),
        reference_transport=torch.zeros(batch, 232),
        desired_delta_xi=torch.zeros(batch, 6))
    assert adapter.calls == [(56, 11), (56, 11), (48, 11)]
    assert torch.equal(result["activation"], torch.ones(batch))
    assert torch.equal(result["teacher_weight"], torch.ones(batch))
    assert torch.equal(result["valid_fraction"], torch.ones(batch))
    adapter.calls.clear()
    diagnostic = planner.teacher(
        mu=torch.zeros(1, ACTION_DIM),
        current_native=torch.zeros(1, ACTION_DIM),
        base_target=torch.zeros(1, ACTION_DIM),
        native_lower=torch.full((ACTION_DIM,), -1.0),
        native_upper=torch.full((ACTION_DIM,), 1.0),
        mimic_scales=tuple(1.0 for _ in range(ACTION_DIM)),
        current_links=links[:1],
        object_pose=torch.eye(4).expand(1, 4, 4).clone(),
        reference_transport=torch.zeros(1, 232),
        desired_delta_xi=torch.zeros(1, 6),
        _diagnostic_candidate_count=2,
    )
    assert adapter.calls == [(2, 11)]
    assert torch.equal(diagnostic["activation"], torch.ones(1))
    assert torch.equal(diagnostic["valid_fraction"], torch.ones(1))


def test_v119_stage_a_agent_skips_task_teacher_when_coef_zero():
    source = (ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/learning/v118_agent.py").read_text()
    assert "self.value_mean_std = (self.central_value_net.model.value_mean_std" in source
    assert 'self.dataset.values_dict["cm_teacher_actions"] = batch_dict["cm_teacher_actions"]' in source
    assert 'self.dataset.values_dict["cm_teacher_weights"] = batch_dict["cm_teacher_weights"]' in source
    assert 'result["prev_neglogp"], result["values"], result["entropy"]' in source
    assert 'result["mus"], result["sigmas"]' in source
    assert 'input_dict["sigma"], not self.is_rnn)' in source
    assert "if self.cm_distill_coef > 0:" in source
    assert '"teacher_action": res_dict["mus"].detach()' in source
    assert '"teacher_weight": torch.zeros' in source
    assert 'infos.get("terminate")' in source
    assert "terminated = self.dones" in source


def test_v118_launcher_honors_explicit_physical_gpu():
    source = (ROOT / "src/task/CmResidual/tools/run_v118_reference_ppo.py").read_text()
    assert 'env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)' in source
    assert '"physical_gpu": args.gpu' in source
    assert "trainer_max_epochs = epochs - 1" in source
