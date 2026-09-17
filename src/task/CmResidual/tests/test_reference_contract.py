from pathlib import Path
import argparse
import importlib.util
import json
import math
import subprocess
import sys
import types
import torch

import numpy as np
import pytest
import yaml

from src.task.CmResidual.tools.data.build_reference import expand_q6


def _geometry_module():
    path = Path("third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py")
    spec = importlib.util.spec_from_file_location("cm_residual_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _action_mapping_module():
    root = Path("third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual")
    package_name = "cm_residual_contract_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    _load_module(f"{package_name}.contract", root / "contract.py")
    return _load_module(f"{package_name}.action_mapping", root / "action_mapping.py")


def _observation_module():
    path = Path("third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/dexplore_observation.py")
    return _load_module("cm_residual_observation_test", path)


def _base_policy_module():
    path = Path("third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/base_policy.py")
    return _load_module("cm_residual_base_policy_test", path)


def _reference_provider_module():
    root = Path("third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual")
    package_name = "cm_residual_reference_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    _load_module(f"{package_name}.dexplore_observation", root / "dexplore_observation.py")
    return _load_module(f"{package_name}.reference_provider", root / "reference_provider.py")


def test_urdf_authoritative_mimic_expansion():
    q = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    native = expand_q6(q, {7: 1.05, 9: 1.05, 11: 1.05, 13: 1.18, 16: 0.6, 17: 0.8})
    np.testing.assert_allclose(native[[6, 8, 10, 12, 14, 15]], q)
    np.testing.assert_allclose(native[[7, 9, 11, 13, 16, 17]], [0.105, 0.21, 0.315, 0.472, 0.36, 0.48])


def test_written_reference_has_explicit_frame_contract():
    root = Path("data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift")
    if not (root / "reference.npz").is_file():
        return
    with np.load(root / "reference.npz", allow_pickle=False) as z:
        assert z["frame_id"].shape == (367,)
        assert z["source_frame_id"][0] == 176
        assert z["source_frame_id"][-1] == 1640
        assert z["wrist_twist_world_ref"].shape == (367, 6)
        assert np.isfinite(z["q_native_ref"]).all()


def test_surface_geometry_has_live_shapes_and_flow():
    module = _geometry_module()
    links = ("hand_base_link", "thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
             "index_proximal", "index_intermediate", "index_tip", "middle_proximal", "middle_intermediate", "middle_tip",
             "ring_proximal", "ring_intermediate", "ring_tip", "pinky_proximal", "pinky_intermediate", "pinky_tip")
    geometry = module.SurfaceGeometry(
        hand_urdf="src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf",
        object_urdf="/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/mjcf/airplane.urdf",
        query_links=links, object_count=32, hand_count=64,
    )
    poses = torch.eye(4).repeat(2, len(links), 1, 1)
    hand, normals = geometry.hand(poses)
    flow = geometry.flow(hand)
    obj, obj_normals = geometry.object(torch.eye(4).repeat(2, 1, 1))
    assert hand.shape == flow.shape == (2, 64, 3)
    assert obj.shape == obj_normals.shape == (2, 32, 3)
    assert torch.isfinite(normals).all() and torch.isfinite(obj).all()
    assert torch.allclose(flow[0], torch.zeros_like(flow[0]))


def test_zero_residual_preserves_dexplore_physical_targets():
    module = _action_mapping_module()
    torch.manual_seed(4)
    base = torch.rand(3, 18) * 2.0 - 1.0
    current = torch.zeros(3, 18)
    lower = torch.tensor([-2.0] * 6 + [0.0] * 12)
    upper = torch.tensor([2.0] * 6 + [2.0] * 12)
    normalized = base.clamp(-1.0, 1.0)
    scale = upper - lower
    scale[:3] = 1.0
    scale[3:6] = torch.pi
    pd_action = torch.cat((normalized[:, :6], (1.0 + normalized[:, 6:]) / 2.0), dim=-1)
    expected = scale * pd_action
    expected[:, :6] += current[:, :6]
    independent = expected[:, [6, 8, 10, 12, 14, 15]].clone()
    expected[:, [7, 9, 11, 13, 16, 17]] = independent[:, [0, 1, 2, 3, 5, 5]] * torch.tensor(
        [1.05, 1.05, 1.05, 1.05, 0.6, 0.8])
    actual, details = module.compose_physical_residual(
        base, torch.zeros_like(base), current, lower, upper,
        translation_scale_m=0.015, rotation_scale_rad=0.20, finger_scale_rad=0.08)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(details["applied_delta"], torch.zeros_like(base), rtol=0.0, atol=0.0)


def test_physical_residual_is_bounded_and_keeps_mimic_contract():
    module = _action_mapping_module()
    base = torch.zeros(1, 18)
    residual = torch.ones_like(base)
    current = torch.zeros_like(base)
    lower = torch.tensor([-2.0] * 6 + [0.0] * 12)
    upper = torch.tensor([2.0] * 6 + [2.0] * 12)
    targets, details = module.compose_physical_residual(
        base, residual, current, lower, upper,
        translation_scale_m=0.015, rotation_scale_rad=0.20, finger_scale_rad=0.08)
    requested = details["requested_delta"]
    torch.testing.assert_close(requested[0, :3], torch.full((3,), 0.015))
    torch.testing.assert_close(requested[0, 3:6], torch.full((3,), 0.20))
    torch.testing.assert_close(requested[0, [6, 8, 10, 12, 14, 15]], torch.full((6,), 0.08))
    torch.testing.assert_close(requested[0, [7, 9, 11, 13, 16, 17]], torch.zeros(6))
    torch.testing.assert_close(targets[0, [7, 9, 11, 13]], targets[0, [6, 8, 10, 12]] * 1.05)
    torch.testing.assert_close(targets[0, [16, 17]], targets[0, 15] * torch.tensor([0.6, 0.8]))


def test_grab_reference_residual_uses_manifest_mimic_and_absolute_target():
    module = _action_mapping_module()
    root = Path("data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift")
    with np.load(root / "reference.npz", allow_pickle=False) as data:
        base = torch.from_numpy(data["q_native_ref"][1:2].copy())
    base[:, :6] = torch.tensor([-.14, -.93, 1.06, 1.59, .07, -1.21])
    mapping = json.loads((root / "manifest.json").read_text())["mimic_mapping"]
    scales = tuple(mapping[str(index)] for index in (7, 9, 11, 13, 16, 17))
    lower = torch.tensor([-3.] * 6 + [0.] * 12)
    upper = torch.tensor([3.] * 18)
    zero, details = module.compose_reference_residual(
        base, torch.zeros_like(base), lower, upper, translation_scale_m=.015,
        rotation_scale_rad=.20, finger_scale_rad=.08, mimic_scales=scales)
    torch.testing.assert_close(zero, base, atol=1e-6, rtol=0)
    assert details["saturation"].sum() == 0
    action = torch.zeros_like(base)
    action[:, 12] = 1
    changed, _ = module.compose_reference_residual(
        base, action, lower, upper, translation_scale_m=.015,
        rotation_scale_rad=.20, finger_scale_rad=.08, mimic_scales=scales)
    torch.testing.assert_close(changed[0, 13], changed[0, 12] * 1.18)
    assert changed[0, 13] > base[0, 13]


def test_reference_residual_reduces_only_outward_authority_at_joint_boundary():
    module = _action_mapping_module()
    scales = (1.05, 1.05, 1.05, 1.18, .6, .8)
    lower = torch.full((18,), -3.0)
    upper = torch.full((18,), 3.0)
    # Native DOF 8 is a source; its coupled DOF 9 remains feasible at 1.05.
    base = torch.zeros(1, 18)
    upper[8] = 1.0
    base[0, 8] = upper[8]
    base[0, 9] = upper[8] * scales[1]
    outward = torch.zeros_like(base)
    outward[0, 8] = 1.0
    target, details = module.compose_reference_residual(
        base, outward, lower, upper, translation_scale_m=.015,
        rotation_scale_rad=.20, finger_scale_rad=.08, mimic_scales=scales)
    torch.testing.assert_close(target, base, atol=0, rtol=0)
    assert details["configured_authority"][0, 8] == pytest.approx(.08)
    assert details["effective_authority"][0, 8] == 0
    assert details["authority_limited"][0, 8] == 1
    assert details["authority_limited"][0, 9] == 1
    assert details["saturation"].sum() == 0

    zero, zero_details = module.compose_reference_residual(
        base, torch.zeros_like(base), lower, upper, translation_scale_m=.015,
        rotation_scale_rad=.20, finger_scale_rad=.08, mimic_scales=scales)
    torch.testing.assert_close(zero, base, atol=0, rtol=0)
    assert zero_details["authority_limited"].sum() == 0

    inward = torch.zeros_like(base)
    inward[0, 8] = -1.0
    target, details = module.compose_reference_residual(
        base, inward, lower, upper, translation_scale_m=.015,
        rotation_scale_rad=.20, finger_scale_rad=.08, mimic_scales=scales)
    assert target[0, 8] == pytest.approx(.92)
    assert target[0, 9] == pytest.approx(.92 * scales[1])
    assert details["authority_limited"].sum() == 0
    assert details["saturation"].sum() == 0


def test_reference_provider_indexed_reset_preserves_per_env_reference_state():
    module = _reference_provider_module()
    provider = object.__new__(module.RetargetedReferenceProvider)
    provider.device = torch.device("cpu")
    provider.length = 4
    provider.robot_q = torch.arange(72, dtype=torch.float32).view(4, 18)
    provider.robot_dq = provider.robot_q + 100
    provider.object_state = torch.arange(52, dtype=torch.float32).view(4, 13)
    indices = torch.tensor([3, 1])
    q, dq, obj = provider.reset_state(2, indices)
    torch.testing.assert_close(q, provider.robot_q[[3, 1]])
    torch.testing.assert_close(dq, provider.robot_dq[[3, 1]])
    torch.testing.assert_close(obj, provider.object_state[[3, 1]])
    with pytest.raises(ValueError, match="reference frame range"):
        provider.reset_state(1, torch.tensor([4]))


def test_contact_selection_uses_net_force_tensor():
    module = _observation_module()
    net_force = torch.zeros(2, 20, 3)
    query = torch.tensor([2, 4, 6, 8, 10, 12])
    contact = torch.tensor([1, 4])
    net_force[:, 4, 0] = 3.0
    net_force[:, 10, 2] = -2.0
    selected = module.select_contact_forces(net_force, query, contact)
    torch.testing.assert_close(selected, net_force[:, [4, 10]])
    assert (selected.abs().amax(-1) > 0.1).all()


def test_dexplore_actor_matches_checkpoint_relu_inference():
    checkpoint = Path("/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth")
    if not checkpoint.is_file():
        pytest.skip("released DExplore checkpoint is unavailable")
    module = _base_policy_module()
    policy = module.InspireDExplorePolicy(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = payload["model"]
    stats = payload["running_mean_std"]
    mean = torch.as_tensor(stats["running_mean"], dtype=torch.float32)
    var = torch.as_tensor(stats["running_var"], dtype=torch.float32)
    standardized = torch.linspace(-7.0, 7.0, module.OBSERVATION_DIM).repeat(2, 1)
    standardized[1] = standardized[1].flip(0)
    observation = mean + standardized * torch.sqrt(var + 1e-5)

    expected = standardized.clamp(-5.0, 5.0)
    for layer in (0, 2, 4, 6):
        expected = torch.nn.functional.linear(
            expected,
            model[f"a2c_network.actor_mlp.{layer}.weight"],
            model[f"a2c_network.actor_mlp.{layer}.bias"],
        )
        expected = torch.relu(expected)
    expected = torch.nn.functional.linear(
        expected, model["a2c_network.mu.weight"], model["a2c_network.mu.bias"]
    ).clamp(-1.0, 1.0)
    actual = policy(observation)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2e-6)
    assert all(isinstance(policy.actor[index], torch.nn.ReLU) for index in (1, 3, 5, 7))
    torch.testing.assert_close(policy.running_var, var, rtol=0.0, atol=0.0)


def test_legacy_reference_and_reset_use_dexplore_source_tensor():
    corrected = Path("data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/reference.npz")
    source_path = Path(
        "data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/"
        "s1_airplane_lift/interaction_hand_inspire.pt"
    )
    object_mesh = Path("/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/mjcf/objects/airplane/airplane.obj")
    if not all(path.is_file() for path in (corrected, source_path, object_mesh)):
        pytest.skip("DExplore reference inputs are unavailable")
    module = _reference_provider_module()
    provider = module.ReferenceProvider(
        corrected,
        "cpu",
        expected_sha256="a2d710b911cf8988750208450c411b3095e187ed2b2f05df459c24d5748812c1",
        source_tensor=source_path,
        source_sha256="19b110dc81c4928b4f3e6197d8549b011fd48a8157e75fd06668bd306dc396cf",
        frame_start=44,
        frame_end=410,
        object_mesh=object_mesh,
    )
    source = torch.load(source_path, map_location="cpu", weights_only=True).float()
    q, dq, object_state = provider.reset_state(2)
    assert provider.length == 367
    assert provider.hoi_data.shape == (367, 428)
    assert provider.object_points.shape == (256, 3)
    assert provider.first_contact_index == 16
    assert provider.training_eligible is False
    assert module.validate_reference_usage(False, "diagnostic") == "diagnostic"
    assert module.validate_reference_usage(False, "ppo_pilot") == "ppo_pilot"
    with pytest.raises(ValueError, match="training_eligible=false"):
        module.validate_reference_usage(False, "")
    with pytest.raises(ValueError, match="allowIneligibleFor"):
        module.validate_reference_usage(False, "training")
    torch.testing.assert_close(q[0], source[44, 373:391], rtol=0.0, atol=0.0)
    torch.testing.assert_close(dq[0], (source[44, 373:391] - source[43, 373:391]) * 30.0)
    torch.testing.assert_close(object_state[0, :7], source[44, 198:205], rtol=0.0, atol=0.0)
    torch.testing.assert_close(provider.table_pose, source[0, 238:245], rtol=0.0, atol=0.0)
    expected_key_positions = source[44, 102:198].view(32, 3)[16:].reshape(-1)
    torch.testing.assert_close(
        provider.hoi_data[0, 119:167], expected_key_positions, rtol=0.0, atol=0.0)
    expected_key_velocity = (
        source[44, 102:198].view(32, 3)[16:]
        - source[43, 102:198].view(32, 3)[16:]
    ).reshape(-1) * 30.0
    torch.testing.assert_close(
        provider.hoi_data[0, 296:344], expected_key_velocity, rtol=0.0, atol=0.0)

    urdf = Path("src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")
    kinematics_path = Path("src/task/CmDecoderv2/kinematics.py")
    kinematics_module = _load_module("cm_residual_kinematics_contract_test", kinematics_path)
    kinematics = kinematics_module.InspireKinematics(urdf)
    position, rotation, velocity, angular = provider.reset_body_state(urdf, kinematics.link_names)
    expected_links = kinematics.link_transforms_native(source[44, 373:391].numpy())
    expected_position = torch.as_tensor(
        np.stack([expected_links[name][:3, 3] for name in kinematics.link_names]), dtype=torch.float32)
    torch.testing.assert_close(position, expected_position, rtol=0.0, atol=2e-7)
    assert position.shape == velocity.shape == angular.shape == (len(kinematics.link_names), 3)
    assert rotation.shape == (len(kinematics.link_names), 4)
    assert all(torch.isfinite(value).all() for value in (position, rotation, velocity, angular))


def test_eligible_corrected_reference_records_raw_contact_gate():
    manifest_path = Path(
        "data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/manifest.json")
    source_path = Path(
        "data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/"
        "s1_airplane_lift/interaction_hand_inspire.pt")
    if not (manifest_path.is_file() and source_path.is_file()):
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    assert source.shape[1] == 598
    assert torch.all((source[44:411, 205] == 0) | (source[44:411, 205] == 1))
    assert manifest["training_eligible"] is True
    metrics = manifest["metrics"]
    assert metrics["raw_contact_source"] == "source_tensor[:,205:206]"
    assert metrics["raw_contact_frames"] == 296
    assert metrics["raw_contact_distance_pass_frames"] == 295
    assert metrics["raw_contact_distance_pass_fraction"] >= 0.90
    assert manifest["gate_status"]["raw_contact_distance"] == "PASS"


def test_observation_matches_unmodified_dexplore_source():
    script = Path("src/task/CmResidual/tests/dexplore_golden_parity.py")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {"shape": [2, 721], "max_abs_error": 0.0}


def test_v14_configs_lock_real_inputs_and_dimensions():
    root = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg/task")
    for name in ("CmResidual.yaml", "CmResidualOnline.yaml"):
        config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert config["env"]["numObservations"] == 2005
        assert config["env"]["asset"]["objectAssetFileName"] == "airplane.urdf"
        assert config["basePolicy"]["useOiCmContext"] is True
        assert config["basePolicy"]["cmFeatureDim"] == 32
        assert config["basePolicy"]["cmNumSlots"] == 16
        assert config["basePolicy"]["oiCmCheckpointSha256"] == "3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283"
        assert config["reference"]["sha256"] == "a2d710b911cf8988750208450c411b3095e187ed2b2f05df459c24d5748812c1"
        assert "reference_tracking_v2/s1_airplane_lift/reference.npz" in config["reference"]["path"]
        assert config["reference"]["profile"] == "dexplore_legacy_parity"
        assert config["reference"]["allowIneligibleFor"] == ""
        assert config["reference"]["sourceTensorSha256"] == "19b110dc81c4928b4f3e6197d8549b011fd48a8157e75fd06668bd306dc396cf"
        assert config["reference"]["frameStart"] == 44
        assert config["reference"]["frameEnd"] == 410
        assert config["env"]["terminateOnSuccess"] is True
        assert config["env"]["plane"] == {
            "staticFriction": 1.0,
            "dynamicFriction": 1.0,
            "restitution": 1.0,
        }
        assert config["sim"]["substeps"] == 4
        assert config["sim"]["physx"]["num_threads"] == 8
        assert config["sim"]["physx"]["contact_offset"] == 0.01
        assert config["sim"]["physx"]["rest_offset"] == 0.0
        assert config["sim"]["physx"]["max_depenetration_velocity"] == 20.0
        assert config["sim"]["physx"]["default_buffer_size_multiplier"] == 25.0
        assert config["residual"] == {
            "translationScaleM": 0.015,
            "rotationScaleRad": 0.20,
            "fingerScaleRad": 0.08,
        }


def test_v15_pilot_config_is_a_two_update_wiring_smoke():
    path = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPilotPPO.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["defaults"] == ["CmResidualPPO", "_self_"]
    pilot = config["params"]["config"]
    assert pilot == {
        "name": "CmResidualPilot",
        "full_experiment_name": "CmResidualPilot",
        "max_epochs": 2,
        "horizon_length": 32,
        "minibatch_size": 2048,
        "mini_epochs": 4,
        "save_best_after": 0,
        "save_frequency": 1,
    }


def test_v18_safe_policy_starts_at_zero_with_fixed_small_sigma():
    vendor = str(Path("third_party/IsaacGymEnvs").resolve())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.learning.cm_models import ModelCmContinuous
    from isaacgymenvs.learning.cm_network_builder import CmBuilder

    config_root = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg").resolve()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
        cfg = compose(config_name="config", overrides=[
            "task=CmResidual",
            "train=CmResidualSafePPO",
            "task.basePolicy.useOiCmContext=false",
        ])
    params = OmegaConf.to_container(cfg.train.params, resolve=True)
    continuous = params["network"]["space"]["continuous"]
    assert continuous["mu_init"] == {"name": "const_initializer", "val": 0.0}
    assert continuous["sigma_init"]["val"] == pytest.approx(math.log(0.1), abs=1e-15)
    assert continuous["fixed_sigma"] is True
    assert continuous["learn_sigma"] is False
    assert cfg.task.basePolicy.useOiCmContext is False

    builder = CmBuilder()
    builder.load(params["network"])
    model = ModelCmContinuous(builder).build({
        "input_shape": (1442,),
        "actions_num": 18,
        "num_seqs": 1,
        "value_size": 1,
        "normalize_input": bool(params["config"]["normalize_input"]),
        "normalize_value": bool(params["config"]["normalize_value"]),
    })
    network = model.a2c_network
    assert network.mu.weight.count_nonzero().item() == 0
    assert network.mu.bias.count_nonzero().item() == 0
    assert network.sigma.requires_grad is False
    torch.testing.assert_close(
        network.sigma.exp(), torch.full((18,), 0.1), rtol=0.0, atol=1e-7)
    with torch.no_grad():
        result = model({"obs": torch.randn(3, 1442), "is_train": False})
    torch.testing.assert_close(result["mus"], torch.zeros(3, 18), rtol=0.0, atol=0.0)


def test_v115_cmv2_actor_transport_pools_tokens_only_for_actor():
    vendor = str(Path("third_party/IsaacGymEnvs").resolve())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.learning.cm_models import ModelCmContinuous
    from isaacgymenvs.learning.cm_network_builder import CmEffectBuilder

    config_root = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg").resolve()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
        cfg = compose(config_name="config", overrides=[
            "task=CmResidualGrabReferenceTransitionCmv2Actor",
            "train=CmResidualGrabReferenceTransitionCmv2ActorPPO",
            "task.cmBuffer.outputDir=/tmp/cmv2_actor_contract",
        ])
    assert cfg.task.env.numObservations == 726
    assert cfg.task.basePolicy.useCmv2ActorContext is True
    assert cfg.task.basePolicy.useCmv2ActionEvaluator is True
    params = OmegaConf.to_container(cfg.train.params, resolve=True)
    assert params["network"]["name"] == "cm_effect_actor_critic"
    assert params["network"]["actor_input_dim"] == 726
    assert params["network"]["critic_input_dim"] == 68
    assert params["config"]["normalize_input"] is False
    builder = CmEffectBuilder()
    builder.load(params["network"])
    model = ModelCmContinuous(builder).build({
        "input_shape": (726,), "actions_num": 18, "num_seqs": 1, "value_size": 1,
        "normalize_input": False, "normalize_value": False,
    })
    network = model.a2c_network
    raw = torch.zeros(2, 726)
    # Row 0 is the all-invalid-token fallback; row 1 has a valid canonical token.
    raw[1, 68 + 39] = 1.0
    raw[:, -18:] = torch.linspace(-1.0, 1.0, 18)
    representation = network._cm_input(raw)
    assert representation.shape == (2, 214)
    torch.testing.assert_close(representation[0, 68:196], torch.zeros(128), rtol=0.0, atol=0.0)
    torch.testing.assert_close(representation[:, :68], raw[:, :68], rtol=0.0, atol=0.0)
    torch.testing.assert_close(representation[:, -18:], raw[:, -18:], rtol=0.0, atol=0.0)
    with torch.no_grad():
        result = model({"obs": raw, "is_train": False})
        critic = network.eval_critic(raw)
    assert result["mus"].shape == (2, 18)
    assert critic.shape == (2, 1)
    assert torch.isfinite(result["mus"]).all() and torch.isfinite(critic).all()


def test_v116_normalizes_only_the_base68_prefix():
    vendor = str(Path("third_party/IsaacGymEnvs").resolve())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.learning.cm_models import ModelCmEffectContinuous
    from isaacgymenvs.learning.cm_network_builder import CmEffectBuilder

    config_root = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg").resolve()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
        cfg = compose(config_name="config", overrides=[
            "task=CmResidualGrabReferenceTransitionCmv2ActorV116",
            "train=CmResidualGrabReferenceTransitionCmv2ActorV116PPO",
            "task.cmBuffer.outputDir=/tmp/cmv2_actor_v116_contract",
        ])
    assert cfg.task.cmBuffer.mode == "transition_only"
    assert cfg.task.cmBuffer.schema == "cmresidual.cm_buffer.transition_only.v2"
    params = OmegaConf.to_container(cfg.train.params, resolve=True)
    assert params["model"]["name"] == "cm_effect_continuous"
    assert params["config"]["normalize_input"] is True
    builder = CmEffectBuilder()
    builder.load(params["network"])
    model = ModelCmEffectContinuous(builder).build({
        "input_shape": (726,), "actions_num": 18, "num_seqs": 1, "value_size": 1,
        "normalize_input": True, "normalize_value": False,
    })
    raw = torch.randn(3, 726)
    raw[:, 68 + 39] = 1.0
    normalized = model.norm_obs(raw)
    assert tuple(model.running_mean_std.running_mean.shape) == (68,)
    torch.testing.assert_close(normalized[:, 68:], raw[:, 68:], rtol=0.0, atol=0.0)


def test_v116_nominal_actor_path_does_not_call_legacy_cpu_fk_evaluator():
    task_source = Path(
        "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py").read_text()
    start = task_source.index("    def evaluate_cmv2_nominal_base")
    end = task_source.index("    def evaluate_cmv2_actions", start)
    nominal = task_source[start:end]
    assert "evaluate_cmv2_actions" not in nominal
    assert "InspireKinematics" not in nominal
    assert ".cpu(" not in nominal and ".numpy(" not in nominal
    assert "reference.link_pose" in nominal


def test_v116_ddp_launcher_requires_explicit_gpu012():
    tools = Path("src/task/CmResidual/tools").resolve()
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    module = _load_module(
        "cm_residual_v116_ddp_runner_test", tools / "run_cmv2_actor_distributed.py")
    assert module._parse_gpus("0,1,2") == (0, 1, 2)
    with pytest.raises(argparse.ArgumentTypeError, match="exactly"):
        module._parse_gpus("0,1,3")


def test_v116_ddp_launcher_uses_python_script_bootstrap():
    tools = Path("src/task/CmResidual/tools").resolve()
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    module = _load_module(
        "cm_residual_v116_ddp_runner_bootstrap_test", tools / "run_cmv2_actor_distributed.py")
    command = module._torchrun_command(["task=CmResidualGrabReferenceTransitionCmv2ActorV116"])
    assert command[:6] == [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc_per_node=3", str(module.BOOTSTRAP)]
    assert module.BOOTSTRAP.suffix == ".py" and module.BOOTSTRAP.is_file()
    assert "-c" not in command


def _build_v19_model(train_config, seed=42):
    vendor = str(Path("third_party/IsaacGymEnvs").resolve())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from isaacgymenvs.learning.cm_models import ModelCmContinuous
    from isaacgymenvs.learning.cm_network_builder import CmBuilder

    config_root = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg").resolve()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_root)):
        cfg = compose(config_name="config", overrides=[
            "task=CmResidual", f"train={train_config}",
            "task.basePolicy.useOiCmContext=true",
        ])
    params = OmegaConf.to_container(cfg.train.params, resolve=True)
    torch.manual_seed(seed)
    builder = CmBuilder()
    builder.load(params["network"])
    model = ModelCmContinuous(builder).build({
        "input_shape": (2005,),
        "actions_num": 18,
        "num_seqs": 1,
        "value_size": 1,
        "normalize_input": False,
        "normalize_value": False,
    })
    return model, params


def test_v19_critic_only_configs_are_a_strict_actor_matched_ablation():
    control, control_params = _build_v19_model("CmResidualSafeCriticControlPPO")
    critic_cm, critic_cm_params = _build_v19_model("CmResidualSafeCriticCmPPO")
    assert control_params["network"]["actor_input_dim"] == 1442
    assert control_params["network"]["critic_input_dim"] == 1442
    assert control_params["network"]["critic_candidate_input_dims"] == [1442, 2005]
    assert critic_cm_params["network"]["actor_input_dim"] == 1442
    assert critic_cm_params["network"]["critic_input_dim"] == 2005
    assert critic_cm_params["network"]["critic_candidate_input_dims"] == [1442, 2005]

    control_state = control.a2c_network.state_dict()
    critic_cm_state = critic_cm.a2c_network.state_dict()
    actor_keys = [
        key for key in control_state
        if key.startswith("actor_mlp.") or key.startswith("mu.") or key == "sigma"
    ]
    assert actor_keys
    for key in actor_keys:
        torch.testing.assert_close(control_state[key], critic_cm_state[key], rtol=0.0, atol=0.0)

    prefix = torch.randn(4, 1442)
    suffix_a = torch.randn(4, 563)
    suffix_b = suffix_a + 10.0
    obs_a = torch.cat((prefix, suffix_a), dim=-1)
    obs_b = torch.cat((prefix, suffix_b), dim=-1)
    with torch.no_grad():
        control_actor_a = control.a2c_network.eval_actor(obs_a)
        control_actor_b = control.a2c_network.eval_actor(obs_b)
        cm_actor_a = critic_cm.a2c_network.eval_actor(obs_a)
        cm_actor_b = critic_cm.a2c_network.eval_actor(obs_b)
        control_value_a = control.a2c_network.eval_critic(obs_a)
        control_value_b = control.a2c_network.eval_critic(obs_b)
        cm_value_a = critic_cm.a2c_network.eval_critic(obs_a)
        cm_value_b = critic_cm.a2c_network.eval_critic(obs_b)
    for left, right in (
            (control_actor_a[0], control_actor_b[0]),
            (control_actor_a[1], control_actor_b[1]),
            (cm_actor_a[0], cm_actor_b[0]),
            (cm_actor_a[1], cm_actor_b[1]),
            (control_actor_a[0], cm_actor_a[0]),
            (control_actor_a[1], cm_actor_a[1]),
            (control_value_a, control_value_b)):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert not torch.equal(cm_value_a, cm_value_b)


def test_v19_matched_build_preserves_rng_and_initial_stochastic_actions():
    models = []
    post_build_rng_states = []
    for train_config in (
            "CmResidualSafeCriticControlPPO", "CmResidualSafeCriticCmPPO"):
        model, _ = _build_v19_model(train_config, seed=42)
        model.eval()
        models.append(model)
        post_build_rng_states.append(torch.get_rng_state().clone())
    assert torch.equal(*post_build_rng_states)

    observation = torch.randn(4, 2005)
    actions = []
    with torch.no_grad():
        for model, rng_state in zip(models, post_build_rng_states):
            torch.set_rng_state(rng_state)
            actions.append(model({"obs": observation, "is_train": False})["actions"])
    torch.testing.assert_close(actions[0], actions[1], rtol=0.0, atol=0.0)


def test_v19_asymmetric_inputs_require_separate_network():
    vendor = str(Path("third_party/IsaacGymEnvs").resolve())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from isaacgymenvs.learning.cm_network_builder import CmBuilder

    _, params = _build_v19_model("CmResidualSafeCriticCmPPO")
    params["network"]["separate"] = False
    builder = CmBuilder()
    builder.load(params["network"])
    with pytest.raises(ValueError, match="separate=true"):
        builder.build(
            "cm_actor_critic", input_shape=(2005,), actions_num=18,
            num_seqs=1, value_size=1)


def test_v18_preservation_gate_requires_both_lift_and_contact():
    tools = Path("src/task/CmResidual/tools").resolve()
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    module = _load_module("cm_residual_v18_eval_test", tools / "eval_residual_stability.py")
    baseline = {"mean_env_max_lift_m": 0.08, "mean_contact_occupancy": 0.50}
    passed = module.preservation_gate(
        {"mean_env_max_lift_m": 0.064, "mean_contact_occupancy": 0.40}, baseline)
    assert passed["passed"] is True
    assert passed["lift_ratio"] == pytest.approx(0.8)
    assert passed["contact_ratio"] == pytest.approx(0.8)
    failed = module.preservation_gate(
        {"mean_env_max_lift_m": 0.063, "mean_contact_occupancy": 0.50}, baseline)
    assert failed["passed"] is False
    with pytest.raises(ValueError, match="positive"):
        module.preservation_gate(
            {"mean_env_max_lift_m": 0.1, "mean_contact_occupancy": 0.1},
            {"mean_env_max_lift_m": 0.0, "mean_contact_occupancy": 0.5})


def test_v18_installs_only_required_isaacgym_numpy_alias():
    tools = Path("src/task/CmResidual/tools").resolve()
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    module = _load_module("cm_residual_v18_numpy_compat_test",
                          tools / "eval_residual_stability.py")
    import numpy as np

    had_float = "float" in np.__dict__
    previous = np.__dict__.get("float")
    np.__dict__.pop("float", None)
    try:
        module._install_isaacgym_numpy_compat()
        assert np.float is float
    finally:
        if had_float:
            np.float = previous
        else:
            np.__dict__.pop("float", None)


def test_v18_training_stage_overrides_rlgames_epoch_budget():
    tools = Path("src/task/CmResidual/tools").resolve()
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    module = _load_module("cm_residual_v18_training_budget_test",
                          tools / "run_ppo_stability.py")
    resolved = module._resolved_config([
        "task=CmResidual",
        "train=CmResidualSafePPO",
        "task.basePolicy.useOiCmContext=false",
        "train.params.config.max_epochs=10",
    ])
    assert resolved["max_iterations"] != 10
    assert resolved["train"]["params"]["config"]["max_epochs"] == 10


def test_v19_smoke_budget_and_checkpoint_frequency_are_explicit():
    tools = Path("src/task/CmResidual/tools").resolve()
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    module = _load_module("cm_residual_v19_smoke_budget_test",
                          tools / "run_ppo_stability.py")
    resolved = module._resolved_config([
        "task=CmResidual",
        "train=CmResidualSafeCriticControlPPO",
        "task.basePolicy.useOiCmContext=true",
        "train.params.config.max_epochs=2",
        "train.params.config.save_frequency=1",
    ])
    assert resolved["train"]["params"]["config"]["max_epochs"] == 2
    assert resolved["train"]["params"]["config"]["save_frequency"] == 1


def test_v110_run_version_is_explicit_and_preserves_legacy_defaults():
    tools = Path("src/task/CmResidual/tools").resolve()
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    module = _load_module("cm_residual_v110_version_test",
                          tools / "eval_residual_stability.py")

    assert module._resolve_modification_version("v18_no_cm") == "V1.8"
    assert module._resolve_modification_version("control") == "V1.9.2"
    assert module._resolve_modification_version("critic_cm") == "V1.9.2"
    assert module._resolve_modification_version("control", "V1.10.1") == "V1.10.1"
    assert module._resolve_modification_version("critic_cm", "V1.10.1") == "V1.10.1"
    with pytest.raises(ValueError, match="Unsupported modification version"):
        module._resolve_modification_version("v18_no_cm", "V1.10.1")
    with pytest.raises(ValueError, match="Unsupported modification version"):
        module._resolve_modification_version("control", "V1.10")


def test_real_oi_cm_checkpoint_contract_when_available():
    checkpoint = Path(
        "outputs/objectinteractioncm/"
        "object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt")
    if not checkpoint.is_file():
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload["config"]["meta"]
    assert meta["feature_dim"] == meta["cm_dim"] == 32
    assert meta["num_cm_tokens"] == 16
    assert meta["num_obj_points"] == 1024
    assert meta["num_hand_points"] == 1538
    scale_manifest = Path(meta["scale_manifest_path"])
    assert scale_manifest.is_file()
    scales = json.loads(scale_manifest.read_text(encoding="utf-8"))
    assert "scales" in scales
