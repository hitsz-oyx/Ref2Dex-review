from pathlib import Path
import importlib.util
import json
import sys
import types
import torch

import numpy as np
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
    expected = module.dexplore_action_to_targets(base, current, lower, upper)
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


def test_v13_configs_lock_real_inputs_and_dimensions():
    root = Path("third_party/IsaacGymEnvs/isaacgymenvs/cfg/task")
    for name in ("CmResidual.yaml", "CmResidualOnline.yaml"):
        config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert config["env"]["numObservations"] == 2005
        assert config["env"]["asset"]["objectAssetFileName"] == "airplane.urdf"
        assert config["basePolicy"]["cmFeatureDim"] == 32
        assert config["basePolicy"]["cmNumSlots"] == 16
        assert config["basePolicy"]["oiCmCheckpointSha256"] == "3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283"
        assert config["reference"]["sha256"] == "a2d710b911cf8988750208450c411b3095e187ed2b2f05df459c24d5748812c1"
        assert config["residual"] == {
            "translationScaleM": 0.015,
            "rotationScaleRad": 0.20,
            "fingerScaleRad": 0.08,
        }


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
