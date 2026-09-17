from pathlib import Path
import xml.etree.ElementTree as ET

import json
import numpy as np
import torch
import yaml

from src.task.CmDecoderv2.kinematics import InspireKinematics
from src.task.CmResidual.tools.data.build_dexycb_base_reference import (
    DEFAULT_URDF,
    select_canonical_tips,
    wrist_native_from_pose,
)


ARTIFACT = Path(
    "data/processed_data/cm_residual/dexycb_base_v1/"
    "subject-10/20201022_110806"
)
CONFIG = Path(
    "third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/"
    "CmResidualDexYCBBase.yaml"
)


def test_tip_identity_uses_fingertip_region_and_is_deterministic():
    canonical = np.asarray([
        [0.02, 0.0, 0.0], [0.03, 0.0, 0.0], [0.20, 0.0, 0.0],
        [0.0, 0.02, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.04],
        [-0.04, 0.0, 0.0], [0.0, -0.04, 0.0],
    ])
    finger = np.asarray([1, 1, 1, 2, 2, 3, 4, 5])
    region = np.asarray([1, 1, 2, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(
        select_canonical_tips(canonical, finger, region), [1, 4, 5, 6, 7])


def test_wrist_native_round_trip_matches_inspire_fk():
    kinematics = InspireKinematics(DEFAULT_URDF)
    native = np.zeros(18, dtype=np.float64)
    native[:6] = [0.12, -0.04, 0.31, 0.25, -0.33, 0.41]
    expected = kinematics.wrist_pose_from_native(native)
    recovered = wrist_native_from_pose(kinematics, expected)
    probe = np.zeros(18, dtype=np.float64)
    probe[:6] = recovered
    np.testing.assert_allclose(kinematics.wrist_pose_from_native(probe), expected, atol=1e-6)


def test_dexycb_base_config_is_isolated_and_disables_cm():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    canonical = yaml.safe_load(Path(
        "third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml"
    ).read_text(encoding="utf-8"))
    assert config["name"] == "CmResidualDexYCBBase"
    assert config["env"]["numObservations"] == 1442
    assert config["env"]["numActions"] == 18
    assert config["env"]["episodeLength"] == 72
    assert config["env"]["terminateOnSuccess"] is False
    assert config["env"]["asset"]["objectAssetFileName"] == "master_chef_can.urdf"
    assert config["env"]["asset"]["initializeDofsAtCreation"] is True
    assert config["basePolicy"]["useOiCmContext"] is False
    assert config["reference"]["frameStart"] == 1
    assert config["reference"]["frameEnd"] == 72
    assert config["reference"]["allowIneligibleFor"] == "diagnostic"
    assert "initializeDofsAtCreation" not in canonical["env"]["asset"]


def test_built_dexycb_artifact_obeys_padded_source_contract():
    if not ARTIFACT.is_dir():
        return
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    source = torch.load(ARTIFACT / "interaction_hand_inspire.pt", map_location="cpu", weights_only=True)
    with np.load(ARTIFACT / "reference.npz", allow_pickle=False) as reference:
        assert reference["q_native_ref"].shape == (72, 18)
        assert reference["link_pose_world_ref"].shape[0] == 72
        assert np.isfinite(reference["link_pose_world_ref"]).all()
    assert source.shape == (74, 598)
    assert torch.equal(source[0], source[1])
    assert torch.equal(source[-1], source[-2])
    assert torch.isfinite(source).all()
    assert manifest["training_eligible"] is False
    assert manifest["evaluation_eligible"] is True
    assert manifest["training_frame_range"] == [1, 72]
    assert all(value == "PASS" for value in manifest["gate_status"].values())
    assert manifest["metrics"]["tip_improvement_fraction"] >= 0.20
    ET.parse(ARTIFACT / "assets/master_chef_can.urdf")
    ET.parse(ARTIFACT / "assets/table.urdf")
