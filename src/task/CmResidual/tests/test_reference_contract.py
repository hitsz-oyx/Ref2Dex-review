from pathlib import Path
import importlib.util
import torch

import numpy as np

from src.task.CmResidual.tools.data.build_reference import expand_q6


def _geometry_module():
    path = Path("third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py")
    spec = importlib.util.spec_from_file_location("cm_residual_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
