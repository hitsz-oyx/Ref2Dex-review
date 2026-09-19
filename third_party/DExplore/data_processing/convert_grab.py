"""Convert GRAB dataset sequences into simulation-ready interaction data.

This script processes raw GRAB motion capture data (SMPL-X parameters + object poses)
and retargets human hand poses to robot hand DOFs using dex_retargeting.

Usage:
    python data_processing/convert_grab.py \
        --robot inspire \
        --grab_dir /path/to/grab \
        --original_grab_dir /path/to/grab/raw \
        --smplx_model_dir /path/to/smplx/models

Requires:
    - GRAB dataset (https://grab.is.tue.mpg.de/)
    - SMPL-X body model (https://smpl-x.is.tue.mpg.de/)
    - dex_retargeting package (pip install dex-retargeting)
    - SAPIEN (pip install sapien)
"""
import argparse
import copy
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

from smpl_constants import (
    SMPLH_BONE_ORDER_NAMES, SMPLH_SEGMENT, SMPL_2_MUJOCO,
    SMPL_2_MUJOCO_JOINTS, load_smplx_vert_segmentation,
)
from skeleton_utils import SkeletonTree, SkeletonState
from robot_configs import ROBOT_CONFIGS, INSPIRE_RETARGET_REORDER


# ── Utility functions ───────────────────────────────────────────────────────

def quat_to_exp_map(q):
    """Convert quaternion (wxyz) to exponential map."""
    w = q[..., 0:1].clamp(-1, 1)
    xyz = q[..., 1:4]
    angle = 2.0 * torch.acos(w.abs())
    sin_half = torch.sin(angle / 2).clamp(min=1e-8)
    axis = xyz / sin_half
    small = (angle.abs() < 1e-6).squeeze(-1)
    exp_map = axis * angle
    exp_map[small] = 0.0
    return exp_map


def quat_conjugate_wxyz(q):
    """Conjugate of wxyz quaternion."""
    return torch.cat([q[..., 0:1], -q[..., 1:4]], dim=-1)


def quat_mul_wxyz(a, b):
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 + y1*w2 + z1*x2 - x1*z2,
        w1*z2 + z1*w2 + x1*y2 - y1*x2,
    ], dim=-1)


def local_rotation_to_dof_smpl(local_rot):
    """Convert local quaternions (xyzw) to exponential map DOF positions."""
    B, J, _ = local_rot.shape
    # Convert xyzw -> wxyz for exp map
    wxyz = torch.cat([local_rot[..., 3:], local_rot[..., :3]], dim=-1)
    dof_pos = quat_to_exp_map(wxyz[:, 1:])  # skip root
    return dof_pos.reshape(B, -1)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def DotDict(in_dict):
    out = copy.copy(in_dict)
    for k, v in out.items():
        if isinstance(v, dict):
            out[k] = DotDict(v)
    return dotdict(out)


def parse_npz(path):
    npz = np.load(path, allow_pickle=True)
    return DotDict({k: npz[k].item() for k in npz.files})


def params2torch(params, device='cuda'):
    # Filter out non-array values and 'fullpose' (conflicts with individual pose params)
    skip_keys = {"fullpose"}
    return {k: torch.from_numpy(v).float().to(device) for k, v in params.items()
            if isinstance(v, np.ndarray) and k not in skip_keys}


def vertex_normals(vertices, faces):
    """Compute per-vertex normals from mesh vertices and faces."""
    assert vertices.ndim == 3 and faces.ndim == 3
    bs, nv = vertices.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3, device=device)
    faces = faces + (torch.arange(bs, dtype=torch.int32, device=device) * nv)[:, None, None]
    vf = vertices.reshape(bs * nv, 3)[faces.long()]
    faces = faces.view(-1, 3)
    vf = vf.view(-1, 3, 3)
    normals.index_add_(0, faces[:, 1].long(), torch.cross(vf[:, 2] - vf[:, 1], vf[:, 0] - vf[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), torch.cross(vf[:, 0] - vf[:, 2], vf[:, 1] - vf[:, 2]))
    normals.index_add_(0, faces[:, 0].long(), torch.cross(vf[:, 1] - vf[:, 0], vf[:, 2] - vf[:, 0]))
    normals = F.normalize(normals, eps=1e-6, dim=1)
    return normals.reshape(bs, nv, 3)


# ── Subject vertex template cache ──────────────────────────────────────────

_sbj_cache = {}

def load_sbj_verts(sbj_id, seq_data, data_root):
    if sbj_id in _sbj_cache:
        return _sbj_cache[sbj_id]
    mesh_path = os.path.join(data_root, seq_data.body.vtemp)
    # The GRAB .npz stores vtemp as "tools/subject_meshes/male/s1.ply" but
    # the official zip extracts to "male/s1.ply" under the tools/ directory.
    # Try both paths: first as-is under grab_dir, then under grab_dir/tools/.
    if not os.path.exists(mesh_path):
        # Strip "tools/subject_meshes/" prefix -> "male/s1.ply", look under tools/
        vtemp_rel = seq_data.body.vtemp
        for prefix in ("tools/subject_meshes/", "subject_meshes/"):
            if vtemp_rel.startswith(prefix):
                vtemp_rel = vtemp_rel[len(prefix):]
                break
        mesh_path = os.path.join(data_root, "tools", vtemp_rel)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(
            f"Subject mesh not found for {sbj_id}. "
            f"Download subject meshes from GRAB and extract under {{grab_dir}}/tools/. "
            f"Tried: {os.path.join(data_root, seq_data.body.vtemp)}, {mesh_path}")
    vtemp = np.array(trimesh.load(mesh_path, process=False).vertices)
    _sbj_cache[sbj_id] = vtemp
    return vtemp


# ── Build vertex-to-joint segmentation mask ─────────────────────────────────

def build_joints2verts_mask():
    """Build a (52, 10475) boolean mask mapping joints to vertices."""
    smplx_seg = load_smplx_vert_segmentation()
    mask = torch.zeros((52, 10475), dtype=torch.bool)
    for i, name in enumerate(SMPLH_BONE_ORDER_NAMES):
        if i > 21:
            break
        verts_list = smplx_seg[SMPLH_SEGMENT[name]]
        mask[i, verts_list] = True
    return mask


# ── Setup retargeting ───────────────────────────────────────────────────────

def setup_retargeting(robot_cfg, dex_retarget_dir):
    """Initialize dex_retargeting and SAPIEN for a given robot hand."""
    from dex_retargeting import yourdfpy as urdf
    from dex_retargeting.constants import RobotName, HandType, get_default_config_path, RetargetingType
    from dex_retargeting.retargeting_config import RetargetingConfig
    import sapien
    # SAPIEN 2.x keeps the Python bindings under ``sapien.core`` while older
    # releases re-exported ``Scene`` at the package root.  Normalize the
    # namespace so the converter works with either layout.
    if not hasattr(sapien, "Scene") and hasattr(sapien, "core"):
        sapien.Scene = sapien.core.Scene

    hands_dir = Path(dex_retarget_dir)
    RetargetingConfig.set_default_urdf_dir(hands_dir)

    robot_name_enum = RobotName[robot_cfg.robot_name if robot_cfg.robot_name != "schunk" else "svh"]
    config_path = get_default_config_path(robot_name_enum, RetargetingType.position, HandType.right)
    override = dict(add_dummy_free_joint=True)
    if robot_cfg.urdf_subpath:
        override["urdf_path"] = robot_cfg.urdf_subpath

    # The current Inspire URDF models its six virtual wrist coordinates as
    # three prismatic + three ``continuous`` joints.  Pinocchio represents a
    # continuous joint with a 2D quaternion, so the resulting model has
    # ``nq != nv`` and recent dex-retargeting releases reject it as a
    # ``special joint``.  For retargeting these coordinates are ordinary
    # scalar wrist DOFs; make a private, runtime-only copy with equivalent
    # revolute joints and no dummy free joint.  This keeps the checked-in
    # vendor URDF untouched while making the official Inspire config usable.
    if robot_cfg.robot_name == "inspire":
        source_urdf = Path(hands_dir) / robot_cfg.urdf_subpath
        if source_urdf.exists():
            import xml.etree.ElementTree as ET
            tree = ET.parse(str(source_urdf))
            changed = False
            active_joint_names = []
            for joint in tree.getroot().findall("joint"):
                if joint.get("type") != "fixed":
                    active_joint_names.append(joint.get("name"))
                if joint.get("type") != "continuous":
                    continue
                joint.set("type", "revolute")
                limit = joint.find("limit")
                if limit is None:
                    limit = ET.SubElement(joint, "limit")
                # Wide finite limits satisfy Pinocchio while preserving the
                # intended unbounded wrist rotation in practice.
                limit.set("lower", "-1000000")
                limit.set("upper", "1000000")
                limit.set("effort", limit.get("effort", "1000"))
                limit.set("velocity", limit.get("velocity", "3.14"))
                changed = True
            if changed:
                # The temporary URDF lives outside the asset directory;
                # absolutize mesh references so yourdfpy does not emit noisy
                # resolution warnings (and so an optional SAPIEN scene can
                # still load the visuals).
                for mesh in tree.getroot().iter("mesh"):
                    filename = mesh.get("filename")
                    if filename and not os.path.isabs(filename):
                        mesh.set("filename", str((source_urdf.parent / filename).resolve()))
                patched_dir = Path(tempfile.mkdtemp(prefix="dexplore-inspire-"))
                patched_urdf = patched_dir / source_urdf.name
                tree.write(str(patched_urdf), encoding="utf-8", xml_declaration=True)
                override["urdf_path"] = str(patched_urdf)
                override["add_dummy_free_joint"] = False
                # The new Inspire asset contains six virtual wrist joints in
                # addition to the six finger source joints and six mimic
                # joints.  Unlike the legacy 12-DOF asset, none can be left
                # fixed with the current SeqRetargeting API (the converter
                # supplies no fixed_qpos).  Optimize the complete 18-DOF
                # scalar model and ignore mimic tags; this yields the native
                # Inspire ordering expected by Dexplore/Isaac Gym.
                override["target_joint_names"] = active_joint_names
                override["ignore_mimic_joint"] = True
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()

    # The converter only needs the robot joint ordering and a ``set_qpos``
    # sink; it does not simulate the hand in SAPIEN.  Prefer a headless path
    # by default because SAPIEN 2.x may initialize a Vulkan display and
    # segfault on compute-only machines.  Set DEXPLORE_ENABLE_SAPIEN=1 to
    # retain the legacy scene setup when interactive rendering is desired.
    if os.environ.get("DEXPLORE_ENABLE_SAPIEN", "0") != "1":
        parsed_urdf = urdf.URDF.load(config.urdf_path,
                                     add_dummy_free_joints=config.add_dummy_free_joint,
                                     build_scene_graph=False)
        active_names = [name for name, joint in parsed_urdf.joint_map.items()
                        if getattr(joint, "type", "fixed") != "fixed"]
        retarget2sapien = np.asarray(
            [retargeting.joint_names.index(name) for name in active_names], dtype=int)

        class _Robot:
            def set_qpos(self, qpos):
                self.qpos = qpos

        class _Scene:
            def update_render(self):
                return None

        return retargeting, _Robot(), retarget2sapien, None, _Scene()

    # Setup SAPIEN scene
    # SAPIEN 2.x does not expose a public ``Scene()`` constructor; scenes are
    # created by an Engine.  Keep the engine alive for the lifetime of the
    # scene (the returned object is attached below for clarity).
    if hasattr(sapien, "core") and hasattr(sapien.core, "Engine"):
        engine = sapien.core.Engine()
        scene = engine.create_scene()
    else:
        scene = sapien.Scene()
    scene.set_timestep(1 / 30)
    from sapien.asset import create_dome_envmap
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_directional_light(np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
    scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
    scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    urdf_path = str(Path(config.urdf_path))
    robot_urdf = urdf.URDF.load(urdf_path, add_dummy_free_joints=True, build_scene_graph=False)
    temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
    temp_path = os.path.join(temp_dir, os.path.basename(urdf_path))
    robot_urdf.write_xml_file(temp_path)
    robot = loader.load(temp_path)

    sapien_joint_names = [j.name for j in robot.get_active_joints()]
    retarget2sapien = np.array([retargeting.joint_names.index(n) for n in sapien_joint_names]).astype(int)

    # Compute sapien2isaac mapping
    sapien2isaac = robot_cfg.get_sapien2isaac()

    return retargeting, robot, retarget2sapien, sapien2isaac, scene


# ── Main processing ─────────────────────────────────────────────────────────

def process_sequence(name, args, robot_cfg, retargeting, robot, retarget2sapien,
                     sapien2isaac, scene, joints2verts, lbs_weight_info):
    """Process a single GRAB sequence and save the interaction data."""
    import smplx
    profile = bool(os.environ.get("DEXPLORE_PROFILE"))
    t_profile = time.time()
    def mark(label):
        nonlocal t_profile
        if profile:
            now = time.time()
            print(f"    [{label}] {now - t_profile:.2f}s", flush=True)
            t_profile = now

    grab_dir = args.grab_dir
    original_dir = args.original_grab_dir
    object_dir = args.object_dir

    # Load sequence data
    with np.load(os.path.join(grab_dir, "sequences", name, "object.npz"), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f["angles"], f["trans"], str(f["name"])

    sub = name.split("_")[0]
    motion_raw_file = os.path.join(original_dir, sub, "_".join(name.split("_")[1:]) + ".npz")
    if not os.path.exists(motion_raw_file):
        print(f"  Skipping {name}: raw motion file not found")
        return False

    seq_data = parse_npz(os.path.join(grab_dir, "sequences", name, "motion.npz"))
    seq_raw_data = parse_npz(motion_raw_file)
    mark("load")

    n_comps = seq_data["n_comps"]
    gender = seq_data["gender"]
    sbj_id = seq_data["sbj_id"]
    T = seq_data.n_frames
    sbj_vtemp = load_sbj_verts(sbj_id, seq_data, grab_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smpl_model = smplx.create(
        model_path=args.smplx_model_dir, model_type="smplx",
        gender=gender, num_pca_comps=n_comps,
        v_template=sbj_vtemp.astype(np.float32), batch_size=T,
    ).to(device)

    # Update LBS weights for vertex segmentation (once)
    if not lbs_weight_info["done"]:
        idx = smpl_model.lbs_weights.max(dim=1)[1]
        for v in range(idx.shape[0]):
            if idx[v] - 3 > 21:
                joints2verts[idx[v] - 3, v] = True
        lbs_weight_info["done"] = True

    sbj_parms = params2torch(seq_data.body.params, device)
    sbj_parms_raw = params2torch(seq_raw_data.body.params, device)

    # Compute coordinate transform between processed and raw GRAB data
    R1 = sRot.from_rotvec(sbj_parms["global_orient"][0].double().cpu().numpy())
    R2 = sRot.from_rotvec(sbj_parms_raw["global_orient"][0].double().cpu().numpy())
    T2 = seq_raw_data["object"]["params"]["transl"][0].astype(np.float64)
    T1 = obj_trans[0].astype(np.float64)
    R12 = sRot.from_matrix(R1.as_matrix() @ np.linalg.inv(R2.as_matrix()))
    T12 = T1 - R12.apply(T2)

    # Process table poses
    table_angles = seq_raw_data["table"]["params"]["global_orient"][::4].astype(np.float64)
    table_trans = seq_raw_data["table"]["params"]["transl"][::4].astype(np.float64)
    table_angles[:, 0] = 0
    table_angles[:, 1] = 0
    table_angles[:, 2] = -np.pi
    table_trans = R12.apply(table_trans) + T12[np.newaxis, :]

    # Process object poses
    object_angles = seq_raw_data["object"]["params"]["global_orient"][::4].astype(np.float64)
    angle_matrix = sRot.from_rotvec(object_angles).as_matrix()
    angle_matrix = np.transpose(angle_matrix, (0, 2, 1))
    rotation_x_neg90 = sRot.from_euler("x", -np.pi / 2, degrees=False)
    obj_angles = (rotation_x_neg90 * sRot.from_matrix(angle_matrix)).as_rotvec()

    B = sbj_parms["transl"].shape[0]

    # Apply 90-degree X rotation to global orient
    rotation_x_90 = sRot.from_euler("x", np.pi / 2, degrees=False)
    rotated_orient = rotation_x_90 * sRot.from_rotvec(
        sbj_parms["global_orient"].double().cpu().numpy())
    sbj_parms["global_orient"] = torch.tensor(rotated_orient.as_rotvec()).float().to(device)

    # Run SMPL-X forward kinematics
    with torch.no_grad():
        smplx_output = smpl_model(**sbj_parms)
    joints = smplx_output.joints
    pelvis = joints[:, 0].detach().clone().double()
    vertices = smplx_output.vertices.cpu().detach().clone()
    mark("smplx_forward")

    # Load object mesh
    obj_mesh_path = os.path.join(object_dir, f"{obj_name}/{obj_name}.obj")
    # Raw GRAB extractions commonly call this file ``mesh.obj``.  Accept that
    # layout as well so the preparation adapter does not need to duplicate
    # licensed meshes.
    if not os.path.exists(obj_mesh_path):
        obj_mesh_path = os.path.join(object_dir, f"{obj_name}/mesh.obj")
    obj_mesh = trimesh.load(obj_mesh_path, force="mesh")
    # InterAct stores contact labels at a rate that can differ from the
    # canonical motion (e.g. 75 labels for a 300-frame clip).  Resample by
    # nearest frame index once so every downstream tensor has exactly B rows;
    # this also handles the original 120 Hz GRAB labels without assumptions
    # about a fixed stride.
    def _resample_contact(labels, target_len):
        labels = np.asarray(labels)
        if labels.shape[0] == target_len:
            return torch.from_numpy(labels)
        if labels.shape[0] == 0:
            return torch.zeros((target_len,) + labels.shape[1:], dtype=torch.int8)
        indices = np.floor(np.arange(target_len) * labels.shape[0] / target_len).astype(np.int64)
        indices = np.clip(indices, 0, labels.shape[0] - 1)
        return torch.from_numpy(labels[indices])

    contact_object = _resample_contact(seq_data["contact"]["object"], B)
    contact_body = _resample_contact(seq_data["contact"]["body"], B)
    use_distance_contact = bool(getattr(args, "distance_contact", False))
    # Raw GRAB already provides per-vertex contact labels.  Sampling 1024
    # object points and constructing a 10475x1024 distance matrix for every
    # frame is prohibitively expensive over the full corpus, so use those
    # labels by default.  The original geometric-distance path remains
    # available with ``--distance-contact`` for exact legacy reproduction.
    object_points, _ = trimesh.sample.sample_surface_even(
        obj_mesh, count=1024 if use_distance_contact else 128, seed=2024)

    # Compute object vertices over time
    obj_verts_all = []
    for t in range(B):
        rot = sRot.from_rotvec(obj_angles[t]).as_matrix()
        obj_verts_all.append(np.matmul(object_points, rot.T) + obj_trans[t])
    obj_verts_all = torch.from_numpy(np.array(obj_verts_all))
    # Keep a device copy for the optional nearest-object queries below.  The
    # original implementation accidentally performed a ~10M-element distance
    # matrix on CPU for every frame even when SMPL-X was on CUDA.
    obj_verts_all_device = obj_verts_all.to(device)
    vertices_device = smplx_output.vertices.detach()

    ground_height = min(torch.min(obj_verts_all[:, :, 1]), torch.min(vertices[:, :, 1])).cpu()
    mark("contact_setup")

    # Compute contact labels
    # Legacy InterAct caches kept contact labels at 120 Hz while motion was
    # downsampled to 30 Hz.  New raw-to-Dexplore caches store them at 30 Hz.
    contact_stride = 1
    is_contact = contact_object.sum(dim=-1) > 0
    if not use_distance_contact:
        # Vectorized fast path: reduce the 10475 vertex labels to the 52
        # skeleton segments in one operation.  This is mathematically the
        # same contact test as the loop below but avoids ~B*52 tensor scans.
        contact_vertices = contact_body.bool()
        contact_part_label = (contact_vertices.float() @ joints2verts.float().t() > 0).float()
    else:
        thres_non_contact = 0.01
        left_foot = joints[:, 10]
        right_foot = joints[:, 11]
        contact_part_label = []
        for t in range(B):
            verts_contact = contact_body[min(t * contact_stride, contact_body.shape[0] - 1)].unsqueeze(0) > 0
            dis = (vertices_device[t].unsqueeze(0) - obj_verts_all_device[t].unsqueeze(1)).norm(dim=-1)
            min_dis_v = dis.min(dim=0)[0].cpu()
            verts_non_contact = (min_dis_v > thres_non_contact).unsqueeze(0)
            verts_not_on_ground = vertices[t, :, 1] - ground_height > 0.1
            not_on_ground = (verts_not_on_ground * joints2verts).sum(dim=-1) == joints2verts.sum(dim=-1)
            if t > 0:
                delta_left = torch.norm(left_foot[t, [0, 2]] - left_foot[t - 1, [0, 2]])
                delta_right = torch.norm(right_foot[t, [0, 2]] - right_foot[t - 1, [0, 2]])
                left_static = delta_left < 0.02
                right_static = delta_right < 0.02
                if not left_static and not right_static:
                    if delta_left > delta_right:
                        right_static = True
                    else:
                        left_static = True
                if not left_static:
                    not_on_ground[7] = True
                    not_on_ground[10] = True
                if not right_static:
                    not_on_ground[8] = True
                    not_on_ground[11] = True
            contact = torch.any(verts_contact * joints2verts, dim=-1).float()
            non_contact = (verts_non_contact * joints2verts).sum(dim=-1) == joints2verts.sum(dim=-1)
            if not not_on_ground[7] or not not_on_ground[10] or contact[7] > 0 or contact[10] > 0:
                non_contact[7] = False
                non_contact[10] = False
            if not not_on_ground[8] or not not_on_ground[11] or contact[8] > 0 or contact[11] > 0:
                non_contact[8] = False
                non_contact[11] = False
            contact[non_contact] = -1
            contact_part_label.append(contact)
        contact_part_label = torch.stack(contact_part_label)
    mark("contact")

    # Compute relative transforms
    obj_trans_delta = rotation_x_90.apply(obj_trans - pelvis.cpu().numpy())
    table_trans_delta = rotation_x_90.apply(table_trans - pelvis.cpu().numpy())
    root_trans = rotation_x_90.apply(sbj_parms["transl"].double().detach().cpu().numpy())

    rotated_obj = rotation_x_90 * sRot.from_rotvec(obj_angles.astype(np.float64))
    rotated_table = rotation_x_90 * sRot.from_rotvec(table_angles.astype(np.float64))

    # Decode hand PCA to full joint angles
    left_hand_pose = torch.einsum("bi,ij->bj", [sbj_parms["left_hand_pose"], smpl_model.left_hand_components.float()])
    right_hand_pose = torch.einsum("bi,ij->bj", [sbj_parms["right_hand_pose"], smpl_model.right_hand_components.float()])
    pose_aa = torch.cat([
        sbj_parms["global_orient"], sbj_parms["body_pose"],
        left_hand_pose, right_hand_pose,
    ], dim=-1).detach().cpu().numpy().astype(np.float64)

    pose_aa_mj = pose_aa.reshape(-1, 52, 3)[..., SMPL_2_MUJOCO, :].copy()
    contact_part_label = contact_part_label[:, SMPL_2_MUJOCO].clone()

    if isinstance(gender, np.ndarray):
        gender = gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")

    # Build skeleton state for coordinate transform.
    # Skeleton XMLs come from InterMimic's interact2mimic.py and follow the
    # `{model_type}_{dataset_name}_{subject}.xml` pattern (GRAB → smplx_grab_s{N}.xml).
    skeleton_tree = SkeletonTree.from_mjcf(
        os.path.join(args.skeleton_dir, f"smplx_grab_{name.split('_')[0]}.xml"))

    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(B, 52, 4)
    root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, torch.from_numpy(pose_quat), root_trans_offset, is_local=True)

    # Apply upright correction
    pose_quat_global = (
        sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
        * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
    ).as_quat().reshape(B, -1, 4)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
    pose_quat = new_sk_state.local_rotation.numpy().astype(np.float64)

    obj_angles_quat = rotated_obj.as_quat().reshape(B, 4)
    table_angles_quat = rotated_table.as_quat().reshape(B, 4)

    # Re-run SMPL-X with corrected poses
    trans = new_sk_state.global_translation[:, 0, :].detach().clone().to(device)
    pose_aa_t = torch.from_numpy(pose_aa).to(device)
    with torch.no_grad():
        smplx_output = smpl_model(
            body_pose=pose_aa_t[:, 3:66].float(),
            global_orient=pose_aa_t[:, :3].float(),
            left_hand_pose=sbj_parms["left_hand_pose"].float(),
            right_hand_pose=sbj_parms["right_hand_pose"].float(),
            v_template=sbj_vtemp, transl=trans.float(), return_full_pose=True,
        )
    mark("smplx_corrected")
    verts = smplx_output.vertices
    joints = smplx_output.joints
    A = smplx_output.full_pose

    # Compute wrist rotations in exp map
    left_wrist_rot = quat_to_exp_map(
        torch.from_numpy(pose_quat_global[:, 17, :]).float().unsqueeze(0)
        .permute(1, 0, 2).squeeze(1)
        # Convert xyzw -> wxyz
    )
    # Actually: pose_quat_global is xyzw, quat_to_exp_map expects wxyz
    pqg_wxyz_left = torch.cat([
        torch.from_numpy(pose_quat_global[:, 17, 3:4]),
        torch.from_numpy(pose_quat_global[:, 17, :3]),
    ], dim=-1).double()
    pqg_wxyz_right = torch.cat([
        torch.from_numpy(pose_quat_global[:, 36, 3:4]),
        torch.from_numpy(pose_quat_global[:, 36, :3]),
    ], dim=-1).double()
    left_wrist_rot = quat_to_exp_map(pqg_wxyz_left)
    right_wrist_rot = quat_to_exp_map(pqg_wxyz_right)

    # Height fix
    offset = joints[:30, 0] - trans[:30]
    diff_fix = ((verts[:30] - offset[:, None])[:30, ..., -1].min(dim=-1).values).min()
    verts = verts - (joints[:, 0:1] - trans[:, None])
    joints = joints - (joints[:, 0:1] - trans[:, None])
    joints[..., -1] -= diff_fix
    trans[..., -1] -= diff_fix
    verts[..., -1] -= diff_fix

    # Build output tensor
    dof_smpl_all = local_rotation_to_dof_smpl(torch.from_numpy(pose_quat).double())
    trans = trans.cpu()
    obj_new_trans = trans + torch.from_numpy(obj_trans_delta).double()
    table_new_trans = trans + torch.from_numpy(table_trans_delta).double()

    data = torch.zeros((B, 331 + 52 + 52 * 4 + 7))
    data[:, 0:3] = joints[..., 20, :].double()
    data[:, 3:6] = left_wrist_rot
    data[:, 6:51] = dof_smpl_all[:, 17 * 3:32 * 3]
    data[:, 51:54] = joints[..., 21, :].double()
    data[:, 54:57] = right_wrist_rot
    data[:, 57:102] = dof_smpl_all[:, 36 * 3:51 * 3]
    data[:, 102:102 + 32 * 3] = joints[..., SMPL_2_MUJOCO_JOINTS, :].view(B, -1).double()
    data[:, 198:201] = obj_new_trans
    data[:, 201:205] = torch.from_numpy(obj_angles_quat).double()
    data[:, 205:206] = is_contact[:, None]
    contact_part_label_hand = torch.cat([
        contact_part_label[:, 17:33], contact_part_label[:, 36:52],
    ], dim=1)
    data[:, 206:206 + 32] = contact_part_label_hand.double()
    data[:, 238:241] = table_new_trans
    data[:, 241:245] = torch.from_numpy(table_angles_quat).double()
    data[:, 245:245 + 32 * 4] = torch.from_numpy(np.concatenate([
        pose_quat_global[..., 17:33, :], pose_quat_global[..., 36:52, :],
    ], axis=-2)).double().view(-1, 32 * 4)
    mark("assemble")

    # ── Retarget hand joints to robot DOFs ──────────────────────────────────
    hand_joints = torch.cat([
        joints[:, 21:22, :], joints[:, 50:65, :],
        verts[:, 8079:8080, :], verts[:, 7669:7670, :],
        verts[:, 7794:7795, :], verts[:, 7905:7906, :],
        verts[:, 8022:8023, :],
    ], dim=1)

    if robot_cfg.hand_joint_reorder is not None:
        hand_joints = hand_joints[:, robot_cfg.hand_joint_reorder, :]

    indices = retargeting.optimizer.target_link_human_indices
    qposes = []
    retarget_stride = max(1, int(getattr(args, "retarget_stride", 1)))
    sample_indices = list(range(0, joints.shape[0], retarget_stride))
    if sample_indices[-1] != joints.shape[0] - 1:
        sample_indices.append(joints.shape[0] - 1)
    for i in sample_indices:
        ref_value = hand_joints[i][indices, :]
        for _ in range(getattr(args, "retarget_iterations", 1)):
            qpos = retargeting.retarget(ref_value.detach().cpu().numpy())[retarget2sapien]
        robot.set_qpos(qpos)
        qposes.append(qpos)
        scene.update_render()

    qposes = np.asarray(qposes)
    if len(sample_indices) != joints.shape[0]:
        # Linear interpolation is appropriate for the 30 Hz retargeted hand
        # trajectory and avoids hundreds of thousands of expensive NLopt
        # solves when exporting the complete GRAB corpus.
        qposes = np.stack([
            np.interp(np.arange(joints.shape[0]), sample_indices, qposes[:, d])
            for d in range(qposes.shape[1])
        ], axis=1)
    qposes = torch.from_numpy(qposes).double()
    mark("retarget")

    # Apply robot-specific joint reordering
    num_dof = robot_cfg.num_dof
    if robot_cfg.robot_name == "inspire":
        qposes = qposes[:, INSPIRE_RETARGET_REORDER]
    elif sapien2isaac is not None:
        qposes = qposes[:, sapien2isaac]

    data[:, 245 + 32 * 4:245 + 32 * 4 + num_dof] = qposes

    # Save output
    output_dir = os.path.join(args.output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"interaction_hand_{robot_cfg.robot_name}.pt")
    torch.save(data, output_path)
    print(f"  Saved {output_path} ({B} frames)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert GRAB data to Dexplore format")
    parser.add_argument("--robot", type=str, required=True,
                        choices=list(ROBOT_CONFIGS.keys()),
                        help="Robot hand to retarget to")
    parser.add_argument("--grab_dir", type=str, required=True,
                        help="Path to processed GRAB data (contains sequences/ and objects/)")
    parser.add_argument("--original_grab_dir", type=str, required=True,
                        help="Path to original GRAB download (for raw motion data)")
    parser.add_argument("--object_dir", type=str, default="dexplore/data/assets/mjcf/objects",
                        help="Path to object meshes directory (default: dexplore/data/assets/mjcf/objects)")
    parser.add_argument("--smplx_model_dir", type=str, required=True,
                        help="Path to SMPL-X model directory")
    parser.add_argument("--skeleton_dir", type=str, default="dexplore/data/assets/smplx",
                        help="Path to MuJoCo skeleton XMLs (default: dexplore/data/assets/smplx)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for processed motion data (default: data)")
    parser.add_argument("--dex_retarget_dir", type=str, default="dexplore/data/assets",
                        help="Path to robot hand URDFs directory (default: dexplore/data/assets)")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only process sequences matching this substring")
    parser.add_argument("--subject", type=str, default=None,
                        help="Only process sequences for one subject directory (e.g. s1)")
    parser.add_argument("--retarget-iterations", dest="retarget_iterations", type=int, default=1,
                        help="Number of warm-started optimizer calls per frame (default: 1)")
    parser.add_argument("--retarget-stride", dest="retarget_stride", type=int, default=1,
                        help="Retarget every N frames and interpolate (default: 1)")
    parser.add_argument("--distance-contact", action="store_true",
                        help="Recompute contact by nearest sampled object points (slow legacy mode)")
    args = parser.parse_args()

    robot_cfg = ROBOT_CONFIGS[args.robot]
    print(f"Processing GRAB data for robot: {args.robot} ({robot_cfg.num_dof} DOFs)")

    # Setup retargeting
    retargeting, robot, retarget2sapien, sapien2isaac, scene = setup_retargeting(
        robot_cfg, args.dex_retarget_dir)

    # Build vertex segmentation mask
    joints2verts = build_joints2verts_mask()
    lbs_weight_info = {"done": False}

    # Process sequences
    seq_dir = os.path.join(args.grab_dir, "sequences")
    sequences = sorted(os.listdir(seq_dir))
    if args.filter:
        sequences = [s for s in sequences if args.filter in s]
    if args.subject:
        sequences = [s for s in sequences if s.split("_", 1)[0] == args.subject]

    print(f"Found {len(sequences)} sequences to process")
    success, fail = 0, 0
    for name in tqdm(sequences, desc="Processing"):
        try:
            ok = process_sequence(
                name, args, robot_cfg, retargeting, robot,
                retarget2sapien, sapien2isaac, scene, joints2verts, lbs_weight_info)
            if ok:
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  Error processing {name}: {e}")
            fail += 1

    print(f"\nDone: {success} succeeded, {fail} failed")


if __name__ == "__main__":
    main()
