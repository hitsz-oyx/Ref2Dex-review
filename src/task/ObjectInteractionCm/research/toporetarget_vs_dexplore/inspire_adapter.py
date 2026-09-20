"""Dexplore Inspire assets adapted to TopoRetarget's generic hand interface.

This pilot preserves Dexplore's independent 12-finger-joint semantics. It is
not a six-actuator hardware model. Original assets are never edited.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

VERSION = "V1.4.2"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
NATIVE_ORDER = [f"joint{i}" for i in range(1, 7)] + [
    f"{finger}_{part}_joint"
    for finger in ("index", "middle", "pinky", "ring")
    for part in ("proximal", "intermediate")
] + ["thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
     "thumb_intermediate_joint", "thumb_distal_joint"]


def info(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare_assets(source_root, destination, sides=("left", "right"), left_assets=None):
    """Materialize hashed URDF/config copies, with original meshes read-only."""
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    configs = destination / "configs"
    (configs / "anchors").mkdir(parents=True)
    provenance = {"work_version": VERSION, "mimic_policy": "dexplore_ignore_mimic_true",
                  "source_files": [], "changes": {}, "native_joint_names": NATIVE_ORDER}
    for side, suffix in (("left", "lh"), ("right", "rh")):
        if side not in sides:
            continue
        side_root = Path(left_assets).resolve() if side == "left" and left_assets is not None else source_root
        source = side_root / f"inspire_hand_{side}.urdf"
        root = ET.parse(source).getroot()
        provenance["source_files"].append(info(source))
        added_wrist = False
        if side == "left" and left_assets is not None:
            # Official dex left provides audited tips but no free wrist. Add
            # Dexplore's exact six-axis chain in this private asset copy.
            root.find("link[@name='base']").set("name", "link6")
            root.find("joint[@name='base_joint']/parent").set("link", "link6")
            reference_path = source_root / "inspire_hand_right.urdf"
            reference_root = ET.parse(reference_path).getroot()
            for i in range(6):
                root.append(copy.deepcopy(reference_root.find(f"link[@name='link{i}']")))
                root.append(copy.deepcopy(reference_root.find(f"joint[@name='joint{i+1}']")))
            provenance["source_files"].append(info(reference_path))
            added_wrist = True
        meshes = set()
        for mesh in root.findall(".//mesh"):
            path = (source.parent / mesh.get("filename")).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            meshes.add(path)
            mesh.set("filename", str(path))
        provenance["source_files"].extend(info(path) for path in sorted(meshes))
        removed = []
        for joint in root.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None:
                removed.append({"joint": joint.get("name"), **mimic.attrib})
                joint.remove(mimic)
            if joint.get("type") == "continuous":
                joint.set("type", "revolute")
                limit = joint.find("limit")
                if limit is None:
                    limit = ET.SubElement(joint, "limit")
                limit.set("lower", "-1000000")
                limit.set("upper", "1000000")
        by_name = {j.get("name"): j for j in root.findall("joint")}
        # Native wrist is translation xyz followed by intrinsic XYZ rotation.
        for i in range(6):
            joint = by_name[f"joint{i + 1}"]
            np.testing.assert_allclose(np.fromstring(joint.find("axis").get("xyz"), sep=" "),
                                       np.eye(3)[i % 3], atol=1e-12)
            np.testing.assert_allclose(np.fromstring(joint.find("origin").get("xyz"), sep=" "), 0)
            np.testing.assert_allclose(np.fromstring(joint.find("origin").get("rpy"), sep=" "), 0)
        full_path = destination / f"inspire_{side}_full.urdf"
        ET.ElementTree(root).write(full_path, encoding="unicode")
        hand = copy.deepcopy(root)
        for element in list(hand):
            if (element.tag == "joint" and element.get("name") in {f"joint{i}" for i in range(1, 7)}
                    or element.tag == "link" and element.get("name") in {f"link{i}" for i in range(6)}):
                hand.remove(element)
        hand_path = destination / f"inspire_{side}_finger.urdf"
        ET.ElementTree(hand).write(hand_path, encoding="unicode")
        anchors = [{"semantic_name": "wrist", "anchor_type": "link_origin", "link_name": "link6"}]
        for semantic, joint in zip(("thumb_cmc", "thumb_mcp", "thumb_ip"),
                                   ("thumb_proximal_yaw_joint", "thumb_intermediate_joint", "thumb_distal_joint")):
            anchors.append({"semantic_name": semantic, "anchor_type": "joint_origin", "joint_name": joint})
        anchors.append({"semantic_name": "thumb_tip", "anchor_type": "link_origin", "link_name": "thumb_tip"})
        for finger in FINGERS[1:]:
            for semantic, part in (("mcp", "proximal"), ("pip", "intermediate")):
                anchors.append({"semantic_name": f"{finger}_{semantic}", "anchor_type": "joint_origin",
                                "joint_name": f"{finger}_{part}_joint"})
            tip_offset = np.fromstring(by_name[f"{finger}_tip_joint"].find("origin").get("xyz"), sep=" ")
            anchors.append({"semantic_name": f"{finger}_dip", "anchor_type": "link_local_point",
                            "link_name": f"{finger}_intermediate", "local_xyz": (tip_offset * .5).tolist(),
                            "notes": "工程锚点：Inspire 无独立 DIP，取 PIP-tip 中点；不按结果调节。"})
            anchors.append({"semantic_name": f"{finger}_tip", "anchor_type": "link_origin", "link_name": f"{finger}_tip"})
        anchor_id = f"inspire_{suffix}_mediapipe21"
        (configs / "anchors" / f"{anchor_id}.yaml").write_text(yaml.safe_dump({
            "profile_id": anchor_id, "layout_name": "mediapipe21", "anchors": anchors,
            "notes": "Dexplore independent-finger pilot; explicit virtual DIP anchors."}, allow_unicode=True))
        joints = hand.findall("joint")
        active = [j for j in joints if j.get("type") != "fixed"]
        spec = {"name": f"inspire_{suffix}", "side": side, "asset_id": "inspire_dexplore_pilot",
                "urdf_relative_path": hand_path.name, "base_link": "link6",
                "semantic_keypoint_layout": "mediapipe21", "keypoint_anchor_profile": anchor_id,
                "dof_order": [j.get("name") for j in active], "neutral_q": [0.] * len(active),
                "expected_link_count": len(hand.findall("link")), "expected_total_joint_count": len(joints),
                "expected_actuated_joint_count": len(active), "expected_fixed_joint_count": len(joints)-len(active),
                "expected_tip_links": [f"{f}_tip" for f in FINGERS],
                "self_collision": {"supported": False}, "notes": "12 independent finger joints; wrist stored separately."}
        assert len(active) == 12
        (configs / f"inspire_{suffix}.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True))
        provenance["changes"][side] = {"removed_mimic_tags": removed,
            "official_left_with_added_wrist": added_wrist,
            "wrist": "six virtual joints factored into base_pose_scene; full URDF retained for FK checks",
            "full_urdf": info(full_path), "finger_urdf": info(hand_path)}
    provenance["tracked_files"] = [{"path": str(p.relative_to(destination)), "sha256": info(p)["sha256"]}
                                   for p in sorted(destination.rglob("*")) if p.is_file()]
    write_json(destination / "asset_manifest.json", provenance)
    return provenance


def load_hand(assets, side):
    from toporetarget.robots.registry import RobotHandRegistry
    root = Path(assets).resolve()
    registry = RobotHandRegistry(config_root=root / "configs")
    return registry.load("inspire_lh" if side == "left" else "inspire_rh", asset_root=root)


def native18_to_state(qpos):
    value = np.asarray(qpos, dtype=np.float64)
    base = np.broadcast_to(np.eye(4), value.shape[:-1] + (4, 4)).copy()
    base[..., :3, :3] = Rotation.from_euler("XYZ", value[..., 3:6]).as_matrix()
    base[..., :3, 3] = value[..., :3]
    return base


def state_to_native18(base, qpos, finger_names):
    base, qpos = np.asarray(base), np.asarray(qpos)
    wrist = np.concatenate((base[..., :3, 3], Rotation.from_matrix(base[..., :3, :3]).as_euler("XYZ")), axis=-1)
    return np.concatenate((wrist, qpos[..., [list(finger_names).index(name) for name in NATIVE_ORDER[6:]]]), axis=-1)
