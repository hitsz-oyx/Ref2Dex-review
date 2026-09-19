"""Interactive Viser viewer for native Inspire trajectories.

This viewer reads the 18 native Inspire DOFs from
``interaction_hand_inspire.pt``, applies them to the Inspire URDF, and serves
the resulting hand through Viser.  The geometric and RL exports can be
selected independently or overlaid.  With ``--show-mano`` it also renders the
right-hand MANO surface from the canonical InterAct SMPL-X/MANO parameters in
the same object frame.  With ``--all`` it scans the export directory and
provides a trajectory browser: choose a sequence from the dropdown, click the
previous/next buttons, and use independent checkboxes to show any combination
of hand layers.

The URDF is parsed locally instead of using ``viser.extras.ViserUrdf`` because
the latter requires the optional ``yourdfpy`` package, which is not part of the
Isaac Gym ``graspenv`` used by this repository.
"""

from __future__ import annotations

import argparse
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


# Native Dexplore tensors place the robot DOFs after 32 human-hand quaternions.
QPOS_START = 245 + 32 * 4
NUM_DOFS = 18
TIP_LINKS = ("index_tip", "middle_tip", "ring_tip", "pinky_tip", "thumb_tip")

# Isaac Gym's native Inspire DOF order is index, middle, pinky, ring, thumb,
# whereas the actuated joints appear in the URDF XML as thumb, index, middle,
# ring, pinky.  This maps an Isaac-native qpos vector to URDF qpos slots.
NATIVE_TO_URDF = np.asarray(
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9],
    dtype=np.int64,
)


def _vec(text: Optional[str], size: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(size, default, dtype=np.float64)
    values = [float(v) for v in text.split()]
    if len(values) != size:
        raise ValueError(f"Expected {size} values, got {len(values)} in {text!r}")
    return np.asarray(values, dtype=np.float64)


def _origin(element: Optional[ET.Element]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if element is None:
        return out
    out[:3, 3] = _vec(element.get("xyz"), 3)
    rpy = _vec(element.get("rpy"), 3)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return out


def _motion_transform(joint_type: str, axis: np.ndarray, q: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if joint_type == "prismatic":
        out[:3, 3] = axis * q
    elif joint_type in ("revolute", "continuous"):
        out[:3, :3] = Rotation.from_rotvec(axis * q).as_matrix()
    elif joint_type != "fixed":
        raise ValueError(f"Unsupported Inspire joint type: {joint_type}")
    return out


def _matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _resolve_mesh(filename: str, urdf_path: Path) -> Path:
    filename = filename.replace("package://", "").replace("file://", "")
    if filename.startswith("$(find ") and ")/" in filename:
        filename = filename.split(")/", 1)[1]
    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = urdf_path.parent / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"URDF mesh does not exist: {candidate}")
    return candidate


@dataclass
class InspireJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    q_index: int


@dataclass
class InspireVisual:
    link: str
    vertices: np.ndarray
    faces: np.ndarray
    local_transform: np.ndarray


class InspireUrdfModel:
    """Small URDF FK/visual loader sufficient for the Inspire asset."""

    def __init__(self, urdf_path: Path):
        self.urdf_path = urdf_path.resolve()
        root = ET.parse(self.urdf_path).getroot()
        link_names = [link.get("name") for link in root.findall("link")]
        if any(name is None for name in link_names):
            raise ValueError(f"Unnamed link in {self.urdf_path}")
        self.link_names = [str(name) for name in link_names]

        self.joints: List[InspireJoint] = []
        child_links = set()
        q_index = 0
        for element in root.findall("joint"):
            name = element.get("name")
            joint_type = element.get("type", "fixed")
            parent = element.find("parent")
            child = element.find("child")
            if name is None or parent is None or child is None:
                raise ValueError(f"Malformed joint in {self.urdf_path}")
            parent_name = parent.get("link")
            child_name = child.get("link")
            if parent_name is None or child_name is None:
                raise ValueError(f"Malformed parent/child in joint {name}")
            axis_element = element.find("axis")
            axis = _vec(axis_element.get("xyz") if axis_element is not None else None,
                        3, default=0.0)
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
            current_q_index = q_index if joint_type != "fixed" else -1
            if joint_type != "fixed":
                q_index += 1
            self.joints.append(InspireJoint(
                name=str(name), joint_type=joint_type,
                parent=str(parent_name), child=str(child_name),
                origin=_origin(element.find("origin")), axis=axis,
                q_index=current_q_index,
            ))
            child_links.add(str(child_name))
        self.actuated_joints = [joint for joint in self.joints if joint.q_index >= 0]
        if len(self.actuated_joints) != NUM_DOFS:
            raise ValueError(
                f"Expected {NUM_DOFS} actuated Inspire joints, got "
                f"{len(self.actuated_joints)}: {[j.name for j in self.actuated_joints]}"
            )
        self.root_link = next(name for name in self.link_names if name not in child_links)
        self.children: Dict[str, List[InspireJoint]] = {}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)

        self.visuals: List[InspireVisual] = []
        for link in root.findall("link"):
            link_name = str(link.get("name"))
            for visual in link.findall("visual"):
                mesh = visual.find("./geometry/mesh")
                if mesh is None or mesh.get("filename") is None:
                    continue
                mesh_path = _resolve_mesh(mesh.get("filename"), self.urdf_path)
                loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
                if not isinstance(loaded, trimesh.Trimesh):
                    raise TypeError(f"Expected a mesh in {mesh_path}, got {type(loaded)}")
                vertices = np.asarray(loaded.vertices, dtype=np.float32)
                faces = np.asarray(loaded.faces, dtype=np.int32)
                scale = _vec(mesh.get("scale"), 3, default=1.0)
                vertices = vertices * scale.astype(np.float32)
                self.visuals.append(InspireVisual(
                    link=link_name, vertices=vertices, faces=faces,
                    local_transform=_origin(visual.find("origin")),
                ))
        if not self.visuals:
            raise ValueError(f"No visual meshes found in {self.urdf_path}")

    def link_transforms(self, urdf_qpos: np.ndarray) -> Dict[str, np.ndarray]:
        urdf_qpos = np.asarray(urdf_qpos, dtype=np.float64)
        if urdf_qpos.shape != (NUM_DOFS,):
            raise ValueError(f"Expected qpos shape ({NUM_DOFS},), got {urdf_qpos.shape}")
        transforms: Dict[str, np.ndarray] = {self.root_link: np.eye(4, dtype=np.float64)}

        def visit(parent: str) -> None:
            for joint in self.children.get(parent, []):
                transforms[joint.child] = (
                    transforms[parent]
                    @ joint.origin
                    @ _motion_transform(joint.joint_type, joint.axis,
                                         urdf_qpos[joint.q_index] if joint.q_index >= 0 else 0.0)
                )
                visit(joint.child)

        visit(self.root_link)
        if len(transforms) != len(self.link_names):
            missing = sorted(set(self.link_names) - set(transforms))
            raise ValueError(f"URDF contains disconnected links: {missing}")
        return transforms

    def qpos_to_urdf_order(self, native_qpos: np.ndarray) -> np.ndarray:
        native_qpos = np.asarray(native_qpos, dtype=np.float64)
        if native_qpos.shape != (NUM_DOFS,):
            raise ValueError(f"Expected native qpos shape ({NUM_DOFS},), got {native_qpos.shape}")
        urdf_qpos = np.empty_like(native_qpos)
        urdf_qpos[NATIVE_TO_URDF] = native_qpos
        return urdf_qpos


def alignment_report(model: InspireUrdfModel, motion: torch.Tensor, name: str) -> None:
    """Print a small FK-vs-object-center sanity check for a native trajectory."""
    frame_ids = sorted({0, int(motion.shape[0] // 2), int(motion.shape[0] - 1)})
    print(f"{name} FK/object-center check (native qpos -> URDF):")
    for frame_id in frame_ids:
        native = motion[frame_id, QPOS_START:QPOS_START + NUM_DOFS].numpy()
        links = model.link_transforms(model.qpos_to_urdf_order(native))
        tips = np.asarray([links[link][:3, 3] for link in TIP_LINKS])
        object_center = motion[frame_id, 198:201].numpy()
        distances = np.linalg.norm(tips - object_center[None, :], axis=1)
        object_contact = bool(motion[frame_id, 205].item() > 0) if motion.shape[1] > 205 else None
        contact_text = (
            "object-contact" if object_contact else "no-object-contact"
            if object_contact is not None else "contact-unknown"
        )
        print(
            f"  frame {frame_id:04d}: object={np.round(object_center, 3).tolist()} "
            f"min_tip_dist={distances.min():.3f}m "
            f"mean_tip_dist={distances.mean():.3f}m ({contact_text})"
        )
        # A large distance is expected before grasping and after release.  It
        # is only a coordinate warning while the source explicitly marks the
        # object as being in contact.
        if object_contact and distances.min() > 0.15:
            print("    WARNING: object is far from every Inspire fingertip; "
                  "inspect the source coordinate transform, not only the viewer.")


def load_motion(path: Path) -> torch.Tensor:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[1] < QPOS_START + NUM_DOFS:
        raise ValueError(f"Invalid native Inspire tensor: {path} (shape={getattr(data, 'shape', None)})")
    data = data.float().contiguous()
    if not bool(torch.isfinite(data[:, QPOS_START:QPOS_START + NUM_DOFS]).all()):
        raise ValueError(f"Non-finite Inspire qpos values in {path}")
    return data


def _load_mano_side(sequence: str, side: str, interact_root: Path,
                    smplx_model_root: Path) -> tuple:
    """Load the MANO hand surface embedded in canonical SMPL-X.

    InterAct's canonical GRAB representation stores the two 24-D MANO PCA
    blocks inside ``human.npz``.  SMPL-X uses the same 778-vertex MANO hand
    topology; selecting the corresponding vertices/faces gives a true MANO
    mesh in precisely the same canonical frame as the object and body.
    """
    canonical_root = interact_root / "data/grab/sequences_canonical" / sequence
    human_path = canonical_root / "human.npz"
    if not human_path.is_file():
        raise FileNotFoundError(f"Canonical human file not found for MANO: {human_path}")
    with np.load(human_path, allow_pickle=True) as human:
        poses = np.asarray(human["poses"], dtype=np.float32)
        transl = np.asarray(human["trans"], dtype=np.float32)
        vtemp = np.asarray(human["vtemp"], dtype=np.float32)
        gender = str(human["gender"])
    frame_count = min(len(poses), len(transl))
    poses, transl = poses[:frame_count], transl[:frame_count]

    import smplx
    model = smplx.create(
        str(smplx_model_root), model_type="smplx", gender=gender,
        use_pca=True, num_pca_comps=24, batch_size=frame_count,
        v_template=vtemp, ext="npz",
    )
    zeros = torch.zeros((frame_count, 3), dtype=torch.float32)
    pose = torch.from_numpy(poses)
    with torch.no_grad():
        output = model(
            global_orient=pose[:, :3], body_pose=pose[:, 3:66],
            left_hand_pose=pose[:, 66:90], right_hand_pose=pose[:, 90:114],
            jaw_pose=zeros, leye_pose=zeros, reye_pose=zeros,
            expression=torch.zeros((frame_count, 10), dtype=torch.float32),
            transl=torch.from_numpy(transl),
        )

    # The public SMPL-X segmentation is exactly the 778 MANO vertices and
    # 1538 faces for each hand when palm and finger segments are combined.
    segmentation_path = Path(__file__).with_name("smplx_vert_segmentation.json")
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"SMPL-X segmentation file not found: {segmentation_path}")
    import json
    with segmentation_path.open("r", encoding="utf-8") as handle:
        segmentation = json.load(handle)
    prefix = "right" if side == "right" else "left"
    vertex_ids = np.unique(np.asarray(
        segmentation[f"{prefix}Hand"] + segmentation[f"{prefix}HandIndex1"],
        dtype=np.int64,
    ))
    full_faces = np.asarray(model.faces, dtype=np.int64)
    face_mask = np.isin(full_faces, vertex_ids).all(axis=1)
    if vertex_ids.size != 778 or int(face_mask.sum()) != 1538:
        raise ValueError(
            f"Unexpected {side} MANO segmentation: vertices={vertex_ids.size}, "
            f"faces={int(face_mask.sum())}"
        )
    remap = np.full(int(full_faces.max()) + 1, -1, dtype=np.int64)
    remap[vertex_ids] = np.arange(vertex_ids.size, dtype=np.int64)
    faces = remap[full_faces[face_mask]].astype(np.int32)
    vertices = output.vertices[:, vertex_ids].detach().cpu().numpy().astype(np.float32)

    # The Inspire exporter rotates InterAct's canonical frame by +90 degrees
    # around X when assembling the Isaac/Dexplore state.  Its object pose uses
    # the same mapping, so apply it to the MANO surface as well.
    export_rotation = Rotation.from_euler("x", np.pi / 2.0).as_matrix().astype(np.float32)
    vertices = vertices @ export_rotation.T
    return vertices, faces


def load_mano_sequence(sequence: str, sides: tuple, interact_root: Path,
                       smplx_model_root: Path) -> tuple:
    """Load requested MANO sides; return ``{side: vertices}`` and faces."""
    vertices = {}
    faces = {}
    for side in sides:
        vertices[side], faces[side] = _load_mano_side(
            sequence, side, interact_root, smplx_model_root
        )
    return vertices, faces


class ViserHand:
    def __init__(self, server, model: InspireUrdfModel, name: str, color,
                 opacity: float, label_text: str, label_height: float):
        self.model = model
        self.handles = []
        for index, visual in enumerate(model.visuals):
            handle = server.scene.add_mesh_simple(
                f"/{name}/{visual.link}/{index}", visual.vertices, visual.faces,
                color=color, opacity=opacity, material="standard",
            )
            self.handles.append((visual, handle))
        self.label = server.scene.add_label(
            f"/{name}/label", label_text, position=(0.0, 0.0, 0.0),
            font_size_mode="screen", font_screen_scale=0.9,
            depth_test=False, anchor="bottom-center",
        )
        self.label_height = float(label_height)
        self.visible = True

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        for _, handle in self.handles:
            handle.visible = self.visible
        self.label.visible = self.visible

    def update(self, native_qpos: np.ndarray, world_offset=(0.0, 0.0, 0.0)) -> None:
        urdf_qpos = self.model.qpos_to_urdf_order(native_qpos)
        links = self.model.link_transforms(urdf_qpos)
        offset = np.asarray(world_offset, dtype=np.float64)
        for visual, handle in self.handles:
            transform = links[visual.link] @ visual.local_transform
            handle.position = transform[:3, 3] + offset
            handle.wxyz = _matrix_to_wxyz(transform[:3, :3])
        tip_center = np.mean([links[link][:3, 3] for link in TIP_LINKS], axis=0)
        self.label.position = tip_center + offset + np.asarray(
            [0.0, 0.0, self.label_height], dtype=np.float64
        )


class ViserObject:
    def __init__(self, server, name: str, color, opacity: float,
                 label_text: str, label_height: float,
                 mesh_path: Optional[Path] = None):
        self.server = server
        self.name = name
        self.color = color
        self.opacity = float(opacity)
        self.label_height = float(label_height)
        self.handle = None
        self.mesh_path: Optional[Path] = None
        self.visible = True
        self.label = server.scene.add_label(
            f"/{name}/label", label_text, position=(0.0, 0.0, 0.0),
            font_size_mode="screen", font_screen_scale=0.85,
            depth_test=False, anchor="bottom-center",
        )
        self.set_mesh(mesh_path)

    def set_mesh(self, mesh_path: Optional[Path]) -> None:
        """Replace the displayed object mesh when the trajectory changes."""
        if mesh_path is None:
            self.mesh_path = None
            if self.handle is not None:
                self.handle.visible = False
            self.label.visible = False
            return
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected object mesh, got {type(mesh)}")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        if self.handle is None:
            self.handle = self.server.scene.add_mesh_simple(
                f"/{self.name}/mesh", vertices, faces, color=self.color,
                opacity=self.opacity, material="standard",
            )
        else:
            # MeshHandle exposes assignable vertices/faces, so reusing the
            # node avoids stale object geometry in the browser on switches.
            self.handle.vertices = vertices
            self.handle.faces = faces
            self.handle.visible = self.visible
        self.label.visible = self.visible
        self.mesh_path = mesh_path

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if self.handle is not None:
            self.handle.visible = self.visible
        self.label.visible = self.visible and self.mesh_path is not None

    def update(self, frame: torch.Tensor) -> None:
        if self.handle is None:
            return
        position = frame[198:201].numpy()
        xyzw = frame[201:205].numpy()
        self.handle.position = position
        self.handle.wxyz = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        self.label.position = position + np.asarray(
            [0.0, 0.0, self.label_height], dtype=np.float32
        )


class ViserMano:
    """A single dynamic mesh for a MANO reference trajectory."""

    def __init__(self, server, name: str, vertices: np.ndarray, faces: np.ndarray,
                 color, opacity: float, label_text: str, label_height: float):
        self.handle = server.scene.add_mesh_simple(
            f"/{name}", vertices[0], faces, color=color, opacity=opacity,
            material="standard",
        )
        self.label = server.scene.add_label(
            f"/{name}/label", label_text, position=(0.0, 0.0, 0.0),
            font_size_mode="screen", font_screen_scale=0.9,
            depth_test=False, anchor="bottom-center",
        )
        self.label_height = float(label_height)
        self.visible = True

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self.handle.visible = self.visible
        self.label.visible = self.visible

    def set_data(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.handle.vertices = np.asarray(vertices[0], dtype=np.float32)
        self.handle.faces = np.asarray(faces, dtype=np.int32)

    def update(self, vertices: np.ndarray, frame_index: int) -> None:
        current_vertices = np.asarray(vertices[frame_index], dtype=np.float32)
        self.handle.vertices = current_vertices
        self.label.position = current_vertices.mean(axis=0) + np.asarray(
            [0.0, 0.0, self.label_height], dtype=np.float32
        )


def _find_object_mesh(sequence: str, interact_root: Path, object_root: Optional[Path]) -> Optional[Path]:
    object_name = sequence.split("_", 1)[1].split("_", 1)[0]
    candidates = []
    if object_root is not None:
        candidates.extend([object_root / object_name / f"{object_name}.obj",
                           object_root / object_name / "mesh.obj"])
    candidates.extend([
        interact_root / "data/grab/objects" / object_name / f"{object_name}.obj",
        interact_root / "data/grab/objects" / object_name / "mesh.obj",
    ])
    return next((path for path in candidates if path.exists()), None)


def _list_sequences(root: Path) -> List[str]:
    """Return exported sequence names available below an export root."""
    if not root.exists():
        raise FileNotFoundError(f"Export root does not exist: {root}")
    return sorted(
        (
            path.parent.name
            for path in root.glob("*/interaction_hand_inspire.pt")
            if path.is_file()
        ),
        key=_sequence_sort_key,
    )


def _sequence_sort_key(sequence: str) -> tuple:
    """Sort subject IDs numerically (s1, s2, ..., s10) before clip names."""
    match = re.match(r"^s(\d+)(?:_|$)", sequence)
    return (int(match.group(1)) if match else 10**9, sequence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sequence_group = parser.add_mutually_exclusive_group(required=True)
    sequence_group.add_argument("--sequence", help="Single sequence, e.g. s1_cup_lift")
    sequence_group.add_argument(
        "--all", action="store_true",
        help="Browse every sequence shared by the selected export roots",
    )
    parser.add_argument("--source", choices=("geometric", "rl", "both"), default="geometric")
    parser.add_argument(
        "--show-mano", action="store_true",
        help="Render the canonical MANO reference alongside the Inspire hand(s)",
    )
    parser.add_argument(
        "--mano-side", choices=("right", "left", "both"), default="right",
        help="MANO side(s) to render when --show-mano is enabled (default: right)",
    )
    parser.add_argument("--geometric-root", type=Path,
                        default=Path("/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric"))
    parser.add_argument("--rl-root", type=Path,
                        default=Path("/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl"))
    parser.add_argument(
        "--smplx-model-root", type=Path,
        default=Path("/home2/wyy/oyx_ws/Ref2Dex/data/raw_data/ARCTIC/arctic/models"),
        help="SMPL-X model root containing smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz",
    )
    parser.add_argument("--urdf", type=Path,
                        default=Path("dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"))
    parser.add_argument("--interact-root", type=Path, default=Path("/home2/wyy/oyx_ws/InterAct"))
    parser.add_argument("--object-root", type=Path, default=None,
                        help="Optional root containing <object>/<object>.obj")
    parser.add_argument("--hide-object", action="store_true")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--check-only", action="store_true",
                        help="Load the URDF/tensor and print mapping without starting a server")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not args.urdf.is_absolute():
        args.urdf = repo_root / args.urdf

    selected = [name for name in ("geometric", "rl") if args.source in (name, "both")]
    roots = {"geometric": args.geometric_root, "rl": args.rl_root}
    if args.all:
        available = [_list_sequences(roots[name]) for name in selected]
        sequence_names = sorted(
            set.intersection(*(set(names) for names in available)),
            key=_sequence_sort_key,
        )
    else:
        sequence_names = [str(args.sequence)]
        missing = [
            name for name in selected
            if not (roots[name] / sequence_names[0] / "interaction_hand_inspire.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Sequence {sequence_names[0]!r} is missing from export root(s): {missing}"
            )
    if not sequence_names:
        raise RuntimeError("No shared Inspire trajectories were found in the selected roots")
    if args.sequence is not None:
        if args.sequence not in sequence_names:
            raise ValueError(
                f"Sequence {args.sequence!r} is not available for source {args.source}; "
                f"found {len(sequence_names)} sequence(s)"
            )
        initial_sequence_index = sequence_names.index(args.sequence)
    else:
        initial_sequence_index = 0

    def paths_for(sequence: str) -> Dict[str, Path]:
        return {
            name: roots[name] / sequence / "interaction_hand_inspire.pt"
            for name in selected
        }

    def load_sequence(sequence: str) -> Dict[str, torch.Tensor]:
        return {name: load_motion(path) for name, path in paths_for(sequence).items()}

    sequence = sequence_names[initial_sequence_index]
    data = load_sequence(sequence)
    num_frames = min(int(value.shape[0]) for value in data.values())
    mano_sides = {
        "right": ("right",),
        "left": ("left",),
        "both": ("right", "left"),
    }[args.mano_side]
    mano_vertices: Dict[str, np.ndarray] = {}
    mano_faces: Dict[str, np.ndarray] = {}
    if args.show_mano:
        print(f"Loading MANO reference for {sequence} ({', '.join(mano_sides)}) ...")
        mano_vertices, mano_faces = load_mano_sequence(
            sequence, mano_sides, args.interact_root, args.smplx_model_root,
        )
        num_frames = min(
            num_frames, *(int(value.shape[0]) for value in mano_vertices.values())
        )
    model = InspireUrdfModel(args.urdf)
    for name, value in data.items():
        alignment_report(model, value, name)
    if args.check_only:
        print(f"URDF actuated joints ({len(model.actuated_joints)}):")
        for joint in model.actuated_joints:
            print(f"  {joint.q_index:02d} {joint.name}")
        print(f"available trajectories: {len(sequence_names)}")
        print(f"sequence: {sequence}")
        print(f"geometric/RL source frames: {num_frames}")
        for name, value in data.items():
            print(f"{name}: qpos columns [{QPOS_START}:{QPOS_START + NUM_DOFS}], shape={tuple(value.shape)}")
        for side, value in mano_vertices.items():
            print(f"MANO {side}: vertices shape={tuple(value.shape)}, faces={mano_faces[side].shape[0]}")
        return

    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("Install Viser with `pip install viser` to use this viewer") from exc

    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/grid", width=2.0, height=2.0, plane="xy", cell_size=0.1)
    server.scene.add_frame("/origin", show_axes=True, axes_length=0.15, axes_radius=0.006)
    hands = {}
    if "geometric" in data:
        hands["geometric"] = ViserHand(
            server, model, "geometric", (40, 130, 255),
            0.62 if args.source == "both" else 1.0,
            "Geometric retargeting", 0.08,
        )
    if "rl" in data:
        hands["rl"] = ViserHand(
            server, model, "rl", (245, 90, 50),
            0.62 if args.source == "both" else 1.0,
            "RL retargeting", 0.14,
        )
    mano_handles = {}
    if args.show_mano:
        mano_colors = {"right": (55, 210, 120), "left": (190, 85, 225)}
        for side in mano_sides:
            mano_handles[side] = ViserMano(
                server, f"mano_{side}", mano_vertices[side], mano_faces[side],
                mano_colors[side], 0.48, f"MANO {side}",
                0.03 if side == "right" else 0.09,
            )

    objects = {}
    if not args.hide_object:
        object_path = _find_object_mesh(sequence, args.interact_root, args.object_root)
        object_styles = {
            "geometric": ((224, 70, 160), "Geometric reference object", 0.06),
            "rl": ((255, 190, 35), "RL simulated object", 0.12),
        }
        for name in data:
            color, label_text, label_height = object_styles[name]
            objects[name] = ViserObject(
                server, f"{name}_object", color, 0.52,
                label_text, label_height, object_path,
            )
        if object_path is not None:
            print(f"Object mesh: {object_path}")
        else:
            print("Object mesh not found; continuing with hand-only view")

    trajectory_gui = server.gui.add_dropdown(
        "Trajectory", tuple(sequence_names), initial_value=sequence
    )
    previous = server.gui.add_button("Previous trajectory")
    next_trajectory = server.gui.add_button("Next trajectory")
    frame_slider = server.gui.add_slider("Frame", 0, max(0, num_frames - 1), 1,
                                         min(max(args.start_frame, 0), num_frames - 1))
    play = server.gui.add_checkbox("Play", True)
    loop = server.gui.add_checkbox("Loop", True)
    speed = server.gui.add_slider("Speed", 0.1, 3.0, 0.1, 1.0)
    restart = server.gui.add_button("Restart")
    server.gui.add_markdown(
        "### Hand layers\n"
        "🟢 **MANO reference** · 🔵 **Geometric retargeting** · "
        "🟠 **RL retargeting**\n\n"
        "### Object layers\n"
        "🟣 **Geometric reference object** · 🟡 **RL simulated object**"
    )
    layer_checkboxes = {}
    if args.show_mano:
        layer_checkboxes["mano"] = server.gui.add_checkbox(
            "Show MANO reference (green)", True
        )
    if "geometric" in hands:
        layer_checkboxes["geometric"] = server.gui.add_checkbox(
            "Show geometric retargeting (blue)", True
        )
    if "rl" in hands:
        layer_checkboxes["rl"] = server.gui.add_checkbox(
            "Show RL retargeting (orange)", True
        )
    if "geometric" in objects:
        layer_checkboxes["geometric_object"] = server.gui.add_checkbox(
            "Show geometric reference object (magenta)", True
        )
    if "rl" in objects:
        layer_checkboxes["rl_object"] = server.gui.add_checkbox(
            "Show RL simulated object (yellow)", True
        )
    status = server.gui.add_markdown("")
    state = {
        "frame": int(frame_slider.value),
        "visible_layers": {name: True for name in layer_checkboxes},
        "sequence": sequence,
        "sequence_index": initial_sequence_index,
        "data": data,
        "mano_vertices": mano_vertices,
        "num_frames": num_frames,
    }
    lock = threading.Lock()

    def render_frame(frame_index: int) -> None:
        with lock:
            current_data = state["data"]
            current_mano_vertices = state["mano_vertices"]
            current_num_frames = int(state["num_frames"])
            current_sequence = str(state["sequence"])
            current_sequence_index = int(state["sequence_index"])
            visible_layers = dict(state["visible_layers"])
            state["frame"] = frame_index
        frame_index = int(np.clip(frame_index, 0, current_num_frames - 1))
        with lock:
            state["frame"] = frame_index
        for name, hand in hands.items():
            hand_visible = bool(visible_layers.get(name, False))
            hand.set_visible(hand_visible)
            if hand_visible:
                hand.update(current_data[name][frame_index, QPOS_START:QPOS_START + NUM_DOFS].numpy())
        mano_visible = bool(visible_layers.get("mano", False))
        for side, handle in mano_handles.items():
            handle.set_visible(mano_visible)
            if mano_visible:
                handle.update(current_mano_vertices[side], frame_index)
        for name, object_mesh in objects.items():
            object_visible = bool(visible_layers.get(f"{name}_object", False))
            object_mesh.set_visible(object_visible)
            if object_visible:
                object_mesh.update(current_data[name][frame_index])
        # Contact columns remain semantic labels from the geometric reference.
        status_source = "geometric" if "geometric" in current_data else next(iter(current_data))
        object_frame = current_data[status_source][frame_index]
        object_contact = bool(object_frame[205].item() > 0) if object_frame.shape[0] > 205 else False
        left_contacts = int((object_frame[206:222] > 0).sum().item()) if object_frame.shape[0] >= 222 else 0
        right_contacts = int((object_frame[222:238] > 0).sum().item()) if object_frame.shape[0] >= 238 else 0
        contact_text = (
            f"object contact · left `{left_contacts}` / right `{right_contacts}`"
            if object_contact else "no object contact"
        )
        object_delta_text = ""
        if "geometric" in current_data and "rl" in current_data:
            object_delta = torch.linalg.norm(
                current_data["rl"][frame_index, 198:201]
                - current_data["geometric"][frame_index, 198:201]
            ).item()
            object_delta_text = f" · object Δ `{object_delta:.3f} m`"
        display_names = {
            "mano": "MANO", "geometric": "Geometric", "rl": "RL",
            "geometric_object": "Geometric object",
            "rl_object": "RL object",
        }
        shown = [display_names[name] for name, visible in visible_layers.items() if visible]
        shown_text = ", ".join(shown) if shown else "none"
        frame_slider.value = frame_index
        status.content = (
            f"**{current_sequence}** ({current_sequence_index + 1}/{len(sequence_names)}) "
            f"· frame `{frame_index}/{current_num_frames - 1}` · shown `{shown_text}` "
            f"· {contact_text}{object_delta_text}"
        )

    def switch_sequence(index_delta: int) -> None:
        with lock:
            current_index = int(state["sequence_index"])
        next_index = (current_index + int(index_delta)) % len(sequence_names)
        next_sequence = sequence_names[next_index]
        next_data = load_sequence(next_sequence)
        next_num_frames = min(int(value.shape[0]) for value in next_data.values())
        next_mano_vertices = {}
        next_mano_faces = {}
        if args.show_mano:
            print(f"Loading MANO reference for {next_sequence} ({', '.join(mano_sides)}) ...")
            next_mano_vertices, next_mano_faces = load_mano_sequence(
                next_sequence, mano_sides, args.interact_root, args.smplx_model_root,
            )
            next_num_frames = min(
                next_num_frames, *(int(value.shape[0]) for value in next_mano_vertices.values())
            )
        with lock:
            state["sequence"] = next_sequence
            state["sequence_index"] = next_index
            state["data"] = next_data
            state["mano_vertices"] = next_mano_vertices
            state["num_frames"] = next_num_frames
            state["frame"] = 0
        trajectory_gui.value = next_sequence
        frame_slider.max = max(0, next_num_frames - 1)
        frame_slider.value = 0
        if objects:
            object_path = _find_object_mesh(next_sequence, args.interact_root, args.object_root)
            for object_mesh in objects.values():
                object_mesh.set_mesh(object_path)
            if object_path is not None:
                print(f"Object mesh: {object_path}")
        if args.show_mano:
            for side, handle in mano_handles.items():
                handle.set_data(next_mano_vertices[side], next_mano_faces[side])
        for name, value in next_data.items():
            alignment_report(model, value, name)
        render_frame(0)

    @frame_slider.on_update
    def _(_: object) -> None:
        render_frame(int(frame_slider.value))

    @play.on_update
    def _(_: object) -> None:
        return None

    @speed.on_update
    def _(_: object) -> None:
        return None

    @restart.on_click
    def _(_: object) -> None:
        render_frame(0)

    @trajectory_gui.on_update
    def _(_: object) -> None:
        selected_sequence = str(trajectory_gui.value)
        with lock:
            current_index = int(state["sequence_index"])
        selected_index = sequence_names.index(selected_sequence)
        if selected_index != current_index:
            switch_sequence(selected_index - current_index)

    @previous.on_click
    def _(_: object) -> None:
        switch_sequence(-1)

    @next_trajectory.on_click
    def _(_: object) -> None:
        switch_sequence(1)

    # Each checkbox controls one layer independently.  This intentionally does
    # not use an exclusive "view mode": any one, any pair, all three, or none
    # can be displayed with mouse clicks.
    for layer_name, checkbox in layer_checkboxes.items():
        @checkbox.on_update
        def _(_: object, selected_layer=layer_name,
              selected_checkbox=checkbox) -> None:
            with lock:
                state["visible_layers"][selected_layer] = bool(selected_checkbox.value)
            render_frame(int(frame_slider.value))

    render_frame(int(frame_slider.value))
    print(f"Viser server: http://localhost:{args.port}")
    print(
        f"Loaded {len(sequence_names)} trajectory/trajectories. "
        "Use the Trajectory dropdown or Previous/Next buttons to switch."
    )
    print("Use the independent hand-layer checkboxes to control visibility.")

    while True:
        if bool(play.value):
            with lock:
                current_num_frames = int(state["num_frames"])
                next_frame = state["frame"] + 1
            if next_frame >= current_num_frames:
                if bool(loop.value):
                    next_frame = 0
                else:
                    play.value = False
                    next_frame = current_num_frames - 1
            render_frame(next_frame)
        time.sleep(1.0 / max(1.0, float(args.fps) * float(speed.value)))


if __name__ == "__main__":
    main()
