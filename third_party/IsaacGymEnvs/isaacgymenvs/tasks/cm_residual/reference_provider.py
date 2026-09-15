"""Legacy DExplore reference reconstruction plus corrected-reference audit metadata."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .dexplore_observation import (
    calc_heading_quat_inv,
    compute_sdf,
    exp_map_to_quat,
    object_points_world,
    quat_rotate,
    quat_to_exp_map,
)


REFERENCE_DIM = 428
SOURCE_WIDTH = 598
FPS = 30.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference(values: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values)
    result[1:] = (values[1:] - values[:-1]) * FPS
    return result


def _sample_object_surface(mesh_path: Path) -> torch.Tensor:
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("DExplore reference reconstruction requires trimesh") from exc
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected one object mesh at {mesh_path}")
    points, _ = trimesh.sample.sample_surface_even(mesh, count=256, seed=2024)
    if len(points) == 0:
        raise ValueError(f"Object sampling returned no points for {mesh_path}")
    if len(points) < 256:
        points = np.concatenate((points, points[:256 - len(points)]), axis=0)
    return torch.as_tensor(points[:256], dtype=torch.float32)


class ReferenceProvider:
    """Rebuild the exact legacy fields consumed by the released teacher.

    ``path`` remains the corrected CmResidual artifact and is used only for its
    eligibility/provenance metadata.  Teacher reference and reset state come
    from the separately checksummed DExplore source tensor.
    """

    def __init__(self, path: str | Path, device: torch.device | str,
                 expected_sha256: str | None = None, *,
                 source_tensor: str | Path,
                 source_sha256: str,
                 frame_start: int,
                 frame_end: int,
                 object_mesh: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Corrected reference artifact not found: {self.path}")
        self.sha256 = _sha256(self.path)
        if expected_sha256 and self.sha256 != str(expected_sha256).lower():
            raise ValueError(f"Corrected reference SHA256 mismatch: {self.sha256} != {expected_sha256}")
        self.manifest_path = self.path.with_name("manifest.json")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Reference manifest not found: {self.manifest_path}")
        self.metadata = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.training_eligible = bool(self.metadata.get("training_eligible", False))
        if self.metadata.get("coordinate_frame") != "world" or self.metadata.get("quaternion_order") != "xyzw":
            raise ValueError("Corrected reference must use world-frame xyzw poses")

        self.source_path = Path(source_tensor).expanduser().resolve()
        if not self.source_path.is_file():
            raise FileNotFoundError(f"DExplore source tensor not found: {self.source_path}")
        self.source_sha256 = _sha256(self.source_path)
        if self.source_sha256 != str(source_sha256).lower():
            raise ValueError(f"DExplore source SHA256 mismatch: {self.source_sha256} != {source_sha256}")
        source = torch.load(self.source_path, map_location="cpu", weights_only=True)
        if not isinstance(source, torch.Tensor) or source.ndim != 2 or source.shape[1] != SOURCE_WIDTH:
            raise ValueError(f"Expected DExplore source [T,{SOURCE_WIDTH}], got {getattr(source, 'shape', None)}")
        source = source.to(dtype=torch.float32)
        if not torch.isfinite(source).all():
            raise ValueError("DExplore source contains non-finite values")
        self.frame_start, self.frame_end = int(frame_start), int(frame_end)
        if self.frame_start < 1 or self.frame_end >= len(source) - 1 or self.frame_start > self.frame_end:
            raise ValueError(f"Invalid DExplore frame interval [{self.frame_start},{self.frame_end}] for {len(source)} frames")
        manifest_range = tuple(int(value) for value in self.metadata.get("training_frame_range", ()))
        if manifest_range and manifest_range != (self.frame_start, self.frame_end):
            raise ValueError(f"Legacy interval {self.frame_start,self.frame_end} disagrees with manifest {manifest_range}")

        self.object_mesh = Path(object_mesh).expanduser().resolve()
        if not self.object_mesh.is_file():
            raise FileNotFoundError(f"DExplore object mesh not found: {self.object_mesh}")
        object_points = _sample_object_surface(self.object_mesh)
        self.device = torch.device(device)
        self.object_points = object_points.to(self.device)
        self._source = source
        full_reference = self._build_reference(source, object_points)
        section = slice(self.frame_start, self.frame_end + 1)
        self.hoi_data = full_reference[section].to(self.device)
        self.robot_q = source[section, 373:391].to(self.device)
        self.robot_dq = _difference(source[:, 373:391])[section].to(self.device)
        self.object_state = torch.cat((
            source[:, 198:205],
            _difference(source[:, 198:201]),
            _difference(quat_to_exp_map(source[:, 201:205])),
        ), dim=-1)[section].to(self.device)
        self.table_pose = source[0, 238:245].to(self.device)
        self.length = int(self.hoi_data.shape[0])
        self.dt = 1.0 / FPS
        contact = source[section, 222:238].amax(dim=-1) > 0.5
        contact_indices = torch.where(contact)[0]
        self.first_contact_index = int(contact_indices[0]) if len(contact_indices) else -1

    @staticmethod
    def _build_reference(source: torch.Tensor, object_points: torch.Tensor) -> torch.Tensor:
        frames = source.shape[0]
        right_hand_rot = exp_map_to_quat(source[:, 54:57])
        right_hand_pos = source[:, 51:54]
        right_dof_pos = source[:, 54:102]
        right_hand_vel = _difference(right_hand_pos)
        right_dof_vel = _difference(right_dof_pos)

        object_pos = source[:, 198:201]
        object_rot = source[:, 201:205]
        object_vel = _difference(object_pos)
        object_ang_vel = _difference(quat_to_exp_map(object_rot))
        object_state = torch.cat((object_pos, object_rot, object_vel, object_ang_vel), dim=-1)

        key_body_pos = source[:, 102:198].view(frames, 32, 3)[:, 16:].clone()
        # The released line is ``body_pos[..., 6:] = remainder(...)``.  Since
        # the final coordinate dimension is 3, that slice is empty: it is an
        # intentional no-op here, not a remainder over body indices 6 onward.
        key_body_vel = _difference(key_body_pos.reshape(frames, -1))
        contact = torch.round(source[:, 205:206])
        contact_parts = torch.round(source[:, 206:238])[:, 16:]

        human_rot = source[:, 245:373].view(frames, 32, 4)[:, 16:].reshape(frames, -1)
        human_rot_exp = quat_to_exp_map(human_rot.reshape(-1, 4)).view(frames, -1)
        human_rot_vel = _difference(human_rot_exp)

        world_points = object_points_world(object_state, object_points)
        ref_ig = compute_sdf(key_body_pos, world_points)
        heading = calc_heading_quat_inv(right_hand_rot)
        ref_ig = quat_rotate(
            heading[:, None].expand(-1, key_body_pos.shape[1], -1), ref_ig).reshape(frames, -1)

        robot_q = source[:, 373:391]
        robot_dq = _difference(robot_q)
        result = torch.cat((
            right_hand_rot, right_hand_pos, right_dof_pos,
            right_hand_vel, right_dof_vel,
            object_state,
            key_body_pos.reshape(frames, -1), contact, contact_parts,
            ref_ig, human_rot, key_body_vel, human_rot_vel,
            robot_q, robot_dq,
        ), dim=-1)
        if result.shape != (frames, REFERENCE_DIM) or not torch.isfinite(result).all():
            raise RuntimeError(f"Invalid reconstructed DExplore reference {tuple(result.shape)}")
        return result

    def frame(self, indices: torch.Tensor, offset: int = 0) -> torch.Tensor:
        selected = indices.to(self.device, dtype=torch.long).clamp(0, self.length - 1)
        selected = (selected + int(offset)).clamp(0, self.length - 1)
        return self.hoi_data.index_select(0, selected)

    def reset_state(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if count <= 0:
            raise ValueError("reset_state count must be positive")
        return (
            self.robot_q[0].expand(count, -1),
            self.robot_dq[0].expand(count, -1),
            self.object_state[0].expand(count, -1),
        )

    def reset_body_state(self, urdf_path: str | Path, body_names) -> tuple[torch.Tensor, ...]:
        """Return deterministic reset FK and adjacent-frame finite differences."""
        repository_root = Path(__file__).resolve().parents[5]
        module_path = repository_root / "src/task/CmDecoderv2/kinematics.py"
        spec = importlib.util.spec_from_file_location("cmresidual_v14_kinematics", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        kinematics = module.InspireKinematics(urdf_path)
        names = tuple(str(name) for name in body_names)
        missing = set(names) - set(kinematics.link_names)
        if missing:
            raise ValueError(f"FK does not contain Isaac Gym bodies: {sorted(missing)}")

        frame_ids = (self.frame_start - 1, self.frame_start, self.frame_start + 1)
        transforms = []
        for frame_id in frame_ids:
            links = kinematics.link_transforms_native(self._source[frame_id, 373:391].numpy())
            transforms.append(np.stack([links[name] for name in names]))
        previous, current, following = transforms
        position = current[:, :3, 3]
        velocity = (following[:, :3, 3] - previous[:, :3, 3]) * (FPS / 2.0)

        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_matrix(current[:, :3, :3]).as_quat()
        relative = following[:, :3, :3] @ np.swapaxes(previous[:, :3, :3], -1, -2)
        angular = Rotation.from_matrix(relative).as_rotvec() * (FPS / 2.0)
        values = tuple(torch.as_tensor(value, dtype=torch.float32, device=self.device)
                       for value in (position, rotation, velocity, angular))
        if not all(torch.isfinite(value).all() for value in values):
            raise RuntimeError("Non-finite reset FK state")
        return values
