"""完整轨迹数据 provider：stage4 cache + geometry cache + raw GRAB object mesh。

V1.0：手部几何（778 顶点/手）来自 geometry cache，`model_inputs` 做双手
在线表面采样（同一 transition 的 t 与 t+g 复用同一组 face id + 权重，按
(i, gap) 播种）；transition 有效性改为窗口 [i, i+g] 内 (R∨L) 连续
active。object mesh 仍由 MeshProvider 从 raw GRAB 重建。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.task.InteractionTransfer.dataset import fixed_point_indices
from src.task.InteractionTransfer.geometry_cache import gather_surface_points, sample_surface_refs
from src.task.InteractionTransfer.viewer.mesh_provider import MeshProvider

GAPS = (1, 2, 4, 8)


@dataclass
class SequenceBundle:
    """一条序列的全部可視化数据（全部为 world 坐标，米）。"""

    name: str                    # 如 s1/airplane_fly_1
    object_name: str
    subject: str
    raw_rel: str                 # raw GRAB 相对路径
    raw_frames_total: int        # raw GRAB 总帧数
    raw_frame_id: np.ndarray     # (T,) cache 帧对应的 raw 帧
    hand_verts: dict             # side -> (T, 778, 3) world（geometry cache）
    hand_faces: dict             # side -> (1538, 3)
    hand_face_normals: dict      # side -> (T, 1538, 3)（stage4）
    hand_face_centers: dict      # side -> (T, 1538, 3)（stage4，V0.x 模型输入）
    obj_verts_canonical: np.ndarray
    obj_faces: np.ndarray
    obj_world: np.ndarray        # (T, V, 3)
    obj_points_world: np.ndarray    # (T, 4096, 3)
    obj_normals_world: np.ndarray   # (T, 4096, 3)
    right_active: np.ndarray    # (T,)
    left_active: np.ndarray     # (T,)
    object_indices: np.ndarray  # (512,)
    hand_points_per_side: int
    hand_parity_mm: float       # geometry 重建顶点回代 face-center 的最大偏差

    @property
    def frames(self) -> int:
        return len(self.raw_frame_id)

    def valid_gaps(self, i: int) -> tuple[int, ...]:
        """start i 处 (R∨L) 在 [i, i+g] 连续 active 的 gap 集合。"""
        active = self.right_active | self.left_active
        out = []
        for g in GAPS:
            if i + g < self.frames and bool(active[i:i + g + 1].all()):
                out.append(g)
        return tuple(out)

    def valid_transition(self, i: int, gap: int = 1) -> bool:
        if not (0 <= i < self.frames - gap):
            return False
        active = self.right_active | self.left_active
        return bool(active[i:i + gap + 1].all())

    def valid_frames(self, gap: int = 1) -> list[int]:
        return [i for i in range(self.frames - gap) if self.valid_transition(i, gap)]

    def first_valid_frame(self, gap: int = 1) -> int:
        valid = self.valid_frames(gap)
        return valid[0] if valid else 0

    def valid_transition_v010(self, i: int) -> bool:
        """V0.x one-step 有效性：i, i+1 处右手 active 且左手 inactive。"""
        ra, la = self.right_active, self.left_active
        if not (0 <= i < self.frames - 1):
            return False
        return bool(ra[i] and ra[i + 1] and not la[i] and not la[i + 1])

    def valid_frames_v010(self) -> list[int]:
        return [i for i in range(self.frames - 1) if self.valid_transition_v010(i)]

    def first_valid_frame_v010(self) -> int:
        valid = self.valid_frames_v010()
        return valid[0] if valid else 0

    def sample_hands(self, i: int, gap: int, seed_extra: int = 0):
        """双手表面采样；返回 (pts_i, nrm, flow, verts_i, verts_j) world 坐标。"""
        j = i + gap
        rng = np.random.default_rng([7, i, gap, seed_extra])
        pts, nrm, flow, v_i, v_j = [], [], [], [], []
        for side in ("left", "right"):
            face_idx, bary = sample_surface_refs(self.hand_faces[side].shape[0],
                                                 self.hand_points_per_side, rng)
            pts.append(gather_surface_points(self.hand_verts[side][i], self.hand_faces[side], face_idx, bary))
            nrm.append(self.hand_face_normals[side][i][face_idx].astype(np.float32))
            flow.append(gather_surface_points(self.hand_verts[side][j], self.hand_faces[side], face_idx, bary)
                        - pts[-1])
            v_i.append(self.hand_verts[side][i])
            v_j.append(self.hand_verts[side][j])
        return (np.concatenate(pts).astype(np.float32), np.concatenate(nrm).astype(np.float32),
                np.concatenate(flow).astype(np.float32), v_i, v_j)

    def model_inputs(self, i: int, device: torch.device, gap: int = 1) -> dict:
        """V1.0 batched 输入：双手 concat 手点 + Δt（normalized gap）。"""
        j = i + gap
        oi = self.object_indices
        o = self.obj_points_world[i, oi].astype(np.float32)
        center = o.mean(0)
        nxt = self.obj_points_world[j, oi].astype(np.float32)
        hp, hn, hf, _, _ = self.sample_hands(i, gap)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)

        return {
            "object_points": t(o - center),
            "object_normals": t(self.obj_normals_world[i, oi].astype(np.float32)),
            "hand_points": t(hp - center),
            "hand_normals": t(hn),
            "hand_flow": t(hf),
            "object_flow": t(nxt - o),
            "gap": torch.tensor([float(gap)], device=device),
            "center": center,
        }

    def model_inputs_v010(self, i: int, device: torch.device) -> dict:
        """V0.x batched 输入：单右手 1538 stage4 face centers，无 Δt。"""
        j = i + 1
        oi = self.object_indices
        o = self.obj_points_world[i, oi].astype(np.float32)
        center = o.mean(0)
        nxt = self.obj_points_world[j, oi].astype(np.float32)
        hp = self.hand_face_centers["right"][i].astype(np.float32)
        nxt_hp = self.hand_face_centers["right"][j].astype(np.float32)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)

        return {
            "object_points": t(o - center),
            "object_normals": t(self.obj_normals_world[i, oi].astype(np.float32)),
            "hand_points": t(hp - center),
            "hand_normals": t(self.hand_face_normals["right"][i].astype(np.float32)),
            "hand_flow": t(nxt_hp - hp),
            "object_flow": t(nxt - o),
            "center": center,
        }


class TrajectoryProvider:
    """加载 split 内序列；重数据（object mesh）由 MeshProvider 跨序列缓存。"""

    def __init__(self, cache_root: str, split: Path, geometry_root: str, grab_root=None,
                 mano_path=None, device: str = "cuda"):
        self.cache_root = Path(cache_root)
        self.geometry_root = Path(geometry_root)
        self.sequences = [line.strip() for line in
                          Path(split).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.sequences:
            raise ValueError(f"split 为空: {split}")
        self.device = torch.device(device)
        self.mesh = MeshProvider(grab_root, mano_path, device)

    def load(self, name: str) -> SequenceBundle:
        d = self.cache_root / name
        with np.load(d / "shared.npz", allow_pickle=False) as s:
            raw_rel = str(s["source_raw_file"].item())
            raw_frame_id = np.asarray(s["raw_frame_id"], np.int64)
            obj_points = np.asarray(s["obj_points_world"], np.float32)
            obj_normals = np.asarray(s["obj_normals_world"], np.float32)
            object_name = str(s["object_name"].item())
            subject = str(s["subject_id"].item())

        hand_verts, hand_faces, hand_face_normals, hand_face_centers = {}, {}, {}, {}
        right_active = left_active = None
        parity = 0.0
        for side in ("left", "right"):
            with np.load(self.cache_root / name / f"{side}.npz", allow_pickle=False) as h:
                centers = np.asarray(h["hand_points_world"], np.float32)
                hand_face_normals[side] = np.asarray(h["hand_normals_world"], np.float32)
                hand_face_centers[side] = centers
                active = np.asarray(h["obj_candidate_mask_5cm"], bool).any(1)
            if side == "right":
                right_active = active
            else:
                left_active = active
            with np.load(self.geometry_root / name / f"{side}.npz", allow_pickle=False) as g:
                hand_verts[side] = np.asarray(g["hand_vertices_world"], np.float32)
            hand_faces[side] = np.load(self.geometry_root / f"faces_{side}.npy")
            # 重建顶点回代 face-center 的一致性检查（geometry -> stage4）。
            parity = max(parity, float(np.abs(
                hand_verts[side][:, hand_faces[side], :].mean(2) - centers).max()) * 1000.0)

        seq = self.mesh.sequence(raw_rel)
        obj_canonical, obj_faces = self.mesh.object_mesh(object_name)
        R, transl = self.mesh.object_poses(raw_rel, raw_frame_id)
        canonical = torch.from_numpy(obj_canonical).to(self.device)
        obj_world = (R @ canonical.T).transpose(1, 2) + transl.unsqueeze(1)
        obj_world = obj_world.cpu().numpy().astype(np.float32)

        return SequenceBundle(
            name=name, object_name=object_name, subject=subject, raw_rel=raw_rel,
            raw_frames_total=seq.n_frames, raw_frame_id=raw_frame_id,
            hand_verts=hand_verts, hand_faces=hand_faces,
            hand_face_normals=hand_face_normals, hand_face_centers=hand_face_centers,
            obj_verts_canonical=obj_canonical, obj_faces=obj_faces,
            obj_world=obj_world, obj_points_world=obj_points,
            obj_normals_world=obj_normals,
            right_active=right_active, left_active=left_active,
            object_indices=fixed_point_indices(obj_points.shape[1], 512),
            hand_points_per_side=769,
            hand_parity_mm=parity,
        )


def load_model(checkpoint: Path, dense_checkpoint: str, device: torch.device):
    """自动识别 V0.x（action 8D）/ V1.0（action 9D）checkpoint。

    返回 ``(model, epoch)``；``model.is_legacy`` 标记 V0.x（调用方据此切换
    单右手输入 / 无 dt forward / 6D rigid inverse）。
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"]
    action_weight = state.get("message.action.0.weight")
    legacy = action_weight is not None and int(action_weight.shape[-1]) == 8
    if legacy:
        from src.task.InteractionTransfer.viewer.legacy_model import LegacyInteractionTransfer

        model = LegacyInteractionTransfer(dense_checkpoint=dense_checkpoint).to(device).eval()
    else:
        from src.task.InteractionTransfer.model import InteractionTransfer

        model = InteractionTransfer(dense_checkpoint=dense_checkpoint).to(device).eval()
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current)
    model.requires_grad_(False)
    model.is_legacy = legacy
    return model, int(ckpt.get("epoch", -1))
