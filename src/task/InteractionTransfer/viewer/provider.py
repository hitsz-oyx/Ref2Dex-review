"""完整轨迹数据 provider：stage4 cache + raw GRAB mesh 重建。

与 ``GRABOneStepDataset`` 不同，这里不做 motion 排序、不筛 transition——
viewer 按 cache 帧序（即真实时间序）遍历整条序列，transition 有效性
只决定"当前帧是否可运行模型"，不影响可视化本身。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.task.InteractionTransfer.dataset import fixed_point_indices
from src.task.InteractionTransfer.viewer.mesh_provider import MeshProvider


@dataclass
class SequenceBundle:
    """一条序列的全部可視化数据（全部为 world 坐标，米）。"""

    name: str                    # 如 s1/airplane_fly_1
    object_name: str
    subject: str
    raw_rel: str                 # raw GRAB 相对路径
    raw_frames_total: int        # raw GRAB 总帧数
    raw_frame_id: np.ndarray     # (T,) cache 帧对应的 raw 帧
    mano_verts: np.ndarray       # (T, 778, 3) world
    mano_faces: np.ndarray       # (1538, 3)
    obj_verts_canonical: np.ndarray
    obj_faces: np.ndarray
    obj_world: np.ndarray        # (T, V, 3)
    obj_points_world: np.ndarray    # (T, 4096, 3)
    obj_normals_world: np.ndarray   # (T, 4096, 3)
    hand_points_world: np.ndarray   # (T, 1538, 3)
    hand_normals_world: np.ndarray  # (T, 1538, 3)
    right_active: np.ndarray    # (T,)
    left_active: np.ndarray     # (T,)
    object_indices: np.ndarray  # (512,)
    hand_parity_mm: float       # MANO face centers 与 cache hand points 的最大偏差

    @property
    def frames(self) -> int:
        return len(self.raw_frame_id)

    def valid_transition(self, i: int) -> bool:
        """与 GRABOneStepDataset 完全一致的 one-step 有效性。"""
        if not (0 <= i < self.frames - 1):
            return False
        ra, la = self.right_active, self.left_active
        return bool(ra[i] and ra[i + 1] and not la[i] and not la[i + 1])

    def valid_frames(self) -> list[int]:
        return [i for i in range(self.frames - 1) if self.valid_transition(i)]

    def first_valid_frame(self) -> int:
        valid = self.valid_frames()
        return valid[0] if valid else 0

    def model_inputs(self, i: int, device: torch.device) -> dict:
        """构造与 GRABOneStepDataset.__getitem__(i) 完全一致的 batched 输入。"""
        j = i + 1
        oi, hi = self.object_indices, slice(None)
        o = self.obj_points_world[i, oi].astype(np.float32)
        center = o.mean(0)
        hp = self.hand_points_world[i, hi].astype(np.float32)
        nxt = self.obj_points_world[j, oi].astype(np.float32)
        nxt_hp = self.hand_points_world[j, hi].astype(np.float32)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(device)

        return {
            "object_points": t(o - center),
            "object_normals": t(self.obj_normals_world[i, oi].astype(np.float32)),
            "hand_points": t(hp - center),
            "hand_normals": t(self.hand_normals_world[i, hi].astype(np.float32)),
            "hand_flow": t(nxt_hp - hp),
            "object_flow": t(nxt - o),
            "center": center,
        }


class TrajectoryProvider:
    """加载 split 内序列；重数据（MANO/mesh）由 MeshProvider 跨序列缓存。"""

    def __init__(self, cache_root: str, split: Path, grab_root=None, mano_path=None,
                 device: str = "cuda"):
        self.cache_root = Path(cache_root)
        self.sequences = [line.strip() for line in
                          Path(split).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.sequences:
            raise ValueError(f"split 为空: {split}")
        self.device = torch.device(device)
        self.mesh = MeshProvider(grab_root, mano_path, device)

    def load(self, name: str) -> SequenceBundle:
        d = self.cache_root / name
        with np.load(d / "shared.npz", allow_pickle=False) as s, \
             np.load(d / "right.npz", allow_pickle=False) as r, \
             np.load(d / "left.npz", allow_pickle=False) as l:
            raw_rel = str(s["source_raw_file"].item())
            raw_frame_id = np.asarray(s["raw_frame_id"], np.int64)
            obj_points = np.asarray(s["obj_points_world"], np.float32)
            obj_normals = np.asarray(s["obj_normals_world"], np.float32)
            hand_points = np.asarray(r["hand_points_world"], np.float32)
            hand_normals = np.asarray(r["hand_normals_world"], np.float32)
            right_active = np.asarray(r["obj_candidate_mask_5cm"], bool).any(1)
            left_active = np.asarray(l["obj_candidate_mask_5cm"], bool).any(1)
            object_name = str(s["object_name"].item())
            subject = str(s["subject_id"].item())

        seq = self.mesh.sequence(raw_rel)
        mano_verts, mano_faces = self.mesh.hand_vertices(raw_rel, "right", raw_frame_id)
        centers = mano_verts[:, mano_faces].mean(2)
        parity = float(np.abs(centers - hand_points).max()) * 1000.0

        obj_canonical, obj_faces = self.mesh.object_mesh(object_name)
        R, transl = self.mesh.object_poses(raw_rel, raw_frame_id)
        canonical = torch.from_numpy(obj_canonical).to(self.device)
        obj_world = (R @ canonical.T).transpose(1, 2) + transl.unsqueeze(1)
        obj_world = obj_world.cpu().numpy().astype(np.float32)

        return SequenceBundle(
            name=name, object_name=object_name, subject=subject, raw_rel=raw_rel,
            raw_frames_total=seq.n_frames, raw_frame_id=raw_frame_id,
            mano_verts=mano_verts, mano_faces=mano_faces,
            obj_verts_canonical=obj_canonical, obj_faces=obj_faces,
            obj_world=obj_world, obj_points_world=obj_points,
            obj_normals_world=obj_normals, hand_points_world=hand_points,
            hand_normals_world=hand_normals,
            right_active=right_active, left_active=left_active,
            object_indices=fixed_point_indices(obj_points.shape[1], 512),
            hand_parity_mm=parity,
        )


def load_model(checkpoint: Path, dense_checkpoint: str, device: torch.device):
    from src.task.InteractionTransfer.model import InteractionTransfer

    model = InteractionTransfer(dense_checkpoint=dense_checkpoint).to(device).eval()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    current = model.state_dict()
    current.update(ckpt["model"])
    model.load_state_dict(current)
    model.requires_grad_(False)
    return model, int(ckpt.get("epoch", -1))
