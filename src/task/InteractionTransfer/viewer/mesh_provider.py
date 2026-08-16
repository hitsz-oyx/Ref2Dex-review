"""MANO hand mesh 与 GRAB object mesh 的加载与刚体变换工具。

复用 ``process/GRAB/raw.py`` 的既有加载路径：
- MANO layer 按 subject v_template 注入（否则 forward 是平均手型，与 GT 不一致）；
- object canonical mesh 从 ``{grab_root}/tools/object_meshes/contact_meshes`` 加载；
- object world pose 直接来自 raw GRAB ``object.params``。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
from smplx import MANO

from process.GRAB.raw import (
    DEFAULT_GRAB_ROOT,
    DEFAULT_MANO_MODEL_DIR,
    GRABSeqData,
    axis_angle_to_rotmat,
    load_object_canonical_mesh,
)

from src.task.InteractionTransfer.inverse_optimize import fit_rigid_twist, rodrigues


def twist_transform(twist: torch.Tensor, points: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """把 rigid twist [B,6] 应用到任意点集 [B,N,3]：v' = R(ω)(v-c)+c+t。"""
    R = rodrigues(twist[:, 3:])
    local = points - center
    return (R @ local.transpose(1, 2)).transpose(1, 2) + center + twist[:, :3].unsqueeze(1)


def flow_to_mesh_twist(flow: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """从 per-point flow [B,N,3] 拟合 rigid twist 并返回 (twist, center)。

    用于把模型输出的 512 点 object flow / 优化得到的 1538 点 hand flow
    提升为整个 mesh 顶点上的刚体变换。
    """
    center = points.mean(dim=1, keepdim=True)
    return fit_rigid_twist(flow, points), center


class MeshProvider:
    """按需缓存 MANO layer、raw sequence 与 object canonical mesh。"""

    def __init__(self, grab_root: str | Path | None = None,
                 mano_path: str | Path | None = None, device: str = "cuda"):
        self.grab_root = Path(grab_root) if grab_root else Path(DEFAULT_GRAB_ROOT) / "data"
        self.mano_path = Path(mano_path) if mano_path else Path(DEFAULT_MANO_MODEL_DIR)
        self.device = torch.device(device)
        self._sequences: dict[str, GRABSeqData] = {}
        self._mano_layers: dict[tuple[str, str], MANO] = {}
        self._object_meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def sequence(self, raw_rel: str) -> GRABSeqData:
        if raw_rel not in self._sequences:
            self._sequences[raw_rel] = GRABSeqData(str(self.grab_root / raw_rel))
        return self._sequences[raw_rel]

    def mano_layer(self, raw_rel: str, side: str = "right") -> MANO:
        key = (raw_rel, side)
        if key not in self._mano_layers:
            seq = self.sequence(raw_rel)
            vtemp_path = self.grab_root / seq.get_hand_vtemp_relpath(side)
            kwargs = {"is_rhand": side == "right", "use_pca": True,
                      "num_pca_comps": seq.n_comps, "flat_hand_mean": True}
            if vtemp_path.exists():
                kwargs["v_template"] = trimesh.load(vtemp_path, process=False).vertices.astype(np.float32)
            layer = MANO(str(self.mano_path), **kwargs).to(self.device)
            layer.requires_grad_(False)
            self._mano_layers[key] = layer
        return self._mano_layers[key]

    @torch.no_grad()
    def hand_vertices(self, raw_rel: str, side: str, frame_ids: np.ndarray,
                      chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """MANO forward，返回 world 顶点 (T,778,3) 与 faces (1538,3)。"""
        layer = self.mano_layer(raw_rel, side)
        params = self.sequence(raw_rel).get_hand_params(side)
        ids = np.asarray(frame_ids, np.int64)
        T = len(ids)

        def pick(name: str, width: int) -> torch.Tensor:
            value = np.asarray(params[name], np.float32)
            return torch.from_numpy(value[ids].reshape(T, width)).to(self.device)

        rot, pose, tsl = pick("global_orient", 3), pick("hand_pose", 24), pick("transl", 3)
        betas = np.asarray(params["betas"], np.float32)
        betas = betas[ids] if betas.ndim > 1 else np.tile(betas[None], (T, 1))
        width = layer.shapedirs.shape[-1]
        betas = betas[:, :width] if betas.shape[1] > width else np.pad(betas, ((0, 0), (0, width - betas.shape[1])))
        betas = torch.from_numpy(betas).to(self.device)

        verts = []
        for start in range(0, T, chunk):
            sl = slice(start, min(start + chunk, T))
            out = layer(global_orient=rot[sl], hand_pose=pose[sl], betas=betas[sl], transl=tsl[sl])
            verts.append(out.vertices.detach().cpu().numpy().astype(np.float32))
        faces = np.asarray(layer.faces, np.int64)
        return np.concatenate(verts, 0), faces

    def object_mesh(self, obj_name: str) -> tuple[np.ndarray, np.ndarray]:
        """canonical object mesh（米制）顶点与面。"""
        if obj_name not in self._object_meshes:
            mesh = load_object_canonical_mesh(obj_name, str(self.grab_root), "m")
            self._object_meshes[obj_name] = (
                np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64))
        return self._object_meshes[obj_name]

    def object_poses(self, raw_rel: str, frame_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """raw GRAB object params -> 每帧 (R (T,3,3), t (T,3))，transl 归一到米。"""
        params = self.sequence(raw_rel).get_object_params()
        ids = np.asarray(frame_ids, np.int64)
        rot = torch.from_numpy(np.asarray(params["global_orient"], np.float32)[ids]).to(self.device)
        transl = np.asarray(params["transl"], np.float32)[ids]
        if np.abs(transl).max() > 5.0:  # 与 stage4 cache 生成一致的 mm -> m 启发式
            transl = transl / 1000.0
        return axis_angle_to_rotmat(rot), torch.from_numpy(transl).to(self.device)
