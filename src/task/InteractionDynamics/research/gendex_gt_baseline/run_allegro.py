"""加载官方 GenDexGrasp CMapAdam 并运行 Allegro 静态优化。"""
from __future__ import annotations

import contextlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@contextlib.contextmanager
def _gendex_import_context(root: Path):
    old_cwd = Path.cwd(); old_path = list(sys.path)
    os.chdir(root); sys.path.insert(0, str(root))
    try:
        yield
    finally:
        os.chdir(old_cwd); sys.path[:] = old_path


def _joined_mesh(handmodel, q: torch.Tensor, particle: int, center: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    meshes = handmodel.get_meshes_from_q(q=q, i=particle)
    vertices=[]; faces=[]; offset=0
    for mesh in meshes:
        vertex = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32)
        face = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.long)
        vertices.append(vertex); faces.append(face + offset); offset += len(vertex)
    return torch.cat(vertices) + center.cpu(), torch.cat(faces)


def _load_official_cmap_adam(root: Path):
    """原样加载官方类，只把 SciPy 已删除的 `as_dcm` 名称替换为等价 `as_matrix`。"""
    path = root / "utils_model/CMapAdam.py"; source = path.read_text()
    source = source.replace(".as_dcm()", ".as_matrix()")
    module = types.ModuleType("gendex_official_cmapadam_compat"); module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.CMapAdam


def run_official_allegro(contact_map_goal: torch.Tensor, object_center: torch.Tensor,
                         gendex_root: Path, steps: int = 100, particles: int = 32,
                         learning_rate: float = 5e-3, seed: int = 0,
                         device: str = "cuda") -> dict:
    """直接调用官方 CMapAdam；仅要求 SciPy 兼容补丁 `as_dcm -> as_matrix`。"""
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    with _gendex_import_context(gendex_root):
        CMapAdam = _load_official_cmap_adam(gendex_root)

        class ChunkedCMapAdam(CMapAdam):
            """官方同一 energy 的显存等价实现，只避免 `.repeat()` 展开完整笛卡尔积。"""
            def compute_energy_align_dist(self):
                hand = self.handmodel.get_surface_points_new()
                contact_parts=[]
                best_distance = torch.full(hand.shape[:2], torch.inf, device=hand.device)
                best_index = torch.zeros(hand.shape[:2], dtype=torch.long, device=hand.device)
                for start in range(0, len(self.object_point_cloud), 256):
                    stop = min(start + 256, len(self.object_point_cloud))
                    points = self.object_point_cloud[start:stop]
                    normals = self.object_normal_cloud[start:stop]
                    delta = hand[:, None] - points[None, :, None]
                    distance = delta.norm(dim=-1)
                    alignment = (delta * normals[None, :, None]).sum(-1) / (distance + 1e-5)
                    aligned = distance * torch.exp(2 * (1 - alignment))
                    contact_distance = torch.sqrt(aligned.min(dim=2).values)
                    contact_parts.append(1 - 2 * (torch.sigmoid(10 * contact_distance) - .5))
                    hand_distance, local_index = distance.transpose(1, 2).min(dim=2)
                    better = hand_distance < best_distance
                    best_distance = torch.where(better, hand_distance, best_distance)
                    best_index = torch.where(better, local_index + start, best_index)
                current = torch.cat(contact_parts, dim=1)
                energy_contact = torch.abs(current - self.contact_value_goal[None]).mean(1)
                nearest_points = self.object_point_cloud[best_index]
                nearest_normals = self.object_normal_cloud[best_index]
                inside = (((nearest_points - hand) * nearest_normals).sum(2) > 0).float()
                energy_penetration = (inside * best_distance).mean(1)
                energy = energy_contact + 100 * energy_penetration
                joint = (F.relu(self.q_current[:, 9:] - self.q_joint_upper) +
                         F.relu(self.q_joint_lower - self.q_current[:, 9:]))
                self.energy = energy + joint.sum(1)
                return (energy, energy_penetration, joint) if self.verbose_energy else energy

        optimizer = ChunkedCMapAdam(robot_name="allegro", contact_map_goal=contact_map_goal,
            num_particles=particles, init_rand_scale=.5, learning_rate=learning_rate,
            running_name="v20_7_gt_mano", energy_func_name="align_dist", device=device,
            verbose_energy=False)
        q_initial = optimizer.get_opt_q().clone()
        chunk_parity = None
        if particles == 1:
            optimizer.handmodel.update_kinematics(q=optimizer.q_current)
            chunked_energy = optimizer.compute_energy_align_dist()
            chunked_grad = torch.autograd.grad(chunked_energy.sum(), optimizer.q_current,
                                               retain_graph=True)[0]
            official_energy = CMapAdam.compute_energy_align_dist(optimizer)
            official_grad = torch.autograd.grad(official_energy.sum(), optimizer.q_current)[0]
            chunk_parity = {"energy_max_abs": float((chunked_energy-official_energy).abs().max()),
                            "gradient_max_abs": float((chunked_grad-official_grad).abs().max())}
            if chunk_parity["energy_max_abs"] > 1e-6 or chunk_parity["gradient_max_abs"] > 1e-5:
                raise RuntimeError(f"分块 GenDex energy parity 失败: {chunk_parity}")

        def energy_rows() -> dict[str, torch.Tensor]:
            optimizer.handmodel.update_kinematics(q=optimizer.q_current)
            optimizer.verbose_energy = True
            contact_plus_penetration, penetration, joint = optimizer.compute_energy()
            optimizer.verbose_energy = False
            joint_sum = joint.sum(1)
            contact = contact_plus_penetration - 100 * penetration
            return {"contact_mae": contact.detach(), "penetration_m": penetration.detach(),
                    "joint_penalty": joint_sum.detach(),
                    "total": (contact_plus_penetration + joint_sum).detach()}

        initial_energy = energy_rows(); history=[]
        for step in range(steps):
            optimizer.step()
            if (step + 1) % 10 == 0 or step + 1 == steps:
                current = energy_rows()
                history.append({"step": step + 1, **{key: float(value.min())
                                                      for key, value in current.items()}})
        final_energy = energy_rows(); best = int(final_energy["total"].argmin())
        q_final = optimizer.get_opt_q().clone()
        initial_vertices, faces = _joined_mesh(optimizer.handmodel, q_initial, best, object_center)
        final_vertices, final_faces = _joined_mesh(optimizer.handmodel, q_final, best, object_center)
        if not torch.equal(faces, final_faces):
            raise RuntimeError("官方 Allegro 初末 mesh topology 不一致")
        optimizer.handmodel.update_kinematics(q=q_final)
        final_surface = optimizer.handmodel.get_surface_points_new()[best].detach().cpu() + object_center.cpu()
    return {"q_initial": q_initial.cpu(), "q_final": q_final.cpu(), "best_particle": best,
        "initial_vertices": initial_vertices, "final_vertices": final_vertices, "faces": faces,
        "final_surface": final_surface, "history": history,
        "chunk_parity": chunk_parity,
        "initial_energy": {key: value.cpu() for key, value in initial_energy.items()},
        "final_energy": {key: value.cpu() for key, value in final_energy.items()}}
