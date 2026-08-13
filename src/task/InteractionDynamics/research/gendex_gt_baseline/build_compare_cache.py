"""由 V20.6 canonical cache 构建 GT MANO→GenDex CMap→Allegro 静态对比。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from src.task.InteractionDynamics.inverse_mano import face_centers
from src.task.InteractionDynamics.penetration import penetration_from_mesh
from src.task.InteractionDynamics.research.gendex_gt_baseline.contact_map import (
    build_gendex_contact_value, build_gendex_gt_map,
)
from src.task.InteractionDynamics.research.gendex_gt_baseline.run_allegro import run_official_allegro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v20-cache", type=Path, required=True)
    parser.add_argument("--gendex-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100); parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    source = torch.load(args.v20_cache, map_location="cpu")
    object_vertices = source["object_vertices"].float(); object_faces = source["object_faces"].long()
    gt_vertices = source["gt_vertices_object"].float(); mano_faces = source["mano_faces"].long()
    old_robot = source["robot_results"]["Allegro"]
    mesh = trimesh.Trimesh(object_vertices.numpy(), object_faces.numpy(), process=False)
    points_np, face_ids = trimesh.sample.sample_surface(mesh, 2048, seed=args.seed)
    object_points = torch.from_numpy(points_np).float()
    object_normals = torch.from_numpy(np.asarray(mesh.face_normals[face_ids])).float()
    object_normals = torch.nn.functional.normalize(object_normals, dim=-1)
    center = object_points.mean(0); centered_points = object_points - center
    mano_surface = face_centers(gt_vertices, mano_faces)
    contact_maps = torch.stack([build_gendex_gt_map(centered_points, object_normals, hand-center)
                                for hand in mano_surface])
    contact_mass = contact_maps[:, :, 6].sum(1); contact_count = (contact_maps[:, :, 6] > .5).sum(1)
    stable_frame = 4 + int(contact_mass[4:].argmax())
    goal = contact_maps[stable_frame]
    high = goal[:, 6] > .5
    euclidean = torch.cdist(centered_points, mano_surface[stable_frame]-center).amin(1)
    sanity = {"high_contact_count": int(high.sum()), "contact_mass": float(goal[:, 6].sum()),
        "high_mean_distance_mm": float(euclidean[high].mean()*1000) if bool(high.any()) else None,
        "low_mean_distance_mm": float(euclidean[~high].mean()*1000) if bool((~high).any()) else None}
    optimized = run_official_allegro(goal.to(args.device), center, args.gendex_root.resolve(),
        steps=args.steps, particles=args.particles, seed=args.seed, device=args.device)
    realized_contact = build_gendex_contact_value(object_points, object_normals,
                                                   optimized["final_surface"])
    realized_high = realized_contact > .5
    intersection = int((high & realized_high).sum()); union = int((high | realized_high).sum())
    realization = {"contact_mass": float(realized_contact.sum()),
        "high_contact_count": int(realized_high.sum()), "high_contact_intersection": intersection,
        "high_contact_union": union, "high_contact_iou": intersection / union if union else 1.,
        "contact_mae": float((goal[:, 6]-realized_contact).abs().mean()),
        "contact_correlation": float(torch.corrcoef(torch.stack((goal[:, 6], realized_contact)))[0, 1])}
    best = optimized["best_particle"]
    initial = {key: float(value[best]) for key, value in optimized["initial_energy"].items()}
    final = {key: float(value[best]) for key, value in optimized["final_energy"].items()}
    penetration = penetration_from_mesh(optimized["final_vertices"].numpy(), object_vertices.numpy(),
                                        object_faces.numpy(), object_points.numpy())
    payload = {"source_raw_file": source["source_raw_file"], "object_name": source["object_name"],
        "subject": source["subject"], "stable_frame": stable_frame, "seed": args.seed,
        "object_vertices": object_vertices, "object_faces": object_faces,
        "object_points": object_points, "object_normals": object_normals,
        "contact_values": goal[:, 6], "contact_mass_by_frame": contact_mass,
        "realized_contact_values": realized_contact,
        "contact_count_by_frame": contact_count, "gt_vertices": gt_vertices[stable_frame],
        "mano_faces": mano_faces, "old_gty_vertices": old_robot["vertices_object"][stable_frame],
        "old_gty_faces": old_robot["faces"], "gendex_initial_vertices": optimized["initial_vertices"],
        "gendex_final_vertices": optimized["final_vertices"], "gendex_faces": optimized["faces"],
        "gendex_final_surface": optimized["final_surface"], "q_initial": optimized["q_initial"],
        "q_final": optimized["q_final"], "best_particle": best, "initial_energy": initial,
        "final_energy": final, "optimization_history": optimized["history"], "map_sanity": sanity,
        "realization": realization,
        "chunk_parity": optimized["chunk_parity"],
        "full_mesh_penetration_mm": penetration.max_penetration_mm if penetration.valid else None,
        "official_commit": "29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7"}
    args.output.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.output)
    print({"stable_frame": stable_frame, "contact_mass_by_frame": contact_mass.tolist(),
        "contact_count_by_frame": contact_count.tolist(), "map_sanity": sanity,
        "best_particle": best, "initial_energy": initial, "final_energy": final,
        "realization": realization,
        "full_mesh_penetration_mm": payload["full_mesh_penetration_mm"]}, flush=True)


if __name__ == "__main__": main()
