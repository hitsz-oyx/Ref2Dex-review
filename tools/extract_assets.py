#!/usr/bin/env python3
"""
Extract deterministic assets for Ref2Dex preprocessing and visualization.

For each ARCTIC object:
  - mesh.obj          (canonical mesh, in meters)
  - parts.json        (vertex part membership: 1=top, 0=bottom)
  - sampling.npz      (canonical surface points, normals, face_idx, barycentric, point_id)

For MANO right/left hand:
  - mesh.obj          (canonical flat-hand mesh)
  - sampling.npz      (face_id, barycentric, point_id, finger_id, region_id)
                      (1538 face centers; topology-stable across shapes)

We sample MANO using **face centers** (one point per triangle) rather than
vertex sampling. This guarantees that face i corresponds to the same triangle
across different hand shapes, providing topology-stable correspondence.

This ensures deterministic sampling points across all downstream uses.
"""

import argparse
import json
import os
import os.path as op
import sys

import numpy as np

# Fix numpy compatibility for chumpy (used by smplx/MANO)
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float_
if not hasattr(np, 'complex'):
    np.complex = np.complex_
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'unicode'):
    np.unicode = np.str_
if not hasattr(np, 'str'):
    np.str = np.str_

import torch
import trimesh
from scipy.spatial import cKDTree
from smplx import MANO
from tqdm import tqdm

# --- Paths ---
MANO_MODEL_DIR = "/home/oyx/test_ws/Ref2Dex/dataset/arctic/data/body_models/mano"
ARCTIC_META_DIR = "/home/oyx/test_ws/Ref2Dex/dataset/arctic/data/arctic_data/data/meta"
OBJECT_VTEMPLATE_DIR = op.join(ARCTIC_META_DIR, "object_vtemplates")

# Output asset root
ASSETS_ROOT = op.join(op.dirname(op.abspath(__file__)), "assets")

# All objects in this ARCTIC subset
OBJECT_NAMES = [
    "box", "capsulemachine", "espressomachine", "ketchup",
    "laptop", "microwave", "mixer", "notebook", "phone",
    "scissors", "waffleiron",
]

# Sampling config
NUM_OBJ_POINTS = 2048
SAMPLING_SEED = 42  # deterministic

# MANO joint-to-finger mapping (smplx order: 16 joints)
# 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
JOINT_TO_FINGER = [0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1]

REGION_PALM = 0
REGION_FINGERTIP = 1
REGION_FINGER_PAD = 2


# ============================================================
def interpolate_vertex_attributes(
    face_idx: np.ndarray,
    barycentric: np.ndarray,
    vert_attr: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    fv = faces[face_idx]
    attr0 = vert_attr[fv[:, 0]]
    attr1 = vert_attr[fv[:, 1]]
    attr2 = vert_attr[fv[:, 2]]
    return (
        barycentric[:, 0:1] * attr0
        + barycentric[:, 1:2] * attr1
        + barycentric[:, 2:3] * attr2
    )


def sample_mesh_surface(mesh: trimesh.Trimesh, n_points: int):
    np.random.seed(SAMPLING_SEED)
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points, seed=SAMPLING_SEED)
    vertices = mesh.vertices
    faces = mesh.faces[face_idx]
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    triangles = np.stack([v0, v1, v2], axis=1)
    barycentric = trimesh.triangles.points_to_barycentric(triangles, points)
    barycentric = np.nan_to_num(barycentric, nan=1.0 / 3.0)
    return points, face_idx, barycentric


def extract_objects(assets_root: str = None, num_obj_points: int = None):
    """Extract canonical object mesh + sampling to assets."""
    assets_root = assets_root or ASSETS_ROOT
    num_obj_points = num_obj_points or NUM_OBJ_POINTS
    dst = op.join(assets_root, "objects")
    os.makedirs(dst, exist_ok=True)

    for obj_name in tqdm(OBJECT_NAMES, desc="Objects"):
        src_dir = op.join(OBJECT_VTEMPLATE_DIR, obj_name)
        obj_dst = op.join(dst, obj_name)
        os.makedirs(obj_dst, exist_ok=True)

        # --- Load canonical mesh ---
        mesh_path = op.join(src_dir, "mesh.obj")
        mesh = trimesh.load(mesh_path, process=False)
        # ARCTIC stores in mm -> convert to meters
        if mesh.vertices.max() > 10:
            mesh.vertices = mesh.vertices.copy() / 1000.0

        # --- Save mesh as .obj (in meters) ---
        mesh.export(op.join(obj_dst, "mesh.obj"))

        # --- Save parts.json (if exists) ---
        parts_path = op.join(src_dir, "parts.json")
        parts = None
        if op.exists(parts_path):
            with open(parts_path) as f:
                parts_raw = json.load(f)
            parts = np.array(parts_raw, dtype=bool)
            with open(op.join(obj_dst, "parts.json"), "w") as f:
                json.dump(parts_raw, f)

        # --- Sample surface points ---
        pts, face_idx, bary = sample_mesh_surface(mesh, NUM_OBJ_POINTS)

        # --- Compute canonical normals ---
        mesh.fix_normals()
        vn = mesh.vertex_normals
        normals = interpolate_vertex_attributes(face_idx, bary, vn, mesh.faces)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.clip(norms, 1e-10, None)

        # --- Part membership for each sampled point ---
        if parts is not None:
            tree = cKDTree(mesh.vertices)
            _, nearest_v = tree.query(pts, k=1)
            sample_parts = parts[nearest_v]
        else:
            sample_parts = np.zeros(NUM_OBJ_POINTS, dtype=bool)

        # --- Save sampling ---
        np.savez_compressed(
            op.join(obj_dst, "sampling.npz"),
            points=pts.astype(np.float32),
            normals=normals.astype(np.float32),
            face_idx=face_idx.astype(np.int32),
            barycentric=bary.astype(np.float32),
            point_id=np.arange(NUM_OBJ_POINTS, dtype=np.int32),
            sample_parts=sample_parts,
        )

        # --- Also save the full mesh face array for interpolation ---
        np.savez_compressed(
            op.join(obj_dst, "mesh_faces.npz"),
            faces=mesh.faces.astype(np.int32),
            vertices=mesh.vertices.astype(np.float32),
        )

        print(f"  [{obj_name}] {len(mesh.vertices)} verts, {NUM_OBJ_POINTS} samples saved")


def extract_mano(assets_root: str = None):
    """Extract canonical MANO hand mesh + face-center sampling to assets.

    Sampling strategy: one point per face (the centroid).
    Since the MANO face topology is fixed, this gives topology-stable
    correspondence across different shapes. The point count is fixed
    at 1538 (= number of MANO faces).
    """
    assets_root = assets_root or ASSETS_ROOT
    device = "cpu"

    for is_right in [True, False]:
        side = "right" if is_right else "left"
        dst = op.join(assets_root, "mano", side)
        os.makedirs(dst, exist_ok=True)

        # --- Build MANO layer (canonical flat-hand mean) ---
        mano = MANO(MANO_MODEL_DIR, is_rhand=is_right, use_pca=False,
                     flat_hand_mean=True).to(device)

        # --- Forward to get canonical vertices and joint positions ---
        with torch.no_grad():
            output = mano()
            verts = output.vertices.squeeze(0).cpu().numpy()  # (778, 3)
            joints = output.joints.squeeze(0).cpu().numpy()   # (16, 3)
            faces = mano.faces  # (1538, 3)

        # --- Save canonical mesh as .obj ---
        hand_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        hand_mesh.export(op.join(dst, "mesh.obj"))

        # --- Sample one point per face (centroid) ---
        # face_centers[f] = (verts[v0] + verts[v1] + verts[v2]) / 3
        face_centers = verts[faces].mean(axis=1)  # (1538, 3)

        # --- Compute face normals (in canonical pose, pointing outward) ---
        # We use the canonical mesh's face normals (trimesh gives per-face normals)
        hand_mesh_for_normals = trimesh.Trimesh(
            vertices=verts, faces=faces, process=False
        )
        hand_mesh_for_normals.fix_normals()
        face_normals = hand_mesh_for_normals.face_normals  # (1538, 3)
        # Ensure outward-facing (trimesh's fix_normals should handle this)
        norms = np.linalg.norm(face_normals, axis=-1, keepdims=True)
        face_normals = face_normals / np.clip(norms, 1e-10, None)

        # --- Assign finger_id for each face (by closest joint) ---
        # Joint ordering in smplx MANO: 0=wrist, 1-3=thumb, 4-6=index,
        # 7-9=middle, 10-12=ring, 13-15=pinky (but standard is 16 joints)
        # We use a robust joint->finger mapping that covers all 16 joints
        # JOINT_TO_FINGER indices:
        #   0  = wrist/palm
        #   1-3  = thumb (MCP, PIP, DIP)
        #   4-6  = index
        #   7-9  = middle
        #   10-12 = ring
        #   13-15 = pinky
        JOINT_TO_FINGER_LOCAL = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        tree = cKDTree(joints)
        _, closest_joint = tree.query(face_centers, k=1)
        finger_id = np.array(
            [JOINT_TO_FINGER_LOCAL[j] if j < len(JOINT_TO_FINGER_LOCAL) else 0
             for j in closest_joint],
            dtype=np.int32,
        )

        # --- Region ID ---
        # palm = closest to joint 0 (wrist)
        # fingertip = closest to tip joints (last joint of each finger)
        tip_joints = {3, 6, 9, 12, 15}  # DIP of thumb/index/middle/ring/pinky
        palm_joints = {0}
        region_id = np.zeros(len(face_centers), dtype=np.int32)
        for i in range(len(face_centers)):
            if closest_joint[i] in tip_joints:
                region_id[i] = REGION_FINGERTIP
            elif closest_joint[i] in palm_joints:
                region_id[i] = REGION_PALM
            else:
                region_id[i] = REGION_FINGER_PAD

        num_faces = len(faces)
        # --- Save sampling ---
        np.savez_compressed(
            op.join(dst, "sampling.npz"),
            points=face_centers.astype(np.float32),       # (1538, 3)
            normals=face_normals.astype(np.float32),      # (1538, 3)
            face_idx=np.arange(num_faces, dtype=np.int32),  # face ids 0..1537
            barycentric=np.tile(
                np.array([[1/3, 1/3, 1/3]], dtype=np.float32),
                (num_faces, 1)
            ),
            point_id=np.arange(num_faces, dtype=np.int32),
            finger_id=finger_id,
            region_id=region_id,
            faces=faces.astype(np.int32),
        )

        print(f"  [mano/{side}] {len(verts)} verts, {num_faces} face samples, "
              f"finger_id distribution: "
              f"{dict(zip(*np.unique(finger_id, return_counts=True)))}")


def main():
    parser = argparse.ArgumentParser(description="Extract deterministic assets")
    parser.add_argument("--output_root", type=str, default=ASSETS_ROOT)
    parser.add_argument("--num_obj_points", type=int, default=NUM_OBJ_POINTS)
    parser.add_argument("--mano_only", action="store_true",
                        help="Only extract MANO assets (skip object sampling)")
    args = parser.parse_args()

    output_root = args.output_root
    num_obj_points = args.num_obj_points

    print(f"[Asset Extractor] Output: {output_root}")
    print(f"[Asset Extractor] Num obj points: {num_obj_points}")
    print()

    if not args.mano_only:
        extract_objects(assets_root=output_root, num_obj_points=num_obj_points)
    extract_mano(assets_root=output_root)

    print(f"\nDone. Assets saved to {output_root}")


if __name__ == "__main__":
    main()