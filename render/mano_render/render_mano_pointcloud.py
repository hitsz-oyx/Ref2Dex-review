#!/usr/bin/env python3
"""Load MANO model and save the mesh as a .ply point cloud file."""

import numpy as np
# Fix numpy compatibility for chumpy
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.unicode = str
np.str = str

import open3d as o3d
import torch
from smplx import MANO

MANO_PATH = "/home/oyx/test_ws/Ref2Dex/dataset/arctic/data/body_models/mano"
OUTPUT_PATH = "/home/oyx/test_ws/mano_right_flat.ply"

# Load MANO right-hand model
model = MANO(
    model_path=MANO_PATH,
    is_rhand=True,
    use_pca=False,
    flat_hand_mean=True,
)

# Forward pass with default (flat) pose
output = model()
vertices = output.vertices.detach().squeeze(0).numpy()  # (778, 3)
faces = model.faces  # (1538, 3)

# Create Open3D mesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()

# Save as PLY
o3d.io.write_triangle_mesh(OUTPUT_PATH, mesh)
print(f"Saved mesh to {OUTPUT_PATH}")
print(f"  Vertices: {vertices.shape[0]}")
print(f"  Faces: {faces.shape[0]}")

# Also save a point cloud only version (just vertices)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
pcd_path = "/home/oyx/test_ws/mano_right_flat_pointcloud.ply"
o3d.io.write_point_cloud(pcd_path, pcd)
print(f"Saved point cloud to {pcd_path}")

# Save face centroids as point cloud
face_centroids = vertices[faces].mean(axis=1)  # (1538, 3)
centroid_pcd = o3d.geometry.PointCloud()
centroid_pcd.points = o3d.utility.Vector3dVector(face_centroids)
centroid_path = "/home/oyx/test_ws/mano_right_flat_face_centroids.ply"
o3d.io.write_point_cloud(centroid_path, centroid_pcd)
print(f"Saved face centroids to {centroid_path}")
print(f"  Face centroids: {face_centroids.shape[0]} points")