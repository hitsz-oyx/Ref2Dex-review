"""Viser viewer for HRDexDB Inspire F1 frames and frozen-Cm input geometry.

Example:
    PYTHONPATH=. python -m src.task.CmDecoder.viewer \
      --dataset-root /home2/wyy/oyx_ws/HRDexDB/v0 \
      --object apple --scene 2 --port 8095
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import viser
from scipy.spatial import cKDTree

from src.task.CmDecoder.dataset import _eval_surface, _load_hrdex_io, _robot_hand_mesh, _surface_spec, _to_world


def _load_pose(path: Path) -> np.ndarray:
    value = np.loadtxt(path, dtype=np.float32)
    return value.reshape(4, 4)


class InspireViewerData:
    def __init__(self, root: Path, object_name: str, scene: str, *, seed: int = 42, hand_points: int = 1538, object_points: int = 512, poisson_depth: int = 6):
        self.root = root.resolve()
        self.episode = self.root / "inspire_f1" / object_name / str(scene)
        if not self.episode.is_dir():
            raise FileNotFoundError(self.episode)
        self.object_name = object_name
        self.scene = str(scene)
        self.robot_urdf = self.root.parent / "assets" / "robots" / "xarm_inspire_f1_right.urdf"
        self.io = _load_hrdex_io(self.root.parent)
        self.urdf = self.io.parse_urdf(self.robot_urdf)
        self.mesh_cache: dict = {}
        self.reconstruction_template: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.poisson_depth = int(poisson_depth)
        c2r_path = self.episode / "C2R.npy"
        self.c2r = np.asarray(np.load(c2r_path, allow_pickle=False), dtype=np.float32) if c2r_path.exists() else np.eye(4, dtype=np.float32)
        self.poses = self._load_poses()
        self.q_full = self._load_q(len(self.poses))
        self.q = self.q_full[:, 6:]
        mesh_path = self.root / "assets" / "mesh_v2" / object_name / f"{object_name}.obj"
        if not mesh_path.exists():
            mesh_path = self.root / "assets" / "mesh" / object_name / f"{object_name}.obj"
        self.object_mesh = trimesh.load(mesh_path, force="mesh", process=False)
        self.object_face, self.object_bary = _surface_spec(np.asarray(self.object_mesh.vertices), np.asarray(self.object_mesh.faces), object_points, seed)
        vertices, faces = _robot_hand_mesh(self.io, self.urdf, self.q_full[0], self.mesh_cache)
        self.hand_faces = faces
        self.hand_face, self.hand_bary = _surface_spec(vertices, faces, hand_points, seed + 991)

    def reconstruct_hand(self, index: int, points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build Poisson topology once, then deform it from corresponding samples."""
        del index  # topology is shared because hand samples have frame correspondence
        points = np.asarray(points, dtype=np.float32)
        if self.reconstruction_template is None:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                cloud, depth=self.poisson_depth, n_threads=4
            )
            densities = np.asarray(densities)
            if len(densities):
                mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
            extent_pad = 0.003
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                points.min(axis=0) - extent_pad, points.max(axis=0) + extent_pad
            )
            mesh = mesh.crop(bbox)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            template_vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.triangles, dtype=np.int64)
            distances, neighbors = cKDTree(points).query(template_vertices, k=4)
            inverse = 1.0 / np.maximum(distances, 1e-6)
            weights = inverse / inverse.sum(axis=1, keepdims=True)
            self.reconstruction_template = (
                neighbors.astype(np.int64), weights.astype(np.float32), faces
            )
        neighbors, weights, faces = self.reconstruction_template
        vertices = (points[neighbors] * weights[..., None]).sum(axis=1)
        return vertices.astype(np.float32), faces

    def _load_q(self, target_count: int) -> np.ndarray:
        q, times = self.io.load_robot_qpos(self.episode, "inspire_f1")
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        ts_path = self.episode / "raw" / "timestamps" / "timestamp.npy"
        if ts_path.exists():
            video_times = np.asarray(np.load(ts_path, allow_pickle=True), dtype=np.float64).reshape(-1)
            q_video = np.stack([np.interp(video_times, times, q[:, i]) for i in range(12)], axis=1)
            idx = np.linspace(0, len(q_video) - 1, target_count).round().astype(int)
            return q_video[idx].astype(np.float32)
        target = np.arange(float(times[0]), float(times[-1]) + 1e-6, 1.0 / 30.0)
        q30 = np.stack([np.interp(target, times, q[:, i]) for i in range(12)], axis=1)
        idx = np.linspace(0, len(q30) - 1, target_count).round().astype(int)
        return q30[idx].astype(np.float32)

    def _load_poses(self) -> np.ndarray:
        files = sorted((self.episode / "object_6d").glob("pose_*.txt"))
        if not files:
            files = sorted((self.episode / "object_6d_pose_v2").glob("pose_*.txt"))
        poses = np.stack([_load_pose(p) for p in files], axis=0)
        return poses.astype(np.float32)

    def frame(self, index: int):
        index = int(np.clip(index, 0, len(self.q) - 1))
        hand_vertices, hand_faces = _robot_hand_mesh(self.io, self.urdf, self.q_full[index], self.mesh_cache)
        hand_vertices = _to_world(hand_vertices, self.c2r)
        hp, hn = _eval_surface(hand_vertices, hand_faces, self.hand_face, self.hand_bary)
        op_local, on = _eval_surface(np.asarray(self.object_mesh.vertices), np.asarray(self.object_mesh.faces), self.object_face, self.object_bary)
        pose = self.poses[index]
        op = op_local @ pose[:3, :3].T + pose[:3, 3]
        on = on @ pose[:3, :3].T
        flow = None
        if index + 1 < len(self.q):
            next_vertices, next_faces = _robot_hand_mesh(self.io, self.urdf, self.q_full[index + 1], self.mesh_cache)
            next_vertices = _to_world(next_vertices, self.c2r)
            next_points, _ = _eval_surface(next_vertices, next_faces, self.hand_face, self.hand_bary)
            flow = next_points - hp
        return {"hand_vertices": hand_vertices, "hand_faces": hand_faces, "hand_points": hp, "hand_normals": hn, "object_points": op, "object_normals": on, "flow": flow, "q": self.q[index]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--object", dest="object_name", required=True)
    parser.add_argument("--scene", default="0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8095)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poisson-depth", type=int, default=6)
    args = parser.parse_args()
    data = InspireViewerData(args.dataset_root, args.object_name, args.scene, seed=args.seed, poisson_depth=args.poisson_depth)
    server = viser.ViserServer(host=args.host, port=args.port)
    frame = server.gui.add_slider("Frame", min=0, max=len(data.q) - 1, step=1, initial_value=0)
    play = server.gui.add_button("Play"); stop = server.gui.add_button("Stop")
    show_object = server.gui.add_checkbox("Show Object Samples", True)
    hand_mesh_mode = server.gui.add_dropdown(
        "Hand Mesh", ("GT", "Reconstructed", "Both", "Hidden"), initial_value="GT"
    )
    show_points = server.gui.add_checkbox("Show Hand Samples", True)
    show_flow = server.gui.add_checkbox("Show Hand Flow", True)
    hand_point_size = server.gui.add_slider("Hand Point Size", min=0.001, max=0.020, step=0.001, initial_value=0.006)
    object_point_size = server.gui.add_slider("Object Point Size", min=0.001, max=0.020, step=0.001, initial_value=0.004)
    flow_line_width = server.gui.add_slider("Flow Line Width", min=0.5, max=10.0, step=0.5, initial_value=2.0)
    flow_scale = server.gui.add_slider("Flow Length Scale", min=0.0, max=20.0, step=0.5, initial_value=8.0)
    status = server.gui.add_markdown("")
    state = {"playing": False}; handles = []; lock = threading.Lock()

    def render() -> None:
        nonlocal handles
        sample = data.frame(int(frame.value))
        reconstructed = None
        if hand_mesh_mode.value in {"Reconstructed", "Both"}:
            reconstructed = data.reconstruct_hand(
                int(frame.value), sample["hand_points"], sample["hand_normals"]
            )
        with server.atomic():
            for handle in handles:
                handle.remove()
            handles = []
            if show_object.value:
                handles.append(server.scene.add_point_cloud("/world/object/samples", sample["object_points"], colors=(180, 180, 180), point_size=float(object_point_size.value)))
            if hand_mesh_mode.value in {"GT", "Both"}:
                handles.append(server.scene.add_mesh_simple("/world/hand/gt_mesh", sample["hand_vertices"], sample["hand_faces"], color=(80, 160, 255), opacity=0.45 if hand_mesh_mode.value == "Both" else 0.65))
            if reconstructed is not None:
                recon_vertices, recon_faces = reconstructed
                handles.append(server.scene.add_mesh_simple("/world/hand/reconstructed_mesh", recon_vertices, recon_faces, color=(255, 170, 45), opacity=0.50 if hand_mesh_mode.value == "Both" else 0.70))
            if show_points.value:
                handles.append(server.scene.add_point_cloud("/world/hand/samples", sample["hand_points"], colors=(40, 220, 120), point_size=float(hand_point_size.value)))
            if show_flow.value and sample["flow"] is not None:
                starts = sample["hand_points"][::8]
                ends = (sample["hand_points"] + sample["flow"] * float(flow_scale.value))[::8]
                segments = np.stack([starts, ends], axis=1)
                handles.append(server.scene.add_line_segments("/world/hand/flow", segments, colors=(255, 80, 50), line_width=float(flow_line_width.value)))
        flow = sample["flow"]
        flow_mm = "N/A" if flow is None else f"{np.linalg.norm(flow, axis=1).mean() * 1000:.3f} mm"
        status.content = (f"**{data.object_name}/{data.scene}**  \nFrame: **{int(frame.value)}/{len(data.q)-1}**  \n"
                          f"q (rad): `{np.array2string(sample['q'], precision=4)}`  \n"
                          f"Mean hand flow: **{flow_mm}**  \n"
                          f"Point counts: hand **{len(sample['hand_points'])}**, object **{len(sample['object_points'])}**  \n"
                          f"Mesh mode: **{hand_mesh_mode.value}**")

    @frame.on_update
    def _(_):
        with lock:
            render()
    @play.on_click
    def _(_): state["playing"] = True
    @stop.on_click
    def _(_): state["playing"] = False
    for control in (show_object, hand_mesh_mode, show_points, show_flow, hand_point_size, object_point_size, flow_line_width, flow_scale):
        control.on_update(lambda _: render())
    render()
    print(f"CmDecoder Inspire F1 viewer: http://localhost:{args.port}", flush=True)
    while True:
        if state["playing"]:
            frame.value = (int(frame.value) + 1) % (len(data.q))
        time.sleep(1.0 / max(args.fps, 1e-3))


if __name__ == "__main__":
    main()
