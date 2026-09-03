"""Convert OakInk2 preview annotations to object-centered correspondence Stage3.

OakInk2 preview annotations contain MANO quaternions/translations and per-frame
object poses in the same mocap/world frame.  This exporter reconstructs MANO
vertices with manotorch, transforms hand geometry into each object's frame, and
stores the complete 4096-point object pool required by CorrStaticDatasetV2.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields[0] == "f" and len(fields) >= 4:
            faces.append([int(x.split("/")[0]) - 1 for x in fields[1:4]])
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError(f"invalid object mesh {path}")
    n = np.zeros_like(v)
    if len(f):
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
        for j in range(3):
            np.add.at(n, f[:, j], fn)
    n /= np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-8, None)
    return v, n


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q /= np.clip(np.linalg.norm(q), 1e-12, None)
    w, x, y, z = q
    return np.asarray(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=np.float32,
    )


def _mano_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        d = pickle.load(f, encoding="latin1")
    def raw(x):
        return np.asarray(getattr(x, "r", x), dtype=np.float32)
    template = raw(d["v_template"])
    shapedirs = raw(d["shapedirs"])
    reg = np.asarray(d["J_regressor"][0].toarray(), dtype=np.float32).reshape(778)
    return template, shapedirs, reg


def _smplx_transl(tsl: np.ndarray, betas: np.ndarray, template: np.ndarray,
                  shapedirs: np.ndarray, wrist_reg: np.ndarray) -> np.ndarray:
    shaped = template + np.einsum("vck,k->vc", shapedirs, betas)
    return (np.asarray(tsl, dtype=np.float32) - wrist_reg @ shaped).astype(np.float32)


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[:, faces]
    centers = tri.mean(axis=2).astype(np.float32)
    normals = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
    normals /= np.clip(np.linalg.norm(normals, axis=2, keepdims=True), 1e-8, None)
    return centers, normals.astype(np.float32)


def _object_pool(cache_path: Path, mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as d:
        points = np.asarray(d["points"], dtype=np.float32)
        mesh_path = Path(str(np.asarray(d["mesh_path"]).item()))
    if mesh_path.as_posix() not in mesh_cache:
        mesh_cache[mesh_path.as_posix()] = _mesh(mesh_path)
    mesh_v, mesh_n = mesh_cache[mesh_path.as_posix()]
    idx = np.linspace(0, len(points) - 1, 4096, dtype=np.int64) if len(points) >= 4096 else np.arange(4096) % len(points)
    pool = points[idx]
    nn = cKDTree(mesh_v).query(pool, workers=-1)[1]
    return pool.astype(np.float32), mesh_n[nn].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oakink2-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--mano-model-dir", required=True)
    ap.add_argument("--mano-assets-root", required=True, help="manotorch root containing models/MANO_*.pkl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-sequences", type=int, default=0)
    ap.add_argument("--start-sequence", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = Path(args.oakink2_root).resolve()
    anno_root = root / "downloads/hf/OakInk-v2/anno_preview"
    cache_root = root / "outputs/dataset_audit/cache/object_query_points_canonical_20k_voxel1mm"
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if not anno_root.is_dir() or not cache_root.is_dir():
        raise FileNotFoundError(f"OakInk2 anno/cache missing: {anno_root} / {cache_root}")

    # Import only at runtime so repository tests do not require manotorch.
    from manotorch.manolayer import ManoLayer

    mano_dir = Path(args.mano_model_dir).resolve()
    mano_root = Path(args.mano_assets_root).resolve()
    mano_arrays = {
        side: _mano_arrays(mano_dir / f"MANO_{side.upper()}.pkl")
        for side in ("right", "left")
    }
    layers = {}
    faces = {}
    for side in ("right", "left"):
        layers[side] = ManoLayer(
            mano_assets_root=str(mano_root), rot_mode="quat", side=side,
            center_idx=0, use_pca=False, flat_hand_mean=True,
        ).to(args.device)
        faces[side] = layers[side].th_faces.detach().cpu().numpy().astype(np.int32)

    all_seq_paths = sorted(anno_root.glob("*.pkl"))
    start_sequence = max(0, int(args.start_sequence))
    seq_paths = all_seq_paths[start_sequence:]
    if args.max_sequences > 0:
        seq_paths = seq_paths[: args.max_sequences]
    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []
    skipped: list[dict] = []
    side_counts = Counter()
    for seq_idx, anno_path in enumerate(seq_paths, start=start_sequence):
        with anno_path.open("rb") as f:
            anno = pickle.load(f)
        frame_ids = sorted(set(int(x) for x in anno["mocap_frame_id_list"]) & set(int(x) for x in anno["raw_mano"].keys()))
        if not frame_ids:
            skipped.append({"sequence": anno_path.stem, "reason": "no_common_frames"})
            continue
        for obj_id in anno.get("obj_list", []):
            cache_path = cache_root / (str(obj_id).replace("@", "_") + ".npz")
            if not cache_path.is_file() or obj_id not in anno.get("obj_transf", {}):
                skipped.append({"sequence": anno_path.stem, "object": obj_id, "reason": "missing_object_cache_or_pose"})
                continue
            try:
                obj_pool, obj_normals = _object_pool(cache_path, mesh_cache)
            except Exception as exc:
                skipped.append({"sequence": anno_path.stem, "object": obj_id, "reason": f"object_load:{exc}"})
                continue
            obj_pose_full = np.stack([np.asarray(anno["obj_transf"][obj_id][fid], dtype=np.float32) for fid in frame_ids])
            for side, prefix in (("right", "rh__"), ("left", "lh__")):
                template, shapedirs, wrist_reg = mano_arrays[side]
                obj_pose = obj_pose_full.copy()
                pose_q = torch.cat([anno["raw_mano"][fid][prefix + "pose_coeffs"] for fid in frame_ids], dim=0).to(args.device)
                betas_t = torch.cat([anno["raw_mano"][fid][prefix + "betas"] for fid in frame_ids], dim=0).to(args.device)
                tsl = torch.cat([anno["raw_mano"][fid][prefix + "tsl"] for fid in frame_ids], dim=0).cpu().numpy().astype(np.float32)
                with torch.no_grad():
                    verts = layers[side](pose_coeffs=pose_q, betas=betas_t).verts.detach().cpu().numpy().astype(np.float32)
                world_v = verts + tsl[:, None, :]
                obj_r = obj_pose[:, :3, :3]
                obj_t = obj_pose[:, :3, 3]
                # Row-vector equivalent of T_world_from_object^{-1}: x_obj =
                # (x_world - t_world) @ R_world_from_object.
                hand_obj_v = np.einsum("tck,tkv->tcv", world_v - obj_t[:, None, :], obj_r)
                hand_obj, hand_n = _face_geometry(hand_obj_v, faces[side])
                # Distances are evaluated in object coordinates; object pool is canonical.
                obj_tree = cKDTree(obj_pool)
                dist = obj_tree.query(hand_obj.reshape(-1, 3), workers=-1)[0].reshape(
                    len(frame_ids), faces[side].shape[0]
                ).astype(np.float32)
                active = np.flatnonzero(np.min(dist, axis=1) <= 0.05)
                if len(active) == 0:
                    skipped.append({"sequence": anno_path.stem, "object": obj_id, "side": side, "reason": "no_interacting_frame"})
                    continue
                # The loader applies the same 5 cm interaction rule; trim
                # permanent non-interacting frames to avoid writing redundant
                # geometry that will never be sampled.
                frame_ids_active = [frame_ids[int(i)] for i in active]
                obj_pose = obj_pose[active]
                tsl = tsl[active]
                hand_obj = hand_obj[active]
                hand_n = hand_n[active]
                dist = dist[active]
                pose_q = pose_q[active]
                betas_t = betas_t[active]
                root_pose = np.tile(np.eye(4, dtype=np.float32), (len(frame_ids_active), 1, 1))
                root_q = torch.cat([anno["raw_mano"][fid][prefix + "pose_coeffs"][:, 0, :] for fid in frame_ids_active], dim=0).numpy()
                root_pose[:, :3, :3] = np.stack([_quat_wxyz_to_matrix(q) for q in root_q])
                root_pose[:, :3, 3] = tsl
                q_np = pose_q.cpu().numpy()
                q_xyzw = np.concatenate([q_np[..., 1:], q_np[..., :1]], axis=-1)
                aa = Rotation.from_quat(q_xyzw.reshape(-1, 4)).as_rotvec().reshape(len(frame_ids_active), 16, 3).astype(np.float32)
                mano_transl = np.stack([
                    _smplx_transl(tsl[i], betas_t[i].detach().cpu().numpy(), template, shapedirs, wrist_reg)
                    for i in range(len(frame_ids_active))
                ])
                stem = f"{seq_idx:05d}_{str(obj_id).replace('@', '_')}_{side}"
                out = out_root / f"{stem}.npz"
                if out.exists() and not args.overwrite:
                    continue
                # Runtime sampling deliberately uses the full pool and does not
                # consult a clean-GT candidate mask.  Omitting this optional
                # legacy field also avoids a second nearest-neighbour pass.
                np.savez(
                    out,
                    schema_name=np.asarray("train_corr_static_v2"), schema_version=np.asarray("2.0.0"),
                    dataset_id=np.asarray("oakink2"), dataset_name=np.asarray("OakInk2"),
                    seq_id=np.asarray(f"{anno_path.stem}/{obj_id}/{side}"), side=np.asarray(side),
                    raw_frame_id=np.asarray(frame_ids_active, dtype=np.int32),
                    # Object geometry is canonical and frame-invariant; keep
                    # one [4096,3] copy. CorrStaticDatasetV2 broadcasts it on
                    # read to the usual [T,4096,3] view.
                    obj_points=obj_pool,
                    obj_normals=obj_normals,
                    hand_points=hand_obj, hand_normals=hand_n,
                    hand_to_obj_min_dist=dist,
                    coordinate_frame=np.asarray("object"),
                    hand_root_pose=root_pose, obj_root_pose_world=obj_pose,
                    mano_global_orient=aa[:, 0, :], mano_transl=mano_transl,
                    mano_pose=aa[:, 1:, :].reshape(len(frame_ids_active), 45),
                    mano_betas=betas_t.cpu().numpy().astype(np.float32),
                    mano_v_template=template,
                    mano_use_pca=np.asarray(False), mano_num_pca_comps=np.asarray(45, dtype=np.int64),
                    mano_flat_hand_mean=np.asarray(True), mano_pose_repr=np.asarray("axis_angle"),
                )
                row = {"file": str(out), "seq_id": f"{anno_path.stem}/{obj_id}/{side}", "object_id": str(obj_id), "side": side, "frames": len(frame_ids_active), "min_dist_m": float(dist.min())}
                rows.append(row); side_counts[side] += 1
        print(f"[{seq_idx + 1}/{len(all_seq_paths)}] {anno_path.stem} outputs={len(rows)}", flush=True)
    summary = {"source": str(root), "output_root": str(out_root), "start_sequence": start_sequence, "num_sequences": len(seq_paths), "num_outputs": len(rows), "num_frames": int(sum(r["frames"] for r in rows)), "side_counts": dict(side_counts), "skipped": skipped, "outputs": rows}
    # Parallel shard workers write disjoint summaries; a later merge step can
    # combine them without a race on one shared JSON file.
    (out_root / f"oakink2_stage3_stats_{start_sequence:04d}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("num_sequences", "num_outputs", "num_frames", "side_counts")}, indent=2))


if __name__ == "__main__":
    main()
