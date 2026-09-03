"""Convert OakInk annotations to compact correspondence Stage 3.

OakInk's released annotations are per-view/per-frame pickles rather than the
project's Stage 2 schema.  The official ``general_info`` annotation contains
the MANO root quaternion, local joint quaternions, shape and root translation.
The converter uses those fields to construct the true hand-root frame, then
converts the released 778-vertex hand mesh to 1538 MANO face centers with
true face normals.

The camera-space hand/object geometry is transformed with the camera-space
hand-root pose ``T_camera_root = T_camera_world @ T_world_root``.  The saved
``hand_root_pose`` remains the world-space ``T_world_root`` required by the
training-side MANO reconstruction path.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _parse_name(path: Path) -> tuple[str, str, int, int, int]:
    parts = path.stem.split("__")
    if len(parts) < 5:
        raise ValueError(f"Unexpected OakInk filename: {path.name}")
    # OakInk names are ``sequence__camera__frame__view``.  The final field is
    # the view identifier used by seq_all.json; camera is retained only for
    # diagnostics.
    return parts[0], parts[1], int(parts[4]), int(parts[3]), int(parts[2])


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
            face = []
            for token in fields[1:4]:
                face.append(int(token.split("/")[0]) - 1)
            faces.append(face)
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError(f"Empty/invalid mesh: {path}")
    normals = np.zeros_like(v)
    if len(f):
        tri = v[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for col in range(3):
            np.add.at(normals, f[:, col], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.clip(norms, 1e-8, None)
    return v, normals


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def _transform_normals(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    out = normals @ transform[:3, :3].T
    return (out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-8, None)).astype(np.float32)


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert OakInk/PyTorch3D ``[w, x, y, z]`` quaternion to ``R``."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Invalid OakInk root quaternion: {quaternion!r}")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _oakink_hand_root(general_info: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world/camera root transforms and MANO parameters for one frame."""
    hand_anno = general_info.get("hand_anno")
    if not isinstance(hand_anno, dict):
        raise ValueError("OakInk general_info is missing hand_anno.")
    raw_pose = np.asarray(hand_anno.get("hand_pose"), dtype=np.float32).reshape(-1)
    if raw_pose.size != 64:
        raise ValueError(f"Expected OakInk hand_pose with 64 values, got {raw_pose.shape}")
    raw_pose = raw_pose.reshape(16, 4)
    hand_tsl = np.asarray(hand_anno.get("hand_tsl"), dtype=np.float32).reshape(-1)
    if hand_tsl.shape != (3,) or not np.isfinite(hand_tsl).all():
        raise ValueError(f"Expected finite OakInk hand_tsl shape (3,), got {hand_tsl.shape}")
    cam_from_world = np.asarray(general_info.get("cam_extr"), dtype=np.float32)
    if cam_from_world.shape != (4, 4) or not np.isfinite(cam_from_world).all():
        raise ValueError(f"Expected finite OakInk cam_extr shape (4, 4), got {cam_from_world.shape}")

    root_world = np.eye(4, dtype=np.float32)
    root_world[:3, :3] = _quat_wxyz_to_matrix(raw_pose[0])
    root_world[:3, 3] = hand_tsl
    root_camera = (cam_from_world @ root_world).astype(np.float32)

    # OakInk stores quaternions in PyTorch3D [w, x, y, z] order.  The runner's
    # non-PCA MANO path expects axis-angle45: root orientation plus 15 local
    # joint rotations, with flat_hand_mean=True.
    pose_xyzw = np.concatenate([raw_pose[:, 1:], raw_pose[:, :1]], axis=1)
    pose_axis_angle = Rotation.from_quat(pose_xyzw).as_rotvec().astype(np.float32)
    global_orient = pose_axis_angle[0]
    hand_pose = pose_axis_angle[1:].reshape(-1).astype(np.float32)
    shape = np.asarray(hand_anno.get("hand_shape"), dtype=np.float32).reshape(-1)
    if shape.shape != (10,) or not np.isfinite(shape).all():
        raise ValueError(f"Expected finite OakInk hand_shape shape (10,), got {shape.shape}")
    return root_world, root_camera, global_orient, hand_pose, shape


def _to_hand_root(points_camera: np.ndarray, root_camera: np.ndarray) -> np.ndarray:
    """Transform camera-space points into the current frame's hand-root."""
    rotation = root_camera[:3, :3]
    translation = root_camera[:3, 3]
    return ((np.asarray(points_camera, dtype=np.float32) - translation) @ rotation).astype(np.float32)


def _normals_to_hand_root(normals_camera: np.ndarray, root_camera: np.ndarray) -> np.ndarray:
    rotation = root_camera[:3, :3]
    result = np.asarray(normals_camera, dtype=np.float32) @ rotation
    return (result / np.clip(np.linalg.norm(result, axis=1, keepdims=True), 1e-8, None)).astype(np.float32)


def _load_mano_faces(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    faces = np.asarray(payload["f"], dtype=np.int32)
    if faces.shape != (1538, 3):
        raise ValueError(f"Expected MANO faces with shape (1538, 3), got {faces.shape}")
    return faces


def _load_mano_shape_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    template = np.asarray(payload["v_template"], dtype=np.float32)
    if template.shape != (778, 3):
        raise ValueError(f"Expected MANO v_template with shape (778, 3), got {template.shape}")
    shapedirs = np.asarray(payload["shapedirs"], dtype=np.float32)
    if shapedirs.shape != (778, 3, 10):
        raise ValueError(f"Expected MANO shapedirs with shape (778, 3, 10), got {shapedirs.shape}")
    wrist_regressor = np.asarray(payload["J_regressor"][0].toarray(), dtype=np.float32).reshape(778)
    return template, shapedirs, wrist_regressor


def _smplx_mano_transl(
    wrist_world: np.ndarray,
    betas: np.ndarray,
    v_template: np.ndarray,
    shapedirs: np.ndarray,
    wrist_regressor: np.ndarray,
) -> np.ndarray:
    """Convert OakInk's wrist position to the translation expected by smplx.MANO.

    OakInk's centered ManoLayer adds ``hand_tsl`` to put joint 0 at the
    annotated wrist.  smplx.MANO instead adds ``transl`` after LBS while its
    wrist joint remains at the shaped template offset.  Subtracting that
    offset makes the runtime reconstruction agree with OakInk's released
    vertices and keeps the reconstructed wrist at ``hand_tsl``.
    """
    shaped_vertices = v_template + np.einsum("vck,k->vc", shapedirs, betas)
    wrist_offset = wrist_regressor @ shaped_vertices
    return (np.asarray(wrist_world, dtype=np.float32) - wrist_offset).astype(np.float32)


def _mano_face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    centers = tri.mean(axis=1).astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8, None)
    return centers, normals.astype(np.float32)


def _sample(points: np.ndarray, normals: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic farthest-like coverage without introducing a dependency on
    # a second mesh sampler; OakInk meshes are already densely tessellated.
    idx = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[idx], normals[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oakink-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--max-groups", type=int, default=0, help="0 means all groups.")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--start-group", type=int, default=0, help="Resume at this sorted group index.")
    ap.add_argument(
        "--mano-model",
        default="/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/MANO_RIGHT.pkl",
        help="MANO model pickle used only for the 1538-face topology.",
    )
    args = ap.parse_args()
    root = Path(args.oakink_root).resolve()
    anno = root / "downloads" / "image" / "anno"
    obj_root = root / "downloads" / "image" / "obj"
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    mano_model_path = Path(args.mano_model).resolve()
    mano_faces = _load_mano_faces(mano_model_path)
    mano_v_template, mano_shapedirs, mano_wrist_regressor = _load_mano_shape_data(mano_model_path)

    hand_dir = anno / "hand_v"
    groups: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in hand_dir.iterdir():
        if path.suffix != ".pkl":
            continue
        obj_id, timestamp, view, frame, camera = _parse_name(path)
        groups[(obj_id, timestamp, view)].append(path)
    sorted_group_keys = sorted(groups)
    max_groups = int(args.max_groups)
    group_keys = sorted_group_keys if max_groups <= 0 else sorted_group_keys[:max_groups]

    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    skipped = []
    for group_idx, key in enumerate(group_keys[int(args.start_group):], start=int(args.start_group)):
        obj_id, timestamp, view = key
        paths = sorted(groups[key], key=lambda p: _parse_name(p)[3])
        if args.max_frames > 0:
            paths = paths[: int(args.max_frames)]
        mesh_id = obj_id.split("_", 1)[0]
        mesh_path = obj_root / f"{mesh_id}.obj"
        if not mesh_path.is_file():
            skipped.append({"seq_id": f"{obj_id}/{timestamp}/view{view}", "reason": "missing_object_mesh", "path": str(mesh_path)})
            print(f"[skip] {obj_id}/{timestamp}/view{view}: missing {mesh_path.name}", flush=True)
            continue
        if mesh_id not in mesh_cache:
            mesh_cache[mesh_id] = _mesh(mesh_path)
        canonical_v, canonical_n = mesh_cache[mesh_id]
        obj_points: list[np.ndarray] = []
        obj_normals: list[np.ndarray] = []
        hand_points: list[np.ndarray] = []
        hand_normals: list[np.ndarray] = []
        hand_root_poses: list[np.ndarray] = []
        mano_global_orients: list[np.ndarray] = []
        mano_transls: list[np.ndarray] = []
        mano_poses: list[np.ndarray] = []
        mano_betas: list[np.ndarray] = []
        raw_ids = []
        for path in paths:
            obj_id2, _timestamp, _view, frame, _camera = _parse_name(path)
            hand_cam = np.asarray(_load_pickle(path), dtype=np.float32)
            transf = np.asarray(_load_pickle(anno / "obj_transf" / path.name), dtype=np.float32)
            general = _load_pickle(anno / "general_info" / path.name)
            root_world, root_camera, global_orient, mano_pose, mano_betas_i = _oakink_hand_root(general)
            # obj_transf maps object coordinates to camera coordinates.  Both
            # object and hand are transformed with the same camera-space root
            # pose; this removes translation and the camera/world wrist
            # orientation rather than only subtracting the wrist point.
            obj_cam = _transform(canonical_v, transf)
            obj_n_cam = _transform_normals(canonical_n, transf)
            joints_cam = np.asarray(_load_pickle(anno / "hand_j" / path.name), dtype=np.float32)
            if joints_cam.shape != (21, 3):
                raise ValueError(f"Expected OakInk hand_j shape (21, 3), got {joints_cam.shape}")
            wrist_from_pose = root_camera[:3, 3]
            if not np.allclose(wrist_from_pose, joints_cam[0], atol=2e-5):
                raise ValueError(
                    f"OakInk hand_tsl/root pose disagrees with hand_j wrist for {path.name}: "
                    f"error={np.linalg.norm(wrist_from_pose - joints_cam[0]):.6g} m"
                )
            obj_o = _to_hand_root(obj_cam, root_camera)
            obj_n = _normals_to_hand_root(obj_n_cam, root_camera)
            hand_o = _to_hand_root(hand_cam, root_camera)
            hand_p, hand_n = _mano_face_geometry(hand_o, mano_faces)
            obj_p, obj_nn = _sample(obj_o, obj_n, 4096)
            obj_points.append(obj_p)
            obj_normals.append(obj_nn)
            hand_points.append(hand_p)
            hand_normals.append(hand_n)
            hand_root_poses.append(root_world)
            mano_global_orients.append(global_orient)
            mano_transls.append(
                _smplx_mano_transl(
                    general["hand_anno"]["hand_tsl"],
                    mano_betas_i,
                    mano_v_template,
                    mano_shapedirs,
                    mano_wrist_regressor,
                )
            )
            mano_poses.append(mano_pose)
            mano_betas.append(mano_betas_i)
            raw_ids.append(frame)
        obj_arr = np.asarray(obj_points, dtype=np.float32)
        obj_n_arr = np.asarray(obj_normals, dtype=np.float32)
        hand_arr = np.asarray(hand_points, dtype=np.float32)
        hand_n_arr = np.asarray(hand_normals, dtype=np.float32)
        hand_root_pose_arr = np.asarray(hand_root_poses, dtype=np.float32)
        mano_global_orient_arr = np.asarray(mano_global_orients, dtype=np.float32)
        mano_transl_arr = np.asarray(mano_transls, dtype=np.float32)
        mano_pose_arr = np.asarray(mano_poses, dtype=np.float32)
        mano_betas_arr = np.asarray(mano_betas, dtype=np.float32)
        distances = np.empty((len(paths), 1538), dtype=np.float32)
        for i in range(len(paths)):
            distances[i] = cKDTree(obj_arr[i]).query(hand_arr[i], workers=1)[0]
        seq_id = f"{obj_id}/{timestamp}/view{view}"
        out = out_root / f"{group_idx:05d}_{obj_id}_view{view}.npz"
        np.savez_compressed(
            out,
            schema_name=np.asarray("train_corr_static_v2"),
            schema_version=np.asarray("2.0.0"),
            dataset_id=np.asarray("oakink"), dataset_name=np.asarray("OakInk"),
            seq_id=np.asarray(seq_id), side=np.asarray("right"),
            raw_frame_id=np.asarray(raw_ids, dtype=np.int32),
            obj_points=obj_arr, obj_normals=obj_n_arr,
            hand_points=hand_arr, hand_normals=hand_n_arr,
            hand_to_obj_min_dist=distances,
            coordinate_frame=np.asarray("hand_root"),
            hand_root_pose=hand_root_pose_arr,
            mano_global_orient=mano_global_orient_arr,
            mano_transl=mano_transl_arr,
            mano_pose=mano_pose_arr,
            mano_betas=mano_betas_arr,
            mano_v_template=mano_v_template,
            mano_use_pca=np.asarray(False),
            mano_num_pca_comps=np.asarray(45, dtype=np.int64),
            mano_flat_hand_mean=np.asarray(True),
            mano_pose_repr=np.asarray("axis_angle"),
        )
        rows.append({
            "seq_id": seq_id, "object_id": obj_id, "view": view,
            "frames": len(paths), "file": str(out),
            "contact_fraction_2cm": float((distances < 0.02).mean()),
            "distance_mean_m": float(distances.mean()),
            "distance_p50_m": float(np.quantile(distances, 0.50)),
            "distance_p95_m": float(np.quantile(distances, 0.95)),
        })
        print(f"[{group_idx + 1}/{len(group_keys)}] {seq_id} frames={len(paths)}", flush=True)
    summary = {
        "source": str(root), "output_root": str(out_root),
        "num_groups": len(rows), "num_frames": int(sum(x["frames"] for x in rows)),
        "schema_version": "2.0.0", "coordinate_frame": "hand_root",
        "hand_root_orientation": "official_oakink_mano_global_orient",
        "hand_points_source": "OakInk hand_v 778 MANO vertices -> MANO 1538 face centers",
        "hand_normals_source": "MANO face cross-product normals",
        "contact_source": "KD-tree distance to 4096 sampled object vertices",
        "mano_source": "OakInk general_info hand_pose/hand_shape/hand_tsl",
        "mano_transl_semantics": "smplx translation; shaped wrist joint equals OakInk hand_tsl",
        "skipped_groups": skipped,
        "groups": rows,
    }
    (out_root / "oakink_pilot_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
