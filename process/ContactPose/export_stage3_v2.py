from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# smplx/chumpy still imports legacy numpy aliases.
for _legacy_name, _legacy_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)

import torch
import trimesh
from smplx import MANO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTACTPOSE_ROOT = Path("/mnt/ugreen_nas/storage/Ref2Dex_storage/ContactPose")
DEFAULT_MANO_PATH = Path("/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
DEFAULT_OUTPUT_ROOT = ROOT / "processed_data" / "generated" / "stage3" / "contactpose_use_stage3_v2"
SCHEMA_NAME = "train_corr_static_v2"
SCHEMA_VERSION = "2.0.0"
HAND_SIDES = ("left", "right")
NUM_HAND_POINTS = 1538


@dataclass(frozen=True)
class SequenceRef:
    subject_id: str
    intent: str
    object_name: str
    path: Path

    @property
    def seq_name(self) -> str:
        return f"{self.subject_id}_{self.intent}_{self.object_name}"

    @property
    def seq_id(self) -> str:
        return f"contactpose:{self.subject_id}_{self.intent}/{self.object_name}"


@dataclass(frozen=True)
class HandSequence:
    ref: SequenceRef
    side: str
    num_frames: int
    hand_points: np.ndarray
    hand_normals: np.ndarray
    obj_points: np.ndarray
    obj_normals: np.ndarray


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    return device


def _seed_from_text(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & 0x7FFFFFFF


def _quat_wxyz_to_matrix(quat_wxyz: list[float] | np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {quat.shape}")
    w, x, y, z = quat
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = quat / norm
    matrix = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return matrix


def _pose_dict_to_matrix(payload: dict[str, Any]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = _quat_wxyz_to_matrix(payload["rotation"])
    matrix[:3, 3] = np.asarray(payload["translation"], dtype=np.float32)
    return matrix


def _invert_pose(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -(rotation.T @ translation)
    return inv


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return (points @ rotation.T + translation).astype(np.float32)


def _transform_normals(normals: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    out = (normals @ rotation.T).astype(np.float32)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return (out / np.clip(norm, 1e-8, None)).astype(np.float32)


def _compute_face_centers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces].mean(axis=1).astype(np.float32)


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / np.clip(norm, 1e-8, None)).astype(np.float32)


def _sample_mesh_surface(mesh: trimesh.Trimesh, num_points: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    except TypeError:
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        finally:
            np.random.set_state(state)
    face_normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    norm = np.linalg.norm(face_normals, axis=-1, keepdims=True)
    face_normals = face_normals / np.clip(norm, 1e-8, None)
    return np.asarray(points, dtype=np.float32), face_normals.astype(np.float32)


def _compute_contact_statistics(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    *,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames, num_obj, _ = obj_points.shape
    num_hand = hand_points.shape[1]
    candidate_mask = np.empty((num_frames, num_obj), dtype=bool)
    hand_to_obj_min_dist = np.empty((num_frames, num_hand), dtype=np.float32)
    batch_size = max(1, int(frame_batch_size))
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        obj = torch.from_numpy(obj_points[start:end]).to(device=device, dtype=torch.float32)
        hand = torch.from_numpy(hand_points[start:end]).to(device=device, dtype=torch.float32)
        dist = torch.cdist(obj, hand)
        candidate_mask[start:end] = (dist.amin(dim=-1) <= float(candidate_threshold)).cpu().numpy()
        hand_to_obj_min_dist[start:end] = dist.amin(dim=1).cpu().numpy().astype(np.float32)
        del obj, hand, dist
    return candidate_mask, hand_to_obj_min_dist


class ManoLayerCache:
    def __init__(self, mano_path: Path) -> None:
        self.mano_path = Path(mano_path)
        self._layers: dict[tuple[str, int], MANO] = {}

    def get(self, side: str, num_pca_comps: int) -> MANO:
        key = (side, int(num_pca_comps))
        layer = self._layers.get(key)
        if layer is None:
            layer = MANO(
                model_path=str(self.mano_path),
                is_rhand=(side == "right"),
                use_pca=True,
                num_pca_comps=int(num_pca_comps),
                flat_hand_mean=False,
                batch_size=1,
            )
            self._layers[key] = layer
        return layer


def _load_hand_mesh_in_hand_frame(
    mano_cache: ManoLayerCache,
    *,
    side: str,
    mano_fit: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(mano_fit["pose"], dtype=np.float32)
    if pose.ndim != 1 or pose.shape[0] < 4:
        raise ValueError(f"Unexpected MANO pose shape for {side}: {pose.shape}")
    betas = np.asarray(mano_fit["betas"], dtype=np.float32)
    num_pca_comps = int(pose.shape[0] - 3)
    mano_layer = mano_cache.get(side, num_pca_comps)
    global_orient = torch.from_numpy(pose[:3]).view(1, 3)
    hand_pose = torch.from_numpy(pose[3:]).view(1, num_pca_comps)
    betas_t = torch.from_numpy(betas).view(1, -1)
    with torch.no_grad():
        out = mano_layer(global_orient=global_orient, hand_pose=hand_pose, betas=betas_t)
    vertices = out.vertices[0].detach().cpu().numpy().astype(np.float32)
    hTm = _invert_pose(_pose_dict_to_matrix(mano_fit["mTc"]))
    vertices_hand = _transform_points(vertices, hTm)
    faces = np.asarray(mano_layer.faces, dtype=np.int64)
    hand_points = _compute_face_centers(vertices_hand, faces)
    hand_normals = _compute_face_normals(vertices_hand, faces)
    if hand_points.shape[0] != NUM_HAND_POINTS:
        raise ValueError(
            f"Expected {NUM_HAND_POINTS} MANO face centers, got {hand_points.shape[0]} for {side}"
        )
    return hand_points, hand_normals


def _enumerate_sequences(contactpose_root: Path, *, intent: str) -> list[SequenceRef]:
    data_root = contactpose_root / "data" / "contactpose_data"
    pattern = f"full*_{intent}"
    refs: list[SequenceRef] = []
    for subject_dir in sorted(data_root.glob(pattern)):
        if not subject_dir.is_dir():
            continue
        subject_id = subject_dir.name.split("_", 1)[0]
        for object_dir in sorted(subject_dir.iterdir()):
            if not object_dir.is_dir():
                continue
            if object_dir.name in {"hands", "__MACOSX"}:
                continue
            if not (object_dir / "annotations.json").exists():
                continue
            refs.append(
                SequenceRef(
                    subject_id=subject_id,
                    intent=intent,
                    object_name=object_dir.name,
                    path=object_dir,
                )
            )
    return refs


def _load_object_surface_samples(ref: SequenceRef, *, num_obj_pool: int) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = ref.path / f"{ref.object_name}.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing object mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(mesh)!r}")
    seed = _seed_from_text(f"{ref.seq_id}:{num_obj_pool}")
    return _sample_mesh_surface(mesh, num_obj_pool, seed=seed)


def _build_hand_sequence(
    ref: SequenceRef,
    *,
    hand_idx: int,
    side: str,
    mano_cache: ManoLayerCache,
    num_obj_pool: int,
) -> HandSequence:
    annotations = json.loads((ref.path / "annotations.json").read_text(encoding="utf-8"))
    mano_fits = json.loads((ref.path / "mano_fits_15.json").read_text(encoding="utf-8"))
    hands_meta = annotations["hands"]
    if hand_idx >= len(hands_meta) or hand_idx >= len(mano_fits):
        raise IndexError(f"{ref.seq_id}: missing hand index {hand_idx}")
    hand_meta = hands_meta[hand_idx]
    mano_fit = mano_fits[hand_idx]
    if not bool(hand_meta.get("valid")) or not bool(mano_fit.get("valid")):
        raise ValueError(f"{ref.seq_id} {side}: hand is not valid")

    hand_points_single, hand_normals_single = _load_hand_mesh_in_hand_frame(
        mano_cache,
        side=side,
        mano_fit=mano_fit,
    )
    obj_points_canonical, obj_normals_canonical = _load_object_surface_samples(
        ref,
        num_obj_pool=num_obj_pool,
    )
    frames = annotations["frames"]
    moving = bool(hand_meta.get("moving", False))
    if moving and any("hTo" not in frame for frame in frames):
        raise ValueError(f"{ref.seq_id} {side}: moving hand but frame['hTo'] is missing")

    num_frames = len(frames)
    hand_points = np.broadcast_to(
        hand_points_single[None, :, :], (num_frames, hand_points_single.shape[0], 3)
    ).copy()
    hand_normals = np.broadcast_to(
        hand_normals_single[None, :, :], (num_frames, hand_normals_single.shape[0], 3)
    ).copy()
    obj_points = np.empty((num_frames, num_obj_pool, 3), dtype=np.float32)
    obj_normals = np.empty((num_frames, num_obj_pool, 3), dtype=np.float32)
    identity = np.eye(4, dtype=np.float32)
    for frame_idx, frame in enumerate(frames):
        hTo = identity
        if moving:
            hTo_payload = frame["hTo"]
            if not isinstance(hTo_payload, list) or hand_idx >= len(hTo_payload):
                raise ValueError(f"{ref.seq_id} {side}: invalid hTo payload at frame {frame_idx}")
            hTo = _pose_dict_to_matrix(hTo_payload[hand_idx])
        obj_points[frame_idx] = _transform_points(obj_points_canonical, hTo)
        obj_normals[frame_idx] = _transform_normals(obj_normals_canonical, hTo)

    return HandSequence(
        ref=ref,
        side=side,
        num_frames=num_frames,
        hand_points=hand_points,
        hand_normals=hand_normals,
        obj_points=obj_points,
        obj_normals=obj_normals,
    )


def _write_stage3_npz(
    sequence: HandSequence,
    *,
    output_root: Path,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> Path:
    candidate_mask, hand_to_obj_min_dist = _compute_contact_statistics(
        sequence.obj_points,
        sequence.hand_points,
        candidate_threshold=candidate_threshold,
        frame_batch_size=frame_batch_size,
        device=device,
    )
    subject_dir = output_root / sequence.ref.subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_path = subject_dir / f"{sequence.ref.seq_name}_{sequence.side}.npz"
    np.savez_compressed(
        output_path,
        schema_name=np.asarray(SCHEMA_NAME),
        schema_version=np.asarray(SCHEMA_VERSION),
        dataset_name=np.asarray("ContactPose"),
        subject_id=np.asarray(sequence.ref.subject_id),
        seq_name=np.asarray(sequence.ref.seq_name),
        seq_id=np.asarray(sequence.ref.seq_id),
        object_name=np.asarray(sequence.ref.object_name),
        side=np.asarray(sequence.side),
        raw_frame_id=np.arange(sequence.num_frames, dtype=np.int32),
        obj_points=sequence.obj_points.astype(np.float32),
        obj_normals=sequence.obj_normals.astype(np.float32),
        hand_points=sequence.hand_points.astype(np.float32),
        hand_normals=sequence.hand_normals.astype(np.float32),
        hand_to_obj_min_dist=hand_to_obj_min_dist.astype(np.float32),
        obj_candidate_mask_5cm=candidate_mask.astype(bool),
        coordinate_frame=np.asarray("hand_root"),
    )
    return output_path


def _write_meta(
    output_root: Path,
    *,
    args: argparse.Namespace,
    written: list[str],
    skipped: list[dict[str, str]],
) -> None:
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "ContactPose",
        "coordinate_frame": "hand_root",
        "num_obj_pool": int(args.num_obj_pool),
        "num_hand_points": NUM_HAND_POINTS,
        "candidate_threshold": float(args.candidate_threshold),
        "intent": str(args.intent),
        "single_hand_only": bool(args.single_hand_only),
        "contactpose_root": str(Path(args.contactpose_root).resolve()),
        "mano_path": str(Path(args.mano_path).resolve()),
        "written_files": written,
        "num_written": len(written),
        "num_skipped": len(skipped),
        "skipped": skipped,
    }
    (output_root / "meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ContactPose sequences to Ref2Dex Stage 3 v2 format.")
    parser.add_argument("--contactpose-root", default=str(DEFAULT_CONTACTPOSE_ROOT))
    parser.add_argument("--mano-path", default=str(DEFAULT_MANO_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--intent", choices=("use", "handoff"), default="use")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-obj-pool", type=int, default=4096)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--single-hand-only", action="store_true", default=True)
    parser.add_argument("--allow-bimanual", dest="single_hand_only", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contactpose_root = Path(args.contactpose_root).resolve()
    mano_path = Path(args.mano_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    refs = _enumerate_sequences(contactpose_root, intent=args.intent)
    if args.limit is not None:
        refs = refs[: int(args.limit)]
    if not refs:
        raise SystemExit("No ContactPose sequences matched the current filters.")

    mano_cache = ManoLayerCache(mano_path)
    written: list[str] = []
    skipped: list[dict[str, str]] = []

    for ref in refs:
        try:
            annotations = json.loads((ref.path / "annotations.json").read_text(encoding="utf-8"))
            mano_fits = json.loads((ref.path / "mano_fits_15.json").read_text(encoding="utf-8"))
            valid_hands = []
            for hand_idx, side in enumerate(HAND_SIDES):
                if hand_idx < len(annotations["hands"]) and hand_idx < len(mano_fits):
                    if bool(annotations["hands"][hand_idx].get("valid")) and bool(mano_fits[hand_idx].get("valid")):
                        valid_hands.append((hand_idx, side))
            if args.single_hand_only and len(valid_hands) != 1:
                skipped.append({"seq_id": ref.seq_id, "reason": f"expected exactly 1 valid hand, got {len(valid_hands)}"})
                continue
            for hand_idx, side in valid_hands:
                out_path = output_root / ref.subject_id / f"{ref.seq_name}_{side}.npz"
                if out_path.exists() and not args.overwrite:
                    written.append(str(out_path))
                    continue
                sequence = _build_hand_sequence(
                    ref,
                    hand_idx=hand_idx,
                    side=side,
                    mano_cache=mano_cache,
                    num_obj_pool=int(args.num_obj_pool),
                )
                result = _write_stage3_npz(
                    sequence,
                    output_root=output_root,
                    candidate_threshold=float(args.candidate_threshold),
                    frame_batch_size=int(args.frame_batch_size),
                    device=device,
                )
                written.append(str(result))
                print(f"[ok] {ref.seq_id} {side} -> {result}")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"seq_id": ref.seq_id, "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[skip] {ref.seq_id}: {type(exc).__name__}: {exc}")

    _write_meta(output_root, args=args, written=written, skipped=skipped)
    print(json.dumps({"written": len(written), "skipped": len(skipped), "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
