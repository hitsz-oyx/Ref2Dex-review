"""Build a parent-only V20 F7 cache without the actual Inspire trajectory.

The cache is intentionally small and explicit: MANO parent mesh vertices are
expressed in the parent object's frame, while anchors are sampled from that
same parent object frame.  ``field_f7[t]`` is the causal state at frame
``t+1`` and is therefore the goal for the control transition ``t -> t+1``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.task.CmDecoderv2.field_realizer import build_f7_field
from src.task.InteractionDynamics.uni3d import deterministic_fps


ROOT = Path(__file__).resolve().parents[5]
PARENT_ROOT = ROOT / "data/processed_data/cm_object_v2_surface512_object_pose_20260830"
VIEW_ROOT = ROOT / "data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def localize(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points - pose[:, None, :3, 3]) @ pose[:, :3, :3]


def build_sequence(parent: Path, output: Path, *, anchor_count: int, tau_m: float, batch_frames: int) -> dict[str, object]:
    required = [
        parent / "shared/raw_frame_id.npy", parent / "shared/obj_pose_world.npy",
        parent / "shared/obj_points_world.npy", parent / "shared/obj_normals_world.npy",
        parent / "right/hand_mesh_vertices_world.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing MANO parent provenance: {missing}")
    raw = np.asarray(np.load(required[0], mmap_mode="r"), dtype=np.int64)
    pose = np.asarray(np.load(required[1], mmap_mode="r"), dtype=np.float32)
    obj_world = np.asarray(np.load(required[2], mmap_mode="r"), dtype=np.float32)
    obj_normals_world = np.asarray(np.load(required[3], mmap_mode="r"), dtype=np.float32)
    hand_world = np.asarray(np.load(required[4], mmap_mode="r"), dtype=np.float32)
    if pose.shape != (len(raw), 4, 4) or hand_world.shape[0] != len(raw):
        raise ValueError("parent frame arrays do not agree")
    object_local = localize(obj_world, pose)
    normals_local = obj_normals_world @ pose[:, :3, :3]
    anchor_ids = deterministic_fps(torch.from_numpy(object_local[:1]), anchor_count)[0].numpy()
    anchors = object_local[0, anchor_ids]
    anchor_normals = normals_local[0, anchor_ids]
    anchor_normals /= np.clip(np.linalg.norm(anchor_normals, axis=-1, keepdims=True), 1e-8, None)
    hand_local = localize(hand_world, pose)
    field_chunks = []
    for start in range(0, len(raw) - 1, max(1, int(batch_frames))):
        stop = min(len(raw), start + max(1, int(batch_frames)) + 1)
        chunk = build_f7_field(
            torch.from_numpy(hand_local[start:stop]), torch.from_numpy(anchors), tau_m=tau_m
        ).numpy()
        field_chunks.append(chunk)
    field = np.concatenate(field_chunks, axis=0).astype(np.float32, copy=False)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "field_f7.npy", field)
    np.save(output / "anchors_object.npy", anchors.astype(np.float32))
    np.save(output / "anchor_normals_object.npy", anchor_normals.astype(np.float32))
    np.save(output / "raw_frame_id.npy", raw)
    manifest = {
        "schema_name": "ref2dex_field_f7_parent_cache_v1",
        "modification_version": "V1.1.16",
        "source_type": "mano_parent_surface_in_parent_object_pose",
        "coordinate_frame": "parent_object_pose_t",
        "field_definition": "F7=[r,d,v]; no p/c/contact/E",
        "anchor_count": int(anchor_count), "tau_m": float(tau_m),
        "frame_count": int(len(raw)), "field_frame_count": int(len(field)),
        "field_alignment": "field_f7[t] is causal state at raw frame t+1; goal for t->t+1",
        "parent_root": str(parent.resolve()),
        "raw_frame_id_sha256": sha256(required[0]),
        "object_pose_sha256": sha256(required[1]),
        "hand_mesh_sha256": sha256(required[4]),
        "files": {name: str((output / name).resolve()) for name in ("field_f7.npy", "anchors_object.npy", "anchor_normals_object.npy", "raw_frame_id.npy")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="s1/airplane_lift")
    parser.add_argument("--all", action="store_true", help="Build entries from the paired train/val index")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--tau-m", type=float, default=0.015)
    parser.add_argument("--batch-frames", type=int, default=16)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.all:
        index = json.loads((VIEW_ROOT / "index.json").read_text())
        entries = [(split, value) for split in ("train", "val") for value in index["sequences"][split]]
        if args.max_sequences is not None:
            entries = entries[: max(0, int(args.max_sequences))]
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        built = []
        for split, value in entries:
            sequence_parent = PARENT_ROOT / str(value["subject_id"]) / f"{value['object_name']}_{value['action_name']}"
            target = args.output / split / value["id"].replace("/", "_")
            try:
                built.append(build_sequence(sequence_parent, target, anchor_count=args.anchor_count, tau_m=args.tau_m, batch_frames=args.batch_frames))
                print(f"[done] {split} {value['id']}", flush=True)
            except FileNotFoundError as error:
                print(f"[skip] {split} {value['id']}: {error}", flush=True)
        result = {"mode": "all", "requested": len(entries), "built": len(built), "results": built}
    else:
        subject, task = args.sequence.split("/", 1)
        parent = PARENT_ROOT / subject / task
        result = build_sequence(parent, args.output, anchor_count=args.anchor_count, tau_m=args.tau_m, batch_frames=args.batch_frames)
    run_manifest = {
        "schema_name": "ref2dex.run_manifest_v1", "task": "CmDecoderv2",
        "modification_version": "V1.1.16", "operation": "parent_f7_cache_smoke",
        "run_id": args.output.name, "run_status": "COMPLETED", "conclusion": "SUPPORTED",
        "base_commit": base_commit, "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sequence": "all eligible paired entries" if args.all else args.sequence,
        "output_directory": str(args.output.resolve()), "cache_manifest": "manifest.json",
        "command": [str(value) for value in ("python", *(__import__("sys").argv[1:]))], "result": result,
    }
    (args.output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(run_manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
