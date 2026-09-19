"""Build the parent-only MANO-H source cache for the V1.1.16 D control."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[5]
VIEW = ROOT / "data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913"
PARENT = ROOT / "data/processed_data/cm_object_v2_surface512_object_pose_20260830"
GRAB = ROOT / "data/raw_data/GRAB/grab"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rot6(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:3, :2].T.reshape(-1), dtype=np.float32)


def build(entry: dict, out: Path) -> dict:
    parent = PARENT / entry["id"]
    raw_ids = np.asarray(np.load(parent / "shared/raw_frame_id.npy"), dtype=np.int64)
    object_pose = np.asarray(np.load(parent / "shared/obj_pose_world.npy"), dtype=np.float32)
    root_pose = np.asarray(np.load(parent / "right/hand_root_pose_world.npy"), dtype=np.float32)
    raw_path = GRAB / (entry["id"] + ".npz")
    with np.load(raw_path, allow_pickle=True) as archive:
        params = archive["rhand"].item()["params"]
    h = np.empty((len(raw_ids), 43), dtype=np.float32)
    for i, frame in enumerate(raw_ids):
        object_inv = np.linalg.inv(object_pose[i])
        wrist_object = object_inv @ root_pose[i]
        h[i, :3] = wrist_object[:3, 3]
        h[i, 3:9] = rot6(wrist_object[:3, :3])
        h[i, 9:33] = np.asarray(params["hand_pose"][frame], dtype=np.float32)
        # GRAB raw rhand entries have no explicit betas; preprocessing uses
        # zero beta together with the subject-specific v_template.
        h[i, 33:] = 0.0
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "mano_h.npy", h)
    np.save(out / "raw_frame_id.npy", raw_ids)
    manifest = {
        "schema_name": "ref2dex_mano_h_parent_cache_v1", "work_version": "V1.1.16",
        "source_type": "mano_parent_pose_in_parent_object_pose", "coordinate_frame": "parent_object_pose_t",
        "field_definition": "H=wrist_object_t[3]+wrist_object_rot6d[6]+hand_pose[24]+betas[10]",
        "frame_count": int(len(raw_ids)), "raw_frame_id_sha256": sha(parent / "shared/raw_frame_id.npy"),
        "source_raw_file": str(raw_path), "source_raw_sha256": sha(raw_path),
        "betas_semantics": "zero_beta_with_subject_specific_v_template",
        "files": {name: str((out / name).resolve()) for name in ("mano_h.npy", "raw_frame_id.npy")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-sequences", type=int)
    args = parser.parse_args()
    out_root = ROOT / "data/processed_data/cm_decoder_v2" / args.run_id
    out_root.mkdir(parents=True, exist_ok=False)
    index = json.loads((VIEW / "index.json").read_text())
    selected = index["sequences"]["train"] + index["sequences"]["val"]
    if args.max_sequences:
        selected = selected[:args.max_sequences]
    rows = []
    for entry in selected:
        out = out_root / entry["split"] / entry["id"].replace("/", "_")
        rows.append(build(entry, out))
    run = {"manifest_schema": "ref2dex.run.v1", "task": "CmDecoderv2", "work_version": "V1.1.16",
           "operation_category": ["data", "diagnostic"], "run_id": args.run_id, "run_status": "COMPLETED",
           "created_at": datetime.now(timezone.utc).isoformat(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "requested": len(selected), "built": len(rows), "source_index": str(VIEW / "index.json"),
           "output_dir": str(out_root)}
    (out_root / "run_manifest.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"run_id": args.run_id, "requested": len(selected), "built": len(rows), "output": str(out_root)}), flush=True)


if __name__ == "__main__":
    main()
