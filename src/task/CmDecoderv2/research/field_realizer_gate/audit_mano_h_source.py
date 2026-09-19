"""Read-only availability and raw-frame alignment audit for the D control."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    index_path = Path("data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/index.json")
    parent_root = Path("data/processed_data/cm_object_v2_surface512_object_pose_20260830")
    grab_root = Path("data/raw_data/GRAB")
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "task": "CmDecoderv2", "work_version": "V1.1.16",
        "operation_category": ["diagnostic"], "run_id": args.output.name, "run_status": "RUNNING",
        "created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "index_path": str(index_path), "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "availability and frame alignment only; not MANO geometry parity",
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    try:
        index = json.loads(index_path.read_text())
        for split in ("train", "val"):
            for entry in index["sequences"][split]:
                raw_path = grab_root / "grab" / (entry["id"] + ".npz")
                row = {"id": entry["id"], "split": split, "raw_path": str(raw_path)}
                try:
                    # Trusted GRAB files store named parameter dictionaries.
                    with np.load(raw_path, allow_pickle=True) as archive:
                        hand = archive["rhand"].item()
                    params = hand["params"]
                    raw_ids = np.load(parent_root / entry["id"] / "shared/raw_frame_id.npy")
                    target_ids = np.load(Path(entry["geometry_root"]) / "source_frame_id.npy")
                    assert np.array_equal(raw_ids, target_ids), "MANO/Inspire raw frame mismatch"
                    assert params["hand_pose"].shape[1] == 24
                    for key in ("global_orient", "hand_pose", "transl"):
                        assert np.isfinite(params[key][raw_ids]).all(), key
                    template = grab_root / hand["vtemp"]
                    assert template.is_file(), f"Missing template {template}"
                    row.update(available=True, frames=len(raw_ids), pose_dimension=24,
                               template_path=str(template),
                               betas_storage="explicit" if isinstance(hand.get("betas"), np.ndarray)
                               else "existing_GRABSeqData_zero_beta_with_subject_template",
                               raw_frame_sha256=hashlib.sha256(raw_ids.tobytes()).hexdigest())
                except Exception as error:
                    row.update(available=False, error=f"{type(error).__name__}: {error}")
                rows.append(row)
        result = {"sequences": rows, "available": sum(x["available"] for x in rows),
                  "total": len(rows), "geometry_parity_tested": False,
                  "D_training_started": False,
                  "remaining": "wrist uses MANO joint0, not raw transl; template/shape parity and matched geometry input remain",
                  "conclusion": "SUPPORTED" if all(x["available"] for x in rows) else "INCONCLUSIVE"}
        (args.output / "availability.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest["run_status"] = "COMPLETED"
        print(json.dumps({k: v for k, v in result.items() if k != "sequences"}), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
