"""V1.1.16 pre-training contract audit.

This audit intentionally checks manifests and source provenance only. It does
not load a checkpoint, generate a cache, or silently reinterpret the legacy
transported MANO source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .contact_logging import contact_api_capability


ROOT = Path(__file__).resolve().parents[5]
VIEW = ROOT / "data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913"
CONTACT = ROOT / "outputs/cmdecoderv2/rl_online_source_airplane_contact50_v15_20260913/manifest.json"
PAIRWISE_IMPLEMENTATION = ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output = ROOT / "src/task/CmDecoderv2/research/field_realizer_gate/output" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema_name": "ref2dex.field_realizer_gate_audit.v1",
        "task": "CmDecoderv2",
        "modification_version": "V1.1.16",
        "run_id": args.run_id,
        "run_status": "STARTED",
        "conclusion": "INCONCLUSIVE",
        "base_commit": base_commit,
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "started_at": started,
        "output_directory": str(output.relative_to(ROOT)),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    try:
        view = json.loads((VIEW / "manifest.json").read_text())
        index = json.loads((VIEW / "index.json").read_text())
        contact = json.loads(CONTACT.read_text())
        assert view["modification_version"] == "V1.1.14"
        assert view["split_contract"] == {"train": "mano_source_actual_inspire", "val": "mano_source_actual_inspire", "test": "not_built"}
        assert view["counts"] == {"train": 254, "val": 30, "test": 0}
        assert view["cm_source_contract"]["coordinate_frame"] == "object_pose_t"
        assert view["cm_source_contract"]["hand_points"] == 10135
        assert any(x["id"] == "s2/mug_drink_2" for x in view["excluded_sequences"])
        assert len(index["sequences"]["train"]) == 254 and len(index["sequences"]["val"]) == 30
        assert contact["initial_state_frame"] == 50 and contact["window_start_frame"] == 50
        assert contact["window_count"] == 378 and contact["seed"] == 42 and contact["fps"] == 30
        assert contact["checkpoint_sha256"] == "0814bdabcbdf484d90c6855a50e3bfebc1053b4d085ebde152b15f65aa8c4494"
        assert contact["oicm_checkpoint_sha256"] == "3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283"
        source_manifests = []
        for split in ("train", "val"):
            for entry in index["sequences"][split]:
                source_manifest = VIEW / entry["cm_source"] / "manifest.json" if "cm_source" in entry else Path(entry["paired_source_manifest"])
                if not source_manifest.is_absolute():
                    source_manifest = (VIEW / source_manifest).resolve()
                source = json.loads(source_manifest.read_text())
                assert source["source_type"] == "mano_parent_surface_at_actual_object_pose"
                assert source["coordinate_frame"] == "object_pose_t"
                source_manifests.append({"id": entry["id"], "split": split, "path": str(source_manifest), "sha256": sha256(source_manifest), "source_type": source["source_type"]})
        task_text = PAIRWISE_IMPLEMENTATION.read_text()
        api = contact_api_capability()
        pairwise_available = bool(api.get("available"))
        result = {
            "frozen_contract": {"field": "F7=[r,d,v]", "anchors": 128, "tau_m": 0.015, "window_size": 4, "fps": 30},
            "paired_split": {"train": 254, "val": 30, "test": 0},
            "contact50": {"initial_state_frame": 50, "window_start_frame": 50, "window_count": 378, "envs": 64, "seed": 42},
            "legacy_source_is_transported": True,
            "source_manifests": source_manifests,
            "pairwise_contact_api_static": pairwise_available,
            "pairwise_contact_api_status": "API_AVAILABLE_ADAPTER_PENDING_TASK_INTEGRATION" if pairwise_available else "UNAVAILABLE",
            "pairwise_contact_api": api,
            "task_currently_uses_net_force_only": "acquire_net_contact_force_tensor" in task_text,
            "future_effect_input": False,
            "training_or_rollout_started": False,
            "conclusion": "SUPPORTED" if pairwise_available else "INCONCLUSIVE",
        }
        (output / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        manifest.update(run_status="COMPLETED", conclusion=result["conclusion"], finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    except Exception as error:
        manifest.update(run_status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
