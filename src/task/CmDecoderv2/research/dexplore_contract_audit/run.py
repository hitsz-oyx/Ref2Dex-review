"""Audit exact paired files before changing decoder supervision or launching training."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ...kinematics import InspireKinematics
from ...pointflow import _v13_surface_samples
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (
    InspireUrdfModel, _fk_surface,
)
from .contracts import (
    existing_splits, joint_least_squares, mimic_residual, reconstruct_native,
    sampled_frames, statistics, validate_pair,
)

ROOT = Path(__file__).resolve().parents[5]
VERSION = "V1.1.13"
TIP_LINKS = ("index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "sha256": digest.hexdigest()}


def tips(model: InspireUrdfModel, q: np.ndarray) -> np.ndarray:
    return np.stack([np.stack([links[name][:3, 3] for name in TIP_LINKS])
                     for links in (model.link_transforms(model.qpos_to_urdf_order(row)) for row in q)])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--frames-per-sequence", type=int, default=8)
    parser.add_argument("--surface-points", type=int, default=10135)
    parser.add_argument("--geometric-root", type=Path, default=ROOT / "data/processed_data/inspire_geometric_dexplore")
    parser.add_argument("--actual-root", type=Path, default=ROOT / "data/processed_data/inspire_rl_object_dexplore")
    parser.add_argument("--canonical-root", type=Path, default=ROOT.parent / "InterAct/data/grab/sequences_canonical")
    parser.add_argument("--source-index", type=Path, default=ROOT / "data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json")
    parser.add_argument("--urdf", type=Path, default=ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")
    return parser.parse_args()


def main():
    args = parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        raise ValueError("run_id must be a single directory name")
    if args.surface_points <= 0 or args.frames_per_sequence <= 0:
        raise ValueError("Sampling counts must be positive")
    output = Path(__file__).parent / "output" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(output / "config.json", config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    code_paths = [Path(__file__), Path(__file__).with_name("contracts.py"),
                  ROOT / "src/task/CmDecoderv2/kinematics.py", ROOT / "src/task/CmDecoderv2/pointflow.py",
                  ROOT / "src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py",
                  ROOT / "src/task/ObjectInteractionCm/research/hand_region_sampling/build_trajectory_preview.py",
                  ROOT / "src/task/ObjectInteractionCm/research/hand_region_sampling/run.py"]
    dexplore = ROOT.parent / "dexplore"
    external_paths = [dexplore / name for name in (
        "data_processing/convert_grab.py", "data_processing/robot_configs.py",
        "data_processing/adapt_interact_canonical.py", "dexplore/env/tasks/dexplore_inspire.py",
        "dexplore/learning/dexplore_players.py")]
    assets = [args.urdf, *sorted((args.urdf.parent / "meshes_right").glob("*.STL"))]
    inputs = [args.source_index, args.geometric_root / "manifest.json", args.actual_root / "manifest.json"]
    manifest = {
        "schema_name": "ref2dex.run.v1", "task": "CmDecoderv2", "run_id": args.run_id,
        "run_status": "RUNNING", "work_version": VERSION,
        "operation_category": ["diagnostic"], "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_commit": commit, "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "command": " ".join(__import__("sys").argv), "seed": 2024, "checkpoint": None,
        "config_snapshot": "config.json", "metadata_snapshot": "metadata.json", "output_dir": str(output),
        "input_files": [fingerprint(path) for path in inputs],
        "code_files": [fingerprint(path) for path in code_paths + external_paths],
        "assets": [fingerprint(path) for path in assets],
    }
    write_json(output / "run_manifest.json", manifest)
    try:
        run(args, output)
        manifest.update(run_status="COMPLETED", elapsed_seconds=time.monotonic() - start)
    except BaseException as error:
        manifest.update(run_status="FAILED", exit_reason=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.monotonic() - start)
        raise
    finally:
        write_json(output / "run_manifest.json", manifest)


def run(args, output):
    geometry_paths = {p.parent.name: p for p in args.geometric_root.glob("*/interaction_hand_inspire.pt")}
    actual_paths = {p.parent.name: p for p in args.actual_root.glob("*/interaction_hand_inspire.pt")}
    if not geometry_paths or geometry_paths.keys() != actual_paths.keys():
        raise ValueError("Geometric/actual sequence sets differ or are empty")
    if len(geometry_paths) != 660:
        raise ValueError(f"Expected approved 660-pair source set, got {len(geometry_paths)}")
    names = sorted(geometry_paths)
    if args.smoke:
        names = names[:2]
    splits = existing_splits(json.loads(args.source_index.read_text()))
    kinematics = InspireKinematics(args.urdf)
    helper = InspireUrdfModel(args.urdf)
    points, normals, visual_ids = _v13_surface_samples(helper, args.surface_points, 2024)
    aggregates = {source: {key: [] for key in (
        "mimic_abs_deg", "max_mimic_abs_deg", "reconstruction_joint_rmse_deg",
        "optimal_joint_rmse_deg", "surface_epe_mm", "tip_epe_mm")}
        for source in ("geometric", "actual")}
    samples = {key: [] for key in ("q", "reconstructed_q", "frames", "sequence", "source",
                                  "surface_epe_mm", "tip_epe_mm")}
    records, metrics = [], []
    log = output / "run.log"
    with log.open("w", encoding="utf-8") as handle:
        for ordinal, name in enumerate(names):
            geo = torch.load(geometry_paths[name], map_location="cpu", weights_only=True)
            actual = torch.load(actual_paths[name], map_location="cpu", weights_only=True)
            if not isinstance(geo, torch.Tensor) or not isinstance(actual, torch.Tensor):
                raise TypeError(f"{name}: native files must contain tensors")
            geo, actual = geo.numpy(), actual.numpy()
            validate_pair(geo, actual)
            human_path = args.canonical_root / name / "human.npz"
            object_path = args.canonical_root / name / "object.npz"
            with np.load(human_path, allow_pickle=False) as human:
                if human["poses"].shape != (len(geo), 114) or human["trans"].shape != (len(geo), 3):
                    raise ValueError(f"{name}: canonical MANO source frame count/shape mismatch")
                if not all(np.isfinite(human[key]).all() for key in ("poses", "trans", "vtemp")):
                    raise ValueError(f"{name}: nonfinite canonical human")
            with np.load(object_path, allow_pickle=False) as obj:
                if len(obj["trans"]) != len(geo) or len(obj["angles"]) != len(geo):
                    raise ValueError(f"{name}: canonical object frame count mismatch")
            record = {"id": name, "frame_count": len(geo), "existing_split": splits.get(name, "unassigned"),
                      "geometric": fingerprint(geometry_paths[name]), "actual": fingerprint(actual_paths[name]),
                      "canonical_human": fingerprint(human_path), "canonical_object": fingerprint(object_path)}
            records.append(record)
            frames = sampled_frames(len(geo), args.frames_per_sequence)
            for source_id, (source, tensor) in enumerate((("geometric", geo), ("actual", actual))):
                q = tensor[:, 373:391].astype(np.float64)
                reconstructed = reconstruct_native(q, kinematics)
                ls = joint_least_squares(q, kinematics)
                residual = np.rad2deg(np.abs(mimic_residual(q)))
                row = {"id": name, "source": source, "frames": len(q), "existing_split": record["existing_split"],
                       "mimic_abs_deg": statistics(residual),
                       "max_mimic_abs_deg": statistics(residual.max(axis=1)),
                       "frame_fraction_over_1deg": float((residual.max(axis=1) > 1).mean()),
                       "reconstruction_joint_rmse_deg": statistics(np.rad2deg(np.sqrt(np.square(q[:, 6:] - reconstructed[:, 6:]).mean(axis=1)))),
                       "optimal_joint_rmse_deg": statistics(np.rad2deg(np.sqrt(np.square(q[:, 6:] - ls[:, 6:]).mean(axis=1))))}
                raw_surface, _ = _fk_surface(helper, q[frames], points, normals, visual_ids)
                rec_surface, _ = _fk_surface(helper, reconstructed[frames], points, normals, visual_ids)
                surface_error = np.linalg.norm(raw_surface.astype(np.float64) - rec_surface, axis=-1).mean(axis=1) * 1000
                tip_error = np.linalg.norm(tips(helper, q[frames]) - tips(helper, reconstructed[frames]), axis=-1).mean(axis=1) * 1000
                row.update(surface_epe_mm=statistics(surface_error), tip_epe_mm=statistics(tip_error))
                values = {"mimic_abs_deg": residual.reshape(-1), "max_mimic_abs_deg": residual.max(axis=1),
                          "reconstruction_joint_rmse_deg": np.rad2deg(np.sqrt(np.square(q[:, 6:] - reconstructed[:, 6:]).mean(axis=1))),
                          "optimal_joint_rmse_deg": np.rad2deg(np.sqrt(np.square(q[:, 6:] - ls[:, 6:]).mean(axis=1))),
                          "surface_epe_mm": surface_error, "tip_epe_mm": tip_error}
                for key, value in values.items():
                    aggregates[source][key].append(value)
                samples["q"].append(q[frames])
                samples["reconstructed_q"].append(reconstructed[frames])
                samples["frames"].append(frames)
                samples["sequence"].append(np.full(len(frames), ordinal, dtype=np.int32))
                samples["source"].append(np.full(len(frames), source_id, dtype=np.int8))
                samples["surface_epe_mm"].append(surface_error)
                samples["tip_epe_mm"].append(tip_error)
                metrics.append(row)
            message = f"[{ordinal + 1}/{len(names)}] {name} frames={len(geo)} geo/actual surface_mm={metrics[-2]['surface_epe_mm']['mean']:.4f}/{metrics[-1]['surface_epe_mm']['mean']:.4f}"
            handle.write(message + "\n")
            handle.flush()
            if ordinal % 25 == 0 or ordinal + 1 == len(names):
                print(message, flush=True)
    np.savez_compressed(output / "state_samples.npz", **{key: np.concatenate(value) for key, value in samples.items()})
    write_json(output / "paired_manifest.json", {"schema_name": "ref2dex_dexplore_pair_audit_v1", "diagnostic_only": True,
               "training_eligible": False, "work_version": VERSION, "sequences": records})
    summary = {"sequences": len(names), "frames": sum(row["frame_count"] for row in records),
               "existing_split_counts": dict(Counter(row["existing_split"] for row in records)),
               "pair_shapes_and_unchanged_fields": "passed", "canonical_identity_and_frame_count": "passed",
               "canonical_mesh_coordinates": "not_checked", "metric_semantics": "FK reconstruction error at retained independent joints, not an optimal FK lower bound", "sources": {}}
    for source, fields in aggregates.items():
        combined = {key: np.concatenate(values) for key, values in fields.items()}
        summary["sources"][source] = {
            "micro": {key: statistics(value) for key, value in combined.items()},
            "macro": {key: statistics(np.array([np.mean(v) for v in values])) for key, values in fields.items()},
            "frame_fraction_over_1deg": float((combined["max_mimic_abs_deg"] > 1).mean()),
            "sequences_with_over_1deg": sum(float(np.max(v)) > 1 for v in fields["max_mimic_abs_deg"]),
        }
    write_json(output / "summary.json", summary)
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "metadata.json", {
        "schema_name": "ref2dex_dexplore_contract_audit_v1", "native_q": ["T", 18],
        "finger_state": 6, "mimic_mode": "existing_decoder_and_dexplore_pd_target",
        "coordinate_frame": "dexplore_native_world", "effective_fps": 30,
        "surface_correspondence": "v1_3_cache", "surface_points": args.surface_points, "surface_seed": 2024,
        "frame_sampling": "unique_integer_linspace_including_endpoints", "frames_per_sequence": args.frames_per_sequence,
        "sources": ["geometric", "actual"], "reference_frames_used_as_actions": False,
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
