"""Fit DExplore geometric Inspire trajectories to the six-dimensional control manifold.

The wrist and all human/object columns stay unchanged.  Only the 12 finger
columns are replaced by ``expand_finger_q(u6)`` after optimizing ``u6`` against
the original FK fingertips and a deterministic surface sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ...kinematics import InspireKinematics
from ...pointflow import DifferentiableInspireSurface, _v13_surface_samples
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel, _fk_surface
from src.task.CmDecoderv2.research.dexplore_contract_audit.contracts import (
    MIMIC_TARGET, MIMIC_SOURCE, MIMIC_SCALE, reconstruct_native,
)

ROOT = Path(__file__).resolve().parents[5]
TIP_LINKS = ("index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor at {path}")
    return value.numpy().astype(np.float32, copy=False)


def tip_targets(model: InspireUrdfModel, q: np.ndarray) -> np.ndarray:
    return np.stack([
        np.stack([model.link_transforms(model.qpos_to_urdf_order(row))[name][:3, 3] for name in TIP_LINKS])
        for row in q
    ]).astype(np.float32)


def fit_sequence(q: np.ndarray, helper: InspireUrdfModel, surface: DifferentiableInspireSurface,
                  points: np.ndarray, normals: np.ndarray, visual_ids: np.ndarray,
                  *, device: torch.device, iterations: int, lr: float,
                  surface_weight: float, tip_weight: float, smooth_weight: float) -> tuple[np.ndarray, dict]:
    kin = InspireKinematics(helper.urdf_path)
    q = q.astype(np.float64, copy=False)
    frames = len(q)
    wrist = np.stack([helper.link_transforms(helper.qpos_to_urdf_order(row))['hand_base_link'] for row in q]).astype(np.float32)
    target_tips = tip_targets(helper, q)
    target_surface, _ = _fk_surface(helper, q, points, normals, visual_ids)
    u0 = kin.clamp_finger_q(q[:, [6, 8, 10, 12, 14, 15]])
    u = torch.tensor(u0, device=device, dtype=torch.float32, requires_grad=True)
    wrist_t = torch.from_numpy(wrist).to(device)
    target_tips_t = torch.from_numpy(target_tips).to(device)
    target_surface_t = torch.from_numpy(target_surface).to(device)
    optimizer = torch.optim.Adam([u], lr=float(lr))
    lower = torch.tensor(kin.finger_lower, device=device, dtype=torch.float32)
    upper = torch.tensor(kin.finger_upper, device=device, dtype=torch.float32)
    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        clipped = torch.maximum(torch.minimum(u, upper), lower)
        pred_tips = surface.link_points(clipped, wrist_t, TIP_LINKS)
        pred_surface = surface(clipped, wrist_t)
        tip_loss = (pred_tips - target_tips_t).square().mean()
        surface_loss = (pred_surface - target_surface_t).square().mean()
        if frames >= 3:
            smooth_loss = (clipped[2:] - 2 * clipped[1:-1] + clipped[:-2]).square().mean()
        else:
            smooth_loss = clipped.sum() * 0.0
        loss = float(tip_weight) * tip_loss + float(surface_weight) * surface_loss + float(smooth_weight) * smooth_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            u.clamp_(lower, upper)
    with torch.no_grad():
        fitted = torch.maximum(torch.minimum(u, upper), lower)
        fitted_tips = surface.link_points(fitted, wrist_t, TIP_LINKS).cpu().numpy()
        fitted_surface = surface(fitted, wrist_t).cpu().numpy()
    # Expand the optimized six values without touching the six virtual wrist DOFs.
    fitted_native = np.zeros((frames, 18), dtype=np.float64)
    fitted_native[:, :6] = q[:, :6]
    fitted_native[:, 6] = fitted[:, 0].cpu().numpy()
    fitted_native[:, 8] = fitted[:, 1].cpu().numpy()
    fitted_native[:, 10] = fitted[:, 2].cpu().numpy()
    fitted_native[:, 12] = fitted[:, 3].cpu().numpy()
    fitted_native[:, 14] = fitted[:, 4].cpu().numpy()
    fitted_native[:, 15] = fitted[:, 5].cpu().numpy()
    fitted_native[:, 7] = fitted_native[:, 6] * 1.05
    fitted_native[:, 9] = fitted_native[:, 8] * 1.05
    fitted_native[:, 11] = fitted_native[:, 10] * 1.05
    fitted_native[:, 13] = fitted_native[:, 12] * 1.05
    fitted_native[:, 16] = fitted_native[:, 15] * .6
    fitted_native[:, 17] = fitted_native[:, 15] * .8
    baseline_q = reconstruct_native(q, kin)
    baseline_tip_error = np.linalg.norm(tip_targets(helper, baseline_q) - target_tips, axis=-1).mean(axis=1) * 1000
    fitted_tip_error = np.linalg.norm(fitted_tips - target_tips, axis=-1).mean(axis=1) * 1000
    fitted_surface_error = np.linalg.norm(fitted_surface - target_surface, axis=-1).mean(axis=1) * 1000
    return fitted_native, {
        "frames": frames, "iterations": int(iterations), "tip_error_before_mm": {"mean": float(baseline_tip_error.mean()), "p95": float(np.quantile(baseline_tip_error, .95))},
        "tip_error_after_mm": {"mean": float(fitted_tip_error.mean()), "p95": float(np.quantile(fitted_tip_error, .95))},
        "surface_error_after_mm": {"mean": float(fitted_surface_error.mean()), "p95": float(np.quantile(fitted_surface_error, .95))},
        "independent_q_rmse_deg": float(np.rad2deg(np.sqrt(np.square(q[:, 6:] - fitted_native[:, 6:]).mean(axis=1))).mean()),
        "mimic_residual_after_deg": float(np.rad2deg(np.abs(fitted_native[:, MIMIC_TARGET] - fitted_native[:, MIMIC_SOURCE] * MIMIC_SCALE)).mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "data/processed_data/inspire_geometric_dexplore")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--surface-points", type=int, default=256)
    parser.add_argument("--tip-weight", type=float, default=10.0)
    parser.add_argument("--surface-weight", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    output = args.output_root
    input_paths = sorted(args.input_root.glob("*/interaction_hand_inspire.pt"))
    if len(input_paths) != 660:
        raise ValueError(f"Expected 660 source sequences, got {len(input_paths)}")
    if args.smoke:
        input_paths = input_paths[:2]
    urdf = ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"
    helper = InspireUrdfModel(urdf)
    points, normals, visual_ids = _v13_surface_samples(helper, args.surface_points, 2024)
    surface = DifferentiableInspireSurface(urdf, sample_count=args.surface_points, surface_sampling="v1_3_cache").to(args.device)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1", "task": "CmDecoderv2", "run_id": args.run_id,
        "run_status": "RUNNING", "work_version": "V1.1.13", "operation_category": ["data", "operation"],
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "base_commit": commit,
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "command": " ".join(__import__("sys").argv), "output_dir": str(output.resolve()),
        "input_root": str(args.input_root.resolve()), "training_eligible": not args.smoke,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records = []
    started = time.time()
    try:
        for index, source_path in enumerate(input_paths):
            source = load_tensor(source_path)
            q = source[:, 373:391]
            if not np.isfinite(source).all() or source.shape[1] != 598:
                raise ValueError(f"Invalid source tensor {source_path}: {source.shape}")
            fitted_q, metrics = fit_sequence(q, helper, surface, points, normals, visual_ids,
                                             device=torch.device(args.device), iterations=args.iterations,
                                             lr=args.lr, surface_weight=args.surface_weight,
                                             tip_weight=args.tip_weight, smooth_weight=args.smooth_weight)
            result = torch.from_numpy(source.copy())
            result[:, 373:391] = torch.from_numpy(fitted_q.astype(np.float32))
            out_path = output / source_path.parent.name / source_path.name
            out_path.parent.mkdir(parents=True, exist_ok=False)
            torch.save(result, out_path)
            metrics.update({"id": source_path.parent.name, "source_sha256": sha256(source_path), "output": str(out_path.resolve())})
            records.append(metrics)
            if index % 10 == 0 or index + 1 == len(input_paths):
                print(f"[{index + 1}/{len(input_paths)}] {source_path.parent.name} tip_after={metrics['tip_error_after_mm']['mean']:.3f}mm", flush=True)
        manifest = {
            "schema_name": "ref2dex_dexplore_coupled_geometric_v1", "diagnostic_only": False,
            "training_eligible": not args.smoke, "run_id": args.run_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "work_version": "V1.1.13", "input_root": str(args.input_root.resolve()),
            "output_root": str(output.resolve()), "source_format": "dexplore_inspire_filtered_v1",
            "native_q_slice": [373, 391], "independent_finger_indices": [6, 8, 10, 12, 14, 15],
            "mimic_source": MIMIC_SOURCE.tolist(), "mimic_target": MIMIC_TARGET.tolist(), "mimic_scale": MIMIC_SCALE.tolist(),
            "wrist_policy": "fixed_from_input_native_q_0_6", "unchanged_columns": list(range(373)) + list(range(391, 598)),
            "optimization": {"objective": "tip_and_surface_FK_with_second_difference", "iterations": args.iterations, "lr": args.lr,
                             "surface_points": args.surface_points, "surface_seed": 2024, "tip_weight": args.tip_weight,
                             "surface_weight": args.surface_weight, "smooth_weight": args.smooth_weight},
            "num_sequences": len(records), "records": records,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        summary = {"num_sequences": len(records), "num_frames": sum(r["frames"] for r in records),
            "tip_error_before_mm": {"mean": float(np.mean([r["tip_error_before_mm"]["mean"] for r in records])),
                                    "p95": float(np.quantile([r["tip_error_before_mm"]["mean"] for r in records], .95))},
            "tip_error_after_mm": {"mean": float(np.mean([r["tip_error_after_mm"]["mean"] for r in records])),
                                   "p95": float(np.quantile([r["tip_error_after_mm"]["mean"] for r in records], .95))},
            "surface_error_after_mm": {"mean": float(np.mean([r["surface_error_after_mm"]["mean"] for r in records])),
                                        "p95": float(np.quantile([r["surface_error_after_mm"]["mean"] for r in records], .95))},
            "mimic_residual_after_deg": {"mean": float(np.mean([r["mimic_residual_after_deg"] for r in records]))},
            "training_eligible": not args.smoke}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        run_manifest.update({"run_status": "COMPLETED", "elapsed_seconds": time.time() - started,
                             "counts": {"sequences": len(records), "frames": summary["num_frames"]},
                             "outputs": {"manifest": str((output / "manifest.json").resolve()), "summary": str((output / "summary.json").resolve())}})
        print(json.dumps({"output_root": str(output), "num_sequences": len(records), "elapsed_seconds": time.time() - started}, ensure_ascii=False))
    except BaseException as error:
        run_manifest.update({"run_status": "FAILED", "elapsed_seconds": time.time() - started,
                             "exit_reason": f"{type(error).__name__}: {error}"})
        raise
    finally:
        (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
