"""Test whether the MANO point decoder corrects perturbed current states.

The source action token is held fixed from the unperturbed ARCTIC sample while
the target hand points are translated by a controlled offset.  This isolates
state-feedback from changes in the action token.  Positive correction
projection means that the predicted flow points toward the unperturbed next
hand state.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.base.utils import import_from_path
from src.task.Cm.dataset_object_v2 import _MmapSequenceDataset


def _as_batch(sample: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _scenario_offsets(
    current: np.ndarray,
    obj_points: np.ndarray,
    obj_valid: np.ndarray,
    magnitude_m: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    valid = obj_points[obj_valid.astype(bool)]
    if valid.size == 0:
        valid = obj_points
    hand_center = current.mean(axis=0)
    object_center = valid.mean(axis=0)
    toward = _normalize(object_center - hand_center)
    random_direction = _normalize(rng.normal(size=3).astype(np.float32))
    amount = float(magnitude_m)
    return {
        "clean": np.zeros(3, dtype=np.float32),
        "toward_object": toward * amount,
        "away_from_object": -toward * amount,
        "random": random_direction * amount,
    }


def _evaluate_prediction(
    current: np.ndarray,
    future: np.ndarray,
    offset: np.ndarray,
    predicted_flow: np.ndarray,
) -> dict[str, float]:
    perturbed = current + offset
    predicted_next = perturbed + predicted_flow
    target_error = future - perturbed
    offset_norm_sq = float(np.dot(offset, offset))
    target_norm = float(np.linalg.norm(target_error, axis=-1).mean())
    flow_norm = float(np.linalg.norm(predicted_flow, axis=-1).mean())
    correction_projection = np.sum(predicted_flow * target_error, axis=-1)
    correction_projection = correction_projection / np.sum(target_error * target_error, axis=-1).clip(1e-10)
    correction_against_offset = np.sum(predicted_flow * (-offset[None, :]), axis=-1)
    if offset_norm_sq > 1e-10:
        correction_against_offset = correction_against_offset / offset_norm_sq
    else:
        correction_against_offset = np.zeros_like(correction_against_offset)
    return {
        "pred_next_epe_mm": float(np.linalg.norm(predicted_next - future, axis=-1).mean() * 1000.0),
        "target_flow_norm_mm": target_norm * 1000.0,
        "pred_flow_norm_mm": flow_norm * 1000.0,
        "correction_projection": float(correction_projection.mean()),
        "correction_against_offset": float(correction_against_offset.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--arctic-root", type=Path,
        default=Path("data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/arctic"),
    )
    parser.add_argument("--sequence", default="s01/box_use_01")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/research/arctic_mano_cm64_perturbation.json"),
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = load_checkpoint(args.decoder_checkpoint.resolve(), map_location="cpu")
    cfg = task_config_from_dict(payload["config"])
    model_cfg = copy.copy(cfg.model)
    model_cfg.meta = cfg.meta
    model = import_from_path(model_cfg.class_path)(model_cfg).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)

    sequence = (args.arctic_root / args.sequence).resolve()
    if not (sequence / "shared" / "meta.json").is_file():
        raise FileNotFoundError(f"ARCTIC object-v2 sequence not found: {sequence}")
    dataset = _MmapSequenceDataset(
        [sequence], num_obj_points=512, num_hand_points=1538,
        fixed_stride=1, min_stride=1, max_stride=1, active_only=True,
    )
    selected_rows = [
        index for index, (path, side, _frame) in enumerate(dataset.rows)
        if path.resolve() == sequence and side == args.side
    ]
    start = int(args.start_frame)
    rows = selected_rows[start:start + int(args.frames)]
    if not rows:
        raise ValueError("No evaluation rows selected")

    magnitudes_m = (0.005, 0.010, 0.020)
    accum: dict[str, list[dict[str, float]]] = {}
    rng = np.random.default_rng(int(args.seed))
    with torch.inference_mode():
        for row_index in rows:
            sample = dataset[row_index]
            batch = _as_batch(sample, device)
            current = sample["hand_points"].numpy().astype(np.float32)
            future = current + sample["hand_flow"].numpy().astype(np.float32)
            obj_points = sample["obj_points"].numpy().astype(np.float32)
            obj_valid = sample["obj_valid_mask"].numpy()

            clean_prediction = model(batch)
            # Hold the source action token fixed.  The perturbed state is
            # therefore tested as a target-side state-feedback input.
            cm_tokens = clean_prediction["cm_tokens"].detach()
            for magnitude_m in magnitudes_m:
                offsets = _scenario_offsets(current, obj_points, obj_valid, magnitude_m, rng)
                for scenario, offset in offsets.items():
                    perturbed_batch = dict(batch)
                    perturbed_batch["hand_points"] = torch.from_numpy(current + offset)[None].to(device)
                    perturbed_batch["cm_tokens"] = cm_tokens
                    prediction = model(perturbed_batch)
                    predicted_flow = prediction["pred_hand_flow"][0].cpu().numpy().astype(np.float32)
                    metrics = _evaluate_prediction(current, future, offset, predicted_flow)
                    metrics.update({"magnitude_mm": magnitude_m * 1000.0})
                    key = f"{scenario}_{int(round(magnitude_m * 1000.0))}mm"
                    accum.setdefault(key, []).append(metrics)

    summary = {}
    for scenario, records in accum.items():
        keys = records[0].keys()
        summary[scenario] = {
            key: float(np.mean([record[key] for record in records]))
            for key in keys
        }
    result = {
        "checkpoint": str(args.decoder_checkpoint),
        "sequence": args.sequence,
        "side": args.side,
        "start_frame": int(args.start_frame),
        "frames": len(rows),
        "magnitudes_mm": [5.0, 10.0, 20.0],
        "token_mode": "fixed_clean_cm_tokens",
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
