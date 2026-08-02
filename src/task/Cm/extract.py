"""Export CmAction tokens and point-flow predictions from a BaseRunner checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.base import build_runner_from_checkpoint
from src.task.Cm.dataset import Stage4CmDataset
from src.task.Cm.runner import CmActionRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CmAction tokens and predicted point flow.")
    parser.add_argument("--checkpoint", "--cm-checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--stage4-input", required=True, help="One Stage 4 NPZ or a Stage 4 directory.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--include-inactive", action="store_true")
    parser.add_argument("--stride", type=int, default=None, help="Fixed endpoint stride; default samples deterministically.")
    return parser.parse_args()


def _ensure_cm_runner(runner) -> CmActionRunner:
    if not isinstance(runner, CmActionRunner):
        raise ValueError("Cm extract.py only supports CmAction BaseRunner checkpoints.")
    return runner


def _scalar_fields(input_path: Path) -> dict[str, np.ndarray]:
    keys = ("seq_id", "subject_id", "seq_name", "object_name", "side", "coordinate_frame")
    shared_path = input_path.parent / "shared.npz"
    fields: dict[str, np.ndarray] = {}
    with np.load(shared_path, allow_pickle=False) as shared:
        fields.update({key: np.asarray(shared[key]) for key in keys if key in shared.files})
    with np.load(input_path, allow_pickle=False) as hand:
        fields.update({key: np.asarray(hand[key]) for key in keys if key in hand.files})
    return fields


def _predict_batch(runner: CmActionRunner, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = runner.inference(runner.model, batch)
    return output


def extract_file(
    *,
    input_path: Path,
    output_path: Path,
    runner: CmActionRunner,
    batch_size: int,
    start_pair: int,
    max_pairs: int,
    active_only: bool,
    stride: int | None,
) -> None:
    dataset = Stage4CmDataset(
        input_path,
        num_obj_points=int(runner.cfg.meta.num_obj_points),
        num_hand_points=int(runner.cfg.meta.num_hand_points),
        base_seed=int(runner.cfg.train.seed),
        active_only=active_only,
        min_stride=int(runner.cfg.data.min_stride),
        max_stride=int(runner.cfg.data.max_stride),
        fixed_stride=stride,
        coordinate_frame=str(runner.cfg.meta.coordinate_frame),
    )
    if start_pair < 0 or start_pair >= len(dataset):
        raise ValueError(f"start_pair={start_pair} is outside [0, {len(dataset) - 1}] for {input_path}")
    indices = list(range(start_pair, len(dataset)))
    if max_pairs > 0:
        indices = indices[:max_pairs]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    fields = (
        "raw_frame_id", "next_raw_frame_id", "stride", "selected_obj_idx", "obj_valid_mask", "obj_flow_gt",
        "pred_obj_flow", "cm_tokens", "cm_tokens_masked", "cm_anchor_pos", "cm_anchor_normal",
        "cm_assignment", "cm_slot_weights", "slot_gate", "slot_hard_mask", "slot_nonzero_prob",
        "decoder_slot_usage", "dynamic_candidate_flow",
    )
    collected: dict[str, list[np.ndarray]] = {key: [] for key in fields}
    total_squared_error = 0.0
    total_coordinate_count = 0
    for batch in loader:
        prediction = _predict_batch(runner, batch)
        prepared = runner.prepare_batch(batch)
        valid = prepared["obj_valid_mask"].unsqueeze(-1).float()
        total_squared_error += float(
            (((prediction["pred_obj_flow"] - prepared["obj_flow_gt"]) ** 2) * valid).sum().item()
        )
        total_coordinate_count += int(valid.sum().item()) * 3
        tensors = {
            "raw_frame_id": batch["raw_frame_id"],
            "next_raw_frame_id": batch["next_raw_frame_id"],
            "stride": batch["stride"],
            "selected_obj_idx": batch["selected_obj_idx"],
            "obj_valid_mask": batch["obj_valid_mask"],
            "obj_flow_gt": batch["obj_flow_gt"],
            **{key: prediction[key].detach().cpu() for key in fields if key in prediction},
        }
        for key, value in tensors.items():
            collected[key].append(value.detach().cpu().numpy())
    payload = _scalar_fields(input_path)
    payload.update(
        {
            "schema_name": np.asarray("ref2dex_cm_action_tokens"),
            "schema_version": np.asarray("4.1.0"),
            "source_stage4_file": np.asarray(str(input_path.resolve())),
            "source_cm_checkpoint": np.asarray(str(Path(runner.cfg.train.output_dir))),
            "num_pairs": np.asarray(len(indices), dtype=np.int32),
            "flow_mse": np.asarray(total_squared_error / max(total_coordinate_count, 1), dtype=np.float32),
        }
    )
    payload.update({key: np.concatenate(values, axis=0) for key, values in collected.items()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"[cm-extract] wrote {output_path} pairs={len(indices)} flow_mse={float(payload['flow_mse']):.8f}")


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(args.checkpoint, mode="eval", device=args.device, build_data=False)
    runner.setup_inference(args.checkpoint)
    runner = _ensure_cm_runner(runner)
    source = Path(args.stage4_input).resolve()
    inputs = (
        sorted([*source.glob("**/left.npz"), *source.glob("**/right.npz")])
        if source.is_dir()
        else [source]
    )
    if not inputs:
        raise FileNotFoundError(f"No Stage 4 files found at {source}")
    output_root = Path(args.output_root).resolve()
    for input_path in inputs:
        relative = input_path.relative_to(source) if source.is_dir() else Path(input_path.name)
        extract_file(
            input_path=input_path,
            output_path=output_root / relative,
            runner=runner,
            batch_size=args.batch_size,
            start_pair=args.start_pair,
            max_pairs=args.max_pairs,
            active_only=not args.include_inactive,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
