"""导出 InteractionDynamics V1 的表征、注意力和 object-effect 诊断结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.base import build_runner_from_checkpoint
from src.task.InteractionDynamics.runner import InteractionDynamicsRunner


PREDICTION_FIELDS = (
    "action_tokens", "action_context_tokens", "interaction_tokens",
    "pred_hand_patch_disp_internal", "pred_obj_patch_disp_internal",
    "pred_obj_disp_chunk", "effect_patch_index", "action_to_world_attention",
    "object_to_action_attention", "effect_to_interaction_attention",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="可选配置；默认读取 checkpoint 内保存的配置。")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output-root", default="output/InteractionDynamics/grab_v1/extraction")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=0, help="0 表示导出整个 split。")
    return parser.parse_args()


def _ensure_runner(runner: Any) -> InteractionDynamicsRunner:
    if not isinstance(runner, InteractionDynamicsRunner):
        raise ValueError("extract.py 只支持 InteractionDynamicsRunner checkpoint")
    return runner


def _sample_identity(hand_path: Path) -> tuple[str, str]:
    with np.load(hand_path, allow_pickle=False) as hand:
        side = str(hand["side"].item())
    with np.load(hand_path.parent / "shared.npz", allow_pickle=False) as shared:
        seq_id = str(shared["seq_id"].item())
    return seq_id, side


def _cpu_array(value: torch.Tensor, index: int) -> np.ndarray:
    return value[index].detach().cpu().numpy()


def extract_split(*, runner: InteractionDynamicsRunner, split: str,
                  output_root: Path, source_checkpoint: Path,
                  max_samples: int = 0) -> dict[str, Any]:
    loaders = runner.val_loaders if split == "val" else runner.test_loaders
    prefix = f"{split}/"
    if prefix not in loaders:
        raise ValueError(f"checkpoint 配置中没有 {split} split")
    loader = loaders[prefix]
    dataset = loader.dataset
    limit = len(dataset) if max_samples <= 0 else min(int(max_samples), len(dataset))
    output_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    runner.eval_mode()
    for raw_batch in loader:
        if exported >= limit:
            break
        batch = runner.prepare_batch(raw_batch)
        with runner.eval_context():
            prediction = runner.model(batch)
        batch_size = int(batch["raw_frame_id"].shape[0])
        for batch_index in range(min(batch_size, limit - exported)):
            hand_path, _ = dataset.sample_location(exported)
            seq_id, side = _sample_identity(hand_path)
            raw_frame_id = int(batch["raw_frame_id"][batch_index].item())
            payload = {
                "schema_name": np.asarray("ref2dex_interaction_dynamics_extraction"),
                "schema_version": np.asarray("1.0.0"),
                "seq_id": np.asarray(seq_id), "side": np.asarray(side),
                "raw_frame_id": np.asarray(raw_frame_id, dtype=np.int64),
                "future_raw_frame_ids": _cpu_array(batch["future_raw_frame_ids"], batch_index),
                "effect_obj_idx": _cpu_array(batch["effect_obj_idx"], batch_index),
                "world_hand_patch_centers_object": _cpu_array(
                    prediction["hand_patch_centers_object"], batch_index),
                "world_obj_patch_centers_object": _cpu_array(
                    prediction["obj_patch_centers_object"], batch_index),
                "hand_patch_center_hand": _cpu_array(prediction["hand_patch_center_hand"], batch_index),
                "hand_patch_center_object": _cpu_array(
                    prediction["hand_patch_center_object"], batch_index),
                "gt_obj_disp_chunk": _cpu_array(batch["effect_obj_disp_gt"], batch_index),
                "effect_obj_points": _cpu_array(batch["effect_obj_points_object"], batch_index),
                "effect_obj_valid_mask": _cpu_array(batch["effect_obj_valid_mask"], batch_index),
            }
            payload.update({key: _cpu_array(prediction[key], batch_index) for key in PREDICTION_FIELDS})
            output_path = output_root / seq_id / side / f"{raw_frame_id:06d}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **payload)
            exported += 1
    summary = {
        "schema_name": "ref2dex_interaction_dynamics_extraction_summary",
        "schema_version": "1.0.0", "split": split,
        "checkpoint": str(source_checkpoint.resolve()),
        "num_samples": exported, "output_root": str(output_root.resolve()),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    runner = _ensure_runner(build_runner_from_checkpoint(
        args.checkpoint, config=args.config, mode="eval", device=args.device,
    ))
    summary = extract_split(
        runner=runner, split=args.split, output_root=Path(args.output_root),
        source_checkpoint=Path(args.checkpoint), max_samples=args.max_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
