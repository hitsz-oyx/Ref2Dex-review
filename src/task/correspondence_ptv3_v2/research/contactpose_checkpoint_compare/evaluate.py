from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.base import load_checkpoint, task_config_from_dict
from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在同一 ContactPose 测试集上比较一个 PTv3 v2 checkpoint。"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    test_root = Path(args.test_root).resolve()
    output_path = Path(args.output).resolve()

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = task_config_from_dict(payload["config"])

    # ContactPose 子集是双方都未训练过的 Stage 3 v2.0 数据。它没有 MANO
    # 参数，因此两边统一采用预存 hand_points 和 legacy candidate mask。
    cfg.data.train_path = str(test_root)
    cfg.data.val_path = str(test_root)
    cfg.data.test_path = None
    cfg.data.val_split = 0.0
    cfg.data.batch_size = args.batch_size
    cfg.data.val_batch_size = args.batch_size
    cfg.data.num_workers = args.num_workers
    cfg.data.persistent_workers = args.num_workers > 0
    cfg.meta.use_mano_reconstruction = False
    cfg.meta.apply_hand_perturb = False
    cfg.meta.runtime_resample_object = False
    cfg.meta.val_obj_perturb_prob = 1.0
    cfg.train.device = args.device
    cfg.train.distributed.enable = False
    cfg.wandb.enable = False

    runner = CorrespondencePTV3V2Runner(
        cfg,
        mode="eval",
        checkpoint=checkpoint_path,
    )
    clean = runner.evaluate_loader(runner.val_loaders["val_clean/"], prefix="test_clean/")
    perturbed = runner.evaluate_loader(
        runner.val_loaders["val_perturbed/"], prefix="test_perturbed/"
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(runner.global_step),
        "checkpoint_epoch": int(runner.start_epoch),
        "test_root": str(test_root),
        "num_sequences": len(runner.val_loader.dataset.file_paths),
        "num_frames": len(runner.val_loader.dataset),
        "protocol": {
            "hand_input": "stored_clean_hand_points",
            "hand_perturb": False,
            "runtime_resample_object": False,
            "object_rotation_std_deg": float(cfg.meta.obj_rot_std_deg),
            "object_translation_std_m": float(cfg.meta.obj_trans_std),
            "object_perturb_probability": 1.0,
        },
        "metrics": {**clean, **perturbed},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
