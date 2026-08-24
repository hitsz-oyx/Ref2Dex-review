from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.base import load_checkpoint, task_config_from_dict
from src.task.correspondence_ptv3_v2.config import Config
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
    parser.add_argument(
        "--condition",
        choices=("object_only", "hand_only", "hand_object"),
        default="object_only",
        help="Perturbation condition for the validation stream.",
    )
    parser.add_argument(
        "--hand-target-rms-mm",
        type=float,
        default=5.0,
        help="ARCTIC axis-angle hand-noise target RMS in mm (hand conditions only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    test_root = Path(args.test_root).resolve()
    output_path = Path(args.output).resolve()

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    cfg = task_config_from_dict(payload["config"])

    # The evaluator keeps the historical object-only protocol by default, but
    # can also use the MANO fields in regenerated ARCTIC Stage 3 data.
    cfg.data.train_path = str(test_root)
    cfg.data.val_path = str(test_root)
    cfg.data.test_path = None
    cfg.data.val_split = 0.0
    cfg.data.batch_size = args.batch_size
    cfg.data.val_batch_size = args.batch_size
    cfg.data.dataset_id = "arctic" if "arctic" in test_root.name.lower() else None
    cfg.data.num_workers = args.num_workers
    cfg.data.persistent_workers = args.num_workers > 0
    is_hand_condition = args.condition in {"hand_only", "hand_object"}
    if is_hand_condition:
        # Historical v2.0 checkpoints predate the MANO config fields. Use the
        # current task default so their weights can still be evaluated on a
        # regenerated MANO test set without mutating the checkpoint itself.
        if not getattr(cfg.meta, "mano_model_dir", None):
            cfg.meta.mano_model_dir = Config.meta.mano_model_dir
        cfg.meta.use_mano_reconstruction = True
        cfg.meta.apply_hand_perturb = True
        cfg.meta.exclusive_hand_object_perturb = False
        # Evaluation conditions describe a deterministic corruption stream,
        # not the checkpoint's training-time hand/object exposure ratio.
        # Without this override H80 and H50 checkpoints inherit 0.8 and 0.5,
        # respectively, so they are evaluated on different fractions of
        # perturbed frames despite sharing the same condition name.
        cfg.meta.hand_perturb_prob = 1.0
        cfg.meta.hand_geometry_calibration_paths = {
            "arctic": str(
                Path("src/task/correspondence_ptv3_v2/calibration/arctic_axis_angle45_target_9mm_sample.json")
                .resolve()
            )
        }
        cfg.meta.hand_geometry_noise_scale = float(args.hand_target_rms_mm) / 9.0
        cfg.meta.hand_geometry_noise_required = True
    else:
        cfg.meta.use_mano_reconstruction = False
        cfg.meta.apply_hand_perturb = False
        cfg.meta.exclusive_hand_object_perturb = False
    cfg.meta.val_apply_obj_perturb = args.condition in {"object_only", "hand_object"}
    cfg.meta.apply_obj_perturb = cfg.meta.val_apply_obj_perturb
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
            "condition": args.condition,
            "hand_perturb": is_hand_condition,
            "hand_perturb_probability": 1.0 if is_hand_condition else 0.0,
            "hand_perturb_representation": "axis_angle45" if is_hand_condition else None,
            "hand_target_rms_mm": float(args.hand_target_rms_mm) if is_hand_condition else None,
            "runtime_resample_object": False,
            "object_perturb": args.condition in {"object_only", "hand_object"},
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
