"""缓存冻结 V20 Free-Y 与 V20.3 Direct-H 的 train/val/test 预测。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.eval_parallel_fusion_v20_4 import load_direct
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.train_h_realizer_v20_2 import load_field, predict_field
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, move


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--field-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-h-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    args = parser.parse_args(); device = torch.device("cuda")
    field, mean, std = load_field(args.field_checkpoint, device)
    direct, h_std = load_direct(args.direct_h_checkpoint, device)
    layers = ManoLayers(args.grab_root, args.mano_path, device); args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        loader = DataLoader(CachedFieldV20Dataset(args.cache, split), args.batch_size,
                            shuffle=False, num_workers=0)
        rows = {key: [] for key in ("y_free", "y_h", "future_y", "current_y")}
        with torch.no_grad():
            for raw in loader:
                batch = move(raw, device); y_free = predict_field(field, batch, mean, std)
                z = direct(batch["current_y"], batch["anchors_cm"],
                           batch["object_patches"], batch["current_h"])
                y_h, _ = decode_mano_field(z * h_std, batch, layers)
                for key, value in (("y_free", y_free), ("y_h", y_h),
                                   ("future_y", batch["future_y"][..., :7]),
                                   ("current_y", batch["current_y"])):
                    rows[key].append(value.cpu())
        payload = {key: torch.cat(value) for key, value in rows.items()}
        payload["field_std"] = std.cpu()
        torch.save(payload, args.output / f"{split}.pt")
        print(f"{split}: {len(payload['y_free'])} samples", flush=True)


if __name__ == "__main__":
    main()
