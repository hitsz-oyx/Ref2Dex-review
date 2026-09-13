"""Export a teacher-forced one-step decoder bank for RL interface smoke.

The exported q/wrist trajectory is produced by the frozen decoder checkpoint
from the paired MANO->actual-Inspire view.  Each frame uses the observed
current target state and the first predicted horizon, so this is an offline
base-policy bank, not an online closed-loop decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.base import load_config
from src.task.CmDecoderv2.dataset import CmDecoderV2Dataset, _collate_cm_decoder
from src.task.CmDecoderv2.kinematics import extract_finger_q, rotvec_to_matrix
from src.task.CmDecoderv2.model import CmDecoderV2


ROOT = Path(__file__).resolve().parents[5]


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "src/task/CmDecoderv2/configs/active/mano_actual_finetune_v1_batch8.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence", default="s1/airplane_lift")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:4")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    index_path = _resolve(config.data.index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = [item for item in index["sequences"]["train"] if str(item["id"]) == args.sequence]
    if len(entries) != 1:
        raise ValueError(f"Expected one train sequence {args.sequence!r}, found {len(entries)}")
    entry = entries[0]
    dataset = CmDecoderV2Dataset(
        entries,
        urdf_path=_resolve(config.data.urdf_path),
        window_size=int(config.meta.window_size),
        num_obj_points=int(config.meta.num_obj_points),
        num_hand_points=int(config.meta.num_hand_points),
        hand_stream_mode=str(config.meta.hand_stream_mode),
        knn_k=int(config.meta.knn_k),
        hand_supervision_radius_m=float(config.meta.hand_supervision_radius_m),
        seed=42,
        perturb=False,
        active_only=False,
        translation_noise_std_m=float(config.data.translation_noise_std_m),
        translation_noise_clip_m=float(config.data.translation_noise_clip_m),
        rotation_noise_std_deg=float(config.data.rotation_noise_std_deg),
        rotation_noise_clip_deg=float(config.data.rotation_noise_clip_deg),
        finger_q_noise_std_rad=float(config.data.finger_q_noise_std_rad),
        finger_q_noise_clip_rad=float(config.data.finger_q_noise_clip_rad),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate_cm_decoder)
    config.model.meta = config.meta
    device = torch.device(args.device)
    model = CmDecoderV2(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    sequence = dataset.sequences[0]
    frame_count = int(sequence.frame_count)
    q_native = np.asarray(sequence.q_native, dtype=np.float32)
    wrist = np.asarray(sequence.wrist_pose, dtype=np.float32)
    q_bank = extract_finger_q(q_native).astype(np.float32)
    wrist_bank = wrist.copy()
    covered = np.zeros(frame_count, dtype=bool)
    row_by_start = {int(start): int(row) for row, (_, start) in enumerate(dataset.rows)}
    with torch.inference_mode():
        for start in range(max(0, frame_count - int(config.meta.window_size))):
            row = row_by_start.get(start)
            if row is None:
                continue
            batch = _collate_cm_decoder([dataset[row]])
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            output = model(batch)
            pred_q = batch["current_finger_q"][0] + output["pred_q_delta"][0, 0]
            pred_t = output["pred_wrist_translation"][0, 0].detach().cpu().numpy()
            pred_r = output["pred_wrist_rotvec"][0, 0].detach().cpu().numpy()
            current_wrist = batch["current_wrist_pose_world"][0].detach().cpu().numpy()
            next_wrist = current_wrist.copy()
            next_wrist[:3, :3] = current_wrist[:3, :3] @ rotvec_to_matrix(pred_r)
            next_wrist[:3, 3] = current_wrist[:3, 3] + current_wrist[:3, :3] @ pred_t
            q_bank[start + 1] = pred_q.detach().cpu().numpy()
            wrist_bank[start + 1] = next_wrist
            covered[start + 1] = True

    args.output_root.mkdir(parents=True, exist_ok=False)
    np.save(args.output_root / "q_finger_decoder_bank.npy", q_bank)
    np.save(args.output_root / "wrist_pose_decoder_bank.npy", wrist_bank)
    manifest = {
        "schema_name": "ref2dex_cm_decoder_v2_rl_decoder_bank_v1",
        "schema_version": "1.0.0",
        "bank_mode": "teacher_forced_one_step",
        "sequence_id": args.sequence,
        "frame_count": frame_count,
        "covered_frames": int(covered.sum()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config": str(_resolve(args.config)),
        "source_index": str(index_path),
        "source_view": str(config.data.view_root),
        "q_path": str((args.output_root / "q_finger_decoder_bank.npy").resolve()),
        "wrist_path": str((args.output_root / "wrist_pose_decoder_bank.npy").resolve()),
        "online_closed_loop": False,
        "conclusion": "SUPPORTED_FOR_INTERFACE_SMOKE_ONLY",
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
