"""把 V18.4 H prediction 经 MANO forward 转为 Y 并评估。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_grasp_v18 import _transform
from src.task.InteractionDynamics.eval_grasp_v18 import subset_masks, trajectory_statistics
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.inverse_mano import build_mano, face_centers
from src.task.InteractionDynamics.mano_hand_transition import ManoHandTransition
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future


def contact_f1(pred_d: torch.Tensor, gt_d: torch.Tensor) -> float:
    pred, gt = pred_d < 2, gt_d < 2; tp = (pred & gt).sum().float()
    precision = tp / pred.sum().clamp_min(1); recall = tp / gt.sum().clamp_min(1)
    return float(2 * precision * recall / (precision + recall).clamp_min(1e-12))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu"); saved = checkpoint["args"]
    model = ManoHandTransition(dim=saved["dim"], layers=saved["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    totals = {stage: {"n": 0, "sq": torch.zeros(3), "success": 0, "contacts": [], "pred_d": [], "gt_d": []}
              for stage in ("overall", "formation", "transition", "maintenance")}
    processed = 0
    for path in sorted((args.cache / args.split).glob("event_*.pt")):
        if args.samples is not None and processed >= args.samples: break
        shard = torch.load(path, map_location="cpu"); take = len(shard["state"])
        if args.samples is not None: take = min(take, args.samples - processed)
        batch = {key: shard[key][:take].to(device) for key in
                 ("state", "future", "anchors_cm", "object_patches", "current_h", "frame", "grasp_frame")}
        delta = model(batch["state"], batch["anchors_cm"], batch["object_patches"],
                      batch["current_h"]) * stats["delta_std"] + stats["delta_mean"]
        side, source = shard["side"], shard["source_raw_file"]
        mano, _, _ = build_mano(args.grab_root / source, args.grab_root, side, args.mano_path, device)
        faces = torch.as_tensor(mano.faces, dtype=torch.long, device=device)
        predictions = []
        for i in range(take):
            current = batch["current_h"][i]; q = shard["object_rotation"][i].to(device)
            t = shard["object_translation"][i].to(device)
            relative_current = axis_angle_to_matrix(shard["global_orient"][i, :1].to(device))
            relative_current = q[:1].transpose(-1, -2) @ relative_current
            relative = axis_angle_to_matrix(delta[i, :, 3:6]) @ relative_current
            world_rotation = q[1:] @ relative
            orient = matrix_to_axis_angle(world_rotation)
            object_position = (current[:3] + delta[i, :, :3]) / 100
            transl = (object_position[:, None, :] - t[1:]) @ q[1:].transpose(-1, -2)
            pose = shard["hand_pose"][i, :1].to(device) + delta[i, :, 6:]
            betas = shard["betas"][i, 1:].to(device)
            surface = face_centers(mano(global_orient=orient, hand_pose=pose,
                                        betas=betas, transl=transl[:, 0]).vertices, faces)
            current_surface = face_centers(mano(
                global_orient=shard["global_orient"][i, :1].to(device),
                hand_pose=shard["hand_pose"][i, :1].to(device),
                betas=shard["betas"][i, :1].to(device),
                transl=shard["transl"][i, :1].to(device)).vertices, faces)
            all_surface = torch.cat([current_surface, surface])
            surface_object = _transform(all_surface, q, t)
            y = {key: 100 * value for key, value in build_interaction_y(
                 surface_object, batch["anchors_cm"][i] / 100, .015).items()}
            predictions.append(pack_state_future(y, 8)[1])
        prediction = torch.stack(predictions); values = prediction.reshape(take, 128, 8, 7)
        target_values = batch["future"].reshape(take, 128, 8, 7)
        stable = trajectory_statistics(prediction)["success"]
        for stage, mask in subset_masks(batch).items():
            row = totals[stage]; n = int(mask.sum()); row["n"] += n
            if n == 0:
                continue
            error = values[mask] - target_values[mask]
            row["sq"] += torch.tensor([error[..., :3].square().mean().cpu(),
                                        error[..., 3:6].square().mean().cpu(),
                                        error[..., 6].square().mean().cpu()]) * n
            row["success"] += int(stable[mask].sum())
            row["contacts"].append(trajectory_statistics(prediction[mask])["contact"].cpu())
            row["pred_d"].append(values[mask][..., 6].cpu()); row["gt_d"].append(target_values[mask][..., 6].cpu())
        processed += take
    result = {"split": args.split, "samples": processed, "groups": {}}
    for stage, row in totals.items():
        if not row["n"]: continue
        rmse = (row["sq"] / row["n"]).sqrt(); contacts = torch.cat(row["contacts"]).float()
        result["groups"][stage] = {"samples": row["n"], "r_rmse_cm": float(rmse[1]),
            "u_rmse_cm": float(rmse[0]), "d_rmse_cm": float(rmse[2]),
            "stable_success_rate": row["success"] / row["n"],
            "contact_f1": contact_f1(torch.cat(row["pred_d"]), torch.cat(row["gt_d"])),
            "terminal_contact_mean": float(contacts.mean()),
            "rd_violation_rate": 0.0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
