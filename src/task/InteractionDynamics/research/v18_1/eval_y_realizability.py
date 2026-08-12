"""比较 V18 生成 Y 与真实 Y 的 MANO inverse 可实现性。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_grasp_v18 import (
    _rigid_world_to_reference, _transform, load_events)
from src.task.InteractionDynamics.grasp_interaction_diffusion import (
    GraspInteractionDiffusion, sample_grasp_v)
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.inverse_mano import build_mano, face_centers
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def optimize(mano, faces, betas, current, object_rotation, object_translation,
             anchors, target, steps, lr) -> tuple[float, float]:
    pose0, orient0, transl0 = current
    pose = torch.nn.Parameter(pose0.expand(8, -1).clone())
    orient = torch.nn.Parameter(orient0.expand(8, -1).clone())
    transl = torch.nn.Parameter(transl0.expand(8, -1).clone())
    optimizer = torch.optim.Adam([pose, orient, transl], lr=lr)

    def predict() -> torch.Tensor:
        output = mano(global_orient=torch.cat([orient0, orient]),
                      hand_pose=torch.cat([pose0, pose]), betas=betas,
                      transl=torch.cat([transl0, transl]))
        surface = face_centers(output.vertices, faces)
        surface_object = _transform(surface, object_rotation, object_translation)
        y = {key: 100 * value for key, value in
             build_interaction_y(surface_object, anchors, .015).items()}
        return pack_state_future(y, 8)[1]

    with torch.no_grad(): initial = float((predict() - target).square().mean().sqrt())
    for _ in range(steps):
        optimizer.zero_grad(); loss = (predict() - target).square().mean()
        loss.backward(); optimizer.step()
    with torch.no_grad(): final = float((predict() - target).square().mean().sqrt())
    return initial, final


def main() -> None:
    args = parse_args(); device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]; model_cfg = cfg["model"]
    model = GraspInteractionDiffusion(model_cfg["horizon"], model_cfg["dim"],
                                      model_cfg["heads"], model_cfg["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    events = load_events(args.events); paths = sorted(args.cache.glob("event_*.pt"))[:args.samples]
    rows = []
    for sample_number, cache_path in enumerate(paths):
        event_index = int(cache_path.stem.rsplit("_", 1)[-1]); event = events[event_index]
        shard = torch.load(cache_path, map_location="cpu")
        state = shard["state"][:1].to(device); gt = shard["future"][:1].to(device)
        frame = int(shard["frame"][0]); anchors = shard["anchors_cm"][0].to(device) / 100
        normalized = (state - stats["state_mean"]) / stats["state_std"]
        torch.manual_seed(1000 + sample_number)
        with torch.no_grad():
            residual = sample_grasp_v(model, normalized, shard["anchors_cm"][:1].to(device),
                                      shard["object_patches"][:1].to(device),
                                      sampling_steps=cfg["diffusion"]["sampling_steps"])
            generated = persistence_future(state, 8) + residual * stats["residual_std"] + stats["residual_mean"]
        with np.load(event.path.parent / "shared.npz", allow_pickle=False) as shared:
            source = str(shared["source_raw_file"].item())
            raw_ids = np.asarray(shared["raw_frame_id"][frame:frame + 9], np.int64)
            object_world = torch.from_numpy(np.asarray(shared["obj_points_world"], np.float32)).to(device)
        with np.load(event.path, allow_pickle=False) as hand: side = str(hand["side"].item())
        mano, _, params = build_mano(args.grab_root / source, args.grab_root,
                                     side, args.mano_path, device)
        def param(name: str, width: int) -> torch.Tensor:
            return torch.from_numpy(np.asarray(params[name][raw_ids], np.float32).reshape(9, width)).to(device)
        orient, pose, transl = param("global_orient", 3), param("hand_pose", 24), param("transl", 3)
        beta = np.asarray(params["betas"], np.float32)
        beta = beta[raw_ids] if beta.ndim > 1 else np.broadcast_to(beta, (9, beta.shape[-1])).copy()
        betas = torch.from_numpy(beta).to(device)
        rotation, translation = _rigid_world_to_reference(object_world)
        transforms = (rotation[frame:frame + 9], translation[frame:frame + 9])
        current = (pose[:1], orient[:1], transl[:1])
        faces = torch.as_tensor(mano.faces, dtype=torch.long, device=device)
        gt_initial, gt_final = optimize(mano, faces,
                                        betas, current, *transforms, anchors, gt[0], args.steps, args.lr)
        gen_initial, gen_final = optimize(mano, faces,
                                          betas, current, *transforms, anchors, generated[0], args.steps, args.lr)
        rows.append({"event_index": event_index, "frame": frame,
                     "gt_initial_rmse_cm": gt_initial, "gt_final_rmse_cm": gt_final,
                     "generated_initial_rmse_cm": gen_initial,
                     "generated_final_rmse_cm": gen_final})
        print(json.dumps(rows[-1]), flush=True)
    result = {"checkpoint_epoch": checkpoint["epoch"], "steps": args.steps, "samples": rows,
              "gt_final_mean_rmse_cm": sum(r["gt_final_rmse_cm"] for r in rows) / len(rows),
              "generated_final_mean_rmse_cm": sum(r["generated_final_rmse_cm"] for r in rows) / len(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__": main()
