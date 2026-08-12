"""从 V18 cache 与 raw GRAB 构建独立 MANO-H cache，并执行 GT H→Y parity。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_grasp_v18 import (
    _rigid_world_to_reference, _transform, load_events)
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.inverse_mano import build_mano, face_centers
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future


def hand_state(global_orient: torch.Tensor, hand_pose: torch.Tensor,
               transl: torch.Tensor, object_rotation: torch.Tensor,
               object_translation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 object-frame H condition 和每帧 object-frame wrist rotation。"""
    world_rotation = axis_angle_to_matrix(global_orient)
    object_hand_rotation = object_rotation.transpose(-1, -2) @ world_rotation
    object_hand_translation = transl[:, None, :] @ object_rotation + object_translation
    condition = torch.cat([100 * object_hand_translation[:, 0],
                           matrix_to_rotation_6d(object_hand_rotation), hand_pose], -1)
    return condition, object_hand_rotation


def delta_h(condition: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    relative_rotation = rotations[1:] @ rotations[:1].transpose(-1, -2)
    return torch.cat([condition[1:, :3] - condition[:1, :3],
                      matrix_to_axis_angle(relative_rotation),
                      condition[1:, 9:] - condition[:1, 9:]], -1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--v18-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); events = load_events(args.events)
    summary = {}
    for split in ("train", "val", "test"):
        output_dir = args.output / split; output_dir.mkdir(parents=True, exist_ok=True)
        total = 0; parity = []
        for source_path in sorted((args.v18_cache / split).glob("event_*.pt")):
            if total >= args.max_samples: break
            event_index = int(source_path.stem.rsplit("_", 1)[-1]); event = events[event_index]
            source = torch.load(source_path, map_location="cpu")
            take = min(len(source["state"]), args.max_samples - total)
            with np.load(event.path.parent / "shared.npz", allow_pickle=False) as shared:
                raw_source = str(shared["source_raw_file"].item())
                raw_frame_id = np.asarray(shared["raw_frame_id"], np.int64)
                object_world = torch.from_numpy(np.asarray(shared["obj_points_world"], np.float32)).to(device)
            with np.load(event.path, allow_pickle=False) as hand:
                side = str(hand["side"].item())
            mano, _, params = build_mano(args.grab_root / raw_source, args.grab_root,
                                         side, args.mano_path, device)
            faces = torch.as_tensor(mano.faces, dtype=torch.long, device=device)
            object_rotation, object_translation = _rigid_world_to_reference(object_world)
            rows = {key: [] for key in ("current_h", "future_delta_h", "betas", "raw_frame_ids",
                    "object_rotation", "object_translation", "global_orient", "hand_pose", "transl")}
            for local in range(take):
                frame = int(source["frame"][local]); cache_ids = np.arange(frame, frame + 9)
                raw_ids = raw_frame_id[cache_ids]
                def param(name: str, width: int) -> torch.Tensor:
                    return torch.from_numpy(np.asarray(params[name][raw_ids], np.float32).reshape(9, width)).to(device)
                orient, pose, transl = param("global_orient", 3), param("hand_pose", 24), param("transl", 3)
                beta = np.asarray(params["betas"], np.float32)
                beta = beta[raw_ids] if beta.ndim > 1 else np.broadcast_to(beta, (9, beta.shape[-1])).copy()
                betas = torch.from_numpy(beta).to(device)
                q, t = object_rotation[cache_ids], object_translation[cache_ids]
                h, rotations = hand_state(orient, pose, transl, q, t)
                surface = face_centers(mano(global_orient=orient, hand_pose=pose,
                                            betas=betas, transl=transl).vertices, faces)
                surface_object = _transform(surface, q, t)
                anchors = source["anchors_cm"][local].to(device) / 100
                reconstructed = {key: 100 * value for key, value in
                                 build_interaction_y(surface_object, anchors, .015).items()}
                _, future = pack_state_future(reconstructed, 8)
                parity.append(float((future.cpu() - source["future"][local]).abs().max()))
                values = (h[0], delta_h(h, rotations), betas, torch.from_numpy(raw_ids), q, t,
                          orient, pose, transl)
                for key, value in zip(rows, values): rows[key].append(value.cpu())
            shard = {key: value[:take] for key, value in source.items()}
            shard.update({key: torch.stack(value) for key, value in rows.items()})
            shard.update({"side": side, "source_raw_file": raw_source,
                          "source_hand_cache": str(event.path)})
            torch.save(shard, output_dir / source_path.name); total += take
        summary[split] = {"samples": total, "parity_max_abs_cm": max(parity),
                          "parity_mean_max_abs_cm": sum(parity) / len(parity)}
        print(split, summary[split], flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "parity.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__": main()
