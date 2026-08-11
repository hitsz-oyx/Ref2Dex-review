"""复用 checkpoint 评估 V16.4 diffusion 的 effect conditioning。"""
from __future__ import annotations

import argparse
import json

import torch

from src.task.InteractionDynamics.eval_interaction_compression import make_dataset
from src.task.InteractionDynamics.interaction_diffusion import InteractionFieldDiffusion
from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
from src.task.InteractionDynamics.train_interaction_diffusion import (
    field_metrics, generate, materialize)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_full.yaml")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tau-m", type=float, default=.015)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    dataset = make_dataset(args.config)
    data = materialize(dataset, checkpoint["indices"], device, args.tau_m)
    model = InteractionFieldDiffusion(checkpoint["dim"], 8, checkpoint["layers"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    _, alpha_bar = cosine_schedule(args.diffusion_steps, device)
    batch = {name: value.to(device) for name, value in data.items()}
    clean = (batch["y"] - checkpoint["y_mean"].to(device)) / checkpoint["y_std"].to(device)
    noise = torch.randn(clean.shape, device=device,
                        generator=torch.Generator(device=device).manual_seed(20260811))
    for timestep_value in [args.diffusion_steps // 2, args.diffusion_steps - 1]:
        timestep = torch.full((len(clean),), timestep_value, device=device, dtype=torch.long)
        scale = alpha_bar[timestep_value]
        noisy = scale.sqrt() * clean + (1 - scale).sqrt() * noise
        for mode in ["correct", "shuffle", "zero"]:
            raw_effect = (batch["effect"] if mode == "correct" else
                          batch["effect"].roll(1, 0) if mode == "shuffle" else
                          torch.zeros_like(batch["effect"]))
            effect = ((raw_effect - checkpoint["effect_mean"].to(device))
                      / checkpoint["effect_std"].to(device))
            prediction = model(noisy, batch["anchors_cm"], batch["anchor_normals"],
                               effect, timestep)
            print(json.dumps({"diagnostic": "denoise", "timestep": timestep_value,
                              "effect_mode": mode,
                              "noise_mse": float(torch.nn.functional.mse_loss(prediction, noise))},
                             ensure_ascii=False))
    for mode in ["correct", "shuffle", "zero"]:
        prediction = generate(
            model, data, checkpoint["y_mean"].to(device), checkpoint["y_std"].to(device),
            checkpoint["effect_mean"].to(device), checkpoint["effect_std"].to(device),
            args.diffusion_steps, args.batch_size, device, mode)
        print(json.dumps({"effect_mode": mode, **field_metrics(prediction, data["y"])},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
