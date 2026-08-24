#!/usr/bin/env python3
"""Calibrate MANO pose noise in millimetres instead of coefficient units.

The input is one or more Stage 3 v2.1 roots.  For every MANO descriptor
(side, pose representation, dimension and hand-mean convention), the tool:

1. measures the root-aligned vertex RMS response of every pose dimension;
2. measures the geometry produced by the legacy uniform coefficient std;
3. derives inverse-sensitivity per-dimension std values; and
4. globally rescales them so their median RMS matches the legacy baseline.

No global orientation or translation is used during calibration.  Each MANO
result is additionally aligned by its own wrist joint before vertex RMS is
computed, so pose-dependent wrist drift cannot leak into the metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.correspondence_ptv3_v2.mano_recon import MANOLayerCache


@dataclass
class SampleGroup:
    side: str
    use_pca: bool
    num_comps: int
    flat_hand_mean: bool
    v_template: np.ndarray | None
    pose: torch.Tensor
    betas: torch.Tensor


def root_aligned_vertices(output: Any) -> torch.Tensor:
    """Return MANO vertices relative to the output wrist joint."""
    return output.vertices - output.joints[:, :1]


def vertex_rms_mm(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Per-sample RMS Euclidean vertex displacement in millimetres."""
    squared_distance = (value - reference).square().sum(dim=-1)
    return squared_distance.mean(dim=-1).sqrt() * 1000.0


def _scalar(data: Any, key: str, default: Any) -> Any:
    if key not in data.files:
        return default
    return np.asarray(data[key]).item()


def _template_key(template: np.ndarray | None) -> str:
    if template is None:
        return "mean"
    value = np.ascontiguousarray(template, dtype=np.float32)
    return hashlib.sha1(value.tobytes()).hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _descriptor(group: SampleGroup) -> tuple[str, bool, int, bool]:
    return group.side, group.use_pca, group.num_comps, group.flat_hand_mean


def _descriptor_name(descriptor: tuple[str, bool, int, bool]) -> str:
    side, use_pca, num_comps, flat_hand_mean = descriptor
    representation = "pca" if use_pca else "axis_angle"
    mean_name = "flat" if flat_hand_mean else "mean"
    return f"{side}_{representation}{num_comps}_{mean_name}"


def _resolve_files(roots: Iterable[Path], max_files: int, seed: int) -> list[Path]:
    files = sorted({path.resolve() for root in roots for path in root.rglob("*.npz")})
    files = [path for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError("No readable Stage 3 .npz files found")
    if max_files > 0 and len(files) > max_files:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(files), size=max_files, replace=False)
        files = [files[index] for index in sorted(chosen.tolist())]
    return files


def load_sample_groups(
    files: list[Path], *, max_samples: int, seed: int
) -> list[SampleGroup]:
    """Load a deterministic, approximately file-balanced Stage 3 sample."""
    rng = np.random.default_rng(seed)
    rows: dict[
        tuple[str, bool, int, bool, str],
        dict[str, Any],
    ] = {}
    per_file = max(1, int(np.ceil(max_samples / len(files))))
    remaining = max_samples
    for path in files:
        if remaining <= 0:
            break
        with np.load(path, allow_pickle=False) as data:
            required = {"mano_pose", "mano_betas"}
            if not required.issubset(data.files):
                continue
            pose = np.asarray(data["mano_pose"], dtype=np.float32)
            betas = np.asarray(data["mano_betas"], dtype=np.float32)
            if pose.ndim != 2 or pose.shape[0] == 0:
                continue
            if betas.ndim == 1:
                betas = np.broadcast_to(betas, (pose.shape[0], betas.shape[0]))
            side = str(_scalar(data, "side", "right"))
            use_pca = bool(_scalar(data, "mano_use_pca", pose.shape[1] != 45))
            num_comps = int(_scalar(data, "mano_num_pca_comps", pose.shape[1]))
            flat_hand_mean = bool(_scalar(data, "mano_flat_hand_mean", True))
            template = (
                np.asarray(data["mano_v_template"], dtype=np.float32)
                if "mano_v_template" in data.files
                else None
            )
            take = min(per_file, remaining, pose.shape[0])
            indices = np.sort(rng.choice(pose.shape[0], size=take, replace=False))
            key = (side, use_pca, num_comps, flat_hand_mean, _template_key(template))
            row = rows.setdefault(
                key,
                {
                    "side": side,
                    "use_pca": use_pca,
                    "num_comps": num_comps,
                    "flat_hand_mean": flat_hand_mean,
                    "v_template": template,
                    "pose": [],
                    "betas": [],
                },
            )
            row["pose"].append(pose[indices, :num_comps].copy())
            row["betas"].append(betas[indices].copy())
            remaining -= take
    if not rows:
        raise ValueError("Stage 3 files contain no usable MANO v2.1 samples")
    return [
        SampleGroup(
            side=row["side"],
            use_pca=row["use_pca"],
            num_comps=row["num_comps"],
            flat_hand_mean=row["flat_hand_mean"],
            v_template=row["v_template"],
            pose=torch.from_numpy(np.concatenate(row["pose"], axis=0)),
            betas=torch.from_numpy(np.concatenate(row["betas"], axis=0)),
        )
        for row in rows.values()
    ]


def _forward_root_aligned(layer: Any, pose: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    batch_size = pose.shape[0]
    output = layer(
        global_orient=torch.zeros((batch_size, 3), device=pose.device),
        hand_pose=pose,
        transl=torch.zeros((batch_size, 3), device=pose.device),
        betas=betas,
    )
    return root_aligned_vertices(output)


def _group_layers(
    groups: list[SampleGroup], cache: MANOLayerCache, device: torch.device
) -> list[tuple[SampleGroup, Any, torch.Tensor]]:
    result = []
    for group in groups:
        layer, _ = cache.get_or_build_for_v_template(
            side=group.side,
            use_pca=group.use_pca,
            num_pca_comps=group.num_comps,
            flat_hand_mean=group.flat_hand_mean,
            v_template=group.v_template,
        )
        group.pose = group.pose.to(device=device, dtype=torch.float32)
        group.betas = group.betas.to(device=device, dtype=torch.float32)
        baseline = _forward_root_aligned(layer, group.pose, group.betas)
        result.append((group, layer, baseline))
    return result


def _rms_for_noise(
    prepared: list[tuple[SampleGroup, Any, torch.Tensor]],
    noise_rows: list[torch.Tensor],
) -> np.ndarray:
    values = []
    for (group, layer, baseline), noise in zip(prepared, noise_rows):
        perturbed = _forward_root_aligned(layer, group.pose + noise, group.betas)
        values.append(vertex_rms_mm(baseline, perturbed).detach().cpu().numpy())
    return np.concatenate(values)


def calibrate_descriptor(
    groups: list[SampleGroup],
    *,
    cache: MANOLayerCache,
    device: torch.device,
    reference_std: float,
    target_rms_mm: float | None,
    clip_sigma: float,
    seed: int,
) -> dict[str, Any]:
    num_comps = groups[0].num_comps
    prepared = _group_layers(groups, cache, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    standard_noise = [
        torch.randn(
            (group.pose.shape[0], num_comps), generator=generator, device=device
        ).clamp(-clip_sigma, clip_sigma)
        for group, _layer, _baseline in prepared
    ]
    reference_values = _rms_for_noise(
        prepared, [noise * reference_std for noise in standard_noise]
    )
    reference_median = float(np.median(reference_values))
    target_median = (
        reference_median if target_rms_mm is None else float(target_rms_mm)
    )
    if target_median <= 0.0:
        raise ValueError("target_rms_mm must be positive")

    impacts: list[list[np.ndarray]] = [[] for _ in range(num_comps)]
    probe = reference_std
    for dim in range(num_comps):
        for group, layer, baseline in prepared:
            delta = torch.zeros_like(group.pose)
            delta[:, dim] = probe
            plus = _forward_root_aligned(layer, group.pose + delta, group.betas)
            minus = _forward_root_aligned(layer, group.pose - delta, group.betas)
            symmetric = 0.5 * (
                vertex_rms_mm(baseline, plus) + vertex_rms_mm(baseline, minus)
            )
            impacts[dim].append(symmetric.detach().cpu().numpy())
    impact_matrix = np.stack(
        [np.concatenate(per_dim) for per_dim in impacts], axis=1
    )
    sensitivity = np.median(impact_matrix, axis=0) / probe
    if np.any(~np.isfinite(sensitivity)) or np.any(sensitivity <= 0):
        raise RuntimeError(f"Invalid per-dimension sensitivity: {sensitivity.tolist()}")

    # One unit of scale gives every dimension the same first-order median
    # vertex RMS. A global search then accounts for simultaneous dimensions
    # and MANO's non-linearity while matching the old geometry distribution.
    inverse_sensitivity = 1.0 / sensitivity
    low, high = 0.0, target_median
    balanced_values = np.zeros_like(reference_values)
    for _ in range(16):
        scale = 0.5 * (low + high)
        std = torch.as_tensor(
            inverse_sensitivity * scale, dtype=torch.float32, device=device
        )
        balanced_values = _rms_for_noise(
            prepared, [noise * std for noise in standard_noise]
        )
        if float(np.median(balanced_values)) < target_median:
            low = scale
        else:
            high = scale
    geometry_scale_mm = 0.5 * (low + high)
    balanced_std = inverse_sensitivity * geometry_scale_mm
    balanced_values = _rms_for_noise(
        prepared,
        [
            noise
            * torch.as_tensor(balanced_std, dtype=torch.float32, device=device)
            for noise in standard_noise
        ],
    )

    descriptor = _descriptor(groups[0])
    return {
        "descriptor": _descriptor_name(descriptor),
        "side": descriptor[0],
        "representation": "pca" if descriptor[1] else "axis_angle",
        "num_components": num_comps,
        "flat_hand_mean": descriptor[3],
        "num_samples": int(sum(group.pose.shape[0] for group in groups)),
        "metric": "wrist-aligned MANO 778-vertex RMS (mm)",
        "reference_coefficient_std": reference_std,
        "target_median_rms_mm": target_median,
        "clip_sigma": clip_sigma,
        "reference_rms_mm": _quantiles(reference_values),
        "per_dimension_sensitivity_mm_per_coefficient": sensitivity.tolist(),
        "per_dimension_probe_rms_mm": {
            "p50": np.median(impact_matrix, axis=0).tolist(),
            "p95": np.quantile(impact_matrix, 0.95, axis=0).tolist(),
        },
        "balanced_geometry_scale_mm": geometry_scale_mm,
        "balanced_coefficient_std_per_dimension": balanced_std.tolist(),
        "balanced_rms_mm": _quantiles(balanced_values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Stage 3 v2.1 roots")
    parser.add_argument("--mano-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--pca-reference-std", type=float, default=0.5)
    parser.add_argument("--axis-angle-reference-std", type=float, default=0.05)
    parser.add_argument(
        "--target-rms-mm",
        type=float,
        default=None,
        help="Target balanced median RMS; default matches the legacy reference median",
    )
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    device = torch.device(args.device)
    files = _resolve_files(args.inputs, args.max_files, args.seed)
    groups = load_sample_groups(files, max_samples=args.max_samples, seed=args.seed)
    by_descriptor: dict[tuple[str, bool, int, bool], list[SampleGroup]] = {}
    for group in groups:
        by_descriptor.setdefault(_descriptor(group), []).append(group)
    cache = MANOLayerCache(model_dir=args.mano_model_dir, device=device)
    results = []
    with torch.inference_mode():
        for descriptor, descriptor_groups in sorted(by_descriptor.items()):
            reference_std = (
                args.pca_reference_std if descriptor[1] else args.axis_angle_reference_std
            )
            print(
                f"[calibrate] {_descriptor_name(descriptor)} "
                f"samples={sum(group.pose.shape[0] for group in descriptor_groups)}"
            )
            results.append(
                calibrate_descriptor(
                    descriptor_groups,
                    cache=cache,
                    device=device,
                    reference_std=reference_std,
                    target_rms_mm=args.target_rms_mm,
                    clip_sigma=args.clip_sigma,
                    seed=args.seed,
                )
            )
    payload = {
        "schema": "ref2dex_mano_geometry_noise_calibration_v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "sampled_files": len(files),
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[calibrate] wrote {args.output}")
    for result in results:
        reference = result["reference_rms_mm"]
        balanced = result["balanced_rms_mm"]
        print(
            f"[calibrate] {result['descriptor']} reference p50={reference['p50']:.3f} mm "
            f"p95={reference['p95']:.3f} mm; balanced p50={balanced['p50']:.3f} mm "
            f"p95={balanced['p95']:.3f} mm"
        )


if __name__ == "__main__":
    main()
