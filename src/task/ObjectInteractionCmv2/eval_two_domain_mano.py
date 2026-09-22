"""Deterministic non-full flow evaluation for the V1.4.4 two-domain MANO run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .model import ObjectInteractionCmv2V13Model
from .multi_domain import ThreeDomainTransitions, collate_three_domain, sha256_file
from .train_multi_domain import utc_now


WORK_VERSION = "V1.4.6"
ANGLE_EPSILON_M = 1e-6
GRAB_STRIDE = 1
ARCTIC_STRIDES = (5, 6, 7, 8, 9, 10)


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _stable_choice(total: int, count: int, *, seed: int, domain: str, stride: int) -> list[int]:
    if count <= 0:
        raise ValueError("sample count must be positive")
    if total < count:
        raise ValueError(f"{domain}/stride={stride}: only {total} transitions, need {count}")
    digest = hashlib.sha256(f"{seed}:{domain}:{stride}:v1_4_6".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return np.sort(rng.choice(total, size=count, replace=False)).astype(np.int64).tolist()


@dataclass
class FlowMetrics:
    angle_epsilon_m: float = ANGLE_EPSILON_M
    samples: int = 0
    points: int = 0
    epe_sum_m: float = 0.0
    pred_norm_sum_m: float = 0.0
    gt_norm_sum_m: float = 0.0
    angle_sum_deg: float = 0.0
    angle_valid_points: int = 0
    gt_static_points: int = 0
    prediction_static_points: int = 0
    both_static_points: int = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
            raise ValueError(f"invalid flow shape: pred={tuple(prediction.shape)} target={tuple(target.shape)}")
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise ValueError("non-finite predicted or target flow")
        prediction, target = prediction.detach(), target.detach()
        pred_norm = torch.linalg.vector_norm(prediction, dim=-1)
        gt_norm = torch.linalg.vector_norm(target, dim=-1)
        epe = torch.linalg.vector_norm(prediction - target, dim=-1)
        valid = (pred_norm >= self.angle_epsilon_m) & (gt_norm >= self.angle_epsilon_m)
        self.samples += int(prediction.shape[0])
        self.points += int(epe.numel())
        self.epe_sum_m += float(epe.sum().cpu())
        self.pred_norm_sum_m += float(pred_norm.sum().cpu())
        self.gt_norm_sum_m += float(gt_norm.sum().cpu())
        self.gt_static_points += int((gt_norm < self.angle_epsilon_m).sum().cpu())
        self.prediction_static_points += int((pred_norm < self.angle_epsilon_m).sum().cpu())
        self.both_static_points += int(((gt_norm < self.angle_epsilon_m) & (pred_norm < self.angle_epsilon_m)).sum().cpu())
        if valid.any():
            cosine = (prediction * target).sum(-1) / (pred_norm * gt_norm).clamp_min(self.angle_epsilon_m ** 2)
            angle = torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))
            self.angle_sum_deg += float(angle[valid].sum().cpu())
            self.angle_valid_points += int(valid.sum().cpu())

    def merge(self, other: "FlowMetrics") -> None:
        if self.angle_epsilon_m != other.angle_epsilon_m:
            raise ValueError("cannot merge metrics with different static thresholds")
        for key in ("samples", "points", "epe_sum_m", "pred_norm_sum_m", "gt_norm_sum_m", "angle_sum_deg",
                    "angle_valid_points", "gt_static_points", "prediction_static_points", "both_static_points"):
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def summary(self) -> dict[str, Any]:
        if not self.points:
            raise ValueError("no evaluated object points")
        return {
            "samples": self.samples,
            "object_points": self.points,
            "flow_epe_micro_mm": 1000.0 * self.epe_sum_m / self.points,
            "prediction_flow_magnitude_micro_mm": 1000.0 * self.pred_norm_sum_m / self.points,
            "gt_flow_magnitude_micro_mm": 1000.0 * self.gt_norm_sum_m / self.points,
            "flow_angle_mean_deg": self.angle_sum_deg / self.angle_valid_points if self.angle_valid_points else None,
            "angle_epsilon_m": self.angle_epsilon_m,
            "angle_valid_points": self.angle_valid_points,
            "angle_excluded_points": self.points - self.angle_valid_points,
            "gt_static_points": self.gt_static_points,
            "prediction_static_points": self.prediction_static_points,
            "both_static_points": self.both_static_points,
        }


def _dataset(training_cfg: Mapping[str, Any], domain: str, stride: int, seed: int) -> ThreeDomainTransitions:
    source = next((dict(item) for item in training_cfg["sources"] if item["name"] == domain), None)
    if source is None:
        raise ValueError(f"missing {domain} source in training config")
    return ThreeDomainTransitions(
        [source], "val", num_obj_points=int(training_cfg["data"]["num_obj_points"]),
        train_stride_values={domain: [int(stride)]}, fixed_stride=int(stride),
        active_only=bool(training_cfg["data"]["active_only"]), base_seed=int(seed),
        allow_manifest_split_override=True,
    )


def _sample_records(raw: Mapping[str, Any], domain: str, stride: int) -> list[dict[str, Any]]:
    records = []
    for index, sequence_id in enumerate(raw["sequence_id"]):
        records.append({
            "domain": domain,
            "stride": int(stride),
            "sequence_id": str(sequence_id),
            "source_frame_id": int(raw["source_frame_id"][index]),
            "next_source_frame_id": int(raw["next_source_frame_id"][index]),
            "delta_time_s": float(raw["delta_time_s"][index]),
        })
    return records


@torch.inference_mode()
def _evaluate_group(model: ObjectInteractionCmv2V13Model, dataset: ThreeDomainTransitions, indices: list[int],
                    *, domain: str, stride: int, device: torch.device, batch_size: int,
                    angle_epsilon_m: float) -> tuple[FlowMetrics, list[dict[str, Any]]]:
    loader = DataLoader(Subset(dataset, indices), batch_size=int(batch_size), shuffle=False,
                        num_workers=0, collate_fn=collate_three_domain)
    metrics = FlowMetrics(angle_epsilon_m=angle_epsilon_m)
    records: list[dict[str, Any]] = []
    model.eval()
    for raw in loader:
        records.extend(_sample_records(raw, domain, stride))
        batch = _move(raw, device)
        metrics.update(model(batch)["obj_flow_pred"], batch["obj_flow_gt"])
    if metrics.samples != len(indices) or len(records) != len(indices):
        raise RuntimeError("evaluation sample accounting mismatch")
    return metrics, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    training_config_path = args.training_config.resolve()
    if not checkpoint.is_file() or not training_config_path.is_file():
        raise FileNotFoundError("checkpoint and training config must exist")
    training_cfg = json.loads(training_config_path.read_text(encoding="utf-8"))
    legacy_work_version_key = "modification" + "_version"
    training_work_version = training_cfg.get("work_version", training_cfg.get(legacy_work_version_key))
    if training_work_version != "V1.4.4":
        raise ValueError("evaluation only accepts the frozen V1.4.4 two-domain training config")
    if int(training_cfg["data"]["num_obj_points"]) != 1024:
        raise ValueError("evaluation requires the V1.4.4 1024-point object contract")
    output = args.output_root.resolve() / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    manifest: dict[str, Any] = {
        "schema_name": "ref2dex_run_manifest_v1", "task": "ObjectInteractionCmv2",
        "operation": "v1_4_6_two_domain_mano_flow_evaluation", "run_id": args.run_id,
        "run_status": "STARTED", "created_at": utc_now(), "work_version": WORK_VERSION,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "training_config": str(training_config_path), "training_config_sha256": sha256_file(training_config_path),
        "coordinate_frame": "object_pose_t", "device": str(device),
        "sampling": {"seed": int(args.seed), "grab_stride": GRAB_STRIDE, "grab_samples": int(args.grab_samples),
                     "arctic_strides": list(ARCTIC_STRIDES), "arctic_samples_per_stride": int(args.arctic_samples_per_stride)},
        "metrics": {"flow_epe": "point-micro mm", "prediction_flow_magnitude": "point-micro mm",
                    "gt_flow_magnitude": "point-micro mm", "flow_angle": "degree, excluding either norm < epsilon",
                    "angle_epsilon_m": float(args.angle_epsilon_m)},
        "outputs": {"config": "config.json", "sample_index": "sample_index.json", "metrics": "metrics.jsonl",
                    "summary": "metrics_summary.json", "log": "eval.log"}, "conclusion": "INCONCLUSIVE",
    }
    evaluation_config = {"schema_name": "object_interaction_cmv2_two_domain_mano_flow_eval_v1", **manifest["sampling"],
                         "checkpoint": str(checkpoint), "training_config": str(training_config_path), "device": str(device),
                         "batch_size": int(args.batch_size), "minimum_free_memory_gib": int(args.minimum_free_memory_gib),
                         "angle_epsilon_m": float(args.angle_epsilon_m)}
    _write_json(output / "config.json", evaluation_config)
    _write_json(output / "run_manifest.json", manifest)
    try:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"CUDA evaluation required, got {device}")
        torch.cuda.set_device(device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        minimum = int(args.minimum_free_memory_gib) * 2**30
        if free_bytes < minimum:
            raise RuntimeError(f"{device} free memory {free_bytes / 2**30:.1f} GiB below {args.minimum_free_memory_gib} GiB")
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = ObjectInteractionCmv2V13Model(SimpleNamespace(**training_cfg["model"])).to(device)
        if payload.get("architecture_version") != model.architecture_version:
            raise ValueError("checkpoint architecture mismatch")
        model.load_state_dict(payload["model"], strict=True)
        groups = [("grab", GRAB_STRIDE, int(args.grab_samples))] + [
            ("arctic", stride, int(args.arctic_samples_per_stride)) for stride in ARCTIC_STRIDES]
        summary: dict[str, Any] = {"run_id": args.run_id, "checkpoint": str(checkpoint), "groups": {}}
        samples: list[dict[str, Any]] = []
        arctic_total, all_total = FlowMetrics(float(args.angle_epsilon_m)), FlowMetrics(float(args.angle_epsilon_m))
        manifest.update({"run_status": "RUNNING", "started_at": utc_now()})
        _write_json(output / "run_manifest.json", manifest)
        for domain, stride, count in groups:
            dataset = _dataset(training_cfg, domain, stride, int(args.seed))
            indices = _stable_choice(len(dataset), count, seed=int(args.seed), domain=domain, stride=stride)
            values, records = _evaluate_group(model, dataset, indices, domain=domain, stride=stride, device=device,
                                               batch_size=int(args.batch_size), angle_epsilon_m=float(args.angle_epsilon_m))
            key = f"{domain}_stride_{stride}"
            summary["groups"][key] = {"available_transitions": len(dataset), **values.summary()}
            samples.extend(records); all_total.merge(values)
            if domain == "arctic":
                arctic_total.merge(values)
        summary["arctic_pooled"] = arctic_total.summary()
        summary["all_pooled"] = all_total.summary()
        _write_json(output / "sample_index.json", {"run_id": args.run_id, "seed": int(args.seed), "samples": samples})
        (output / "metrics.jsonl").write_text(json.dumps(summary, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_json(output / "metrics_summary.json", summary)
        (output / "eval.log").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.update({"run_status": "COMPLETED", "finished_at": utc_now(), "evaluated_transitions": len(samples),
                         "best_metric": None, "last_step": None, "last_epoch": None, "conclusion": "INCONCLUSIVE"})
        _write_json(output / "run_manifest.json", manifest)
        return summary
    except Exception as error:
        manifest.update({"run_status": "FAILED", "finished_at": utc_now(), "error": repr(error),
                         "conclusion": "INVALID_IMPLEMENTATION"})
        _write_json(output / "run_manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grab-samples", type=int, default=12000)
    parser.add_argument("--arctic-samples-per-stride", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--minimum-free-memory-gib", type=int, default=20)
    parser.add_argument("--angle-epsilon-m", type=float, default=ANGLE_EPSILON_M)
    args = parser.parse_args()
    if args.angle_epsilon_m <= 0 or args.batch_size <= 0:
        raise ValueError("angle epsilon and batch size must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
