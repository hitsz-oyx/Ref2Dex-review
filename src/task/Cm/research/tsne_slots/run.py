"""Shared t-SNE diagnostics for GRAB/Inspire Cm representations.

The main fit combines held-out transitions from both datasets. GRAB uses
stride 1--10, while Inspire-F1's underlying data stride is 2, so its effective
stride labels are the even values 2--20. Sixteen slot tokens are mean-pooled to one sample vector; one shared
t-SNE coordinate system is recolored by dataset, stride, hand-flow RMS and
object-flow RMS. Fixed-stride=5, matched hand-motion, and per-slot views are
also exported.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset, Subset

from src.base import build_runner_from_checkpoint
from src.base.run_manifest import build_run_manifest, write_run_manifest
from src.task.Cm.dataset.hrdexdb import HrdexdbGeometryDataset, _manifest_specs
from src.task.Cm.dataset.object_v2 import _MmapSequenceDataset, _read_sequence_split

DEFAULT_CHECKPOINT = (
    "outputs/cm/cm_hrdexdb_inspire_f1_decoder_only_resume_20260827_011451/"
    "checkpoints/latest.pt"
)
DEFAULT_GRAB_ROOT = "data/processed_data/cm_object_v2_subject_template_20260820/grab"
DEFAULT_GRAB_SPLIT = DEFAULT_GRAB_ROOT + "/splits_seed42/test.txt"
DEFAULT_HRDEXDB_ROOT = "data/processed_data/cm_decoder/hrdexdb_all_v1/v4"
DEFAULT_HRDEXDB_MANIFEST = DEFAULT_HRDEXDB_ROOT + "/selection_all_object_disjoint_seed42.json"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "output"
SOURCE_STRIDES = {
    "GRAB": tuple(range(1, 11)),
    "Inspire-F1": tuple(range(2, 21, 2)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--grab-test-split", default=DEFAULT_GRAB_SPLIT)
    parser.add_argument("--hrdexdb-root", default=DEFAULT_HRDEXDB_ROOT)
    parser.add_argument("--hrdexdb-manifest", default=DEFAULT_HRDEXDB_MANIFEST)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory. Defaults to this experiment's output/<run_id> directory.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=512, help="Maximum samples per source across strides 1..10; 0 means all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--matched-hand-min-mm", type=float, default=4.0)
    parser.add_argument("--matched-hand-max-mm", type=float, default=6.0)
    parser.add_argument("--matched-bin-width-mm", type=float, default=2.0)
    parser.add_argument("--matched-bin-max-mm", type=float, default=42.0)
    parser.add_argument("--coordinate-frame", default=None, choices=("hand_root_t", "object_pose_t"))
    parser.add_argument("--grab-frame-activity-mask-name", default=None)
    parser.add_argument("--hrdexdb-candidate-mask-name", default="obj_candidate_mask_5cm.npy")
    return parser.parse_args()


def _make_dataset(source: str, args: argparse.Namespace, *, num_obj_points: int, num_hand_points: int, seed: int, fixed_stride: int | None, stride_values: Sequence[int] | None = None) -> tuple[Dataset, int]:
    coordinate_frame = getattr(args, "coordinate_frame", None)
    if source == "GRAB":
        root, split = Path(args.grab_root).resolve(), Path(args.grab_test_split).resolve()
        sequences = _read_sequence_split(split, root)
        return _MmapSequenceDataset(
            sequences, num_obj_points=num_obj_points, num_hand_points=num_hand_points,
            base_seed=seed, min_stride=1, max_stride=10, fixed_stride=fixed_stride,
            active_only=True, sampling_bank_size=4, fixed_eval_bank=0,
            coordinate_frame=coordinate_frame,
            frame_activity_mask_name=getattr(args, "grab_frame_activity_mask_name", None),
        ), len(sequences)
    if source == "Inspire-F1":
        root, manifest = Path(args.hrdexdb_root).resolve(), Path(args.hrdexdb_manifest).resolve()
        episodes = _manifest_specs(root, manifest, "test", "inspire_f1")
        return HrdexdbGeometryDataset(
            episodes, num_obj_points=num_obj_points, num_obj_pool=4096,
            num_hand_points=num_hand_points, base_seed=seed, min_stride=1,
            max_stride=max(SOURCE_STRIDES[source]), stride_values=stride_values,
            fixed_stride=fixed_stride, active_only=True,
            coordinate_frame=coordinate_frame,
            candidate_mask_name=getattr(args, "hrdexdb_candidate_mask_name", "obj_candidate_mask_5cm.npy"),
        ), len(episodes)
    raise ValueError(f"Unsupported source: {source}")


def _balanced_subset(dataset: Dataset, count: int, seed: int) -> Subset:
    if count <= 0 or len(dataset) <= count:
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.sort(np.random.default_rng(seed).choice(len(dataset), size=count, replace=False))
    return Subset(dataset, indices.tolist())


def _iter_records(loader: DataLoader, runner) -> Iterable[dict[str, np.ndarray]]:
    for batch in loader:
        with torch.inference_mode():
            output = runner.inference(runner.model, batch)
        tokens = output["cm_tokens"].detach().cpu()
        if tokens.ndim != 3:
            raise ValueError(f"Expected cm_tokens [B,K,C], got {tuple(tokens.shape)}")
        hand_flow = batch["hand_flow"].float()
        hand_rms = torch.linalg.vector_norm(hand_flow, dim=-1).square().mean(dim=-1).sqrt() * 1000.0
        obj_flow, valid = batch["obj_flow_gt"].float(), batch["obj_valid_mask"].bool()
        obj_sq = obj_flow.square().sum(dim=-1).masked_fill(~valid, 0.0)
        obj_rms = (obj_sq.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).sqrt() * 1000.0
        yield {
            "tokens": tokens.numpy().astype(np.float32, copy=False),
            "stride": batch["stride"].detach().cpu().numpy().astype(np.int64, copy=False),
            "hand_flow_rms_mm": hand_rms.numpy().astype(np.float32, copy=False),
            "obj_flow_rms_mm": obj_rms.numpy().astype(np.float32, copy=False),
        }


def _collect_records(runner, dataset: Dataset, *, count: int, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    loader = DataLoader(_balanced_subset(dataset, count, seed), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    chunks = list(_iter_records(loader, runner))
    if not chunks:
        raise ValueError("Dataset produced no samples")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


def _concat_records(records: Sequence[dict[str, np.ndarray]], labels: Sequence[int]) -> dict[str, np.ndarray]:
    merged = {key: np.concatenate([record[key] for record in records], axis=0) for key in records[0]}
    merged["labels"] = np.concatenate([np.full(record["tokens"].shape[0], label, dtype=np.int64) for record, label in zip(records, labels)], axis=0)
    return merged


def _run_tsne(vectors: np.ndarray, *, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    if vectors.shape[0] < 4:
        raise ValueError(f"Need at least 4 samples for t-SNE, got {vectors.shape[0]}")
    effective = min(float(perplexity), float(max(2, (vectors.shape[0] - 1) // 3)))
    return TSNE(n_components=2, perplexity=effective, init="pca", learning_rate="auto", n_iter=int(n_iter), random_state=int(seed), metric="euclidean").fit_transform(vectors).astype(np.float32, copy=False)


def _safe_silhouette(vectors: np.ndarray, labels: np.ndarray) -> float | None:
    return None if len(np.unique(labels)) < 2 or len(vectors) <= 3 else float(silhouette_score(vectors, labels, metric="euclidean"))


def _finish_axes(axes) -> None:
    for axis in np.asarray(axes).reshape(-1):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)


def _plot_shared_panels(embedding: np.ndarray, records: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    labels = records["labels"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), squeeze=False)
    names, colors = {0: "GRAB", 1: "Inspire-F1"}, {0: "#2878b5", 1: "#d95f02"}
    for label in (0, 1):
        mask = labels == label
        axes[0, 0].scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.65, c=colors[label], label=names[label], linewidths=0, rasterized=True)
    axes[0, 0].set_title("Dataset")
    axes[0, 0].legend(frameon=False)
    cmap = plt.get_cmap("tab10")
    for stride in sorted(np.unique(records["stride"])):
        mask = records["stride"] == stride
        axes[0, 1].scatter(embedding[mask, 0], embedding[mask, 1], s=9, alpha=0.6, c=[cmap((int(stride) - 1) % 10)], label=f"s={int(stride)}", linewidths=0, rasterized=True)
    axes[0, 1].set_title("Stride")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)
    for axis, key, label in ((axes[1, 0], "hand_flow_rms_mm", "hand-flow RMS (mm)"), (axes[1, 1], "obj_flow_rms_mm", "object-flow RMS (mm)")):
        scatter = axis.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.7, c=records[key], cmap="viridis", linewidths=0, rasterized=True)
        axis.set_title(label)
        fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label="mm")
    _finish_axes(axes)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dataset_and_magnitude(embedding: np.ndarray, records: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    labels = records["labels"]
    names, colors = {0: "GRAB", 1: "Inspire-F1"}, {0: "#2878b5", 1: "#d95f02"}
    for label in (0, 1):
        mask = labels == label
        axes[0, 0].scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.7, c=colors[label], label=names[label], linewidths=0, rasterized=True)
    axes[0, 0].set_title("Dataset (matched hand-flow magnitude)")
    axes[0, 0].legend(frameon=False)
    scatter = axes[0, 1].scatter(embedding[:, 0], embedding[:, 1], s=12, alpha=0.7, c=records["hand_flow_rms_mm"], cmap="viridis", linewidths=0, rasterized=True)
    axes[0, 1].set_title("Matched hand-flow RMS")
    fig.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.04, label="mm")
    _finish_axes(axes)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dataset_only(embedding: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(9, 7))
    names, colors = {0: "GRAB", 1: "Inspire-F1"}, {0: "#2878b5", 1: "#d95f02"}
    for label in (0, 1):
        mask = labels == label
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=11, alpha=0.68, c=colors[label], label=names[label], linewidths=0, rasterized=True)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_value_only(embedding: np.ndarray, values: np.ndarray, output_path: Path, title: str, colorbar_label: str) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(9, 7))
    scatter = axis.scatter(embedding[:, 0], embedding[:, 1], s=11, alpha=0.70, c=values, cmap="viridis", linewidths=0, rasterized=True)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _bin_match_records(records: dict[str, np.ndarray], tokens: np.ndarray, *, width_mm: float, max_mm: float, seed: int) -> tuple[dict[str, np.ndarray] | None, int, list[dict[str, int]]]:
    """Equalize the two datasets independently inside hand-RMS bins."""
    if width_mm <= 0 or max_mm <= width_mm:
        raise ValueError("Invalid matched hand-flow bin range")
    values = records["hand_flow_rms_mm"]
    candidates = []
    bin_counts = []
    for lower in np.arange(0.0, max_mm, width_mm):
        upper = lower + width_mm
        per_label = [np.flatnonzero((records["labels"] == label) & (values >= lower) & (values < upper)) for label in (0, 1)]
        count = min(len(per_label[0]), len(per_label[1]))
        bin_counts.append({"lower_mm": int(lower), "upper_mm": int(upper), "samples_per_source": int(count)})
        if count:
            candidates.append((per_label, count))
    rng = np.random.default_rng(seed)
    chosen = []
    for per_label, count in candidates:
        chosen.extend(np.sort(rng.choice(per_label[0], count, replace=False)))
        chosen.extend(np.sort(rng.choice(per_label[1], count, replace=False)))
    if not chosen:
        return None, 0, bin_counts
    indices = np.asarray(chosen, dtype=np.int64)
    matched = {key: value[indices] for key, value in records.items()}
    matched["tokens"] = tokens[indices]
    return matched, int(min(np.sum(matched["labels"] == 0), np.sum(matched["labels"] == 1))), bin_counts


def _plot_slot_tsne(embedding: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    slots, rows = embedding.shape[1], int(np.ceil(embedding.shape[1] / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), squeeze=False)
    names, colors = {0: "GRAB", 1: "Inspire-F1"}, {0: "#2878b5", 1: "#d95f02"}
    for slot in range(slots):
        axis = axes[slot // 4, slot % 4]
        for label in (0, 1):
            mask = labels == label
            axis.scatter(embedding[mask, slot, 0], embedding[mask, slot, 1], s=8, alpha=0.62, c=colors[label], label=names[label], linewidths=0, rasterized=True)
        axis.set_title(f"Cm slot {slot}")
    for slot in range(slots, rows * 4):
        axes[slot // 4, slot % 4].axis("off")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    _finish_axes(axes)
    fig.suptitle(title, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _fixed_stride_records(runner, args, *, num_obj_points: int, num_hand_points: int) -> dict[str, np.ndarray]:
    records, labels = [], []
    for source, label, salt in (("GRAB", 0, 0x47524142), ("Inspire-F1", 1, 0x494E5350)):
        # Inspire-F1 has no physical stride=5 under its stride-2 data
        # convention; use the nearest central even stride as an auxiliary
        # source-specific control and record that distinction in metadata.
        fixed_stride = 5 if source == "GRAB" else 10
        dataset, _ = _make_dataset(source, args, num_obj_points=num_obj_points, num_hand_points=num_hand_points, seed=args.seed ^ salt, fixed_stride=fixed_stride, stride_values=SOURCE_STRIDES[source])
        count = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
        records.append(_collect_records(runner, dataset, count=count, batch_size=args.batch_size, seed=args.seed ^ salt ^ 0x5005))
        labels.append(label)
    return _concat_records(records, labels)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_samples < 0:
        raise ValueError("--batch-size must be positive and --max-samples non-negative")
    if args.n_iter < 250:
        raise ValueError("--n-iter must be at least 250 for the installed scikit-learn")
    if args.matched_hand_min_mm < 0 or args.matched_hand_max_mm <= args.matched_hand_min_mm:
        raise ValueError("Invalid matched hand-flow magnitude range")
    if args.matched_bin_width_mm <= 0 or args.matched_bin_max_mm <= args.matched_bin_width_mm:
        raise ValueError("Invalid matched hand-flow bin range")
    if args.output_root is None:
        run_id = "tsne_slots_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_OUTPUT_ROOT / run_id
    else:
        output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Checkpoints created before the Cm directory migration still carry the
    # old import paths.  Keep their hyperparameters intact while resolving
    # those two moved classes to the current modules.
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint_payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"Checkpoint has no dictionary config: {args.checkpoint}")
    checkpoint_config = json.loads(json.dumps(checkpoint_config))
    if checkpoint_config.get("runner_class") == "src.task.Cm.runner.CmActionRunner":
        checkpoint_config["runner_class"] = "src.task.Cm.src.runner.CmActionRunner"
    model_config = checkpoint_config.get("model")
    if isinstance(model_config, dict) and model_config.get("class_path") == "src.task.Cm.model.CmFlowModel":
        model_config["class_path"] = "src.task.Cm.src.model.CmFlowModel"
    runner = build_runner_from_checkpoint(args.checkpoint, config=checkpoint_config, mode="eval", device=args.device, build_data=False)
    checkpoint_data = runner.setup_inference(args.checkpoint)
    num_obj_points, num_hand_points = int(runner.cfg.meta.num_obj_points), int(runner.cfg.meta.num_hand_points)
    if args.coordinate_frame is None:
        args.coordinate_frame = str(runner.cfg.meta.coordinate_frame)
    checkpoint_data_cfg = checkpoint_config.get("data", {})
    if args.grab_root == DEFAULT_GRAB_ROOT and checkpoint_data_cfg.get("base_root"):
        args.grab_root = str(checkpoint_data_cfg["base_root"])
        args.grab_test_split = str(Path(args.grab_root) / "splits_seed42" / "test.txt")
    if args.hrdexdb_root == DEFAULT_HRDEXDB_ROOT and checkpoint_data_cfg.get("hrdexdb_root"):
        args.hrdexdb_root = str(checkpoint_data_cfg["hrdexdb_root"])
    if args.hrdexdb_manifest == DEFAULT_HRDEXDB_MANIFEST and checkpoint_data_cfg.get("hrdexdb_manifest"):
        args.hrdexdb_manifest = str(checkpoint_data_cfg["hrdexdb_manifest"])
    if args.grab_frame_activity_mask_name is None:
        args.grab_frame_activity_mask_name = checkpoint_data_cfg.get("base_frame_activity_mask_name")
    if args.hrdexdb_candidate_mask_name == "obj_candidate_mask_5cm.npy" and checkpoint_data_cfg.get("hrdexdb_candidate_mask_name"):
        args.hrdexdb_candidate_mask_name = str(checkpoint_data_cfg["hrdexdb_candidate_mask_name"])

    all_records, all_labels, episode_counts = [], [], {}
    per_stride = {
        source: (0 if args.max_samples <= 0 else max(1, int(np.ceil(args.max_samples / len(SOURCE_STRIDES[source])))))
        for source in SOURCE_STRIDES
    }
    for source, label, salt in (("GRAB", 0, 0x47524142), ("Inspire-F1", 1, 0x494E5350)):
        source_strides = SOURCE_STRIDES[source]
        _, episode_counts[source] = _make_dataset(source, args, num_obj_points=num_obj_points, num_hand_points=num_hand_points, seed=args.seed ^ salt, fixed_stride=None, stride_values=source_strides)
        for stride in source_strides:
            dataset, _ = _make_dataset(source, args, num_obj_points=num_obj_points, num_hand_points=num_hand_points, seed=args.seed ^ salt ^ stride, fixed_stride=stride, stride_values=source_strides)
            count = len(dataset) if args.max_samples <= 0 else min(per_stride[source], len(dataset))
            if count:
                all_records.append(_collect_records(runner, dataset, count=count, batch_size=args.batch_size, seed=args.seed ^ salt ^ (stride * 0x9E37)))
                all_labels.append(label)
    records = _concat_records(all_records, all_labels)
    tokens = records.pop("tokens")
    if len(tokens) < 4 or len(np.unique(records["labels"])) < 2:
        raise ValueError("Combined GRAB/Inspire sample set is too small or has one dataset only")
    pooled = tokens.mean(axis=1)
    pooled_tsne = _run_tsne(pooled, perplexity=args.perplexity, n_iter=args.n_iter, seed=args.seed)
    slot_tsne = np.stack([_run_tsne(tokens[:, slot, :], perplexity=args.perplexity, n_iter=args.n_iter, seed=args.seed + slot + 1) for slot in range(tokens.shape[1])], axis=1)
    pooled_silhouette = _safe_silhouette(pooled, records["labels"])
    slot_silhouette = [_safe_silhouette(tokens[:, slot, :], records["labels"]) for slot in range(tokens.shape[1])]
    _plot_shared_panels(pooled_tsne, records, output_root / "pooled_tsne_panels.png", "Cm pooled t-SNE: GRAB 1--10 / Inspire-F1 even 2--20")
    _plot_slot_tsne(slot_tsne, records["labels"], output_root / "slot_tsne.png", "Cm slot-wise t-SNE: GRAB 1--10 / Inspire-F1 even 2--20")

    # Natural all-stride view: keep each dataset's deterministic random
    # stride distribution instead of enforcing equal counts per stride.
    natural_records, natural_labels = [], []
    for source, label, salt in (("GRAB", 0, 0x47524142), ("Inspire-F1", 1, 0x494E5350)):
        natural_dataset, _ = _make_dataset(source, args, num_obj_points=num_obj_points, num_hand_points=num_hand_points, seed=args.seed ^ salt, fixed_stride=None, stride_values=SOURCE_STRIDES[source])
        natural_count = len(natural_dataset) if args.max_samples <= 0 else min(args.max_samples, len(natural_dataset))
        natural_records.append(_collect_records(runner, natural_dataset, count=natural_count, batch_size=args.batch_size, seed=args.seed ^ salt ^ 0x4E415455))
        natural_labels.append(label)
    natural = _concat_records(natural_records, natural_labels)
    natural_tokens = natural.pop("tokens")
    natural_embedding = _run_tsne(natural_tokens.mean(axis=1), perplexity=args.perplexity, n_iter=args.n_iter, seed=args.seed ^ 0x4E4154)
    _plot_dataset_only(natural_embedding, natural["labels"], output_root / "all_stride_natural_tsne.png", "Cm pooled t-SNE: natural stride sampling (GRAB 1--10 / Inspire-F1 even 2--20), dataset")
    _plot_value_only(natural_embedding, natural["hand_flow_rms_mm"], output_root / "all_stride_natural_hand_magnitude_tsne.png", "Cm pooled t-SNE: natural stride sampling (GRAB 1--10 / Inspire-F1 even 2--20), hand-flow RMS", "hand-flow RMS (mm)")

    fixed = _fixed_stride_records(runner, args, num_obj_points=num_obj_points, num_hand_points=num_hand_points)
    fixed_tokens = fixed.pop("tokens")
    fixed_embedding = _run_tsne(fixed_tokens.mean(axis=1), perplexity=args.perplexity, n_iter=args.n_iter, seed=args.seed ^ 0x53545235)
    _plot_shared_panels(fixed_embedding, fixed, output_root / "fixed_stride5_tsne_panels.png", "Cm pooled t-SNE: source-specific control (GRAB s=5 / Inspire-F1 s=10)")

    # The bin-matched plot is the explicit fixed-stride control: equal counts
    # are selected independently in each 2 mm hand-RMS bin.
    matched_bins, matched_bins_count, matched_bin_counts = _bin_match_records(
        fixed, fixed_tokens, width_mm=args.matched_bin_width_mm, max_mm=args.matched_bin_max_mm, seed=args.seed ^ 0x4D42494E
    )
    if matched_bins is not None and matched_bins_count >= 2:
        matched_bins_tokens = matched_bins.pop("tokens")
        matched_bins_embedding = _run_tsne(matched_bins_tokens.mean(axis=1), perplexity=args.perplexity, n_iter=args.n_iter, seed=args.seed ^ 0x4D42494E)
        _plot_dataset_and_magnitude(matched_bins_embedding, matched_bins, output_root / "matched_hand_bins_stride5_tsne.png", f"Cm pooled t-SNE: source-specific control, hand-flow RMS matched in {args.matched_bin_width_mm:g} mm bins")
    else:
        matched_bins_count = 0
        (output_root / "matched_hand_bins_stride5_tsne.txt").write_text("Insufficient samples for bin-matched hand-flow plot.\n", encoding="utf-8")

    matched_mask = (records["hand_flow_rms_mm"] >= args.matched_hand_min_mm) & (records["hand_flow_rms_mm"] <= args.matched_hand_max_mm)
    candidates = [np.flatnonzero(matched_mask & (records["labels"] == label)) for label in (0, 1)]
    matched_count = min((len(values) for values in candidates), default=0)
    if matched_count >= 2:
        rng = np.random.default_rng(args.seed ^ 0x4D415443)
        chosen = np.concatenate([np.sort(rng.choice(values, matched_count, replace=False)) for values in candidates])
        matched = {key: value[chosen] for key, value in records.items()}
        matched_tokens = tokens[chosen]
        matched_embedding = _run_tsne(matched_tokens.mean(axis=1), perplexity=args.perplexity, n_iter=args.n_iter, seed=args.seed ^ 0x4D4154)
        _plot_dataset_and_magnitude(matched_embedding, matched, output_root / "matched_hand_magnitude_tsne.png", f"Cm pooled t-SNE: matched hand-flow RMS {args.matched_hand_min_mm:g}--{args.matched_hand_max_mm:g} mm")
    else:
        (output_root / "matched_hand_magnitude_tsne.txt").write_text("Insufficient samples in requested matched hand-flow range.\n", encoding="utf-8")

    np.savez_compressed(output_root / "tsne.npz", cm_tokens=tokens, pooled_cm=pooled, pooled_tsne=pooled_tsne, slot_tsne=slot_tsne, labels=records["labels"], strides=records["stride"], hand_flow_rms_mm=records["hand_flow_rms_mm"], obj_flow_rms_mm=records["obj_flow_rms_mm"], natural_all_stride_tokens=natural_tokens, natural_all_stride_tsne=natural_embedding, natural_all_stride_labels=natural["labels"], natural_all_stride_values=natural["stride"], natural_all_stride_hand_flow_rms_mm=natural["hand_flow_rms_mm"], natural_all_stride_obj_flow_rms_mm=natural["obj_flow_rms_mm"], fixed_stride5_tokens=fixed_tokens, fixed_stride5_tsne=fixed_embedding, fixed_stride5_labels=fixed["labels"], fixed_stride5_hand_flow_rms_mm=fixed["hand_flow_rms_mm"], fixed_stride5_obj_flow_rms_mm=fixed["obj_flow_rms_mm"])
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": checkpoint_data.get("step"), "checkpoint_epoch": checkpoint_data.get("epoch"),
        "split": "test", "sources": ["GRAB", "Inspire-F1"], "source_strides": {key: list(value) for key, value in SOURCE_STRIDES.items()}, "sample_level_pooling": "mean over 16 Cm slots",
        "samples_per_source": {name: int(np.sum(records["labels"] == label)) for name, label in (("GRAB", 0), ("Inspire-F1", 1))},
        "samples_per_source_stride": {name: {str(stride): int(np.sum((records["labels"] == label) & (records["stride"] == stride))) for stride in SOURCE_STRIDES[name]} for name, label in (("GRAB", 0), ("Inspire-F1", 1))},
        "episode_counts": episode_counts, "num_slots": int(tokens.shape[1]), "cm_dim": int(tokens.shape[2]), "perplexity_requested": float(args.perplexity), "n_iter": int(args.n_iter), "seed": int(args.seed), "coordinate_frame": str(runner.cfg.meta.coordinate_frame),
        "pooled_silhouette_original_space": pooled_silhouette, "slot_silhouette_original_space": slot_silhouette,
        "matched_hand_flow_range_mm": [float(args.matched_hand_min_mm), float(args.matched_hand_max_mm)], "matched_samples_per_source": int(matched_count if matched_count >= 2 else 0),
        "matched_bin_width_mm": float(args.matched_bin_width_mm), "matched_bin_max_mm": float(args.matched_bin_max_mm), "matched_bins_samples_per_source": int(matched_bins_count), "matched_bin_counts": matched_bin_counts,
        "fixed_control_strides": {"GRAB": 5, "Inspire-F1": 10},
        "note": "All four pooled panels share one t-SNE fit; Inspire-F1 stride labels are even 2..20 because source data stride is 2. Flow magnitudes use valid object points and all hand points. t-SNE is exploratory.",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    experiment_config = {
        "experiment_id": "tsne_slots",
        "guide_version": "V1.2.1",
        "plan_version": None,
        "inputs": {
            "checkpoint": str(args.checkpoint),
            "grab_root": str(args.grab_root),
            "grab_test_split": str(args.grab_test_split),
            "hrdexdb_root": str(args.hrdexdb_root),
            "hrdexdb_manifest": str(args.hrdexdb_manifest),
        },
        "arguments": vars(args),
    }
    (output_root / "config.json").write_text(
        json.dumps(experiment_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_manifest = build_run_manifest(
        task="cm",
        run_name=output_root.name,
        output_dir=output_root,
        mode="diagnostic",
        config=experiment_config,
        metadata=metadata,
        config_source=Path(__file__).with_name("experiment.yaml"),
        initial_checkpoint=args.checkpoint,
        config_snapshot=output_root / "config.json",
        metadata_snapshot=output_root / "metadata.json",
        repo_root=Path(__file__).resolve().parents[5],
    )
    write_run_manifest(output_root / "run_manifest.json", run_manifest)
    print(f"[cm-tsne] wrote {output_root / 'pooled_tsne_panels.png'}")
    print(f"[cm-tsne] wrote {output_root / 'slot_tsne.png'}")
    print(f"[cm-tsne] wrote {output_root / 'fixed_stride5_tsne_panels.png'}")
    print(f"[cm-tsne] wrote {output_root / 'all_stride_natural_hand_magnitude_tsne.png'}")
    print(f"[cm-tsne] matched-bin samples_per_source={matched_bins_count}")
    print(f"[cm-tsne] samples={len(tokens)} pooled_silhouette={pooled_silhouette}")
    print(f"[cm-tsne] matched_samples_per_source={matched_count}")


if __name__ == "__main__":
    main()
