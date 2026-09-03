"""t-SNE diagnostics for an ObjectInteractionCm checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Subset

from src.base import build_runner_from_checkpoint
from src.task.ObjectInteractionCm.dataset import ObjectInteractionCmDataset, _resolve_index_entries


SOURCE_STRIDES = {"grab": tuple(range(1, 11)), "inspire_f1": tuple(range(2, 21, 2))}
SOURCE_LABELS = {"grab": 0, "inspire_f1": 1}
SOURCE_NAMES = {0: "GRAB", 1: "Inspire-F1"}
SOURCE_COLORS = {0: "#2878b5", 1: "#d95f02"}


def _tsne(x: np.ndarray, *, seed: int, n_iter: int, perplexity: float) -> np.ndarray:
    p = min(float(perplexity), float(max(2, (len(x) - 1) // 3)))
    return TSNE(n_components=2, perplexity=p, init="pca", learning_rate="auto", n_iter=n_iter, random_state=seed).fit_transform(x).astype(np.float32)


def _collect(runner, dataset, *, count: int, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    if count < len(dataset):
        indices = np.sort(np.random.default_rng(seed).choice(len(dataset), count, replace=False)).tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for batch in loader:
        with torch.inference_mode():
            prediction = runner.inference(runner.model, batch)
        hand_flow = batch["hand_flow"].float()
        hand_valid = batch["hand_valid_mask"].bool()
        hand_sq = hand_flow.square().sum(dim=-1).masked_fill(~hand_valid, 0.0)
        hand_rms = (hand_sq.sum(dim=-1) / hand_valid.sum(dim=-1).clamp_min(1)).sqrt() * 1000.0
        obj_flow = batch["obj_flow_gt"].float()
        obj_rms = obj_flow.square().sum(dim=-1).mean(dim=-1).sqrt() * 1000.0
        chunks.append({
            "tokens": prediction["cm_tokens"].detach().cpu().numpy().astype(np.float32),
            "stride": batch["stride"].cpu().numpy().astype(np.int64),
            "hand_flow_rms_mm": hand_rms.cpu().numpy().astype(np.float32),
            "obj_flow_rms_mm": obj_rms.cpu().numpy().astype(np.float32),
        })
    return {key: np.concatenate([item[key] for item in chunks]) for key in chunks[0]}


def _merge(parts, labels):
    result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    result["labels"] = np.concatenate([np.full(len(part["tokens"]), label, dtype=np.int64) for part, label in zip(parts, labels)])
    return result


def _finish(axis):
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_panels(embedding, records, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), squeeze=False)
    for label in (0, 1):
        mask = records["labels"] == label
        axes[0, 0].scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=.65, c=SOURCE_COLORS[label], label=SOURCE_NAMES[label], linewidths=0, rasterized=True)
    axes[0, 0].set_title("Dataset")
    axes[0, 0].legend(frameon=False)
    cmap = plt.get_cmap("tab10")
    for stride in sorted(np.unique(records["stride"])):
        mask = records["stride"] == stride
        axes[0, 1].scatter(embedding[mask, 0], embedding[mask, 1], s=9, alpha=.6, c=[cmap((int(stride) - 1) % 10)], label=f"s={int(stride)}", linewidths=0, rasterized=True)
    axes[0, 1].set_title("Stride")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)
    for axis, key, title_text in ((axes[1, 0], "hand_flow_rms_mm", "hand-flow RMS (mm)"), (axes[1, 1], "obj_flow_rms_mm", "object-flow RMS (mm)")):
        points = axis.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=.7, c=records[key], cmap="viridis", linewidths=0, rasterized=True)
        axis.set_title(title_text)
        fig.colorbar(points, ax=axis, fraction=.046, pad=.04, label="mm")
    for axis in axes.reshape(-1):
        _finish(axis)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_simple(embedding, records, path, title, *, magnitude=False):
    fig, axis = plt.subplots(figsize=(9, 7))
    if magnitude:
        points = axis.scatter(embedding[:, 0], embedding[:, 1], s=11, alpha=.7, c=records["hand_flow_rms_mm"], cmap="viridis", linewidths=0, rasterized=True)
        fig.colorbar(points, ax=axis, label="hand-flow RMS (mm)")
    else:
        for label in (0, 1):
            mask = records["labels"] == label
            axis.scatter(embedding[mask, 0], embedding[mask, 1], s=11, alpha=.68, c=SOURCE_COLORS[label], label=SOURCE_NAMES[label], linewidths=0, rasterized=True)
        axis.legend(frameon=False)
    axis.set_title(title)
    _finish(axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no dictionary config")
    runner = build_runner_from_checkpoint(args.checkpoint, config=config, mode="eval", device=args.device, build_data=False)
    checkpoint_data = runner.setup_inference(args.checkpoint)
    data_cfg, meta_cfg = config["data"], config["meta"]
    index_path = Path(data_cfg["index_path"])
    if not index_path.is_absolute():
        index_path = (Path.cwd() / index_path).resolve()
    test_entries = _resolve_index_entries(index_path, "test")
    parts, labels, counts = [], [], {}
    source_count = 0 if args.max_samples <= 0 else max(1, int(np.ceil(args.max_samples / 10)))
    for source in ("grab", "inspire_f1"):
        entries = [item for item in test_entries if item["source"] == source]
        counts[source] = len(entries)
        source_stride_count = 0 if args.max_samples <= 0 else max(1, int(np.ceil(args.max_samples / len(SOURCE_STRIDES[source]))))
        for stride in SOURCE_STRIDES[source]:
            dataset = ObjectInteractionCmDataset(
                Path("."), sequence_entries=entries, num_obj_points=int(meta_cfg["num_obj_points"]),
                num_hand_points=int(meta_cfg["num_hand_points"]), max_hand_points=int(meta_cfg["max_hand_points"]),
                min_stride=1, max_stride=max(SOURCE_STRIDES[source]), fixed_stride=stride,
                active_only=bool(data_cfg.get("active_only", True)),
                source_stride_values={source: SOURCE_STRIDES[source]},
            )
            count = len(dataset) if args.max_samples <= 0 else min(source_stride_count, len(dataset))
            if count:
                part = _collect(runner, dataset, count=count, batch_size=args.batch_size, seed=args.seed ^ stride ^ (0x47524142 if source == "grab" else 0x494E5350))
                parts.append(part)
                labels.append(SOURCE_LABELS[source])
    records = _merge(parts, labels)
    tokens = records.pop("tokens")
    pooled = tokens.mean(axis=1)
    embedding = _tsne(pooled, seed=args.seed, n_iter=args.n_iter, perplexity=args.perplexity)
    _plot_panels(embedding, records, output / "pooled_tsne_panels.png", "ObjectInteractionCm pooled t-SNE: GRAB 1--10 / Inspire-F1 even 2--20")
    # Natural source-specific stride sampling, used only for the single all-stride view.
    natural_parts, natural_labels = [], []
    for source in ("grab", "inspire_f1"):
        entries = [item for item in test_entries if item["source"] == source]
        dataset = ObjectInteractionCmDataset(
            Path("."), sequence_entries=entries, num_obj_points=int(meta_cfg["num_obj_points"]),
            num_hand_points=int(meta_cfg["num_hand_points"]), max_hand_points=int(meta_cfg["max_hand_points"]),
            min_stride=1, max_stride=max(SOURCE_STRIDES[source]), active_only=bool(data_cfg.get("active_only", True)),
            source_stride_values={source: SOURCE_STRIDES[source]},
        )
        count = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
        natural_parts.append(_collect(runner, dataset, count=count, batch_size=args.batch_size, seed=args.seed ^ 0x4E4154))
        natural_labels.append(SOURCE_LABELS[source])
    natural = _merge(natural_parts, natural_labels)
    natural_embedding = _tsne(natural["tokens"].mean(axis=1), seed=args.seed ^ 0x4E4154, n_iter=args.n_iter, perplexity=args.perplexity)
    _plot_simple(natural_embedding, natural, output / "all_stride_natural_tsne.png", "ObjectInteractionCm: natural all-stride sampling, dataset")
    _plot_simple(natural_embedding, natural, output / "all_stride_natural_hand_magnitude_tsne.png", "ObjectInteractionCm: natural all-stride sampling, hand-flow RMS", magnitude=True)
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": checkpoint_data.get("step"), "checkpoint_epoch": checkpoint_data.get("epoch"),
        "coordinate_frame": config["meta"].get("coordinate_frame"), "split": "test", "source_strides": {k: list(v) for k, v in SOURCE_STRIDES.items()},
        "samples_per_source": {SOURCE_NAMES[label]: int((records["labels"] == label).sum()) for label in (0, 1)},
        "samples_per_source_stride": {SOURCE_NAMES[label]: {str(s): int(((records["labels"] == label) & (records["stride"] == s)).sum()) for s in SOURCE_STRIDES[source]} for source, label in SOURCE_LABELS.items()},
        "test_sequence_counts": counts, "pooled_silhouette_original_space": float(silhouette_score(pooled, records["labels"])),
        "num_slots": int(tokens.shape[1]), "cm_dim": int(tokens.shape[2]), "note": "Inspire-F1 stride labels are even 2..20 because source data stride is 2; t-SNE is exploratory.",
    }
    np.savez_compressed(output / "tsne.npz", cm_tokens=tokens, pooled_cm=pooled, pooled_tsne=embedding, labels=records["labels"], strides=records["stride"], hand_flow_rms_mm=records["hand_flow_rms_mm"], obj_flow_rms_mm=records["obj_flow_rms_mm"])
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[object-interaction-cm-tsne] wrote {output / 'pooled_tsne_panels.png'}")
    print(f"[object-interaction-cm-tsne] samples={len(tokens)} pooled_silhouette={metadata['pooled_silhouette_original_space']}")


if __name__ == "__main__":
    main()
