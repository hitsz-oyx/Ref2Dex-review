"""Slot-wise t-SNE visualization for the Inspire-F1 fine-tuned Cm model.

The script evaluates the same Cm checkpoint on held-out GRAB and Inspire-F1
transitions, using a fixed endpoint stride.  Each slot is embedded separately
so the resulting PNG contains one panel per Cm slot.  The exported ``.npz``
keeps the high-dimensional vectors and labels for later quantitative tests.

Example::

    python -m src.task.Cm.research.tsne_slots \
      --checkpoint outputs/cm/cm_hrdexdb_inspire_f1_decoder_only_resume_20260827_011451/checkpoints/latest.pt \
      --output-root output/research/cm_tsne_inspire_f1_latest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset, Subset

from src.base import build_runner_from_checkpoint
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
DEFAULT_OUTPUT = "output/research/cm_tsne_inspire_f1_latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--grab-test-split", default=DEFAULT_GRAB_SPLIT)
    parser.add_argument("--hrdexdb-root", default=DEFAULT_HRDEXDB_ROOT)
    parser.add_argument("--hrdexdb-manifest", default=DEFAULT_HRDEXDB_MANIFEST)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum transitions per dataset; both datasets are capped equally (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    return parser.parse_args()


def _build_datasets(args: argparse.Namespace, *, num_obj_points: int, num_hand_points: int, seed: int):
    grab_root = Path(args.grab_root).resolve()
    grab_split = Path(args.grab_test_split).resolve()
    grab_sequences = _read_sequence_split(grab_split, grab_root)
    grab = _MmapSequenceDataset(
        grab_sequences,
        num_obj_points=num_obj_points,
        num_hand_points=num_hand_points,
        base_seed=seed,
        min_stride=5,
        max_stride=5,
        fixed_stride=5,
        active_only=True,
        sampling_bank_size=4,
        fixed_eval_bank=0,
    )

    hroot = Path(args.hrdexdb_root).resolve()
    manifest = Path(args.hrdexdb_manifest).resolve()
    inspire_episodes = _manifest_specs(hroot, manifest, "test", "inspire_f1")
    inspire = HrdexdbGeometryDataset(
        inspire_episodes,
        num_obj_points=num_obj_points,
        num_obj_pool=4096,
        num_hand_points=num_hand_points,
        base_seed=seed,
        min_stride=5,
        max_stride=5,
        fixed_stride=5,
        active_only=True,
    )
    return grab, inspire, grab_sequences, inspire_episodes


def _balanced_subset(dataset: Dataset, max_samples: int, seed: int) -> Subset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False))
    return Subset(dataset, indices.tolist())


def _iter_tensors(loader: DataLoader, runner) -> Iterable[np.ndarray]:
    for batch in loader:
        with torch.inference_mode():
            output = runner.inference(runner.model, batch)
        tokens = output["cm_tokens"].detach().cpu().numpy()
        if tokens.ndim != 3:
            raise ValueError(f"Expected cm_tokens [B,K,C], got {tokens.shape}")
        yield tokens


def _collect_tokens(runner, dataset: Dataset, *, batch_size: int, label: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    chunks = list(_iter_tensors(loader, runner))
    if not chunks:
        raise ValueError("Dataset produced no samples")
    tokens = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    labels = np.full(tokens.shape[0], label, dtype=np.int64)
    return tokens, labels


def _run_tsne(tokens: np.ndarray, *, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    n_samples = int(tokens.shape[0])
    if n_samples < 4:
        raise ValueError(f"Need at least 4 samples for t-SNE, got {n_samples}")
    effective_perplexity = min(float(perplexity), float(max(2, (n_samples - 1) // 3)))
    model = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=int(n_iter),
        random_state=int(seed),
        metric="euclidean",
    )
    return model.fit_transform(tokens).astype(np.float32, copy=False)


def _plot_slots(embedding: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    num_slots = embedding.shape[1]
    rows = int(np.ceil(num_slots / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows), squeeze=False)
    colors = {0: "#2878b5", 1: "#d95f02"}
    names = {0: "GRAB", 1: "Inspire-F1"}
    for slot in range(num_slots):
        ax = axes[slot // 4][slot % 4]
        for label in (0, 1):
            mask = labels == label
            ax.scatter(
                embedding[mask, slot, 0],
                embedding[mask, slot, 1],
                s=8,
                alpha=0.62,
                c=colors[label],
                label=names[label],
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(f"Cm slot {slot}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for slot in range(num_slots, rows * 4):
        axes[slot // 4][slot % 4].axis("off")
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=2, frameon=False)
    fig.suptitle("Inspire-F1 fine-tuned Cm: slot-wise t-SNE (stride=5)", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runner = build_runner_from_checkpoint(
        args.checkpoint,
        mode="eval",
        device=args.device,
        build_data=False,
    )
    checkpoint_data = runner.setup_inference(args.checkpoint)
    num_obj_points = int(runner.cfg.meta.num_obj_points)
    num_hand_points = int(runner.cfg.meta.num_hand_points)
    grab, inspire, grab_sequences, inspire_episodes = _build_datasets(
        args,
        num_obj_points=num_obj_points,
        num_hand_points=num_hand_points,
        seed=int(args.seed),
    )
    target_count = min(len(grab), len(inspire)) if args.max_samples <= 0 else min(args.max_samples, len(grab), len(inspire))
    if target_count < 4:
        raise ValueError(f"Equalized sample count is too small: {target_count}")
    grab_subset = _balanced_subset(grab, target_count, int(args.seed) ^ 0x47524142)
    inspire_subset = _balanced_subset(inspire, target_count, int(args.seed) ^ 0x494E5350)
    grab_tokens, grab_labels = _collect_tokens(runner, grab_subset, batch_size=args.batch_size, label=0)
    inspire_tokens, inspire_labels = _collect_tokens(runner, inspire_subset, batch_size=args.batch_size, label=1)
    tokens = np.concatenate([grab_tokens, inspire_tokens], axis=0)
    labels = np.concatenate([grab_labels, inspire_labels], axis=0)
    num_slots = int(tokens.shape[1])
    embedding = np.stack(
        [_run_tsne(tokens[:, slot, :], perplexity=args.perplexity, n_iter=args.n_iter, seed=args.seed + slot) for slot in range(num_slots)],
        axis=1,
    )
    slot_silhouette = [float(silhouette_score(tokens[:, slot, :], labels, metric="euclidean")) for slot in range(num_slots)]
    np.savez_compressed(
        output_root / "slot_tsne.npz",
        cm_tokens=tokens,
        tsne=embedding,
        labels=labels,
        dataset_names=np.asarray(["GRAB", "Inspire-F1"], dtype="U10")[labels],
    )
    _plot_slots(embedding, labels, output_root / "slot_tsne.png")
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_data.get("step"),
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "stride": 5,
        "datasets": {"GRAB": int(target_count), "Inspire-F1": int(target_count)},
        "grab_test_sequences": len(grab_sequences),
        "inspire_f1_test_episodes": len(inspire_episodes),
        "num_slots": num_slots,
        "cm_dim": int(tokens.shape[2]),
        "perplexity_requested": float(args.perplexity),
        "n_iter": int(args.n_iter),
        "seed": int(args.seed),
        "coordinate_frame": str(runner.cfg.meta.coordinate_frame),
        "slot_silhouette_original_space": slot_silhouette,
        "slot_silhouette_mean": float(np.mean(slot_silhouette)),
        "note": "Each slot is embedded independently; t-SNE is exploratory and not a trainability metric.",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cm-tsne] wrote {output_root / 'slot_tsne.png'}")
    print(f"[cm-tsne] wrote {output_root / 'slot_tsne.npz'} samples_per_dataset={target_count} slots={num_slots}")
    print(f"[cm-tsne] original-space slot silhouette mean={float(np.mean(slot_silhouette)):.4f}")


if __name__ == "__main__":
    main()
