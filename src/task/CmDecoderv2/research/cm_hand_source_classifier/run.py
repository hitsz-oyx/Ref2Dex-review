"""Classify MANO versus Inspire source using frozen OICM Cm tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn

from src.base import load_config
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


MODIFICATION_VERSION = "V1.1.6"
LABELS = {"mano": 0, "inspire_rl": 1}


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _world_to_frame(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    return ((np.asarray(points, dtype=np.float32) - pose[:3, 3]) @ pose[:3, :3]).astype(np.float32)


def _normal_world_to_frame(normals: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose, dtype=np.float32)[:3, :3]
    values = np.asarray(normals, dtype=np.float32) @ rotation
    return (values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") & 0x7FFFFFFF


class GeometrySequence:
    """Minimal single right-hand view over one ObjectInteractionCm cache."""

    def __init__(self, entry: dict[str, Any], index_root: Path) -> None:
        self.entry = entry
        self.path = (index_root / str(entry["path"])).resolve()
        geometry = self.path / "geometry"
        required = {
            "object_points": geometry / "obj_points_pool_world.npy",
            "object_normals": geometry / "obj_normals_pool_world.npy",
            "object_pose": geometry / "obj_pose_world.npy",
            "hand_points": geometry / "hand_points_world.npy",
            "hand_normals": geometry / "hand_normals_world.npy",
            "source_frame": geometry / "source_frame_id.npy",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing geometry for {entry['id']}: {missing}")
        self.object_points = np.load(required["object_points"], mmap_mode="r")
        self.object_normals = np.load(required["object_normals"], mmap_mode="r")
        self.object_pose = np.load(required["object_pose"], mmap_mode="r")
        self.hand_points = np.load(required["hand_points"], mmap_mode="r")
        self.hand_normals = np.load(required["hand_normals"], mmap_mode="r")
        self.source_frame = np.load(required["source_frame"], mmap_mode="r")
        if self.object_points.shape[1:] != (4096, 3) or self.hand_points.shape[1:] != (1538, 3):
            raise ValueError(f"Unexpected cache shapes for {entry['id']}: object={self.object_points.shape}, hand={self.hand_points.shape}")
        if any(len(value) != len(self.object_points) for value in (self.object_normals, self.object_pose, self.hand_points, self.hand_normals, self.source_frame)):
            raise ValueError(f"Frame count mismatch for {entry['id']}")


def _load_entries(index_path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != "ref2dex_object_interaction_cm_index_v1_1":
        raise ValueError(f"Unsupported index schema: {payload.get('schema_name')!r}")
    entries = payload.get("sequences", {})
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val"):
        result[split] = []
        for raw in entries.get(split, []):
            item = dict(raw)
            path = (index_path.parent / str(item["path"])).resolve()
            if not path.is_dir():
                raise FileNotFoundError(path)
            item["resolved_path"] = str(path)
            result[split].append(item)
    return result


def _balanced_entries(entries: list[dict[str, Any]], common_objects: set[str], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        if entry.get("variant") not in LABELS or str(entry.get("object_name")) not in common_objects:
            continue
        grouped.setdefault((str(entry["object_name"]), str(entry["variant"])), []).append(entry)
    selected: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for object_name in sorted(common_objects):
        mano = list(grouped.get((object_name, "mano"), []))
        inspire = list(grouped.get((object_name, "inspire_rl"), []))
        count = min(len(mano), len(inspire))
        if count <= 0:
            continue
        for values in (mano, inspire):
            order = rng.permutation(len(values))[:count]
            selected.extend(values[int(index)] for index in order)
    selected.sort(key=lambda item: (str(item["object_name"]), str(item["variant"]), str(item["id"])))
    return selected


def _sample_frames(frame_count: int, per_sequence: int) -> np.ndarray:
    transitions = max(0, int(frame_count) - 1)
    count = min(int(per_sequence), transitions)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.linspace(0, transitions - 1, count, dtype=np.int64))


def _pool_tokens(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.concatenate([values.mean(axis=1), values.max(axis=1), values.std(axis=1)], axis=-1).astype(np.float32)


def _make_batch(sequence: GeometrySequence, frames: np.ndarray, num_obj_points: int) -> dict[str, np.ndarray]:
    object_points: list[np.ndarray] = []
    object_normals: list[np.ndarray] = []
    hand_points: list[np.ndarray] = []
    hand_normals: list[np.ndarray] = []
    hand_flow: list[np.ndarray] = []
    for frame in frames.tolist():
        pose = np.asarray(sequence.object_pose[frame], dtype=np.float32)
        rng = np.random.default_rng(_stable_seed(sequence.entry["id"], frame, 2024) ^ 0xA17)
        selected = rng.choice(4096, size=int(num_obj_points), replace=False)
        current_object = np.asarray(sequence.object_points[frame], dtype=np.float32)
        future_object = np.asarray(sequence.object_points[frame + 1], dtype=np.float32)
        current_hand = np.asarray(sequence.hand_points[frame], dtype=np.float32)
        future_hand = np.asarray(sequence.hand_points[frame + 1], dtype=np.float32)
        object_points.append(_world_to_frame(current_object[selected], pose))
        object_normals.append(_normal_world_to_frame(np.asarray(sequence.object_normals[frame, selected]), pose))
        hand_points.append(_world_to_frame(current_hand, pose))
        hand_normals.append(_normal_world_to_frame(np.asarray(sequence.hand_normals[frame]), pose))
        hand_flow.append(_world_to_frame(future_hand, pose) - _world_to_frame(current_hand, pose))
    return {
        "obj_points": np.stack(object_points).astype(np.float32),
        "obj_normals": np.stack(object_normals).astype(np.float32),
        "hand_points": np.stack(hand_points).astype(np.float32),
        "hand_normals": np.stack(hand_normals).astype(np.float32),
        "hand_flow": np.stack(hand_flow).astype(np.float32),
    }


def extract_features(
    model: ObjectInteractionCmModel,
    entries: dict[str, list[dict[str, Any]]],
    *,
    index_path: Path,
    device: torch.device,
    num_obj_points: int,
    frames_per_sequence: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    records: dict[str, list[Any]] = {"cm": [], "anchor": [], "label": [], "split": [], "source": [], "sequence": [], "object": [], "frame": [], "valid": [], "active": []}
    sequence_counts: dict[str, int] = {}
    for split in ("train", "val"):
        sequence_counts[split] = 0
        for entry_index, entry in enumerate(entries[split]):
            sequence = GeometrySequence(entry, index_path.parent)
            frames = _sample_frames(len(sequence.object_points), frames_per_sequence)
            if len(frames) == 0:
                continue
            label = LABELS[str(entry["variant"])]
            sequence_counts[split] += 1
            for start in range(0, len(frames), int(batch_size)):
                batch_frames = frames[start:start + int(batch_size)]
                arrays = _make_batch(sequence, batch_frames, num_obj_points)
                torch_batch = {
                    "obj_points": torch.from_numpy(arrays["obj_points"]).to(device),
                    "obj_normals": torch.from_numpy(arrays["obj_normals"]).to(device),
                    "obj_valid_mask": torch.ones((len(batch_frames), num_obj_points), dtype=torch.bool, device=device),
                    "hand_points": torch.from_numpy(arrays["hand_points"]).to(device),
                    "hand_normals": torch.from_numpy(arrays["hand_normals"]).to(device),
                    "hand_flow": torch.from_numpy(arrays["hand_flow"]).to(device),
                    "hand_valid_mask": torch.ones((len(batch_frames), 1538), dtype=torch.bool, device=device),
                }
                with torch.inference_mode():
                    output = model(torch_batch)
                cm = output["cm_tokens"].detach().cpu().numpy().astype(np.float32)
                anchor = output["cm_anchor_pos"].detach().cpu().numpy().astype(np.float32)
                valid = output["sample_valid"].detach().cpu().numpy().astype(bool)
                active = output["sampled_active_count"].detach().cpu().numpy().astype(np.int32)
                records["cm"].append(_pool_tokens(cm))
                records["anchor"].append(_pool_tokens(anchor))
                records["label"].append(np.full(len(batch_frames), label, dtype=np.int64))
                records["split"].append(np.full(len(batch_frames), split))
                records["source"].append(np.full(len(batch_frames), str(entry["variant"])))
                records["sequence"].append(np.full(len(batch_frames), str(entry["id"])))
                records["object"].append(np.full(len(batch_frames), str(entry["object_name"])))
                records["frame"].append(np.asarray(batch_frames, dtype=np.int64))
                records["valid"].append(valid)
                records["active"].append(active)
    arrays = {
        "cm_features": np.concatenate(records["cm"], axis=0),
        "anchor_features": np.concatenate(records["anchor"], axis=0),
        "label": np.concatenate(records["label"], axis=0),
        "split": np.concatenate(records["split"], axis=0).astype("U16"),
        "source": np.concatenate(records["source"], axis=0).astype("U16"),
        "sequence": np.concatenate(records["sequence"], axis=0).astype("U128"),
        "object": np.concatenate(records["object"], axis=0).astype("U64"),
        "frame": np.concatenate(records["frame"], axis=0),
        "sample_valid": np.concatenate(records["valid"], axis=0),
        "sampled_active_count": np.concatenate(records["active"], axis=0),
    }
    meta = {"sequences_processed": sequence_counts, "frames_total": int(len(arrays["label"]))}
    return arrays, meta


class SmallClassifier(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    counts = [int((labels == value).sum()) for value in (0, 1)]
    count = min(counts)
    if count <= 0:
        raise ValueError(f"Both classes are required, got counts={counts}")
    chosen = [rng.choice(np.flatnonzero(labels == value), count, replace=False) for value in (0, 1)]
    result = np.concatenate(chosen)
    rng.shuffle(result)
    return result.astype(np.int64)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= 0.5).astype(np.int64)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": auc,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "samples": int(len(labels)),
        "class_counts": {"mano": int((labels == 0).sum()), "inspire_rl": int((labels == 1).sum())},
    }


def train_one(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], SmallClassifier, dict[str, np.ndarray]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    train_indices = _balanced_indices(train_labels, seed)
    test_indices = _balanced_indices(test_labels, seed + 1)
    train_x = np.asarray(train_features[train_indices], dtype=np.float32)
    test_x = np.asarray(test_features[test_indices], dtype=np.float32)
    train_y = np.asarray(train_labels[train_indices], dtype=np.float32)
    test_y = np.asarray(test_labels[test_indices], dtype=np.int64)
    mean = train_x.mean(axis=0)
    std = np.clip(train_x.std(axis=0), 1e-6, None)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    model = SmallClassifier(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    tensor_x = torch.from_numpy(train_x)
    tensor_y = torch.from_numpy(train_y)
    generator = torch.Generator().manual_seed(int(seed))
    last_loss = None
    for _ in range(int(epochs)):
        order = torch.randperm(len(tensor_x), generator=generator)
        model.train()
        for start in range(0, len(order), int(batch_size)):
            indices = order[start:start + int(batch_size)]
            logits = model(tensor_x[indices].to(device))
            loss = loss_fn(logits, tensor_y[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(torch.from_numpy(test_x).to(device))).cpu().numpy()
    result = {
        "train": _metrics(train_y.astype(np.int64), torch.sigmoid(model(torch.from_numpy(train_x).to(device))).detach().cpu().numpy()),
        "test": _metrics(test_y, probabilities),
        "train_loss_last_batch": last_loss,
        "epochs": int(epochs),
        "input_dim": int(train_x.shape[1]),
    }
    return result, model, {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "test_probabilities": probabilities, "test_indices": test_indices}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json")
    parser.add_argument("--oicm-config", default="src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml")
    parser.add_argument("--oicm-checkpoint", default="outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="src/task/CmDecoderv2/research/cm_hand_source_classifier/output")
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-sequences-per-class", type=int, default=None)
    args = parser.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    index_path = _resolve(args.index)
    index_entries = _load_entries(index_path)
    per_split_common: list[set[str]] = []
    for split in ("train", "val"):
        mano_objects = {str(item["object_name"]) for item in index_entries[split] if item.get("variant") == "mano"}
        inspire_objects = {str(item["object_name"]) for item in index_entries[split] if item.get("variant") == "inspire_rl"}
        per_split_common.append(mano_objects & inspire_objects)
    common_objects = set.intersection(*per_split_common) if per_split_common else set()
    selected_entries = {
        split: _balanced_entries(index_entries[split], common_objects, int(args.seed) + (0 if split == "train" else 1))
        for split in ("train", "val")
    }
    if args.max_sequences_per_class is not None:
        limit = int(args.max_sequences_per_class)
        for split in ("train", "val"):
            by_source = {source: [entry for entry in selected_entries[split] if entry["variant"] == source] for source in LABELS}
            count = min(limit, *(len(values) for values in by_source.values()))
            selected_entries[split] = []
            for source in LABELS:
                selected_entries[split].extend(by_source[source][:count])
            selected_entries[split].sort(key=lambda item: str(item["id"]))

    device = torch.device(args.device)
    oicm_cfg = load_config(args.oicm_config)
    oicm_cfg.model.meta = oicm_cfg.meta
    model = ObjectInteractionCmModel(oicm_cfg.model).to(device)
    checkpoint_path = _resolve(args.oicm_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    model.requires_grad_(False)
    arrays, extraction_meta = extract_features(
        model,
        selected_entries,
        index_path=index_path,
        device=device,
        num_obj_points=int(oicm_cfg.meta.num_obj_points),
        frames_per_sequence=int(args.frames_per_sequence),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    run_id = f"cmdecoderv2-cm-source-classifier-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output = _resolve(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "features.npz", **arrays)

    results: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for sample_name, sample_mask in (("all", np.ones(len(arrays["label"]), dtype=bool)), ("valid_only", arrays["sample_valid"])):
        split_train = arrays["split"] == "train"
        split_test = arrays["split"] == "val"
        train_mask = split_train & sample_mask
        test_mask = split_test & sample_mask
        results[sample_name] = {"available": {"train": int(train_mask.sum()), "test": int(test_mask.sum())}}
        for feature_name in ("cm_features", "anchor_features"):
            try:
                result, classifier, norm = train_one(
                    arrays[feature_name][train_mask], arrays["label"][train_mask],
                    arrays[feature_name][test_mask], arrays["label"][test_mask],
                    seed=int(args.seed) + len(results[sample_name]), epochs=int(args.epochs),
                    batch_size=128, device=device,
                )
            except ValueError as error:
                results[sample_name][feature_name] = {"status": "SKIPPED", "reason": str(error)}
                continue
            results[sample_name][feature_name] = result
            model_key = f"{sample_name}_{feature_name}"
            torch.save({"model": classifier.state_dict(), "mean": norm["mean"], "std": norm["std"], "feature_name": feature_name, "sample_name": sample_name}, output / f"classifier_{model_key}.pt")
            models[model_key] = str(output / f"classifier_{model_key}.pt")

    metrics = {
        "run_id": run_id,
        "modification_version": MODIFICATION_VERSION,
        "label_map": LABELS,
        "common_object_names": sorted(common_objects),
        "sequence_counts": {split: {source: int(sum(1 for item in selected_entries[split] if item["variant"] == source)) for source in LABELS} for split in ("train", "val")},
        "frames_per_sequence": int(args.frames_per_sequence),
        "seed": int(args.seed),
        "results": results,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_name": "ref2dex_cmdecoderv2_cm_hand_source_classifier_v1",
        "task": "CmDecoderv2",
        "activity_id": args.activity_id,
        "run_id": run_id,
        "run_status": "COMPLETED",
        "conclusion": "INCONCLUSIVE",
        "conclusion_scope": "small classifier diagnostic of source/embodiment information in frozen OICM cm_tokens; not a hand-shape disentanglement proof",
        "modification_version": MODIFICATION_VERSION,
        "operation_category": ["diagnostic", "experiment", "operation"],
        "created_at": _now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "index": str(index_path),
        "index_sha256": _sha256(index_path),
        "oicm_config": str(_resolve(args.oicm_config)),
        "oicm_checkpoint": str(checkpoint_path),
        "oicm_checkpoint_sha256": _sha256(checkpoint_path),
        "split_contract": "existing ObjectInteractionCm train sequences for classifier training, val sequences for held-out test; no frame-level mixing",
        "feature_contract": "primary=cm_tokens mean/max/std pooling; control=cm_anchor_pos mean/max/std pooling; no hand flow or object points in classifier input",
        "invalid_cm_contract": "all samples and sample_valid=true subset are reported separately; invalid dummy tokens have no physical semantics",
        "common_object_names": sorted(common_objects),
        "sequence_counts": metrics["sequence_counts"],
        "extraction": extraction_meta,
        "outputs": {
            "features": str(output / "features.npz"),
            "metrics": str(output / "metrics.json"),
            "run_manifest": str(output / "run_manifest.json"),
            "classifiers": models,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "output": str(output), "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
