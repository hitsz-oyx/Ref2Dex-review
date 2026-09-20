"""Assemble the missing DExplore GRAB ``motion.npz`` input contract for V1.20.

The public DExplore README requires this file but public InterAct writes only
``human.npz`` and ``object.npz``.  This adapter preserves the existing 30 Hz
body payload and replaces only its downsampled contact payload with untouched
native-rate contact from the matching raw GRAB sequence.  It is explicitly a
reconstructed baseline, not a claim of access to the authors' unpublished
producer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LEGACY_ROOT = REPOSITORY_ROOT / "data/processed_data/dexplore_grab/sequences"
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "data/raw_data/GRAB/grab"
DEFAULT_RAW_OBJECT_ROOT = REPOSITORY_ROOT / "data/raw_data/GRAB/objects"
DEFAULT_RAW_TOOLS_ROOT = REPOSITORY_ROOT / "data/raw_data/GRAB/tools"
DEFAULT_SKELETON_ROOT = REPOSITORY_ROOT.parent / "InterAct/simulation/intermimic/data/assets/smplx"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "data/processed_data/dexplore_reconstructed_v120"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.ndarray) and value.shape == () else value


def _load_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as archive:
        return {key: _scalar(archive[key]) for key in archive.files}


def _validate(legacy: dict[str, Any], raw: dict[str, Any]) -> tuple[int, dict[str, np.ndarray]]:
    if "body" not in legacy or "contact" not in legacy:
        raise ValueError("legacy motion must contain body and contact")
    if "contact" not in raw:
        raise ValueError("raw GRAB sequence must contain contact")
    frames = int(_scalar(legacy["n_frames"]))
    body = legacy["body"]
    if not isinstance(body, dict) or not isinstance(body.get("params"), dict):
        raise ValueError("legacy motion body.params must be a mapping")
    lengths = {np.asarray(value).shape[0] for value in body["params"].values()
               if isinstance(value, np.ndarray) and value.ndim > 0}
    if lengths != {frames}:
        raise ValueError(f"legacy body parameter lengths {sorted(lengths)} do not equal n_frames={frames}")
    contact = raw["contact"]
    if not isinstance(contact, dict) or not {"body", "object"}.issubset(contact):
        raise ValueError("raw contact must contain body and object arrays")
    native = {key: np.asarray(value) for key, value in contact.items()}
    required = 4 * (frames - 1) + 1
    if native["body"].shape[0] < required or native["object"].shape[0] < required:
        raise ValueError(f"native contact must contain at least {required} frames for {frames} motion frames")
    if native["body"].shape[0] <= frames or native["object"].shape[0] <= frames:
        raise ValueError("native contact is already downsampled")
    return frames, native


def derive_body_translation(*, baseline_tensor: Path, reference_tensor: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive a per-frame source-frame body translation from DExplore tensors.

    DExplore rotates SMPL-X translations by +90 degrees about X while emitting
    the 598-D tensor.  The correction aligns the right-hand/object relative
    position (slots 51:54 and 198:201) with the geometric reference and then
    applies the inverse rotation: ``[x, y, z]_source = [x, z, -y]_output``.
    """
    import torch

    baseline = torch.load(baseline_tensor, map_location="cpu", weights_only=True).float()
    reference = torch.load(reference_tensor, map_location="cpu", weights_only=True).float()
    if baseline.ndim != 2 or reference.ndim != 2 or baseline.shape != reference.shape or baseline.shape[1] != 598:
        raise ValueError("alignment tensors must have matching (T, 598) shapes")
    if not torch.isfinite(baseline).all() or not torch.isfinite(reference).all():
        raise ValueError("alignment tensors must be finite")
    output_delta = ((reference[:, 51:54] - reference[:, 198:201])
                    - (baseline[:, 51:54] - baseline[:, 198:201])).detach().numpy()
    source_delta = np.stack((output_delta[:, 0], output_delta[:, 2], -output_delta[:, 1]), axis=1).astype(np.float32)
    provenance = {
        "method": "right_hand_object_relative_inverse_rotation_x90",
        "baseline_tensor": {"path": str(baseline_tensor.resolve()), "sha256": _sha256(baseline_tensor)},
        "reference_tensor": {"path": str(reference_tensor.resolve()), "sha256": _sha256(reference_tensor)},
        "output_relative_delta_m": {
            "shape": list(output_delta.shape), "min": output_delta.min(axis=0).tolist(),
            "max": output_delta.max(axis=0).tolist(), "mean": output_delta.mean(axis=0).tolist(),
        },
    }
    return source_delta, provenance


def build(*, sequence: str, legacy_root: Path, raw_root: Path, raw_object_root: Path,
          raw_tools_root: Path, skeleton_root: Path, output_root: Path,
          body_translation: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
          alignment_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    subject, action = sequence.split("_", 1)
    legacy_dir = legacy_root / sequence
    legacy_motion = legacy_dir / "motion.npz"
    legacy_object = legacy_dir / "object.npz"
    raw_motion = raw_root / subject / f"{action}.npz"
    if not legacy_motion.is_file() or not legacy_object.is_file() or not raw_motion.is_file():
        raise FileNotFoundError("missing legacy motion/object or matching raw GRAB sequence")
    target_dir = output_root / "sequences" / sequence
    if target_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing reconstructed sequence: {target_dir}")
    legacy = _load_payload(legacy_motion)
    raw = _load_payload(raw_motion)
    frames, native_contact = _validate(legacy, raw)
    legacy_object_payload = _load_payload(legacy_object)
    object_name = str(legacy_object_payload["name"])
    raw_mesh = raw_object_root / object_name / "mesh.obj"
    if not raw_mesh.is_file():
        raise FileNotFoundError(f"missing raw object mesh: {raw_mesh}")
    subject_template = legacy.get("body", {}).get("vtemp")
    if not isinstance(subject_template, str) or not subject_template.startswith("tools/"):
        raise ValueError("legacy motion body.vtemp must be a tools-relative subject template path")
    raw_subject_mesh = raw_tools_root.parent / subject_template
    if not raw_subject_mesh.is_file():
        raise FileNotFoundError(f"missing raw subject mesh: {raw_subject_mesh}")
    skeleton = skeleton_root / f"smplx_grab_{subject}.xml"
    if not skeleton.is_file():
        raise FileNotFoundError(f"missing InterAct skeleton: {skeleton}")
    target_dir.mkdir(parents=True)
    translation = np.asarray(body_translation, dtype=np.float32)
    if translation.shape not in {(3,), (frames, 3)}:
        raise ValueError(f"body_translation must have shape (3,) or {(frames, 3)}, got {translation.shape}")
    payload = copy.deepcopy(legacy)
    transl = np.asarray(payload["body"]["params"].get("transl"))
    if transl.shape != (frames, 3):
        raise ValueError(f"body.params.transl must have shape {(frames, 3)}, got {transl.shape}")
    payload["body"]["params"]["transl"] = transl.astype(np.float32, copy=True) + translation
    payload["contact"] = native_contact
    np.savez(target_dir / "motion.npz", **payload)
    shutil.copy2(legacy_object, target_dir / "object.npz")
    output_mesh = output_root / "objects" / object_name / f"{object_name}.obj"
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_mesh, output_mesh)
    output_subject_mesh = output_root / subject_template
    output_subject_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_subject_mesh, output_subject_mesh)
    output_skeleton = output_root / "skeletons" / skeleton.name
    output_skeleton.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(skeleton, output_skeleton)
    manifest = {
            "schema": "dexplore_reconstructed_motion_input_v1",
            "classification": "reconstructed_baseline",
            "sequence": sequence,
            "motion_frames_30hz": frames,
            "contact_source": "raw_grab_native_rate",
            "contact_frames": {key: int(value.shape[0]) for key, value in native_contact.items()
                               if isinstance(value, np.ndarray) and value.ndim > 0},
            "body_world_translation_m": {
                "kind": "constant" if translation.ndim == 1 else "per_frame",
                "shape": list(translation.shape),
                "min": translation.min(axis=0).tolist(), "max": translation.max(axis=0).tolist(),
                "mean": translation.mean(axis=0).tolist(),
            },
            "inputs": {
                "legacy_motion": {"path": str(legacy_motion.resolve()), "sha256": _sha256(legacy_motion)},
                "legacy_object": {"path": str(legacy_object.resolve()), "sha256": _sha256(legacy_object)},
                "raw_grab": {"path": str(raw_motion.resolve()), "sha256": _sha256(raw_motion)},
                "raw_object_mesh": {"path": str(raw_mesh.resolve()), "sha256": _sha256(raw_mesh)},
                "raw_subject_mesh": {"path": str(raw_subject_mesh.resolve()), "sha256": _sha256(raw_subject_mesh)},
                "legacy_interact_skeleton": {"path": str(skeleton.resolve()), "sha256": _sha256(skeleton)},
            },
            "outputs": {
                "motion": {"path": str((target_dir / "motion.npz").resolve()),
                           "sha256": _sha256(target_dir / "motion.npz")},
                "object": {"path": str((target_dir / "object.npz").resolve()),
                           "sha256": _sha256(target_dir / "object.npz")},
                "object_mesh": {"path": str(output_mesh.resolve()), "sha256": _sha256(output_mesh)},
                "subject_mesh": {"path": str(output_subject_mesh.resolve()), "sha256": _sha256(output_subject_mesh)},
                "skeleton": {"path": str(output_skeleton.resolve()), "sha256": _sha256(output_skeleton)},
            },
            "invariants": [
                "body metadata and parameters are copied from legacy 30 Hz motion",
                "object.npz is copied byte-for-byte from legacy input",
                "raw object mesh is copied byte-for-byte into the public converter's filename convention",
                "subject template is copied byte-for-byte into the public converter's vtemp convention",
                "legacy InterAct-generated skeleton is copied byte-for-byte for this reconstructed converter input",
                "raw contact arrays are copied without interpolation, repetition, filtering, or numeric conversion",
            ],
        }
    if alignment_provenance is not None:
        manifest["body_translation_alignment"] = alignment_provenance
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="s1_airplane_lift")
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--raw-object-root", type=Path, default=DEFAULT_RAW_OBJECT_ROOT)
    parser.add_argument("--raw-tools-root", type=Path, default=DEFAULT_RAW_TOOLS_ROOT)
    parser.add_argument("--skeleton-root", type=Path, default=DEFAULT_SKELETON_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--body-translation", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=(0.0, 0.0, 0.0),
                        help="Explicit world-frame translation applied only to body.params.transl")
    parser.add_argument("--alignment-baseline-tensor", type=Path,
                        help="598-D baseline tensor used to derive a per-frame body translation")
    parser.add_argument("--alignment-reference-tensor", type=Path,
                        help="598-D geometric tensor used as the alignment reference")
    args = parser.parse_args()
    if bool(args.alignment_baseline_tensor) != bool(args.alignment_reference_tensor):
        parser.error("--alignment-baseline-tensor and --alignment-reference-tensor must be supplied together")
    translation: np.ndarray | tuple[float, float, float] = tuple(args.body_translation)
    alignment_provenance = None
    if args.alignment_baseline_tensor:
        translation, alignment_provenance = derive_body_translation(
            baseline_tensor=args.alignment_baseline_tensor, reference_tensor=args.alignment_reference_tensor)
    print(json.dumps(build(sequence=args.sequence, legacy_root=args.legacy_root,
                           raw_root=args.raw_root, raw_object_root=args.raw_object_root,
                           raw_tools_root=args.raw_tools_root,
                           skeleton_root=args.skeleton_root,
                           output_root=args.output_root,
                           body_translation=translation,
                           alignment_provenance=alignment_provenance), indent=2))


if __name__ == "__main__":
    main()
