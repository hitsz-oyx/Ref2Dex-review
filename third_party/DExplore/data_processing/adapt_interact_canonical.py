"""Adapt InterAct's canonical GRAB files to Dexplore's converter input.

InterAct stores canonical clips as ``human.npz``/``object.npz`` while the
Dexplore retargeter consumes the nested ``motion.npz`` representation.  This
adapter keeps the canonical human/object transforms, and takes contact and
metadata from the original GRAB clip (the latter is still required by
``convert_grab.py`` for the table/object alignment).
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


def _load_dict(path):
    with np.load(path, allow_pickle=True) as z:
        return {key: (z[key].item() if z[key].shape == () else z[key]) for key in z.files}


def _scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def adapt(interact_root, raw_root, output_root, subjects=None, metadata_root=None):
    interact_root = Path(interact_root)
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    canonical_root = interact_root / "data" / "grab" / "sequences_canonical"
    object_root = interact_root / "data" / "grab" / "objects"
    output_seq_root = output_root / "sequences"
    output_seq_root.mkdir(parents=True, exist_ok=True)

    # Keep meshes/templates in one place; symlinks avoid copying licensed data.
    for name, source in (("objects", object_root),
                         ("tools", raw_root / "tools")):
        dest = output_root / name
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(source, target_is_directory=True)

    names = sorted(path.name for path in canonical_root.iterdir() if path.is_dir())
    if subjects:
        subjects = set(subjects)
        names = [name for name in names if name.split("_", 1)[0] in subjects]
    if len(names) != 1335:
        print(f"Warning: found {len(names)} canonical clips (expected 1335)")

    for name in names:
        subject, raw_stem = name.split("_", 1)
        raw_path = raw_root / "grab" / subject / f"{raw_stem}.npz"
        metadata_path = (Path(metadata_root) / "sequences" / name / "motion.npz"
                         if metadata_root is not None else None)
        if metadata_path is not None and metadata_path.exists():
            raw_path = metadata_path
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing metadata/raw clip for {name}: {raw_path}")
        raw = _load_dict(raw_path)
        human = _load_dict(canonical_root / name / "human.npz")
        obj_path = canonical_root / name / "object.npz"
        out_dir = output_seq_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        poses = np.asarray(human["poses"], dtype=np.float32)
        trans = np.asarray(human["trans"], dtype=np.float32)
        if poses.shape[0] != trans.shape[0]:
            raise ValueError(f"{name}: poses/trans length mismatch")
        body_params = {
            "global_orient": poses[:, :3],
            "body_pose": poses[:, 3:66],
            "left_hand_pose": poses[:, 66:90],
            "right_hand_pose": poses[:, 90:114],
            "transl": trans,
        }

        # Contacts in the raw file are 120 Hz; InterAct process_grab and
        # canonicalization retain every fourth frame (30 Hz).
        contact_raw = raw.get("contact", {})
        contact = {}
        for key, value in contact_raw.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                contact[key] = value[::4]
            else:
                contact[key] = value

        seq = {
            "n_comps": np.array(int(_scalar(raw["n_comps"]))),
            "gender": np.array(str(_scalar(raw["gender"]))),
            "sbj_id": np.array(str(_scalar(raw["sbj_id"]))),
            "n_frames": np.array(poses.shape[0]),
            "body": {
                "params": body_params,
                "vtemp": raw["body"]["vtemp"],
            },
            "contact": contact,
        }
        np.savez(out_dir / "motion.npz", **seq)
        shutil.copyfile(obj_path, out_dir / "object.npz")

    print(f"Adapted {len(names)} InterAct canonical clips to {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interact-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--subjects", default="",
                        help="Comma-separated subject IDs for parallel processing")
    parser.add_argument("--metadata-root", default=None,
                        help="Optional processed GRAB root containing sequences/motion.npz")
    args = parser.parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    adapt(args.interact_root, args.raw_root, args.output_root,
          subjects=subjects, metadata_root=args.metadata_root)


if __name__ == "__main__":
    main()
