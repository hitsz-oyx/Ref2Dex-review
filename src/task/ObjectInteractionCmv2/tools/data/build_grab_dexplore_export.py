#!/usr/bin/env python3
"""Build reproducible GRAB inputs for the DExplore Inspire RL exporter.

This tool keeps all generated data below ``data/processed_data`` and treats
the external GRAB, SMPL-X, InterAct and DExplore repositories as read-only
inputs.  It combines the GRAB-only InterAct preprocessing and canonicalization
steps into a per-sequence operation, so workers can safely own disjoint shards.

The DExplore-compatible selection is derived from the same SMPL-X segment and
LBS mapping used by ``convert_grab.py``: clips with any left-hand contact and
clips whose name contains ``doorknob`` are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


SCHEMA_NAME = "ref2dex_object_interaction_cmv2_grab_dexplore_export_v1"
MOTION_FILENAME = "interaction_hand_inspire.pt"
MODIFICATION_VERSION = "V1.0.3"
HISTORIC_FILTERED_COUNT = 660
LEFT_TARGET_JOINTS = tuple(range(17, 33))
OUTPUT_WIDTH = 598


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return {"root": str(root.resolve()), "commit": commit, "dirty": dirty}


def _load_scalar_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as archive:
        return {
            key: archive[key].item() if archive[key].shape == () else archive[key]
            for key in archive.files
        }


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _raw_files(raw_root: Path) -> list[Path]:
    paths = sorted((raw_root / "grab").glob("s*/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No GRAB sequences below {raw_root / 'grab'}")
    return paths


def _sequence_name(path: Path) -> str:
    return f"{path.parent.name}_{path.stem}"


def _subject_template(raw_root: Path, raw: dict[str, Any]) -> np.ndarray:
    relative = str(raw["body"]["vtemp"])
    candidates = [raw_root / relative]
    for prefix in ("tools/subject_meshes/", "subject_meshes/"):
        if relative.startswith(prefix):
            candidates.append(raw_root / "tools" / relative[len(prefix):])
    for path in candidates:
        if path.is_file():
            return np.asarray(
                trimesh.load(path, process=False, force="mesh").vertices,
                dtype=np.float32,
            )
    raise FileNotFoundError(f"Subject template is missing; tried {candidates}")


def _load_dexplore_constants(dexplore_root: Path):
    data_processing = dexplore_root / "data_processing"
    if not data_processing.is_dir():
        raise FileNotFoundError(f"DExplore data_processing is missing: {data_processing}")
    sys.path.insert(0, str(data_processing))
    try:
        from smpl_constants import (  # type: ignore
            SMPLH_BONE_ORDER_NAMES,
            SMPLH_SEGMENT,
            SMPL_2_MUJOCO,
            load_smplx_vert_segmentation,
        )
    finally:
        sys.path.pop(0)
    return (
        SMPLH_BONE_ORDER_NAMES,
        SMPLH_SEGMENT,
        np.asarray(SMPL_2_MUJOCO, dtype=np.int64),
        load_smplx_vert_segmentation,
    )


def _left_contact_vertices(
    raw_root: Path,
    smplx_model_root: Path,
    dexplore_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import smplx

    first = _raw_files(raw_root)[0]
    raw = _load_scalar_archive(first)
    vtemplate = _subject_template(raw_root, raw)
    model = smplx.create(
        str(smplx_model_root),
        model_type="smplx",
        gender=str(_scalar(raw["gender"])).lower(),
        num_pca_comps=int(_scalar(raw["n_comps"])),
        v_template=vtemplate,
        batch_size=1,
    )
    names, segments, smpl_to_mujoco, load_segmentation = _load_dexplore_constants(
        dexplore_root
    )
    segmentation = load_segmentation()
    mask = torch.zeros((52, 10475), dtype=torch.bool)
    for index, name in enumerate(names):
        if index > 21:
            break
        mask[index, segmentation[segments[name]]] = True
    dominant_joint = model.lbs_weights.argmax(dim=1)
    for vertex in range(len(dominant_joint)):
        row = int(dominant_joint[vertex]) - 3
        if 21 < row < 52:
            mask[row, vertex] = True
    source_rows = smpl_to_mujoco[list(LEFT_TARGET_JOINTS)]
    vertices = torch.any(mask[source_rows.tolist()], dim=0).nonzero().flatten().numpy()
    return vertices.astype(np.int64), source_rows


def command_select(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.resolve()
    vertices, source_rows = _left_contact_vertices(
        raw_root, args.smplx_model_root.resolve(), args.dexplore_root.resolve()
    )
    selected: list[str] = []
    excluded_left: list[str] = []
    excluded_doorknob: list[str] = []
    paths = _raw_files(raw_root)
    for index, path in enumerate(paths, 1):
        raw = _load_scalar_archive(path)
        contact_body = np.asarray(raw["contact"]["body"])
        name = _sequence_name(path)
        if bool((contact_body[:, vertices] > 0).any()):
            excluded_left.append(name)
        elif "doorknob" in name:
            excluded_doorknob.append(name)
        else:
            selected.append(name)
        if index % 100 == 0 or index == len(paths):
            print(f"Scanned {index}/{len(paths)} raw GRAB clips", flush=True)
    payload = {
        "schema_name": SCHEMA_NAME,
        "created_at": _now(),
        "modification_version": MODIFICATION_VERSION,
        "seed": args.seed,
        "raw_root": str(raw_root),
        "smplx_model_root": str(args.smplx_model_root.resolve()),
        "dexplore": _git_state(args.dexplore_root.resolve()),
        "filter": {
            "exclude_any_left_hand_contact": True,
            "left_target_joint_indices": list(LEFT_TARGET_JOINTS),
            "left_source_rows": source_rows.tolist(),
            "num_left_vertices": int(len(vertices)),
            "exclude_name_contains": ["doorknob"],
        },
        "counts": {
            "source": len(paths),
            "selected": len(selected),
            "excluded_left_contact": len(excluded_left),
            "excluded_doorknob_after_contact_filter": len(excluded_doorknob),
            "historic_documented_selected": HISTORIC_FILTERED_COUNT,
            "historic_count_delta": len(selected) - HISTORIC_FILTERED_COUNT,
        },
        "selected": selected,
        "excluded_left_contact": excluded_left,
        "excluded_doorknob": excluded_doorknob,
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload["counts"], indent=2))


def _load_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != SCHEMA_NAME:
        raise ValueError(
            f"Expected selection schema {SCHEMA_NAME}, got {payload.get('schema_name')!r}"
        )
    selected = payload.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"Selection contains no clips: {path}")
    return payload


def _selected_names(selection: dict[str, Any], limit: int | None) -> list[str]:
    names = [str(name) for name in selection["selected"]]
    return names if limit is None else names[: max(0, int(limit))]


def _raw_path(raw_root: Path, name: str) -> Path:
    subject, stem = name.split("_", 1)
    path = raw_root / "grab" / subject / f"{stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Raw GRAB clip is missing: {path}")
    return path


def _export_object_assets(raw_root: Path, object_root: Path) -> int:
    source_root = raw_root / "tools" / "object_meshes" / "contact_meshes"
    sources = sorted(source_root.glob("*.ply"))
    if not sources:
        raise FileNotFoundError(f"No GRAB contact meshes below {source_root}")
    object_root.mkdir(parents=True, exist_ok=True)
    for source in sources:
        output = object_root / source.stem / f"{source.stem}.obj"
        if output.exists():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load(source, process=False, force="mesh")
        temporary = output.with_suffix(".obj.tmp")
        mesh.export(temporary, file_type="obj")
        temporary.replace(output)
    return len(sources)


def _skeleton_xml(model: Any, output: Path, mapping: Iterable[int]) -> None:
    mapping = [int(value) for value in mapping]
    selected = list(range(22)) + list(range(25, 55))
    model_parents = [int(value) for value in model.parents.detach().cpu().tolist()]
    parents_source = []
    for source in selected:
        parent = model_parents[source]
        parents_source.append(-1 if parent < 0 else selected.index(parent))
    with torch.no_grad():
        device = next(model.parameters()).device
        zero = torch.zeros((1, 3), device=device)
        body = torch.zeros((1, 63), device=device)
        hand = torch.zeros((1, model.num_pca_comps), device=device)
        result = model(
            global_orient=zero,
            body_pose=body,
            left_hand_pose=hand,
            right_hand_pose=hand,
            transl=zero,
            betas=torch.zeros((1, 10), device=device),
            expression=torch.zeros((1, 10), device=device),
            jaw_pose=zero,
            leye_pose=zero,
            reye_pose=zero,
        )
        source_joints = torch.cat(
            [result.joints[:, :22], result.joints[:, 25:55]], dim=1
        )[0]
        target_joints = source_joints[mapping].detach().cpu().numpy()
    target_parents = []
    for source_index in mapping:
        source_parent = parents_source[source_index]
        target_parents.append(-1 if source_parent < 0 else mapping.index(source_parent))
    children: list[list[int]] = [[] for _ in mapping]
    for index, parent in enumerate(target_parents):
        if parent >= 0:
            children[parent].append(index)
    root = ET.Element("mujoco", attrib={"model": "smplx_grab"})
    worldbody = ET.SubElement(root, "worldbody")
    bodies: list[Any] = [None] * len(mapping)

    def add_body(index: int) -> None:
        parent = target_parents[index]
        xml_parent = worldbody if parent < 0 else bodies[parent]
        position = (
            np.zeros(3, dtype=np.float32)
            if parent < 0
            else target_joints[index] - target_joints[parent]
        )
        bodies[index] = ET.SubElement(
            xml_parent,
            "body",
            attrib={
                "name": f"joint_{index:02d}",
                "pos": "%.9g %.9g %.9g" % tuple(position),
            },
        )
        for child in children[index]:
            add_body(child)

    for root_index, parent in enumerate(target_parents):
        if parent < 0:
            add_body(root_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def command_assets(args: argparse.Namespace) -> None:
    import smplx

    selection = _load_selection(args.selection.resolve())
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    tools_link = output_root / "tools"
    tools_link.symlink_to(raw_root / "tools", target_is_directory=True)
    num_objects = _export_object_assets(raw_root, output_root / "objects")
    _, _, mapping, _ = _load_dexplore_constants(args.dexplore_root.resolve())
    subjects: dict[str, str] = {}
    for name in selection["selected"]:
        subjects.setdefault(name.split("_", 1)[0], name)
    for subject, name in sorted(subjects.items()):
        raw = _load_scalar_archive(_raw_path(raw_root, name))
        model = smplx.create(
            str(args.smplx_model_root.resolve()),
            model_type="smplx",
            gender=str(_scalar(raw["gender"])).lower(),
            num_pca_comps=int(_scalar(raw["n_comps"])),
            v_template=_subject_template(raw_root, raw),
            batch_size=1,
        ).to(args.device)
        _skeleton_xml(model, output_root / "skeletons" / f"smplx_grab_{subject}.xml", mapping)
        print(f"Prepared skeleton for {subject}", flush=True)
    print(f"Prepared {num_objects} object meshes and {len(subjects)} subject skeletons")


def command_init(args: argparse.Namespace) -> None:
    selection_path = args.selection.resolve()
    selection = _load_selection(selection_path)
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    repo_root = Path(__file__).resolve().parents[5]
    run_manifest = {
        "task": "ObjectInteractionCmv2",
        "modification_version": MODIFICATION_VERSION,
        "run_id": args.run_id,
        "run_status": "STARTED",
        "started_at": _now(),
        "operation_category": ["data", "operation"],
        "seed": args.seed,
        "devices": args.gpus,
        "limit": args.limit,
        "num_selected": len(_selected_names(selection, args.limit)),
        "selection_manifest": os.path.relpath(selection_path, run_root),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "external_code": {
            "ref2dex": _git_state(repo_root),
            "dexplore": _git_state(args.dexplore_root.resolve()),
        },
        "outputs": {
            "canonical": "canonical",
            "geometric": "geometric",
            "rl": "rl",
            "manifest": "manifest.json",
            "validation": "validation.json",
        },
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(run_root / "run_manifest.json", run_manifest)
    (run_root / "commands.log").write_text(
        f"{run_manifest['started_at']} run initialized\n", encoding="utf-8"
    )
    print(json.dumps(run_manifest, indent=2))


def command_mark(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["run_status"] = args.status
    payload["completed_at"] = _now()
    payload["failure_reason"] = args.reason
    payload["conclusion"] = (
        "INVALID_IMPLEMENTATION" if args.status == "FAILED" else "INCONCLUSIVE"
    )
    _write_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))


def _transform_points(points: np.ndarray, rotation: Rotation) -> np.ndarray:
    shape = points.shape
    return rotation.apply(points.reshape(-1, 3)).reshape(shape)


def _canonicalize_one(
    raw_path: Path,
    raw_root: Path,
    output_root: Path,
    smplx_model_root: Path,
    device: str,
    stride: int,
) -> tuple[str, int]:
    import smplx

    raw = _load_scalar_archive(raw_path)
    name = _sequence_name(raw_path)
    body = raw["body"]
    source_params = body["params"]
    body_params = {
        key: np.asarray(value)[::stride]
        for key, value in source_params.items()
        if key != "fullpose" and isinstance(value, np.ndarray)
    }
    required = (
        "global_orient",
        "body_pose",
        "left_hand_pose",
        "right_hand_pose",
        "transl",
    )
    missing = [key for key in required if key not in body_params]
    if missing:
        raise KeyError(f"{name}: body parameters missing {missing}")
    frames = len(body_params["transl"])
    model = smplx.create(
        str(smplx_model_root),
        model_type="smplx",
        gender=str(_scalar(raw["gender"])).lower(),
        num_pca_comps=int(_scalar(raw["n_comps"])),
        v_template=_subject_template(raw_root, raw),
        batch_size=frames,
    ).to(device)
    tensor_params = {
        key: torch.from_numpy(value).float().to(device) for key, value in body_params.items()
    }
    model.eval()
    with torch.no_grad():
        body_output = model(**tensor_params)
    raw_joints = body_output.joints.detach().cpu().numpy()

    upright = Rotation.from_euler("x", -np.pi / 2, degrees=False)
    processed_trans = upright.apply(body_params["transl"])
    processed_global = (
        upright * Rotation.from_rotvec(body_params["global_orient"])
    ).as_rotvec()
    processed_params = dict(body_params)
    processed_params["global_orient"] = processed_global
    processed_params["transl"] = processed_trans
    processed_tensor_params = {
        key: torch.from_numpy(value).float().to(device)
        for key, value in processed_params.items()
    }
    # SMPL-X applies global orientation around the model root rather than the
    # world origin.  A direct rigid transform of the first forward result
    # therefore gives the wrong body translation.  Re-run with the transformed
    # root parameters exactly as InterAct's GRAB preprocessing does.
    with torch.no_grad():
        processed_output = model(**processed_tensor_params)
    processed_joints = processed_output.joints.detach().cpu().numpy()
    processed_vertices = processed_output.vertices.detach().cpu().numpy()

    object_params = raw["object"]["params"]
    raw_object_rotation = Rotation.from_rotvec(
        np.asarray(object_params["global_orient"])[::stride]
    )
    processed_object_rotation = upright * raw_object_rotation.inv()
    raw_object_trans = np.asarray(object_params["transl"])[::stride]
    object_trans_delta = upright.apply(raw_object_trans - raw_joints[:, 0])
    processed_object_trans = processed_joints[:, 0] + object_trans_delta
    if len(processed_object_trans) != frames:
        raise ValueError(f"{name}: body/object frame count mismatch")

    first_floor = float(processed_vertices[: min(30, frames), :, 1].min())
    processed_trans[:, 1] -= first_floor
    processed_joints[:, :, 1] -= first_floor
    processed_vertices[:, :, 1] -= first_floor
    processed_object_trans[:, 1] -= first_floor

    centroid = processed_joints[0, 0].copy()
    initial = Rotation.from_rotvec(processed_global[0]).as_matrix()
    denominator = float(np.hypot(initial[0, 0], initial[2, 0]))
    if denominator < 1e-8:
        raise ValueError(f"{name}: degenerate initial heading")
    cosine = initial[0, 0] / denominator
    sine = initial[2, 0] / denominator
    heading = np.eye(3, dtype=np.float32)
    heading[[0, 2, 0, 2], [0, 2, 2, 0]] = [cosine, cosine, -sine, sine]
    canonical_rotation_matrix = np.linalg.inv(heading).astype(np.float32)
    canonical_rotation = Rotation.from_matrix(canonical_rotation_matrix)

    pelvis = processed_joints[:, 0] - centroid
    centered_trans = processed_trans - centroid
    pelvis_original = pelvis - centered_trans
    canonical_trans = (
        (centered_trans + pelvis_original) @ canonical_rotation_matrix.T
        - pelvis_original
    )
    canonical_global = (
        canonical_rotation * Rotation.from_rotvec(processed_global)
    ).as_rotvec()
    canonical_object_trans = (
        (processed_object_trans - centroid) @ canonical_rotation_matrix.T
    )
    canonical_object_rotation = canonical_rotation * processed_object_rotation
    canonical_vertices = (
        (processed_vertices - centroid) @ canonical_rotation_matrix.T
    )

    object_name = str(_scalar(raw["obj_name"]))
    mesh_path = output_root / "objects" / object_name / f"{object_name}.obj"
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    object_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    object_minimum = np.inf
    for frame in range(min(30, frames)):
        rotation = canonical_object_rotation[frame].as_matrix()
        world = object_vertices @ rotation.T + canonical_object_trans[frame]
        object_minimum = min(object_minimum, float(world[:, 1].min()))
    human_minimum = float(canonical_vertices[: min(30, frames), :, 1].min())
    final_floor = min(human_minimum, object_minimum)
    canonical_trans[:, 1] -= final_floor
    canonical_object_trans[:, 1] -= final_floor

    contact = {}
    for key, value in raw["contact"].items():
        contact[key] = (
            np.asarray(value)[::stride]
            if isinstance(value, np.ndarray) and value.ndim > 0
            else value
        )
    canonical_params = {
        "global_orient": canonical_global.astype(np.float32),
        "body_pose": body_params["body_pose"].astype(np.float32),
        "left_hand_pose": body_params["left_hand_pose"].astype(np.float32),
        "right_hand_pose": body_params["right_hand_pose"].astype(np.float32),
        "transl": canonical_trans.astype(np.float32),
    }
    motion = {
        "n_comps": np.array(int(_scalar(raw["n_comps"]))),
        "gender": np.array(str(_scalar(raw["gender"]))),
        "sbj_id": np.array(str(_scalar(raw["sbj_id"]))),
        "n_frames": np.array(frames),
        "body": {"params": canonical_params, "vtemp": body["vtemp"]},
        "contact": contact,
    }
    sequence_dir = output_root / "sequences" / name
    sequence_dir.mkdir(parents=True, exist_ok=False)
    np.savez(sequence_dir / "motion.npz", **motion)
    np.savez(
        sequence_dir / "object.npz",
        angles=canonical_object_rotation.as_rotvec().astype(np.float32),
        trans=canonical_object_trans.astype(np.float32),
        name=np.array(object_name),
    )
    return name, frames


def command_build(args: argparse.Namespace) -> None:
    selection = _load_selection(args.selection.resolve())
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Run assets must be prepared before building sequences: {output_root}"
        )
    names = _selected_names(selection, args.limit)
    names = [name for index, name in enumerate(names) if index % args.num_shards == args.shard_index]
    report = {
        "schema_name": SCHEMA_NAME,
        "created_at": _now(),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "requested": len(names),
        "completed": [],
        "skipped": [],
        "failed": [],
    }
    for index, name in enumerate(names, 1):
        destination = output_root / "sequences" / name
        if destination.exists():
            if not args.resume:
                raise FileExistsError(f"Output sequence already exists: {destination}")
            report["skipped"].append(name)
            continue
        try:
            built_name, frames = _canonicalize_one(
                _raw_path(raw_root, name),
                raw_root,
                output_root,
                args.smplx_model_root.resolve(),
                args.device,
                args.stride,
            )
            report["completed"].append({"name": built_name, "frames": frames})
            print(f"[{index}/{len(names)}] built {built_name} ({frames} frames)", flush=True)
        except Exception as error:
            report["failed"].append({"name": name, "error": repr(error)})
            print(f"[{index}/{len(names)}] FAILED {name}: {error}", flush=True)
    report_path = output_root / "_reports" / f"canonical_shard_{args.shard_index:02d}.json"
    _write_json(report_path, report)
    print(
        f"Shard {args.shard_index}: completed={len(report['completed'])} "
        f"skipped={len(report['skipped'])} failed={len(report['failed'])}"
    )
    if report["failed"]:
        raise SystemExit(1)


def _motion_paths(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    return {
        path.parent.name: path
        for path in root.glob(f"*/{MOTION_FILENAME}")
        if path.is_file()
    }


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected Tensor in {path}, got {type(value)}")
    return value


def _alignment_probe(dexplore_root: Path, name: str, motion: torch.Tensor) -> dict[str, Any]:
    data_processing = dexplore_root / "data_processing"
    sys.path.insert(0, str(data_processing))
    try:
        from visualize_inspire_trajectory import (  # type: ignore
            InspireUrdfModel,
            NUM_DOFS,
            QPOS_START,
            TIP_LINKS,
        )
    finally:
        sys.path.pop(0)
    right_contact = torch.any(motion[:, 222:238] > 0, dim=1).nonzero().flatten()
    if len(right_contact) == 0:
        raise ValueError(f"{name}: no right-hand contact frame for alignment probe")
    frame = int(right_contact[len(right_contact) // 2])
    model = InspireUrdfModel(
        dexplore_root / "dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"
    )
    native = motion[frame, QPOS_START:QPOS_START + NUM_DOFS].numpy()
    transforms = model.link_transforms(model.qpos_to_urdf_order(native))
    tips = np.asarray([transforms[link][:3, 3] for link in TIP_LINKS])
    object_center = motion[frame, 198:201].numpy()
    distances = np.linalg.norm(tips - object_center[None, :], axis=1)
    minimum = float(distances.min())
    if minimum > 0.15:
        raise ValueError(
            f"{name}: alignment probe frame {frame} has fingertip/object-center distance {minimum:.6f} m"
        )
    return {
        "sequence": name,
        "frame": frame,
        "right_contact": True,
        "min_fingertip_to_object_center_m": minimum,
        "mean_fingertip_to_object_center_m": float(distances.mean()),
        "threshold_m": 0.15,
    }


def command_validate(args: argparse.Namespace) -> None:
    selection_path = args.selection.resolve()
    selection = _load_selection(selection_path)
    expected = _selected_names(selection, args.limit)
    expected_sorted = sorted(expected)
    run_root = args.run_root.resolve()
    dexplore_root = args.dexplore_root.resolve()
    canonical_root = run_root / "canonical"
    geometric_root = run_root / "geometric"
    rl_root = run_root / "rl"
    canonical_names = sorted(
        path.parent.name for path in canonical_root.glob("sequences/*/motion.npz")
    )
    geometric = _motion_paths(geometric_root)
    rl = _motion_paths(rl_root)
    if canonical_names != expected_sorted:
        raise ValueError(
            f"Canonical sequence set mismatch: expected={len(expected)} actual={len(canonical_names)}"
        )
    if sorted(geometric) != expected_sorted or sorted(rl) != expected_sorted:
        raise ValueError(
            "Geometric/RL sequence sets differ from the selected canonical set: "
            f"expected={len(expected)} geometric={len(geometric)} rl={len(rl)}"
        )
    total_frames = 0
    changed_object = 0
    changed_hand = 0
    max_unchanged_difference = 0.0
    unchanged_slices = ((0, 198), (205, 373), (391, OUTPUT_WIDTH))
    per_sequence = []
    for index, name in enumerate(expected, 1):
        motion = _load_scalar_archive(canonical_root / "sequences" / name / "motion.npz")
        frames = int(_scalar(motion["n_frames"]))
        geometric_tensor = _load_tensor(geometric[name])
        rl_tensor = _load_tensor(rl[name])
        if geometric_tensor.shape != (frames, OUTPUT_WIDTH):
            raise ValueError(f"{name}: geometric shape {tuple(geometric_tensor.shape)} != {(frames, OUTPUT_WIDTH)}")
        if rl_tensor.shape != geometric_tensor.shape:
            raise ValueError(f"{name}: RL shape {tuple(rl_tensor.shape)} != {tuple(geometric_tensor.shape)}")
        if not torch.isfinite(geometric_tensor).all() or not torch.isfinite(rl_tensor).all():
            raise ValueError(f"{name}: non-finite geometric or RL tensor")
        sequence_unchanged = 0.0
        for start, stop in unchanged_slices:
            difference = float((geometric_tensor[:, start:stop] - rl_tensor[:, start:stop]).abs().max())
            sequence_unchanged = max(sequence_unchanged, difference)
        if sequence_unchanged != 0.0:
            raise ValueError(f"{name}: RL exporter changed protected columns ({sequence_unchanged})")
        object_delta = float((geometric_tensor[:, 198:205] - rl_tensor[:, 198:205]).abs().max())
        hand_delta = float((geometric_tensor[:, 373:391] - rl_tensor[:, 373:391]).abs().max())
        changed_object += int(object_delta > 1e-7)
        changed_hand += int(hand_delta > 1e-7)
        max_unchanged_difference = max(max_unchanged_difference, sequence_unchanged)
        total_frames += frames
        per_sequence.append(
            {
                "name": name,
                "frames": frames,
                "object_max_abs_delta": object_delta,
                "hand_max_abs_delta": hand_delta,
            }
        )
        if index % 100 == 0 or index == len(expected):
            print(f"Validated {index}/{len(expected)} paired exports", flush=True)

    alignment = _alignment_probe(dexplore_root, expected[0], _load_tensor(geometric[expected[0]]))

    checkpoint = args.checkpoint.resolve()
    repo_root = Path(__file__).resolve().parents[5]
    summary = {
        "schema_name": SCHEMA_NAME,
        "created_at": _now(),
        "modification_version": MODIFICATION_VERSION,
        "selection_manifest": os.path.relpath(selection_path, run_root),
        "selection_sha256": _sha256(selection_path),
        "filter_counts": selection["counts"],
        "num_sequences": len(expected),
        "num_frames": total_frames,
        "tensor_shape": ["T", OUTPUT_WIDTH],
        "coordinate_contract": "DExplore canonical GRAB / Isaac Gym world frame; meters; quaternion xyzw",
        "pair_contract": {
            "geometric": "deterministic Inspire retargeting from canonical GRAB reference",
            "rl": "teacher-policy actual Inspire DOF and actual dynamic object pose",
            "protected_equal_columns": [[start, stop] for start, stop in unchanged_slices],
            "rl_hand_columns": [373, 391],
            "rl_object_pose_columns": [198, 205],
        },
        "changed_sequence_counts": {
            "object_pose": changed_object,
            "inspire_dof": changed_hand,
        },
        "max_protected_column_abs_difference": max_unchanged_difference,
        "geometric_alignment_probe": alignment,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "inputs": {
            "raw_grab_root": selection["raw_root"],
            "smplx_model_root": selection["smplx_model_root"],
        },
        "external_code": {
            "ref2dex": _git_state(repo_root),
            "dexplore": _git_state(dexplore_root),
        },
        "outputs": {
            "canonical": "canonical",
            "geometric": "geometric",
            "rl": "rl",
            "validation": "validation.json",
        },
        "per_sequence": per_sequence,
    }
    _write_json(run_root / "manifest.json", summary)
    _write_json(
        run_root / "validation.json",
        {
            "validated_at": summary["created_at"],
            "num_sequences": len(expected),
            "num_frames": total_frames,
            "changed_sequence_counts": summary["changed_sequence_counts"],
            "max_protected_column_abs_difference": max_unchanged_difference,
            "geometric_alignment_probe": alignment,
            "result": "SUPPORTED",
        },
    )
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.is_file()
        else {}
    )
    run_manifest.update(
        {
            "task": "ObjectInteractionCmv2",
            "modification_version": MODIFICATION_VERSION,
            "run_id": args.run_id,
            "run_status": "COMPLETED",
            "completed_at": summary["created_at"],
            "operation_category": ["data", "operation"],
            "seed": args.seed,
            "devices": args.gpus,
            "checkpoint": summary["checkpoint"],
            "metadata_snapshot": "manifest.json",
            "selection_manifest": summary["selection_manifest"],
            "commands_log": "commands.log",
            "outputs": summary["outputs"],
            "conclusion": "SUPPORTED",
        }
    )
    _write_json(run_root / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest, indent=2))


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--smplx-model-root", type=Path, required=True)
    parser.add_argument("--dexplore-root", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select", help="Derive the DExplore-compatible GRAB list")
    _add_shared_arguments(select)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=42)
    select.set_defaults(func=command_select)

    assets = subparsers.add_parser("assets", help="Prepare object meshes and subject skeletons")
    _add_shared_arguments(assets)
    assets.add_argument("--selection", type=Path, required=True)
    assets.add_argument("--output-root", type=Path, required=True)
    assets.add_argument("--device", default="cpu")
    assets.set_defaults(func=command_assets)

    init = subparsers.add_parser("init", help="Create a non-overwriting data-processing run")
    init.add_argument("--selection", type=Path, required=True)
    init.add_argument("--run-root", type=Path, required=True)
    init.add_argument("--dexplore-root", type=Path, required=True)
    init.add_argument("--checkpoint", type=Path, required=True)
    init.add_argument("--run-id", required=True)
    init.add_argument("--limit", type=int, default=None)
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--gpus", default="1,2,3")
    init.set_defaults(func=command_init)

    mark = subparsers.add_parser("mark", help="Record a terminal status for an incomplete run")
    mark.add_argument("--run-root", type=Path, required=True)
    mark.add_argument("--status", choices=("FAILED", "STOPPED"), required=True)
    mark.add_argument("--reason", required=True)
    mark.set_defaults(func=command_mark)

    build = subparsers.add_parser("build", help="Build one disjoint canonical sequence shard")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--raw-root", type=Path, required=True)
    build.add_argument("--smplx-model-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--stride", type=int, default=4)
    build.add_argument("--device", default="cuda")
    build.add_argument("--num-shards", type=int, default=1)
    build.add_argument("--shard-index", type=int, default=0)
    build.add_argument("--resume", action="store_true")
    build.set_defaults(func=command_build)

    validate = subparsers.add_parser("validate", help="Validate paired geometric/RL exports")
    validate.add_argument("--selection", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--dexplore-root", type=Path, required=True)
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--limit", type=int, default=None)
    validate.add_argument("--seed", type=int, default=42)
    validate.add_argument("--gpus", default="1,2,3")
    validate.set_defaults(func=command_validate)

    args = parser.parse_args()
    if getattr(args, "num_shards", 1) < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= getattr(args, "shard_index", 0) < getattr(args, "num_shards", 1):
        parser.error("--shard-index must be in [0, num_shards)")
    args.func(args)


if __name__ == "__main__":
    main()
