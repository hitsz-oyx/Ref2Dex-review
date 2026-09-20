"""Prepare raw GRAB archives for :mod:`convert_grab`.

The public Dexplore converter consumes the simulation-oriented GRAB layout
(``sequences/<id>/motion.npz`` and ``object.npz``).  The GRAB download itself
only contains ``grab/<subject>/*.npz``.  This small adapter bridges that gap
without requiring the full InterAct training stack: it applies the same
upright coordinate transform used by InterAct, downsamples 120 Hz motion to
30 Hz, and writes the minimal fields consumed by Dexplore.

The generated directory is a derived cache.  It contains symlinks to the
licensed GRAB meshes and should not be committed or redistributed.
"""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as sRot


def _scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _params_to_torch(params, device):
    # ``fullpose`` is redundant and is not accepted together with the
    # individual pose arguments by smplx.
    return {
        key: torch.from_numpy(value).float().to(device)
        for key, value in params.items()
        if isinstance(value, np.ndarray) and key != "fullpose"
    }


def _load_raw(path):
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key].item() for key in archive.files}


def _save_scalar_dict(path, values):
    # np.savez stores dictionaries as object scalars, which is the convention
    # used by the original GRAB files and parse_npz in convert_grab.py.
    np.savez(path, **{key: np.array(value, dtype=object) if isinstance(value, dict) else value
                      for key, value in values.items()})


def _mesh_link(raw_root, object_name, object_root):
    source = Path(raw_root) / "objects" / object_name / "mesh.obj"
    if not source.exists():
        # Some GRAB extractions retain the official tools path instead.
        source = Path(raw_root) / "tools" / "object_meshes" / "contact_meshes" / f"{object_name}.ply"
    if not source.exists():
        raise FileNotFoundError(f"Object mesh not found for {object_name}: {source}")
    dest_dir = Path(object_root) / object_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{object_name}.obj"
    if dest.exists() or dest.is_symlink():
        return
    # trimesh can read PLY but Isaac Gym and the converter consistently use OBJ.
    if source.suffix.lower() == ".obj":
        dest.symlink_to(source)
    else:
        mesh = trimesh.load(str(source), force="mesh", process=False)
        mesh.export(str(dest))


def _make_skeleton_xml(model, output_path, mapping):
    """Write a minimal 52-joint MuJoCo hierarchy for skeleton_utils."""
    import xml.etree.ElementTree as ET

    # The pose vector used by Dexplore is root + 21 body joints + 15 joints per
    # hand.  Mapping is the same source->MuJoCo ordering used in smpl_constants.
    # SMPL-X's model uses 55 joint indices, while the pose vector consumed by
    # Dexplore is a compact 52-joint vector (22 body + 15 left hand + 15
    # right hand).  Build the compact parent table directly from the model's
    # kinematic tree instead of hard-coding original indices; this also keeps
    # the generated XML compatible with different SMPL-X releases.
    selected = list(range(22)) + list(range(25, 55))
    model_parents = [int(x) for x in model.parents.detach().cpu().tolist()]
    parents_source = []
    for source in selected:
        parent = model_parents[source]
        parents_source.append(-1 if parent < 0 else selected.index(parent))
    with torch.no_grad():
        # ``smplx.create(..., batch_size=n_frames)`` registers default betas
        # and expression tensors at that batch size.  Use the model's batch
        # size here instead of a singleton batch to avoid implicit broadcast
        # mismatches in the landmark layer.
        batch_size = int(getattr(model, "batch_size", 1))
        zero = torch.zeros((batch_size, 3), device=next(model.parameters()).device)
        body = torch.zeros((batch_size, 63), device=zero.device)
        hand = torch.zeros((batch_size, model.num_pca_comps), device=zero.device)
        out = model(global_orient=zero, body_pose=body,
                    left_hand_pose=hand, right_hand_pose=hand,
                    transl=zero, betas=torch.zeros((batch_size, 10), device=zero.device),
                    expression=torch.zeros((batch_size, 10), device=zero.device),
                    jaw_pose=zero, leye_pose=zero, reye_pose=zero)
        source_joints = torch.cat([out.joints[:, :22], out.joints[:, 25:55]], dim=1)[0]
        target_joints = source_joints[mapping].cpu().numpy()

    parent_target = []
    for source_index in mapping:
        source_parent = parents_source[source_index]
        parent_target.append(-1 if source_parent < 0 else mapping.index(source_parent))

    root = ET.Element("mujoco", attrib={"model": "smplx_grab"})
    worldbody = ET.SubElement(root, "worldbody")
    names = [f"joint_{i:02d}" for i in range(len(mapping))]

    def add_body(parent_xml, index):
        parent = parent_target[index]
        if parent < 0:
            pos = np.zeros(3, dtype=np.float32)
            xml_parent = parent_xml
        else:
            pos = target_joints[index] - target_joints[parent]
            xml_parent = body_xml[parent]
        body_xml[index] = ET.SubElement(xml_parent, "body", attrib={
            "name": names[index], "pos": "%.9g %.9g %.9g" % tuple(pos)})
        for child in children[index]:
            add_body(parent_xml, child)

    children = [[] for _ in mapping]
    for index, parent in enumerate(parent_target):
        if parent >= 0:
            children[parent].append(index)
    body_xml = [None] * len(mapping)
    for root_index in [i for i, p in enumerate(parent_target) if p < 0]:
        add_body(worldbody, root_index)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def prepare_sequence(raw_path, out_root, model_root, skeleton_root, device, stride):
    import smplx
    from smpl_constants import SMPL_2_MUJOCO

    raw = _load_raw(raw_path)
    body = raw["body"]
    body_params = body["params"]
    gender = str(_scalar(raw["gender"])).lower()
    sbj_id = str(_scalar(raw["sbj_id"]))
    n_comps = int(_scalar(raw["n_comps"]))
    n_raw = int(_scalar(raw["n_frames"]))
    if n_raw != len(body_params["transl"]):
        raise ValueError(f"{raw_path}: n_frames={n_raw} but transl has {len(body_params['transl'])}")

    # Cache one upright skeleton per subject; the subject template is already
    # encoded in the GRAB body mesh, so no shape betas are needed.
    sbj_vtemp_path = body["vtemp"]
    if sbj_vtemp_path.startswith("tools/subject_meshes/"):
        sbj_vtemp_path = sbj_vtemp_path[len("tools/subject_meshes/"):]
    sbj_vtemp_file = Path(raw_path).parents[2] / "tools" / "subject_meshes" / sbj_vtemp_path
    if not sbj_vtemp_file.exists():
        sbj_vtemp_file = Path(raw_path).parents[2] / "tools" / sbj_vtemp_path
    sbj_vtemp = np.asarray(trimesh.load(str(sbj_vtemp_file), process=False).vertices, dtype=np.float32)

    model = smplx.create(model_root, model_type="smplx", gender=gender,
                         num_pca_comps=n_comps, v_template=sbj_vtemp,
                         batch_size=n_raw).to(device)
    params = _params_to_torch(body_params, device)
    raw_obj = raw["object"]["params"]
    raw_table = raw["table"]["params"]
    with torch.no_grad():
        raw_out = model(**params)
        raw_pelvis = raw_out.joints[:, 0].detach().cpu().numpy()
        rx_minus = sRot.from_euler("x", -np.pi / 2, degrees=False)
        proc_global = (rx_minus * sRot.from_rotvec(body_params["global_orient"])).as_rotvec()
        proc_trans = rx_minus.apply(body_params["transl"])
        proc_params = copy.deepcopy(body_params)
        proc_params["global_orient"] = proc_global
        proc_params["transl"] = proc_trans
        proc_torch = _params_to_torch(proc_params, device)
        proc_out = model(**proc_torch)
        proc_verts = proc_out.vertices
        proc_pelvis = proc_out.joints[:, 0].detach().cpu().numpy()
        diff_fix = proc_verts[..., 1].amin().item()

    proc_trans = proc_trans.copy()
    proc_trans[:, 1] -= diff_fix
    obj_delta = rx_minus.apply(raw_obj["transl"] - raw_pelvis)
    proc_obj_trans = proc_pelvis + obj_delta
    proc_obj_trans[:, 1] -= diff_fix
    proc_obj_angles = (rx_minus * sRot.from_rotvec(raw_obj["global_orient"]).inv()).as_rotvec()

    name = f"{sbj_id}_{Path(raw_path).stem}"
    seq_dir = Path(out_root) / "sequences" / name
    seq_dir.mkdir(parents=True, exist_ok=True)
    B = (n_raw + stride - 1) // stride
    # Keep contacts aligned with the processed 30 Hz motion.  The converter
    # accepts both 30 Hz and legacy 120 Hz contacts.
    contact = raw["contact"]
    contact_ds = {key: value[::stride] if isinstance(value, np.ndarray) and value.ndim > 0 else value
                  for key, value in contact.items()}
    motion = {
        "n_comps": np.array(n_comps), "gender": np.array(gender),
        "sbj_id": np.array(sbj_id), "n_frames": np.array(B),
        "body": {"params": {key: value[::stride] if isinstance(value, np.ndarray) and value.ndim > 0 else value
                              for key, value in proc_params.items()},
                  "vtemp": body["vtemp"]},
        "contact": contact_ds,
    }
    _save_scalar_dict(seq_dir / "motion.npz", motion)
    _save_scalar_dict(seq_dir / "object.npz", {
        "angles": proc_obj_angles[::stride], "trans": proc_obj_trans[::stride],
        "name": np.array(str(_scalar(raw["obj_name"]))),
    })
    _mesh_link(Path(raw_path).parents[2], str(_scalar(raw["obj_name"])), Path(out_root) / "objects")

    skeleton_path = Path(skeleton_root) / f"smplx_grab_{sbj_id}.xml"
    # Rewriting is intentional: earlier derived caches may have been made
    # with a different SMPL-X parent convention.
    _make_skeleton_xml(model, skeleton_path, SMPL_2_MUJOCO)
    return name, B


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-grab-root", required=True,
                        help="GRAB root containing grab/ and tools/")
    parser.add_argument("--output-root", required=True,
                        help="Derived root to create (sequences/ and objects/)")
    parser.add_argument("--smplx-model-root", required=True,
                        help="Directory containing smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz")
    parser.add_argument("--skeleton-root", required=True)
    parser.add_argument("--filter", default=None)
    parser.add_argument("--subject", default=None,
                        help="Only process one subject directory (e.g. s1)")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    raw_root = Path(args.raw_grab_root)
    # ``convert_grab.py`` resolves the subject template through
    # ``<prepared-root>/tools``.  Keep that path as a symlink to the original
    # GRAB tools directory so the derived cache never duplicates licensed
    # subject meshes.
    prepared_tools = Path(args.output_root) / "tools"
    source_tools = raw_root / "tools"
    if source_tools.exists() and not prepared_tools.exists():
        prepared_tools.parent.mkdir(parents=True, exist_ok=True)
        prepared_tools.symlink_to(source_tools, target_is_directory=True)
    files = sorted(raw_root.glob("grab/s*/*.npz"))
    if args.filter:
        files = [path for path in files if args.filter in path.name]
    if args.subject:
        files = [path for path in files if path.parent.name == args.subject]
    if not files:
        raise FileNotFoundError(f"No raw GRAB sequences found below {raw_root / 'grab'}")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ok = 0
    for index, raw_path in enumerate(files, 1):
        try:
            name, frames = prepare_sequence(raw_path, args.output_root, args.smplx_model_root,
                                            args.skeleton_root, device, args.stride)
            ok += 1
            print(f"[{index}/{len(files)}] prepared {name} ({frames} frames)", flush=True)
        except Exception as exc:
            print(f"[{index}/{len(files)}] FAILED {raw_path}: {exc}", flush=True)
    print(f"Prepared {ok}/{len(files)} sequences")


if __name__ == "__main__":
    main()
