"""Render a small, headless preview of an InterAct GRAB clip.

The MP4 shows the canonical SMPL-X body and object.  A companion PNG compares
the 18 native Inspire DOFs from geometric retargeting and the RL checkpoint
rollout.  It is intentionally GRAB-only so it does not require SMPL-H or
``human_body_prior``.
"""

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def _look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = eye
    return pose


def _load_npz_dict(path):
    with np.load(path, allow_pickle=True) as z:
        return {key: (z[key].item() if z[key].shape == () else z[key])
                for key in z.files}


def load_canonical(interact_root, sequence, model_root, max_frames):
    import smplx

    seq_root = Path(interact_root) / "data/grab/sequences_canonical" / sequence
    human = _load_npz_dict(seq_root / "human.npz")
    obj = _load_npz_dict(seq_root / "object.npz")
    poses = np.asarray(human["poses"], dtype=np.float32)
    trans = np.asarray(human["trans"], dtype=np.float32)
    n = min(len(poses), int(max_frames)) if max_frames > 0 else len(poses)
    poses, trans = poses[:n], trans[:n]
    gender = str(human["gender"])
    vtemp = np.asarray(human["vtemp"], dtype=np.float32)

    # InterAct's GRAB canonical files keep 24-D PCA hand poses (the same
    # convention as the raw GRAB archive), rather than 45-D axis-angle hands.
    model = smplx.create(
        str(model_root), model_type="smplx", gender=gender, use_pca=True,
        num_pca_comps=24, v_template=vtemp, batch_size=n, ext="npz",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pose = torch.from_numpy(poses).to(device)
    trans_t = torch.from_numpy(trans).to(device)
    zeros = torch.zeros((n, 3), device=device)
    expression = torch.zeros((n, 10), device=device)
    with torch.no_grad():
        output = model(
            global_orient=pose[:, :3], body_pose=pose[:, 3:66],
            left_hand_pose=pose[:, 66:90], right_hand_pose=pose[:, 90:114],
            jaw_pose=zeros, leye_pose=zeros, reye_pose=zeros,
            expression=expression, transl=trans_t,
        )
    verts = output.vertices.detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int64)

    object_name = str(obj["name"])
    object_path = Path(interact_root) / "data/grab/objects" / object_name / f"{object_name}.obj"
    object_mesh = trimesh.load(str(object_path), force="mesh", process=False)
    object_local = np.asarray(object_mesh.vertices, dtype=np.float32)
    object_faces = np.asarray(object_mesh.faces, dtype=np.int64)
    angles = np.asarray(obj["angles"], dtype=np.float64)[:n]
    object_trans = np.asarray(obj["trans"], dtype=np.float64)[:n]
    rotations = Rotation.from_rotvec(angles).as_matrix()
    object_world = np.einsum("vj,tij->tvi", object_local, rotations) + object_trans[:, None, :]
    return verts, faces, object_world.astype(np.float32), object_faces, object_name


def render_mesh_video(verts, faces, object_world, object_faces, output, fps=30):
    import pyrender

    all_points = np.concatenate([verts.reshape(-1, 3), object_world.reshape(-1, 3)], axis=0)
    lo, hi = np.nanpercentile(all_points, [1, 99], axis=0)
    center = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    span = max(span, 0.5)
    floor_y = float(lo[1] - 0.03 * span)

    scene = pyrender.Scene(bg_color=[245, 245, 245, 255], ambient_light=[0.35, 0.35, 0.35])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4.0 / 3.0)
    eye = center + np.array([1.8 * span, 1.25 * span, 1.8 * span])
    scene.add(camera, pose=_look_at(eye, center))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
              pose=_look_at(eye, center))
    floor = trimesh.creation.box(extents=[3.0 * span, 0.03 * span, 3.0 * span])
    floor.apply_translation([center[0], floor_y, center[2]])
    scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False), name="floor")
    renderer = pyrender.OffscreenRenderer(640, 480)
    # Stream raw RGB frames to the system ffmpeg binary.  This avoids relying
    # on the optional ``imageio-ffmpeg`` Python plugin, which is not installed
    # in the default graspenv.
    ffmpeg = subprocess.Popen([
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "640x480", "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output),
    ], stdin=subprocess.PIPE)
    human_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.95, 0.72, 0.22, 1.0], metallicFactor=0.0, roughnessFactor=0.8)
    object_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.88, 0.20, 0.52, 1.0], metallicFactor=0.0, roughnessFactor=0.75)
    for frame_id in range(len(verts)):
        human = trimesh.Trimesh(vertices=verts[frame_id], faces=faces, process=False)
        obj = trimesh.Trimesh(vertices=object_world[frame_id], faces=object_faces, process=False)
        human_node = scene.add(pyrender.Mesh.from_trimesh(human, material=human_material, smooth=False),
                               name="human")
        object_node = scene.add(pyrender.Mesh.from_trimesh(obj, material=object_material, smooth=False),
                                name="object")
        color, _ = renderer.render(scene)
        image = Image.fromarray(color).convert("RGB")
        ImageDraw.Draw(image).text((10, 10), f"frame {frame_id:04d}", fill=(25, 25, 25))
        ffmpeg.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        scene.remove_node(human_node)
        scene.remove_node(object_node)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing preview video")
    renderer.delete()


def render_qpos_plot(geometric_path, rl_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geometric = torch.load(geometric_path, map_location="cpu", weights_only=True).numpy()
    rl = torch.load(rl_path, map_location="cpu", weights_only=True).numpy()
    q0, q1 = 245 + 32 * 4, 245 + 32 * 4 + 18
    n = min(len(geometric), len(rl))
    fig, axes = plt.subplots(6, 3, figsize=(15, 12), sharex=True)
    for dof, ax in enumerate(axes.flat):
        ax.plot(geometric[:n, q0 + dof], color="#1976d2", lw=1.0, label="geometric" if dof == 0 else None)
        ax.plot(rl[:n, q0 + dof], color="#d32f2f", lw=1.0, label="RL checkpoint" if dof == 0 else None)
        ax.set_title(f"Inspire DOF {dof}")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[-1, 1].set_xlabel("frame (30 Hz)")
    fig.suptitle("Geometric retargeting vs RL checkpoint rollout")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--interact-root", default="/home2/wyy/oyx_ws/InterAct")
    parser.add_argument("--model-root", default="/home2/wyy/oyx_ws/Ref2Dex/data/raw_data/ARCTIC/arctic/models")
    parser.add_argument("--geometric-root", default="/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric")
    parser.add_argument("--rl-root", default="/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl")
    parser.add_argument("--output-dir", default="/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/visualization_preview")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = output_dir / f"{args.sequence}_human_object.mp4"
    qpos = output_dir / f"{args.sequence}_qpos_compare.png"
    verts, faces, object_world, object_faces, object_name = load_canonical(
        args.interact_root, args.sequence, args.model_root, args.max_frames)
    render_mesh_video(verts, faces, object_world, object_faces, video)
    render_qpos_plot(
        Path(args.geometric_root) / args.sequence / "interaction_hand_inspire.pt",
        Path(args.rl_root) / args.sequence / "interaction_hand_inspire.pt",
        qpos,
    )
    print(f"Wrote {video}")
    print(f"Wrote {qpos}")
    print(f"Object: {object_name}; frames: {len(verts)}")


if __name__ == "__main__":
    main()
