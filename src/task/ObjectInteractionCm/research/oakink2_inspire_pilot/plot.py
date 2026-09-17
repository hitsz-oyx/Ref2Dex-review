"""Plot exported MANO, Inspire visual-mesh FK and original object geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def add_mesh(ax, vertices, faces, color, alpha, max_faces=2500):
    chosen = np.arange(len(faces))
    if len(chosen) > max_faces:
        chosen = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
    triangles = vertices[faces[chosen]]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=color, alpha=alpha, linewidths=0))


def transformed(vertices, pose):
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def robot_meshes(urdf, link_names, poses):
    result = []
    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):
        name = link.get("name")
        if name not in link_names:
            continue
        for visual in link.findall("visual"):
            element = visual.find("geometry/mesh")
            if element is None:
                raise ValueError(f"Unsupported visual geometry: {name}")
            mesh = trimesh.load(urdf.parent / element.get("filename"), force="mesh", process=False)
            vertices = np.asarray(mesh.vertices) * np.fromstring(element.get("scale", "1 1 1"), sep=" ")
            origin = visual.find("origin")
            local = np.eye(4)
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            result.append((transformed(vertices, poses[link_names.index(name)] @ local), np.asarray(mesh.faces)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    record = report["sequences"][0]
    data = np.load(record["output"]["path"], allow_pickle=False)
    candidates = np.linspace(0, len(data["source_frame_id"]) - 1, 40, dtype=int)
    # Pick a nearby hand/object frame for inspection, without filtering export.
    proximity = []
    for frame in candidates:
        obj = data["object_sample_vertices_world"][frame].reshape(-1, 3)
        hands = np.concatenate([data[side + "_mano_vertices_world"][frame] for side in ("left", "right")])
        proximity.append(cKDTree(obj).query(hands)[0].min())
    frame = int(candidates[np.argmin(proximity)])
    fig = plt.figure(figsize=(13, 10))
    for column, side in enumerate(("left", "right")):
        ax = fig.add_subplot(2, 2, column + 1, projection="3d")
        mano = data[side + "_mano_vertices_world"][frame]
        add_mesh(ax, mano, data[side + "_mano_faces"], "#eb9a41", .28)
        robot = robot_meshes(Path(record["hands"][side]["urdf"]["path"]),
                            data[side + "_inspire_link_names"].tolist(),
                            data[side + "_inspire_link_poses_world"][frame])
        for vertices, faces in robot:
            add_mesh(ax, vertices, faces, "#427fcb", .4)
        for i, obj in enumerate(record["objects"]):
            mesh = trimesh.load(obj["mesh"]["path"], force="mesh", process=False)
            points = transformed(np.asarray(mesh.vertices), data["object_poses_world"][frame, i])
            add_mesh(ax, points, np.asarray(mesh.faces), "#9ca3af", .2)
        target = data[side + "_mano_joints_world"][frame, [4, 8, 12, 16, 20]]
        predicted = data[side + "_inspire_tips_world"][frame]
        ax.scatter(*target.T, c="#e68726", s=20, label="MANO tips")
        ax.scatter(*predicted.T, c="#1b56a5", s=20, label="Inspire tip FK")
        for a, b in zip(target, predicted):
            ax.plot(*np.stack([a, b]).T, c="#cf3333", lw=1)
        geometry = np.concatenate([mano] + [v for v, _ in robot])
        center = (geometry.min(0) + geometry.max(0)) / 2
        half = max(np.ptp(geometry, axis=0).max() * .65, .12)
        for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(mid - half, mid + half)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=25, azim=-65)
        ax.set(xlabel="native X (m)", ylabel="native Y (m)", zlabel="native Z (m)",
               title=f"{side.capitalize()}: source frame {data['source_frame_id'][frame]}")
        ax.legend(fontsize=8)
        curve = fig.add_subplot(2, 2, column + 3)
        errors = data[side + "_inspire_tip_error_m"] * 1000
        ids = data["source_frame_id"]
        curve.plot(ids, errors.mean(1), lw=1, label="Mean of 5 tips")
        curve.plot(ids, errors.max(1), lw=.7, alpha=.5, label="Max of 5 tips")
        curve.axvline(ids[frame], color="gray", linestyle="--", lw=.7)
        curve.set(xlabel="Original mocap frame ID", ylabel="Tip error (mm)", title=f"{side.capitalize()} geometric fit")
        curve.legend()
        curve.grid(alpha=.2)
    fig.suptitle("OakInk2 bilateral Inspire geometric pilot | orange: MANO; blue: Inspire; gray: objects\n"
                 "Native world preserved. Visual/FK inspection; collision and contact preservation are not validated.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(args.run / "geometry_review.png", dpi=160)
    plt.close(fig)
    (args.run / "plot_source.py").write_bytes(Path(__file__).read_bytes())
    print(args.run / "geometry_review.png")


if __name__ == "__main__":
    main()
