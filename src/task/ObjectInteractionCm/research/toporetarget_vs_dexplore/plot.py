"""Render the fixed first frame of each comparison with actual visual meshes."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inspire_adapter import NATIVE_ORDER


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    helpers = Path(__file__).parent.parent / "oakink2_inspire_pilot/plot.py"
    spec = importlib.util.spec_from_file_location("mesh_plot", helpers)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from dex_retargeting.retargeting_config import RetargetingConfig
    fig = plt.figure(figsize=(12, 6*len(args.runs)))
    for row, run in enumerate(args.runs):
        config = json.loads((run / "config.json").read_text())
        data = np.load(config["input"], allow_pickle=False)
        source_report = json.loads((Path(config["input"]).parent / "source_report.json").read_text())
        side = next(iter(source_report["sides"]))
        compared = np.load(run / f"{side}_comparison.npz", allow_pickle=False)
        asset = Path(config["assets"]) / f"inspire_{side}_full.urdf"
        rt = RetargetingConfig.from_dict({"type":"position", "urdf_path":str(asset.resolve()),
             "target_link_names":[f"{f}_tip" for f in ("thumb","index","middle","ring","pinky")],
             "target_link_human_indices":[4,8,12,16,20], "target_joint_names":NATIVE_ORDER,
             "add_dummy_free_joint":False,"ignore_mimic_joint":True}).build()
        objects = module.transformed(data["object_vertices"], data["object_poses"][0])
        mano = data[f"{side}_mano_vertices"][0]
        geometry_by_method = []
        for method in ("dex", "final"):
            native = compared[f"{method}_native18"][0]
            rt.optimizer.robot.compute_forward_kinematics(native[[NATIVE_ORDER.index(n) for n in rt.joint_names]])
            poses = [rt.optimizer.robot.get_link_pose(i) for i in range(len(rt.optimizer.robot.link_names))]
            predicted = np.stack([poses[list(rt.optimizer.robot.link_names).index(f"{finger}_tip")][:3,3]
                                  for finger in ("thumb","index","middle","ring","pinky")])
            np.testing.assert_allclose(predicted, compared[f"{method}_points"][0,[4,8,12,16,20]],atol=1e-6)
            geometry_by_method.append(module.robot_meshes(asset, list(rt.optimizer.robot.link_names), poses))
        extent = np.concatenate([objects,mano] + [v for meshes in geometry_by_method for v,_ in meshes])
        center = (extent.min(0)+extent.max(0))/2
        radius = max(np.ptp(extent,axis=0).max()*.55,.1)
        for col,(label,meshes) in enumerate(zip(("Dexplore-style position", "Topo constrained final"),geometry_by_method)):
            ax = fig.add_subplot(len(args.runs),2,row*2+col+1,projection="3d")
            module.add_mesh(ax,objects,data["object_faces"],"#848a92",.25)
            sampled = objects[np.linspace(0,len(objects)-1,min(3000,len(objects)),dtype=int)]
            ax.scatter(*sampled.T,s=1,c="#59616d",alpha=.5)
            module.add_mesh(ax,mano,data[f"{side}_mano_faces"],"#ee9b45",.22)
            for v,f in meshes:
                module.add_mesh(ax,v,f,"#2375b9" if col==0 else "#24946b",.4)
            for points,color in [(compared["source_points"][0],"#d47c22"),
                                 (compared["dex_points" if col==0 else "final_points"][0],"#124b83" if col==0 else "#09633d")]:
                for indices in ([0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]):
                    ax.plot(*points[indices].T,c=color,lw=1.8,marker="o",markersize=2)
            for setter,mid in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):
                setter(mid-radius,mid+radius)
            ax.set_box_aspect([1,1,1]); ax.view_init(elev=28,azim=-65)
            ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",
                   title=f"{side} | frame {data['source_frame_id'][0]} | {label}")
    fig.suptitle("Inspire retargeting pilot | orange: source MANO; gray: object\n"
                 "Same asset within each row; 12 independent finger joints; geometry smoke only.")
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(args.output,dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
