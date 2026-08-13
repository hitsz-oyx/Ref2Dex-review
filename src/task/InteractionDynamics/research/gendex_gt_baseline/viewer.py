"""V20.7 GT MANO / old GT-Y Allegro / GenDex GT-CMap Allegro 静态 Viewer。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import viser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0"); parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args(); row = torch.load(args.cache, map_location="cpu")
    array = lambda value: value.numpy() if torch.is_tensor(value) else np.asarray(value)
    obj=array(row["object_vertices"]); faces=array(row["object_faces"]); diameter=np.linalg.norm(np.ptp(obj, axis=0))
    offsets=[np.array([-1.5*diameter,0,0]), np.zeros(3), np.array([1.5*diameter,0,0])]
    server=viser.ViserServer(host=args.host, port=args.port)
    cmap=array(row["contact_values"]); color=np.stack((255*cmap, 40*np.ones_like(cmap), 255*(1-cmap)), -1).astype(np.uint8)
    for name, offset in zip(("gt_mano", "old_gty", "gendex"), offsets):
        server.scene.add_mesh_simple(f"/world/{name}/object", obj+offset, faces, color=(180,180,180), opacity=.45)
        server.scene.add_point_cloud(f"/world/{name}/contact_map", array(row["object_points"])+offset,
            colors=color, point_size=.003)
    server.scene.add_mesh_simple("/world/gt_mano/hand", array(row["gt_vertices"])+offsets[0],
        array(row["mano_faces"]), color=(80,220,120))
    server.scene.add_mesh_simple("/world/old_gty/hand", array(row["old_gty_vertices"])+offsets[1],
        array(row["old_gty_faces"]), color=(255,165,65))
    initial_hand = server.scene.add_mesh_simple("/world/gendex/initial", array(row["gendex_initial_vertices"])+offsets[2],
        array(row["gendex_faces"]), color=(160,160,160), opacity=.2, visible=False)
    server.scene.add_mesh_simple("/world/gendex/hand", array(row["gendex_final_vertices"])+offsets[2],
        array(row["gendex_faces"]), color=(180,100,255))
    show_initial = server.gui.add_checkbox("Show Initial GenDex Robot", False)
    @show_initial.on_update
    def _(_): initial_hand.visible = show_initial.value
    server.gui.add_markdown(f"**GT MANO | old GT-Y Allegro | GenDex GT-CMap Allegro**  \n"
        f"frame={row['stable_frame']} | high CMap points={row['map_sanity']['high_contact_count']}  \n"
        f"GenDex total: {row['initial_energy']['total']:.4f} → {row['final_energy']['total']:.4f} | "
        f"contact MAE: {row['initial_energy']['contact_mae']:.4f} → {row['final_energy']['contact_mae']:.4f}  \n"
        f"CMap corr={row['realization']['contact_correlation']:.3f} | high-C IoU={row['realization']['high_contact_iou']:.3f}  \n"
        f"full-mesh penetration={row['full_mesh_penetration_mm']} mm")
    print(f"V20.7 GenDex Viewer: http://localhost:{args.port}", flush=True)
    while True: time.sleep(3600)


if __name__ == "__main__": main()
