#!/usr/bin/env python3
"""Read-only bilateral MANO trajectory viewer for GRAB and ARCTIC."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    import viser

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8145)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--grab", type=Path, default=Path("data/processed_data/oicm_v1_4_raw/grab_mano_30hz/s1/airplane_fly_1"))
    ap.add_argument("--arctic", type=Path, default=Path("data/processed_data/oicm_v1_4_raw/arctic_mano_30hz/s01/box_use_01"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {"GRAB / s1/airplane_fly_1": args.grab.resolve(), "ARCTIC / s01/box_use_01": args.arctic.resolve()}
    data = {}
    for label, root in paths.items():
        with np.load(root / "shared.npz") as z: shared = {k: np.asarray(z[k]) for k in z.files}
        with np.load(root / "left.npz") as z: left = {k: np.asarray(z[k]) for k in z.files}
        with np.load(root / "right.npz") as z: right = {k: np.asarray(z[k]) for k in z.files}
        data[label] = {"root": root, "shared": shared, "left": left, "right": right}
    manifest = {"schema_name": "ref2dex_run_manifest_v1", "task": "ObjectInteractionCm", "operation": "bilateral_mano_trajectory_visualization", "run_id": args.output.name, "run_status": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "modification_version": "V1.4.6", "inputs": {k: str(v) for k, v in paths.items()}, "viewer": f"http://{args.host}:{args.port}", "output": str(args.output.resolve()), "conclusion": "INCONCLUSIVE"}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    with server.gui.add_folder("MANO 轨迹"):
        sequence = server.gui.add_dropdown("数据集 / 轨迹", options=tuple(data), initial_value=next(iter(data)))
        frame = server.gui.add_slider("帧", min=0, max=len(data[next(iter(data))]["shared"]["raw_frame_id"]) - 1, step=1, initial_value=0)
        status = server.gui.add_markdown("")
    handles = []

    def render(_: object = None) -> None:
        nonlocal handles
        for handle in handles:
            handle.remove()
        handles = []
        item = data[sequence.value]
        idx = min(int(frame.value), len(item["shared"]["raw_frame_id"]) - 1)
        for side, color in (("left", (40, 130, 255)), ("right", (40, 210, 100))):
            hand = item[side]
            points = np.asarray(hand["hand_points_world"][idx], dtype=np.float32)
            handles.append(server.scene.add_point_cloud(f"{side}_mano_points", points=points, colors=color, point_size=0.004))
            if "hand_mesh_vertices_world" in hand:
                handles.append(server.scene.add_mesh_simple(f"{side}_mano_mesh", vertices=np.asarray(hand["hand_mesh_vertices_world"][idx], dtype=np.float32), faces=np.asarray(hand["hand_mesh_faces"], dtype=np.int32), color=color, opacity=0.28))
        obj = item["shared"]
        handles.append(server.scene.add_point_cloud("object_points", points=np.asarray(obj["obj_points_world"][idx], dtype=np.float32), colors=(190, 190, 190), point_size=0.002))
        status.content = f"raw frame: `{int(obj['raw_frame_id'][idx])}`  |  30 Hz  |  left/right MANO: `{item['left']['hand_points_world'].shape[1]}/{item['right']['hand_points_world'].shape[1]}` points  |  object: `{obj['obj_points_world'].shape[1]}` points"

    @sequence.on_update
    def _(_: object) -> None:  # type: ignore[no-untyped-def]
        frame.max = len(data[sequence.value]["shared"]["raw_frame_id"]) - 1
        render()

    @frame.on_update
    def _(_: object) -> None:  # type: ignore[no-untyped-def]
        render()

    render()
    print(json.dumps({"viewer": f"http://{args.host}:{args.port}", "run_id": args.output.name, "sequences": list(data)}, ensure_ascii=False), flush=True)
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        manifest["run_status"] = "STOPPED"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
