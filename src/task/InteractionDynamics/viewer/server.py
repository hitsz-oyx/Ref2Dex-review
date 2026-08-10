#!/usr/bin/env python
"""InteractionDynamics 点流与 MANO inverse 浏览器服务。"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE
while not (REPO_ROOT / "src").is_dir() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.base import build_runner_from_checkpoint  # noqa: E402


def flat(value) -> list[float]:
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def effect_payload(state: dict, index: int) -> dict:
    dataset = state["dataset"]
    if not 0 <= index < len(dataset):
        raise IndexError(f"sample index must be in [0, {len(dataset) - 1}]")
    item = dataset[index]
    batch = {key: value.unsqueeze(0) for key, value in item.items() if torch.is_tensor(value)}
    with torch.no_grad():
        output = state["runner"].inference(state["runner"].model, batch)
    valid = item["effect_obj_valid_mask"].bool().numpy()
    current = item["effect_obj_points_object"].numpy()[valid]
    gt = current[None] + item["effect_obj_disp_gt"].numpy()[:, valid]
    pred_disp = output["pred_obj_disp_chunk"].detach().cpu().numpy()[0]
    pred = current[None] + pred_disp[:, valid]
    hand_current = item["world_hand_points_object"].numpy()
    hand_future = hand_current[None] + item["action_hand_disp_chunk_object"].numpy()
    error = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    return {
        "index": index,
        "raw_frame": int(item["raw_frame_id"]),
        "future_raw_frames": item["future_raw_frame_ids"].tolist(),
        "current": flat(current), "gt": flat(gt), "pred": flat(pred),
        "hand_current": flat(hand_current), "hand_future": flat(hand_future),
        "step_epe_mm": error.mean(1).tolist(), "ade_mm": float(error.mean()),
    }


def mano_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.npz") if path.is_file())


def mano_payload(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"initial_points_world", "optimized_points_world", "gt_points_world"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path.name} missing fields: {missing}")
        initial = np.asarray(data["initial_points_world"], dtype=np.float32)
        optimized = np.asarray(data["optimized_points_world"], dtype=np.float32)
        gt = np.asarray(data["gt_points_world"], dtype=np.float32)
        error = np.linalg.norm(optimized - gt, axis=-1) * 1000.0
        history = json.loads(str(data["history_json"].item())) if "history_json" in data else []
        return {
            "name": path.name, "side": str(data["side"].item()) if "side" in data else "",
            "current_raw_frame": int(data["current_raw_frame"]) if "current_raw_frame" in data else None,
            "future_raw_frames": data["future_raw_frames"].tolist() if "future_raw_frames" in data else [],
            "initial": flat(initial), "optimized": flat(optimized), "gt": flat(gt),
            "step_epe_mm": error.mean(1).tolist(), "ade_mm": float(error.mean()),
            "history": history,
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "InteractionDynamicsViewer/0.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode(),
                        "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        state = self.server.state  # type: ignore[attr-defined]
        try:
            if parsed.path in {"/", "/index.html"}:
                self.send_bytes((HERE / "viewer.html").read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/info":
                files = mano_files(state["mano_root"])
                self.send_json({"num_samples": len(state["dataset"]), "split": state["split"],
                                "checkpoint": state["checkpoint"],
                                "mano_files": [path.name for path in files]})
                return
            if parsed.path == "/api/effect":
                index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                with state["lock"]:
                    self.send_json(effect_payload(state, index))
                return
            if parsed.path == "/api/mano":
                name = Path(parse_qs(parsed.query).get("name", [""])[0]).name
                path = state["mano_root"] / name
                if not name or not path.is_file():
                    raise FileNotFoundError(name)
                self.send_json(mano_payload(path))
                return
            if parsed.path == "/api/healthz":
                self.send_json({"ok": True})
                return
            self.send_json({"error": "not found"}, 404)
        except Exception as exc:  # HTTP request boundary: return the concrete failure to the UI.
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--mano-root", type=Path,
                        default=Path("output/InteractionDynamics/mano_inverse"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(args.checkpoint, config=args.config,
                                          mode="eval", device=args.device)
    dataset = runner.test_loader.dataset if args.split == "test" else runner.val_loader.dataset
    state = {"runner": runner, "dataset": dataset, "split": args.split,
             "checkpoint": str(Path(args.checkpoint).resolve()),
             "mano_root": args.mano_root.resolve(), "lock": threading.Lock()}
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state  # type: ignore[attr-defined]
    print(f"InteractionDynamics viewer: http://{args.host}:{args.port}")
    print(f"samples={len(dataset)} mano_files={len(mano_files(state['mano_root']))}")
    server.serve_forever()


if __name__ == "__main__":
    main()
