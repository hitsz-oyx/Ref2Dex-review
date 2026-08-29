#!/usr/bin/env python
"""Minimal Cm viewer HTTP server.

Loads a CmAction checkpoint once, exposes a tiny HTTP API so a browser-based
three.js viewer can pull inference results pair-by-pair without holding the
GPU/long inference in the page itself.

Supports two data layouts via the ``--scene-cache`` flag:

* default (Stage 4 NPZ):  ``--data-root data/processed_data/stage4/data`` and
  ``--input s1/banana_lift/right.npz``; uses ``Stage4CmDataset``.
* ``--scene-cache``:      ``--data-root data/processed_data/cm_scene_v1_1_full``
  and ``--input s1/banana_lift`` (a sequence directory under the scene root);
  uses ``Stage4CmSceneDataset``.

Endpoints
---------
GET /                         serves viewer.html
GET /api/info                 {n_pairs, ds_rate, source_fps, coord_frame,
                                available_strides}
GET /api/pair?idx=N           {raw, next_raw, stride, points, hand,
                                cm_assignment, anchors}
GET /api/horizon?idx=N&stride=S
                               one fixed-start, non-autoregressive prediction
                               at the requested horizon
GET /api/files?query=TEXT     Stage4 left/right NPZ suggestions
                              (or scene sequence dirs in ``--scene-cache`` mode)
POST /api/input               select one Stage4 NPZ under --data-root
                              (or one scene sequence dir under --data-root)
GET /api/healthz              liveness probe

Usage (run from Ref2Dex root)
-----------------------------
    # Stage 4 NPZ (legacy)
    python src/task/Cm/visualization/server.py \
        --checkpoint outputs/cm/cm_full_grab_16slot_20e_fresh_20260805_203822/checkpoints/best.pt \
        --input      data/processed_data/stage4/data/grab/s1/banana_lift/right.npz \
        --port 8765

    # Scene cache v1.1
    python src/task/Cm/visualization/server.py \
        --scene-cache \
        --checkpoint outputs/cm/cm_full_grab_scene_v1_1_16slot_10epoch_20260818_001836/checkpoints/best.pt \
        --input      data/processed_data/cm_scene_v1_1_full/s1/banana_lift \
        --port 8765

Open http://<host>:8765/ in a browser.
"""
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
# Walk up to the Ref2Dex repo root (the ancestor that contains the ``src/`` dir).
# Works regardless of where this script is dropped inside the repo.
REPO_ROOT = HERE
for _ in range(8):
    if (REPO_ROOT / "src").is_dir():
        break
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / "src").is_dir():
    raise SystemExit(f"Could not locate the Ref2Dex repo root from {HERE}")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.base import build_runner_from_checkpoint  # noqa: E402
from src.task.Cm.dataset.cache_schema import (  # noqa: E402
    SIDES as _SCENE_SIDES,
    iter_sequence_dirs as _iter_scene_sequence_dirs,
)
from src.task.Cm.dataset import Stage4CmDataset  # noqa: E402
from src.task.Cm.src.runner import CmActionRunner  # noqa: E402


SLOT_COLORS = np.asarray(
    [
        [0.894, 0.102, 0.110], [0.216, 0.494, 0.722], [0.302, 0.686, 0.290], [0.596, 0.306, 0.639],
        [1.000, 0.498, 0.000], [1.000, 1.000, 0.200], [0.651, 0.337, 0.157], [0.969, 0.506, 0.749],
        [0.600, 0.600, 0.600], [0.122, 0.471, 0.706], [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
        [0.580, 0.404, 0.741], [0.549, 0.337, 0.294], [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
    ],
    dtype=np.float64,
)


def _to_jsonable(arr: np.ndarray) -> list:
    return np.asarray(arr, dtype=np.float32).reshape(-1).tolist()


def _is_scene_cache_mode(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "scene_cache", False))


def _build_dataset(
    args,
    runner: CmActionRunner,
    input_path: Path,
    *,
    stride: int | None = None,
):
    """Return either ``Stage4CmDataset`` (Stage 4 NPZ) or ``Stage4CmSceneDataset``.

    Callers branch on the concrete class to read raw NPZ / scene npy fields.
    """
    common = dict(
        num_obj_points=int(runner.cfg.meta.num_obj_points),
        num_hand_points=int(runner.cfg.meta.num_hand_points),
        base_seed=int(runner.cfg.train.seed),
        active_only=not args.include_inactive,
        min_stride=int(runner.cfg.data.min_stride),
        max_stride=int(runner.cfg.data.max_stride),
        fixed_stride=int(args.stride if stride is None else stride),
        coordinate_frame=str(runner.cfg.meta.coordinate_frame),
    )
    if _is_scene_cache_mode(args):
        from src.task.Cm.dataset.scene import Stage4CmSceneDataset  # noqa: PLC0415
        return Stage4CmSceneDataset(
            input_path,
            sampling_bank_size=int(getattr(runner.cfg.data, "sampling_bank_size", 4)),
            **common,
        )
    return Stage4CmDataset(input_path, **common)


def _input_path_for_client(path: Path, data_root: Path) -> str:
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def _resolve_input_path(value: str, data_root: Path, *, scene_cache: bool) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested] if requested.is_absolute() else [REPO_ROOT / requested, data_root / requested]
    candidate = next((path.resolve() for path in candidates if path.exists()), candidates[-1].resolve())
    try:
        candidate.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"Input must be inside data root: {data_root}") from exc
    if scene_cache:
        # Scene cache: a sequence directory contains shared/, left/, right/, sampling_bank/.
        if not (candidate / "shared").is_dir():
            raise ValueError(f"{candidate}: not a scene-cache sequence (expected sibling 'shared/' dir)")
        present = [side for side in _SCENE_SIDES if (candidate / side).is_dir()]
        if not present:
            raise ValueError(f"{candidate}: sequence has no 'left' or 'right' side directory")
    else:
        # Stage 4 NPZ: must be a hand-side NPZ with sibling shared.npz.
        if candidate.name not in {"left.npz", "right.npz"}:
            raise ValueError("Select a Stage4 left.npz or right.npz file.")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if not (candidate.parent / "shared.npz").is_file():
            raise FileNotFoundError(f"{candidate}: missing sibling shared.npz")
    return candidate


def _list_files(data_root: Path, *, scene_cache: bool) -> list[Path]:
    if scene_cache:
        return _iter_scene_sequence_dirs(data_root)
    return sorted([*data_root.glob("**/left.npz"), *data_root.glob("**/right.npz")])


def _dataset_info(state: dict) -> dict:
    dataset = state["dataset"]
    data_root = state["data_root"]
    input_path = state["input_path"]
    ds_rate = getattr(dataset, "ds_rate", None)
    source_fps = getattr(dataset, "source_fps", None)
    info: dict = {
        "n_pairs": len(dataset),
        "ds_rate": int(ds_rate) if ds_rate is not None else None,
        "source_fps": float(source_fps) if source_fps is not None else None,
        "coord_frame": str(getattr(dataset, "coordinate_frame", "hand_root_t")),
        "checkpoint": str(state["args"].checkpoint),
        "input": _input_path_for_client(input_path, data_root),
        "data_root": str(data_root),
        "base_stride": int(state["args"].stride),
        "available_strides": list(range(int(state["runner"].cfg.data.min_stride), int(state["runner"].cfg.data.max_stride) + 1)),
        "scene_cache": bool(state["scene_cache"]),
    }
    return info


def _dataset_for_stride(state: dict, stride: int):
    min_stride = int(state["runner"].cfg.data.min_stride)
    max_stride = int(state["runner"].cfg.data.max_stride)
    if stride < min_stride or stride > max_stride:
        raise ValueError(f"stride must be in [{min_stride}, {max_stride}]")
    cached = state["datasets_by_stride"].get(stride)
    if cached is not None:
        return cached
    dataset = _build_dataset(state["args"], state["runner"], state["input_path"], stride=stride)
    state["datasets_by_stride"][stride] = dataset
    state["raw_index_by_stride"][stride] = {
        int(dataset[index]["raw_frame_id"]): index for index in range(len(dataset))
    }
    return dataset


def _infer_horizon(state: dict, base_idx: int, stride: int) -> dict:
    base_dataset = state["dataset"]
    if base_idx < 0 or base_idx >= len(base_dataset):
        raise IndexError(f"idx out of range [0, {len(base_dataset) - 1}]")
    raw_frame_id = int(base_dataset[base_idx]["raw_frame_id"])
    dataset = _dataset_for_stride(state, stride)
    dataset_idx = state["raw_index_by_stride"][stride].get(raw_frame_id)
    if dataset_idx is None:
        raise ValueError(f"raw frame {raw_frame_id} is unavailable for stride {stride}")
    payload = _infer_one(state["runner"], dataset, dataset_idx)
    payload["source_idx"] = base_idx
    payload["horizon_stride"] = stride
    payload["horizon_raw_delta"] = payload["next_raw"] - payload["raw"]
    return payload


def _scene_pose_pair(dataset, idx: int, stride: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(pose_current, pose_next, side)`` for a scene-cache sample.

    The dataset's ``sample_location`` does not surface the hand side, so we read
    it back from ``dataset._samples`` (the public ``__getitem__`` already relies
    on the same private tuple).
    """
    samples = getattr(dataset, "_samples", None)
    if not samples or idx >= len(samples):
        raise IndexError(f"scene dataset has no sample {idx}")
    sequence_dir, side, current = samples[idx][:3]
    cache = dataset._get_cache(sequence_dir)  # noqa: SLF001 — mirrors __getitem__
    record = dataset._load_record(sequence_dir, side)  # noqa: SLF001
    side_arrays = record["side_arrays"]
    pose_current = np.asarray(side_arrays["hand_root_pose_world"][int(current)], dtype=np.float32)
    pose_next = np.asarray(
        side_arrays["hand_root_pose_world"][int(current) + int(stride)], dtype=np.float32
    )
    return pose_current, pose_next, str(side)


def _is_scene_dataset(dataset) -> bool:
    """Detect ``Stage4CmSceneDataset`` without importing it eagerly."""
    return type(dataset).__name__ == "Stage4CmSceneDataset"


def _infer_one(runner: CmActionRunner, dataset, idx: int) -> dict:
    sample = dataset[idx]
    batch = {k: v.unsqueeze(0) for k, v in sample.items() if torch.is_tensor(v)}
    with torch.no_grad():
        output = runner.inference(runner.model, batch)
    prediction = {k: v.detach().cpu() for k, v in output.items()}

    valid = sample["obj_valid_mask"].bool()
    current = sample["obj_points"][valid].numpy().astype(np.float64)
    gt_flow = sample["obj_flow_gt"][valid].numpy().astype(np.float64)
    gt_next = current + gt_flow
    pred_flow = prediction["pred_obj_flow"].squeeze(0)[valid].numpy().astype(np.float64)
    pred_next = current + pred_flow

    hand_current = sample["hand_points"].numpy().astype(np.float64)
    hand_future = hand_current + sample["hand_flow"].numpy().astype(np.float64)

    cm_assignment = prediction["cm_assignment"].squeeze(0).numpy()
    slot_ids = cm_assignment.argmax(axis=0)
    num_slots = int(cm_assignment.shape[0])
    slot_colors = SLOT_COLORS[:num_slots]
    hand_slot_colors = slot_colors[slot_ids]

    anchors_pos = prediction["cm_anchor_pos"].squeeze(0).numpy().astype(np.float64)
    decoder_usage = prediction["decoder_slot_usage"].squeeze(0).numpy().astype(np.float64)

    stride_npz = int(sample["stride"])
    if _is_scene_dataset(dataset):
        pose_current, pose_next, side = _scene_pose_pair(dataset, idx, stride_npz)
    else:
        # Stage 4 NPZ layout.
        path, raw_idx = dataset.sample_location(idx)
        data = dataset._load_file(path)  # noqa: SLF001 — mirrors __getitem__
        pose_current = np.asarray(data["hand_root_pose_world"][raw_idx], dtype=np.float32)
        pose_next = np.asarray(
            data["hand_root_pose_world"][raw_idx + stride_npz], dtype=np.float32
        )
        side = path.stem

    return {
        "idx": int(idx),
        "raw": int(sample["raw_frame_id"]),
        "next_raw": int(sample["next_raw_frame_id"]),
        "stride": int(sample["stride"]),
        "side": side,
        "valid_obj_count": int(valid.sum().item()),
        "points": {
            "current": _to_jsonable(current),
            "gt_next": _to_jsonable(gt_next),
            "pred_next": _to_jsonable(pred_next),
        },
        "hand": {
            "current": _to_jsonable(hand_current),
            "future": _to_jsonable(hand_future),
            "slot_colors": _to_jsonable(hand_slot_colors),
        },
        "anchors": {
            "positions": _to_jsonable(anchors_pos),
            "usages": _to_jsonable(decoder_usage),
            "colors": _to_jsonable(slot_colors),
        },
        "pose_world_current": pose_current.reshape(-1).tolist(),
        "pose_world_next": pose_next.reshape(-1).tolist(),
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "CmHTMLViewer/0.2"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status=status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 65536:
            raise ValueError("Request body must be a JSON object smaller than 64 KiB.")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        path = parsed.path
        state = self.server.state  # type: ignore[attr-defined]

        if path in ("/", "/index.html"):
            html_path = HERE / "viewer.html"
            self._send_bytes(html_path.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/healthz":
            with state["lock"]:
                self._send_json({"ok": True, "n_pairs": len(state["dataset"])})
            return

        if path == "/api/info":
            with state["lock"]:
                self._send_json(_dataset_info(state))
            return

        if path == "/api/files":
            query = parse_qs(parsed.query).get("query", [""])[0].strip().lower()
            data_root = state["data_root"]
            scene_cache = bool(state["scene_cache"])
            with state["lock"]:
                candidates = [
                    _input_path_for_client(candidate, data_root)
                    for candidate in _list_files(data_root, scene_cache=scene_cache)
                ]
            candidates = [
                candidate
                for candidate in candidates
                if not query or query in str(candidate).lower()
            ]
            self._send_json({"files": candidates[:300], "total": len(candidates), "query": query})
            return

        if path == "/api/pair":
            qs = parse_qs(parsed.query)
            try:
                idx = int(qs.get("idx", ["0"])[0])
            except (TypeError, ValueError):
                self._send_json({"error": "idx must be an integer"}, status=400)
                return
            try:
                with state["lock"]:
                    ds = state["dataset"]
                    if idx < 0 or idx >= len(ds):
                        self._send_json({"error": f"idx out of range [0, {len(ds) - 1}]"}, status=404)
                        return
                    payload = _infer_one(state["runner"], ds, idx)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": repr(exc)}, status=500)
                return
            self._send_json(payload)
            return

        if path == "/api/horizon":
            qs = parse_qs(parsed.query)
            try:
                idx = int(qs.get("idx", ["0"])[0])
                stride = int(qs.get("stride", ["1"])[0])
                with state["lock"]:
                    payload = _infer_horizon(state, idx, stride)
            except (TypeError, ValueError, IndexError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": repr(exc)}, status=500)
                return
            self._send_json(payload)
            return

        self._send_json({"error": "not found", "path": path}, status=404)

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        state = self.server.state  # type: ignore[attr-defined]
        if parsed.path != "/api/input":
            self._send_json({"error": "not found", "path": parsed.path}, status=404)
            return
        try:
            requested_path = str(self._read_json()["path"])
            with state["lock"]:
                input_path = _resolve_input_path(
                    requested_path, state["data_root"], scene_cache=bool(state["scene_cache"]),
                )
                dataset = _build_dataset(state["args"], state["runner"], input_path)
                state["dataset"] = dataset
                state["input_path"] = input_path
                state["datasets_by_stride"] = {}
                state["raw_index_by_stride"] = {}
                payload = _dataset_info(state)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Root exposed by the browser selector. "
            "Defaults to data/processed_data/stage4/data (Stage 4 NPZ) or "
            "data/processed_data/cm_scene_v1_1_full when --scene-cache is set."
        ),
    )
    parser.add_argument("--scene-cache", action="store_true",
                        help="Use Stage4CmSceneDataset (Scene Cache V1.1) and treat --input as a sequence directory.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--include-inactive", action="store_true")
    args = parser.parse_args()

    runner = build_runner_from_checkpoint(
        args.checkpoint, mode="eval", device=args.device, build_data=False,
    )
    runner.setup_inference(args.checkpoint)
    if not isinstance(runner, CmActionRunner):
        raise SystemExit("This server only supports CmAction BaseRunner checkpoints.")
    if args.data_root is None:
        args.data_root = str(
            REPO_ROOT / "data" / "processed_data" / "cm_scene_v1_1_full"
            if args.scene_cache else
            REPO_ROOT / "data" / "processed_data" / "stage4" / "data"
        )
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")
    input_path = _resolve_input_path(args.input, data_root, scene_cache=args.scene_cache)
    dataset = _build_dataset(args, runner, input_path)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.state = {  # type: ignore[attr-defined]
        "args": args,
        "runner": runner,
        "dataset": dataset,
        "data_root": data_root,
        "input_path": input_path,
        "scene_cache": bool(args.scene_cache),
        "datasets_by_stride": {},
        "raw_index_by_stride": {},
        "lock": threading.RLock(),
    }
    print(
        f"[cm-html-viewer] serving on http://{args.host}:{args.port}/  "
        f"(scene_cache={args.scene_cache}, pairs={len(dataset)}, ckpt={args.checkpoint})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[cm-html-viewer] shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
