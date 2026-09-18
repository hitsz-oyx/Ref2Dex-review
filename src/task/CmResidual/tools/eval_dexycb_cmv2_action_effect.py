"""Run a bounded structured Cmv2 action-effect diagnostic.

This CLI intentionally evaluates synthetic geometry unless it is embedded by a
physical task.  It is useful for checkpoint/schema and candidate-batch gates;
it is not a substitute for the later IsaacGym counterfactual physics run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.CmResidual.cm_v2_action_evaluator import (  # noqa: E402
    Cmv2ActionEvaluator, NominalHandSweep, EVALUATOR_SCHEMA)
from src.task.CmResidual.cm_v2_adapter import (  # noqa: E402
    MODEL_CONFIG, FrozenCmv2Adapter)
from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2V13Model  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_adapter(device: str):
    model = ObjectInteractionCmv2V13Model(SimpleNamespace(**MODEL_CONFIG))
    temporary = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    temporary.close()
    path = Path(temporary.name)
    torch.save({"architecture_version": "v1_3_rigid_only", "model": model.state_dict()}, path)
    return FrozenCmv2Adapter(path, _sha256(path), device), path


def _synthetic_inputs(candidates: int, device: str):
    x = torch.linspace(-0.05, 0.05, 1024, device=device)
    object_points = torch.stack((x, x.square(), torch.zeros_like(x)), -1)[None]
    object_normals = torch.zeros_like(object_points)
    object_normals[..., 2] = 1.0
    current = torch.zeros(1, 1538, 3, device=device)
    current_normals = torch.zeros_like(current)
    current_normals[..., 2] = 1.0
    next_points = current[:, None].expand(1, candidates, 1538, 3).clone()
    for index in range(candidates):
        next_points[:, index, :, 0] += float(index) * 0.001
    next_normals = current_normals[:, None].expand_as(next_points)
    return object_points, object_normals, NominalHandSweep(
        current, current_normals, next_points, next_normals)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="Frozen Cmv2 checkpoint; requires --checkpoint-sha256")
    parser.add_argument("--checkpoint-sha256", default="",
                        help="Exact SHA256 for --checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a temporary deterministic model and synthetic geometry")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON output for this diagnostic")
    args = parser.parse_args(argv)
    if args.candidates <= 0:
        parser.error("--candidates must be positive")
    temporary = None
    if args.synthetic:
        adapter, temporary = _synthetic_adapter(args.device)
    else:
        if args.checkpoint is None or not args.checkpoint_sha256:
            parser.error("real evaluation requires --checkpoint and --checkpoint-sha256")
        adapter = FrozenCmv2Adapter(args.checkpoint, args.checkpoint_sha256, args.device)
    object_points, object_normals, sweep = _synthetic_inputs(args.candidates, args.device)
    result = Cmv2ActionEvaluator(adapter).evaluate(
        object_points, object_normals, sweep,
        candidate_actions=torch.zeros(1, args.candidates, 18, device=args.device),
        desired_delta_xi=torch.zeros(1, 6, device=args.device))
    summary = {
        "schema": EVALUATOR_SCHEMA,
        "geometry": "synthetic",
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "candidate_count": args.candidates,
        "finite": bool(torch.isfinite(result["predicted_delta_xi"]).all()),
        "best_candidate": int(Cmv2ActionEvaluator.select_best(result)[0].item()),
        "effect_score": result["effect_score"][0].detach().cpu().tolist(),
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if temporary is not None:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
