"""Collect six snapshots and run the V1.21e.1 independent-simulator parity test."""
from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.task.CmResidual.tools.run_v121c_calibration_collection import (
    DEXPLORE_PYTHON, DEXPLORE_ROOT, ISAAC_GYM_PYTHON, MOTION_ROOT, OUTPUT_ROOT,
    _gpu_used_mib, _input_identity, _sha256, _write_json,
)
from src.task.CmResidual.v121e_snapshot import PHASE_QUOTAS, evaluate_state

BOOTSTRAP = Path(__file__).with_name("v121e_snapshot_parity_bootstrap.py")
PLAN = ROOT / "src/task/CmResidual/docs/plan/V1.21e.1.md"
SOURCE_RUN = OUTPUT_ROOT / "cmresidual_v121e_snapshot_parity_gpu3_20260920_2130"

def _now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def _git(*args): return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def _parity_command(seed: int) -> list[str]:
    return [str(DEXPLORE_PYTHON), str(BOOTSTRAP.resolve()), "--test", "--task", "Dexplore_Inspire",
            "--cfg_env", "dexplore/data/cfg/inspire.yaml", "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
            "--motion_file", str(MOTION_ROOT), "--num_envs", "1", "--seed", str(seed), "--headless",
            "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0"]

def _arm_specs():
    return (("prefix", 1), ("prefix", 2), ("restore", 1), ("restore", 2))

def _source_rows(source: Path):
    manifest_path = source / "run_manifest.json"
    selected_path = source / "selected_states.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "COMPLETED" or manifest.get("run_id") != source.name:
        raise ValueError("source V1.21e run manifest identity/status mismatch")
    with np.load(selected_path, allow_pickle=False) as p:
        required = {"snapshot_dof_state", "snapshot_actor_root_state", "snapshot_task_indices",
                    "snapshot_reset_buf", "snapshot_terminate_buf", "snapshot_contact_reset", "canonical_current_ig",
                    "state_id", "phase_id", "episode_id", "seed", "collection_batch_id", "raw_obs",
                    "policy_mu", "progress"}
        if missing := required - set(p.files): raise ValueError(f"snapshot artifact missing {sorted(missing)}")
        count = len(p["state_id"])
        if count != 16 or len(set(str(value) for value in p["state_id"])) != count:
            raise ValueError("source V1.21e selection must contain 16 unique states")
        counts = {phase: int(np.count_nonzero(p["phase_id"] == phase)) for phase in PHASE_QUOTAS}
        if counts != {0: 6, 1: 6, 2: 4}:
            raise ValueError(f"source V1.21e phase composition mismatch: {counts}")
        rows = [{k: p[k][i].copy() if hasattr(p[k][i], "copy") else p[k][i] for k in p.files}
                for i in range(count)]
    for row in rows:
        episode = source / f'episodes/{int(row["episode_id"]):04d}/episode.npz'
        if not episode.is_file(): raise ValueError(f"source episode missing: {episode}")
        with np.load(episode, allow_pickle=False) as artifact:
            episode_required = {"initial_dof_state", "initial_actor_root_state", "initial_task_indices",
                                "executed_action_history", "done_history"}
            if missing := episode_required - set(artifact.files):
                raise ValueError(f"source episode artifact missing {sorted(missing)}")
    return rows, {
        "run_id": source.name,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
        "selected_states": {"path": str(selected_path.resolve()), "sha256": _sha256(selected_path)},
    }

def _select(rows):
    result=[]
    for phase, quota in PHASE_QUOTAS.items():
        candidates=sorted((r for r in rows if int(r["phase_id"]) == phase), key=lambda r: str(r["state_id"]))
        if len(candidates) < quota: return None
        result += candidates[:quota]
    return sorted(result, key=lambda r:(int(r["phase_id"]), str(r["state_id"])))

def _save(path: Path, rows):
    keys=rows[0].keys(); payload={}
    for key in keys:
        values=[r[key] for r in rows]
        payload[key] = np.stack(values) if np.asarray(values[0]).ndim else np.asarray(values)
    np.savez_compressed(path, **payload)

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--run-id", required=True); ap.add_argument("--gpu", type=int, default=3); ap.add_argument("--dry-run", action="store_true"); args=ap.parse_args()
    output=(OUTPUT_ROOT/args.run_id).resolve()
    if output.exists(): raise FileExistsError(output)
    source_rows, source_identity = _source_rows(SOURCE_RUN.resolve())
    selected = _select(source_rows)
    if selected is None: raise RuntimeError("source collection insufficient for 2/2/2 snapshot quota")
    inputs=_input_identity(); inputs["parity_bootstrap"]={"path":str(BOOTSTRAP.resolve()),"sha256":_sha256(BOOTSTRAP)}; inputs["source_v121e_run"] = source_identity
    used=_gpu_used_mib(args.gpu)
    if used > 1024: raise RuntimeError(f"GPU{args.gpu} capacity gate failed: {used} MiB used")
    if _git("status", "--porcelain"): raise RuntimeError("formal V1.21e run requires a clean worktree")
    if args.dry_run:
        print(json.dumps({"preflight":"passed","gpu":args.gpu,"gpu_used_mib":used,"git_commit":_git("rev-parse","HEAD"),"quotas":PHASE_QUOTAS,"inputs":inputs}, sort_keys=True)); return 0
    output.mkdir(parents=True); (output/"arms").mkdir(); (output/"logs").mkdir()
    log_path=output/"logs/eval.log"; metrics_path=output/"metrics.jsonl"; selected_path=output/"selected_states.npz"; manifest_path=output/"run_manifest.json"
    config={"phase_quotas":PHASE_QUOTAS,"states":6,"arms":["A1_prefix","A2_prefix","B1_restore","B2_restore"],"processes_per_state":4,"num_envs_per_process":1,"physical_gpu":args.gpu,"control_hz":30,"action":"actor_mean","plan":str(PLAN),"source_run":str(SOURCE_RUN.resolve())}
    _write_json(output/"config.json", config)
    manifest={"manifest_schema":"ref2dex.cmresidual.v121e.snapshot_parity.v2","created_at":_now(),"task":"CmResidual","work_version":"V1.21","run_id":args.run_id,"run_status":"STARTED","git_commit":_git("rev-parse","HEAD"),"branch":_git("branch","--show-current"),"output_dir":str(output),"config":str(output/"config.json"),"metrics":str(metrics_path),"log":str(log_path),"inputs":inputs,"plan_sha256":_sha256(PLAN),"gpu_preflight_mib":used,"scientific_conclusion":"INCONCLUSIVE"}
    _write_json(manifest_path, manifest)
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(args.gpu); env["PYTHONPATH"]=os.pathsep.join((str((DEXPLORE_ROOT/"dexplore").resolve()),str(ISAAC_GYM_PYTHON.resolve()),str(ROOT),env.get("PYTHONPATH","")))
    try:
        with log_path.open("w",encoding="utf-8",buffering=1) as log, metrics_path.open("w",encoding="utf-8",buffering=1) as metrics:
            _save(selected_path,selected)
            results=[]
            for index,state in enumerate(selected):
                records={"prefix": [], "restore": []}
                for mode, repeat in _arm_specs():
                    arm_out=output/f"arms/{index:02d}_{mode}_{repeat}.json"; command=_parity_command(int(state["seed"]))
                    arm_env=env.copy(); arm_env.update({"REF2DEX_V121E_SELECTED":str(selected_path),"REF2DEX_V121E_EPISODE":str(SOURCE_RUN/f'episodes/{int(state["episode_id"]):04d}/episode.npz'),"REF2DEX_V121E_STATE_INDEX":str(index),"REF2DEX_V121E_MODE":mode,"REF2DEX_V121E_REPEAT":str(repeat),"REF2DEX_V121E_OUTPUT":str(arm_out)})
                    log.write(f"state={index} mode={mode} repeat={repeat} command={shlex.join(command)}\n"); subprocess.run(command,cwd=DEXPLORE_ROOT,env=arm_env,check=True,stdout=log,stderr=subprocess.STDOUT)
                    records[mode].append(json.loads(arm_out.read_text()))
                result=evaluate_state(records["prefix"],records["restore"]); result["phase_id"]=int(state["phase_id"]); results.append(result); metrics.write(json.dumps(result,sort_keys=True)+"\n"); metrics.flush()
        valid=sum(r["valid"] for r in results); duplicate_ok=all(r["gates"]["prefix_duplicate_object_ceiling"] and r["gates"]["restore_duplicate_object_ceiling"] for r in results)
        conclusion="SUPPORTED" if valid==6 else "REFUTED" if duplicate_ok else "INVALID_IMPLEMENTATION"
        summary={"state_count":6,"valid_count":valid,"phase_counts":{str(p):sum(int(r["phase_id"])==p for r in results) for p in PHASE_QUOTAS},"conclusion":conclusion,"results":results}; _write_json(output/"summary.json",summary)
        scientific = conclusion if conclusion in {"SUPPORTED", "REFUTED"} else "INCONCLUSIVE"
        manifest.update({"run_status":"COMPLETED","completed_at":_now(),"selected_states":str(selected_path),"summary":str(output/"summary.json"),"conclusion":conclusion,"scientific_conclusion":scientific}); _write_json(manifest_path,manifest,overwrite=True)
        print(json.dumps({"run_id":args.run_id,"run_status":"COMPLETED","valid_count":valid,"conclusion":conclusion},sort_keys=True)); return 0
    except BaseException as error:
        manifest.update({"run_status":"FAILED","completed_at":_now(),"failure_reason":f"{type(error).__name__}: {error}","traceback":traceback.format_exc(),"conclusion":"INVALID_IMPLEMENTATION","scientific_conclusion":"INCONCLUSIVE"}); _write_json(manifest_path,manifest,overwrite=True); raise

if __name__ == "__main__": raise SystemExit(main())
