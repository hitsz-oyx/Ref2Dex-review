"""Collect 16 snapshots and run the formal V1.21e prefix/restore parity test."""
from __future__ import annotations
import argparse, hashlib, json, os, shlex, subprocess, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.task.CmResidual.tools.run_v121c_calibration_collection import (
    DEXPLORE_PYTHON, DEXPLORE_ROOT, ISAAC_GYM_PYTHON, MOTION_ROOT, OUTPUT_ROOT,
    _command as collection_command, _gpu_used_mib, _input_identity, _sha256, _write_json,
)
from src.task.CmResidual.v121e_snapshot import PHASE_QUOTAS, evaluate_state

BOOTSTRAP = Path(__file__).with_name("v121e_snapshot_parity_bootstrap.py")
PLAN = ROOT / "src/task/CmResidual/docs/plan/V1.21e.md"
MAX_EPISODES = 64

def _now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def _git(*args): return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def _parity_command(seed: int) -> list[str]:
    return [str(DEXPLORE_PYTHON), str(BOOTSTRAP.resolve()), "--test", "--task", "Dexplore_Inspire",
            "--cfg_env", "dexplore/data/cfg/inspire.yaml", "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
            "--motion_file", str(MOTION_ROOT), "--num_envs", "2", "--seed", str(seed), "--headless",
            "--sim_device", "cuda:0", "--rl_device", "cuda:0", "--graphics_device_id", "0"]

def _rows(path: Path, episode_id: int, seed: int, batch: str):
    with np.load(path, allow_pickle=False) as p:
        required = {"snapshot_dof_state", "snapshot_actor_root_state", "snapshot_task_indices",
                    "snapshot_reset_buf", "snapshot_terminate_buf", "snapshot_contact_reset", "canonical_current_ig"}
        if missing := required - set(p.files): raise ValueError(f"snapshot artifact missing {sorted(missing)}")
        return [{**{k: p[k][i].copy() if hasattr(p[k][i], "copy") else p[k][i] for k in p.files},
                 "episode_id": episode_id, "seed": seed, "collection_batch_id": batch}
                for i in range(len(p["state_id"]))]

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
    inputs=_input_identity(); inputs["parity_bootstrap"]={"path":str(BOOTSTRAP.resolve()),"sha256":_sha256(BOOTSTRAP)}
    used=_gpu_used_mib(args.gpu)
    if used > 1024: raise RuntimeError(f"GPU{args.gpu} capacity gate failed: {used} MiB used")
    if _git("status", "--porcelain"): raise RuntimeError("formal V1.21e run requires a clean worktree")
    if args.dry_run:
        print(json.dumps({"preflight":"passed","gpu":args.gpu,"gpu_used_mib":used,"git_commit":_git("rev-parse","HEAD"),"quotas":PHASE_QUOTAS,"inputs":inputs}, sort_keys=True)); return 0
    output.mkdir(parents=True); (output/"episodes").mkdir(); (output/"arms").mkdir(); (output/"logs").mkdir()
    log_path=output/"logs/eval.log"; metrics_path=output/"metrics.jsonl"; selected_path=output/"selected_states.npz"; manifest_path=output/"run_manifest.json"
    batch=f"{args.run_id}:snapshot"; config={"phase_quotas":PHASE_QUOTAS,"states":16,"arms":["prefix_duplicate","restore_duplicate"],"physical_gpu":args.gpu,"control_hz":30,"action":"actor_mean","plan":str(PLAN)}
    _write_json(output/"config.json", config)
    manifest={"manifest_schema":"ref2dex.cmresidual.v121e.snapshot_parity.v1","created_at":_now(),"task":"CmResidual","work_version":"V1.21","run_id":args.run_id,"run_status":"STARTED","git_commit":_git("rev-parse","HEAD"),"branch":_git("branch","--show-current"),"output_dir":str(output),"config":str(output/"config.json"),"metrics":str(metrics_path),"log":str(log_path),"inputs":inputs,"plan_sha256":_sha256(PLAN),"gpu_preflight_mib":used,"scientific_conclusion":"INCONCLUSIVE"}
    _write_json(manifest_path, manifest)
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=str(args.gpu); env["PYTHONPATH"]=os.pathsep.join((str((DEXPLORE_ROOT/"dexplore").resolve()),str(ISAAC_GYM_PYTHON.resolve()),str(ROOT),env.get("PYTHONPATH",""))); env["REF2DEX_V121E_CAPTURE_SNAPSHOT"]="1"
    rows=[]; selected=None
    try:
        with log_path.open("w",encoding="utf-8",buffering=1) as log, metrics_path.open("w",encoding="utf-8",buffering=1) as metrics:
            for episode_id in range(MAX_EPISODES):
                seed=5909+episode_id; ep=output/f"episodes/{episode_id:04d}"; command=collection_command(seed,ep)
                sim_sha=hashlib.sha256(json.dumps({"seed":seed,"num_envs":1,"control_hz":30,"motion":str(MOTION_ROOT)},sort_keys=True).encode()).hexdigest()
                ep_env=env.copy(); ep_env.update({"REF2DEX_V121C_EPISODE_OUTPUT":str(ep),"REF2DEX_V121C_COLLECTION_BATCH_ID":batch,"REF2DEX_V121C_EPISODE_ID":str(episode_id),"REF2DEX_V121C_EPISODE_SEED":str(seed),"REF2DEX_V121C_SIM_CONFIG_SHA256":sim_sha})
                log.write(f"collect episode={episode_id} command={shlex.join(command)}\n"); subprocess.run(command,cwd=DEXPLORE_ROOT,env=ep_env,check=True,stdout=log,stderr=subprocess.STDOUT)
                rows += _rows(ep/"active_states.npz",episode_id,seed,batch); selected=_select(rows)
                if selected is not None: break
            if selected is None: raise RuntimeError("collection insufficient for 6/6/4 snapshot quota")
            _save(selected_path,selected)
            results=[]
            for index,state in enumerate(selected):
                records={}
                for mode in ("prefix","restore"):
                    arm_out=output/f"arms/{index:02d}_{mode}.json"; command=_parity_command(int(state["seed"]))
                    arm_env=env.copy(); arm_env.update({"REF2DEX_V121E_SELECTED":str(selected_path),"REF2DEX_V121E_EPISODE":str(output/f'episodes/{int(state["episode_id"]):04d}/episode.npz'),"REF2DEX_V121E_STATE_INDEX":str(index),"REF2DEX_V121E_MODE":mode,"REF2DEX_V121E_OUTPUT":str(arm_out)})
                    log.write(f"state={index} mode={mode} command={shlex.join(command)}\n"); subprocess.run(command,cwd=DEXPLORE_ROOT,env=arm_env,check=True,stdout=log,stderr=subprocess.STDOUT)
                    records[mode]=json.loads(arm_out.read_text())
                result=evaluate_state(records["prefix"],records["restore"]); result["phase_id"]=int(state["phase_id"]); results.append(result); metrics.write(json.dumps(result,sort_keys=True)+"\n"); metrics.flush()
        valid=sum(r["valid"] for r in results); duplicate_ok=all(r["gates"]["prefix_duplicate_object_ceiling"] and r["gates"]["restore_duplicate_object_ceiling"] for r in results)
        conclusion="SUPPORTED" if valid==16 else "REFUTED" if duplicate_ok else "INVALID_IMPLEMENTATION"
        summary={"state_count":16,"valid_count":valid,"phase_counts":{str(p):sum(int(r["phase_id"])==p for r in results) for p in PHASE_QUOTAS},"conclusion":conclusion,"results":results}; _write_json(output/"summary.json",summary)
        scientific = conclusion if conclusion in {"SUPPORTED", "REFUTED"} else "INCONCLUSIVE"
        manifest.update({"run_status":"COMPLETED","completed_at":_now(),"selected_states":str(selected_path),"summary":str(output/"summary.json"),"conclusion":conclusion,"scientific_conclusion":scientific}); _write_json(manifest_path,manifest,overwrite=True)
        print(json.dumps({"run_id":args.run_id,"run_status":"COMPLETED","valid_count":valid,"conclusion":conclusion},sort_keys=True)); return 0
    except BaseException as error:
        manifest.update({"run_status":"FAILED","completed_at":_now(),"failure_reason":f"{type(error).__name__}: {error}","traceback":traceback.format_exc(),"conclusion":"INVALID_IMPLEMENTATION","scientific_conclusion":"INCONCLUSIVE"}); _write_json(manifest_path,manifest,overwrite=True); raise

if __name__ == "__main__": raise SystemExit(main())
