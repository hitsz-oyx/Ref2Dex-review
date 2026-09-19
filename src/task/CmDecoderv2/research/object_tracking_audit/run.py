"""CPU审计三段物体轨迹，不改写参考/实际数据。"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from src.task.CmDecoderv2.research.cm_condition_dependence.run import ROOT,sha256,write_json
from src.task.CmDecoderv2.research.cross_hand_cm_swap.diagnostics import local_object_flow,paired_statistics
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import _load_tensor
from .diagnostics import poses,rotation_error,translation_offset,point_epe,lag_audit,LAGS

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150'
DEX = ROOT.parent/'dexplore'
SOURCES = [DEX/'data_processing/prepare_grab.py',DEX/'data_processing/convert_grab.py',
    DEX/'dexplore/learning/dexplore_players.py',DEX/'dexplore/env/tasks/base_dexplore_task.py',
    ROOT/'src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py']


def describe(values):
    return dict(count=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.quantile(values,.95))) if len(values) else None


def run(args):
    output = HERE/'output'/args.run_id
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    def log(message):
        line = datetime.now().astimezone().isoformat(timespec='seconds')+' '+message
        print(line,flush=True)
        with (output/'run.log').open('a') as f:f.write(line+'\n')
    manifest = dict(task='CmDecoderv2',work_version='V1.1.11',run_id=args.run_id,activity_id=args.run_id,
        schema_name='ref2dex.object_tracking_audit.v1',run_status='STARTED',conclusion='INCONCLUSIVE',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),worktree_dirty=True,
        started_at=datetime.now().astimezone().isoformat(timespec='seconds'),seed=42,checkpoint='not_loaded',
        config_snapshot='config.json',metadata_snapshot='metadata.json',output=str(output))
    write_json(output/'run_manifest.json',manifest)
    try:
        meta = json.loads((BASE/'metadata.json').read_text())
        protected = dict(meta['protected_sha256'])
        for p in list(BASE.rglob('*'))+list(HERE.iterdir())+SOURCES:
            if p.is_file():protected[str(p)] = sha256(p)
        stats = dict(meta['input_stats'])
        entries = meta['sequence_entries'][:2] if args.smoke else meta['sequence_entries']
        for e in entries:
            for folder in ('inspire_geometric_dexplore','inspire_rl_object_dexplore'):
                p = ROOT/'data/processed_data'/folder/e['id'].replace('/','_')/'interaction_hand_inspire.pt'
                protected[str(p)] = sha256(p)
            p = Path(e['geometry_root'])/'frame_time.npy'
            s=p.stat();stats[str(p)] = [s.st_size,s.st_mtime_ns]
        def check():
            for p,digest in protected.items():assert sha256(p)==digest,p
            for p,expected in stats.items():
                s=Path(p).stat();assert [s.st_size,s.st_mtime_ns]==expected,p
        check()
        records,values,lag_curves,lag_results = [],{},[],[]
        heldout_values,heldout_baseline,heldout_sequence = [],[],[]
        for sn,e in enumerate(entries):
            if time.monotonic()-started>600:raise TimeoutError('10 minute CPU budget')
            d = dict(np.load(BASE/'descriptors'/f'{sn:02d}.npz'))
            ref = _load_tensor(ROOT/'data/processed_data/inspire_geometric_dexplore'/e['id'].replace('/','_')/'interaction_hand_inspire.pt')
            actual = _load_tensor(ROOT/'data/processed_data/inspire_rl_object_dexplore'/e['id'].replace('/','_')/'interaction_hand_inspire.pt')
            assert ref.shape==actual.shape and len(ref)==len(d['actual_pose'])
            columns = np.ones(ref.shape[1],bool);columns[198:205]=False;columns[373:391]=False
            assert np.array_equal(ref[:,columns],actual[:,columns]),e['id']
            time_array = np.load(Path(e['geometry_root'])/'frame_time.npy')
            raw = np.load(Path(e['geometry_root'])/'source_frame_id.npy')
            np.testing.assert_allclose(time_array,raw/120.,atol=2e-6)
            assert (np.diff(raw)==4).all()
            a,g = poses(actual),poses(ref)
            cached_a,cached_g = poses(actual,True),poses(ref,True)
            np.testing.assert_allclose(cached_a,d['actual_pose'],atol=2e-6)
            parent = d['reference_pose'].astype(np.float64)
            bias,residual = translation_offset(cached_g,parent)
            tracking_bias,tracking_residual = translation_offset(a,g)
            native_position = np.linalg.norm(a[:,:3,3]-g[:,:3,3],axis=-1)*1000
            native_rotation = rotation_error(a,g)
            # Preserve the exact previous cache contract for segment comparisons.
            fa,fg,fp = [local_object_flow(d['canonical'],p) for p in (d['actual_pose'],cached_g,parent)]
            covered = np.zeros(len(fa),bool)
            for s in d['starts'][d['target_valid']]:covered[s:s+4]=True
            moving = covered & (np.sqrt((fa**2).sum(-1).mean(-1))>=1.)
            series = dict(tracking_position_mm=native_position,tracking_rotation_deg=native_rotation,
                geometric_parent_offset_residual_mm=residual,tracking_after_mean_offset_mm=tracking_residual,
                geometric_parent_effect_mm=point_epe(fg,fp)[moving],
                actual_geometric_effect_mm=point_epe(fa,fg)[moving],actual_parent_effect_mm=point_epe(fa,fp)[moving])
            for k,v in series.items():values.setdefault(k,[]).append(v)
            lag,t,curve = lag_audit(fa,fg,moving)
            lag_results.append(lag)
            lag_curves.append(curve.mean(-1) if len(t) else np.full(len(LAGS),np.nan))
            if lag['selected_lag'] is not None:
                split=len(t)//2;idx=int(np.flatnonzero(LAGS==lag['selected_lag'])[0])
                heldout_values.extend(curve[idx,split:].tolist());heldout_baseline.extend(curve[15,split:].tolist())
                heldout_sequence.extend([sn]*(len(t)-split))
            row = dict(sequence_number=sn,id=e['id'],frames=len(ref),moving_transitions=int(moving.sum()),
                geometric_parent_offset_m=bias.tolist(),tracking_mean_offset_m=tracking_bias.tolist(),
                native_geometric_vs_parent_rotation_deg=describe(rotation_error(g,parent)),
                inverse_geometric_vs_parent_rotation_deg=describe(rotation_error(cached_g,parent)),
                initial_tracking_position_mm=float(native_position[0]),initial_tracking_rotation_deg=float(native_rotation[0]),
                unchanged_reference_columns=True,cache_pose_replayed=True,lag=lag,**{k:describe(v) for k,v in series.items()})
            records.append(row)
            np.savez_compressed(output/f'trajectory_{sn:02d}.npz',**series,lag_support=t,lag_errors_mm=curve,
                parent_pose=parent,geometric_pose=cached_g,actual_pose=d['actual_pose'],moving=moving)
            log(f'{sn+1}/{len(entries)} {e["id"]}: position={native_position.mean():.1f}mm rotation={native_rotation.mean():.1f}deg lag={lag["selected_lag"]}')
        check()
        aggregates={k:dict(micro=describe(np.concatenate(v)),parent_macro=float(np.mean([x.mean() for x in v if len(x)]))) for k,v in values.items()}
        lag_stats=paired_statistics(np.asarray(heldout_values),np.asarray(heldout_baseline),np.asarray(heldout_sequence))
        summary=dict(aggregates=aggregates,heldout_lag=lag_stats,parents=len(entries),
            initial_position_max_mm=max(r['initial_tracking_position_mm'] for r in records),
            initial_rotation_max_deg=max(r['initial_tracking_rotation_deg'] for r in records),
            protected_inputs_unchanged=True,all_reference_columns_preserved=True,all_cache_poses_replayed=True,
            conclusion='INCONCLUSIVE',elapsed_seconds=time.monotonic()-started)
        with (output/'metrics.jsonl').open('w') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        write_json(output/'summary.json',summary)
        write_json(output/'config.json',dict(arguments=vars(args),lags=LAGS.tolist(),fps=30,baseline=str(BASE),
            lag_protocol='common support; chronological first half chooses lag, second half evaluates',
            effect_contract='existing transposed-native cache convention; 4096 canonical points',
            tracking_contract='native XYZW active rotation; geometric reference vs RL actual',
            moving='unique transitions covered by target-valid K4 windows; actual RMS >=1mm'))
        write_json(output/'metadata.json',dict(protected_sha256=protected,input_stats=stats,sequence_entries=entries,
            source_files=[str(p) for p in SOURCES]))
        plot(records,lag_curves,output)
        size=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        assert size<1024**3 and time.monotonic()-started<600
        manifest.update(run_status='COMPLETED',last_step=sum(r['frames'] for r in records),output_bytes=size,
            elapsed_seconds=time.monotonic()-started,finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))
        write_json(output/'run_manifest.json',manifest)
        log('COMPLETED')
    except Exception as error:
        manifest.update(run_status='FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json',manifest);log(manifest['error']);raise


def plot(records,curves,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(records))
    axes[0].bar(x,[r['tracking_position_mm']['mean'] for r in records])
    axes[0].set(xlabel='Parent index',ylabel='Native object center tracking error (mm)')
    for curve in curves:axes[1].plot(LAGS,curve,alpha=.35)
    axes[1].set(xlabel='Reference lag (frames; negative = actual behind)',ylabel='Object flow EPE (mm)',yscale='log')
    fig.tight_layout();fig.savefig(output/'tracking.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
