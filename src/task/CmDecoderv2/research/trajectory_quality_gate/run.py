"""只读质量标记、修复队列和已有跨手候选的交叉检查。"""
import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from src.task.CmDecoderv2.research.cm_condition_dependence.run import ROOT,sha256,write_json
from src.task.CmDecoderv2.research.cross_hand_cm_swap.diagnostics import local_object_flow
from src.task.CmDecoderv2.research.cross_hand_pair_coverage.matching import selection_stats
from .gates import PROFILES,quality_flags

HERE=Path(__file__).resolve().parent
TRACK=HERE.parent/'object_tracking_audit/output/tracking_val_20260912_154240'
PAIR=HERE.parent/'cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150'
MODES=('baseline','effect','pose20','combined20','pose40','combined40','pose80','combined80')


def run(args):
    output=HERE/'output'/args.run_id;output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def log(s):
        line=datetime.now().astimezone().isoformat(timespec='seconds')+' '+s
        print(line,flush=True)
        with (output/'run.log').open('a') as f:f.write(line+'\n')
    manifest=dict(task='CmDecoderv2',modification_version='V1.1.12',run_id=args.run_id,activity_id=args.run_id,
        schema_name='ref2dex.trajectory_quality_gate.v1',diagnostic_only=True,run_status='STARTED',conclusion='INCONCLUSIVE',
        started_at=datetime.now().astimezone().isoformat(timespec='seconds'),seed=42,checkpoint='not_loaded',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),worktree_dirty=True,
        config_snapshot='config.json',metadata_snapshot='metadata.json',output=str(output))
    write_json(output/'run_manifest.json',manifest)
    try:
        meta=json.loads((TRACK/'metadata.json').read_text());protected=dict(meta['protected_sha256'])
        for p in list(TRACK.iterdir())+list(HERE.iterdir()):
            if p.is_file():protected[str(p)]=sha256(p)
        def check():
            for p,digest in protected.items():assert sha256(p)==digest,p
            for p,expected in meta['input_stats'].items():
                s=Path(p).stat();assert [s.st_size,s.st_mtime_ns]==expected,p
        check()
        edges=dict(np.load(PAIR/'candidate_pairs.npz'))
        windows,records,candidates,queue=[],[],[],[]
        for sn,entry in enumerate(meta['sequence_entries']):
            if time.monotonic()-started>300:raise TimeoutError('5 minute CPU budget')
            d=dict(np.load(PAIR/'descriptors'/f'{sn:02d}.npz'))
            t=dict(np.load(TRACK/f'trajectory_{sn:02d}.npz'))
            starts,ids=d['starts'],d['global_rows']
            fa,fg=[local_object_flow(d['canonical'],t[key]) for key in ('actual_pose','geometric_pose')]
            flow_epe=np.linalg.norm(fa-fg,axis=-1).mean(-1)
            wa=starts[:,None]+np.arange(4);wp=starts[:,None]+np.arange(5)
            flags=quality_flags(flow_epe[wa],np.sqrt((fa**2).sum(-1).mean(-1))[wa],
                np.sqrt((fg**2).sum(-1).mean(-1))[wa],t['tracking_position_mm'][wp],t['tracking_rotation_deg'][wp])
            eligible=d['target_valid']&d['moving']
            flags['baseline']=np.ones(len(ids),bool)
            lookup={int(row):i for i,row in enumerate(ids)}
            mask=edges['sequence_number']==sn
            local={k:v[mask] for k,v in edges.items() if k!='sequence_number'}
            for k in ('target','source'):local[k]=np.array([lookup[int(i)] for i in local[k]],np.int64)
            matched=np.zeros(len(ids),bool);matched[local['target']]=True
            raw=np.load(Path(entry['geometry_root'])/'source_frame_id.npy')
            for i,row_id in enumerate(ids):
                windows.append(dict(row=int(row_id),id=entry['id'],sequence_number=sn,start=int(starts[i]),
                    target_valid=bool(d['target_valid'][i]),moving=bool(d['moving'][i]),cross_hand_compatible=bool(matched[i]),
                    **{k:(bool(v[i]) if v.dtype==bool else int(v[i]) if 'reasons' in k else float(v[i])) for k,v in flags.items()}))
            for mode in MODES:
                result=selection_stats(local,starts,eligible&flags[mode],'any_time')
                for kind,key in [('best','selected_edge_indices'),('nonoverlap_greedy','nonoverlap_edge_indices')]:
                    for ei in result.pop(key):
                        i,j=int(local['target'][ei]),int(local['source'][ei])
                        candidates.append(dict(diagnostic_only=True,split='val',mode=mode,kind=kind,id=entry['id'],
                            target_row=int(ids[i]),source_row=int(ids[j]),target_start=int(starts[i]),source_start=int(starts[j]),
                            target_raw_frame=int(raw[starts[i]]),source_raw_frame=int(raw[starts[j]]),
                            source_hand='mano_original',target_hand='inspire_rl',
                            effect_tracking=bool(flags['effect'][i]),pose20=bool(flags['pose20'][i]),
                            reason_bits20=int(flags['pose20_reasons'][i]),pair_effect_ratio_max=float(local['ratio'][ei].max())))
                records.append(dict(id=entry['id'],mode=mode,eligible_motion_windows=int(eligible.sum()),
                    quality_motion_windows=int((eligible&flags[mode]).sum()),**result))
            n=int(eligible.sum());failed=int((eligible&~flags['effect']).sum())
            queue.append(dict(id=entry['id'],moving_windows=n,effect_tracking_failed=failed,
                failure_fraction=failed/n if n else None,compatible_motion_windows=int((eligible&matched).sum()),
                compatible_but_effect_failed=int((eligible&matched&~flags['effect']).sum()),
                compatible_but_pose20_failed=int((eligible&matched&~flags['pose20']).sum())))
            log(f'{sn+1}/30 {entry["id"]}: moving={n}, effect_tracking={n-failed}, compatible={queue[-1]["compatible_motion_windows"]}')
        check()
        summary={}
        for mode in MODES:
            rows=[r for r in records if r['mode']==mode]
            summary[mode]={k:sum(r[k] for r in rows) for k in ('eligible_motion_windows','quality_motion_windows','windows','edges','unique_selected_sources','one_to_one_capacity','nonoverlap_greedy')}
            summary[mode]['parents']=sum(r['windows']>0 for r in rows)
        assert summary['baseline']['windows']==53 and summary['baseline']['parents']==10
        assert summary['baseline']['nonoverlap_greedy']==18
        summary['compatible_but_effect_failed']=sum(r['compatible_but_effect_failed'] for r in queue)
        summary['compatible_but_pose20_failed']=sum(r['compatible_but_pose20_failed'] for r in queue)
        queue.sort(key=lambda r:(-(r['failure_fraction'] if r['failure_fraction'] is not None else -1),-r['effect_tracking_failed'],r['id']))
        with (output/'repair_queue.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
        for filename,rows in [('window_metrics.jsonl',windows),('metrics.jsonl',records),('candidate_manifest.jsonl',candidates)]:
            with (output/filename).open('w') as f:
                for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        write_json(output/'summary.json',summary)
        write_json(output/'config.json',dict(arguments=vars(args),profiles=PROFILES,modes=MODES,effect_absolute_mm=1.,effect_relative=.25,
            diagnostic_only=True,tracking_input=str(TRACK),pair_input=str(PAIR),window=4,fps=30,
            reason_bits20={'1':'effect','2':'position','4':'rotation'},threshold_status='engineering diagnostics, not validated admission criteria'))
        write_json(output/'metadata.json',dict(protected_sha256=protected,input_stats=meta['input_stats'],sequence_entries=meta['sequence_entries']))
        plot(summary,output)
        size=sum(p.stat().st_size for p in output.iterdir() if p.is_file());assert size<100*1024**2
        manifest.update(run_status='COMPLETED',last_step=len(windows),elapsed_seconds=time.monotonic()-started,
            output_bytes=size,finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))
        write_json(output/'run_manifest.json',manifest);log('COMPLETED '+json.dumps(summary['combined20']))
    except Exception as error:
        manifest.update(run_status='FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json',manifest);log(manifest['error']);raise


def plot(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(MODES))
    ax.bar(x-.2,[summary[m]['windows'] for m in MODES],width=.4,label='Compatible moving windows')
    ax.bar(x+.2,[summary[m]['nonoverlap_greedy'] for m in MODES],width=.4,label='Disjoint greedy pairs')
    ax.set_xticks(x);ax.set_xticklabels(MODES,rotation=35,ha='right');ax.legend();fig.tight_layout()
    fig.savefig(output/'quality_coverage.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);run(p.parse_args())
