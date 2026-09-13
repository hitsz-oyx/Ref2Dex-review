"""允许时间偏移，诊断原始 MANO 与 Inspire 的固定门槛配对覆盖。"""
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points
from src.task.CmDecoderv2.research.cm_condition_dependence.run import ROOT,sha256,write_json
from src.task.CmDecoderv2.research.cross_hand_cm_swap.run import BASE as SAME_SOURCE
from src.task.CmDecoderv2.research.cross_hand_cm_swap.diagnostics import LIMITS,local_object_flow,contact_comparison,gates
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_v1_3_cache import (
    _build_mano_pool,_sample_correspondence,_sample_mano_surface,
)
from .matching import POLICIES,allowed,exact_effect_pairs,contact_filter,selection_stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'cross_hand_cm_swap/output/cross_hand_val_20260912_151850'


def run(args):
    output = HERE/'output'/args.run_id
    output.mkdir(parents=True,exist_ok=False)
    (output/'descriptors').mkdir()
    started = time.monotonic()
    def log(message):
        line = datetime.now().astimezone().isoformat(timespec='seconds')+' '+message
        print(line,flush=True)
        with (output/'run.log').open('a') as f:
            f.write(line+'\n')
    def budget():
        if time.monotonic()-started>1200:
            raise TimeoutError('20 minute budget')
        if torch.cuda.max_memory_allocated()>4*1024**3:
            raise MemoryError('4 GiB allocation budget')
    manifest = dict(task='CmDecoderv2',modification_version='V1.1.10',
        schema_name='ref2dex.cross_hand_pair_coverage.v1',run_id=args.run_id,activity_id=args.run_id,
        run_status='STARTED',conclusion='INCONCLUSIVE',seed=42,
        started_at=datetime.now().astimezone().isoformat(timespec='seconds'),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        worktree_dirty=True,config_snapshot='config.json',metadata_snapshot='metadata.json',
        checkpoint='not_loaded; validity masks inherited from frozen V1.1.9',baseline=str(BASE),output=str(output))
    write_json(output/'run_manifest.json',manifest)
    try:
        previous = json.loads((BASE/'metadata.json').read_text())
        protected = dict(previous['protected_sha256'])
        for p in list(BASE.iterdir())+list(HERE.iterdir()):
            if p.is_file():
                protected[str(p)] = sha256(p)
        stats = previous['input_stats']
        def check_inputs():
            for p,digest in protected.items():
                assert sha256(p)==digest,p
            for p,expected in stats.items():
                s = Path(p).stat()
                assert [s.st_size,s.st_mtime_ns]==expected,p
        check_inputs()
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        pair = dict(np.load(BASE/'pair_bank.npz'))
        entries = json.loads((SAME_SOURCE/'metadata.json').read_text())['sequence_entries']
        if args.smoke:
            entries = entries[:2]
        index = json.loads((ROOT/'data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json').read_text())
        parent_base = Path(index['source_roots']['grab_parent_cache'])
        pool,_ = _build_mano_pool(ROOT/'dataset/arctic/data/body_models/mano',.20)
        correspondence = _sample_correspondence(pool,2048,2024)
        old_masks = gates(pair)
        records,all_edges,selected = [],[],[]
        checks = dict(max_diagonal_descriptor_error=0.,diagonal_effect_replayed=True,diagonal_strict_replayed=True)
        eligible,moving_eligible = 0,0
        with torch.inference_mode():
            for sn,entry in enumerate(entries):
                budget()
                ids = np.flatnonzero(pair['sequence_number']==sn)
                starts = pair['start_frame'][ids]
                target_valid,source_valid = pair['target_valid'][ids].all(-1),pair['mano_valid'][ids].all(-1)
                moving = pair['actual_rms_mm'][ids].mean(-1)>=LIMITS['moving_rms_mm']
                eligible += int(target_valid.sum())
                moving_eligible += int((target_valid&moving).sum())
                geo,parent = Path(entry['geometry_root']),parent_base/entry['id']
                actual_pose = np.load(geo/'obj_pose_world.npy').astype(np.float32)
                raw = np.load(geo/'source_frame_id.npy')
                lookup = {int(r):i for i,r in enumerate(np.load(parent/'shared/raw_frame_id.npy'))}
                aligned = np.array([lookup[int(r)] for r in raw])
                reference_pose = np.load(parent/'shared/obj_pose_world.npy')[aligned].astype(np.float32)
                obj = np.load(geo/'obj_points_pool_world.npy',mmap_mode='r')
                local_obj = (np.asarray(obj)-actual_pose[:,None,:3,3])@actual_pose[:,:3,:3]
                canonical = local_obj[0]
                points,normals = np.empty((len(raw),2048,3),np.float32),np.empty((len(raw),2048,3),np.float32)
                _sample_mano_surface(parent,raw,reference_pose,correspondence,points,normals,batch_size=16)
                local_hand = (points-reference_pose[:,None,:3,3])@reference_pose[:,:3,:3]
                actual_contact = np.load(geo/'obj_candidate_mask_2cm.npy').astype(bool)
                source_contact = np.empty_like(actual_contact)
                for t in range(0,len(raw),8):
                    result = knn_points(torch.as_tensor(local_obj[t:t+8],device=args.device),
                                        torch.as_tensor(local_hand[t:t+8],device=args.device),K=32)
                    source_contact[t:t+8] = (result.dists[:,:,0]<=.02**2).cpu().numpy()
                actual_flow,reference_flow = local_object_flow(canonical,actual_pose),local_object_flow(canonical,reference_pose)
                iou,gap = contact_comparison(canonical,actual_contact,source_contact)
                descriptors = dict(actual_rms_mm=np.sqrt((actual_flow**2).sum(-1).mean(-1)),
                    reference_rms_mm=np.sqrt((reference_flow**2).sum(-1).mean(-1)),
                    effect_epe_mm=np.linalg.norm(actual_flow-reference_flow,axis=-1).mean(-1),
                    contact_iou=iou,contact_centroid_mm=gap)
                for key,value in descriptors.items():
                    rebuilt = value[starts[:,None]+np.arange(4)]
                    error = float(np.abs(rebuilt-pair[key][ids]).max())
                    checks['max_diagonal_descriptor_error'] = max(checks['max_diagonal_descriptor_error'],error)
                    np.testing.assert_allclose(rebuilt,pair[key][ids],atol=2e-5,rtol=2e-6,err_msg=key)
                effect,edges = exact_effect_pairs(actual_flow,reference_flow,starts,target_valid,source_valid,args.device,budget)
                strict = contact_filter(edges,actual_contact,source_contact,canonical,starts)
                matrix = np.zeros_like(effect)
                matrix[strict['target'],strict['source']] = True
                np.testing.assert_array_equal(np.diag(effect),old_masks['effect'][ids])
                np.testing.assert_array_equal(np.diag(matrix),old_masks['strict'][ids])
                np.savez_compressed(output/'descriptors'/f'{sn:02d}.npz',canonical=canonical,
                    actual_pose=actual_pose,reference_pose=reference_pose,
                    actual_contact=np.packbits(actual_contact,axis=-1),source_contact=np.packbits(source_contact,axis=-1),
                    starts=starts,global_rows=ids,moving=moving,target_valid=target_valid,source_valid=source_valid)
                all_edges.append(dict(sequence_number=np.full(len(strict['target']),sn,np.int64),
                    target=ids[strict['target']],source=ids[strict['source']],
                    **{k:v for k,v in strict.items() if k not in ('target','source')}))
                delta = starts[None,:]-starts[:,None]
                for policy in POLICIES:
                    constraint = allowed(policy,delta)
                    for group,mask in [('strict',np.ones(len(ids),bool)),('strict_moving',moving)]:
                        result = selection_stats(strict,starts,mask,policy)
                        result['effect_windows'] = int(((effect&constraint).any(-1)&mask).sum())
                        for kind,key in [('best','selected_edge_indices'),('nonoverlap_greedy','nonoverlap_edge_indices')]:
                            for e in result.pop(key):
                                selected.append(dict(sequence_number=sn,policy=policy,group=group,kind=kind,
                                    target=int(ids[strict['target'][e]]),source=int(ids[strict['source'][e]])))
                        records.append(dict(sequence_number=sn,id=entry['id'],policy=policy,group=group,**result))
                log(f'{sn+1}/{len(entries)} {entry["id"]}: strict edges={len(strict["target"])}; moving recipients={int((matrix.any(-1)&moving).sum())}')
        check_inputs()
        checks['protected_inputs_and_previous_outputs_unchanged'] = True
        coverage = {}
        for policy in POLICIES:
            coverage[policy] = {}
            for group in ('strict','strict_moving'):
                rows = [r for r in records if r['policy']==policy and r['group']==group]
                fields = ('effect_windows','windows','edges','unique_selected_sources','one_to_one_capacity','nonoverlap_greedy')
                coverage[policy][group] = {k:sum(r[k] for r in rows) for k in fields}
                coverage[policy][group]['parents'] = sum(r['windows']>0 for r in rows)
        for group in ('strict','strict_moving'):
            ordered = [coverage[p][group]['windows'] for p in POLICIES[:5]]
            assert ordered==sorted(ordered)
        expected = [sum(old_masks[g][pair['sequence_number']<len(entries)]) for g in ('strict','strict_moving')]
        assert [coverage['sync'][g]['windows'] for g in ('strict','strict_moving')]==expected
        np.savez_compressed(output/'candidate_pairs.npz',**{k:np.concatenate([e[k] for e in all_edges]) for k in all_edges[0]})
        for name,rows in [('metrics.jsonl',records),('selected_pairs.jsonl',selected)]:
            with (output/name).open('w') as f:
                for row in rows:
                    f.write(json.dumps(row,ensure_ascii=False)+'\n')
        summary = dict(coverage=coverage,target_valid_windows=eligible,target_valid_moving_windows=moving_eligible,
            parent_count=len(entries),engineering_checks=checks,elapsed_seconds=time.monotonic()-started,
            peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
            conclusion='INCONCLUSIVE',interpretation='Availability diagnostic within fixed val bank; no decoder or training. Nonoverlap greedy is a lower bound, not independent trials.')
        write_json(output/'summary.json',summary)
        write_json(output/'verification.json',checks)
        write_json(output/'config.json',dict(arguments=vars(args),limits=LIMITS,policies=POLICIES,
            baseline=str(BASE),fps=30,window=4,source_surface_seed=2024,matching='same parent; no time scaling',
            minimum_windows=100,minimum_parents=10))
        write_json(output/'metadata.json',dict(protected_sha256=protected,input_stats=stats,
            sequence_entries=entries,previous_metadata=str(BASE/'metadata.json')))
        plot(summary,output)
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        assert size<1024**3
        budget()
        manifest.update(run_status='COMPLETED',last_step=int((pair['sequence_number']<len(entries)).sum()),
            evaluated_active_windows=int((pair['sequence_number']<len(entries)).sum()),output_bytes=size,
            elapsed_seconds=time.monotonic()-started,finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))
        write_json(output/'run_manifest.json',manifest)
        log('COMPLETED '+json.dumps(coverage['any_time']))
    except Exception as error:
        manifest.update(run_status='FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json',manifest)
        log('FAILED '+manifest['error'])
        raise


def plot(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    x = np.arange(len(POLICIES))
    rows = [summary['coverage'][p]['strict_moving'] for p in POLICIES]
    axes[0].bar(x-.2,[r['windows'] for r in rows],width=.4,label='Covered target windows')
    axes[0].bar(x+.2,[r['nonoverlap_greedy'] for r in rows],width=.4,label='Disjoint greedy pairs')
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('Strict moving count')
    axes[1].bar(x,[r['parents'] for r in rows])
    axes[1].set_ylabel('Parents with strict moving matches')
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(POLICIES,rotation=40,ha='right')
    fig.tight_layout()
    fig.savefig(output/'coverage.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
