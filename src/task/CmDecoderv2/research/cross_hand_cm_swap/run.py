"""冻结模型的 MANO→Inspire 单步配对诊断；不写正式 cache。"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pytorch3d.ops import knn_points

from src.task.CmDecoderv2.dataset import CmDecoderV2Dataset, _collate_cm_decoder
from src.task.CmDecoderv2.model import CmDecoderV2
from src.task.CmDecoderv2.research.cm_condition_dependence.run import (
    ROOT, CHECKPOINT, CHECKPOINT_SHA, namespace, sha256, state_digest, write_json,
    decode_core, error_values,
)
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_v1_3_cache import (
    _build_mano_pool, _sample_correspondence, _sample_mano_surface,
)
from .diagnostics import (CONDITIONS, LIMITS, source_window, local_object_flow,
                          contact_comparison, gates, shifted_donors, summarize)

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'cm_condition_dependence/output/cm_dependence_val_20260912_145016'
CM_KEYS = ('cm_tokens', 'anchor_pos', 'anchor_normal', 'cm_valid')


def run(args):
    started = time.monotonic()
    output = HERE / 'output' / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    def log(message):
        line = datetime.now().astimezone().isoformat(timespec='seconds') + ' ' + message
        print(line, flush=True)
        with (output / 'run.log').open('a') as f:
            f.write(line + '\n')
    def budget():
        if time.monotonic() - started > 1800:
            raise TimeoutError('30 minute budget')
        if torch.cuda.max_memory_allocated() > 8 * 1024**3:
            raise MemoryError('8 GiB allocation budget')
    manifest = dict(task='CmDecoderv2', work_version='V1.1.9',
        schema_name='ref2dex.cross_hand_cm_swap.v1', run_id=args.run_id,
        activity_id=args.run_id, run_status='STARTED', seed=42,
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        worktree_dirty=True, config_snapshot='config.json', metadata_snapshot='metadata.json',
        checkpoint=str(CHECKPOINT), output=str(output), conclusion='INCONCLUSIVE',
        started_at=datetime.now().astimezone().isoformat(timespec='seconds'))
    write_json(output / 'run_manifest.json', manifest)
    try:
        oldmeta = json.loads((BASE / 'metadata.json').read_text())
        protected = dict(oldmeta['protected_sha256'])
        for path in list(BASE.glob('*')) + list(HERE.glob('*.py')):
            if path.is_file():
                protected[str(path)] = sha256(path)
        for path in [ROOT/'src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_v1_3_cache.py',
                     ROOT/'src/task/ObjectInteractionCm/research/hand_region_sampling/run.py',
                     ROOT/'src/task/ObjectInteractionCm/research/hand_region_sampling/build_trajectory_preview.py',
                     HERE.parent/'inspire_rollout_effect/run.py']:
            protected[str(path)] = sha256(path)
        for path, digest in protected.items():
            assert sha256(path) == digest, path
        stats = dict(oldmeta['geometry_stats'])
        def watch(path):
            path = Path(path)
            s = path.stat()
            key = str(path.resolve())
            if key in stats:
                assert stats[key] == [s.st_size, s.st_mtime_ns], key
            stats[key] = [s.st_size, s.st_mtime_ns]
        for path in stats.copy():
            watch(path)
        assert sha256(CHECKPOINT) == CHECKPOINT_SHA
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        cfg = namespace(ckpt['config'])
        cfg.model.meta = cfg.meta
        torch.set_num_threads(4)
        torch.manual_seed(42)
        model = CmDecoderV2(cfg.model).to(args.device).requires_grad_(False).eval()
        model.load_state_dict(ckpt['model'], strict=True)
        oicm = torch.load(model.oicm_checkpoint, map_location='cpu', weights_only=False)['model']
        assert all(torch.equal(model.oicm.state_dict()[k].cpu(), v) for k, v in oicm.items())
        before = state_digest(model)
        entries = oldmeta['sequence_entries'][:2] if args.smoke else oldmeta['sequence_entries']
        dataset = CmDecoderV2Dataset(entries, urdf_path=ROOT / cfg.data.urdf_path,
            window_size=4, num_obj_points=1024, num_hand_points=10135,
            hand_stream_mode='unique_knn_edges', knn_k=32,
            hand_supervision_radius_m=.02, seed=42, perturb=False, active_only=True)
        dataset.set_epoch(0)
        with np.load(BASE / 'cm_bank.npz') as saved:
            selected = saved['sequence_number'] < len(entries)
            bank = {k: saved[k][selected] for k in saved.files}
        assert len(bank['start_frame']) == len(dataset)
        count = len(bank['start_frame'])
        index_path = ROOT / cfg.data.index_path
        index = json.loads(index_path.read_text())
        parent_base = Path(index['source_roots']['grab_parent_cache'])
        pool, _ = _build_mano_pool(ROOT / 'dataset/arctic/data/body_models/mano', .20)
        correspondence = _sample_correspondence(pool, 2048, 2024)
        for p in (ROOT / 'dataset/arctic/data/body_models/mano').glob('*'):
            if p.is_file():
                watch(p)
        source_banks = {name: {k: np.zeros_like(bank[k]) for k in CM_KEYS}
                        for name in ('mano_sync', 'mano_transported')}
        pair = dict(sequence_number=bank['sequence_number'], start_frame=bank['start_frame'],
                    target_valid=bank['cm_valid'])
        for key in ('actual_rms_mm', 'reference_rms_mm', 'effect_epe_mm',
                    'contact_iou', 'contact_centroid_mm'):
            pair[key] = np.zeros((count, 4), dtype=np.float32)
        checks = dict(source_window_max_difference=0., correct_cm_replay_max_difference=0.,
                      canonical_template_max_difference_m=0., mano_cache_replay_max_difference_m=0.)
        provenance = []

        def encode(windows):
            batch = _collate_cm_decoder(windows)
            return [v.cpu().numpy() for v in model._encode_cm_window(
                {k: v.to(args.device) for k, v in batch.items()})]

        # Replay official validation MANO sampling without using it as an evaluation recipient.
        official = next(e for e in index['sequences']['val'] if e['variant'] == 'mano')
        geo = index_path.parent / official['path'] / 'geometry'
        raw = np.load(geo / 'source_frame_id.npy')[:5]
        pose = np.load(geo / 'obj_pose_world.npy')[:5]
        replay = np.empty((5, 2048, 3), np.float32)
        normals = np.empty_like(replay)
        official_parent = parent_base / official['id']
        for p in [geo/'source_frame_id.npy', geo/'obj_pose_world.npy',
                  official_parent/'shared/obj_pose_world.npy', official_parent/'shared/raw_frame_id.npy',
                  official_parent/'right/hand_mesh_vertices_world.npy', official_parent/'right/hand_mesh_faces.npy']:
            watch(p)
        _sample_mano_surface(official_parent, raw, pose, correspondence, replay, normals, batch_size=8)
        for key, value in [('knn_hand_points_world.npy', replay), ('knn_hand_normals_world.npy', normals)]:
            p = geo / key
            watch(p)
            error = float(np.max(np.abs(value - np.load(p, mmap_mode='r')[:5])))
            checks['mano_cache_replay_max_difference_m'] = max(checks['mano_cache_replay_max_difference_m'], error)
            assert error < 2e-6, (key, error)

        with torch.inference_mode():
            for sn, target in enumerate(dataset.sequences):
                budget()
                ids = np.flatnonzero(bank['sequence_number'] == sn)
                parent = parent_base / target.id
                for p in [parent/'shared'/n for n in ('obj_pose_world.npy', 'obj_points_world.npy', 'raw_frame_id.npy')]:
                    watch(p)
                for n in ('hand_mesh_vertices_world.npy', 'hand_mesh_faces.npy'):
                    watch(parent/'right'/n)
                raw_parent = np.load(parent/'shared/raw_frame_id.npy')
                lookup = {int(r): i for i, r in enumerate(raw_parent)}
                aligned = np.array([lookup[int(r)] for r in target.source_frame])
                reference_pose = np.load(parent/'shared/obj_pose_world.npy')[aligned].astype(np.float32)
                actual_pose = np.asarray(target.object_pose)
                canonical = ((target.object_points[0] - actual_pose[0, :3, 3]) @ actual_pose[0, :3, :3]).astype(np.float32)
                original = np.load(parent/'shared/obj_points_world.npy', mmap_mode='r')
                # Every aligned frame, with exact point correspondence; no registration fit.
                for t in range(len(aligned)):
                    a = (target.object_points[t]-actual_pose[t,:3,3]) @ actual_pose[t,:3,:3]
                    b = (original[aligned[t]]-reference_pose[t,:3,3]) @ reference_pose[t,:3,:3]
                    err = float(max(np.abs(a-canonical).max(), np.abs(b-canonical).max()))
                    checks['canonical_template_max_difference_m'] = max(checks['canonical_template_max_difference_m'], err)
                    assert err < 2e-5, (target.id, t, err)
                native_points = np.empty((len(aligned), 2048, 3), np.float32)
                native_normals = np.empty_like(native_points)
                _sample_mano_surface(parent, np.asarray(target.source_frame), reference_pose,
                    correspondence, native_points, native_normals, batch_size=16)
                local_hand = (native_points-reference_pose[:,None,:3,3]) @ reference_pose[:,:3,:3]
                local_normals = native_normals @ reference_pose[:,:3,:3]
                local_obj = (np.asarray(target.object_points)-actual_pose[:,None,:3,3]) @ actual_pose[:,:3,:3]
                local_obj_normals = np.asarray(target.object_normals) @ actual_pose[:,:3,:3]
                edges = np.empty((len(aligned), 4096, 32), np.int64)
                contact = np.empty((len(aligned), 4096), bool)
                for t in range(0, len(aligned), 8):
                    knn = knn_points(torch.as_tensor(local_obj[t:t+8], device=args.device),
                                     torch.as_tensor(local_hand[t:t+8], device=args.device), K=32)
                    edges[t:t+8] = knn.idx.cpu().numpy()
                    contact[t:t+8] = (knn.dists[:,:,0] <= .02**2).cpu().numpy()
                actual_flow = local_object_flow(canonical, actual_pose)
                reference_flow = local_object_flow(canonical, reference_pose)
                target_contact = np.load(Path(entries[sn]['geometry_root'])/'obj_candidate_mask_2cm.npy')
                iou, centroid = contact_comparison(canonical, target_contact, contact)
                desc = dict(actual_rms_mm=np.sqrt((actual_flow**2).sum(-1).mean(-1)),
                    reference_rms_mm=np.sqrt((reference_flow**2).sum(-1).mean(-1)),
                    effect_epe_mm=np.linalg.norm(actual_flow-reference_flow, axis=-1).mean(-1),
                    contact_iou=iou, contact_centroid_mm=centroid)
                for key, value in desc.items():
                    pair[key][ids] = value[bank['start_frame'][ids,None] + np.arange(4)]
                if len(ids):
                    row = int(ids[0])
                    window = source_window(target, int(bank['start_frame'][row]))
                    official_window = dataset[row]
                    for k, v in window.items():
                        delta = float((v.double()-official_window[k].double()).abs().max())
                        checks['source_window_max_difference'] = max(checks['source_window_max_difference'], delta)
                        assert delta < 2e-6, (k, delta)
                    replay_cm = encode([window])
                    for key, value in zip(CM_KEYS, replay_cm):
                        delta = float(np.max(np.abs(value.astype(float)-bank[key][row:row+1].astype(float))))
                        checks['correct_cm_replay_max_difference'] = max(checks['correct_cm_replay_max_difference'], delta)
                        assert delta < 5e-4, (key, delta)
                for name, poses in [('mano_sync', reference_pose), ('mano_transported', actual_pose)]:
                    source = SimpleNamespace(id=target.id, source_frame=target.source_frame,
                        object_pose=poses,
                        object_points=local_obj @ poses[:,:3,:3].transpose(0,2,1)+poses[:,None,:3,3],
                        object_normals=local_obj_normals @ poses[:,:3,:3].transpose(0,2,1),
                        knn_hand_points=local_hand @ poses[:,:3,:3].transpose(0,2,1)+poses[:,None,:3,3],
                        knn_hand_normals=local_normals @ poses[:,:3,:3].transpose(0,2,1),
                        knn_edge_indices=edges)
                    for j in range(0, len(ids), args.batch_size):
                        budget()
                        rows = ids[j:j+args.batch_size]
                        values = encode([source_window(source, int(bank['start_frame'][i])) for i in rows])
                        for key, value in zip(CM_KEYS, values):
                            source_banks[name][key][rows] = value
                provenance.append(dict(id=target.id, parent=str(parent), windows=len(ids), frames=len(aligned)))
                log(f'encoded {sn+1}/{len(entries)} {target.id}: {len(ids)} windows')

            pair['mano_valid'] = source_banks['mano_sync']['cm_valid']
            pair['donor'] = shifted_donors(pair['sequence_number'], pair['start_frame'], pair['mano_valid'].all(-1))
            predictions, metrics = {}, {}
            for condition in CONDITIONS:
                active_bank = dict(bank)
                if condition.startswith('mano'):
                    source_name = 'mano_sync' if condition == 'mano_shift' else condition
                    source = source_banks[source_name]
                    donor = np.maximum(pair['donor'], 0) if condition == 'mano_shift' else np.arange(count)
                    active_bank.update({k: source[k][donor] for k in CM_KEYS})
                available = np.ones(count, bool) if condition in ('correct','identity') else active_bank['cm_valid'].all(-1)
                if condition == 'mano_shift':
                    available &= pair['donor'] >= 0
                results = {k: [] for k in ('q','wrist','hand_epe_mm','q_mae_rad','wrist_translation_mm','wrist_rotation_deg','output_change_mm')}
                for start in range(0, count, args.batch_size):
                    budget()
                    ids = np.arange(start, min(count, start+args.batch_size))
                    q, wrist = decode_core(model, active_bank, ids, 'identity' if condition=='identity' else 'correct', None, args.device)
                    points = model.point_surface(q,wrist)
                    gt = np.stack([dataset.sequences[int(bank['sequence_number'][i])].knn_hand_points[int(bank['start_frame'][i])+1] for i in ids])
                    errors = error_values(points, torch.as_tensor(gt, device=args.device), q,
                        torch.as_tensor(bank['target_q'][ids], device=args.device), wrist,
                        torch.as_tensor(bank['target_wrist'][ids], device=args.device))
                    if condition == 'correct':
                        change = np.zeros(len(ids), np.float32)
                    else:
                        cq = torch.as_tensor(predictions['correct_q'][ids], device=args.device)
                        cw = torch.as_tensor(predictions['correct_wrist'][ids], device=args.device)
                        change = (torch.linalg.vector_norm(points-model.point_surface(cq,cw), dim=-1).mean(-1)*1000).cpu().numpy()
                    for k, v in dict(q=q.cpu().numpy(), wrist=wrist.cpu().numpy(), output_change_mm=change, **errors).items():
                        results[k].append(v)
                results = {k: np.concatenate(v) for k, v in results.items()}
                predictions.update({condition+'_'+k: results.pop(k) for k in ('q','wrist')})
                metrics[condition] = dict(available=available, **results)
                log(f'decoded {condition}')
        # Replay all correct h1 predictions against the earlier independent diagnostic.
        with np.load(BASE/'teacher_predictions.npz') as old:
            mask = (old['condition_id']==0) & (old['sample_id'] < count)
            order = np.argsort(old['sample_id'][mask])
            for key in ('q','wrist'):
                error = float(np.max(np.abs(old[key][mask][order]-predictions['correct_'+key])))
                checks['correct_'+key+'_replay_max_difference'] = error
                assert error < 5e-5, (key,error)
        assert state_digest(model) == before
        for path, digest in protected.items():
            assert sha256(path) == digest, path
        for path, expected in stats.items():
            s = Path(path).stat()
            assert [s.st_size,s.st_mtime_ns] == expected, path
        checks.update(model_unchanged=True, inputs_unchanged=True, fixed_recipient_state=True,
                      complete_cm_replacement=True, future_target_not_in_core=True)
        coverage = {k: dict(windows=int(v.sum()), parents=len(np.unique(pair['sequence_number'][v]))) for k,v in gates(pair).items()}
        summary = dict(coverage=coverage, conditions=summarize(pair,metrics), engineering_checks=checks,
            conclusion='INCONCLUSIVE', elapsed_seconds=time.monotonic()-started,
            peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
            interpretation='Single-step reference compatibility; transported source injects target object motion; validation is not held-out test.')
        np.savez_compressed(output/'pair_bank.npz', **pair)
        np.savez_compressed(output/'source_cm.npz', **{c+'_'+k:v for c,b in source_banks.items() for k,v in b.items()})
        np.savez_compressed(output/'predictions.npz', **predictions)
        np.savez_compressed(output/'metric_arrays.npz', **{c+'_'+k:v for c,b in metrics.items() for k,v in b.items()})
        with (output/'metrics.jsonl').open('w') as f:
            for c, arrays in metrics.items():
                for i in range(count):
                    f.write(json.dumps(dict(condition=c, sample_id=i, sequence=int(pair['sequence_number'][i]),
                        **{k:bool(v[i]) if k=='available' else float(v[i]) for k,v in arrays.items()}))+'\n')
        write_json(output/'summary.json', summary)
        write_json(output/'verification.json', checks)
        write_json(output/'config.json', dict(arguments=vars(args), limits=LIMITS, checkpoint_config=ckpt['config'],
            baseline=str(BASE), source_surface_seed=2024, coordinate_frame='each_transition_object_pose_t', fps=30))
        write_json(output/'metadata.json', dict(provenance=provenance, protected_sha256=protected,
            input_stats=stats, model_state_digest=before, checkpoint_sha256=CHECKPOINT_SHA))
        plot(summary, output)
        size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        assert size < 2*1024**3
        budget()
        manifest.update(run_status='COMPLETED', last_step=count, output_bytes=size,
                        finished_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                        elapsed_seconds=time.monotonic()-started)
        write_json(output/'run_manifest.json', manifest)
        log('COMPLETED '+json.dumps(coverage))
    except Exception as error:
        manifest.update(run_status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(output/'run_manifest.json', manifest)
        log('FAILED '+manifest['error'])
        raise


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,4,figsize=(15,4))
    for ax,(group,conditions) in zip(axes,summary['conditions'].items()):
        values = [conditions[c]['hand_epe_mm'] for c in CONDITIONS]
        ax.bar(np.arange(5), [v['micro'] if v else 0 for v in values])
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(CONDITIONS,rotation=65,ha='right',fontsize=8)
        ax.set_title(group+' n='+str(summary['coverage'][group]['windows']))
        ax.set_ylabel('Reference hand EPE (mm)')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
