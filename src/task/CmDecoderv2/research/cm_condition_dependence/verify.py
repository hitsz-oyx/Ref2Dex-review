"""Verify archived swaps, paired summaries and predictions with independent NumPy FK."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.task.CmDecoderv2.kinematics import InspireKinematics
from src.task.CmDecoderv2.pointflow import DifferentiableInspireSurface
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel
from .diagnostics import CONDITIONS, continuous_starts, paired_summary, select_donors, rotation_difference_deg
from .run import ROOT, sha256, write_json


def verify(directory):
    directory=Path(directory)
    metadata=json.loads((directory/'metadata.json').read_text())
    summary=json.loads((directory/'dependence_summary.json').read_text())
    bank=dict(np.load(directory/'cm_bank.npz',allow_pickle=False))
    donors=dict(np.load(directory/'donors.npz',allow_pickle=False))
    teacher=dict(np.load(directory/'teacher_predictions.npz',allow_pickle=False))
    rows=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
    rollout_rows=[json.loads(line) for line in (directory/'rollout_metrics.jsonl').read_text().splitlines()]
    for path,digest in metadata['protected_sha256'].items():
        assert sha256(path)==digest,path
    for path,expected in metadata['geometry_stats'].items():
        stat=Path(path).stat()
        assert [stat.st_size,stat.st_mtime_ns]==expected,path
    rebuilt=select_donors(bank)
    for key in rebuilt:
        np.testing.assert_array_equal(rebuilt[key],donors[key])
    eligible,_=continuous_starts(bank)
    rebuilt_rollout=select_donors(bank,eligible=eligible)
    for condition in ('swap','matched_swap'):
        np.testing.assert_array_equal(rebuilt_rollout[condition],donors['rollout_'+condition])
    lookup={(r['sample_id'],r['condition']):r for r in rows}
    assert len(lookup)==len(rows)==len(teacher['sample_id'])
    assert paired_summary([r for r in rows if r['current_valid']])==summary['teacher']
    for step in (1,4,8,16):
        assert paired_summary([r for r in rollout_rows if r['step']==step])==summary['rollout'][str(step)]
    for i,(sample,condition_id) in enumerate(zip(teacher['sample_id'],teacher['condition_id'])):
        sample=int(sample)
        condition=CONDITIONS[int(condition_id)]
        row=lookup[sample,condition]
        q_mae=float(np.mean(np.abs(teacher['q'][i]-bank['target_q'][sample])))
        t_error=float(np.linalg.norm(teacher['wrist'][i,:3,3]-bank['target_wrist'][sample,:3,3])*1000)
        np.testing.assert_allclose(row['q_mae_rad'],q_mae,atol=1e-6)
        np.testing.assert_allclose(row['wrist_translation_mm'],t_error,atol=1e-3)
        if condition in ('swap','matched_swap'):
            assert row['donor_id']==int(donors[condition][sample])
            assert bank['sequence_number'][row['donor_id']]==bank['sequence_number'][sample]
    urdf=ROOT/'src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf'
    kinematics=InspireKinematics(urdf)
    helper=InspireUrdfModel(urdf)
    surface=DifferentiableInspireSurface(urdf,sample_count=10135,surface_seed=2024,surface_sampling='v1_3_cache')
    local=surface.surface_points_local.numpy()
    visual_ids=surface.surface_visual_ids.numpy()
    def points(q,wrist):
        links=kinematics.link_transforms_from_state(q,wrist)
        output=np.empty(local.shape,dtype=np.float64)
        for visual_id in np.unique(visual_ids):
            mask=visual_ids==visual_id
            visual=helper.visuals[int(visual_id)]
            transform=links[visual.link]@visual.local_transform
            output[mask]=local[mask]@transform[:3,:3].T+transform[:3,3]
        return output
    checks=[]
    for condition_id,condition in enumerate(CONDITIONS):
        indices=np.flatnonzero(teacher['condition_id']==condition_id)
        if len(indices):
            checks.extend(indices[np.unique(np.linspace(0,len(indices)-1,3).round().astype(int))].tolist())
    max_error=0.
    for i in checks:
        sample=int(teacher['sample_id'][i])
        entry=metadata['sequence_entries'][int(bank['sequence_number'][sample])]
        target=np.load(Path(entry['geometry_root'])/'knn_hand_points_world.npy',mmap_mode='r')[int(bank['start_frame'][sample])+1]
        prediction=points(teacher['q'][i],teacher['wrist'][i])
        epe=float(np.linalg.norm(prediction-target,axis=-1).mean()*1000)
        recorded=lookup[sample,CONDITIONS[int(teacher['condition_id'][i])]]['hand_epe_mm']
        max_error=max(max_error,abs(epe-recorded))
        np.testing.assert_allclose(epe,recorded,atol=2e-3,rtol=1e-5)
    result=dict(teacher_rows=len(rows),rollout_rows=len(rollout_rows),donors_reproducible=True,
                paired_summaries_reproducible=True,protected_inputs_and_code_unchanged=True,
                independent_numpy_fk_samples=len(checks),max_numpy_fk_epe_difference_mm=max_error)
    write_json(directory/'verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
