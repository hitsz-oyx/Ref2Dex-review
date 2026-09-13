"""独立 NumPy FK 与归档指标、配对覆盖核验。"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.task.CmDecoderv2.kinematics import InspireKinematics
from src.task.CmDecoderv2.pointflow import DifferentiableInspireSurface
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel
from .run import ROOT, BASE, sha256, write_json
from .diagnostics import CONDITIONS, gates, shifted_donors, summarize


def verify(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    pair = dict(np.load(directory/'pair_bank.npz'))
    predictions = dict(np.load(directory/'predictions.npz'))
    arrays = dict(np.load(directory/'metric_arrays.npz'))
    metrics = {c: {k[len(c)+1:]:v for k,v in arrays.items() if k.startswith(c+'_')} for c in CONDITIONS}
    for path,digest in meta['protected_sha256'].items():
        assert sha256(path)==digest,path
    for path,expected in meta['input_stats'].items():
        s = Path(path).stat()
        assert [s.st_size,s.st_mtime_ns]==expected,path
    assert summarize(pair,metrics)==summary['conditions']
    np.testing.assert_array_equal(pair['donor'],shifted_donors(pair['sequence_number'],pair['start_frame'],pair['mano_valid'].all(-1)))
    for name, mask in gates(pair).items():
        assert summary['coverage'][name]==dict(windows=int(mask.sum()),parents=len(np.unique(pair['sequence_number'][mask])))
    entries = json.loads((BASE/'metadata.json').read_text())['sequence_entries']
    urdf = ROOT/'src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf'
    kinematics, helper = InspireKinematics(urdf), InspireUrdfModel(urdf)
    surface = DifferentiableInspireSurface(urdf,sample_count=10135,surface_seed=2024,surface_sampling='v1_3_cache')
    local, visual_ids = surface.surface_points_local.numpy(), surface.surface_visual_ids.numpy()
    def points(q,wrist):
        links = kinematics.link_transforms_from_state(q,wrist)
        out = np.empty(local.shape,np.float64)
        for vid in np.unique(visual_ids):
            mask = visual_ids==vid
            visual = helper.visuals[int(vid)]
            transform = links[visual.link] @ visual.local_transform
            out[mask] = local[mask] @ transform[:3,:3].T+transform[:3,3]
        return out
    max_error, checked = 0.,0
    # Cover every condition and every represented parent, including invalid-Cm windows.
    for c in CONDITIONS:
        for sn in np.unique(pair['sequence_number']):
            candidates = np.flatnonzero(pair['sequence_number']==sn)
            i = int(candidates[len(candidates)//2])
            entry = entries[int(sn)]
            gt = np.load(Path(entry['geometry_root'])/'knn_hand_points_world.npy',mmap_mode='r')[int(pair['start_frame'][i])+1]
            predicted = points(predictions[c+'_q'][i],predictions[c+'_wrist'][i])
            correct = points(predictions['correct_q'][i],predictions['correct_wrist'][i])
            for key,value in [('hand_epe_mm', np.linalg.norm(predicted-gt,axis=-1).mean()*1000),
                              ('output_change_mm',np.linalg.norm(predicted-correct,axis=-1).mean()*1000)]:
                error = abs(value-metrics[c][key][i])
                max_error = max(max_error,float(error))
                np.testing.assert_allclose(value,metrics[c][key][i],atol=.002,rtol=1e-5)
            checked += 1
    result = dict(archived_summaries_and_donors_reproducible=True,protected_inputs_unchanged=True,
                  independent_numpy_fk_samples=checked,max_numpy_fk_error_mm=max_error)
    write_json(directory/'independent_verification.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
