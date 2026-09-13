"""独立四元数测地距离、刚体位移及lag选择复核。"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .run import ROOT,BASE,sha256,write_json,_load_tensor
from .diagnostics import LAGS


def verify(directory):
    directory=Path(directory)
    meta=json.loads((directory/'metadata.json').read_text())
    rows=[json.loads(x) for x in (directory/'metrics.jsonl').read_text().splitlines()]
    for p,digest in meta['protected_sha256'].items():assert sha256(p)==digest,p
    for p,expected in meta['input_stats'].items():
        s=Path(p).stat();assert [s.st_size,s.st_mtime_ns]==expected,p
    checked,max_error,max_rotation_error=0,0.,0.
    for sn,entry in enumerate(meta['sequence_entries']):
        d=dict(np.load(directory/f'trajectory_{sn:02d}.npz'))
        geom=dict(np.load(BASE/'descriptors'/f'{sn:02d}.npz'))
        raw=[]
        for folder in ('inspire_geometric_dexplore','inspire_rl_object_dexplore'):
            raw.append(_load_tensor(ROOT/'data/processed_data'/folder/entry['id'].replace('/','_')/'interaction_hand_inspire.pt'))
        ref,actual=raw
        np.testing.assert_allclose(np.linalg.norm(actual[:,198:201].astype(float)-ref[:,198:201],axis=-1)*1000,d['tracking_position_mm'],atol=1e-7)
        angle=np.degrees((Rotation.from_quat(ref[:,201:205]).inv()*Rotation.from_quat(actual[:,201:205])).magnitude())
        max_rotation_error=max(max_rotation_error,float(np.abs(angle-d['tracking_rotation_deg']).max()))
        np.testing.assert_allclose(angle,d['tracking_rotation_deg'],atol=1e-4)
        t=d['lag_support'];curve=d['lag_errors_mm']
        for index in np.unique(np.linspace(0,len(t)-1,min(12,len(t))).astype(int)):
            for lag in (-15,0,15):
                flows=[]
                for key,frame in [('actual_pose',t[index]),('geometric_pose',t[index]+lag)]:
                    p=d[key].astype(float)
                    relative=np.linalg.solve(p[frame],p[frame+1])
                    flows.append((geom['canonical']@(relative[:3,:3]-np.eye(3)).T+relative[:3,3])*1000)
                error=np.linalg.norm(flows[0]-flows[1],axis=-1).mean()
                delta=abs(error-curve[lag+15,index]);max_error=max(max_error,float(delta))
                np.testing.assert_allclose(error,curve[lag+15,index],atol=2e-5,rtol=1e-6)
                checked+=1
        if len(t)>=4:
            split=len(t)//2
            best=int(np.lexsort((LAGS,np.abs(LAGS),curve[:,:split].mean(-1)))[0])
            assert int(LAGS[best])==rows[sn]['lag']['selected_lag']
            np.testing.assert_allclose(curve[best,split:].mean(),rows[sn]['lag']['heldout_selected_mm'])
    result=dict(protected_inputs_unchanged=True,native_rotation_checked_frames=sum(r['frames'] for r in rows),
        max_native_rotation_difference_deg=max_rotation_error,independent_pointflow_checks=checked,
        max_pointflow_difference_mm=max_error,lag_selection_and_heldout_reproduced=True)
    write_json(directory/'independent_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');print(json.dumps(verify(p.parse_args().directory),indent=2))
