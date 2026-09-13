"""独立重建阈值标记，检查导出候选及时间区间。"""
import argparse
import json
from pathlib import Path
import numpy as np
from .run import TRACK,PAIR,sha256,write_json


def read_rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def verify(directory):
    directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text())
    rows=read_rows(directory/'window_metrics.jsonl');candidates=read_rows(directory/'candidate_manifest.jsonl')
    summary=json.loads((directory/'summary.json').read_text());config=json.loads((directory/'config.json').read_text())
    for p,digest in meta['protected_sha256'].items():assert sha256(p)==digest,p
    for p,expected in meta['input_stats'].items():
        s=Path(p).stat();assert [s.st_size,s.st_mtime_ns]==expected,p
    max_error=0.
    for sn,entry in enumerate(meta['sequence_entries']):
        d=dict(np.load(PAIR/'descriptors'/f'{sn:02d}.npz'));t=dict(np.load(TRACK/f'trajectory_{sn:02d}.npz'))
        flow=[]
        for key in ('actual_pose','geometric_pose'):
            p=t[key].astype(float);delta=np.linalg.solve(p[:-1],p[1:])
            flow.append((d['canonical']@(delta[:,:3,:3]-np.eye(3)).transpose(0,2,1)+delta[:,None,:3,3])*1000)
        epe=np.linalg.norm(flow[0]-flow[1],axis=-1).mean(-1)
        limit=np.maximum(1,.25*np.maximum(*[np.sqrt((v**2).sum(-1).mean(-1)) for v in flow]))
        local=[r for r in rows if r['sequence_number']==sn]
        assert len(local)==len(d['starts'])
        for r,start in zip(local,d['starts']):
            ratio=(epe/limit)[start:start+4].max();max_error=max(max_error,abs(ratio-r['effect_ratio_max']))
            assert r['effect']==bool(ratio<=1)
            np.testing.assert_allclose(ratio,r['effect_ratio_max'],atol=1e-8)
            for name,(mm,deg) in config['profiles'].items():
                bp=bool((t['tracking_position_mm'][start:start+5]>mm).any())
                br=bool((t['tracking_rotation_deg'][start:start+5]>deg).any())
                assert r[name]==(not(bp or br))
                assert r['combined'+name[4:]]==(r['effect'] and r[name])
                assert r[name+'_reasons']==int(not r['effect'])+2*bp+4*br
    row_by_id={r['row']:r for r in rows}
    original=read_rows(PAIR/'selected_pairs.jsonl')
    best={r['target']:r['source'] for r in original if r['policy']=='any_time' and r['group']=='strict_moving' and r['kind']=='best'}
    edge_file=dict(np.load(PAIR/'candidate_pairs.npz'));edge_set=set(zip(edge_file['target'],edge_file['source']))
    occupied={}
    for c in candidates:
        r=row_by_id[c['target_row']];s=row_by_id[c['source_row']]
        assert c['diagnostic_only'] and c['split']=='val'
        assert r['id']==s['id']==c['id'] and r['target_valid'] and r['moving'] and r[c['mode']]
        assert (c['target_row'],c['source_row']) in edge_set
        assert r['start']==c['target_start'] and s['start']==c['source_start']
        key=(c['mode'],c['kind'],c['id'])
        a,b=occupied.setdefault(key,(set(),set()))
        if c['kind']=='best':
            assert best[c['target_row']]==c['source_row'] and c['target_row'] not in a
            a.add(c['target_row'])
        else:
            tf=set(range(c['target_start'],c['target_start']+5));sf=set(range(c['source_start'],c['source_start']+5))
            assert not(a&tf or b&sf);a.update(tf);b.update(sf)
    for mode in config['modes']:
        best_rows=[c for c in candidates if c['mode']==mode and c['kind']=='best']
        assert len(best_rows)==summary[mode]['windows']
        assert len({c['id'] for c in best_rows})==summary[mode]['parents']
        assert sum(c['mode']==mode and c['kind']=='nonoverlap_greedy' for c in candidates)==summary[mode]['nonoverlap_greedy']
        assert sum(r['target_valid'] and r['moving'] and r[mode] for r in rows)==summary[mode]['quality_motion_windows']
    result=dict(windows_verified=len(rows),candidate_rows_verified=len(candidates),max_effect_ratio_difference=max_error,
        independent_flags_reproduced=True,candidate_identity_and_nonoverlap_valid=True,old_inputs_unchanged=True)
    write_json(directory/'independent_verification.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');print(json.dumps(verify(p.parse_args().directory),indent=2))
