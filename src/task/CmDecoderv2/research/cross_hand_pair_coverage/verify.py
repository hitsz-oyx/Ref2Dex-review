"""从归档姿态独立计算 point displacement，对局部网格穷举复核。"""
import argparse
import json
from pathlib import Path
import numpy as np
from .run import sha256,write_json
from .matching import allowed,POLICIES


def verify(directory):
    directory = Path(directory)
    meta = json.loads((directory/'metadata.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    all_edges = dict(np.load(directory/'candidate_pairs.npz'))
    selected = [json.loads(s) for s in (directory/'selected_pairs.jsonl').read_text().splitlines()]
    records = [json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
    for p,digest in meta['protected_sha256'].items():
        assert sha256(p)==digest,p
    for p,expected in meta['input_stats'].items():
        s = Path(p).stat()
        assert [s.st_size,s.st_mtime_ns]==expected,p
    assert (all_edges['ratio']<=1).all()
    assert (all_edges['contact_iou']>=.25).all()
    assert (all_edges['centroid_mm']<=20).all()
    checked,max_error = 0,0.
    for sn,entry in enumerate(meta['sequence_entries']):
        d = dict(np.load(directory/'descriptors'/f'{sn:02d}.npz'))
        ids,starts = d['global_rows'],d['starts']
        lookup = {int(row):i for i,row in enumerate(ids)}
        points = d['canonical'].astype(np.float64)
        a = np.unpackbits(d['actual_contact'],axis=-1,count=len(points)).astype(bool)
        b = np.unpackbits(d['source_contact'],axis=-1,count=len(points)).astype(bool)
        def flow(pose):
            pose = pose.astype(np.float64)
            relative = np.linalg.solve(pose[:-1],pose[1:])
            return (points @ (relative[:,:3,:3]-np.eye(3)).transpose(0,2,1)+relative[:,None,:3,3])*1000
        actual,reference = flow(d['actual_pose']),flow(d['reference_pose'])
        existing = { (int(t),int(s)):i for i,(t,s) in enumerate(zip(all_edges['target'],all_edges['source']))
                     if all_edges['sequence_number'][i]==sn }
        chosen = {int(t) for pair in existing for t in pair}
        subset = np.unique(np.linspace(0,len(ids)-1,min(6,len(ids))).round().astype(int)).tolist()
        if chosen:
            more = sorted(chosen)
            subset += [lookup[more[0]],lookup[more[len(more)//2]],lookup[more[-1]]]
        subset = sorted(set(subset))
        for i in subset:
            for j in subset:
                t,s = int(starts[i]),int(starts[j])
                error = np.linalg.norm(actual[t:t+4]-reference[s:s+4],axis=-1).mean(-1)
                limit = np.maximum(1,.25*np.maximum(np.sqrt((actual[t:t+4]**2).sum(-1).mean(-1)),
                                                     np.sqrt((reference[s:s+4]**2).sum(-1).mean(-1))))
                iou,gap = [],[]
                for k in range(4):
                    ma,mb = a[t+k],b[s+k]
                    iou.append((ma&mb).sum()/max((ma|mb).sum(),1))
                    gap.append(np.linalg.norm(points[ma].mean(0)-points[mb].mean(0))*1000 if ma.any() and mb.any() else 1e6)
                good = bool(d['target_valid'][i] and d['source_valid'][j] and (error<=limit).all()
                            and (np.asarray(iou)>=.25).all() and (np.asarray(gap)<=20).all())
                key = (int(ids[i]),int(ids[j]))
                assert good == (key in existing),(entry['id'],i,j)
                if good:
                    ei = existing[key]
                    for name,value in [('epe_mm',error),('ratio',error/limit),('contact_iou',iou),('centroid_mm',gap)]:
                        discrepancy = float(np.abs(np.asarray(value)-all_edges[name][ei]).max())
                        max_error = max(max_error,discrepancy)
                        np.testing.assert_allclose(value,all_edges[name][ei],atol=2e-5,rtol=1e-6)
                checked += 1
        for policy in POLICIES:
            for group in ('strict','strict_moving'):
                selected_rows = [r for r in selected if r['sequence_number']==sn and r['policy']==policy and r['group']==group]
                metric = next(r for r in records if r['sequence_number']==sn and r['policy']==policy and r['group']==group)
                for kind,count_key in [('best','windows'),('nonoverlap_greedy','nonoverlap_greedy')]:
                    rows = [r for r in selected_rows if r['kind']==kind]
                    assert len(rows)==metric[count_key]
                    used_t,used_s = set(),set()
                    for r in rows:
                        key = (r['target'],r['source'])
                        assert key in existing
                        i,j = lookup[key[0]],lookup[key[1]]
                        assert allowed(policy,np.array(starts[j]-starts[i]))
                        if group=='strict_moving':
                            assert d['moving'][i]
                        if kind=='nonoverlap_greedy':
                            tf,sf = set(range(int(starts[i]),int(starts[i])+5)),set(range(int(starts[j]),int(starts[j])+5))
                            assert not (tf&used_t or sf&used_s)
                            used_t.update(tf)
                            used_s.update(sf)
                        else:
                            assert i not in used_t
                            used_t.add(i)
                            # Independent lexicographic best donor check, using only archived edge descriptors.
                            options = [(all_edges['ratio'][ei].mean(),abs(starts[lookup[s]]-starts[i]),starts[lookup[s]],s)
                                       for (t,s),ei in existing.items() if t==key[0] and
                                       allowed(policy,np.array(starts[lookup[s]]-starts[i]))]
                            assert min(options)[-1]==key[1]
        for r in [r for r in records if r['sequence_number']==sn]:
            assert r['nonoverlap_greedy']<=r['one_to_one_capacity']<=r['windows']
    for policy in POLICIES:
        for group in ('strict','strict_moving'):
            rows = [r for r in records if r['policy']==policy and r['group']==group]
            for key,value in summary['coverage'][policy][group].items():
                expected = sum(r['windows']>0 for r in rows) if key=='parents' else sum(r[key] for r in rows)
                assert value==expected
    result = dict(protected_inputs_unchanged=True,independent_bruteforce_pairs=checked,
        max_descriptor_difference=max_error,saved_edge_thresholds_valid=True,
        selected_donors_and_disjoint_intervals_valid=True,summary_reproducible=True)
    write_json(directory/'independent_verification.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    print(json.dumps(verify(parser.parse_args().directory),indent=2))
