#!/usr/bin/env python3
"""Build the V1.4 geometry index while preserving the GRAB split."""
import argparse,json
from pathlib import Path

def lines(p): return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 a=argparse.ArgumentParser(); a.add_argument('--grab-root',type=Path,required=True); a.add_argument('--arctic-root',type=Path,required=True); a.add_argument('--oak-root',type=Path); a.add_argument('--split-root',type=Path,required=True); a.add_argument('--output',type=Path,required=True); x=a.parse_args()
 entries={k:[] for k in ('train','val','test')}; missing=[]
 for split in entries:
  for sid in lines(x.split_root/(split+'.txt')):
   p=x.grab_root/sid
   if (p/'geometry/manifest.json').is_file(): entries[split].append({'source':'inspire_f1','id':'grab/'+sid,'path':str(p.resolve()),'dataset':'grab'})
   else: missing.append('grab/'+sid)
 for p in sorted(x.arctic_root.glob('**/geometry/manifest.json')):
  seq=p.parent.parent; entries['train'].append({'source':'inspire_f1','id':'arctic/'+seq.relative_to(x.arctic_root).as_posix(),'path':str(seq.resolve()),'dataset':'arctic'})
 if x.oak_root and x.oak_root.exists():
  for p in sorted(x.oak_root.glob('**/geometry/manifest.json')):
   seq=p.parent.parent; entries['train'].append({'source':'inspire_f1','id':'oakink2/'+seq.relative_to(x.oak_root).as_posix(),'path':str(seq.resolve()),'dataset':'oakink2'})
 if missing: raise SystemExit(f'missing GRAB geometry {len(missing)}; first={missing[:3]}')
 payload={'schema_name':'ref2dex_object_interaction_cm_index_v1_2','schema_version':'1.2.0','created_at':'2026-09-14','knn_k':8,'object_pool_points':4096,'model_object_points':1024,'hand_points_per_stream':3076,'max_union_hand_points':3076,'source_probability':{'inspire_f1':1.0},'split_policy':{'grab':'existing InteractionTransfer grab_seed42 split; val/test unchanged','arctic':'all train','oakink2':'single-object geometry only, train'},'sequences':entries,'counts':{s:len(v) for s,v in entries.items()}}
 x.output.parent.mkdir(parents=True,exist_ok=True); x.output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n'); print(json.dumps(payload['counts']))
if __name__=='__main__': main()
