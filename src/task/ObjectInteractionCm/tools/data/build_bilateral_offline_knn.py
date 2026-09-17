#!/usr/bin/env python3
"""Build offline KNN arrays for the bilateral geometry cache on one GPU."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

K=32; RADIUS=0.02

def process_sequence(seq: Path, device: torch.device, obj_chunk: int) -> tuple[int,int]:
    g=seq/'geometry'; marker=g/'obj_knn_indices.npy'
    complete=False
    if marker.is_file() and (g/'obj_candidate_mask_2cm.npy').is_file() and (g/'hand_supervision_mask_2cm.npy').is_file() and (g/'hand_min_object_distance_m.npy').is_file() and (g/'knn_hand_points_world.npy').is_file() and (g/'knn_hand_normals_world.npy').is_file():
        try: complete=bool(json.loads((g/'manifest.json').read_text()).get('offline_knn_min_global',False))
        except Exception: complete=False
    if complete: return 0, int(len(np.load(g/'source_frame_id.npy',mmap_mode='r')))
    obj=np.load(g/'obj_points_pool_world.npy',mmap_mode='r'); hand=np.load(g/'hand_points_world.npy',mmap_mode='r'); pose=np.load(g/'obj_pose_world.npy',mmap_mode='r')
    if obj.shape[1:]!=(4096,3) or hand.shape[1:]!=(3076,3): raise ValueError(f'{seq}: invalid shape obj={obj.shape} hand={hand.shape}')
    T=len(obj); idx=np.lib.format.open_memmap(str(g/'obj_knn_indices.npy.partial'),mode='w+',dtype=np.uint16,shape=(T,4096,K)); active=np.lib.format.open_memmap(str(g/'obj_candidate_mask_2cm.npy.partial'),mode='w+',dtype=np.bool_,shape=(T,4096)); sup=np.lib.format.open_memmap(str(g/'hand_supervision_mask_2cm.npy.partial'),mode='w+',dtype=np.bool_,shape=(T,3076)); mind=np.lib.format.open_memmap(str(g/'hand_min_object_distance_m.npy.partial'),mode='w+',dtype=np.float32,shape=(T,))
    try:
      with torch.inference_mode():
       for t in range(T):
        R=torch.as_tensor(np.asarray(pose[t,:3,:3],dtype=np.float32),device=device); tr=torch.as_tensor(np.asarray(pose[t,:3,3],dtype=np.float32),device=device)
        o=torch.as_tensor(np.asarray(obj[t],dtype=np.float32),device=device); h=torch.as_tensor(np.asarray(hand[t],dtype=np.float32),device=device); o=(o-tr)@R; h=(h-tr)@R
        ids=[]; act=[]
        for start in range(0,4096,obj_chunk):
         d=torch.cdist(o[start:start+obj_chunk],h); vals,ii=torch.topk(d,k=K,dim=1,largest=False,sorted=True); ids.append(ii.cpu().numpy().astype(np.uint16)); act.append((vals[:,0]<=RADIUS).cpu().numpy())
        idx[t]=np.concatenate(ids); active[t]=np.concatenate(act)
        minv_global=None
        for start in range(0,3076,obj_chunk):
         d=torch.cdist(h[start:start+obj_chunk],o); v=d.min(dim=1).values; sup[t,start:start+len(v)]=(v<=RADIUS).cpu().numpy();
         chunk_min=v.min()
         minv_global=chunk_min if minv_global is None else torch.minimum(minv_global,chunk_min)
        mind[t]=float(minv_global.cpu())
        if (t+1)%100==0: print(f'{seq.name} frames={t+1}/{T}',flush=True)
      for x in (idx,active,sup,mind): x.flush()
      for stem in ('obj_knn_indices','obj_candidate_mask_2cm','hand_supervision_mask_2cm','hand_min_object_distance_m'):
       partial=g/(stem+'.npy.partial'); final=g/(stem+'.npy'); os.replace(partial,final)
      for name in ('knn_hand_points_world.npy','knn_hand_normals_world.npy'):
       target=g/name
       if not target.exists(): os.symlink(('hand_points_world.npy' if 'points' in name else 'hand_normals_world.npy'),target)
      meta=g/'manifest.json'; d=json.loads(meta.read_text()); d.update({'offline_knn':True,'offline_knn_min_global':True,'knn_k':K,'knn_hand_points':3076,'knn_distance_radius_m':RADIUS,'knn_source':'merged_bilateral_hand_stream'}); meta.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
      return 1,T
    except Exception:
      for p in g.glob('*.npy.partial'):
       try: p.unlink()
       except OSError: pass
      raise

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--roots',nargs='+',type=Path,required=True); ap.add_argument('--device',default='cuda:1'); ap.add_argument('--obj-chunk',type=int,default=512); ap.add_argument('--limit',type=int,default=0); a=ap.parse_args(); device=torch.device(a.device); assert device.type=='cuda' and torch.cuda.is_available()
 seqs=[]
 for root in a.roots: seqs.extend(sorted(p.parent.parent for p in root.resolve().glob('**/geometry/manifest.json')))
 if a.limit: seqs=seqs[:a.limit]
 done=frames=0; started=time.time()
 for i,seq in enumerate(seqs,1):
  try:
   d,f=process_sequence(seq,device,a.obj_chunk); done+=d; frames+=f; print(json.dumps({'sequence':str(seq),'index':i,'total':len(seqs),'written':d,'frames':f},ensure_ascii=False),flush=True)
  except Exception as e: print(f'ERROR {seq}: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
 print(json.dumps({'sequences':len(seqs),'written':done,'frames':frames,'elapsed_s':time.time()-started}))
if __name__=='__main__': main()
