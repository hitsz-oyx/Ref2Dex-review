"""为既有 V20 field shard 补充 V20.1 MANO projector metadata。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from process.GRAB.raw import DEFAULT_GRAB_ROOT, GRABSeqData
from src.task.InteractionDynamics.dataset_grasp_v18 import _rigid_world_to_reference, load_events
from src.task.InteractionDynamics.research.v18_4.build_mano_h_cache import delta_h, hand_state


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--events",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--event-index",type=int,default=0);p.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    p.add_argument("--device",default="cuda");args=p.parse_args();device=torch.device(args.device)
    event=load_events(args.events)[args.event_index];payload=torch.load(args.input,map_location="cpu")
    with np.load(event.path.parent/"shared.npz",allow_pickle=False) as shared:
        source=str(shared["source_raw_file"].item());raw_frame_id=np.asarray(shared["raw_frame_id"],np.int64)
        object_world=torch.from_numpy(np.asarray(shared["obj_points_world"],np.float32)).to(device)
    with np.load(event.path,allow_pickle=False) as hand:side=str(hand["side"].item())
    params=GRABSeqData(str(args.grab_root/source)).get_hand_params(side)
    rotation,translation=_rigid_world_to_reference(object_world);sample_frames=[frame for frame in
        range(max(1,event.grasp_frame-8),event.grasp_frame+5) if frame+8<len(object_world)]
    rows={key:[] for key in ("current_h","future_delta_h","betas","object_rotation",
        "object_translation","global_orient","hand_pose","transl","raw_frame_ids")}
    for frame in sample_frames:
        ids=np.arange(frame,frame+9);raw_ids=raw_frame_id[ids]
        def value(name,width):return torch.from_numpy(np.asarray(params[name][raw_ids],np.float32).reshape(9,width)).to(device)
        orient,pose,transl=value("global_orient",3),value("hand_pose",24),value("transl",3)
        beta=np.asarray(params["betas"],np.float32);beta=beta[raw_ids] if beta.ndim>1 else np.broadcast_to(beta,(9,beta.shape[-1])).copy()
        betas=torch.from_numpy(beta).to(device);q,t=rotation[ids],translation[ids];h,h_rotation=hand_state(orient,pose,transl,q,t)
        values=(h[0],delta_h(h,h_rotation),betas,q,t,orient,pose,transl,torch.from_numpy(raw_ids))
        for key,item in zip(rows,values):rows[key].append(item.cpu())
    if len(sample_frames)!=len(payload["current_y"]):raise ValueError("Field shard and event windows mismatch")
    payload.update({key:torch.stack(value) for key,value in rows.items()});payload.update({
        "side":side,"source_raw_file":source,"source_hand_cache":str(event.path),
        "mano_key":str(event.path.parent.parent.name)+":"+side})
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,args.output);print(args.output,flush=True)


if __name__=="__main__":main()
