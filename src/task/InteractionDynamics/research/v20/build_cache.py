"""从 V18 event 构建独立 V20 causal field cache。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from process.GRAB.raw import DEFAULT_GRAB_ROOT, GRABSeqData, load_object_canonical_mesh
from src.task.InteractionDynamics.dataset_grasp_v18 import (
    _rigid_world_to_reference, _transform, load_events, sequence_interaction)
from src.task.InteractionDynamics.field_state_v20 import build_causal_field, residual_target
from src.task.InteractionDynamics.penetration import penetration_from_mesh
from src.task.InteractionDynamics.uni3d import gather_points


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--events",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--event-indices",type=int,nargs="+")
    parser.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data")
    parser.add_argument("--device",default="cuda");args=parser.parse_args();events=load_events(args.events)
    selected=args.event_indices or list(range(len(events)))
    for event_index in selected:
        event=events[event_index]; data=sequence_interaction(event.path,device=args.device)
        with np.load(event.path.parent/"shared.npz",allow_pickle=False) as shared, np.load(event.path) as hand:
            hand_world=torch.from_numpy(np.asarray(hand["hand_points_world"],np.float32)).to(args.device)
            object_world=torch.from_numpy(np.asarray(shared["obj_points_world"],np.float32)).to(args.device)
            raw_ids=np.asarray(shared["raw_frame_id"],np.int64)
            source_raw_file=str(shared["source_raw_file"].item())
        rotation,translation=_rigid_world_to_reference(object_world)
        hand_reference=_transform(hand_world,rotation,translation)
        sequence=GRABSeqData(str(args.grab_root/source_raw_file))
        mesh=load_object_canonical_mesh(sequence.obj_name,str(args.grab_root),"m")
        params=sequence.get_object_params(); raw_rotation=axis_angle_to_matrix(torch.from_numpy(
            np.asarray(params["global_orient"][raw_ids],np.float32))).to(args.device)
        raw_translation=torch.from_numpy(np.asarray(params["transl"][raw_ids],np.float32)).to(args.device)
        sample_frames=[frame for frame in range(max(1,event.grasp_frame-8),event.grasp_frame+5)
                       if frame+8<len(hand_reference)]
        needed=range(min(sample_frames)-1,max(sample_frames)+9)
        penetration=torch.zeros(hand_reference.shape[:2],device=args.device);valid={}
        for frame in needed:
            canonical=torch.from_numpy(np.asarray(mesh.vertices,np.float32)).to(args.device)
            mesh_world=canonical @ raw_rotation[frame].T+raw_translation[frame]
            mesh_reference=_transform(mesh_world,rotation[frame],translation[frame])
            point=penetration_from_mesh(hand_reference[frame].cpu().numpy(),mesh_reference.cpu().numpy(),
                np.asarray(mesh.faces,np.int32))
            if point.valid: penetration[frame]=torch.from_numpy(point.point_penetration_m).to(args.device)
            valid[frame]=point.valid
        field=100*build_causal_field(hand_reference,data["anchors"],penetration)
        rows=[]
        for frame in sample_frames:
            current=field[frame-1];future=field[frame:frame+8].transpose(0,1)
            points=data["object_object"][frame];knn=data["knn"];anchors=data["anchors"]
            patches=torch.cat([100*(gather_points(points[None],knn[None])-anchors[None,:,None]),
                gather_points(data["normals_object"][frame][None],knn[None])],-1)[0]
            rows.append({"current_y":current.cpu(),"future_y":future.cpu(),
                "delta_y":residual_target(current,future).cpu(),"anchors_cm":(100*anchors).cpu(),
                "object_patches":patches.cpu(),"p_valid":torch.tensor(all(valid[x] for x in range(frame,frame+9)))})
        if rows:
            folder=args.output/"train";folder.mkdir(parents=True,exist_ok=True)
            torch.save({k:torch.stack([row[k] for row in rows]) for k in rows[0]},folder/f"event_{event_index:05d}.pt")
            print(event_index,len(rows),flush=True)


if __name__=="__main__":main()
