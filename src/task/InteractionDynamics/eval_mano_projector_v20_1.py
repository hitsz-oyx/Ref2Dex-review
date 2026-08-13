"""冻结 V20 field predictor 的 controlled MANO feasibility 评估。"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix

from process.GRAB.raw import (DEFAULT_GRAB_ROOT,DEFAULT_MANO_MODEL_DIR,GRABSeqData,
                              load_object_canonical_mesh)
from src.task.InteractionDynamics.dataset_field_v20 import CachedFieldV20Dataset
from src.task.InteractionDynamics.field_dynamics_v20 import FieldDynamicsTransition
from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field
from src.task.InteractionDynamics.mano_projector_v20_1 import project_mano
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers
from src.task.InteractionDynamics.viewer_v2.stable import CausalStableDetector
from src.task.InteractionDynamics.penetration import penetration_from_mesh


def move_single(item,device):
    return {k:([v] if isinstance(v,str) else v[None].to(device)) for k,v in item.items()}


def metrics(a,b):
    error=a-b;result={}
    for name,value in (("r",error[...,:3]),("d",error[...,3]),("v",error[...,4:7])):
        result[name+"_mae_cm"]=float(value.abs().mean());result[name+"_rmse_cm"]=float(value.square().mean().sqrt())
    return result


def stable(field):
    detector=CausalStableDetector(consecutive_frames=3);state=None
    for frame in range(field.shape[2]):state=detector.observe(field[0,:,frame,3].cpu().numpy(),field[0,:,frame,4:7].cpu().numpy())
    return bool(state.latched)


def terminal_penetration(surface,batch,grab_root):
    sequence=GRABSeqData(str(grab_root/batch["source_raw_file"][0]));mesh=load_object_canonical_mesh(
        sequence.obj_name,str(grab_root),"m");raw_id=int(batch["raw_frame_ids"][0,-1])
    params=sequence.get_object_params();rotation=axis_angle_to_matrix(torch.from_numpy(
        np.asarray(params["global_orient"][raw_id],np.float32))).numpy()
    translation=np.asarray(params["transl"][raw_id],np.float32)
    world=np.asarray(mesh.vertices,np.float32)@rotation.T+translation
    q=batch["object_rotation"][0,-1].cpu().numpy();t=batch["object_translation"][0,-1,0].cpu().numpy()
    reference=world@q+t;result=penetration_from_mesh(surface[0,-1].cpu().numpy(),reference,
        np.asarray(mesh.faces,np.int32))
    return {"valid":result.valid,"max_mm":result.max_penetration_mm,
            "mean_mm":result.mean_penetration_mm,"ratio":result.penetrating_point_ratio}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--field-checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=4);p.add_argument("--steps",type=int,default=100);p.add_argument("--lr",type=float,default=1e-2)
    p.add_argument("--target",choices=("prediction","gt"),default="prediction")
    p.add_argument("--smooth-weight",type=float,default=.01);p.add_argument("--prior-weight",type=float,default=1e-4)
    p.add_argument("--grab-root",type=Path,default=Path(DEFAULT_GRAB_ROOT)/"data");p.add_argument("--mano-path",type=Path,default=Path(DEFAULT_MANO_MODEL_DIR));args=p.parse_args()
    device=torch.device("cuda");payload=torch.load(args.field_checkpoint,map_location="cpu");saved=payload["args"]
    model=FieldDynamicsTransition(dim=saved["dim"],layers=saved["layers"]).to(device);model.load_state_dict(payload["model"]);model.eval()
    mean,std=payload["mean"].to(device),payload["std"].to(device);dataset=CachedFieldV20Dataset(args.cache,"train")
    layers=ManoLayers(args.grab_root,args.mano_path,device);rows=[]
    for index in range(min(args.samples,len(dataset))):
        batch=move_single(dataset[index],device)
        with torch.no_grad():delta=model(batch["current_y"],batch["anchors_cm"],batch["object_patches"])*std+mean
        predicted=batch["current_y"][:,:,None]+delta;gt=batch["future_y"]
        parity,gt_surface=decode_mano_field(batch["future_delta_h"],batch,layers)
        parity_error=float((parity-gt[...,:7]).abs().max())
        target=predicted if args.target=="prediction" else gt
        _,projected,surface,history=project_mano(target,batch,layers,std,args.steps,args.lr,
            args.smooth_weight,args.prior_weight)
        rows.append({"index":index,"parity_max_abs_cm":parity_error,"free_vs_gt":metrics(predicted[...,:7],gt[...,:7]),
            "projection_gap":metrics(projected,predicted[...,:7]),"projected_vs_gt":metrics(projected,gt[...,:7]),
            "free_stable":stable(predicted),"projected_stable":stable(projected),"gt_stable":stable(gt),
            "projected_penetration":terminal_penetration(surface,batch,args.grab_root),
            "gt_penetration":terminal_penetration(gt_surface,batch,args.grab_root),
            "history":history,"surface_shape":list(surface.shape)})
        print(json.dumps(rows[-1]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
