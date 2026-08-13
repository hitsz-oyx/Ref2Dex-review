"""V20.1 offline MANO feasibility projector。"""
from __future__ import annotations

import torch

from src.task.InteractionDynamics.mano_field_decoder_v20_1 import decode_mano_field


def project_mano(predicted_y: torch.Tensor,batch:dict,layer_for,std:torch.Tensor,
                 steps:int=100,lr:float=1e-2,smooth_weight:float=.01,
                 prior_weight:float=1e-4,record_steps=(0,10,25,50,100,250,500)):
    delta=torch.zeros((1,8,30),device=predicted_y.device,requires_grad=True)
    optimizer=torch.optim.Adam([delta],lr=lr);history=[]
    def measure(step:int,backward:bool):
        mano_y,surface=decode_mano_field(delta,batch,layer_for)
        error=(mano_y-predicted_y[...,:7])/std[...,:7]
        field=error.square().mean();trajectory=torch.cat([torch.zeros_like(delta[:,:1]),delta],1)
        difference=torch.diff(trajectory,dim=1)
        smooth=(difference[...,:3]/1.).square().mean()+(difference[...,3:6]/.2).square().mean()+(difference[...,6:]/.2).square().mean()
        prior=(delta[...,6:]/.5).square().mean();loss=field+smooth_weight*smooth+prior_weight*prior
        if backward: optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step in record_steps:
            raw=mano_y-predicted_y[...,:7]
            history.append({"step":step,"field_loss":float(field.detach()),"r_projection_rmse":float(raw[...,:3].square().mean().sqrt()),
                "d_projection_rmse":float(raw[...,3].square().mean().sqrt()),"v_projection_rmse":float(raw[...,4:7].square().mean().sqrt()),
                "smooth":float(smooth.detach()),"prior":float(prior.detach())})
        return mano_y,surface
    measure(0,False)
    for step in range(1,steps+1):measure(step,True)
    mano_y,surface=measure(steps,False)
    return delta.detach(),mano_y.detach(),surface.detach(),history
