"""可微 ΔH→MANO surface→V20 causal [r,d,v] decoder。"""
from __future__ import annotations

from collections.abc import Callable

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from src.task.InteractionDynamics.inverse_mano import face_centers


def build_causal_rdv_batched(surface: torch.Tensor, anchors: torch.Tensor,
                             tau_m: float = .015, anchor_chunk: int = 16) -> torch.Tensor:
    """surface `[B,H+1,P,3]` → `[B,N,H,7]`，速度使用当前帧权重。"""
    outputs=[]; displacement=torch.diff(surface,dim=1)
    for start in range(0,anchors.shape[1],anchor_chunk):
        anchor=anchors[:,None,start:start+anchor_chunk,None]
        relative=surface[:,:,None]-anchor
        weights=torch.softmax(-relative.square().sum(-1)/tau_m**2,dim=-1)
        r=(weights[...,None]*relative).sum(-2)[:,1:]
        d=(weights*relative.norm(dim=-1)).sum(-1)[:,1:,...,None]
        v=(weights[:,1:,...,None]*displacement[:,:,None]).sum(-2)
        outputs.append(torch.cat([r,d,v],-1))
    return torch.cat(outputs,2).transpose(1,2)*100


def decode_mano_field(delta_h: torch.Tensor, batch: dict, layer_for: Callable
                      ) -> tuple[torch.Tensor,torch.Tensor]:
    device=delta_h.device; surfaces=[]
    for index in range(len(delta_h)):
        layer=layer_for(batch["mano_key"][index],batch["side"][index],batch["source_raw_file"][index])
        q=batch["object_rotation"][index:index+1];t=batch["object_translation"][index:index+1]
        current_rotation=q[:,:1].transpose(-1,-2)@axis_angle_to_matrix(batch["global_orient"][index:index+1,:1])
        relative=axis_angle_to_matrix(delta_h[index:index+1,:,3:6])@current_rotation
        world_rotation=q[:,1:]@relative
        object_position=(batch["current_h"][index:index+1,None,:3]+delta_h[index:index+1,:,:3])/100
        world_translation=(object_position[...,None,:]-t[:,1:])@q[:,1:].transpose(-1,-2)
        future_pose=batch["hand_pose"][index:index+1,:1]+delta_h[index:index+1,:,6:]
        faces=torch.as_tensor(layer.faces,dtype=torch.long,device=device)
        current=face_centers(layer(global_orient=batch["global_orient"][index:index+1,0],
            hand_pose=batch["hand_pose"][index:index+1,0],betas=batch["betas"][index:index+1,0],
            transl=batch["transl"][index:index+1,0]).vertices,faces)
        future=face_centers(layer(global_orient=matrix_to_axis_angle(world_rotation).flatten(0,1),
            hand_pose=future_pose.flatten(0,1),betas=batch["betas"][index:index+1,1:].flatten(0,1),
            transl=world_translation[...,0,:].flatten(0,1)).vertices,faces).reshape(1,8,-1,3)
        world=torch.cat([current[:,None],future],1)
        surfaces.append(torch.einsum("btpi,btij->btpj",world,q)+t[:,:,None,0])
    surface=torch.cat(surfaces,0)
    return build_causal_rdv_batched(surface,batch["anchors_cm"]/100),surface
