"""Self-contained Inspire/DExplore tensor contract (no Ref2Dex import side effects)."""
from __future__ import annotations
import torch

QUERY_LINKS = (
    "hand_base_link", "thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_tip", "middle_proximal", "middle_intermediate", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_tip", "pinky_proximal", "pinky_intermediate", "pinky_tip",
)
NATIVE_DOF_NAMES = ("joint1","joint2","joint3","joint4","joint5","joint6","index_proximal_joint","index_intermediate_joint","middle_proximal_joint","middle_intermediate_joint","pinky_proximal_joint","pinky_intermediate_joint","ring_proximal_joint","ring_intermediate_joint","thumb_proximal_yaw_joint","thumb_proximal_pitch_joint","thumb_intermediate_joint","thumb_distal_joint")
INDEPENDENT_NATIVE = (6, 8, 10, 12, 14, 15)
MIMIC_NATIVE = (7, 9, 11, 13, 16, 17)
MIMIC_SOURCE = (0, 1, 2, 3, 5, 5)
MIMIC_SCALE = (1.05, 1.05, 1.05, 1.05, 0.6, 0.8)

def native_sim_indices(sim_names):
    if len(sim_names) != 18 or set(sim_names) != set(NATIVE_DOF_NAMES):
        raise ValueError(f"Unexpected Inspire DOF names: {sim_names}")
    return [sim_names.index(name) for name in NATIVE_DOF_NAMES]

def sim_to_native(x, indices): return x.index_select(-1, indices)
def native_to_sim(x, indices):
    y = torch.empty_like(x); y[..., indices] = x; return y

def _quat_xyzw_to_matrix(q):
    q = q / (q.square().sum(-1, keepdim=True).sqrt() + 1e-8)
    x,y,z,w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)

def _matrix_to_quat_xyzw(m):
    # Stable enough for pose observations; returns xyzw.
    t = m[...,0,0]+m[...,1,1]+m[...,2,2]
    w = torch.sqrt(torch.clamp(1+t,min=1e-8))/2
    x = (m[...,2,1]-m[...,1,2])/(4*w+1e-8); y=(m[...,0,2]-m[...,2,0])/(4*w+1e-8); z=(m[...,1,0]-m[...,0,1])/(4*w+1e-8)
    return torch.stack((x,y,z,w),-1)

def pose_matrix(p):
    r = _quat_xyzw_to_matrix(p[..., [3,4,5,6]])
    out=torch.eye(4,device=p.device,dtype=p.dtype).expand(*p.shape[:-1],4,4).clone(); out[...,:3,:3]=r; out[...,:3,3]=p[...,:3]; return out

def matrix_pose(m): return torch.cat((m[...,:3,3], _matrix_to_quat_xyzw(m[...,:3,:3]),),-1)
def inverse_pose(m):
    out=torch.zeros_like(m); r=m[...,:3,:3].transpose(-1,-2); out[...,:3,:3]=r; out[...,:3,3]=-(r@m[...,:3,3,None]).squeeze(-1); out[...,3,3]=1; return out

def coupled_finger_bounds(lower, upper):
    lo=lower[list(INDEPENDENT_NATIVE)].clone(); hi=upper[list(INDEPENDENT_NATIVE)].clone()
    for m,s,k in zip(MIMIC_NATIVE,MIMIC_SOURCE,MIMIC_SCALE): lo[s]=torch.maximum(lo[s],lower[m]/k); hi[s]=torch.minimum(hi[s],upper[m]/k)
    if (lo>hi).any(): raise ValueError('Empty coupled control range')
    return lo,hi
