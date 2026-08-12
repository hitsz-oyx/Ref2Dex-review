import torch

from src.task.InteractionDynamics.dataset_grasp_v18 import _rigid_world_to_reference
from src.task.InteractionDynamics.eval_grasp_v18 import trajectory_statistics
from src.task.InteractionDynamics.grasp_interaction_diffusion import GraspInteractionDiffusion


def test_v18_rigid_alignment_recovers_reference():
    reference=torch.randn(1,64,3)
    angle=torch.tensor(.4); c,s=torch.cos(angle),torch.sin(angle)
    rotation=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    moved=reference@rotation.T+torch.tensor([1.,2.,3.])
    points=torch.cat([reference,moved])
    recovered_rotation,translation=_rigid_world_to_reference(points)
    recovered=points@recovered_rotation+translation
    assert torch.allclose(recovered[0],recovered[1],atol=1e-5)


def test_v18_stable_metric_requires_contact_and_low_motion():
    future=torch.zeros(2,128,56)
    values=future.reshape(2,128,8,7)
    values[...,6]=3.
    values[0,:5,-3:,6]=1.
    values[1,:5,-3:,6]=1.; values[1,:,:,:3]=1.
    stats=trajectory_statistics(future)
    assert stats["success"].tolist()==[True,False]


def test_v18_diffusion_has_no_goal_input():
    model=GraspInteractionDiffusion(horizon=8,dim=32,heads=4,layers=1)
    output=model(torch.randn(2,8,56),torch.randn(2,8,4),torch.randn(2,8,3),
                 torch.randn(2,8,5,6),torch.tensor([1,2]))
    assert output.shape==(2,8,56)
