import torch

from src.task.InteractionDynamics.dataset_grasp_v18 import _rigid_world_to_reference
from src.task.InteractionDynamics.eval_grasp_v18 import subset_masks, trajectory_statistics
from src.task.InteractionDynamics.grasp_interaction_diffusion import GraspInteractionDiffusion
from src.task.InteractionDynamics.research.v18_2.audit_y_consistency import unpack
from src.task.InteractionDynamics.research.v18_2.eval_y_realizability_full import target_loss


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


def test_v18_1_formation_transition_maintenance_masks():
    state=torch.zeros(3,128,4); state[...,3]=3.; state[1:,:5,3]=1.
    batch={"state":state,"frame":torch.tensor([2,2,5]),"grasp_frame":torch.tensor([5,5,5])}
    masks=subset_masks(batch)
    assert masks["formation"].tolist()==[True,False,False]
    assert masks["transition"].tolist()==[False,True,False]
    assert masks["maintenance"].tolist()==[False,False,True]


def test_v18_2_unpack_temporal_consistency():
    state=torch.zeros(1,2,4)
    values=torch.zeros(1,2,8,7)
    values[:,:,0,3:6]=1.; values[:,:,0,:3]=1.
    _,_,error=unpack(state,values.flatten(-2))
    assert torch.allclose(error[:,:,0],torch.zeros(1,2))


def test_v18_2_target_loss_modes_select_components():
    target=torch.zeros(1,2,56); prediction=target.clone().reshape(1,2,8,7)
    prediction[...,0:3]=3.; prediction[...,3:6]=2.; prediction[...,6]=1.
    prediction=prediction.flatten(-2)
    assert torch.allclose(target_loss(prediction,target,"r"),torch.tensor([4.]))
    assert target_loss(prediction,target,"rd").item()==3.25
    assert torch.allclose(target_loss(prediction,target,"rdu"),torch.tensor([40/7]))
