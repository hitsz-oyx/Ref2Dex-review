from pathlib import Path
import torch
from src.task.InteractionDynamics.dataset_v17 import split_sequences
from src.task.InteractionDynamics.eval_v17 import Metrics, masks_for
from src.task.InteractionDynamics.residual_interaction_regression_v17_1 import temporal_goal
from src.task.InteractionDynamics.train_v17_1 import bucket_masks, tau_statistics

def test_v17_split_is_sequence_disjoint_and_deterministic():
    paths=[]
    for i in range(10):
        paths.extend([Path(f"s{i}/demo/right.npz"),Path(f"s{i}/demo/left.npz")])
    first=split_sequences(paths,42); second=split_sequences(paths,42)
    assert first==second
    groups=[{p.parent for p in values} for values in [first.train,first.val,first.test]]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert [len(x) for x in groups]==[8,1,1]

def test_v17_stats_goal_keeps_channels(tmp_path):
    from src.task.InteractionDynamics.train_v17 import stats_for
    rows=[{"state":torch.randn(128,4),"goal":torch.randn(25),"residual":torch.randn(128,28)} for _ in range(3)]
    stats=stats_for(rows,tmp_path/"stats.pt")
    assert stats["state_mean"].shape==(1,1,4)
    assert stats["goal_mean"].shape==(1,25)
    assert stats["residual_mean"].shape==(1,1,28)

def test_v17_eval_masks_and_exact_metrics():
    batch={"residual":torch.zeros(2,128,28), "goal":torch.zeros(2,25),
           "distance_cm":torch.tensor([20.,80.])}
    batch["residual"][1]=1.; batch["goal"][1,0]=1.
    masks=masks_for(batch)
    assert masks["dynamic"].tolist()==[False,True]
    assert masks["active"].tolist()==[False,True]
    assert masks["far_60cm"].tolist()==[False,True]
    target=torch.zeros(2,128,28); prediction=torch.ones_like(target)
    metrics=Metrics(); metrics.update(prediction,target,masks)
    result=metrics.result()
    assert result["overall"]["u_rmse_cm"]==1.
    assert result["dynamic"]["r_rmse_cm"]==1.
    assert result["far_60cm"]["d_rmse_cm"]==1.

def test_v17_1_temporal_goal_and_buckets():
    tau=torch.tensor([-1,0,9,10,29,30,99,100])
    mean,std=tau_statistics(tau)
    goal=torch.zeros(len(tau),25)
    baseline=temporal_goal(goal,tau,mean,std,False)
    temporal=temporal_goal(goal,tau,mean,std,True)
    assert baseline.shape==temporal.shape==(8,26)
    assert torch.count_nonzero(baseline)==0
    assert temporal[0,-1]==0 and temporal[-1,-1]!=0
    masks=bucket_masks(tau,torch.ones(8,dtype=torch.bool))
    assert [int(masks[name].sum()) for name in
            ["missing","0_10","10_30","30_100","100_plus"]]==[1,2,2,2,1]
