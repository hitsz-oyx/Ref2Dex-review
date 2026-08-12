from pathlib import Path
import torch
from src.task.InteractionDynamics.dataset_v17 import split_sequences
from src.task.InteractionDynamics.eval_v17 import Metrics, masks_for

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
