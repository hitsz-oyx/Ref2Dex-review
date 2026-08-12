"""把 V17 full-frame analytic tensors 物化为训练 shard。"""
import argparse
from pathlib import Path
import torch, yaml
from src.task.InteractionDynamics.dataset_v17 import V17Dataset, dominant_sequences, split_sequences

p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="src/task/InteractionDynamics/configs/v17_full.yaml"); p.add_argument("--output",type=Path,required=True); p.add_argument("--shard-size",type=int,default=256); a=p.parse_args()
cfg=yaml.safe_load(Path(a.config).read_text()); paths=dominant_sequences(cfg["data"]["root"],cfg["data"]["dominant_hand_manifest"]); split=split_sequences(paths,cfg["data"]["split_seed"])
for name,seqs in [("train",split.train),("val",split.val),("test",split.test)]:
 d=V17Dataset(cfg["data"]["root"],seqs); folder=a.output/name; folder.mkdir(parents=True,exist_ok=True)
 for start in range(0,len(d),a.shard_size):
  rows=[d[i] for i in range(start,min(start+a.shard_size,len(d)))]; keys=["state","future","residual","anchors_cm","object_patches","goal","distance_cm","frame"]
  torch.save({k:torch.stack([torch.as_tensor(r[k]) for r in rows]) for k in keys},folder/f"{start:06d}.pt")
  print(f"{name}: {min(start+a.shard_size,len(d))}/{len(d)}",flush=True)
