"""V17 full-frame sequence split 与数据分布审计。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import yaml

from src.task.InteractionDynamics.dataset_v17 import V17Dataset, dominant_sequences, split_sequences


def summarize(dataset, max_samples=0):
    indices = np.arange(len(dataset))
    if max_samples and len(indices) > max_samples:
        indices = np.rint(np.linspace(0, len(indices) - 1, max_samples)).astype(int)
    active, dynamic, distance, goal_t, goal_r, residual = [], [], [], [], [], []
    for number, index in enumerate(indices, 1):
        row = dataset[int(index)]
        magnitude = float(row["residual"].square().mean().sqrt())
        active.append(float(row["goal"][0] > .5)); dynamic.append(float(magnitude >= .2))
        distance.append(float(row["distance_cm"])); residual.append(magnitude)
        motion = row["goal"][1:].reshape(-1, 6)
        goal_t.append(float(motion[:, :3].norm(dim=-1).max()))
        goal_r.append(float(motion[:, 3:].norm(dim=-1).max()))
        if number % 1000 == 0: print(json.dumps({"audited": number}), flush=True)
    def stats(values):
        x=np.asarray(values); return {"min":float(x.min()),"median":float(np.median(x)),
            "mean":float(x.mean()),"p90":float(np.quantile(x,.9)),"max":float(x.max())}
    d=np.asarray(distance)
    bins={name:int(((d>=lo)&(d<hi)).sum()) for lo,hi,name in
          [(0,10,"0-10"),(10,30,"10-30"),(30,60,"30-60"),(60,100,"60-100"),(100,np.inf,">100")]}
    return {"chunks":len(dataset),"audited_chunks":len(indices),"active_ratio":np.mean(active),
            "dynamic_ratio":np.mean(dynamic),"distance_cm":stats(distance),"distance_bins":bins,
            "goal_translation_cm":stats(goal_t),"goal_rotation_rad":stats(goal_r),
            "residual_rms_cm":stats(residual)}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",default="src/task/InteractionDynamics/configs/v17_full.yaml")
    p.add_argument("--max-audit-per-split",type=int,default=0); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); data=cfg["data"]
    paths=dominant_sequences(data["root"],data["dominant_hand_manifest"])
    split=split_sequences(paths,data["split_seed"],(data["train_ratio"],data["val_ratio"],data["test_ratio"]))
    a.output.mkdir(parents=True,exist_ok=True)
    for name,values in [("train",split.train),("val",split.val),("test",split.test)]:
        (a.output/f"{name}_sequences.txt").write_text("\n".join(map(str,values))+"\n")
    datasets={name:V17Dataset(data["root"],values) for name,values in
              [("train",split.train),("val",split.val),("test",split.test)]}
    result={"hand_file_count":len(paths),"sequence_count":len({p.parent for p in paths}),
            "split_sequences":{k:len({p.parent for p in v}) for k,v in
            [("train",split.train),("val",split.val),("test",split.test)]},
            "splits":{k:summarize(v,a.max_audit_per_split) for k,v in datasets.items()}}
    (a.output/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)); print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__": main()
