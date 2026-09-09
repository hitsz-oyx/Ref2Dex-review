# MANO/Inspire Cm source classifier

该实验冻结当前 ObjectInteractionCm checkpoint，从 MANO 和 Inspire-F1 的物体条件 hand flow 生成
`cm_tokens`，训练一个小型二分类器区分 `mano=0` 与 `inspire_rl=1`。主输入只有
`cm_tokens` 的无 slot-ID pooling 特征，不把 hand flow、object points 或 anchor position 作为主特征。

数据使用 ObjectInteractionCm 现有 train/val sequence split，并只保留 train/val 两边都出现的 object
类别；每个 split 内按 object 类别平衡两种 source 的 sequence 数量，每条 sequence 固定抽取少量 transition。
主结果同时报告全部样本和 `sample_valid=true` 子集，另外报告 anchor/object control。

运行：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.CmDecoderv2.research.cm_hand_source_classifier.run \
  --activity-id <activity_id>
```

产物写入 `output/<run_id>/`，包括 `features.npz`、`metrics.json`、分类器 checkpoint 和
`run_manifest.json`。
