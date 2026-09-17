# GRAB MANO / Inspire 配对可视化

本诊断为同一个 GRAB sequence 生成只读 viewer index，使
`src.task.ObjectInteractionCm.visualize_grab` 能在 MANO bilateral raw NPZ 与 Inspire geometric
之间切换。准备阶段要求 frame ID、物体点和物体 pose 完全一致，不修改任何输入数据。

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python \
  src/task/ObjectInteractionCm/research/paired_grab_visualization/prepare.py \
  --sequence s1/airplane_fly_1 \
  --output src/task/ObjectInteractionCm/research/paired_grab_visualization/output/<run_id>
```

生成的 `index.json` 供 `visualize_grab.py` 使用；MANO 与 Inspire 记录具有相同 sequence ID，GUI 标签
通过 variant 区分。命令行检查使用 `--sequence-index 0`（MANO）和 `--sequence-index 1`（Inspire）。
