# GRAB 跨帧点流查看器

本诊断复用 `src.task.ObjectInteractionCm.visualize_grab`，只读加载当前 V1.4 GRAB 双手 Inspire
geometry。绿色/紫色分别叠加未来物体点和未来手点；“未来 Δ（cache 帧）”可在 0–10 间调整。

GRAB cache 为 30 Hz，因此 `Δ=10` 在未触及序列末尾时对应约 0.333 秒，并对应原始 120 Hz
时间轴上的 40 帧。状态栏同时显示手和物体对应点位移的 median、P95、max；这些数值用于判断
跨度，不改变训练 flow GT。

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m \
  src.task.ObjectInteractionCm.visualize_grab \
  --index data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/index.json \
  --split train --sequence grab/s1/airplane_fly_1 --future-delta 10 \
  --point-display both --mesh-display off --host 127.0.0.1 --port 8143 \
  --output src/task/ObjectInteractionCm/research/grab_stride_visualization/output/<run_id>
```
