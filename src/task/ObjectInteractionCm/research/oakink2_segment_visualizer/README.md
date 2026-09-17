# OakInk2 主动工具切分查看器

本实验只读消费 OakInk2 active-tool v1.1 索引、原始 annotation 与 Stage3 几何，用于人工检查
primitive 时间窗、最终保留帧、选中对象以及场景上下文。它不生成训练 cache，也不改变 split。

查看器默认打开 v1 相比 v1.1 新增的 47 条有效段，并提供以下筛选：

- 本次新增 47 条；
- 双主物体拆分后保留的 44 条；
- 显式语义修复后保留的 3 条；
- 因既有运动阈值排除的 7 条对照；
- v1.1 全部 2643 条有效段。

手部使用 OakInk2 原始 quaternion MANO 参数在原生世界坐标重建；物体点来自 Stage3 固定采样，
并用原 annotation 的逐帧 SE(3) 放回同一世界坐标。GUI 可以在“最终保留帧”和“primitive 全帧”
间切换，非保留帧只用于诊断，不会被写回 index。

只读检查示例：

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -m \
  src.task.ObjectInteractionCm.research.oakink2_segment_visualizer.run \
  --category new --check-only
```

交互查看器需要显式提供独立输出目录：

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m \
  src.task.ObjectInteractionCm.research.oakink2_segment_visualizer.run \
  --category new --host 127.0.0.1 --port 8142 \
  --output src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/output/<run_id>
```
