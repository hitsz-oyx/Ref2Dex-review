# OakInk2 时序切分 pilot

该诊断按 V1.4.22 修正合同，先用 annotation `frame_id_list` 将 120 Hz mocap 对齐到官方 30 Hz
视频时间轴，再从物体 SE(3) 生成时序稳健的整体运动候选，并读取 Stage3 已缓存的
`hand_to_obj_min_dist`，从与运动重叠的 `<2 cm` seed 向时间轴两侧扩展接触区间。

禁止用固定 `raw_frame_id % 4` 代替官方映射，因为实际相邻 ID 间隔可为 3、4 或 5。可用
`--exclude-segment` 在运动/contact 计算前落实已经人工确认的任务语义排除。

pilot 默认检查削笔器、烧杯、三脚架和 v1.1 的全部静止排除对照。现有 v1/v1.1 index、annotation、
Stage3、split 和训练 cache 全部只读；输出只用于旧/新切分人工比较，不自动成为正式 index。

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -m \
  src.task.ObjectInteractionCm.research.oakink2_temporal_segmentation.run \
  --exclude-segment selected:1772 \
  --output data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/<run_id>
```
