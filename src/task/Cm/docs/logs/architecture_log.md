# Cm 任务架构记录

## V1.2 主 Pipeline

```text
GRAB/ARCTIC Stage4
  → 4096-point object correspondence cache
  → ragged 5cm candidate + uint32 B=4 sampling bank
  → sequence-disjoint object-v2 Dataset
  → current object/hand → Frozen DenseToken
  → hard-gated Slot Attention (K=16) → C_m
  → object point-flow decoder → weighted EPE evaluator
```

单个 sample 始终只有一侧 1538 个 hand points，不拼双手。未来 hand flow 进入 action branch；未来 object 只用于构造 `obj_flow_gt`，禁止 future leakage。模型和 loss 不区分 GRAB/ARCTIC，`dataset_id` 只用于分数据集统计。

## 数据与评估合同

输入为 `obj_points/obj_normals [512,3]`、`hand_points/hand_normals [1538,3]`、`hand_flow [1538,3]`、valid mask 和 `delta_time_s`；输出为米制 `pred_obj_flow [512,3]`。stride 为 1--10，train/val/test 按 sequence 互斥。主指标为全 loader 点加权 EPE，checkpoint 使用 `val/mean_stride_epe_mm`；另报 zero-flow、action shuffle/reverse 和 slot gate 诊断。

## Scale 与缓存

`geometry_input_scale`、`hand_flow_input_scale`、`object_flow_target_scale` 职责分离；正式 scale 仅由 train split calibration 产生并在启动时校验。V1.1.2 的 geometry/candidate/sampling cache 保留，约 517GB 的全量 DenseToken bank 已撤回，正式路径在线 Frozen DenseToken。
