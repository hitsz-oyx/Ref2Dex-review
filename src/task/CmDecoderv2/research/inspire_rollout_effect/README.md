# Inspire rollout self-effect diagnostic

该实验沿用纯 Inspire held-out rollout。对每一步 decoder 生成的
`state_t -> state_{t+1}` 运动，通过固定 URDF 表面对应生成 hand point flow，再将当前物体几何、当前
Inspire 几何和该预测 hand flow 送入冻结 ObjectInteractionCm。

V1.3 配置使用 `unique_knn_edges`：诊断会在本次 run 目录内临时生成纯 Inspire `10135` 点、`K=32`
的 source KNN 索引，供 decoder 的 Cm source window 使用；旁路 effect 则按每个 rollout 预测状态，
对当前 `1024` 个物体采样点重新计算到预测手点的 KNN，再交给 OICM 按 32 条边重算距离和 2 cm
validity。正式 cache、split、checkpoint 和训练配置均不修改。

实验把每一步 decoder 产生的 Inspire hand flow 送入冻结 OICM，观察 OICM 自身输出的 object point-flow
effect。实际 object flow 只作为界面的 GT 对照：它在当前帧单独计算，不会输入 decoder/OICM，也不改变
rollout。由于 OICM 的 hard interaction contract 不定义语义 null-Cm，产物同时保存：

- `raw effect`：dummy 计算路径也会生成的有限 head 输出；
- `effective effect`：当 `sample_valid=false` 时显式归零的合同有效输出。

`--serve` 启动中文交互界面，右侧控件支持：

- `教师强制`：每一帧用该帧真实 Inspire 状态预测下一步；
- `递归Rollout`：点击“从当前帧开始 Rollout”后，从选定 handoff 帧开始递归反馈预测状态；
- 从任意合法帧开始 rollout、前后跳帧、播放/停止；
- 切换 `预测+GT`、`仅预测`、`仅GT`，并查看当前帧预测 effect、GT effect、OICM validity 和
  prediction-GT EPE。GT 仅作显示，不参与模型计算。

运行：

```bash
CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.CmDecoderv2.research.inspire_rollout_effect.run \
  --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml \
  --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt \
  --device cuda:0 \
  --sequence s1/mouse_lift \
  --rl-root data/processed_data/inspire_rl_object_dexplore \
  --activity-id <activity_id> \
  --knn-batch-size 4 \
  --serve
```

运行产物写入本目录 `output/<run_id>/`，包括 `effect.npz`、`effect_summary.json` 和
`run_manifest.json`；V1.3 run 另包含临时 `source_knn_indices.npy`。`effect.npz` 中的
`gt_obj_flow_*`、`gt_effect_rms_mm` 和 `pred_gt_effect_epe_mm` 均为 display-only 字段。
