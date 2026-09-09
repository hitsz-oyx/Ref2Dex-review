# MANO rollout self-effect diagnostic

该实验使用正式 MANO qualitative-only test source。每个序列在 handoff 帧以 `q=0 + MANO wrist`
初始化 Inspire，随后递归执行 CmDecoderv2 的预测状态。每个 rollout transition 产生的 Inspire hand
point flow 会被送入冻结 OICM，记录 OICM 自己预测的 object effect。

实验不读取 paired Inspire GT，不读取未来 object pose/flow，也不与 object-flow GT 比较。产物同时保存：

- `raw effect`：OICM 无效 sample 的有限 dummy head output；
- `effective effect`：`sample_valid=false` 时归零后的合同有效输出。

运行：

```bash
CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.CmDecoderv2.research.mano_rollout_effect.run \
  --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml \
  --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt \
  --device cuda:0 \
  --sequence s1/camera_takepicture_3_Retake \
  --activity-id <activity_id> \
  --serve
```

运行产物写入本目录 `output/<run_id>/`，包括 `effect.npz`、`effect_summary.json` 和
`run_manifest.json`。Viser 默认显示 effective effect，也可切换到 raw effect。
