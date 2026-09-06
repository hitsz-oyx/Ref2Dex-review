# CmDecoderv2 实验记录

## 2026-09-06 — V1.1.1 Temporal-D2 工程 smoke

- experiment_id: `cmdecoderv2-temporal-d2-smoke-v1.1.1`
- activity_id: [`cmdecoderv2-v1.1.1-20260906-102433-implementation`](activity_log.md)
- hypothesis: 冻结的 Dexplore OICM 能以 `K=4` window 接入 Temporal-D2，并只对新 decoder 反向传播。
- dataset: pilot view；RL-Inspire train/val 各 1 sequence，MANO test 1 sequence 不参与训练或定量评估。
- run: `cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619`（最终 smoke；此前 smoke
  `..._102937` / `..._104051` / `..._104215` / `..._104811` 用于逐项补齐 OICM SHA256、cache manifest、
  perturbation provenance 和 translation loss 语义，均不作为
  provenance，均不作为
  最终证据入口）
- result: 两个 optimizer step 与完整 pilot val 完成；OICM 保持 `eval/no-grad`，decoder head/core 可反向；
  `cm_sample_valid` 和 per-horizon q/wrist 指标正常写入。
- conclusion: `SUPPORTED`（只支持工程 wiring；两步 smoke 不支持收敛、泛化或跨 embodiment 效果结论）。
- evidence: [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/)、
  [run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/metrics.jsonl)、
  [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/train.log)、
  [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/best.pt)。
