# V1.20 官方 checkpoint 单环境瞬时抬升诊断

- experiment_id: EXP-20260919-175215-CMRESIDUAL-V120-OFFICIAL-SINGLE-LIFT
- timestamp: 2026-09-19 17:52:15 +0800
- work_version: V1.20
- run_id: `dexplore_v120c_official_single_seed5909_retry2_20260919_174935`
- git_commit: `07badcf1ead7fa88d73296fc01d30ca88a1bd688`
- input_classification: `reconstructed_baseline`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅限下述假设）

## 假设

当前 vendor DExplore、V1.20 `coordfix_v4` 的 `s1_airplane_lift` 输入、官方只读 `inspire.pth`、单 GPU 单环境和 seed 5909 的组合能够完成可观测的瞬时物体抬升。

## 协议

- 使用 [V1.20 最终计划](../plan/V1.20.md) 的官方 checkpoint Gym rollout 门；不训练、不续训、不改上游 reward/observation/action/physics，也不改输入 tensor。
- 输入为 [reconstructed-baseline manifest](../../../../../data/processed_data/dexplore_reconstructed_v120_coordfix_v4/manifest.json)，被加载 tensor 的 SHA256 为 `305dfd16d9bac0e93a95de5b6fa8b9e721d1edfee6946e49100fe231760a205b`。
- 运行 `num_envs=1`、GPU 0、seed 5909；官方 checkpoint SHA256 为 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- 以导出 `(432, 598)` rollout 中 object position slice `198:201` 的 z 列 `200` 计算相对首帧 lift；不以 reward 替代物理轨迹指标。

## 证据

- [运行目录](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/)、[run manifest](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/run_manifest.json)、[eval log](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/eval.log)、[导出 rollout](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/rl_export/s1_airplane_lift/interaction_hand_inspire.pt)、[指标](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/rollout_metrics.json)。
- 432 帧全部 finite。object z 从 `0.918260634 m` 到最高 `1.190827727 m`，最大相对 lift 为 `0.272567093 m`；分别有 325、102、48、18 帧超过 `0.05`、`0.10`、`0.15`、`0.20 m`。
- 末帧相对首帧为 `-0.001053214 m`，因此行为是瞬时抬升，而非保持放置。

## 结论与限制

结论为 `SUPPORTED`：在这个精确的当前输入/环境/seed 组合中，官方 checkpoint 确实产生了约 27.26 cm 的瞬时抬升。这支持继续把当前环境用作官方 policy 的评估入口。

它不证明稳定抓取或放置成功、跨 seed/序列泛化、153 epoch 从零训练效果、与作者未公开 producer 的等价性，亦不构成 CmResidual 或 DDP 的科研效果结论；这些均为 `INCONCLUSIVE`。
