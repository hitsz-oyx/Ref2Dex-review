# V1.20 官方 checkpoint 单环境 rollout

- timestamp: 2026-09-19 17:52:15 +0800
- activity_id: ACT-20260919-175215-CMRESIDUAL-V120-OFFICIAL-SINGLE-LIFT
- work_version: V1.20
- mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户要求先使用当前环境验证官方 checkpoint 能否 lift，并明确指定无需完整复现、单环境任意 seed 即可。
- branch: ai/cmresidual/ddp-adapter-plan
- git_commit: 07badcf1ead7fa88d73296fc01d30ca88a1bd688
- run_id: dexplore_v120c_official_single_seed5909_retry2_20260919_174935
- run_status: COMPLETED

## Contract

- 适用 [V1.20](../plan/V1.20.md) 与 [V1.20c](../plan/V1.20c.md)。直接运行 immutable vendor snapshot 的 `dexplore/run.py`，不经过 Horovod/DDP facade；不改 checkpoint、输入或 vendor 源码。
- 官方只读 checkpoint 为外部 `/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth`，SHA256 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- 输入固定为 `coordfix_v4/converted_attempt1` 的 [reconstructed-baseline manifest](../../../../../data/processed_data/dexplore_reconstructed_v120_coordfix_v4/manifest.json)、`s1_airplane_lift`、1 env、GPU 0、seed 5909；运行时 airplane/table mesh 均为已验证的 V1.20c Git-ignored raw GRAB 物化资产。

## Evidence

- [运行目录](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/)、[run manifest](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/run_manifest.json)、[eval log](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/eval.log)、[导出 rollout](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/rl_export/s1_airplane_lift/interaction_hand_inspire.pt)、[rollout 指标](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/rollout_metrics.json)。
- 上游日志显示 `Physics Device: cuda:0`、`num_motions: 1`、`num_envs: 1`、两次成功加载该 checkpoint，并导出一个 RL rollout；进程正常退出，未生成训练 checkpoint、`metrics.jsonl` 或 `train.log`（这是 `--test --export_rl` 评估，不是训练）。
- 导出 tensor `(432, 598)` 全部 finite。object z 的最大相对首帧抬升为 `0.272567093 m`，末帧相对首帧为 `-0.001053214 m`。

## Result and limits

工程与该窄行为假设结论为 `SUPPORTED`：当前环境可加载并执行官方 checkpoint，且这个固定 rollout 出现约 27.26 cm 的瞬时抬升。完整实验解释见 [experiment card](../experiments/EXP-20260919-175215-CMRESIDUAL-V120-OFFICIAL-SINGLE-LIFT.md)。

这不是稳定抓取/放置成功、跨 seed 泛化、官方未公开 producer 的等价性、153 epoch 正式训练结果，亦不构成 CmResidual/DDP 效果结论；这些均保持 `INCONCLUSIVE`。未修改模型、原始数据、processed input、checkpoint、vendor source、`src/base/` 或既有 output。

## Verification and rollback

- 以 [rollout 指标](../../../../../outputs/Dexplore/dexplore_v120c_official_single_seed5909_retry2_20260919_174935/rollout_metrics.json) 复算 object z；输入/输出 tensor 均有 SHA256 记录，导出 tensor finite。
- 本次只新增 ignored 运行产物和版本化的 Activity/experiment 索引记录；没有可回滚的代码变更。若要撤销记录，回滚本 Activity/experiment 提交即可；不删除运行证据。
