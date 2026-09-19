# V1.20e 四卡正式训练停止

- timestamp: 2026-09-19 20:40:49 +0800
- activity_id: ACT-20260919-204049-CMRESIDUAL-V120E-TRAIN-STOPPED
- work_version: V1.20
- mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认将四卡 local minibatch 修正为 4096 后启动正式训练；随后明确要求停止复现并释放 GPU。
- branch: ai/cmresidual/ddp-adapter-plan
- git_commit: 96ecba9060b4fc4c5fdd07c7d3472abba40407f0
- base_commit: 96ecba9060b4fc4c5fdd07c7d3472abba40407f0
- run_id: dexplore_v120e_ddp4_2048_globalmb16384_e5000_20260919_180729
- run_status: STOPPED
- last_epoch: 638
- best_metric: training reward 283.37（epoch 545；仅优化诊断）
- latest_checkpoint: [GRAB_00000500.pth](../../../../../outputs/Dexplore/dexplore_v120e_ddp4_2048_globalmb16384_e5000_20260919_180729/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000500.pth)

## Contract and evidence

- 先完成 [4×2048 容量 smoke](../../../../../outputs/Dexplore/dexplore_v120e_ddp4_2048_globalmb16384_smoke_20260919_180616/)：`local minibatch=4096`、global minibatch `16384`、每 PPO epoch 192 optimizer steps；四 rank、checkpoint 与 TensorBoard event 均正常。
- 正式 run 使用 GPU `0,1,3,6`、每 rank 2048 env、horizon 64、local minibatch 4096、seed 42、从零初始化；完整 [run manifest](../../../../../outputs/Dexplore/dexplore_v120e_ddp4_2048_globalmb16384_e5000_20260919_180729/run_manifest.json) 与 [train log](../../../../../outputs/Dexplore/dexplore_v120e_ddp4_2048_globalmb16384_e5000_20260919_180729/train.log) 保留。
- 用户停止时已完成 epoch 638；日志内最高训练 reward 为 `283.37`（epoch 545），最后/唯一周期 checkpoint 是 [epoch 500](../../../../../outputs/Dexplore/dexplore_v120e_ddp4_2048_globalmb16384_e5000_20260919_180729/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000500.pth)。训练中未见 OOM、NCCL、non-finite 或 traceback。
- epoch-500 的独立单环境 lift rollout 记录在 [评估 manifest](../../../../../outputs/Dexplore/dexplore_v120e_e500_eval_seed42_20260919_202636/run_manifest.json) 与 [指标](../../../../../outputs/Dexplore/dexplore_v120e_e500_eval_seed42_20260919_202636/rollout_metrics.json)：最大正向 lift `0.0 m`。

## Result and limits

工程运行结论为 `SUPPORTED`：正确 global minibatch 的 4×2048 DDP 训练可以稳定启动并运行至用户停止。行为假设“epoch-500 checkpoint 在固定 seed-42 rollout 产生正向 lift”为 `REFUTED`；训练 reward 不是抓取证据。正式训练未完成 5000 epoch，故对收敛、抓取、上游复现等价性和泛化均为 `INCONCLUSIVE`。

所有训练/评估进程已经退出，GPU `0,1,3,6` 已释放。没有删除 checkpoint、log、event 或其他运行输出；恢复训练必须重新确认同一 checkpoint、预算和数据合同。
