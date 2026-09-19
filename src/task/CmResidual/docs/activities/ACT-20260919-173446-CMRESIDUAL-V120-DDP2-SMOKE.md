# V1.20a 两 rank DDP PhysX/PPO smoke

- timestamp: 2026-09-19 17:34:46 +0800
- activity_id: ACT-20260919-173446-CMRESIDUAL-V120-DDP2-SMOKE
- work_version: V1.20
- mode: change → run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认 V1.20a 两 rank smoke，并在验证 raw GRAB mesh 后授权采用 V1.20c 运行时物化方案。
- branch: ai/cmresidual/ddp-adapter-plan
- git_commit: 22dc69f75e79cc3d629d179c4de414935171ee32
- run_id: dexplore_v120a_ddp2_smoke_coordfixv4_clean_20260919_173328
- run_status: COMPLETED

## Contract

- 计划：[V1.20a](../plan/V1.20a.md)、[V1.20c](../plan/V1.20c.md)。
- 两 rank × 64 env/rank、horizon 64、minibatch 256、`max_iterations=1`、seed 42；物理 GPU `0,1` 映射到逻辑 `cuda:0,1`。
- 输入为 `coordfix_v4/converted_attempt1`，其 [input manifest](../../../../../data/processed_data/dexplore_reconstructed_v120_coordfix_v4/manifest.json) 标记为 `reconstructed_baseline`；从零初始化，不加载 checkpoint。
- 运行前由 launcher 从 raw GRAB 物化 airplane/table mesh。两项 SHA256 分别为 `dcbb1cce…a0d64f0` 与 `25c6fb8b…0e383`；未使用 Horovod package，遗留 `--horovod` 仅选择进程内 DDP facade。

## Output and evidence

- [运行目录](../../../../../outputs/Dexplore/dexplore_v120a_ddp2_smoke_coordfixv4_clean_20260919_173328/)、[run manifest](../../../../../outputs/Dexplore/dexplore_v120a_ddp2_smoke_coordfixv4_clean_20260919_173328/run_manifest.json)、[config](../../../../../outputs/Dexplore/dexplore_v120a_ddp2_smoke_coordfixv4_clean_20260919_173328/config.json)、[train log](../../../../../outputs/Dexplore/dexplore_v120a_ddp2_smoke_coordfixv4_clean_20260919_173328/train.log)。
- 两个 rank 分别记录 `Physics Device: cuda:0` / `cuda:1`，均完成 GPU PhysX 初始化；日志记录 rank 0/1。
- 训练记录 epoch 1 与 epoch 2（外部 DExplore 的 `epoch_num > max_epochs` 退出语义），rank 0 仅写入一个 [TensorBoard event](../../../../../outputs/Dexplore/dexplore_v120a_ddp2_smoke_coordfixv4_clean_20260919_173328/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/summaries/events.out.tfevents.1789810423.server) 与 [checkpoint](../../../../../outputs/Dexplore/dexplore_v120a_ddp2_smoke_coordfixv4_clean_20260919_173328/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth)。
- checkpoint `epoch=2`、`frame=49152`，递归 tensor finite；上游仅写 TensorBoard，不生成 `metrics.jsonl`。

## Result and limits

工程接线为 `SUPPORTED`：无 Horovod wheel 的 PyTorch/NCCL facade 在两张 GPU 上完成初始化、rollout、更新、同步退出、rank-0 event/checkpoint 写盘及 checkpoint finite 检查。科学结论为 `INCONCLUSIVE`：短 smoke 不评估抓取能力、上游复现等价性或训练效果。

尚未直接输出实际两 rank model/optimizer state hash；受控两 rank numerical fixture 已通过，但该观测缺口意味着阶段 1 的“实际 rank state 一致性”子门尚未独立证实，不能进入四 rank 阶段。第一次运行因缺失 ignored runtime mesh 失败；其后一次预提交成功运行仅用于资产修复诊断，不作为本 Activity 的可复现主证据。

## Verification and rollback

- V1.20c 提交前：定向 pytest `10 passed`、`py_compile`、`git diff --check`、`python tools/verify.py --changed` 通过。
- 本次运行：exit code 0；manifest `COMPLETED`；event/checkpoint 存在并通过有限性检查。
- 回滚代码入口：`git revert 22dc69f`；该操作不删除 Git-ignored runtime mesh 或任何运行输出。若需删除其一，必须另获用户授权。
