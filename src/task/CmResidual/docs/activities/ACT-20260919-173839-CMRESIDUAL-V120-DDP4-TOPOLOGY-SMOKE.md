# V1.20a 四 rank DDP topology smoke

- timestamp: 2026-09-19 17:38:39 +0800
- activity_id: ACT-20260919-173839-CMRESIDUAL-V120-DDP4-TOPOLOGY-SMOKE
- work_version: V1.20
- mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户在审阅两 rank 工程 smoke 后，明确要求不补实际 rank-state hash，直接进入四卡。
- branch: ai/cmresidual/ddp-adapter-plan
- git_commit: 65f645e52c8ec912754732057f1d16f144e0ad04
- run_id: dexplore_v120a_ddp4_topology_smoke_coordfixv4_20260919_173750
- run_status: COMPLETED

## Contract

- 适用计划：[V1.20a](../plan/V1.20a.md)、[V1.20c](../plan/V1.20c.md)。
- 物理 GPU `0,1,3,6`，`CUDA_VISIBLE_DEVICES=0,1,3,6`；四 rank 分别使用逻辑 `cuda:0,1,2,3`。
- 每 rank 64 env、horizon 64、minibatch 256、`max_iterations=1`、seed 42；输入仍为 `coordfix_v4/converted_attempt1` 的 [reconstructed-baseline manifest](../../../../../data/processed_data/dexplore_reconstructed_v120_coordfix_v4/manifest.json)，从零初始化。
- 未安装或导入 Horovod package；遗留 `--horovod` 只选择进程内 PyTorch/NCCL facade。airplane/table runtime mesh 均已通过 V1.20c 固定哈希检查。

## Evidence

- [运行目录](../../../../../outputs/Dexplore/dexplore_v120a_ddp4_topology_smoke_coordfixv4_20260919_173750/)、[run manifest](../../../../../outputs/Dexplore/dexplore_v120a_ddp4_topology_smoke_coordfixv4_20260919_173750/run_manifest.json)、[config](../../../../../outputs/Dexplore/dexplore_v120a_ddp4_topology_smoke_coordfixv4_20260919_173750/config.json)、[train log](../../../../../outputs/Dexplore/dexplore_v120a_ddp4_topology_smoke_coordfixv4_20260919_173750/train.log)。
- 日志记录 rank 0–3、`Physics Device: cuda:0..3`、epoch 1/2、正常退出；无 NCCL timeout 或 traceback。
- rank 0 仅写出一个 [TensorBoard event](../../../../../outputs/Dexplore/dexplore_v120a_ddp4_topology_smoke_coordfixv4_20260919_173750/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/summaries/events.out.tfevents.1789810688.server) 与 [checkpoint](../../../../../outputs/Dexplore/dexplore_v120a_ddp4_topology_smoke_coordfixv4_20260919_173750/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth)；checkpoint `epoch=2`、`frame=327680`，递归 tensor finite。上游未生成 `metrics.jsonl`。

## Result and limits

工程 topology/通信结论为 `SUPPORTED`：四 rank 的 GPU PhysX、NCCL、rollout、更新、同步退出与 rank-0 写盘在该短预算下完成。

这不是 V1.20a 原定义的 `4×2048` 语义保持 smoke：当前 launcher 的已提交工程 smoke 合同固定每 rank 64 env/minibatch 256，未实现历史 Horovod 路径的 64×256 梯度累积，也未输出实际 rank 间 model/optimizer hash。依据用户明确授权，本次没有补该观测；不得将它解释为 8192-env 语义、正式训练准入、抓取能力或上游复现等价性。科学结论为 `INCONCLUSIVE`。

## Verification and rollback

- exit code 0；manifest `COMPLETED`；event/checkpoint 存在且 checkpoint finite。
- 已有 V1.20c 静态验证：10 个定向 pytest、`py_compile`、`git diff --check`、`python tools/verify.py --changed` 通过。
- 此次仅产生 ignored output；回滚代码入口仍为 `git revert 22dc69f`，不删除任何运行输出或 ignored runtime mesh。
