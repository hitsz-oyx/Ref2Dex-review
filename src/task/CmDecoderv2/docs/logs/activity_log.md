# CmDecoderv2 活动记录

- scope: `src/task/CmDecoderv2/`
- last_updated: 2026-09-13
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)

## 2026-09-13 10:29:45 +0800 - V1.1.14 IsaacGymEnvs vendor 迁移记录

- timestamp: 2026-09-13 10:29:45 +0800
- activity_id: ACT-20260913-102945-CM-ISAACGYM-VENDOR
- modification_version: V1.1.14
- type: governance / code / operation / documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认将 IsaacGymEnvs 纳入 Ref2Dex 的 `third_party/IsaacGymEnvs` subtree/vendor 路径。
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- import_commit: e1aabc1897bd27c793a765ec58e6145fb47ee02f
- scope: upstream `release/1.5.1` commit `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` 已以 squashed snapshot 导入；外部 checkout 保留为只读 mirror；当前 CmResidual 修改随 vendor 路径保留为未提交差异；未迁移 runs、日志、cache 或 checkpoint。
- run_id: `cm_residual_decoder_bank_vendor_smoke_20260913_102945`
- run_status: COMPLETED
- output: [runs/CmResidual_13-10-29-45](../../../../../runs/CmResidual_13-10-29-45/)、[run_manifest.json](../../../../../runs/CmResidual_13-10-29-45/run_manifest.json)。vendor 路径 smoke exit_code=0，action `(12,)`、observation `(71,)`。
- conclusion: SUPPORTED（vendor 路径和现有 RL task 可运行）；INCONCLUSIVE（正式RL和物理任务效果）。

**原因**

统一到 Ref2Dex 后，VSCode 的同一 Git 工作区可以直接显示 IsaacGymEnvs 基线和 CmResidual 本地修改；上游 provenance 与回滚入口保留在 [third_party/IsaacGymEnvs/UPSTREAM.md](../../../../../third_party/IsaacGymEnvs/UPSTREAM.md)。

**验证**

- vendor task/config `py_compile` 通过；Ref2Dex 定向测试 `8 passed`。
- 从 Ref2Dex 根目录执行 vendor IsaacGymEnvs GPU smoke，object asset 解析到 vendor 内部路径，单 PPO epoch 完成；未完成 episode，`rew=-inf` 不作策略结论。
- 根 `runs/` 已加入 `.gitignore`；外部仓库未删除、reset 或改写历史。

**累计工作树路径**

`src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/runner.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`; `src/task/CmDecoderv2/configs/active/mano_actual_finetune_v1_batch8.yaml`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/rl/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_residual_contract.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/prepare_mano_actual_finetune_view.py`; `src/task/CmDecoderv2/tools/rl/`。

## 2026-09-13 10:13:44 +0800 - V1.1.14 当前 best.pt 接入残差 RL smoke

- timestamp: 2026-09-13 10:13:44 +0800
- activity_id: ACT-20260913-101344-DECODER-BANK-RL-SMOKE
- modification_version: V1.1.14
- type: code / operation / diagnostic / documentation
- task_mode: change + run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户要求先用当前 best.pt 跑通 RL 训练接口；沿用已批准的 reference-conditioned residual RL 范围，不启动正式策略训练。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true（保留既有差异；外部 IsaacGymEnvs 修改不回滚）
- scope: 新增 Task-local decoder bank exporter；外部 IsaacGymEnvs `CmResidual` 增加显式 `decoder_bank` provider 和 `CmResidualDecoderBank` alias/config。当前 decoder best 在离线 paired view 上以 teacher-forced one-step 生成 q/wrist bank，RL task 启动时校验 bank schema 与 checkpoint SHA，再冻结 bank 作为 base；未声称在线 decoder 闭环。
- decoder_bank_run_id: `rl_decoder_bank_s1_airplane_best_20260913`; status: COMPLETED; 432帧、覆盖428帧；checkpoint SHA=`420ba5798bb668cc96bfbb266bf999bb17ef920b247410afc860245eca276201`。
- run_id: `cm_residual_decoder_bank_isaac_smoke_20260913_101311`
- run_status: COMPLETED
- command: `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home2/wyy/oyx_ws/Ref2Dex:/home2/wyy/oyx_ws/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python isaacgymenvs/train.py task=CmResidualDecoderBank train=CmResidualPPO headless=True num_envs=4 max_iterations=1 force_render=False pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 train.params.config.minibatch_size=128`
- output: `/home2/wyy/oyx_ws/IsaacGymEnvs/runs/CmResidual_13-10-13-11/`；one PPO epoch completed, action space `(12,)`, observation space `(71,)`；未有 episode termination，`rew=-inf` 是一轮 smoke 的统计伪影。
- conclusion: SUPPORTED（best.pt→decoder bank→IsaacGymEnvs provider→PPO 单迭代接口）；INCONCLUSIVE（在线闭环解码、残差策略学习、抓取成功）。

**原因**

现有 task 的 `decoder` provider 原先 fail-closed，直接运行会绕过 best.pt。此次增加可审计的离线 bank provider，先验证 checkpoint 身份、base 轨迹加载、物理环境、动作/观测合同和 PPO wiring；bank 明确不是在线 receding-horizon decoder。

**验证**

- exporter 在 GPU4 strict-load 当前 best.pt，输出 432 帧 q/wrist bank，manifest 中 checkpoint SHA 与文件一致。
- IsaacGymEnvs alias task 启动时成功校验 decoder bank schema/SHA，加载 Inspire 18-DOF 资产，报告 action `(12,)`、observation `(71,)`，完成一轮 PPO；exit_code=0。
- `run_manifest.json` 已写入 RL run 目录；外部日志中的 `rew=-inf` 仅因 max_iterations=1 且无 episode 完成，不作为策略结果。

**产物与回滚**

- [decoder bank](../../../../../outputs/cmdecoderv2/rl_decoder_bank_s1_airplane_best_20260913/)、[bank manifest](../../../../../outputs/cmdecoderv2/rl_decoder_bank_s1_airplane_best_20260913/manifest.json)。
- RL smoke目录：`/home2/wyy/oyx_ws/IsaacGymEnvs/runs/CmResidual_13-10-13-11/`；run_manifest：`/home2/wyy/oyx_ws/IsaacGymEnvs/runs/CmResidual_13-10-13-11/run_manifest.json`。
- 修改入口：`src/task/CmDecoderv2/tools/rl/export_decoder_bank.py`、外部 `IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py`、`tasks/__init__.py`、`cfg/task/CmResidualDecoderBank.yaml`；回滚只隔离这些新增/修改的接口文件与 bank/run 输出，不删除当前 decoder 微调或旧 RL smoke。

**累计工作树路径**

`docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/runner.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`; `src/task/CmDecoderv2/configs/active/mano_actual_finetune_v1_batch8.yaml`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/rl/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_residual_contract.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/prepare_mano_actual_finetune_view.py`; `src/task/CmDecoderv2/tools/rl/`。

## 2026-09-13 10:02:37 +0800 - V1.1.14 微调中期状态核对

- timestamp: 2026-09-13 10:02:37 +0800
- activity_id: ACT-20260913-100237-MANO-ACTUAL-FINETUNE-STATUS
- modification_version: V1.1.14
- type: diagnostic / operation / documentation
- task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问当前微调是否收敛及剩余时间；只读运行日志、metrics、进程和checkpoint，不改变训练变量。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true（保留既有差异）
- scope: 核对 run_id `cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251` 的中期状态。
- run_status: RUNNING；last_observed_step=8500/31940，已完成5个epoch，epoch6进行中；GPU三卡进程仍存在。
- validation_snapshot: val/loss 为 epoch1/2/3/4/5 = 0.00615713/0.00601656/0.00596945/0.00603457/0.00600916；当前 best 为 epoch3/step4791；val h1 point-flow EPE 最佳约8.428mm，epoch5为8.432mm；val h1 q MAE 最佳约0.03986rad（epoch3），epoch5回升至0.04821rad。
- convergence_assessment: 尚不能称最终收敛；epoch3后验证指标进入平台并有轻微波动，属于中期 plateau 信号，20 epoch 计划仍继续执行。当前吞吐估计剩余约1.7–1.9小时，随验证和机器负载变化。
- conclusion: SUPPORTED（运行正常、指标有限、checkpoint持续写入）；INCONCLUSIVE（最终收敛、微调收益和跨手/RL效果）。

**原因**

中期验证集改善在 epoch3 后停止，需区分暂时平台与完整训练收敛；不提前停止，以免把 cosine 退火尚未完成的运行误判为终态。

**验证**

- `metrics.jsonl` 已有94条记录和5条完整 val 记录，全部可解析；`best.pt` 为epoch3附近，`latest.pt` 已更新到epoch5。
- `train.log` 最新记录 step8500/epoch6，未见异常退出、OOM或非有限值；`pgrep` 显示 torch.distributed 三进程仍在运行。

**产物**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/)、[metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/train.log)、[best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/checkpoints/best.pt)、[latest.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/checkpoints/latest.pt)。

**累计工作树路径**

`docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/runner.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`; `src/task/CmDecoderv2/configs/active/mano_actual_finetune_v1_batch8.yaml`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/rl/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_residual_contract.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/prepare_mano_actual_finetune_view.py`。

## 2026-09-13 09:24:02 +0800 - V1.1.14 MANO→actual Inspire 微调启动

- timestamp: 2026-09-13 09:24:02 +0800
- activity_id: ACT-20260913-092402-MANO-ACTUAL-FINETUNE-START
- modification_version: V1.1.14
- type: data / experiment / operation / diagnostic
- task_mode: change + run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求开始已确认的 MANO source→actual Inspire 两阶段微调；沿用最终计划，不改变6维独立输出、实际12维状态或三卡设置。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true（保留既有差异；未覆盖旧代码、数据或checkpoint）
- scope: 使用 paired view 将 MANO parent surface 作为 source、Dexplore actual Inspire geometry/q/wrist 作为 target；254 train/30 val，排除 `s2/mug_drink_2` 的缺失 MANO provenance。冻结 OICM，加载阶段一 best 作为 decoder 初始化，三卡每卡 batch8/global24，20 epoch；不修改RL任务或旧运行。
- paired_view_run_id: `mano_actual_finetune_v1_1_14_eligible_20260913`; paired_view_status: COMPLETED; counts: train=254, val=30, test=0; excluded=1。
- smoke_run_id: `cm_decoder_v2_mano_actual_finetune_smoke_20260913_091945`; smoke_status: COMPLETED; 10 steps/42s；有限 train/val 指标，strict 初始化加载通过。
- run_id: `cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251`
- run_status: RUNNING
- command: `CUDA_VISIBLE_DEVICES=0,1,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 timeout --signal=INT --kill-after=30s 21600 /home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --distributed --config src/task/CmDecoderv2/configs/active/mano_actual_finetune_v1_batch8.yaml`
- output: [outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/)（RUNNING；终态指标/checkpoint待补写）
- progress_snapshot: 2026-09-13 09:24 左右 step=200/31940，train h1 point-flow EPE=6.04841mm，loss=0.00434146，进程仍运行；当前吞吐估计剩余约2.4小时，仅作动态估计。
- conclusion: SUPPORTED（paired 数据契约、三卡初始化与训练循环 smoke/正式运行可用）；INCONCLUSIVE（MANO→actual 微调是否改善、跨手效果与RL任务成功）。

**原因**

用户已确认先完成两阶段 decoder 微调；本次先完成 source/target 数据隔离、eligible 样本门控和短程 smoke，再启动正式三卡运行。当前只记录工程运行证据，不把中途指标解释为科研结论。

**验证**

- `CUDA_VISIBLE_DEVICES='' ... pytest -q src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py src/task/CmDecoderv2/tests/test_kinematics.py src/task/CmDecoderv2/tests/test_pointflow.py src/task/CmDecoderv2/tests/test_dataset.py src/task/CmDecoderv2/tests/test_residual_contract.py`：17 passed。
- paired view `index.json`、根 `manifest.json`、`run_manifest.json` 均可解析；source/target stream smoke shape、finite、source active mask 和 target supervision 均通过。
- smoke train/val metrics 全部有限，最终 exit_code=0；正式 run 已完成 setup、三进程均加载 `initial.pt`，未检测到OOM或配置回退。

**产物与回滚**

- paired view：[data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913](../../../../../data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/)、[index.json](../../../../../data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/index.json)、[manifest.json](../../../../../data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/manifest.json)、[run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/run_manifest.json)。
- formal output：[运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/)、[run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/run_manifest.json)、[metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/train.log)（运行中；终态checkpoint链接待补写）。
- 初始化权重：[initial.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_20260913/checkpoints/initial.pt)；旧阶段一权重和旧失败paired prep均保留，不删除、不覆盖。
- 回滚入口：停止 `run_id` 对应进程并隔离本次新增配置/paired view/运行目录；不回滚用户既有工作树差异。

**累计工作树路径**

`docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/runner.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`; `src/task/CmDecoderv2/configs/active/mano_actual_finetune_v1_batch8.yaml`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/rl/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_residual_contract.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/prepare_mano_actual_finetune_view.py`。

## 2026-09-13 08:50:12 +0800 - V1.1.14 长训终态核对

- timestamp: 2026-09-13 08:50:12 +0800
- activity_id: ACT-20260913-085012-COUPLED-LONG-END
- modification_version: V1.1.14
- type: diagnostic / operation / documentation
- task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问训练是否完成；只读已有运行并补记原用户批准长训的终态，不重跑或改变实验变量。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true（保留检查前全部既有差异）
- scope: 只追加本活动；不修改代码、配置、数据、checkpoint、版本指针、实验结论、指导或架构快照。
- run_id: cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622
- run_id_correction: 启动条目的 `cm_decoder_v2_coupled_geometric_batch8_long_20260912_234000` 是预先登记别名；此处以实际目录及原 run_manifest 的 run_name 为准，对应 ACT-20260912-234000-COUPLED-LONG-START，不是新运行。
- run_status: COMPLETED
- last_step: 83150
- last_epoch: 50
- best_metric: val/loss=0.0017697377727122（epoch47/step78161）
- elapsed: 06:31:48（train.log 训练循环记录）；最新checkpoint于2026-09-13约06:28落盘。
- exit_code: 0（恢复原执行session 73337取得）；pgrep 未发现该训练进程。
- command: `CUDA_VISIBLE_DEVICES=0,1,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 timeout --signal=INT --kill-after=30s 36000 /home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --distributed --config src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml --set name=cm_decoder_v2_coupled_geometric_batch8_long --set train.epochs=50 --set train.max_steps=null`
- conclusion: SUPPORTED（正常完成与已有指标/checkpoint一致）；INCONCLUSIVE（Cm跨手能力、闭环适配和RL效果）。不新增科研结论。

**文件**

仅追加 [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md)；沿用 [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)，未修改计划。

**原因**

用户查询既有训练状态；用进程退出码、完成日志、50轮验证记录和checkpoint字段交叉核对，避免把进程消失误报为正常完成。

**验证**

- `pgrep -af '[c]m_decoder_v2_coupled_geometric_batch8_long'` 无匹配；`tail -n 10 outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/train.log` 含 `Training finished at step 83150 in 06:31:48.`；原执行session返回exit_code=0。
- graspenv Python按JSON解析metrics.jsonl：931条记录、50条val记录，无非有限数；argmin(val/loss)为epoch47，与best.pt一致。CPU读取best/latest，各124个model tensor全部有限；latest为epoch50/step83150。此核对不替代重新推理评估或strict模型重建测试。
- 最佳epoch47的h1表面EPE=2.530714mm、四horizon平均=6.944699mm、腕平移=3.654358mm、腕旋转=1.796353deg；h1 q MAE=0.007450926rad，identity=0.006188740rad，手指角度尚未超过该基线。
- epoch45/47/50 val/loss=0.001779085/0.001769738/0.001771657；末几轮变化很小，但cosine学习率已趋近零，不据此证明换优化日程也不会改善。
- 本次为几何监督50epoch；未运行MANO参考到实际Inspire微调、在线decoder仿真或正式残差RL训练。

**产物与回滚**

- 运行目录：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/)；manifest：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/run_manifest.json)。
- 指标：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/metrics.jsonl)；日志：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/train.log)。
- 最佳：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/checkpoints/best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/checkpoints/best.pt)；最近：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/checkpoints/latest.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/checkpoints/latest.pt)。
- 只追加状态记录，无运行产物写入，无需模型/数据回滚；不修改或清理已有dirty工作树。
- 规范反馈：无新增阻碍；审计仅限定本次追加的activity文件，不把其他既有dirty文件误记成本次修改。

## 2026-09-13 00:19:02 +0800 — V1.1.14 IsaacGymEnvs CmResidual wiring smoke

- timestamp: 2026-09-13 00:19:02 +0800
- activity_id: ACT-20260913-001902-CM-RESIDUAL-SMOKE
- modification_version: V1.1.14
- type: code / operation / diagnostic
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认 airplane_lift、12维残差动作、冻结 Cm/OICM/base、71维实际状态观测四项接口。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- scope: Ref2Dex 新增残差动作展开/映射合同与测试；外部 IsaacGymEnvs 注册 `CmResidual` task、task/train 配置。未修改 Cm decoder/OICM 主干、数据 split、GT 或长训配置。
- files: [src/task/CmDecoderv2/rl/residual_contract.py](../../rl/residual_contract.py)、[src/task/CmDecoderv2/tests/test_residual_contract.py](../../tests/test_residual_contract.py)、外部 `/home2/wyy/oyx_ws/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py`、`tasks/__init__.py`、`cfg/task/CmResidual.yaml`、`cfg/train/CmResidualPPO.yaml`。
- environment: 在共享 `graspenv` 安装/升级 hydra-core、omegaconf、pyvirtualdisplay、warp-lang、pysdf、urdfpy、rl-games==1.6.5；其中 urdfpy 将 networkx 降至2.2，rl-games 同时更新 gym/psutil 等，属于环境漂移风险，未改变长训进程。
- run_id: `cm_residual_isaac_smoke_20260913_001902`
- run_status: COMPLETED
- command: `CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home2/wyy/oyx_ws/Ref2Dex:/home2/wyy/oyx_ws/IsaacGymEnvs OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python isaacgymenvs/train.py task=CmResidual train=CmResidualPPO headless=True num_envs=4 max_iterations=1 force_render=False pipeline=gpu sim_device=cuda:0 rl_device=cuda:0 train.params.config.minibatch_size=128`
- output: `/home2/wyy/oyx_ws/IsaacGymEnvs/runs/CmResidual_13-00-18-30/nn/last_CmResidual_ep_1_rew_-inf.pth`
- evidence: Hydra registration, Inspire 18-DOF asset load, GPU PhysX, action space `(12,)`, observation space `(71,)`, actor forward and one PPO epoch completed. `rew=-inf` is a one-epoch/no-terminated-episode artifact, not a policy result.
- conclusion: SUPPORTED（任务注册、动作/观测合同和单步 GPU wiring）；INCONCLUSIVE（残差策略学习、物理抓取、CmDecoder 在线闭环）。`reference_frozen` 是唯一可运行 provider；`decoder` provider 在缺少 Cm-bank adapter 时 fail-closed。
- validation: Ref2Dex focused pytest `17 passed`; external task py_compile passed. Smoke 前三次失败均为局部兼容/索引问题，已修复并在上述终态重跑通过。
- rollback: 隔离外部 checkout 的四个新增/修改文件和 Ref2Dex 新增 `rl/`；不删除长训、旧输出或用户已有改动。
- cumulative_dirty_paths: `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`, `src/task/CmDecoderv2/dataset.py`, `src/task/CmDecoderv2/docs/logs/experiment_log.md`, `src/task/CmDecoderv2/docs/plan/v1.1.md`, `src/task/CmDecoderv2/kinematics.py`, `src/task/CmDecoderv2/pointflow.py`, `src/task/CmDecoderv2/research/dexplore_contract_audit/`, `src/task/CmDecoderv2/rl/`, `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`, `src/task/CmDecoderv2/tests/test_pointflow.py`, `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`, `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`, `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`。

**原因**

按用户确认的四项接口搭建 IsaacGymEnvs 残差策略最小任务，并在不占用长训 GPU 的 GPU4 上做单迭代 wiring 验收；decoder 在线 provider 依赖尚未实现的 Cm-bank adapter，因此保持显式 fail-closed。

**验证**

`CUDA_VISIBLE_DEVICES='' ... pytest ...`：17 passed；外部 task 与 Ref2Dex contract `py_compile` 通过；IsaacGymEnvs 单迭代 smoke 完成并生成 checkpoint。当前长训仍为 `RUNNING`，不在本条中提前写终态指标。

## 2026-09-12 23:40:00 +0800 — V1.1.14 三卡50epoch长训启动

- timestamp: 2026-09-12 23:40:00 +0800
- activity_id: ACT-20260912-234000-COUPLED-LONG-START
- modification_version: V1.1.14
- type: operation / experiment
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求启动长训；沿用已完成的全量FK验收和5epoch batch8配置，不改变研究变量。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- scope: 从随机decoder开始，冻结OICM，使用V1.1.14耦合view、`v1_3_cache`、三卡每卡batch8、50epoch；不resume 5epoch checkpoint，不修改RL代码、外部IsaacGymEnvs、数据/GT/split或共享src/base。
- run_id: `cm_decoder_v2_coupled_geometric_batch8_long_20260912_234000`
- run_status: STARTED
- command: `CUDA_VISIBLE_DEVICES=0,1,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 timeout --signal=INT --kill-after=30s 36000 /home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --distributed --config src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml --set name=cm_decoder_v2_coupled_geometric_batch8_long --set train.epochs=50 --set train.max_steps=null`
- initial_checkpoint: none；每卡batch8/global24、seed42、FP32、无状态扰动、workers0；预计约6–7小时，超时或资源异常安全停止。
- output: [outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_long_20260912_235622/)（RUNNING；实际目录已生成，终态指标/最佳checkpoint待补写）。
- progress_snapshot: 2026-09-13 00:19 左右已完成约3个epoch/4989步，`val/loss=0.00938727`；进程仍在运行，预计总耗时约6–7小时。

**原因**

5epoch仍有持续但变小的验证集改善，且未观察到过拟合；用户要求继续长训。该运行只回答几何阶段优化是否继续改善，不代表跨手迁移或物理RL成功。

**验证**

- 启动前FK gate：`coupled_fk_gate_v14_20260912_224950` COMPLETED；最大标签到FK误差0.000491738mm。
- 5epoch前置运行：`cm_decoder_v2_coupled_geometric_batch8_20260912_225225` COMPLETED，作为速度和显存基准，不作为初始化权重。
- 终态将补写last_step/last_epoch、best metric、checkpoint、metrics/train log及结论；运行期间不创建RL代码。

## 2026-09-12 23:34:52 +0800 — V1.1.14 三卡batch8的5epoch训练完成

- timestamp: 2026-09-12 23:34:52 +0800
- activity_id: ACT-20260912-233452-COUPLED-FIVE-EPOCH-END
- modification_version: V1.1.14
- type: code / diagnostic / experiment / operation / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户要求耦合标签与预测FK一致后训练、扩大batch并继续；沿用已说明的三卡200步smoke和5epoch初步训练范围。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- scope: 新增batch8显式配置、只读全帧FK验收及定向测试、新版本同内容轻量view与本次运行/记录；不改Cm主干、6维输出、原始参考/实际状态、旧cache/split/checkpoint、共享src/base、外部仓库、指导和架构快照。
- run_id: cm_decoder_v2_coupled_geometric_batch8_20260912_225225
- run_status: COMPLETED
- last_step: 8315
- last_epoch: 5
- best_metric: val/loss=0.00901879200305763（epoch5/step8315）
- actual_elapsed: 40分24秒（训练循环含5次验证和checkpoint），exit_code=0；未触发1小时timeout，无OOM，GPU2/4/5训练进程均已退出。
- command: `CUDA_VISIBLE_DEVICES=2,4,5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 timeout --signal=INT --kill-after=30s 3600 /home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --distributed --config src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`
- conclusion: SUPPORTED（全量标签FK合同及batch8三卡工程可运行）；INCONCLUSIVE（Cm跨手表征、完整手运动学习、实际状态适配或RL成功）。不将本次无扰动初步训练称为闭环训练或合格RL base。

**文件**

- 本次新增：[src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml](../../configs/active/coupled_geometric_v1_batch8.yaml)、[src/task/CmDecoderv2/research/dexplore_contract_audit/verify_coupled_view.py](../../research/dexplore_contract_audit/verify_coupled_view.py)；扩展[src/task/CmDecoderv2/tests/test_pointflow.py](../../tests/test_pointflow.py)；更新计划/版本/活动/实验记录。生成view和输出均被Git忽略，没有提交。
- 为累计worktree审计列出本边界保留文件，不表示本事件再次改写它们：`docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`; `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`。

**原因**

修正旧启动遗漏的V1.3表面采样配置，按用户扩大batch的请求确认运行稳定性及真实epoch耗时。4个旧耦合smoke的学习证据已在本版本重新标记INVALID_IMPLEMENTATION，旧目录不删、权重不resume。

**验证**

- 全量FK gate run_id `coupled_fk_gate_v14_20260912_224950`、run_status COMPLETED：285条/72935帧，最大逐点误差0.000491738mm；mimic和全部手指限位通过。新旧view的285条q/wrist逐位一致，348条身份/split/geometry路径不变。原数据不覆盖。
- batch8/global24、GPU2/4/5、graspenv、FP32、workers0、无状态扰动、seed42、新初始化decoder，冻结OICM v1.3。8315步性能窗口均值266.802ms，窗口p05/p95为261.491/271.726ms，data_wait平均占46.51%；验证间隔实测约484–488秒/epoch。同设置50epoch约6.7小时，非收敛时间承诺。
- 5轮val/loss依次0.00943987、0.00947883、0.00910965、0.00906360、0.00901879；h1表面EPE依次11.4056、11.3262、10.9544、10.8358、10.7753mm；最终四horizon平均25.9471mm。数字使用现有Runner聚合，不冒充独立样本micro或序列macro统计。
- 最终h1腕部平移12.1966mm，identity为15.6140mm；手指q MAE0.00779017rad，identity为0.00618874rad；腕部旋转2.57239deg，identity为2.47760deg。改善不是各分量一致发生；没有Cm交换对照、独立test或物理rollout，不能归因于跨手表征成立。
- 定向命令：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py src/task/CmDecoderv2/tests/test_kinematics.py src/task/CmDecoderv2/tests/test_pointflow.py src/task/CmDecoderv2/tests/test_dataset.py`：15 passed；包含新配置与view版本/表面对应，以及legacy采样不对应的回归。`git diff --check`通过。
- 终态读取93条metrics记录，所有数字有限；best/latest均为step8315/epoch5，内嵌V1.1.14、新view及`v1_3_cache`。由checkpoint内嵌config重建模型并`load_state_dict(strict=True)`全key匹配，权重有限；60个OICM state tensor与原checkpoint逐位一致，全部OICM参数仍冻结。
- best.pt SHA256：`1739175b789e22158b67d4c2f7b653f32f519dbba5e31f1bb4713027dac134b6`；原OICM文件SHA仍为`3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。

**产物与回滚**

- 运行目录（56MiB）：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/)。
- [outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/config.json)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/run_manifest.json)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/metrics.jsonl)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/train.log)。
- 最佳：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/checkpoints/best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/checkpoints/best.pt)；最近：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/checkpoints/latest.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/checkpoints/latest.pt)。
- 全量验收：[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/run_manifest.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/run_manifest.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/verification.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/verification.json)。
- 轻量view（27MiB）：[data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912](../../../../../data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912/)、[data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912/run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912/run_manifest.json)。
- 计划/结果：[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)、[src/task/CmDecoderv2/docs/logs/experiment_log.md](experiment_log.md)。回滚只隔离新增配置/诊断/轻量view和本次训练输出；已有模型与产物保持可用。未启动50epoch、MANO→actual微调或RL训练。

**规范反馈**

当前Runner把view生成版本与训练运行版本强制相等，扩大batch或修复配置升级版本也必须重建27MiB同内容view，产生重复sidecar和误写数据谱系的风险。本次workaround为新建view并逐位核验，不修改校验。建议后续仅在本Task版本校验入口区分数据schema兼容性与运行modification_version，保留全部shape/坐标/split校验和旧checkpoint读法；涉及合同变更，需用户确认后另行实施。本轮不修改AGENTS、Skill或公共合同。活动审计要求累计dirty文件重复列入，已明确标为保留差异；未覆盖用户改动。

## 2026-09-12 22:52:56 +0800 — V1.1.14 batch8短跑通过，启动三卡5epoch初步训练

- timestamp: 2026-09-12 22:52:56 +0800
- activity_id: ACT-20260912-225256-COUPLED-FIVE-EPOCH-START
- modification_version: V1.1.14
- type: diagnostic / experiment / operation / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户“你扩大batch吧”“继续”，已说明一致性闸门、200步smoke及5epoch预算。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- scope: 完成V1.1.14同内容轻量view/配置FK验收、三卡短跑，再启动无扰动5epoch初步训练。模型/6维控制/旧数据/划分/cache/checkpoint、共享src/base与外部仓库不变，不训练阶段二或RL。
- run_id: cm_decoder_v2_coupled_geometric_batch8_20260912_225225
- run_status: RUNNING
- command: `CUDA_VISIBLE_DEVICES=2,4,5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 timeout --signal=INT --kill-after=30s 3600 /home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --distributed --config src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`
- 初始checkpoint：decoder随机初始化，不resume任一smoke；OICM固定原v1.3 SHA。每卡batch8/global24、lr3e-4/cosine、5epoch、perturb_train=false、workers0、FP32；预算1小时，异常或超时停止。22:52:56 GPU2/4/5已用显存为14081/16292/14221MiB，只是瞬时值，不是峰值。
- 累计未提交files: `docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`; `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`。本事件只增加运行与验证记录，既有实现不重写。

**原因**

采样错误已定位并隔离；需要在正确点对应下验证扩大batch的工程可行性，然后观察完整epoch。新view只满足Runner版本锁，不改变GT和split；修正版本锁的公共合同不在本次范围。

**验证**

- view run_id `cmdecoderv2-view-full-20260912-224828`，run_status COMPLETED。使用旧source index和既有builder，生成255/30/63分配；285条新旧q/wrist用`np.testing.assert_array_equal`逐位相同，348条geometry_root/id/frame_count/variant相同。命令与统计：[data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912/run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912/run_manifest.json)、[data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912/manifest.json](../../../../../data/processed_data/cm_decoder_v2/coupled_geometric_v1_1_14_20260912/manifest.json)。
- FK run_id `coupled_fk_gate_v14_20260912_224950`，run_status COMPLETED，44.695秒；重跑配置同上一活动，仅run-id改变。285条/72935帧结果完全复现：max点误差0.000491738mm，mimic和限位通过。加入训练/view版本一致及legacy对应必失败的回归测试，15 passed（4.39秒）。
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950](../../research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/run_manifest.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/run_manifest.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/verification.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_v14_20260912_224950/verification.json)。
- smoke run_id `cm_decoder_v2_coupled_batch8_smoke_20260912_225004`，run_status COMPLETED，last_step200/last_epoch1（不足一个epoch），95秒含validation；命令与上一失败尝试相同，配置已改为新view。warmup20后四段perf=270.275/269.599/268.227/265.683ms，88.80–90.33样本/秒，data_wait占46%–47%；无OOM。best_metric val/loss=0.0131651，h1 EPE14.3633mm，仅smoke不是收敛或迁移结论。
- [outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/config.json)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/run_manifest.json)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/metrics.jsonl)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/train.log)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/checkpoints/best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/checkpoints/best.pt)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/checkpoints/latest.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_225004/checkpoints/latest.pt)。
- `git diff --check`通过；`python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/CmDecoderv2/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmDecoderv2 --scope-prefix docs/current_versions.yaml --check-links`上一事件20路径/8链接通过，本事件交接前再审计。
- conclusion: SUPPORTED（标签FK与batch8三卡工程可运行）；INCONCLUSIVE（初步训练、跨手及物理效果）。按1663step/epoch和约0.27s训练步长，估算5epoch约40分钟，50epoch约6–7小时，需实测完整epoch校准。

**运行入口**

[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/config.json)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/run_manifest.json)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/metrics.jsonl)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/train.log)。
计划：[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)；配置：[src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml](../../configs/active/coupled_geometric_v1_batch8.yaml)。回滚为隔离新配置/脚本/view/运行，不删除旧产物。

## 2026-09-12 22:44:21 +0800 — V1.1.14 更正采样配置、全量FK验收与batch8短跑启动

- timestamp: 2026-09-12 22:44:21 +0800
- activity_id: ACT-20260912-224421-COUPLED-BATCH8-START
- modification_version: V1.1.14
- type: code / diagnostic / experiment / operation / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户要求标签到预测FK一致性后才三卡训练，并要求“你扩大batch吧”“继续”；按已批准目标修正遗漏配置。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- scope: 新增Task配置、只读验收脚本及定向测试，更新计划/版本/记录；不修改模型主干、6维控制、旧数据/划分/cache/checkpoint、共享src/base、外部仓库和其他运行。实际观测的完整native18路径保持，无扰动设置不变。
- files: `docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`; `src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml`。列表覆盖本边界累计未提交差异；本事件只修改上述新配置、验收脚本、测试与文档。

**原因**

此前4个耦合训练smoke遗漏`model.surface_sampling`，模型默认`legacy_urdf`而目标是`v1_3_cache`。逐点监督不对应；此前COMPLETED只代表进程结束，学习证据重新标记`INVALID_IMPLEMENTATION`，不resume其权重、不据此评价Cm。
受影响run_id为`cm_decoder_v2_20260912_212342`（10步）、`cm_decoder_v2_20260912_212609`（100步）、`cm_decoder_v2_20260912_215725`（20步batch8）、`cm_decoder_v2_20260912_215917`（20步batch4）；均只执行了部分epoch。
22:01活动用“20步+整套validation”的总时间推算1–2天50epoch没有依据，撤回该估算及未计稳定训练时间的batch吞吐对比。性能monitor预热20步，这两次20步运行没有稳定perf数据。

**验证**

- run_id: `coupled_fk_gate_20260912_224800`；run_status: COMPLETED；实际22:43:09启动、54.906秒。全285条/72935帧/739196225点，均值`0.000016003mm`、最大`0.000491738mm`，mimic最大`5.96046e-8rad`，限位最大浮点超差`2.38419e-8rad`，均通过。结论`SUPPORTED`仅指拟合参考与配置FK对应，不是Cm学习/跨手/物理成功。
- 验收命令：`CUDA_VISIBLE_DEVICES=5 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.dexplore_contract_audit.verify_coupled_view --config src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml --run-id coupled_fk_gate_20260912_224800 --device cuda --batch-size 32`。
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800](../../research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/run_manifest.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/run_manifest.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/verification.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/verification.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/metrics.jsonl](../../research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/metrics.jsonl)。
- `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py src/task/CmDecoderv2/tests/test_kinematics.py src/task/CmDecoderv2/tests/test_pointflow.py src/task/CmDecoderv2/tests/test_dataset.py`：15 passed；`git diff --check`通过。
- run_id: `cm_decoder_v2_coupled_batch8_smoke_20260912_224424`；run_status: FAILED；22:44:27初始化失败，last_step=0/last_epoch=0，best_metric/checkpoint均不存在。Runner要求metadata版本V1.1.13等于训练V1.1.14，未加载模型或进入训练；下一步用既有builder生成同内容V1.1.14轻量view再核验，不绕过Runner合同。
- command: `CUDA_VISIBLE_DEVICES=2,4,5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --distributed --config src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml --set name=cm_decoder_v2_coupled_batch8_smoke --set train.epochs=1 --set train.max_steps=200 --set train.log_every_steps=50`
- [outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_224424](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_224424/)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_224424/run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_batch8_smoke_20260912_224424/run_manifest.json)。初始化在BaseRunner写快照前失败，无config.json/metrics.jsonl/train.log；manifest由Agent依据终端traceback补录，保留该事实，不伪称Runner产物。
- 计划/配置：[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)、[src/task/CmDecoderv2/configs/active/coupled_geometric_v1_batch8.yaml](../../configs/active/coupled_geometric_v1_batch8.yaml)。回滚为隔离新增配置/诊断/输出，旧文件与权重不删除。

## 2026-09-12 22:01:43 +0800 — V1.1.13 扩大 batch 的三卡吞吐验证

- timestamp: 2026-09-12 22:01:43 +0800
- activity_id: ACT-20260912-220143-BATCH-SCALE
- modification_version: V1.1.13
- type: operation / experiment / diagnostic
- change_level: L3
- approval: user-approved
- approval_basis: 用户要求扩大训练 batch。
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- scope: 只改变本次训练运行的 `data.batch_size`/`val_batch_size`，不改模型、数据、cache、split或旧运行；使用冻结 OICM v1.3 SHA `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。
- run_id: `cm_decoder_v2_20260912_215725`; run_status: COMPLETED；三卡`CUDA_VISIBLE_DEVICES=0,1,6`，每卡batch=8、global batch=24，20 step+validation，`00:51`；输出：[batch8 run](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_215725/)、[metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_215725/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_215725/train.log)、[checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_215725/checkpoints/latest.pt)。
- 对照 run_id: `cm_decoder_v2_20260912_215917`; run_status: COMPLETED；每卡batch=4、global batch=12，20 step+validation，`00:49`；输出：[batch4 run](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_215917/)。batch=8 没有显存问题，且单位样本吞吐不低于batch=4，选为正式配置。
- 结果：batch8 val loss `0.0139416`、h1 point-flow EPE `16.6417mm`；这是工程吞吐/显存 smoke，不能视为收敛结论。按当前短程速度，直接50 epoch长训可能需要约1–2天，故本活动不启动未校准的长任务。
- **原因**：扩大 batch 后需要重新测量单步耗时；OICM的10135点流计算不一定随batch线性加速，不能仅按batch倍数推算训练时间。
- **验证**：14项Task测试此前通过；本轮两个三卡运行均完成反传、validation和checkpoint生成；无残留训练进程。
- files: `docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`。

## 2026-09-12 21:27:35 +0800 — V1.1.13 耦合几何参考验收与三卡阶段一短程训练完成

- timestamp: 2026-09-12 21:27:35 +0800
- activity_id: ACT-20260912-212735-COUPLED-GEOMETRIC-TRAIN
- modification_version: V1.1.13
- type: data / code / experiment / operation / diagnostic
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求保留6维控制、按RL耦合规则重定向几何、实际状态保留完整12个手指关节后再训练。
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- skills_used: research-change-control, research-experiment-workflow
- scope: 新增耦合几何拟合、native观测FK、source/cache/view新目录和阶段一三卡运行；原始geometric/actual tensor、旧cache、旧checkpoint、split、外部dexplore/IsaacGymEnvs和共享src/base均未覆盖。
- run_id: `coupled_geometric_v1_full_20260912_194544`; run_status: COMPLETED；660条/166337帧；5021.06秒；输出：[fitted run_manifest](../../../../../data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/run_manifest.json)、[summary](../../../../../data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/summary.json)、[independent validation](../../../../../data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/independent_validation.json)。
- 适配验收：非q列和腕部逐位不变；fitted mimic最大残差`5.96e-8 rad`；指尖适配前均值35.1269mm、适配后18.9055mm、p95 34.3876mm；新参考表面残差均值1.6071mm、p95 2.7154mm。该结论是`SUPPORTED`（数据/接口修复），不是Cm效果结论。
- run_id: `oicm-dexplore-rl-v1-3-...`（3 worker）/finalize；run_status: COMPLETED；285条、72935帧，train255/65008帧、val30/7927帧；三卡KNN和sequence validation通过。输出：[coupled cache run_manifest](../../../../../data/processed_data/coupled_geometric_cache_v1_20260912/run_manifest.json)、[cache index](../../../../../data/processed_data/coupled_geometric_cache_v1_20260912/index.json)、[view run_manifest](../../../../../data/processed_data/cm_decoder_v2/coupled_geometric_v1_20260912/run_manifest.json)。
- 三卡阶段一 smoke：首次脚本路径启动失败（`ModuleNotFoundError: src`，无产物）；模块启动第一次被checkpoint SHA锁定合同拒绝（无训练）；随后 `CUDA_VISIBLE_DEVICES=1,3,6` 三卡10 step+validation完成，输出：[smoke run](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_212342/)。随后以`CUDA_VISIBLE_DEVICES=0,3,6`完成100 step短程训练，输出：[100-step run](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_212609/)，[metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_212609/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_212609/train.log)、[latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_212609/checkpoints/latest.pt)已生成。
- 100-step验证：val loss/point-flow `0.0141249`，val hand point-flow EPE `36.391mm`（h1 `16.6984mm`）；这是工程 smoke/短程优化证据，科研结论仍为`INCONCLUSIVE`，没有证明decoder收敛、跨手能力或物理任务成功，也没有启动第二阶段MANO→actual微调和RL rollout。
- 保护边界：未修改decoder输出为12维；未把实际12维状态压成6维；未启动正式50 epoch长训；未修改任何旧输入/输出。回滚入口为隔离本次新增目录和代码，旧路径保持可用。
- files: `docs/current_versions.yaml`; `src/task/CmDecoderv2/dataset.py`; `src/task/CmDecoderv2/kinematics.py`; `src/task/CmDecoderv2/pointflow.py`; `src/task/CmDecoderv2/docs/plan/v1.1.md`; `src/task/CmDecoderv2/docs/logs/activity_log.md`; `src/task/CmDecoderv2/docs/logs/experiment_log.md`; `src/task/CmDecoderv2/research/dexplore_contract_audit/`; `src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`; `src/task/CmDecoderv2/tests/test_pointflow.py`; `src/task/CmDecoderv2/tools/data/fit_coupled_geometric_retarget.py`; `src/task/CmDecoderv2/tools/data/prepare_coupled_geometric_source.py`; `src/task/CmDecoderv2/tools/data/merge_coupled_view_index.py`。

**原因**

原始几何轨迹的12维手指状态与RL执行的6维独立控制接口不一致；直接沿用旧点云会把参数化误差混入decoder学习。新流程固定腕部，用6维驱动量展开后优化指尖/表面目标，并把物体点按新参考物体位姿重放。

**验证**

14项Task定向测试通过；660条输出独立 shape/finite/列保护/耦合残差验收通过；285条缓存 finalize 和 view metadata 通过；三卡10 step和100 step工程运行完成。上述训练只属于工程 smoke/短程证据，阶段一效果、阶段二微调和RL任务结论仍为INCONCLUSIVE。

## 2026-09-12 19:09:54 +0800 — V1.1.13 配对和FK核验完成，正式训练在状态合同闸门暂停

- timestamp: 2026-09-12 19:09:54 +0800
- activity_id: ACT-20260912-190954-DEXPLORE-CONTRACT-END
- modification_version: V1.1.13
- type: code / diagnostic / experiment / operation / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认两阶段解码训练、三卡和reference-conditioned RL适配范围；先完成必要输入检查，发现GT/状态不匹配后按闸门暂停，不擅自改变科研合同。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a
- worktree_dirty: true
- scope: 本Task新增合同审计和专属测试、计划/活动/实验与指针；未修改核心模型、GT、正式split/cache、旧checkpoint、共享src/base、外部dexplore/IsaacGymEnvs、环境依赖及其他进程。
- run_id: contract_full_20260912_191500
- run_status: COMPLETED
- actual_run_time: 2026-09-12 19:05:19 +0800启动，134.1109秒；独立验证在19:09:54前完成。
- last_step: 660条/166337帧；每源5280个抽样FK状态；两源共10560。
- last_epoch: not_applicable
- best_metric: not_applicable（无训练/模型选择）；best_checkpoint/recent_checkpoint: not_applicable；未生成train.log。
- conclusion: SUPPORTED（原始状态与6维固定耦合表示不等价）；REFUTED（几何重定向已经符合现有decoder耦合约束）；INCONCLUSIVE（两阶段decoder效果、Cm跨手能力及物理任务成功）。
- 未完成的用户目标: 正式paired训练view、两阶段三卡训练、RL任务/策略适配及物理rollout均未实施或启动。需要先确认GT/状态/动作合同调整；这不是训练失败或环境不兼容。

**文件**

- [src/task/CmDecoderv2/research/dexplore_contract_audit](../../research/dexplore_contract_audit/) — 新`__init__.py`、`contracts.py`、`run.py`、`verify.py`、`asset_check.py`、`README.md`、`experiment.yaml`。
- [src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py](../../tests/test_dexplore_contract_audit.py) — 约束、最小二乘、限位、逐位字段保护、异常shape/NaN和split身份测试。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — V1.1.13 final输入闸门和暂停条件。
- [src/task/CmDecoderv2/docs/logs/experiment_log.md](experiment_log.md) — 协议、全量/抽样分母、结果、局限及待批准调整。
- [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 唯一运行时间线及本Task指针。

**原因**

全部660对的shape/finite/未替换列通过；canonical human/object身份和帧数一致，但MANO mesh坐标尚未核验，不能把它写成完整对齐结论。
630条既有split保留（509/58/63），其余30条仅在诊断manifest标unassigned，不产生正式训练划分。
几何从动关节耦合偏差均值68.7413deg，100%帧至少一个从动关节偏差>1deg；actual均值7.9929deg、对应比例76.8933%。
保留原6维并mimic重建的抽样指尖EPE几何35.1815mm/actual5.0975mm，全手10135点均值分别2.3702/0.2674mm。
误差不是最优几何下界；本轮不能推断Cm差的主因，也不能推断几何预训练绝无价值。
建议下一轮重新优化受现有6维控制约束的几何目标，并区分12维实际手指观测与6维控制动作；涉及GT和输入合同，需要确认后实施，不将12个状态关节误当12个独立执行器。

**验证**

- `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py src/task/CmDecoderv2/tests/test_kinematics.py src/task/CmDecoderv2/tests/test_pointflow.py`：10 passed。
- 正式命令：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.dexplore_contract_audit.run --run-id contract_full_20260912_191500`。
- 独立核验：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.dexplore_contract_audit.verify src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500`：通过；原q身份、10560状态FK、独立lstsq、summary及所有保护输入SHA/stat通过。
- 独立surface EPE最大差1.421e-6mm，tip EPE最大差1.774e-13mm；16个torch/NumPy表面样本逐点最大差0.000140mm。
- 资产检查：`CUDA_VISIBLE_DEVICES=5 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmDecoderv2/research/dexplore_contract_audit/asset_check.py --urdf src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf --output src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/gym_asset_check.json`：通过，native18关节名称/顺序一致，physics_steps=0。第一次`-m`失败详见START事件；属于Task提前加载torch，不是conda不兼容。
- 两条smoke run_id `contract_smoke_20260912_190500`、run_status `COMPLETED`，实际19:03:07 +0800启动、1.227秒，详见START活动及其manifest/verification；仅工程证据。
- 产物3.9MiB，均被Git忽略。审计与资产检查进程均已退出。未提交代码。

**产物**

- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/run_manifest.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/run_manifest.json)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/config.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/config.json)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/metrics.jsonl](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/metrics.jsonl)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/run.log](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/run.log)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/summary.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/summary.json)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/paired_manifest.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/paired_manifest.json)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/state_samples.npz](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/state_samples.npz)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/verification.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/verification.json)
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/gym_asset_check.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/gym_asset_check.json)

**回滚与规范反馈**

仅隔离新增审计/测试与本轮文档即可回滚，不需要迁移、删除或重算任何旧数据/权重。
本轮触发科研GT/状态合同的正常审批闸门；无目录、格式或版本规则阻碍，不修改治理规则。
`git diff --check`通过；`audit_diff.py --log src/task/CmDecoderv2/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmDecoderv2 --scope-prefix docs/current_versions.yaml --check-links`通过，11个变更路径、16个本地链接。审计要求的原因段标题已统一为`原因`，不改变内容。

## 2026-09-12 19:05:29 +0800 — V1.1.13 两阶段训练前合同审计启动

- timestamp: 2026-09-12 19:05:29 +0800
- activity_id: ACT-20260912-190529-DEXPLORE-CONTRACT-START
- modification_version: V1.1.13
- type: code / diagnostic / operation / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户在两阶段训练、三卡与reference-conditioned RL适配范围说明后回复“可以，你继续吧”。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 5cf7eaa
- worktree_dirty: true
- scope: 本Task训练前输入合同闸门；不改旧模型/GT/split/cache、其他Task、共享src/base、外部dexplore/IsaacGymEnvs、环境依赖与运行进程。
- run_id: contract_full_20260912_191500
- run_status: RUNNING
- command: `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.dexplore_contract_audit.run --run-id contract_full_20260912_191500`

**文件**

- [src/task/CmDecoderv2/research/dexplore_contract_audit](../../research/dexplore_contract_audit/) — 配对、状态耦合、10135点完整FK及独立核验。
- [src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py](../../tests/test_dexplore_contract_audit.py) — 输入与约束测试。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — 已批准闸门、指针和本记录。

**原因**

几何导出器明确`ignore_mimic_joint=True`；现有decoder保留6个finger q并强制重建从动关节，可能不能表示原始几何/实际状态。需先核验再长训，不自动投影GT或开放12维控制。

**验证与产物**

- `python -m pytest -q src/task/CmDecoderv2/tests/test_dexplore_contract_audit.py`（graspenv、CPU）：5 passed。
- run_id `contract_smoke_20260912_190500`，run_status `COMPLETED`；两条/385帧/32个采样状态通过，独立NumPy FK/torch点最大差0.000144mm以内，仅工程smoke。
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_smoke_20260912_190500](../../research/dexplore_contract_audit/output/contract_smoke_20260912_190500/)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_smoke_20260912_190500/run_manifest.json](../../research/dexplore_contract_audit/output/contract_smoke_20260912_190500/run_manifest.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_smoke_20260912_190500/verification.json](../../research/dexplore_contract_audit/output/contract_smoke_20260912_190500/verification.json)。
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/run_manifest.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/run_manifest.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/run.log](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/run.log)。
- 可选资产检查第一次`-m`入口失败：Task包提前import torch触发Isaac Gym导入顺序限制，未创建仿真；使用独立文件入口重试，不修改Task公共初始化。
- 回滚：仅隔离新审计/测试和本轮文档增补。科研与可训练性结论等待全量核验。

## 2026-09-12 15:53:16 +0800 — V1.1.12 轨迹质量标记与候选交叉检查完成

- activity_id: ACT-20260912-155316-QUALITY-END
- timestamp: 2026-09-12 15:53:16 +0800
- modification_version: V1.1.12
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2
- approval: user-approved
- approval_basis: 用户在质量准入与跨手正对建议后回复“继续”，落实独立诊断规则与候选检查。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 新trajectory_quality_gate、专属测试、Task计划/日志和指针；不修改正式cache/GT/split、模型、旧研究/输出、外部dexplore、其他Task或用户删除的AGENTS.md。
- run_id: quality_val_20260912_155200
- run_status: COMPLETED
- actual_run_time: 2026-09-12 15:51:22–15:51:28 +0800，6.06秒。
- last_step: 4254窗口；3017个有效运动窗口；240条分层指标；178条含条件/类别重复的候选清单行。
- last_epoch: not_applicable；best_metric: not_applicable；recent_checkpoint: not_applicable（无模型加载、训练或仿真）。
- conclusion: SUPPORTED（参考跟踪标记与实际跨手几何兼容选出不同样本）；INCONCLUSIVE（正式质量准入或物理成功标准）。

**文件**

- [src/task/CmDecoderv2/research/trajectory_quality_gate](../../research/trajectory_quality_gate/) — run.py、gates.py、verify.py、README.md、experiment.yaml、__init__.py，复用质量标记、CSV检查队列、带raw/cache帧的候选清单与独立核验。
- [src/task/CmDecoderv2/tests/test_trajectory_quality_gate.py](../../tests/test_trajectory_quality_gate.py) — 四步/五帧、阈值边界、独立标记、原因位和无效输入测试。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — V1.1.12 final；[src/task/CmDecoderv2/docs/logs/experiment_log.md](experiment_log.md) — 固定阈值对照、结果与路线调整；[src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — 运行与审计记录。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 仅本Task指针推进V1.1.12。
- 累计未提交差异还含此前的 [src/task/CmDecoderv2/research/cm_condition_dependence](../../research/cm_condition_dependence/)、[src/task/CmDecoderv2/research/cross_hand_cm_swap](../../research/cross_hand_cm_swap/)、[src/task/CmDecoderv2/research/cross_hand_pair_coverage](../../research/cross_hand_pair_coverage/)、[src/task/CmDecoderv2/research/object_tracking_audit](../../research/object_tracking_audit/)，以及 [src/task/CmDecoderv2/tests/test_cm_condition_dependence.py](../../tests/test_cm_condition_dependence.py)、[src/task/CmDecoderv2/tests/test_cross_hand_cm_swap.py](../../tests/test_cross_hand_cm_swap.py)、[src/task/CmDecoderv2/tests/test_cross_hand_pair_coverage.py](../../tests/test_cross_hand_pair_coverage.py)、[src/task/CmDecoderv2/tests/test_object_tracking_audit.py](../../tests/test_object_tracking_audit.py)；本轮只读消费，未修改。

**原因**

将参考跟踪检查做成可复用诊断，同时检验其能否直接作为跨手正对准入条件。既有53个兼容运动窗口中40个未达原参考effect门槛、53个均未达20mm/15deg五帧pose门槛；不能把这两类目标混同。
effect与跨手兼容交集13窗口/4parent/5不重叠贪心对；combined20为0；combined40为8/2/2，combined80为12/4/4。所有阈值运行前声明，没有后验择优放宽。
原53/10/18基线完整保留，新增清单diagnostic_only=true、split=val；跟踪质量不自动应用于正式训练过滤。

**验证**

- 测试：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_trajectory_quality_gate.py`：4 passed。
- 正式命令：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.trajectory_quality_gate.run --run-id quality_val_20260912_155200`。
- 独立核验：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.trajectory_quality_gate.verify src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200`：通过；4254窗口标记、178候选行，效果比值最大差4.78e-13；源/目标身份、原始候选成员资格、最佳供体和双流不重叠区间通过。
- 旧输入/代码/产物SHA与stat不变；CPU6.06秒、输出约2.33MiB；output按现有规则忽略，`git diff --check`通过，无暂存提交。
- `audit_diff.py --log src/task/CmDecoderv2/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmDecoderv2 --scope-prefix docs/current_versions.yaml --check-links`：通过，38个累计变更路径和24个本地链接可导航；此前差异明确只读保留。
- 可回滚为隔离新诊断目录/测试及本轮文档增补；不影响旧运行或正式数据。无额外确认或规范阻碍。

**产物**

- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/run_manifest.json](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/run_manifest.json)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/config.json](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/config.json)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/window_metrics.jsonl](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/window_metrics.jsonl)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/metrics.jsonl](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/metrics.jsonl)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/candidate_manifest.jsonl](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/candidate_manifest.jsonl)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/repair_queue.csv](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/repair_queue.csv)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/summary.json](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/summary.json)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/run.log](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/run.log)
- [src/task/CmDecoderv2/research/trajectory_quality_gate/output/quality_val_20260912_155200/independent_verification.json](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/independent_verification.json)

## 2026-09-12 15:45:43 +0800 — V1.1.11 物体轨迹来源与跟踪审计完成

- activity_id: ACT-20260912-154543-TRACKING-END
- timestamp: 2026-09-12 15:45:43 +0800
- modification_version: V1.1.11
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2
- approval: user-approved
- approval_basis: 用户在改善两手共同物体动作轨迹对应的建议后回复“继续”。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 本Task新object_tracking_audit、专属测试、计划/日志与指针；dexplore源码/输入只读；核心模型、正式cache/GT/split、旧实验代码/产物/权重、其他Task及用户删除的AGENTS.md保持不变。
- run_id: tracking_val_20260912_154240
- run_status: COMPLETED
- actual_run_time: 2026-09-12 15:42:29–15:42:53 +0800，23.95秒。
- last_step: 7927帧/30 parent；2983去重运动transition；1498个后半段lag验证transition。
- last_epoch: not_applicable；best_metric: not_applicable；recent_checkpoint: not_applicable（无decoder加载、训练或仿真）。
- conclusion: SUPPORTED（现有参考→RL实际轨迹存在明显跟踪差异）；REFUTED（本协议固定lag改善后半效果误差）；INCONCLUSIVE（Cm总体跨手能力及具体物理失效根因）。

**文件**

- [src/task/CmDecoderv2/research/object_tracking_audit](../../research/object_tracking_audit/) — 新run.py、diagnostics.py、verify.py、README.md、experiment.yaml、__init__.py；三段轨迹来源、native跟踪与同支持分半lag审计。
- [src/task/CmDecoderv2/tests/test_object_tracking_audit.py](../../tests/test_object_tracking_audit.py) — 四元数约定、固定偏移、lag符号/支持和缺失处理。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — V1.1.11 final范围。
- [src/task/CmDecoderv2/docs/logs/experiment_log.md](experiment_log.md) — 来源链、指标分母、定量结果和限制。
- [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — 运行与复核终态。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 仅本Task指针推进V1.1.11。
- 工作区累计未提交差异还包含此前的 [src/task/CmDecoderv2/research/cm_condition_dependence](../../research/cm_condition_dependence/)、[src/task/CmDecoderv2/research/cross_hand_cm_swap](../../research/cross_hand_cm_swap/)、[src/task/CmDecoderv2/research/cross_hand_pair_coverage](../../research/cross_hand_pair_coverage/)、[src/task/CmDecoderv2/tests/test_cm_condition_dependence.py](../../tests/test_cm_condition_dependence.py)、[src/task/CmDecoderv2/tests/test_cross_hand_cm_swap.py](../../tests/test_cross_hand_cm_swap.py)、[src/task/CmDecoderv2/tests/test_cross_hand_pair_coverage.py](../../tests/test_cross_hand_pair_coverage.py)；本轮未改写，旧输入摘要保持不变。

**原因**

上一轮仅53个严格运动接收窗口，需定位物体效果差来源。native参考→实际的中心位置误差mean123.91mm/median41.50mm，旋转mean53.54deg；初始帧误差最大仅0.00192mm，偏离发生于执行过程中。
2983个去重运动transition上，parent→几何参考effect EPE0.4801mm，几何参考→实际14.7682mm；分段EPE不作可加分解。
前半选固定lag、后半验证的EPE由14.9919升至15.8312mm，增加0.8393mm，parent聚类CI95=[0.1303,1.8224]。
上游显式取逆旋转与重算骨盆/地面偏移有源码依据；仅核验合同，不自动修改转置或把诊断拟合写成新GT。

**验证**

- 测试：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_object_tracking_audit.py`：4 passed。
- 正式命令：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.object_tracking_audit.run --run-id tracking_val_20260912_154240`。
- 独立核验：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.object_tracking_audit.verify src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240`：通过；7927帧旋转测地距离最大差2.96e-6deg，1080项独立pointflow最大差4.55e-13mm，lag选择与heldout复现。
- 全部参考未替换字段逐元素不变、缓存actual pose回放通过、raw/time合同通过；新tensor、外部源码、旧代码/产物摘要与input stat不变。
- CPU运行23.95秒，输出2.78MiB，output按现有规则忽略；`git diff --check`通过，无暂存提交。
- `audit_diff.py --log src/task/CmDecoderv2/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmDecoderv2 --scope-prefix docs/current_versions.yaml --check-links`：通过，31个累计变更路径、21个本地链接可导航；前阶段差异明确只读保留。
- smoke run_id=tracking_smoke_20260912_154300，run_status=COMPLETED，实际15:41:43–15:41:50，1096帧、6.21秒，仅工程证据。
- 解释边界：历史export未提供实际模拟器逐帧时间戳，本轮未锁定当时controlFrequencyInv；未仿真验证上游逆旋转对物理资产的合理性。结果不等于物理任务成功率。
- 回滚仅隔离新研究目录、专属测试和本轮文档；无额外确认或规范阻碍。

**产物**

- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240](../../research/object_tracking_audit/output/tracking_val_20260912_154240/)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/run_manifest.json](../../research/object_tracking_audit/output/tracking_val_20260912_154240/run_manifest.json)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/config.json](../../research/object_tracking_audit/output/tracking_val_20260912_154240/config.json)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/metadata.json](../../research/object_tracking_audit/output/tracking_val_20260912_154240/metadata.json)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/metrics.jsonl](../../research/object_tracking_audit/output/tracking_val_20260912_154240/metrics.jsonl)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/run.log](../../research/object_tracking_audit/output/tracking_val_20260912_154240/run.log)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/summary.json](../../research/object_tracking_audit/output/tracking_val_20260912_154240/summary.json)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/tracking.png](../../research/object_tracking_audit/output/tracking_val_20260912_154240/tracking.png)
- [src/task/CmDecoderv2/research/object_tracking_audit/output/tracking_val_20260912_154240/independent_verification.json](../../research/object_tracking_audit/output/tracking_val_20260912_154240/independent_verification.json)

## 2026-09-12 15:42:29 +0800 — V1.1.11 全val物体轨迹审计启动

- activity_id: tracking_val_20260912_154240
- timestamp: 2026-09-12 15:42:29 +0800
- modification_version: V1.1.11
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2
- approval: user-approved
- approval_basis: 用户在改善两手共同物体动作轨迹对应的建议后回复“继续”。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 新物体轨迹来源/跟踪诊断及定向测试、Task计划/日志/指针；外部dexplore只读；不改正式数据或模型。
- run_id: tracking_val_20260912_154240
- run_status: STARTED
- command: `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.object_tracking_audit.run --run-id tracking_val_20260912_154240`
- output: [run_manifest.json](../../research/object_tracking_audit/output/tracking_val_20260912_154240/run_manifest.json)

**文件**

- [src/task/CmDecoderv2/research/object_tracking_audit](../../research/object_tracking_audit/) — native参考/实际跟踪、parent到几何参考偏移、既有cache合同effect和分半lag验证。
- [src/task/CmDecoderv2/tests/test_object_tracking_audit.py](../../tests/test_object_tracking_audit.py) — 4项四元数、固定偏移、lag符号/支持及空支持测试。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — final V1.1.11；[docs/current_versions.yaml](../../../../../docs/current_versions.yaml) 仅本Task推进。

**原因**

追踪低配对覆盖的数据生成来源。上游prepare_grab显式取逆旋转，convert_grab重算骨盆/地面偏移，RL导出只替换实际物体pose和native q。先核验合同再区分跟踪误差，不能单凭转置就修改缓存。

**验证**

4项定向测试通过；smoke run_id=tracking_smoke_20260912_154300，run_status=COMPLETED，实际15:41:43–15:41:50，2序列1096帧，6.21秒；只作工程证据。完整运行保留原protocol，新增独立核验入口。回滚仅新目录、测试和本轮文档。

## 2026-09-12 15:33:30 +0800 — V1.1.10 全 val 时间偏移配对覆盖完成

- activity_id: ACT-20260912-153330-PAIR-COVERAGE-END
- timestamp: 2026-09-12 15:33:30 +0800
- modification_version: V1.1.10
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2（独立配对诊断）；单卡限时运行按 L3
- approval: user-approved
- approval_basis: 用户在 V1.1.9 结果与补足同效果/接触配对的建议后回复“继续”。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 本Task新 cross_hand_pair_coverage、专属测试、计划/日志和版本指针；核心模型/loader/训练变量、正式数据/GT/cache/split、旧研究代码/产物/权重、其他Task和用户删除的AGENTS.md保持不变。
- run_id: pair_coverage_val_20260912_153150
- run_status: COMPLETED
- actual_run_time: 2026-09-12 15:30:52–15:31:06 +0800，13.88秒。
- last_step: 4254 active-only候选窗口；30 parent；420条逐parent/policy/group指标。
- last_epoch: not_applicable；best_metric: not_applicable；recent_checkpoint: not_applicable（没有加载decoder或训练；只继承V1.1.9冻结Cm有效mask）。
- conclusion: REFUTED（仅时间偏移达到100严格运动窗口/10 parent的当前库充足性命题）；INCONCLUSIVE（总体跨手迁移，未运行decoder）。

**文件**

- [src/task/CmDecoderv2/research/cross_hand_pair_coverage](../../research/cross_hand_pair_coverage/) — 新增run.py、matching.py、verify.py、README.md、experiment.yaml、__init__.py；精确效果配对、全parent时间搜索、供体复用/不重叠统计与独立穷举复核。
- [src/task/CmDecoderv2/tests/test_cross_hand_pair_coverage.py](../../tests/test_cross_hand_pair_coverage.py) — 5项定向合同验证。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — V1.1.10 final执行增补。
- [src/task/CmDecoderv2/docs/logs/experiment_log.md](experiment_log.md) — 分层覆盖、受限命题结论与下一步依据。
- [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — smoke、全量运行与验证终态。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 仅本Task指针推进V1.1.10；其他已有差异保留。
- 工作区累计差异还含此前的 [src/task/CmDecoderv2/research/cm_condition_dependence](../../research/cm_condition_dependence/)、[src/task/CmDecoderv2/research/cross_hand_cm_swap](../../research/cross_hand_cm_swap/)、[src/task/CmDecoderv2/tests/test_cm_condition_dependence.py](../../tests/test_cm_condition_dependence.py)、[src/task/CmDecoderv2/tests/test_cross_hand_cm_swap.py](../../tests/test_cross_hand_cm_swap.py)；本轮只读消费，未重新修改。

**原因**

同步严格运动只有16窗口/4 parent，固定效果与接触门槛后放开时间偏移，检查现有数据能否补足有效跨手配对。
任意偏移达到53窗口/10 parent，仅覆盖3017个有效运动接收窗口的1.76%；43个唯一选中源、最大一对一容量48、保守提取18个双流不重叠配对。
只按效果也仅76个运动窗口，不足100。20096条严格边中只有90条属于运动接收，91.1%全部边来自一个无严格运动匹配的torussmall序列；不能据总边数扩大训练正对规模。

**验证**

- 测试：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_cross_hand_pair_coverage.py`：5 passed。
- 正式命令：`CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_pair_coverage.run --run-id pair_coverage_val_20260912_153150`。
- 独立核验：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_pair_coverage.verify src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150`：通过，1749个局部网格候选穷举一致，描述最大差1.33e-5；全部保存边门槛、选中供体排序、parent/有效性、不重叠区间和summary通过。
- V1.1.9同步effect/strict对角线逐窗口精确重现，严格85/运动16不变；描述最大差7.44e-6；旧输入/代码/产物SHA256及stat不变。
- GPU3 peak allocated199.9 MiB，输出5.20 MiB；output按现有规则忽略。`git diff --check`通过；无暂存提交。
- `audit_diff.py --log src/task/CmDecoderv2/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmDecoderv2 --scope-prefix docs/current_versions.yaml --check-links`：通过，24个累计变更路径、19个本地链接可导航；前阶段差异已明确标为只读保留。
- smoke run_id=pair_coverage_smoke_20260912_153000，run_status=COMPLETED，实际15:29:49–15:29:54；882窗口，只作工程证据。正式运行前补充verify并修正manifest last_step为active-only实际数，科学协议未变。
- 回滚：隔离新研究目录、专属测试与本轮文档增补，不影响旧运行或正式数据；无需要额外确认的规范阻碍。

**产物**

- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/run_manifest.json](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/run_manifest.json)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/config.json](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/config.json)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/metrics.jsonl](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/metrics.jsonl)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/run.log](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/run.log)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/summary.json](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/summary.json)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/selected_pairs.jsonl](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/selected_pairs.jsonl)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/coverage.png](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/coverage.png)
- [src/task/CmDecoderv2/research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/independent_verification.json](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/independent_verification.json)

## 2026-09-12 15:29:49 +0800 — V1.1.10 时间偏移配对覆盖 smoke 启动

- activity_id: pair_coverage_smoke_20260912_153000
- timestamp: 2026-09-12 15:29:49 +0800
- modification_version: V1.1.10
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2；单卡限时运行按 L3
- approval: user-approved
- approval_basis: 用户在 V1.1.9 结果与补齐同效果跨手配对的建议后回复“继续”。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 新跨时间配对诊断、专属测试、计划/日志及本Task版本指针；正式数据/模型/旧产物保持不变。
- run_id: pair_coverage_smoke_20260912_153000
- run_status: STARTED
- command: `CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_pair_coverage.run --run-id pair_coverage_smoke_20260912_153000 --smoke`
- output: [run_manifest.json](../../research/cross_hand_pair_coverage/output/pair_coverage_smoke_20260912_153000/run_manifest.json)

**文件**

- [src/task/CmDecoderv2/research/cross_hand_pair_coverage](../../research/cross_hand_pair_coverage/) — 固定逐步效果/接触门槛，全同parent候选搜索与供体复用/不重叠统计。
- [src/task/CmDecoderv2/tests/test_cross_hand_pair_coverage.py](../../tests/test_cross_hand_pair_coverage.py) — 必要条件预筛对穷举、有效性、接触、复用和共享端点测试。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — final V1.1.10。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 本Task指针V1.1.10。

**原因**

保持原始 MANO 轨迹与匹配门槛，检查只允许源时间偏移能否补足严格运动配对；不会按decoder结果或目标未来q/wrist挑选。

**验证**

`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_cross_hand_pair_coverage.py`：5 passed。smoke 首先复现 V1.1.9 对角线描述/覆盖；只作工程验证。回滚隔离新目录、测试和本轮文档增补。

## 2026-09-12 15:23:08 +0800 — V1.1.9 全 val 跨手交换完成与独立复核

- activity_id: ACT-20260912-152308-CROSS-HAND-END
- timestamp: 2026-09-12 15:23:08 +0800
- modification_version: V1.1.9
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2（跨手离线诊断）；限时单卡运行按 L3
- approval: user-approved
- approval_basis: 用户在已说明的跨手动作表征路线后连续回复“继续”，执行已定稿 V1.1.9。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 本 Task 新 cross_hand_cm_swap、专属测试、计划/日志和既有 CmDecoderv2 版本指针；核心模型/loader/runner、训练变量、GT/cache/split、旧输出/checkpoint、其他 Task 和用户删除的 AGENTS.md 无本轮写入。
- run_id: cross_hand_val_20260912_151850
- run_status: COMPLETED
- actual_run_time: 2026-09-12 15:18:08 至 15:21:28 +0800；200.53 秒。
- last_step: 4254 窗口；21270 条单步 metrics；30 条 val。
- last_epoch: not_applicable（无训练；固定 decoder epoch5、step7240）。
- best_metric: not_applicable（不作 checkpoint 选择；严格运动 mano_sync EPE=10.8513 mm）。
- recent_checkpoint: [outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)
- conclusion: INCONCLUSIVE（总体跨手迁移；严格运动只有16窗口/4 parent，低于100/10预设门槛）。

**文件**

- [src/task/CmDecoderv2/research/cross_hand_cm_swap](../../research/cross_hand_cm_swap/) — 新增 run.py、diagnostics.py、verify.py、README.md、experiment.yaml、__init__.py；只读重建原始 MANO/transported，固定接收手状态交换完整 Cm。
- [src/task/CmDecoderv2/tests/test_cross_hand_cm_swap.py](../../tests/test_cross_hand_cm_swap.py) — 5项坐标/单位、空接触、全窗口门槛、供体与 parent 统计测试。
- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — V1.1.9 final 增补，明确实际 pose 的 `198:205` 是 tensor slice。
- [src/task/CmDecoderv2/docs/logs/experiment_log.md](experiment_log.md) — 记录完整协议、分层覆盖、配对指标、局限和下一步依据。
- [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — smoke、全量运行与核验终态。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 本 Task 指针 V1.1.9，保留其他作用域已有差异。
- 工作区累计差异还含上一阶段的 [src/task/CmDecoderv2/research/cm_condition_dependence](../../research/cm_condition_dependence/) 和 [src/task/CmDecoderv2/tests/test_cm_condition_dependence.py](../../tests/test_cm_condition_dependence.py)；本次只读消费，前后摘要一致，未重新改写。

**原因**

将已确认的同源条件依赖推进到跨手。两源有效3665/30，效果匹配176/17，严格接触匹配85/9，严格运动16/4。
同时间戳不保证物体效果等价，有效窗口平均效果差中位数8.0708 mm。严格运动上 correct/identity/原始MANO/错时MANO/oracle EPE为8.7499/46.7284/10.8513/54.0171/9.0612 mm，但样本过少且集中，不能据此推广迁移结论。
下一步优先检查真实同效果跨手配对覆盖；oracle 注入目标物体运动，不能单独作为成功证据。

**验证**

- 定向测试：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_cross_hand_cm_swap.py`，5 passed。
- 完整命令：`CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_cm_swap.run --run-id cross_hand_val_20260912_151850`。
- 独立复核：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_cm_swap.verify src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850`，通过；150个 NumPy FK 样本手点EPE/输出变化最大差1.68e-5 mm；donor、summary、覆盖可重现。
- 正式loader窗口回放差0，官方val MANO表面回放差0，canonical object点最大差2.46e-7 m，correct q/wrist回放差1.19e-7；模型参数/buffer、受保护的旧代码/输出/checkpoint SHA256与所有输入stat保持不变。
- GPU3 peak allocated5459.5 MiB，输出75.9 MiB；无新 checkpoint，未启动训练。`git diff --check` 通过，output 被现有规则忽略。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/CmDecoderv2/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmDecoderv2 --scope-prefix docs/current_versions.yaml --check-links`：通过，17个变更路径与18个本地链接一致；包含前阶段累计差异的只读说明。
- smoke run_id=cross_hand_smoke_20260912_151800，run_status=COMPLETED，实际15:16:33–15:17:33；仅工程证据，完整运行前补充了输入脚本保护和独立复核入口。
- 回滚：隔离 cross_hand_cm_swap 新目录、专属测试和本轮文档增补；保留旧实验和所有正式数据。本轮未遇到需要额外确认的规范阻碍。

**产物**

- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/run_manifest.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/run_manifest.json)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/config.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/config.json)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/metadata.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/metadata.json)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/metrics.jsonl](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/metrics.jsonl)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/run.log](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/run.log)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/summary.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/summary.json)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/comparison.png](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/comparison.png)
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/independent_verification.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/independent_verification.json)

## 2026-09-12 15:18:08 +0800 — V1.1.9 全 val 跨手交换启动

- activity_id: cross_hand_val_20260912_151850
- timestamp: 2026-09-12 15:18:08 +0800
- modification_version: V1.1.9
- type: experiment / operation
- change_level: L2；限时单卡运行按 L3
- approval: user-approved
- approval_basis: 用户连续回复“继续”，完成 final 跨手诊断。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 只读加载 30 条 val 和固定 checkpoint，新 output 保存结果。
- run_id: cross_hand_val_20260912_151850
- run_status: STARTED
- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_cm_swap.run --run-id cross_hand_val_20260912_151850`
- outputs: [run_manifest.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/run_manifest.json)

**文件**

- [research/cross_hand_cm_swap](../../research/cross_hand_cm_swap/) — 增补来源脚本保护、精确起止时间和独立 NumPy FK 复核入口。

**原因**

smoke 在 15:17:33 COMPLETED，882 窗口通过所有工程断言；valid=796、effect=92、strict=53、strict_moving=1，只用于检查实现。其 run_id 中时间是标识，不作为实际启动时间。

**验证**

[smoke summary](../../research/cross_hand_cm_swap/output/cross_hand_smoke_20260912_151800/summary.json)、[smoke manifest](../../research/cross_hand_cm_swap/output/cross_hand_smoke_20260912_151800/run_manifest.json)、[smoke run.log](../../research/cross_hand_cm_swap/output/cross_hand_smoke_20260912_151800/run.log) 已生成；完整运行保持同一科学协议。

## 2026-09-12 15:16:33 +0800 — V1.1.9 跨手交换 smoke 启动

- activity_id: cross_hand_smoke_20260912_151800
- timestamp: 2026-09-12 15:16:33 +0800
- modification_version: V1.1.9
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2；限时单卡运行按 L3
- approval: user-approved
- approval_basis: 用户在已说明的冻结跨手交换路线后连续回复“继续”。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 本 Task cross_hand_cm_swap、专属测试、计划/日志及既有版本指针；旧实验、核心实现和正式 cache 不修改。
- run_id: cross_hand_smoke_20260912_151800
- run_status: STARTED
- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_cm_swap.run --run-id cross_hand_smoke_20260912_151800 --smoke`
- outputs: [run_manifest.json](../../research/cross_hand_cm_swap/output/cross_hand_smoke_20260912_151800/run_manifest.json)

**文件**

- [research/cross_hand_cm_swap](../../research/cross_hand_cm_swap/) — 原始 MANO/transported 源重建、KNN、完整 Cm 交换、分层配对与统计。
- [tests/test_cross_hand_cm_swap.py](../../tests/test_cross_hand_cm_swap.py) — 坐标、单位、空接触、全窗口筛选、供体和 parent 统计。
- [docs/plan/v1.1.md](../plan/v1.1.md) — final V1.1.9 范围；明确 `198:205` 是原始 tensor slice，非代码行号。

**原因**

同源条件依赖已完成，推进真正的同步跨手单步兼容性诊断，显式区分注入目标实际运动的 oracle 对照。

**验证**

`python -m pytest -q src/task/CmDecoderv2/tests/test_cross_hand_cm_swap.py`：5 passed。完整冻结回放与输入核验在运行内执行；smoke 不构成科学结论。回滚仅隔离新研究目录、测试与本轮文档。

## 2026-09-12 14:55:33 +0800 — V1.1.8 全 val Cm 条件依赖诊断完成

- activity_id: ACT-20260912-145016-CM-DEPENDENCE-END
- timestamp: 2026-09-12 14:55:33 +0800
- modification_version: V1.1.8
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2（冻结对照/指标）；运行预算按 L3
- approval: user-approved
- approval_basis: 用户回复“按照你的想法继续”，完成已讨论路线的首项冻结诊断。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true（其他 Task 与根级已有差异保留）
- scope: 本 Task 独立 cm_condition_dependence 实现、测试、计划/日志与 CmDecoderv2 版本指针；核心模型/loader/runner、正式训练配置、GT/cache/split、旧权重、指导/架构和其他 Task 无本轮写入。
- run_id: cm_dependence_val_20260912_145016
- run_status: COMPLETED
- last_step: 4254 active-only 窗口；17912 单步指标行；5504 递归指标行
- last_epoch: not_applicable（无训练；固定 checkpoint epoch=5、step=7240）
- best_metric: not_applicable（无 checkpoint 选择；h1 有效集 correct EPE=7.096838 mm）
- recent_checkpoint: [outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)
- conclusion: SUPPORTED（当前同源 Cm 条件依赖）；INCONCLUSIVE（跨手复用与物理闭环）。

**文件**

- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — V1.1.8 final 范围、固定对照和 native/FK 检查说明。
- [src/task/CmDecoderv2/research/cm_condition_dependence](../../research/cm_condition_dependence) — 新增实现、测试入口说明、定义、统计和独立复核；运行产物按规范忽略。
- [src/task/CmDecoderv2/tests/test_cm_condition_dependence.py](../../tests/test_cm_condition_dependence.py) — donor 条件/有效性、完整 Cm 交换、GT 隔离、配对统计、单位和递归状态更新。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmDecoderv2 从 V1.1.7 到 V1.1.8；保留 ObjectInteractionCm 原有指针差异。
- [src/task/CmDecoderv2/docs/logs/experiment_log.md](experiment_log.md) — 按命题记录科研证据及局限。
- [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — 登记失败 smoke、通过 smoke 和正式诊断终态。

**原因**

先隔离目标手是否有效使用 Cm，避免把原 decoder 忽略动作或递归误差归因于跨手。正确/identity h1 EPE
为 7.0968/13.6469 mm；严格匹配的 1302 窗口上 correct/swap 为 5.3292/24.3919 mm，惩罚 CI
[12.3278,30.5853] mm。83 起点的正确 Cm 第 16 步 EPE 为 53.1931 mm，仍有累计状态误差。
原生从动关节与固定 mimic 的差异已单独量化；没有改 GT 或据此重训。

**验证**

- `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_cm_condition_dependence.py src/task/CmDecoderv2/tests/test_model.py src/task/CmDecoderv2/tests/test_kinematics.py`：13 passed。
- 正式命令：`CUDA_VISIBLE_DEVICES=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_condition_dependence.run --run-id cm_dependence_val_20260912_145016 --activity-id ACT-20260912-145016-CM-DEPENDENCE-VAL`。
- 复核命令：`CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_condition_dependence.verify src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016`：通过，15 个独立 NumPy FK 样本最大 EPE 差 0.00000783 mm。
- 正式运行耗时 104.9 秒；GPU 6 peak allocated 2745.3 MiB，输出 49.1 MiB；未启动训练、未生成新 checkpoint。
- core replay / native GT cache replay 差为 0；所有输入/代码摘要、geometry 文件 stat、模型参数及 buffer 保持不变。
- `git diff --check -- docs/current_versions.yaml src/task/CmDecoderv2`：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/CmDecoderv2/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmDecoderv2 --scope-prefix docs/current_versions.yaml --check-links`：通过，覆盖 10 个变更路径，最新条目 16 个本地链接有效；等级与审批另由本记录声明。

**产物**

- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run_manifest.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run_manifest.json)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/config.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/config.json)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/metrics.jsonl](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/metrics.jsonl)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/rollout_metrics.jsonl](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/rollout_metrics.jsonl)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence_summary.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence_summary.json)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence.png](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence.png)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/verification.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/verification.json)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run.log](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run.log)

**回滚与规范反馈**

仅移除本轮新增研究/测试、V1.1.8 增补和活动/实验条目，并恢复 CmDecoderv2 指针；输出可独立隔离。
不执行整个工作区 reset/revert。首次 smoke 的失败是已定位的诊断断言假设，不是权限或规范阻碍；
本轮无审批阻碍，无治理规则改动。后续跨手实验尚未执行，不把本轮同源收益当作跨手结论。

## 2026-09-12 14:50:16 +0800 — V1.1.8 smoke 通过并启动全 val 条件依赖诊断

- activity_id: ACT-20260912-145016-CM-DEPENDENCE-VAL
- timestamp: 2026-09-12 14:50:16 +0800
- modification_version: V1.1.8
- type: diagnostic / experiment / operation / code / documentation
- change_level: L2；单 GPU 运行预算保守按 L3
- approval: user-approved
- approval_basis: 用户回复“按照你的想法继续”，实施已讨论的首项冻结条件依赖诊断。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true
- scope: 本 Task 独立诊断和定向测试，当前计划、Task 日志与版本指针；核心模型、loader、训练、GT、split/cache、旧权重和其他任务保持不变。
- run_id: cm_dependence_val_20260912_145016
- run_status: STARTED
- conclusion: INCONCLUSIVE（正式冻结诊断尚未完成）

**文件**

- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — V1.1.8 final 增补和 native/6-DOF FK 检查说明。
- [src/task/CmDecoderv2/research/cm_condition_dependence/](../../research/cm_condition_dependence/) — 实现、定义、匹配、统计、重放与独立 NumPy FK 复核。
- [src/task/CmDecoderv2/tests/test_cm_condition_dependence.py](../../tests/test_cm_condition_dependence.py) — 7 个定向合同测试。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 仅推进本 Task 指针，其他已有差异保留。
- [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — 运行追溯。

**原因**

首次 smoke 的差异来自真实 native 从动关节与 decoder 固定 mimic/限位假设不一致。已改为以 native
18 维 q 重放 cache 验证采样对应，逐窗口保留 6 维 GT 重建残差；没有改变 GT、预测指标或核心实现。
通过 smoke 后执行既定 30 条 Inspire val 的 4254 active-only 窗口以及最多每序列 3 个 16 步递归起点。

**验证**

- smoke run_id: cm_dependence_smoke_20260912_145100；run_status: COMPLETED；实际启动 2026-09-12 14:47:53 +0800，结束 14:48:16；23.5 秒、2744.8 MiB peak allocated。
- smoke 覆盖 882 窗口、858 h1 有效、835 完整有效；6 个递归起点；原生 FK/cache 抽查最大坐标差 0、原 core 重放差 0，参数/输入不变。只支持工程验证。
- 独立复核 3881 teacher 行、416 rollout 行；donor/配对 summary 可重现，15 个 NumPy FK 样本与归档 EPE 最大差 0.00000780 mm。
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_smoke_20260912_145100/run_manifest.json](../../research/cm_condition_dependence/output/cm_dependence_smoke_20260912_145100/run_manifest.json)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_smoke_20260912_145100/verification.json](../../research/cm_condition_dependence/output/cm_dependence_smoke_20260912_145100/verification.json)
- command: `CUDA_VISIBLE_DEVICES=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_condition_dependence.run --run-id cm_dependence_val_20260912_145016 --activity-id ACT-20260912-145016-CM-DEPENDENCE-VAL`
- PENDING [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/)
- PENDING [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run_manifest.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run_manifest.json)

**回滚**

仅隔离本次新增诊断/测试/记录和输出，并恢复 CmDecoderv2 指针；不回退整个工作区。单 GPU 6、30 分钟、8 GiB allocated、1 GiB output，不启动训练。

## 2026-09-12 14:45:00 +0800 — V1.1.8 首次 smoke 因 GT 重放假设失败

- activity_id: ACT-20260912-144500-CM-DEPENDENCE-SMOKE
- timestamp: 2026-09-12 14:45:00 +0800
- modification_version: V1.1.8
- type: code / diagnostic / experiment / operation / documentation
- change_level: L2（冻结对照/指标）；L3（有限 GPU 运行预算）
- approval: user-approved
- approval_basis: 用户在跨手路线及先做 Cm 条件依赖诊断的建议后回复“按照你的想法继续”。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true（开始时的其他 Task/根诊断差异保留）
- scope: 本 Task 独立 cm_condition_dependence 诊断、定向测试、当前计划/活动/实验记录与版本指针。
- run_id: cm_dependence_smoke_20260912_144500
- run_status: FAILED
- last_step: 首个 batch
- exit_reason: 约束后的 6 维 GT FK 不等于 native simulator 的实际从动关节状态；原检查错误地要求两者相等，最大坐标差 0.577569 mm。
- conclusion: INVALID_IMPLEMENTATION（只针对诊断中的 GT 重放断言；不据此判定已有模型训练无效）

**文件**

- [src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) — 定稿 V1.1.8 首项执行边界。
- [src/task/CmDecoderv2/research/cm_condition_dependence/](../../research/cm_condition_dependence/) — 完整 Cm bank、原状态下交换/置零、匹配/统计与短递归诊断。
- [src/task/CmDecoderv2/tests/test_cm_condition_dependence.py](../../tests/test_cm_condition_dependence.py) — donor/GT 隔离、有效性、时间覆盖、误差单位与配对统计验证。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmDecoderv2 指针推进到 V1.1.8，保留 ObjectInteractionCm 原有改动。
- [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) — 记录实现与运行。

**原因**

先确认同源目标手是否真正依赖 Cm，随后才能解释跨手失败。复用 checkpoint 内嵌配置与现有
model/dataset/FK，保护 core、cache、split、GT、旧 checkpoint、指导/架构及运行中的其他进程。
本轮没有新训练；随机/严格匹配 donor 均来自同一序列，完整交换 tokens/anchors/window。

**验证**

- `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests/test_cm_condition_dependence.py`：6 passed。
- `git diff --check -- docs/current_versions.yaml src/task/CmDecoderv2`：通过。
- physical GPU 6 启动前无 compute process，5 MiB/0% 使用；单 GPU、30 分钟、8 GiB allocated、1 GiB output 上限。
- command: `CUDA_VISIBLE_DEVICES=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_condition_dependence.run --run-id cm_dependence_smoke_20260912_144500 --activity-id ACT-20260912-144500-CM-DEPENDENCE-SMOKE --smoke`
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_smoke_20260912_144500/](../../research/cm_condition_dependence/output/cm_dependence_smoke_20260912_144500/)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_smoke_20260912_144500/run_manifest.json](../../research/cm_condition_dependence/output/cm_dependence_smoke_20260912_144500/run_manifest.json)
- [src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_smoke_20260912_144500/run.log](../../research/cm_condition_dependence/output/cm_dependence_smoke_20260912_144500/run.log)

**回滚**

只移除本次新增诊断/测试/记录及对应 output，并恢复本 Task 指针；不回退整个工作区，不修改用户已有差异。

## 2026-09-11 09:26:31 +0800 — V1.1.7 GT 接触起点 Inspire effect viewer

- activity_id: `cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441`
- timestamp: `2026-09-11 09:26:31 +0800`
- modification_version: `V1.1.7`
- type: `diagnostic / operation / documentation`
- change_level: `L0`（复用既有 viewer 脚本启动交互可视化；不改变研究变量、训练、正式 cache/schema、split、模型或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户要求把刚完成的 contact-start Inspire rollout/effect 结果可视化，并指出此前已有类似脚本。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `fc90fcea25df449eac5947175c611b78e027b992`
- worktree_dirty: `false`（viewer 启动时）
- run_id: `cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441`
- run_status: `RUNNING`
- last_step: `323 transitions`（viewer 初始化时已复算 contact-start effect arrays）
- best_metric: `not_applicable`
- conclusion: `INCONCLUSIVE`（交互 viewer 工程启动成功；可视化本身不新增科研结论）
- scope: [inspire_rollout_effect](../../research/inspire_rollout_effect/) 既有中文 Viser viewer；正式训练、cache、split 和 checkpoint 未修改。

**文件**

- [activity_log.md](activity_log.md) — 记录本次 RUNNING viewer 操作、端口、tmux session 和停止入口。
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441/run_manifest.json) — 记录 viewer run 的 checkpoint、配置、起点 frame `45` 和输出入口。
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441/effect.npz) — viewer 初始化时复算的 contact-start effect arrays。
- [viewer log](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441.log) — Viser 启动输出。

**原因**

需要直观看递归 Inspire hand、预测/GT object effect、OICM validity 和接触后漂移过程；复用已有 `--serve` 中文交互入口，避免修改正式实验结果。

**运行**

- command: `tmux new-session -d -s cmdecoderv2_inspire_viewer_092441 "cd /home2/wyy/oyx_ws/Ref2Dex && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441 --run-id cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441 --knn-batch-size 4 --port 8104 --fps 8 --serve > src/task/CmDecoderv2/research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-viewer-20260911-092441.log 2>&1"`
- viewer: `http://localhost:8104`
- tmux session: `cmdecoderv2_inspire_viewer_092441`
- stop command: `tmux kill-session -t cmdecoderv2_inspire_viewer_092441`
- process: pane pid `3722066`，server pid `3722069`，physical GPU `1`。

**验证**

- Viser 日志打印 `Viser Inspire effect viewer: http://localhost:8104`。
- `ss -ltnp` 显示 `0.0.0.0:8104` 正在监听，server pid `3722069`。
- `curl -sS --max-time 5 http://127.0.0.1:8104/ | head -5` 返回 HTML 入口。
- `tmux list-sessions` 显示 `cmdecoderv2_inspire_viewer_092441` 正在运行。
- manifest 记录 `run_status=RUNNING`、`rollout_start_frame=45`、`frame_count=323`、`worktree_dirty=false`。

**保护边界与回滚**

- 未修改代码、训练配置、正式 cache、split、checkpoint 或旧的完整诊断 run；viewer 产物按仓库规则忽略。
- 关闭 viewer 使用 `tmux kill-session -t cmdecoderv2_inspire_viewer_092441`；如需清理产物，删除/隔离本 viewer run 目录和同名 `.log`。

## 2026-09-11 00:14:44 +0800 — V1.1.7 GT 接触起点 Inspire recursive rollout 完整诊断

- activity_id: `cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021`
- timestamp: `2026-09-11 00:14:44 +0800`
- modification_version: `V1.1.7`
- type: `diagnostic / experiment / operation / documentation`
- change_level: `L2`（按已批准接触起点合同运行长时递归诊断；不修改训练、正式 cache/schema、split、模型或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户确认递归测试从首个 GT Inspire 接触帧开始，并使用完整 Inspire `10135` 点、2 cm 半径和后续纯预测反馈。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `aba8a3650714b62823f996d1f1fde3d383b82fbf`
- worktree_dirty: `false`（run 启动时）
- run_id: `cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021`
- run_status: `COMPLETED`
- last_step: `323 transitions`（实际 sequence frame `45..367`）
- best_metric: `not_applicable`
- recent_checkpoint: [decoder best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)
- conclusion: `INCONCLUSIVE`（接触起点和 effect 协议已支持；单序列长期递归的 hand position drift 较大，不能据此宣称 rollout 或跨序列效果成立）
- scope: [V1.1 plan](../plan/v1.1.md) 第 16 节批准的纯 Inspire rollout/effect 诊断、运行产物及本 Task 实验/活动记录；未改变正式训练与数据产物。

**文件**

- [experiment_log.md](experiment_log.md) — 记录本次完整 run 的假设、运行参数、结果和结论边界。
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/run_manifest.json) — 锁定运行 commit、checkpoint、接触起点、输出和 posthoc 分段分析入口。
- [diagnostic_tables.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/diagnostic_tables.json) — 保存 GT 接触/远离、预测 rollout validity 和里程碑数据。

**原因**

此前 V1.3 完整诊断从 frame 0 启动，预测 hand 未进入 2 cm 有效区，无法评价接触段 Cm/OICM effect。本次按用户确认的首个 GT 接触帧重新运行，以区分接触段 effect 质量和递归 hand state 漂移。

**运行**

- command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021 --run-id cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021 --knn-batch-size 4`
- device: 物理 GPU `1`（通过 `CUDA_VISIBLE_DEVICES=1` 映射为进程内 `cuda:0`）；省略 `--rollout-start-frame`，自动选择首个 GT `<=20 mm` 合法接触帧。
- decoder checkpoint SHA256: `e60f0e954c062d15e7c2fa217e1bebcb0a4b0be8795d1b5729771ee1f8f5fef7`。
- frozen OICM checkpoint SHA256: `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。

**结果**

- 起点: 0-based sequence frame `45`、source frame `180`、GT hand-object 最近距离 `0.564 mm`；只在该帧做一次 GT Inspire state handoff，后续不再输入 GT hand state。
- 轨迹范围: `323` 个 transition，sequence frame `45..367`；GT hand 到 object 的 `<=20 mm` 接触段为 `273` 帧（`45..317`），之后 `50` 帧为 GT 远离段。
- OICM validity: 递归预测 hand 的 `<=20 mm` 段为 `285` 帧（`45..329`），`sample_valid=true` 为 `285/323=88.24%`；sequence frame `330..367` 的 `38` 帧无效，effective effect 按合同归零。
- GT 接触段 `45..317`: Cm/OICM 对 display-only 实际 object flow 的 `pred-GT effect EPE` mean/median/p90/max=`3.545/3.053/4.437/47.174 mm`；effective effect RMS mean=`5.566 mm`，GT effect RMS mean=`3.611 mm`。
- 同一 GT 接触段的递归 hand position EPE mean/median/max=`87.909/88.100/156.048 mm`，hand-flow EPE mean=`2.363 mm`；说明状态位置已明显漂移，但单步 flow 仍部分跟随。
- 全部 transition 的递归 hand position EPE mean/median/max=`246.452/90.633/1517.173 mm`；effective effect RMS mean=`4.923 mm`，raw effect RMS mean=`8.183 mm`，无效帧的 raw 输出不作物理解释。
- `effect_summary.json` 的 `near_le_20mm`/`far_gt_20mm` 分组沿用历史字段，但其距离来源是预测 rollout hand 的 `min_hand_object_distance_mm`；GT 接触/远离分组和里程碑以 `diagnostic_tables.json` 为准。

**验证**

- run 状态为 `COMPLETED`，生成 `323` 个 transition；manifest 记录 `base_commit=aba8a36`、`worktree_dirty=false`、起点 frame `45`。
- `effect.npz`、`effect_summary.json`、`run_manifest.json`、`source_knn_indices.npy` 和 posthoc `diagnostic_tables.json` 均存在且非空。
- 接触段分段统计与 effect 里程碑由 `effect.npz` 独立复核，GT 接触 `273` 帧、OICM valid `285/323` 与 manifest/日志一致。

**结论边界**

- `SUPPORTED`（工程协议）: 自动接触起点解析为 frame `45`，真实 sequence/source frame 映射正确，单次 GT handoff 后递归反馈正确执行；KNN、2 cm validity 和 display-only GT object flow 均按计划隔离。
- `INCONCLUSIVE`（科研效果）: GT 接触段内有效 effect EPE 均值为 `3.545 mm`，表明该序列上可以测到有意义的 Cm/OICM object-effect 输出；但递归 hand position EPE 已达 `87.909 mm` 均值且后段完全失效，单序列不能证明长期 rollout 或一般化效果。
- `INCONCLUSIVE`（display-only 对照解释）: GT object flow 不进入 decoder/OICM，也不反馈 rollout；后段 GT object flow 接近零时，invalid effective effect 与 GT 的低 EPE 不能当作预测能力证据。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/effect_summary.json)
- [diagnostic_tables.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/diagnostic_tables.json)
- [source_knn_indices.npy](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/source_knn_indices.npy)
- [最终计划](../plan/v1.1.md)

**保护边界与回滚**

- 未修改训练配置、训练运行、正式 view/cache、split、checkpoint、`src/base/` 或旧的 frame 0 输出；生成目录按仓库规则忽略。
- 回滚入口：回退记录本次终态的 Git 提交并删除/隔离本 run 输出目录；实现提交 `aba8a36` 及旧 frame 0 诊断保持独立。

## 2026-09-11 00:06:30 +0800 — V1.1.7 recursive rollout 改为 GT 接触起点

- activity_id: `cmdecoderv2-inspire-effect-contact-start-impl-20260911-000630`
- timestamp: `2026-09-11 00:06:30 +0800`
- modification_version: `V1.1.7`
- type: `code / diagnostic / documentation / operation`
- change_level: `L2`（改变诊断 rollout 起点和结果帧索引合同；不改变训练、正式 cache/schema、split、模型或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户确认递归测试从 GT Inspire 10135 点首次进入 2 cm 接触半径的合法帧开始，本序列使用 0-based frame 45，之后只递归反馈预测状态。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `7e30a987e5965de47d0e0e5428091fe4b0d36916`
- worktree_dirty: `true`（本条记录随实现一同提交前）
- run_id: `cmdecoderv2-inspire-effect-v13-contact-smoke-20260911-000533`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（接触起点自动解析、真实 sequence frame 映射和接触区 2-step smoke 均符合合同；完整科研结果待独立 full run）
- scope: [V1.1 plan](../plan/v1.1.md) 第 16 节、[inspire_rollout_effect](../../research/inspire_rollout_effect/) 诊断脚本/说明/元数据和对应 Task tests；正式训练与数据产物未修改。

**文件**

- [plan/v1.1.md](../plan/v1.1.md) — 追加用户批准的 GT 接触起点 recursive rollout 合同，覆盖原 frame 0 正式复测口径。
- [research/inspire_rollout_effect/run.py](../../research/inspire_rollout_effect/run.py) — 自动计算 GT Inspire 10135 点到完整 object pool 的首个 `<=20 mm` 帧；full run 从该帧启动，保存 `sequence_frame`、GT 距离和 manifest 起点字段。
- [research/inspire_rollout_effect/README.md](../../research/inspire_rollout_effect/README.md) — 说明默认接触起点和显式起点参数。
- [research/inspire_rollout_effect/experiment.yaml](../../research/inspire_rollout_effect/experiment.yaml) — 增加 rollout 起点合同。
- [tests/test_inspire_effect.py](../../tests/test_inspire_effect.py) — 增加接触起点与窗口容量回归测试。

**原因**

此前完整 V1.3 recursive run 从 frame 0 启动，属于训练分布外的远距离起点；用户确认递归测试应从 GT 已接触帧启动，以评估接触段内的 Inspire rollout 和 Cm/OICM effect。

**验证**

- py_compile: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/research/inspire_rollout_effect/run.py`，通过。
- Task tests: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`，`20 passed`。
- `git diff --check`：通过。
- contact-start smoke: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-contact-smoke-20260911-000533 --run-id cmdecoderv2-inspire-effect-v13-contact-smoke-20260911-000533 --max-steps 2 --knn-batch-size 4`。
- smoke evidence: [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-smoke-20260911-000533/)、[run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-smoke-20260911-000533/run_manifest.json)、[effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-smoke-20260911-000533/effect.npz)。
- smoke result: `rollout_start_frame=45`、`rollout_start_source_frame_id=180`、GT 起点距离 `0.564 mm`；输出 `sequence_frame=[45,46]`，两步 `oicm_sample_valid=true`，接触区 effect RMS 均值 `10.661 mm`。

**保护边界与回滚**

- 未修改训练配置、训练运行、正式 view/cache、split、checkpoint 或 `src/base/`；未覆盖旧的 frame 0 输出。
- 回滚入口：回退本次实现提交并隔离 smoke 输出目录；旧 frame 0 运行和其日志保持不变。

## 2026-09-10 18:02:00 +0800 — V1.1.7 训练窗口起点与 rollout 语义核查

- activity_id: `cmdecoderv2-training-window-rollout-audit-20260910-180200`
- timestamp: `2026-09-10 18:02:00 +0800`
- modification_version: `V1.1.7`
- type: `diagnostic / documentation`
- change_level: `L0`（只读代码、配置和既有 cache 诊断；不改变训练变量、数据、模型或科研结论）
- approval: `auto`
- approval_basis: 仓库规则允许对只读诊断结果追加 activity 记录。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `4528ed9`
- worktree_dirty: `false`（核查开始时）
- run_id: `not_applicable`
- run_status: `not_applicable`
- conclusion: `SUPPORTED`（训练不是递归 rollout；窗口选择接触感知但不要求窗口起点本身接触）
- scope: `src/task/CmDecoderv2/dataset.py`、`model.py`、`runner.py`、V1.3 配置和既有 V1.3 view/cache；无运行产物变更。

**文件**

- [dataset.py](../../dataset.py) — 核查完整窗口起点枚举、2 cm active mask 筛选、horizon mask 和 dynamic padding。
- [model.py](../../model.py) — 核查 4 个 future horizon 是一次并行解码，没有把前一 horizon 的预测反馈给后一 horizon。
- [runner.py](../../runner.py) — 核查 loss 使用 `active_mask & cm_sample_valid`，只对接触且采样后有效的 horizon 监督。
- [dexplore_rl_v1_3_full10135.yaml](../../configs/active/dexplore_rl_v1_3_full10135.yaml) — 核查正式训练配置 `active_only=true`、`window_size=4` 和 2 cm 合同。

**原因**

需要解释 V1.3 训练是否从接触位置开始，以及它与此前从 frame 0 开始的递归 rollout 诊断之间的分布差异。

**验证**

- `dataset.py` 的窗口起点为 `start=0..T-window_size-1`；`active_only=true` 时保留 `active[start:start+4].any()` 的窗口，因此首个接触帧为 `f` 时最早可保留 `max(0,f-3)`，并非必须从 `f` 开始。
- V1.3 view 每个 train/val sequence 使用 `obj_candidate_mask_2cm.npy`；active 是完整 object pool 上“存在 2 cm 内 KNN hand 邻居”的逐帧 any-mask。
- 既有正式 run metadata 为 train/val windows=`34746/4254`；不是所有完整窗口都进入训练。
- 代表性 cache 核查：`s1/airplane_lift` 首个 active frame=`56`，首个保留 window start=`53`。
- model/runner 核查：同一窗口的 h1..h4 以 frame `start` 的 current state 为基准并行预测；监督 mask=`active_mask & cm_sample_valid`。
- 此前 V1.3 full Inspire rollout 诊断显式从 frame `0` 开始递归，因此该诊断不是“从训练接触窗口起点启动”。

**保护边界与回滚**

- 未修改代码、配置、正式 cache、checkpoint、split 或已有实验结果；仅新增本 activity 记录。
- 回滚入口：移除本条 activity 记录即可，不涉及任何模型或数据产物。

## 2026-09-10 17:57:08 +0800 — V1.1.7 纯 Inspire V1.3 rollout/effect 完整诊断

- activity_id: `cmdecoderv2-inspire-effect-v13-full-20260910-175216`
- timestamp: `2026-09-10 17:57:08 +0800`
- modification_version: `V1.1.7`
- type: `diagnostic / experiment / operation / documentation`
- change_level: `L2`（按已批准 Task-local 诊断协议运行 V1.3 `unique_knn_edges`、10135 点、K=32 和 2 cm validity；只更新运行记录，不修改正式 cache/schema、split、模型或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户确认按 V1.3 decoder `best.pt`、冻结 OICM V1.3、纯 Inspire `s1/mouse_lift`、逐步 `t -> t+1` display-only object flow 对照和临时 Inspire KNN 运行完整诊断。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `6aa2c13b53b4748b9d8205edc459e61463ddeeb1`
- worktree_dirty: `false`（run 启动时；本条为终态记录）
- run_id: `cmdecoderv2-inspire-effect-v13-full-20260910-175216`
- run_status: `COMPLETED`
- last_step: `368 transitions`
- best_metric: `not_applicable`
- conclusion: `REFUTED`（该 V1.3 decoder `best.pt` 在 `s1/mouse_lift` 递归 rollout 中未能进入 GT 接触段；OICM 有效 object-flow 预测因全程 `sample_valid=false` 只能判为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/` 完整 368-step 诊断输出与 Task 级实验记录；正式 V1.3 cache、split、训练配置、decoder/OICM checkpoint 和 `src/base/` 均未修改。

**文件**

- [activity_log.md](activity_log.md) — 记录本次完整 run 的终态、命令、证据、结论边界和保护范围。
- [experiment_log.md](experiment_log.md) — 新增 V1.3 full10135 纯 Inspire recursive rollout/effect 诊断结果。

**原因**

用户要求测试 Inspire 自身 rollout 效果，并查看 rollout 若干步对应的 Cm/OICM 对实际 object flow 的预测效果。
此前仅完成 V1.3 wiring smoke，需要在实现提交后以干净 HEAD 对 `s1/mouse_lift` 跑满 368 个 transition。

**验证**

- full command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-full-20260910-175216 --run-id cmdecoderv2-inspire-effect-v13-full-20260910-175216 --knn-batch-size 4`。
- full evidence: [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/)、[run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/run_manifest.json)、[effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/effect.npz)、[effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/effect_summary.json)、[source_knn_indices.npy](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/source_knn_indices.npy)、[diagnostic_tables.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/diagnostic_tables.json)。
- full result: 368 个 transition，`distance_threshold_mm=20.0`；rollout predicted hand 到 object 最近距离 min/median/max=`298.091/480.982/1245.904 mm`，368/368 均为 `>20mm`，`oicm_valid_ratio=0.0`，有效 object-flow RMS 均值/最大值=`0/0 mm`。
- self-rollout result: hand EPE mean/median/max=`640.851/580.975/1192.516 mm`；future hand EPE mean=`643.875 mm`；hand-flow EPE mean/median/max=`14.610/4.927/89.366 mm`。
- display-only object flow: GT effect RMS mean/max=`2.693/55.968 mm`，effective prediction-vs-GT EPE mean=`2.672 mm`；该 EPE 主要是零有效预测对 display-only GT 的差异，不代表接触区预测质量。
- GT distance audit: 同一 V1.3 `10135` surface sampling 的真实 Inspire 轨迹有 `273/371` 帧 `<=20mm`，首个 `<=20mm` step 为 `46`；因此全程 invalid 来自递归 rollout 未进入接触区，而不是序列本身无接触。

**保护边界与回滚**

- 未写入 `data/processed_data/`，未修改正式 V1.3 view/cache/split、训练配置、decoder/OICM checkpoint 或 `src/base/`；临时 KNN 和诊断表仅在本次 ignored output 目录内。
- 回滚入口：删除或隔离 [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/) 并回退本条 activity/experiment 记录；实现代码回滚入口为提交 `6aa2c13b53b4748b9d8205edc459e61463ddeeb1`。

## 2026-09-10 17:48:24 +0800 — V1.1.7 纯 Inspire rollout/effect 诊断接入 V1.3 KNN

- activity_id: `cmdecoderv2-inspire-effect-v13-impl-20260910-174824`
- timestamp: `2026-09-10 17:48:24 +0800`
- modification_version: `V1.1.7`
- type: `code / diagnostic / documentation / operation`
- change_level: `L2`（Task-local 诊断输入合同从旧 full-hand stream 适配到 V1.3 `unique_knn_edges`、10135 点、K=32 和 2 cm validity；不改变正式 cache/schema、split、模型或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户确认按 `s1/mouse_lift`、V1.3 decoder `best.pt`、冻结 OICM V1.3、逐步 `t -> t+1` display-only object flow 对照和临时 Inspire KNN 执行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `59f1d809a910041f888e68565f6e3c6bc9dac229`
- worktree_dirty: `true`（本条记录随实现一同提交前）
- run_id: `cmdecoderv2-inspire-effect-v13-smoke-20260910-174418`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅支持 V1.3 诊断 wiring 和 2-step smoke；完整 368-step 科研结果待后续独立 run）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/`、对应 Task test 和 [V1.1 plan](../plan/v1.1.md) 第 15 节；正式 V1.3 cache、split、训练配置、decoder/OICM checkpoint 和 `src/base/` 均未修改。

**文件**

- [plan/v1.1.md](../plan/v1.1.md) — 追加用户批准的 V1.3 纯 Inspire rollout/effect 诊断边界：单序列 `s1/mouse_lift`、368 个逐步 transition、临时 source KNN、真实 object flow 只作 display-only。
- [research/inspire_rollout_effect/run.py](../../research/inspire_rollout_effect/run.py) — 对 V1.3 `unique_knn_edges` 添加临时 GT Inspire source KNN、V1.3 surface sampling、compact unique hand stream、per-step predicted-hand KNN effect path、20 mm 动态分桶和 rollout hand/flow EPE。
- [research/inspire_rollout_effect/README.md](../../research/inspire_rollout_effect/README.md) — 更新 V1.3 运行命令、10135/K=32/2 cm 合同和临时 KNN 产物说明。
- [research/inspire_rollout_effect/experiment.yaml](../../research/inspire_rollout_effect/experiment.yaml) — 将实验元数据更新到 V1.1.7/V1.3 诊断口径。
- [tests/test_inspire_effect.py](../../tests/test_inspire_effect.py) — 增加 20 mm summary 分桶和 compact KNN edge stream 回归。

**原因**

V1.3 decoder/OICM 已改为 `unique_knn_edges`，旧诊断脚本会用 full hand stream 或误用 MANO test KNN，
不能无歧义评估纯 Inspire `10135` 点 rollout effect。需要为诊断单独生成临时 Inspire KNN，并保持真实
object flow 只作为 display-only 对照。

**验证**

- py_compile: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/research/inspire_rollout_effect/run.py`，通过。
- Task tests: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`，`18 passed`。
- `git diff --check`：通过。
- smoke command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-smoke-20260910-174418 --run-id cmdecoderv2-inspire-effect-v13-smoke-20260910-174418 --max-steps 2 --knn-batch-size 4`。
- smoke evidence: [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/)、[run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/run_manifest.json)、[effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/effect.npz)、[effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/effect_summary.json)、[source_knn_indices.npy](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/source_knn_indices.npy)。
- smoke result: 2 个 transition 完成；`distance_threshold_mm=20.0`，2/2 为远距离无效 sample，effective object-flow RMS 为 `0 mm`；真实 object flow 为 display-only，`overall_pred_gt_effect_epe_mm=1.58898`。

**保护边界与回滚**

- 未复用 MANO test cache 的 `obj_knn_indices` 作为纯 Inspire KNN；未写入 `data/processed_data/`，未修改正式 V1.3 decoder view、OICM cache/checkpoint、训练配置、正式 split 或旧输出。
- 回滚入口：移除本次 Task-local 诊断代码/文档改动，隔离 smoke 输出目录；不删除原始数据、正式 cache 或 checkpoint。

## 2026-09-10 16:35:24 +0800 — V1.1.7 Inspire 全点监督 decoder 正式训练完成

- activity_id: `cmdecoderv2-v1.1.7-formal-20260910-163524`
- timestamp: `2026-09-10 16:35:24 +0800`
- modification_version: `V1.1.7`
- type: `experiment / operation`
- change_level: `L3 + L2`（按已定稿 [V1.1 plan](../plan/v1.1.md) 执行 V1.3 OICM、KNN hand stream、GT/point-flow 点数和三卡正式长训练）
- approval: `user-approved`
- approval_basis: 用户确认使用最终 OICM V1.3 `best.pt`，Inspire 使用完整 `10135` 点监督，并回复“是的，就按你的来”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `cb4161cab47678a46b69ec440d3dd06334e5645f`
- worktree_dirty: `false`（正式 run 启动和完成时；本条为终态补记）
- run_id: `cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812`
- run_status: `COMPLETED`
- last_step: `72400`
- last_epoch: `50`
- best_metric: `val/loss=0.003993955866854461`（epoch 5 / step 7240）
- conclusion: `INCONCLUSIVE`（工程训练协议 `SUPPORTED`；validation 在早期达到最好，且该 run 不包含 MANO→Inspire 定量 test，因此不单独宣称跨 embodiment 效果成立）
- scope: `src/task/CmDecoderv2/` 的 V1.3 decoder view、冻结 OICM 接入、unique KNN edge stream、动态 hand padding、Inspire `10135` 点 point-flow supervision 和正式训练输出；不改旧 cache、checkpoint、split 或 `src/base/`

**固定输入与运行**

- frozen OICM checkpoint: [V1.3 best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)，SHA256 `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。
- decoder view: [dexplore_rl_v1_3_full10135](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/)，`K=32`、2 cm 仅用于 edge validity/active-only、运行时重算 32 条边距离、全局 hand id 去重、batch-max 动态 padding、point-flow target 全部 `10135` 点。
- view evidence: [view manifest](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/manifest.json)、[view run manifest](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/run_manifest.json)；train/val/test sequence=`255/30/63`，view train/val windows=`63988/7807`，启用 `active_only` 后实际 train/val windows=`34746/4254`，test 仍为 MANO qualitative-only。
- command: `CUDA_VISIBLE_DEVICES=0,2,3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --distributed`
- device: physical GPU `0,2,3`；per-device batch `8`；global batch `24`；seed `42`；decoder 从头训练；50 epochs / `72400` steps；耗时约 `03:52:52`。

**结果与证据**

- validation 首个 epoch：`val/loss=0.0093770677`，point-flow EPE=`26.4114 mm`。
- best `val/loss`：epoch `5` / step `7240`，`0.0039939559`；point-flow EPE=`12.8782 mm`，h1=`7.1633 mm`；[best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)。
- best total point-flow EPE：epoch `27` / step `39096`，`12.6071 mm`；该指标不是当前 checkpoint 选择指标。
- final epoch `50` / step `72400`：`val/loss=0.0041478906`，point-flow EPE=`12.7156 mm`，h1=`9.4626 mm`，wrist translation=`11.8189 mm`，wrist rotation=`6.1306 deg`；[latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/latest.pt)。
- [正式运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/)、[config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/config.json)、[metadata.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metadata.json)、[run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/run_manifest.json)、[metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/train.log)。

**原因**

用户已批准在最终 OICM V1.3 上重新从头训练 decoder；本次终态核对用于确认完整 `10135` 点 point-flow supervision、V1.3 unique-KNN 输入和三卡长任务均按最终 plan 执行，并确定可供后续诊断使用的 best checkpoint。

**验证**

- `py_compile`、`pytest -q src/task/CmDecoderv2/tests`：实现前已通过，Task tests=`15 passed`。
- 正式 `metrics.jsonl`：`824` 条记录，train/val epoch 各 `50`；最终 step/epoch=`72400/50`；所有数值 finite，未发现 `Traceback`、`NaN`、`Inf`、`OOM` 或数据加载错误。
- `torch.load` 核对：`best.pt` 对应 epoch `5` / step `7240`，`latest.pt` 对应 epoch `50` / step `72400`；best metric 与 metrics 一致。
- 训练终止后 torchrun 和 3 个 worker 均已退出；其他 GPU 上的既有任务未停止。
- [V1.1.7 最终计划](../plan/v1.1.md) 与当前配置、view manifest、metadata 的 KNN/10135-point contract 一致。

**保护边界与回滚**

- 未修改既有 V1.2.5 decoder view、V1.3 OICM cache/checkpoint、原始数据、正式 split、旧 decoder output、`src/base/` 或其他 GPU 任务。
- 回滚入口：隔离本次 V1.1.7 的 Task-local 代码/config、`dexplore_rl_v1_3_full10135` view 和本 run output；不删除旧 cache、checkpoint 或运行。

**规范反馈**

- 本次没有审批或目录阻碍。当前 BaseRunner 的 `run_manifest.json` 记录启动时 provenance，训练终态按 Skill 规范登记在本条 activity；manifest 中通用路径扫描器会把 OICM SHA256 字符串额外显示为一个 `missing` 路径，但实际 checkpoint 路径、大小和 SHA256 均已在 metadata/activity 中核验，不影响本次 run 可复现性。

## 2026-09-10 11:09:19 +0800 — V1.3 中间 best.pt 的 MANO/Inspire Cm 来源分类诊断完成

- activity_id: `cmdecoderv2-cm-source-classifier-v13-20260910-110919`
- timestamp: `2026-09-10 11:09:19 +0800`
- modification_version: `V1.1.6`（分类诊断 Task）；上游冻结 OICM 为 `V1.3`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local 分类诊断输入适配；不改变 OICM、cache、正式 split 或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户确认使用 V1.3 训练中的中间 `best.pt`，按 train 训练、val-held-out 评估，并只修改分类诊断脚本输入适配。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d5711026a6bd6687a8e7835785108034f21389a9`
- worktree_dirty: `true`（本条启动记录尚未提交）
- run_id: `cmdecoderv2-cm-source-classifier-20260910-111037`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（Cm 来源信号子结论为 `SUPPORTED`；纯静态手型归因仍不充分）
- scope: `src/task/CmDecoderv2/research/cm_hand_source_classifier/run.py`；使用 V1.3 unique-KNN cache 和当前中间 OICM `best.pt`，保持旧分类协议

**文件**

- `src/task/CmDecoderv2/research/cm_hand_source_classifier/run.py` — 接受 V1.3 index/cache，按有效 `K=32` 边去重构造动态 hand stream，并将 held-out 结果明确标记为 `val_held_out`。
- `src/task/CmDecoderv2/docs/logs/experiment_log.md` — 记录本次 V1.3 中间 checkpoint 分类结果、证据和结论边界。

**原因**

复现实验“只用冻结 Cm 区分 MANO/Inspire”，检验 V1.3 Cm 是否仍包含可识别的手来源/embodiment 信息；主特征为 `cm_tokens` mean/max/std pooling，anchor pooling 继续作为 object-side control。

**运行**

- command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_hand_source_classifier.run --index data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json --oicm-config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml --oicm-checkpoint outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt --device cuda:0 --activity-id cmdecoderv2-cm-source-classifier-v13-20260910-110919 --frames-per-sequence 8 --batch-size 8 --epochs 40`
- device: physical GPU `1`（进程内 `cuda:0`）；当前 OICM V1.3 训练继续使用物理 GPU `0,2,3`
- split: ObjectInteractionCm `train` 训练分类器，`val` 作为 `val-held-out`；正式 `test` 不使用
- output: [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/)
- completed_at: `2026-09-10 11:10:39 +0800`

**验证**

- 代码提交前已通过 `py_compile`、`git diff --check` 和单序列 CUDA smoke；完整运行 `COMPLETED`。
- 共同 object 类别 `11` 个；train `60 MANO + 60 Inspire`，val-held-out `11 MANO + 11 Inspire`；抽取 `1136` 个 transition。
- 全部样本 Cm：val-held-out accuracy/balanced accuracy `0.5795`，F1 `0.4714`，AUROC `0.6475`；anchor control AUROC `0.6058`。
- `sample_valid=true`：Cm val-held-out accuracy/balanced accuracy `0.7386`，F1 `0.7473`，AUROC `0.7531`；anchor control accuracy `0.6250`，AUROC `0.6689`。
- 证据：[metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/metrics.json)、[features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/features.npz)、[run_manifest.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/run_manifest.json)。

**保护边界与解释限制**

- 使用的中间 OICM `best.pt` SHA256：`3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`；未使用随后完成训练产生的最终状态重新运行。
- 未修改 V1.3 cache、OICM 训练配置/模型、正式 train/val/test split 或 checkpoint；只提交了分类诊断脚本适配。
- 结果支持 Cm 仍有弱到中等的可识别 MANO/Inspire source/embodiment 信号，但不能单独证明 Cm 编码纯静态手型；运动、接触状态和 source-domain 统计仍可能贡献分类结果。

## 2026-09-08 20:26:26 +0800 — V1.1.4 teacher-forced effect 误差归因诊断

- activity_id: `cmdecoderv2-teacherforced-effect-attribution-20260908-202626`
- timestamp: `2026-09-08 20:26:26 +0800`
- modification_version: `V1.1.4`
- type: `diagnostic / operation`
- change_level: `L0`（只读逐帧诊断；不修改代码、checkpoint、cache、split 或 viewer 运行）
- approval: `user-approved`
- approval_basis: 用户询问 teacher-forced 下 GT effect 近零而预测 effect 约 6 mm 是否合理。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动）
- run_id: `cmdecoderv2-teacherforced-effect-attribution-20260908-202626`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（6 mm 现象主要由 teacher-forced decoder 单步手流误差解释；OICM GT-hand-flow 对照不支持 OICM 固有 6 mm 偏置）
- scope: `s1/mouse_lift`、修正数据 decoder best、冻结 OICM；无持久化新输出目录，证据记录如下。

**原因**

确认 viewer 的“教师强制”语义：当前 Inspire state 使用 GT，但下一状态仍由 decoder 预测；它不是把 GT hand flow 原样送入 OICM。因此需要把 decoder teacher-forced 路径与 GT hand-flow→OICM 路径分开比较。

**文件**

- `src/task/CmDecoderv2/docs/logs/activity_log.md` — 记录 teacher-forced 与 GT hand-flow→OICM 归因证据。
- `src/task/CmDecoderv2/docs/logs/experiment_log.md` — 追加本次诊断结果和结论边界。

**运行与证据**

- 使用 `EffectDiagnostic.teacher_record` 对 `s1/mouse_lift` 的 `367` 个 transition 逐帧 forward；另用同一 decoder checkpoint 中冻结的 OICM，将真实 GT Inspire hand flow 直接 forward，未写入文件。
- teacher-forced 近物体（距离 `<=50 mm`）`275` 帧全部有效；其中 GT object effect `<=0.1 mm` 的 `126` 帧：GT effect RMS 均值 `0.048 mm`，预测 effect RMS 均值/中位数/最大值 `7.317/7.293/10.120 mm`，prediction-GT EPE 均值 `7.286 mm`。
- 同一 `126` 帧中，teacher-forced decoder hand-flow EPE 均值 `3.850 mm`；GT hand-flow RMS 均值 `1.179 mm`。
- 严格 GT hand-flow→OICM 对照的同一 `126` 帧：预测 object effect RMS 均值/中位数/最大值 `0.706/0.688/1.786 mm`，prediction-GT EPE 均值 `0.660 mm`。
- 远距离帧的 raw 输出仍是无效 sample 的 dummy head path；不用于物理解释。
- [完整 viewer 运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)，其中的 GT/pred effect 字段仍仅作显示和诊断。

**验证**

- 结论：GT object effect 为零而 teacher-forced prediction 约 `6–7 mm` 在当前实现和当前 checkpoint 下是可复现的，但不是“正确的物理预测”；它是 decoder 单步 hand-state/hand-flow 误差经接触敏感 OICM 放大的结果。
- “教师强制”并未消除单步 decoder 误差；它只消除了前序 rollout 状态漂移。若要单独测 OICM，应使用 GT hand-flow→OICM 对照。
- 现有测试仍为 `13 passed`，8104 viewer 保持 HTTP `200`；本次未改变任何运行代码或研究变量。

**保护边界**

- 未修改 decoder/OICM checkpoint、数据、cache、split、配置、模型代码或运行中的 8104/8102/8103 viewer。

## 2026-09-08 19:53:57 +0800 — V1.1.4 中文 Inspire effect viewer 与修正数据完整重跑

- activity_id: `cmdecoderv2-inspire-effect-v1.2.5-best-20260908-194900`
- timestamp: `2026-09-08 19:53:57 +0800`
- modification_version: `V1.1.4`
- type: `code / diagnostic / experiment / operation / documentation`
- change_level: `L1`（Task-local viewer 交互和 display-only GT 诊断字段；不改变 decoder/OICM、训练变量或正式 split）
- approval: `user-approved`
- approval_basis: 用户要求右侧注释中文化、允许任意帧启动递归 rollout、可切换 teacher-forcing/rollout，并同时显示预测 effect 与 GT effect。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动；未覆盖其他 Task 或旧 viewer）
- run_id: `cmdecoderv2-inspire-effect-20260908-195042`
- run_status: `RUNNING`（effect 计算完成；中文 Viser 8104 保持运行）
- conclusion: `INCONCLUSIVE`（支持 OICM 远距离 validity gate 子结论；单序列和跨 embodiment rollout 整体质量仍不能由此判定）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/` 及其 Task 测试/说明

**原因**

将上次会话未完成的交互需求收口：右侧说明中文化，允许从任意合法帧重新 handoff，支持教师强制与递归
Rollout 随时切换，并把真实 object flow 作为不进入模型的 GT effect 对照显示。

**文件**

- `src/task/CmDecoderv2/research/inspire_rollout_effect/run.py` — 增加中文交互 viewer、任意帧 handoff、教师强制/递归 Rollout 切换、预测/GT effect 显示；GT flow 仅 display-only；Ctrl-C 收口 manifest。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/README.md` — 更新中文控件、运行入口和 GT display-only 合同。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/experiment.yaml` — 明确 future object flow 仅允许 display-only，禁止作为模型输入。
- `src/task/CmDecoderv2/tests/test_inspire_effect.py` — 增加可选 GT 摘要字段回归。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/run_manifest.json` — 将被超时终止的旧 viewer 如实标为 `STOPPED`。

**运行与产物**

- full command: `CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v1.2.5-best-20260908-194900 --port 8104 --fps 8 --serve`
- smoke command: 同一 checkpoint/config 加 `--max-steps 2`，run `cmdecoderv2-inspire-effect-20260908-194949`，`COMPLETED`。
- decoder checkpoint: epoch `24` / step `36120`；SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`。
- frozen OICM checkpoint SHA256: `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。
- sequence: `s1/mouse_lift`，368 个 recursive rollout transition；GT object flow 仅用于显示和离线 EPE，不进入模型。
- [完整运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)
- [run manifest](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect_summary.json)
- Viser: `http://localhost:8104`，HTTP `200`；进程保持运行。

**验证**

- `py_compile`（run.py、visualize_inspire_test.py）：通过；`pytest -q src/task/CmDecoderv2/tests`：`13 passed`。
- 2-step smoke 已生成 `gt_obj_flow_object/world`、`gt_effect_rms_mm` 和 `pred_gt_effect_epe_mm`，验证 display-only 字段合同。
- 完整结果：远于 `50 mm` 的 12 帧全部 `sample_valid=false`，effective effect RMS 均值/最大值 `0 mm`；近距离 356 帧全部有效，effective effect RMS 均值 `4.528 mm`、中位数 `1.647 mm`、最大 `52.341 mm`。
- 全局 raw/effective effect RMS 均值 `4.998/4.380 mm`，GT effect RMS 均值 `2.693 mm`，预测-GT EPE 均值 `2.512 mm`，OICM valid ratio `0.9674`。
- 子结论 `SUPPORTED`：该 held-out 序列中远距离 hand flow 仍由 OICM validity gate 屏蔽；总体 decoder rollout/跨 embodiment 结论保持 `INCONCLUSIVE`。

**保护边界与旧运行收口**

- 未读取 GT flow 作为 decoder/OICM 输入，未改变 checkpoint、cache、split、训练配置或旧 8102/8103 viewer。
- 旧 run `cmdecoderv2-inspire-effect-20260908-151014` 的 effect 已完成但 viewer 被执行会话超时终止，manifest 已更正为 `STOPPED`；旧产物保留可审计。

## 2026-09-08 15:12:08 +0800 — V1.1.4 当前修正数据 best.pt 的 Inspire rollout self-effect 诊断

- activity_id: `cmdecoderv2-inspire-effect-v1.2.5-best-20260908-125300`
- timestamp: `2026-09-08 15:12:08 +0800`
- modification_version: `V1.1.4`（复用既有 rollout self-effect 诊断协议；输入 decoder/OICM 为本次修正数据终态）
- type: `diagnostic / experiment / operation`
- change_level: `L0`（按既有诊断协议运行，不修改模型、数据、split、checkpoint 或指标合同）
- approval: `user-approved`
- approval_basis: 用户要求用当前 `best.pt`，将 Inspire rollout 每步产生的 Cm 送入 OICM 预测 object point flow。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动；本次仅产生独立诊断输出并追加记录）
- run_id: `cmdecoderv2-inspire-effect-20260908-151014`
- run_status: `STOPPED`（368-step effect 已计算完成；端口 8104 的 Viser 后续被执行会话超时终止，产物保留）
- conclusion: `INCONCLUSIVE`（远距离 validity gate 子结论成立；单序列 rollout 整体质量仍未判定）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/`

**原因**

需要使用修正数据训练后的当前 decoder `best.pt` 重做既有纯 Inspire self-effect 诊断，观察 decoder 递归
rollout 产生的 Inspire hand flow 经冻结 OICM 后，对物体点流的预测是否仍遵守 5 cm interaction gate；
不与 GT object flow 比较。

**文件**

- [实验记录](experiment_log.md) — 追加当前终态 `best.pt` 的 Inspire rollout effect 证据及正式训练终态。
- [活动记录](activity_log.md) — 登记本次运行状态、命令、输入 checkpoint 和可视化入口。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v1.2.5-best-20260908-125300 --port 8104 --fps 8 --serve`
- decoder checkpoint: epoch `24` / step `36120`；SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`。
- frozen OICM checkpoint SHA256: `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。
- sequence: `s1/mouse_lift`，修正后 MANO-only test parent 中具备 DExplore Inspire tensor 的 held-out 序列；368 个 rollout transition。
- Viser: `http://localhost:8104` 曾返回 HTTP `200`；进程已停止。
- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/)
- [run manifest](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect_summary.json)

**验证**

- 距离 `>50 mm`：12 帧全部 `sample_valid=false`，effective effect RMS 均值/最大值均为 `0 mm`；raw dummy head RMS 均值 `18.949 mm`，不作物理解释。
- 距离 `<=50 mm`：356 帧全部有效，effective effect RMS 均值 `4.528 mm`、中位数 `1.647 mm`、最大值 `52.341 mm`。
- 全局 effective effect RMS 均值 `4.380 mm`，OICM valid ratio `0.9674`；旧 manifest 未包含 display-only GT 字段，已由新完整重跑补齐。
- 初次尝试的历史序列 `s1/camera_takepicture_3_Retake` 不属于修正后的 63 条 test split，loader 在生成输出前按预期拒绝；随后选择同一新 test split 内的 `s1/mouse_lift`，未绕过 split 闸门。

**保护边界**

- 未读取未来 object pose/flow 或 GT object flow；未修改正式 decoder/OICM checkpoint、数据 cache、split、训练配置或已有 viewer。

## 2026-09-08 10:14:02 +0800 — V1.1.6 修正 DExplore RL 数据上的 CmDecoderv2 正式训练完成

- activity_id: `cmdecoderv2-decoder-v1.2.5-20260908-101402`
- timestamp: `2026-09-08 10:14:02 +0800`
- modification_version: `V1.1.6`
- type: `data / experiment / operation`
- change_level: `L2`（切换到修正后的 RL 物体轨迹、OICM V1.2.5 终态 checkpoint 和对应 decoder view；模型结构、loss、split 合同保持不变）
- approval: `user-approved`
- approval_basis: 用户确认按修正后的 DExplore RL 数据和 OICM V1.2.5 best checkpoint 从头训练 decoder。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动；本次只新增独立 decoder config/view/output 并追加记录）
- run_id: `cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（正式训练和 checkpoint 选择按冻结合同完成；该结论不等于 MANO→Inspire rollout 质量成立）
- scope: `src/task/CmDecoderv2/`、`data/processed_data/cm_decoder_v2/dexplore_rl_v1_2_5/`、`outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/`

**原因**

旧 decoder view 和 checkpoint 绑定旧版 OICM 及旧物体轨迹；本次需要在修正后的 DExplore RL 物体轨迹和
OICM V1.2.5 终态 `best.pt` 上重新从头训练，以隔离数据修正对 decoder 的影响。

**验证**

- decoder view builder：`cmdecoderv2-view-full-20260908-100512`，train/val/test=`255/30/63`，RL
  sidecars=`285`，windows=`63988/7807`，split contract 为 Inspire RL train/val + MANO qualitative-only test。
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5_smoke.yaml`：`last_step=2`，完成 train/val，OICM strict-load、point-flow loss 和反向传播通过。
- `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`12 passed`。
- 2026-09-08 11:09:55 +0800 检查：正式 run 已完成约 `26600/75250` steps、进入 epoch 18，进程和三卡 worker 正常；当前最近验证为 step `25585` 的 `val/loss=0.0043540`、`val/hand/point_flow_epe_mm=13.3516`，历史最佳 `val/loss=0.0041447`（epoch 5 / step 7525）。这些仍是中途训练指标，不作为终态科研结论。
- 2026-09-08 11:56:43 +0800 检查：正式 run 已完成约 `50200/75250` steps、进入 epoch 34，进程和三卡 worker 仍正常；最近验证为 step `49665` 的 `val/loss=0.00412665`、`val/hand/point_flow_epe_mm=12.6080`，历史最佳已改善为 `val/loss=0.00400710`（epoch 24 / step 36120）。预计剩余约 45–55 分钟；仍不作终态科研结论。
- 终态：50 epochs / `75250` steps 正常完成，用时 `02:32:18`；best 为 epoch `24` / step `36120`，`val/loss=0.0040070958`、`val/hand/point_flow_epe_mm=12.4907`、h1=`9.0049 mm`；final epoch 50 的 `val/loss=0.0041402271`。

**文件/输入**

- [正式配置](../../configs/active/dexplore_rl_v1_2_5.yaml) — 指向 OICM V1.2.5 `best.pt` 和修正数据 view；从头训练。
- [smoke 配置](../../configs/active/dexplore_rl_v1_2_5_smoke.yaml) — 同一输入合同，关闭扰动并限制 2 steps。
- [decoder view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_2_5/) — 基于修正 OICM index，train/val Inspire RL，test MANO qualitative-only。
- OICM checkpoint SHA256: `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。

**运行**

- command: `CUDA_VISIBLE_DEVICES=4,5,6 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5.yaml --distributed`
- device: `cuda:0,1,2` within `CUDA_VISIBLE_DEVICES=4,5,6`；world size `3`；per-device batch `8`；global batch `24`。
- total steps: `75250`；last epoch: `50`；best metric: `val/loss=0.0040070958`。
- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/)（`COMPLETED`）
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/run_manifest.json)
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/config.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt) — epoch `24` / step `36120`，SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`。
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/latest.pt) — epoch `50` / step `75250`，SHA256 `23c33e1022e970b659d0ae651df04620f9ec1894af98a1b8ae39881f0dfa6ac2`。

**保护边界**

- 旧 decoder view/checkpoint、旧 OICM cache/checkpoint、原始数据、MANO-only test 口径和已有可视化进程不修改。

## 2026-09-07 21:39:30 +0800 — V1.1.6 MANO/Inspire Cm 来源分类诊断

- activity_id: `cmdecoderv2-cm-source-classifier-final-20260907-214000`
- timestamp: `2026-09-07 21:39:30 +0800`
- modification_version: `V1.1.6`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local 表征诊断脚本和小分类器；冻结 OICM、数据 cache、正式 split、decoder checkpoint 和训练变量）
- approval: `user-approved`
- approval_basis: 用户确认训练小分类器检验当前 Cm 是否包含可识别的 MANO/Inspire 手来源信息。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留前序 CmDecoderv2/ObjectInteractionCm 和 rollout 诊断工作区改动）
- run_id: `cmdecoderv2-cm-source-classifier-20260907-213748`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（冻结 OICM 的 `cm_tokens` 含有明显可识别的 MANO/Inspire source/embodiment signal；纯静态手型归因仍需额外控制）
- scope: `src/task/CmDecoderv2/research/cm_hand_source_classifier/`；不改变正式训练或数据合同

**文件**

- `src/task/CmDecoderv2/research/cm_hand_source_classifier/run.py` — 从 MANO/Inspire geometry cache 生成冻结 OICM `cm_tokens`，按 sequence split 训练小型 source classifier，并报告 anchor control。
- `src/task/CmDecoderv2/research/cm_hand_source_classifier/README.md`、`experiment.yaml` — 记录标签、特征、共同 object 类别、split 和输出合同。
- `src/task/CmDecoderv2/docs/plan/v1.1.md` — 记录用户批准的 Cm 来源分类诊断边界。
- `src/task/CmDecoderv2/docs/README.md`、`docs/current_versions.yaml` — 增加分类诊断入口并更新 CmDecoderv2 指针为 `V1.1.6`。
- `src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md` — 其他工作区差异保留，未由本实验覆盖。

**原因**

需要判断当前 Cm 是否保留了足够的 MANO/Inspire 手来源信息，使一个小型分类器能够在未见过的 sequence 上区分两类 source。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_hand_source_classifier.run --activity-id cmdecoderv2-cm-source-classifier-final-20260907-214000 --device cuda:0 --frames-per-sequence 8 --batch-size 8 --epochs 40`
- OICM config: `src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml`；checkpoint 为冻结 `best.pt`，SHA256 `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。
- split: train `226 MANO + 226 Inspire` sequences；val/test `36 MANO + 36 Inspire` sequences；仅使用两种 source 在 train/val 都共同出现的 23 个 object 类别；每条 sequence 抽取 8 个 transition。
- [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/)
- [run manifest](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/run_manifest.json)
- [features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/features.npz)
- [metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/metrics.json)

**验证**

- 主特征是 `cm_tokens` 的 mean/max/std pooling（96D），不输入 hand flow、object points、anchor 或 decoder 输出；anchor mean/max/std（9D）只作为 control。
- 全部样本：Cm test accuracy `0.7448`、balanced accuracy `0.7448`、AUROC `0.8467`；anchor control test AUROC `0.6263`。
- `sample_valid=true` 子集：Cm test accuracy `0.9752`、balanced accuracy `0.9752`、F1 `0.9752`、AUROC `0.9958`；anchor control accuracy `0.6281`、AUROC `0.6702`。
- 特征提取和分类器训练共生成 `4192` 个样本；`PYTHONPATH=. pytest -q src/task/CmDecoderv2/tests`：`12 passed`；脚本 `py_compile` 和 `git diff --check` 通过。
- activity 链接审计待本条写入后执行。

**保护边界与解释限制**

- 这是 source/embodiment 可分性证据，不是“纯静态手型”因果证明；Cm 可能同时编码 hand geometry、motion statistics 和 source-domain 差异。
- 旧 smoke/中间输出保留但不作为最终证据；最终 run 使用小 manifest 和上述 metrics 作为入口。
- 未修改 OICM/decoder checkpoint、正式 train/val/test split、geometry cache、训练配置或共享代码。

## 2026-09-07 20:34:30 +0800 — V1.1.5 MANO rollout self-effect 348-step 对照诊断

- activity_id: `cmdecoderv2-mano-effect-full-20260907-203300`
- timestamp: `2026-09-07 20:34:30 +0800`
- modification_version: `V1.1.5`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local MANO rollout 旁路诊断与可视化；冻结 decoder/OICM、正式 MANO test source、split/cache 和既有 viewer）
- approval: `user-approved`
- approval_basis: 用户要求使用 MANO rollout 做同一 self-effect 诊断并可视化比较。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 和前序诊断工作区改动）
- run_id: `cmdecoderv2-mano-effect-20260907-203231`
- run_status: `RUNNING`（348 步 effect 计算完成；8103 Viser 仍运行）
- conclusion: `SUPPORTED`（本序列中 MANO-source rollout 的远距离 hand flow 同样被 OICM 5 cm validity gate 屏蔽；decoder 整体跨 embodiment 动作质量仍不由此单独判定）
- scope: `src/task/CmDecoderv2/research/mano_rollout_effect/`；不改变正式 test 或训练合同

**文件**

- `src/task/CmDecoderv2/research/mano_rollout_effect/run.py` — 使用正式 MANO source 递归 rollout，并将每步 Inspire hand flow 旁路 forward 冻结 OICM。
- `src/task/CmDecoderv2/research/mano_rollout_effect/README.md`、`experiment.yaml` — 记录 MANO 对照实验合同和入口。
- `src/task/CmDecoderv2/docs/plan/v1.1.md` — 记录用户批准的 MANO rollout self-effect 对照边界。
- `src/task/CmDecoderv2/docs/README.md`、`docs/current_versions.yaml` — 增加诊断导航并更新 CmDecoderv2 指针为 `V1.1.5`。
- `src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md` — 其他工作区差异保留，未由本运行覆盖。

**原因**

需要判断同一 OICM effect 诊断在正式 MANO-source rollout 下是否也把远离物体的 decoder 产生运动识别为无有效 object effect，并和纯 Inspire rollout 结果对照。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.mano_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt --device cuda:0 --sequence s1/camera_takepicture_3_Retake --activity-id cmdecoderv2-mano-effect-full-20260907-203300 --port 8103 --fps 8 --serve`
- tmux session: `cmdecoderv2_mano_effect_20260907_2033`；Viser：`http://localhost:8103`。
- [运行目录](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/)
- [run manifest](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/run_manifest.json)
- [effect.npz](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect.npz)
- [effect_summary.json](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect_summary.json)

**验证**

- 2-step smoke：完成，MANO handoff 初始距离约 `1.27–1.31 m`，OICM 无效且 effective effect 为 `0`。
- `PYTHONPATH=. pytest -q src/task/CmDecoderv2/tests`：`12 passed`；新脚本 `py_compile` 通过。
- 348 个 MANO-source rollout transition 完成；输入使用当前 GRAB/MANO object geometry/pose 和当前 rollout Inspire geometry，不读取 paired Inspire GT、未来 object pose/flow 或 object-flow GT。
- 距离 `>50 mm` 的 258 帧：`sample_valid=0/258`，raw effect RMS 均值 `12.535 mm`，effective effect RMS 均值 `0 mm`。
- 距离 `<=50 mm` 的 90 帧：`89/90` 有效，effective effect RMS 均值 `9.194 mm`，最大 `41.302 mm`；唯一无效近距离帧为 frame `258`，距离约 `47.42 mm` 且 sampled active count 为 `0`。
- 8103 HTTP `200`；`git diff --check` 通过；activity 链接审计待本条写入后执行。

**保护边界**

- raw effect 仅作为无效 OICM sample 的 dummy path 审计量；Viser 默认显示 effective effect，可切换 raw。
- 未修改正式 MANO test index、split/cache、decoder/OICM checkpoint、训练配置、paired Inspire 数据或现有 8101/8102 viewer。

## 2026-09-07 17:31:59 +0800 — V1.1.4 rollout self-effect 348-step OICM 旁路诊断

- activity_id: `cmdecoderv2-inspire-effect-full-20260907-171315`
- timestamp: `2026-09-07 17:31:59 +0800`
- modification_version: `V1.1.4`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local 诊断脚本、实验索引和可视化；冻结 decoder/OICM、数据 split、cache 和既有 viewer 不变）
- approval: `user-approved`
- approval_basis: 用户确认在 rollout 每一步用自身 Inspire point flow 重新 forward OICM effect，不与 GT object flow 比较。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2 point-flow/active-only 及其他 Task 工作区改动）
- run_id: `cmdecoderv2-inspire-effect-20260907-171349`
- run_status: `RUNNING`（effect 已完成；8102 Viser 仍运行）
- conclusion: `SUPPORTED`（本序列中 OICM 的 5 cm validity gate 对 rollout hand flow 生效；decoder 的动作质量仍不由本诊断单独判定）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/`；不改变正式训练或已有 viewer

**文件**

- `src/task/CmDecoderv2/research/inspire_rollout_effect/run.py` — 生成 rollout hand flow，冻结 OICM forward，并提供 raw/effective effect viewer。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/README.md`、`experiment.yaml` — 记录实验合同和入口。
- `src/task/CmDecoderv2/tests/test_inspire_effect.py` — raw/effective mask、分桶摘要和颜色映射测试。
- `src/task/CmDecoderv2/docs/plan/v1.1.md` — 记录用户批准的 rollout self-effect 诊断边界。
- `docs/current_versions.yaml` — CmDecoderv2 指针更新为 `V1.1.4`。
- `src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md` — 本次之前已存在的工作区差异，仅作保护范围，未由本运行修改。

**原因**

需要判断“远离物体时的 rollout Inspire 运动是否被 Cm/OICM 识别为无 object effect”，同时避免把 GT object flow 或未来 object pose 引入诊断。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt --device cuda:0 --sequence s1/camera_takepicture_3_Retake --activity-id cmdecoderv2-inspire-effect-full-20260907-171315 --port 8102 --fps 8 --serve`
- tmux session: `cmdecoderv2_inspire_effect_20260907_1713`；Viser：`http://localhost:8102`。
- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/)
- [run manifest](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect_summary.json)

**验证**

- `py_compile` 和 `PYTHONPATH=. pytest -q src/task/CmDecoderv2/tests`：`12 passed`；包含本诊断新增的 3 个合同测试。
- 348 个 rollout transition 完成；只使用当前 object geometry/pose 作为 `object_pose_t` 输入，没有读取未来 object pose/flow。
- 距离 `>50 mm` 的 122 帧：`sample_valid=0/122`，raw effect RMS 均值 `12.533 mm`，但合同有效 effect 为 `0 mm`。
- 距离 `<=50 mm` 的 226 帧：`225/226` 有效，effective effect RMS 均值 `7.959 mm`，最大 `39.277 mm`；唯一无效帧为 frame `127`，距离 `48.29 mm` 且 sampled active count 为 0，符合 1024 点采样未命中 candidate 的语义。
- 8102 HTTP `200`；`git diff --check` 通过；`audit_diff.py --check-links` 通过（13 个变更路径、4 个本地链接）。

**保护边界**

- raw effect 仅作为 dummy computational path 审计量；不能解释为无效 Cm 的物理 effect。Viser 默认显示 effective effect，可切换 raw 观察该差异。
- 未停止 8101 纯 Inspire viewer；未修改 decoder/OICM checkpoint、正式 split/cache、训练配置或既有输出。

## 2026-09-07 15:45:00 +0800 — V1.1.1 best.pt 纯 Inspire 自回归 viewer 与跳帧控件运行核验

- activity_id: `cmdecoderv2-inspire-best-jump-20260907_103836`
- timestamp: `2026-09-07 15:45:00 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic / operation`
- change_level: `L0`（核验既有 viewer、trajectory 和运行清单；本次未修改模型、数据、配置或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户要求继续查看 point-flow `best.pt` 的 Inspire 自回归 rollout，并增加跳帧观察漂移；viewer 实现和启动范围已在前一轮确认。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留用户已有 CmDecoderv2 point-flow/active-only 改动及其他 Task 改动）
- run_id: `cmdecoderv2-inspire-test-vis-20260907-103857`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（纯 Inspire viewer 工程运行正常；尚未据此形成 decoder 科研效果结论）
- scope: `src/task/CmDecoderv2/`

**运行与控件**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize_inspire_test --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt --device cuda:0 --sequence s1/camera_takepicture_3_Retake --port 8100 --fps 8 --activity-id cmdecoderv2-inspire-best-jump-20260907_103836`
- tmux session: `cmdecoderv2_inspire_best_20260907_103836`；GPU3 进程仍在运行。
- 实际 Viser 地址：`http://localhost:8101`。请求端口 `8100` 被 Viser 自动递增处理，命令行摘要和 manifest 中的请求端口仍为 8100；本次以监听端口 8101 为准。
- 序列：`s1/camera_takepicture_3_Retake`，352 帧；使用 point-flow 正式训练的 `best.pt`（epoch 3 / step 10296）。
- viewer 默认 `rollout`；支持 `teacherforced` / `rollout`、`GT` / `pred` / `both`、点大小滑块、`Jump size (frames)`（1–30）、`Previous jump` / `Next jump`、播放/停止和当前帧重启 rollout。

**产物**

- [运行目录](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260907-103857/)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260907-103857/run_manifest.json)
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260907-103857/trajectory.npz)
- [viewer 实现](../../visualize_inspire_test.py)

**原因**

本次继续的目的只是把上次会话中已经启动的 viewer 运行状态、实际监听端口和跳帧控件证据补齐；不重新训练，也不改变纯 Inspire 诊断与正式 MANO-only test 的边界。

**验证**

- `ps` / `tmux capture-pane`：viewer 主进程存活，Viser 已报告监听 8101。
- `urllib.request http://127.0.0.1:8101/`：HTTP `200`；8097–8100 当前无监听。
- trajectory 文件已存在且可读；运行清单记录 checkpoint、sequence、source contract 和 `run_status=RUNNING`。

**保护边界**

- 未停止或重启正式训练；未修改 best/latest checkpoint、数据 view/cache、split、配置或旧 viewer 输出。
- 下列工作区差异在本次继续前已存在，本次未修改，仅作为交接保护范围：`src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md`。
- 本次继续操作不改变 MANO→Inspire 科研结论；仍需人工通过 viewer 比较 teacherforced 与 rollout 的漂移。

## 2026-09-07 10:21:16 +0800 — V1.1.3 point-flow 三卡正式训练终态与收敛诊断

- activity_id: `cmdecoderv2-pointflow-train-final-20260907-102116`
- timestamp: `2026-09-07 10:21:16 +0800`
- modification_version: `V1.1.3`
- type: `experiment / diagnostic / operation`
- change_level: `L0`（只读解析既有训练产物并补记终态，不修改模型、数据或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户要求核对当前训练结果；原正式训练由用户明确批准。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 Task-local point-flow/active-only 实现和用户工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046`
- run_status: `COMPLETED`
- last_step: `171600`
- last_epoch: `50`
- best_metric: `val/loss=0.003370641680534728`（epoch `3`，step `10296`）
- conclusion: `INCONCLUSIVE`（正式训练工程上完整结束且 point-flow validation 明显优于 epoch 1；但验证最优过早、后续回退，且 q/rotation 诊断劣于 identity baseline，跨 embodiment 效果需用 best.pt 可视化确认）
- scope: `src/task/CmDecoderv2/`

**终态产物**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/)
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/run_manifest.json)
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/config.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt) — epoch `3` / step `10296`。
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/latest.pt) — epoch `50` / step `171600`。

**原因**

训练进程和 tmux session 均已退出，需要区分“训练完整结束”“优化收敛”和“跨 embodiment 方案成立”，并核对 best.pt 是否确实对应最低 validation loss。

**验证**

- 50 epochs / `171600` steps 正常完成，用时 `07:28:30`；`train.log` 末行明确报告结束，无残留 CmDecoderv2 train/torchrun 进程。
- train loss 从 epoch 1 的 `0.0116015` 持续降到 epoch 50 的 `0.00296866`；val loss 从 `0.00757946` 降至 epoch 3 最佳 `0.00337064`，之后未再刷新，epoch 50 为 `0.00382680`。
- best epoch 3：validation 全 horizon point-flow EPE `11.3794 mm`，h1 `6.8813 mm`；final epoch 50 为 `12.1087 / 9.1764 mm`。因此选模必须使用 best.pt，不应使用 latest.pt。
- best epoch 3 的 h1 诊断：q MAE `0.04289 rad`、wrist translation `5.935 mm`、wrist rotation `2.570°`；对应 identity baseline 为 `0.01529 rad / 11.046 mm / 2.010°`。模型明显改善 wrist translation，但 q 与 wrist rotation 尚未超过 identity baseline。
- validation `cm/supervision_frame_ratio=0.966668`，5 cm active-only 与 OICM valid gate 正常生效。

**保护边界与下一证据**

- 未修改或覆盖 best/latest checkpoint、数据 view/cache、split、旧 run 或可视化产物。
- 下一步应固定使用 best.pt 做 teacher-forced 与 rollout 可视化；仅凭本次 loss 曲线不能把 MANO→Inspire 可行性写为 `SUPPORTED`。

## 2026-09-06 20:31:18 +0800 — V1.1.3 启动 CmDecoderv2 point-flow 三卡正式训练

- activity_id: `cmdecoderv2-pointflow-train-20260906-203118`
- timestamp: `2026-09-06 20:31:18 +0800`
- modification_version: `V1.1.3`
- type: `operation / experiment`
- change_level: `L3 + L2`（按已定稿 v1.1 plan 启动长任务；point-flow、active-only 和 RL-only 数据合同已由用户批准）
- approval: `user-approved`
- approval_basis: 用户明确要求“现在开始训练”，沿用此前 GPU 0/1/2 三卡正式训练范围。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（沿用已批准的 Task-local point-flow/active-only 改动；不覆盖旧 run、OICM checkpoint 或 viewer）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（已确认训练 wiring 正常；尚无收敛或 MANO→Inspire 效果结论）
- scope: `src/task/CmDecoderv2/`

**命令与运行状态**

- command: `CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --distributed`
- tmux session: `cmdecoderv2_pointflow_20260906_2030`
- launcher/rank 已启动，world size `3`，global batch `24`，per-device batch `8`，训练上限 `171600` steps。
- step `2000` 已写入 point-flow loss `0.0086414`，h1 点流 EPE `16.6742 mm`，active/supervision ratio `1.0`；最近 20 个记录点的 loss 均值约 `0.01177`、标准差约 `0.00206`，属于 batch-level 波动，不能据此判断收敛。

**产物**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/)（`RUNNING`）
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/run_manifest.json)
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/config.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/train.log)
- best/latest checkpoint：`outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/`（尚未生成，`PENDING`）

**原因**

在 point-flow loss、可微 Inspire FK、5 cm active-only gate 和旧 OICM best checkpoint strict-load smoke 均通过后，开始新的独立三卡正式 run。旧 decoder run `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701` 保持停止并保留。

**验证**

- OICM frozen best checkpoint SHA256 与配置一致：`fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。
- 训练启动后已生成 run manifest/config/metadata/metrics/train.log，3 个 rank 正常运行；GPU 0/1/2 当前显存约 `5.6/3.1/3.1 GB`（总显存 24 GB），利用率随 rank 负载波动。
- 早期 step 已产生 finite point-flow loss 和诊断指标；正式科研结论等待完整 validation 与最终 checkpoint。

**保护边界与停止入口**

- 未覆盖旧 checkpoint、旧输出、decoder view/cache、split 或 8098/8099 viewer。
- 如需停止，使用 tmux session `cmdecoderv2_pointflow_20260906_2030` 发送安全中断，并在本条补记终态、last step/epoch、best metric 和 checkpoint。

## 2026-09-06 20:22:37 +0800 — V1.1.3 将 CmDecoderv2 loss 改为可微 Inspire 点流监督

- activity_id: `cmdecoderv2-pointflow-20260906-202237`
- timestamp: `2026-09-06 20:22:37 +0800`
- modification_version: `V1.1.3`
- type: `architecture / code / experiment`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认纯 point-flow loss、q/wrist 输出保持不变、固定 1538 点/seed 2024、`object_pose_t` 坐标、可微 FK 和 active-only mask 全部按默认方案执行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（新增 Task-local point-flow FK/loss/测试/计划；不覆盖旧 checkpoint、split 或 cache）
- run_id: `cmdecoderv2-pointflow-smoke-20260906-202237`
- run_status: `COMPLETED`（仅完成静态、CPU smoke 和真实 checkpoint forward/loss smoke；未启动长训练）
- conclusion: `SUPPORTED`（工程 point-flow wiring、FK parity、梯度和 active-only gate 通过；科研效果尚未重新训练，仍为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/`；旧 q/wrist-loss decoder run、旧 checkpoint 和 8098/8099 viewer 保持不变。

**原因**

用户要求保留 q/wrist 作为 decoder 输出，但将训练监督改为由 q/wrist 经 Inspire FK 生成的对应表面点流，避免直接 q/wrist loss 与最终几何目标不一致。

**修改**

- [pointflow.py](../../pointflow.py) — 使用同一 URDF、area-weighted surface sampling、seed `2024` 构造固定 1538 点，并实现可微 Inspire FK、SE(3) delta 和 `object_pose_t` 变换；surface buffers 不进入 checkpoint state dict。
- [model.py](../../model.py) — 保留 q/wrist 输出，训练 batch 额外生成预测点和预测点流；可视化 batch 缺少 pose 字段时保持旧输出合同。
- [dataset.py](../../dataset.py) — 提供扰动当前 wrist/object pose 及未来 GT hand points，GT 点流与扰动后的当前状态对齐。
- [runner.py](../../runner.py)、[config.py](../../config.py) — 纯 point-flow Smooth-L1（beta `0.005 m`），q/wrist 只保留诊断指标；5 cm active mask 与 `cm_sample_valid` 继续 gating。
- [test_pointflow.py](../../tests/test_pointflow.py) — 增加 cache parity、可微梯度和 horizon 点坐标变换测试。
- [plan/v1.1.md](../plan/v1.1.md) — 记录 point-flow supervision 合同。

**验证**

- `py_compile`：通过。
- `CUDA_VISIBLE_DEVICES='' ... pytest -q src/task/CmDecoderv2/tests`：`9 passed`。
- differentiable FK 与现有 1538-point cache parity：最大误差约 `0.00014 mm`，梯度 finite。
- 旧 decoder `best.pt` strict load：通过；固定 surface buffers 不破坏旧 checkpoint 兼容。
- 真实 OICM checkpoint + pilot batch（1024 object points）forward/loss smoke：预测点流 `[1,4,1538,3]`、loss finite，active-only supervision ratio 正常。
- 未启动新的长训练，因此没有新的 `metrics.jsonl`、`train.log` 或正式 checkpoint。

**保护边界与回滚**

- 未修改 decoder view/cache、正式 train/val/test split、OICM checkpoint、旧 decoder checkpoint 或 viewer 进程。
- 回滚入口为恢复旧 runner q/wrist loss、移除 point-flow module/model 输出字段，并隔离本次未启动的新训练目录。

## 2026-09-06 19:49:47 +0800 — V1.1.2 停止旧 decoder 训练并启用 5 cm active-only supervision

- activity_id: `cmdecoderv2-active-only-20260906-194947`
- timestamp: `2026-09-06 19:49:47 +0800`
- modification_version: `V1.1.2`
- type: `operation / code / experiment`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户要求停止训练并加入 `active_only`，确保进入 5 cm 范围后才开始 q/wrist 监督；该条沿用此前 5 cm diagnostic 的明确修正方向。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（Task-local active-only 代码/配置/测试/计划增补；不覆盖旧 checkpoint 或正式 split/cache）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `STOPPED`（进程核查时已不存在；未向训练进程发送 kill；旧 run 输出保留）
- last_step: `255900`；last_epoch: `40`
- best_metric: `val/loss=1.3800124928725293`（step `83213`，epoch `13`）
- best_checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- latest_checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt`
- conclusion: `INCONCLUSIVE`（工程 mask smoke 通过；active-only 版本尚未重新长训，不能据此下科研效果结论）
- scope: `src/task/CmDecoderv2/`；8098/8099 viewer 保持运行，正式 train/val/test split、decoder view/cache、OICM checkpoint 和旧 decoder checkpoint 保持不变。

**原因**

旧 decoder 训练的 `data.active_only=false`，且 q/wrist loss 未使用 `cm_sample_valid`；历史指标中约一半窗口帧可能来自无效 Cm。用户要求停止旧训练并只在 5 cm active frame 监督。

**修改**

- [config.py](../../config.py)、[dexplore_rl_v1_1_1.yaml](../../configs/active/dexplore_rl_v1_1_1.yaml)、[dexplore_rl_v1_1_1_smoke.yaml](../../configs/active/dexplore_rl_v1_1_1_smoke.yaml) — `data.active_only=true`。
- [dataset.py](../../dataset.py) — 返回与 `C_t...C_{t+K-1}` 对齐的完整 object-pool 5 cm `active_mask`。
- [runner.py](../../runner.py) — 将 `active_mask` 与冻结 OICM `cm_sample_valid` 相交，q/wrist 三项 loss 和 metrics 只使用有效 horizon，并对剩余 horizon 重新归一化。
- [test_dataset.py](../../tests/test_dataset.py)、[test_model.py](../../tests/test_model.py) — 增加 active mask 和全无效 loss 的回归测试。
- [plan/v1.1.md](../plan/v1.1.md) — 记录本次用户追加批准的 5 cm active-only 训练合同。

**验证**

- `py_compile`：通过。
- `CUDA_VISIBLE_DEVICES='' ... pytest -q src/task/CmDecoderv2/tests`：`7 passed`。
- CPU active-only dataset smoke：pilot windows `428 -> 359`，样本 `active_mask` 与窗口形状一致。
- CPU runner smoke：active mask 与 `cm_sample_valid` 相交、全无效 batch 返回有限零 loss，反向梯度 finite。
- [旧 metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)、[旧 train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log) — 旧无 active-only run 的证据保留。

**保护边界与回滚**

- 未删除或覆盖旧 decoder/OICM checkpoint，未修改 formal split、view/cache、原始数据和可视化进程。
- 回滚入口为恢复 `active_only=false`、移除 `active_mask` loss gate，并隔离本次未启动的新训练 run；旧 run 仍可作为 baseline 使用。

## 2026-09-06 19:31:01 +0800 — V1.1.1 修复 Inspire rollout restart 回调死锁并重启 viewer

- activity_id: `cmdecoderv2-inspire-test-camera-20260906-192900`
- timestamp: `2026-09-06 19:31:01 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic / code / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户报告 rollout 卡死；诊断确认 restart 回调中的同步 Viser property update 与普通锁重入造成 futex deadlock，修复后重启同一 camera test viewer。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（新增 Task-local UI 修复和运行记录；不改正式 split/cache/checkpoint）
- run_id: `cmdecoderv2-inspire-test-vis-20260906-193101`
- run_status: `RUNNING`
- conclusion: `SUPPORTED`（已定位并修复工程死锁；纯 Inspire teacher/rollout 科研效果仍为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/`；正式 MANO-only test index、训练和 checkpoint 选择保持不变。

**文件**

- [visualize_inspire_test.py](../../visualize_inspire_test.py) — 将 Viser 回调锁改为 `threading.RLock`，并避免 restart 回调重复触发 rollout 初始化。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-193101/run_manifest.json) — 修复后 viewer 运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-193101/trajectory.npz) — 修复后 camera test 轨迹输出。

**原因**

旧进程停在 `futex_wait_queue_me`：`on_restart` 持有普通 `threading.Lock` 时设置 `mode.value`，Viser 同步触发 `on_mode_change`，后者再次获取同一把锁。该死锁与 decoder 推理或数据无关。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_inspire_test.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- 修复后 viewer 已在 `http://localhost:8099` 重新启动，序列仍为 `s1/camera_takepicture_3_Retake`，Viser HTTP 200。
- `git diff --check` 与 activity link audit：通过。

**保护边界与回滚**

- 旧死锁运行 `cmdecoderv2-inspire-test-vis-20260906-192300` 输出保留；未修改 raw data、正式 index、cache、checkpoint 或训练配置。
- 回滚入口为停止 8099 viewer、隔离本次输出，或单独回退 RLock/restart 回调改动。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 19:23:00 +0800 — V1.1.1 切换纯 Inspire viewer 到右手接触轨迹并修复 teacher 尾帧边界

- activity_id: `cmdecoderv2-inspire-test-camera-20260906-191200`
- timestamp: `2026-09-06 19:23:00 +0800`
- modification_version: `V1.1.1`
- type: `code / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户要求换一条轨迹可视化；根据右手几何接触诊断切换到 `s1/camera_takepicture_3_Retake`，并修复播放到不完整 Cm window 尾帧时的 teacher 边界错误。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（本次新增 viewer 边界修复和运行记录；不改正式 split/cache/checkpoint）
- run_id: `cmdecoderv2-inspire-test-vis-20260906-192300`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（仅为右手接触轨迹的定性 viewer；不作 decoder 科研效果结论）
- scope: `src/task/CmDecoderv2/`；正式 MANO-only test index、训练和 checkpoint 选择保持不变。

**文件**

- [visualize_inspire_test.py](../../visualize_inspire_test.py) — teacher 模式在最后不足 `K+1` 帧时停止预测、保留 GT playback，避免尾帧 `IndexError`。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-192300/run_manifest.json) — 切换后的 camera test viewer 清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-192300/trajectory.npz) — `s1/camera_takepicture_3_Retake` 的 Inspire GT/object 和 teacher/rollout 槽位。

**原因**

`s1/alarmclock_offhand_1` 的右手 Inspire 几何距离中位数约 `217.7 mm`，仅 `33/271` 帧小于 `2 cm`，不适合评估右手抓取。`s1/camera_takepicture_3_Retake` 有 `253/352` 帧小于 `2 cm`（约 `71.9%`），因此作为本次可视化轨迹。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_inspire_test.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- 新 viewer 已在 `http://localhost:8099` 启动，序列 `s1/camera_takepicture_3_Retake`、352 帧；Viser HTTP 200。
- `git diff --check` 与 activity link audit：通过。

**保护边界与回滚**

- 旧 offhand run 输出保留，不覆盖；未修改正式 index、raw data、cache、checkpoint 或训练配置。
- 回滚入口为停止当前 8099 viewer、恢复旧 run 目录；代码边界修复可单独回退。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 19:04:23 +0800 — V1.1.1 新增纯 Inspire held-out test teacher/rollout 诊断 viewer

- activity_id: `cmdecoderv2-inspire-test-vis-20260906-implementation`
- timestamp: `2026-09-06 19:04:23 +0800`
- modification_version: `V1.1.1`
- type: `architecture / code / data / experiment / operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认使用原 test parent 的 Dexplore RL-Inspire hand/object/q/wrist 完整轨迹，teacher 每帧 GT state，rollout 只在 handoff 使用一次 GT state 后递归；该分支仅作纯 Inspire 诊断。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（新增诊断 viewer 和 plan/activity 修改；不改正式 split/cache/checkpoint）
- run_id: `cmdecoderv2-inspire-test-vis-20260906-190423`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（工程 wiring smoke 通过；该 viewer 不产生正式 test 科研结论）
- scope: `src/task/CmDecoderv2/`；原正式 `dexplore_rl_v1_1/index.json` 的 `test=mano` 合同、训练和 checkpoint 选择均保持不变。

**文件**

- [visualize_inspire_test.py](../../visualize_inspire_test.py) — 新增独立 viewer，按需从 raw `interaction_hand_inspire.pt`、对应 Dexplore object pose、原 parent object surface 和 Inspire URDF FK 构造 held-out Inspire test source；支持 teacherforced/rollout、GT/pred/both 和点大小滑块。
- [plan/v1.1.md](../plan/v1.1.md) — 记录用户追加批准的 Inspire-test diagnostic 分支及其不改变正式 split 的边界。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-190423/run_manifest.json) — 当前 GPU3 诊断运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-190423/trajectory.npz) — 当前序列的 Inspire GT、Dexplore object 和 teacher/rollout 槽位。

**原因**

用户希望区分“纯 Inspire decoder 是否能学好”和 MANO→Inspire 跨 embodiment 效果。正式 decoder view 的 125 条 test entry 只标记 MANO，但每个 held-out parent 都存在对应 raw Dexplore RL-Inspire tensor；新 viewer 使用这些 raw test trajectories 在线构造 source，不把它们写回正式 index，也不用于训练或选 checkpoint。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_inspire_test.py`：通过。
- GPU3 real-checkpoint smoke：`s1/alarmclock_offhand_1` 的 q `[271,18]`、hand `[271,1538,3]`、object `[271,4096,3]`、wrist `[271,4,4]` 全部构造成功；teacher h=1 和 rollout h=1 均输出 finite Inspire points `[1538,3]`。
- viewer 已在 `http://localhost:8099` 启动，HTTP/Viser server 正常；formal MANO viewer 8098 未改动。
- `git diff --check`：通过；activity link audit：通过（5 个本地链接可导航）。

**保护边界与回滚**

- 未生成或修改新的全量 OICM/decoder cache；未修改训练数据 index、模型配置、checkpoint、原始 Dexplore/GRAB 数据。
- 回滚入口为移除 [visualize_inspire_test.py](../../visualize_inspire_test.py)、诊断 plan 增补和本次输出目录；不影响正式 MANO viewer 和 decoder 训练。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 17:28:42 +0800 — V1.1.1 修复 handoff 帧 Viser 回调崩溃并重启 GPU3 viewer

- activity_id: `cmdecoderv2-mano-grab-vis-20260906-172800`
- timestamp: `2026-09-06 17:28:42 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic / code / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户报告 viewer 中断；根据 traceback 修复 Task-local UI 对 handoff 初始状态的错误处理并重新启动同一 viewer。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 CmDecoder、ObjectInteractionCm、CmDecoderv2 及训练改动）
- run_id: `cmdecoderv2-mano-grab-vis-20260906-172842`
- run_status: `RUNNING`
- conclusion: `SUPPORTED`（已定位并修复 handoff UI 的工程错误；MANO→Inspire 科研效果仍为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/`；未修改正式 decoder 训练、GPU0/1/2 训练进程、数据 cache、split 或 checkpoint。

**文件**

- [visualize_mano.py](../../visualize_mano.py) — handoff 帧的 Inspire 初始状态 `cm_valid=None` 时显示 initialized 文本，不再执行 `len(None)`。
- `docs/current_versions.yaml` — 保留既有 `CmDecoderv2: V1.1.1` 指针；本次不改变版本值。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172842/run_manifest.json) — 修复后 GPU3 viewer 的运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172842/trajectory.npz) — 修复后运行的 GT 和 rollout 槽位。

**原因**

旧运行在切换 rollout 后推进 frame 时崩溃。traceback 显示 handoff 初始状态的 `cm_valid=None` 被 UI 当成数组执行 `len(valid)`；这是状态语义正常、显示分支错误，不是模型或 parent cache 数据错误。

**验证**

- 原始 traceback 已保存于旧 Viser session：`TypeError: len() of unsized object`，位置为 `visualize_mano.py` 的 Cm-valid 文本分支。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_mano.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- 修复后 GPU3 viewer 已在 `http://localhost:8098` 启动，HTTP 200；新的 run manifest 已生成，formal decoder training GPU0/1/2 未被中断。
- `git diff --check` 与 activity link audit：通过。

**保护边界与回滚**

- 旧崩溃运行 `cmdecoderv2-mano-grab-vis-20260906-172219` 的输出保留作诊断证据；修复后运行使用新的 run_id，不覆盖旧输出。
- 回滚入口为移除本次 handoff 分支改动或隔离新 viewer 输出；不删除原始 GRAB parent cache、decoder checkpoint 或正式训练输出。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 17:22:19 +0800 — V1.1.1 新增 MANO/GRAB-source Viser 定性 viewer 并启动 GPU3

- activity_id: `cmdecoderv2-mano-grab-vis-20260906-172000`
- timestamp: `2026-09-06 17:22:19 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确确认最终可视化必须使用原始 GRAB MANO 手和物体轨迹；Dexplore RL 只用于 decoder 训练；允许从任意帧以 q=0 + MANO wrist 切换到 rollout。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 CmDecoder、ObjectInteractionCm、CmDecoderv2 及训练改动）
- run_id: `cmdecoderv2-mano-grab-vis-20260906-172219`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（仅确认数据来源、坐标和 viewer wiring；没有 paired Inspire test GT，不作定量或科学效果结论）
- scope: `src/task/CmDecoderv2/`；未修改正式 decoder 训练、GPU0/1/2 训练进程、Dexplore/GRAB 原始数据、已有 RL-Inspire 诊断 viewer。

**文件**

- [visualize_mano.py](../../visualize_mano.py) — 新增最终定性 viewer：GT playback 渲染原始 GRAB parent cache 的 MANO hand/object；rollout 在当前帧以零 finger q 和 MANO `hand_root_pose_world` 初始化，使用 MANO source Cm window 递归预测 Inspire；GT/pred/both 和点大小滑块均可切换。
- `docs/current_versions.yaml` — 保留既有 `CmDecoderv2: V1.1.1` 指针；本次不改变版本值。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172219/run_manifest.json) — 当前 GPU3 Viser 运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172219/trajectory.npz) — 原始 MANO/GRAB GT、物体轨迹和当前已生成的 rollout 槽位。

**原因**

之前的 `visualize_teacherforce.py` 使用 Dexplore RL-Inspire validation GT，因此不能作为最终 MANO→Inspire 测试。新 viewer 将 Dexplore RL 限定为 decoder 训练来源，最终可视化只使用原始 GRAB parent cache；没有把对应 Inspire 轨迹作为测试标签，避免把不同抓取方式混在比较中。

**验证**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize_mano --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence s1/alarmclock_offhand_1 --port 8098 --fps 8 --activity-id cmdecoderv2-mano-grab-vis-20260906-172000`
- checkpoint SHA256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`。
- viewer: `http://localhost:8098`；GPU3 进程正在运行，formal decoder training 的 GPU0/1/2 进程未停止。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_mano.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- GPU3 real-checkpoint smoke：加载 `s1/alarmclock_offhand_1` 原始 parent cache，GT shapes 为 hand `[271,1538,3]` / object `[271,4096,3]`；rollout handoff frame 0 后递归到 frame 1，预测 Inspire points `[1538,3]`、q `[6]`、wrist `[4,4]`，输出 finite。
- `curl http://localhost:8098/`：HTTP 200；run manifest 与 trajectory 已生成。
- `git diff --check`：通过。

**保护边界与回滚**

- 未修改旧 RL-Inspire teacher-forcing viewer；未修改任何数据 cache、split、checkpoint 或正式训练配置。
- 回滚入口为删除/隔离新增 [visualize_mano.py](../../visualize_mano.py) 和本次 viewer 输出目录；不删除原始 GRAB parent cache 或 decoder checkpoint。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 16:42:31 +0800 — V1.1.1 object pose 非接触运动因果诊断

- activity_id: `ACT-20260906-164231-CMDECODERV2-OBJECT-MOTION-CAUSALITY-DIAGNOSTIC`
- timestamp: `2026-09-06 16:42:31 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户追问“没有接触为什么物体会动”；只读核对 Dexplore native tensor 字段定义、cache builder 和 frame 200 轨迹，不修改数据或训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（确认数据没有“无接触冻结物体”的物理约束；是否按右手接触重构/裁剪物体轨迹仍待研究决定）
- scope: `src/task/CmDecoderv2/`、ObjectInteractionCm cache builder 和 Dexplore export schema 的只读诊断；未修改研究变量。

**原因**

确认 frame 200 无右手几何接触时 object pose 仍变化的直接数据来源，并区分 object trajectory、contact label 与 viewer 渲染行为。

**关键证据**

- Dexplore native tensor 的 `198:201` 是 object position、`201:205` 是 object quaternion、`205` 是 object contact flag；cache builder 分别读取 pose 和 flag，没有用 contact flag 对 pose 做冻结或门控。
- 因此 `contact=0` 只表示该帧的 object contact label 为 0，不代表 `T_object[t+1] == T_object[t]`；数据管线不承诺无接触时物体静止。
- `s1/banana_lift` frame 200 的 native contact flag 为 `0`，右手到物体最近距离约 `32.3 mm`，但 object frame 200→201 仍有约 `1.2 mm` translation 和 `2.57°` rotation；这是原始 object pose trajectory 中的轻微抖动/轨迹变化。
- Dexplore 文档明确指出 object motion 可能由未被当前 Inspire 右手显示的其他手/身体接触驱动；当前 tensor 的 object flag 是 object-level contact，不是本 viewer 右手的独立因果门控。
- 更大的 object motion 出现在 frame 220–230，此前 frame 220–226 仍存在 `<2 cm` 接触，随后 object z 约从 `1.129 m` 上升到 `1.335 m`，说明该段是接触后的真实 object trajectory，而非 Viser 自动生成。

**证据入口**

- [Dexplore export schema](../../../../../../dexplore/data_processing/README_GRAB_INSPIRE_EXPORT.md)
- [cache builder](../../tools/data/build_dexplore_view.py)
- [ObjectInteractionCm RL cache builder](../../../ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读核对 native tensor 字段、builder 字段映射、contact flag、world-frame object pose 和 frame 200 hand-object 距离；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:38:21 +0800 — V1.1.1 Viser 世界坐标与 frame 200 物体运动诊断

- activity_id: `ACT-20260906-163821-CMDECODERV2-VISER-COORD-FRAME200-DIAGNOSTIC`
- timestamp: `2026-09-06 16:38:21 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 Viser 使用的坐标系，并指出 frame 200 GT 未接触但物体仍在动；只读核对 viewer 数组索引、world/object pose 和 frame 200 的 hand-object 距离。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（当前 world-frame 物体运动与数组一致；是否要改成 object-centric viewer 仍待用户决定）
- scope: `src/task/CmDecoderv2/` 与 validation geometry 的只读诊断；未修改 viewer、数据或训练。

**原因**

需要确认 frame 200 的物体运动是否来自 Viser 坐标/索引错误，还是来自 Dexplore geometry 中记录的 object pose trajectory。

**关键证据**

- 当前 Viser 使用 `obj_points_world.npy`、`hand_points_world.npy` 和 `wrist_pose_world.npy` 的 Dexplore world frame（manifest 标记为 `dexplore_native_object_pose_world`），没有把所有帧变换到固定 object-centric frame。
- teacherforced frame `t` 显示：object 使用 frame `t+1`，GT current 使用 frame `t`，GT next 使用 frame `t+1`；因此 frame 200 的场景物体是 object frame 201，绿色 GT next 也是 hand frame 201。
- frame 200 的 hand→object 最近距离约 `32.3 mm`，`<20 mm` 接触比例为 `0%`，所以此帧确实没有 2 cm 级物理接触；但此前 frame 175–195 有多段 `<2 cm` 接触，frame 200 是一次短暂分离后的状态。
- frame 200→201 物体质心移动约 `1.4 mm`，object pose translation 约 `1.2 mm`，旋转约 `2.57°`；因此 world-frame 播放时物体确实会轻微移动，主要是该帧的 object pose/rotation trajectory，不是 viewer 凭空移动。
- 更明显的 object motion 出现在 frame 220–230：此区间 hand 在 frame 220–226 仍有 `<2 cm` 接触，之后 object pose z 从约 `1.129 m` 上升到 `1.335 m`；这与前一段接触后的物体运动一致。

**证据入口**

- [teacherforce viewer](../../visualize_teacherforce.py)
- [geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读比较 world-frame object/hand 数组、object pose 帧间增量和 frame 200 的近邻距离；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:29:24 +0800 — V1.1.1 validation GT 预接触运动诊断

- activity_id: `ACT-20260906-162924-CMDECODERV2-GT-PRECONTACT-DIAGNOSTIC`
- timestamp: `2026-09-06 16:29:24 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户反馈 validation Viser 中 GT 在接触物体前已经运动；只读检查 RL q、wrist、hand/object geometry、5 cm candidate mask 和 source contact flag，不修改 viewer、数据或训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（数值上 GT 与 cache/FK 自洽，观察到的是动作的预接触阶段；尚未决定是否新增 grasp-only crop）
- scope: `src/task/CmDecoderv2/` 与 `data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/sequences/val/s1_banana_lift/` 的只读诊断；未改变研究变量。

**原因**

需要确认“接触前运动”是数据包含 reach/approach 动作，还是 q/wrist、hand geometry、object pose 或 frame index 错位。

**关键证据**

- `s1/banana_lift` 为 30 Hz、578 帧；物体基本静止，手从 frame 0 就在接近，hand-object 最近距离从约 `1316 mm` 降到 frame 50 的 `72 mm`。
- 按生成的 5 cm candidate mask，首个 active frame 为 `53`；按实际最近距离，首个 `<20 mm` 接触帧为 `54`；Dexplore source contact flag 首个为 `56`。因此前约 `1.8 s` 是正常 reach/approach 段，不是 hand 静止等待接触。
- frame 0→1 的 GT 手指 q 变化约 `24.18°`（6 维 q 的 L2），腕部平移约 `40.76 mm`、旋转约 `3.39°`；这与“动作一开始就运动”一致。
- `geometry/manifest.json` 标明 RL hand geometry 由 Dexplore native q 经 Inspire URDF FK 生成，decoder view 的 wrist pose 也由同一 q/FK 生成；当前证据未发现 GT frame index 或 coordinate-frame 错位。
- source flag 与几何 5 cm mask 不完全同帧：source flag 从 frame 56 开始，几何 mask 从 frame 53 开始；这是 5 cm candidate 与 Dexplore contact label 的语义差异，不等于 GT 错位。

**证据入口**

- [geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)
- [decoder view manifest](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/sequences/val/s1_banana_lift/manifest.json)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读计算 hand/object 最近距离、候选 active、source contact flag、GT q/wrist 帧间增量和 object/hand centroid 轨迹；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:20:49 +0800 — V1.1.1 Viser 点云显示与 teacherforced/rollout 切换

- activity_id: `ACT-20260906-162100-CMDECODERV2-TEACHERFORCE-TOGGLE`
- timestamp: `2026-09-06 16:20:49 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户确认暂不使用 mesh，要求点云大小滑块、GT/pred/both 显示切换以及 teacherforced/rollout 可随时切换；采用已说明的 GT handoff 语义。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、旧 rollout 和用户改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser server 使用 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（交互可视化 wiring 已通过 smoke；不构成 decoder 科研效果结论）
- scope: `src/task/CmDecoderv2/`（更新 `visualize_teacherforce.py`）与独立 teacher-forcing Viser 输出目录；未修改训练变量、checkpoint 或正式训练。

**原因**

需要在同一 validation sequence 中直接比较 teacher forcing 与 state-feedback rollout，并能按帧选择 GT、预测或两者，同时调整点云可见性。

**实现与运行**

- Viser 新增 `Execution mode`：`teacherforced` / `rollout`；切到 rollout 时以当前帧 GT state 初始化，之后按预测 state 递归，跳帧会从该帧 GT state 重新 handoff；切回 teacherforced 时自动回到 GT state。
- 新增 `Hand display`：`GT` / `pred` / `both`；新增 `Point size (m)` 滑块（`0.001–0.020 m`）。当前仍只渲染点云，不加载 mesh。
- [teacherforce.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)、[run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/run_manifest.json)。Viser 地址：`http://localhost:8097`。

**验证**

- 8-frame Viser smoke 和无服务器 rollout handoff smoke 均通过；完整 validation sequence `s1/banana_lift` 574 帧预计算完成。
- `python3 -m py_compile src/task/CmDecoderv2/visualize_teacherforce.py`、`git diff --check` 和 Viser HTTP 探活通过。
- 正式 decoder 训练 GPU0/1/2 未停止；本次可视化只使用 GPU3。

## 2026-09-06 15:58:41 +0800 — V1.1.1 RL-Inspire validation teacher-forcing Viser 可视化

- activity_id: `ACT-20260906-161100-CMDECODERV2-TEACHERFORCE-VISER`
- timestamp: `2026-09-06 15:58:41 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户要求先不看 closed-loop rollout，改用 teacher forcing 检查 decoder 表现并使用 Viser 可视化；默认使用 RL-Inspire validation，因为 MANO test 没有对应 Inspire GT。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、rollout 输出和用户改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-155735`
- run_status: `RUNNING`（Viser server 仍在 GPU3 / port 8097 提供交互式查看）
- conclusion: `INCONCLUSIVE`（teacher-forced wiring 和定性/诊断指标已生成；不等同于最终科研结论）
- scope: `src/task/CmDecoderv2/`（新增 `visualize_teacherforce.py`）与独立 validation teacher-forcing 输出目录；不修改训练变量、checkpoint 或 MANO rollout。

**原因**

需要隔离 closed-loop state feedback drift：每一帧使用 RL-Inspire validation 的真实当前 q/wrist 和真实 Cm window，只预测 h=1，不把预测状态反馈到下一帧；Viser 同时显示 GT 当前手、GT 下一帧手和 decoder 预测。

**实现与运行**

- 新增 [visualize_teacherforce.py](../../visualize_teacherforce.py)，只接受 `variant=inspire_rl` 的 validation entry，使用 frozen OICM + CmDecoderv2 checkpoint，输出 teacher-forced h=1 轨迹并启动 Viser。
- 使用 GPU3（进程内 `cuda:0`），序列 `s1/banana_lift`，574 个完整 window/frame；Viser 地址为 `http://localhost:8097`。
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-155735/run_manifest.json)、[teacherforce.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-155735/teacherforce.npz)。

**诊断指标**

- 全部帧：q MAE `1.367°`，wrist translation `13.98 mm`，wrist rotation `3.38°`，hand nearest-neighbor EPE `8.40 mm`。
- 当前 Cm 有效帧（379/574，`66.0%`）：q MAE `1.289°`，wrist translation `5.15 mm`，wrist rotation `2.47°`，hand EPE `5.50 mm`。
- 当前 Cm 无效帧：wrist translation `31.14 mm`，hand EPE `14.03 mm`；该差异是 teacher-forcing 诊断中必须单独报告的 mask 分层结果。

**验证**

- 8-frame smoke 已完成；checkpoint load、GT state/window 对齐、forward 和 Viser HTTP 均正常。
- `python3 -m py_compile src/task/CmDecoderv2/visualize_teacherforce.py`、`git diff --check` 通过。
- 正式 decoder 训练 GPU0/1/2 未停止；本次可视化仅使用 GPU3。当前 run 仍保持 RUNNING，未覆盖历史 rollout 或 checkpoint。

## 2026-09-06 15:50:00 +0800 — V1.1.1 核对 5 cm 帧屏蔽与 decoder loss mask

- activity_id: `ACT-20260906-155000-CMDECODERV2-5CM-MASK-DIAGNOSTIC`
- timestamp: `2026-09-06 15:50:00 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问训练是否按 5 cm 屏蔽帧；本次只读核对数据集 active mask、OICM 交互半径和 CmDecoderv2 loss，不修改代码、配置或运行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、CmDecoderv2 实现和用户改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（已确认 5 cm 机制的作用层级；当前 decoder 未按该 mask 重新训练）
- scope: `src/task/CmDecoderv2/` 与其引用的 ObjectInteractionCm 配置/代码；未改变研究变量

**原因**

需要区分 OICM 提取 Cm 时的 5 cm 几何筛选、OICM loss 的 sample mask，以及 CmDecoderv2 自身是否按 5 cm 有效帧筛选和屏蔽 decoder loss。

**核对结果**

- ObjectInteractionCm 配置的 `interaction_radius_m=0.05`、`active_only=true`：数据集会根据预计算的 `obj_candidate_mask_5cm` 丢弃当前帧完全没有 5 cm 候选的行；运行时局部交互边也要求距离不超过 5 cm，`sample_valid` 再表示采样到的 object pool 是否仍有有效交互。OICM runner 会用 `sample_valid` 屏蔽自身的 object/hand flow loss。
- CmDecoderv2 配置的 `data.active_only=false`：decoder window 不会因 5 cm candidate mask 在数据集层被丢弃。
- CmDecoderv2 runner 只记录 `cm/valid_frame_ratio`，q、wrist translation、wrist rotation 三项 loss 没有用 `cm_sample_valid` 做 mask。因此当前 decoder 训练并不是“按 5 cm 有效帧训练”；无效 Cm 仍进入 loss。

**证据入口**

- [CmDecoderv2 config](../../config.py)、[CmDecoderv2 dataset](../../dataset.py)、[CmDecoderv2 runner](../../runner.py)
- [ObjectInteractionCm RL config](../../../../../src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml)、[ObjectInteractionCm dataset](../../../../../src/task/ObjectInteractionCm/dataset.py)、[ObjectInteractionCm runner](../../../../../src/task/ObjectInteractionCm/runner.py)
- [formal training metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)

**验证**

- 只读检查配置、数据集行筛选、5 cm 交互半径、`sample_valid` 传播和 decoder loss 计算；未停止 GPU0/1/2 正式训练。

## 2026-09-06 15:33:31 +0800 — V1.1.1 完整 rollout 的“乱动”原因诊断

- activity_id: `ACT-20260906-153331-CMDECODERV2-ROLLOUT-DIAGNOSTIC`
- timestamp: `2026-09-06 15:33:31 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求分析完整 rollout 中 Inspire 手持续乱动的原因；本次仅做只读轨迹、代码、坐标和训练指标核对，不修改训练、checkpoint、数据或可视化代码。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、CmDecoderv2 实现和用户改动）
- run_id: `cmdecoderv2-mano-viz-20260906-151601`；关联正式训练 run：`cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `COMPLETED`（诊断）；正式训练保持运行状态，未被本次检查中断
- conclusion: `INCONCLUSIVE`（已定位实现与坐标问题，但尚未按修复方案重新训练/复测，不能据此下科研效果结论）
- scope: `src/task/CmDecoderv2/` 与独立 rollout/训练产物；未修改研究变量

**原因**

用户反馈完整 rollout 中 Inspire 手持续乱动，需要区分模型本身、无效 Cm 处理、初始化坐标和视频完整性四类因素；本次只做证据核对，不改变正式训练运行。

**关键证据**

- 可视化 rollout 共 267 steps，MP4 完整；`cm_sample_valid` 为 `[267,4]`，仅 504/1068 个 window-frame 位置有效。当前帧在约 0–90、220–266 均无效，但 visualizer 仍无条件执行 `finger += q_delta`、`wrist = wrist @ delta`，因此 dummy Cm 产生的非零预测会持续累积漂移。前 0–90 帧的每步 q 增量中位数约 `0.00965 rad`，腕部平移中位数约 `5.61 mm`；到第 219 帧 q 范数约 `1.548`、腕部相对初始位移约 `0.944 m`。
- CmDecoderv2 runner 当前只记录 `cm/valid_frame_ratio`，没有用 `cm_sample_valid` mask q/translation/rotation loss。正式训练已完成 epoch 的 valid-frame ratio 约 `0.51816`（train）/`0.52366`（val），约 48% 的 frame supervision 来自无效/dummy Cm；因此当前 best checkpoint 不能视作已按有效交互帧严格训练的最终模型。
- 本次 `mano_wrist` 初始化直接读取 parent GRAB cache 的 `hand_root_pose_world.npy`，但 decoder 测试 geometry view 的 object pose 与 parent cache 的 object pose 存在共同平移偏差，frame 0 约为 `[-39.7, -583.3, +17.6] mm`（旋转基本一致）。直接把 parent-world MANO wrist 当作 view-world Inspire base 会造成初始空间错位；应通过 object pose 关系先变换到 view world。
- 完整 MP4 [rollout](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/rollout.mp4) 已由 `ffprobe` 验证为 267 帧、30 FPS、8.9 秒；此前提前结束的版本是显式 `--max-steps 120`，不是当前编码或接触阶段截断。

**证据入口**

- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/trajectory.npz)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/run_manifest.json)
- [visualize.py](../../visualize.py)、[runner.py](../../runner.py)、[model.py](../../model.py)
- [formal training metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)

**建议的修复顺序（本次未执行）**

1. 用 `T_view_obj @ inv(T_parent_obj) @ T_mano_wrist` 把 MANO wrist 初始化转换到 decoder view world，并确认 MANO root 与 Inspire `hand_base_link` 的固定坐标约定。
2. rollout 只在当前帧 `cm_sample_valid[0,0]` 为真时应用 h=1 action；无效帧明确保持状态，不能用 window 内任意未来有效帧替代当前帧判断。
3. 在 CmDecoderv2 训练中对无效当前帧 mask/drop q、translation、rotation supervision，再重新训练并复测；当前正式 run 和 best.pt 仅作为诊断基线保留。

**验证**

- 只读检查 `trajectory.npz` 的有效帧掩码、状态累积和模型输出范数；对 `visualize.py`、`runner.py`、数据 schema 和 object pose 做源码/数值对齐。
- 本条记录写入后运行 `audit_diff.py --check-links` 与 `git diff --check`；本次未停止正式 GPU0/1/2 训练，也未覆盖旧输出。

## 2026-09-06 15:19:06 +0800 — V1.1.1 修正 MANO wrist 初始化并完成完整 sequence 可视化

- activity_id: `ACT-20260906-151000-CMDECODERV2-MANO-VIZ-MANO-WRIST`
- timestamp: `2026-09-06 15:19:06 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户确认“用 MANO wrist 初始化、Inspire 手指 q 保持 0”，并要求完整跑完接触过程；本次不改变训练变量。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、旧可视化和用户改动）
- run_id: `cmdecoderv2-mano-viz-20260906-151601`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（仅为 MANO-only qualitative rollout；无 paired Inspire test GT，不构成定量或科研效果结论）
- scope: `src/task/CmDecoderv2/visualize.py` 与独立 `outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/` 产物；正式训练 GPU0/1/2 未停止
- checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- checkpoint_sha256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`

**文件**

- `src/task/CmDecoderv2/` — 本 Task 现有实现、配置、数据工具、测试和文档均保留；本次实际代码改动集中在 `visualize.py`，其余路径仅由 activity scope 归档。
- [visualize.py](../../visualize.py) — 默认新增 `mano_wrist` 初始化模式；从 MANO parent cache 读取右手 `hand_root_pose_world.npy`，初始化 Inspire wrist；保留 `manual` 模式和 q=0 手指初始化；MP4 改为 ffmpeg 编码并校验文件大小。
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/run_manifest.json)、[MP4](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/rollout.mp4)、[trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/trajectory.npz)、[PNG frames](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/frames/)

**原因**

旧 rollout 显式限制为 `120` steps，而 `s1/alarmclock_offhand_1` 的完整可用长度为 `frame_count-K=267`，因此视频在接触阶段提前结束；旧默认 wrist 也没有对齐 MANO source。按用户确认，新的 rollout 使用第 0 帧 MANO wrist 世界位姿，手指 q 保持全 0，并跑完整 267 steps。

**验证**

- 命令：`CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence-index 0 --max-steps 10000 --initial-state mano_wrist --render-every 1 --write-mp4 --output-root outputs/cmdecoderv2 --activity-id ACT-20260906-151000-CMDECODERV2-MANO-VIZ-MANO-WRIST`
- `py_compile src/task/CmDecoderv2/visualize.py`、`git diff --check`：通过。
- sequence `s1/alarmclock_offhand_1`：267 steps；MP4 经 `ffprobe` 验证为 H.264、267 帧、30 FPS、8.9 秒；checkpoint SHA 在运行前后保持一致。
- 旧正式 decoder 训练 torchrun 仍在 GPU0/1/2 运行；本次只使用 GPU3。
- 首帧、133 帧和末帧已抽查；画面仅作人工观察，MANO→Inspire 的定性质量仍需用户结合完整视频评估。

## 2026-09-06 15:06:22 +0800 — V1.1.1 使用当前 best.pt 完成 GPU3 MANO 定性 rollout

- activity_id: `ACT-20260906-150300-CMDECODERV2-MANO-VIZ`
- timestamp: `2026-09-06 15:06:22 +0800`
- modification_version: `V1.1.1`
- type: `experiment / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户要求“拿 GPU3 用目前的 best.pt 来可视化”；使用独立输出目录，不触碰 0/1/2 正式训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cmdecoderv2-mano-viz-20260906-150326`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（PNG/MP4 仅供人工定性观察，不含 paired Inspire test GT，不作定量或科研效果结论）
- scope: `src/task/CmDecoderv2/` 的可视化入口与独立 `outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/` 产物
- device: `CUDA_VISIBLE_DEVICES=3`（进程内 `cuda:0`）；正式 decoder 训练 GPU 0/1/2 保持运行
- checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- checkpoint_sha256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`

**原因**

在不停止正式三卡训练的情况下，用当前 decoder best checkpoint 对 MANO-only test sequence 做 receding-horizon `h=1` 定性 rollout，供人工检查 MANO source 与 Inspire predicted 的空间关系。

**命令与产物**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence-index 0 --max-steps 120 --render-every 1 --write-mp4 --output-root outputs/cmdecoderv2 --activity-id ACT-20260906-150300-CMDECODERV2-MANO-VIZ`
- sequence: `s1/alarmclock_offhand_1`；120 steps；初始 Inspire state 由脚本默认参数显式记录。
- [运行目录](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/run_manifest.json)
- [MP4 rollout](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/rollout.mp4)
- [PNG frames](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/frames/)
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/trajectory.npz)

**验证**

- 可视化进程在 GPU3 正常完成；manifest 记录 `paired_inspire_test_gt=false`，未读取配对 Inspire GT。
- 120 张 PNG 已生成；最初的 imageio MP4 写出因 `write() got an unexpected keyword argument 'fps'` 失败，随后使用 ffmpeg 将同一批 PNG 转换为有效 H.264 MP4（120 帧、30 FPS、4 秒），并通过 `ffprobe` 验证。
- `best.pt` 在可视化前后 SHA256 保持一致；正式训练进程仍在 GPU0/1/2 运行，未被中断。
- 首帧/中段/末帧已做肉眼抽查；这是人工观察入口，不把当前画面直接解释为 decoder 科研效果成立。

## 2026-09-06 15:01:07 +0800 — V1.1.1 训练中期 loss/validation 状态诊断

- activity_id: `ACT-20260906-150107-CMDECODERV2-LOSS-STATUS-DIAGNOSTIC`
- timestamp: `2026-09-06 15:01:07 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练状态；只读检查进程、metrics、validation 和 checkpoint，不中断或调整训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- last_step: `156100`（epoch 25 进行中）
- last_completed_epoch: `24`
- best_metric: `val/loss=1.3800124928725293`（epoch 13 / step 83213）
- conclusion: `INCONCLUSIVE`（train loss 仍下降，但 validation 在 epoch 13 后平台化并波动，训练尚未结束）
- scope: `src/task/CmDecoderv2/` 与正式 run 的只读 loss/validation/进程/checkpoint 诊断

**原因**

需要区分优化是否仍在下降与泛化 validation 是否继续改善，避免把持续下降的 train loss 误判为模型已收敛。

**验证**

- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)：已完成 epoch 1–24；train loss 从 `2.059890` 降到 `1.159078`，仍在下降。
- validation 最佳为 epoch 13 的 `1.380012`；epoch 14–24 在约 `1.3867–1.4349` 间波动，epoch 24 为 `1.399368`，尚未恢复 epoch 13 最佳。
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt) 为 epoch 13 / step 83213；[latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt) 为 epoch 24 / step 153624。
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)：torchrun 与 3 个 rank 仍在运行；GPU 0/1/2 利用率约 `68%/57%/71%`。
- 当前判断：训练过程没有异常退出，但 validation 已出现早期平台/波动；是否最终收敛需继续观察后续 epochs，当前科研结论仍为 `INCONCLUSIVE`。

## 2026-09-06 12:54:47 +0800 — V1.1.1 正式训练早期 loss 趋势诊断

- activity_id: `ACT-20260906-125447-CMDECODERV2-LOSS-TREND-DIAGNOSTIC`
- timestamp: `2026-09-06 12:54:47 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练状态及 loss 是否持续下降；只读检查进程、metrics 和 checkpoint，不中断或调整训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- last_step: `49300`（查询时 epoch 8 进行中）
- last_completed_epoch: `7`
- best_metric: `val/loss=1.424412033150659`（epoch 7 / step 44807）
- conclusion: `INCONCLUSIVE`（前 7 个 epoch 的 train/val loss 持续下降，但训练仅完成约 15%，尚不能判断最终收敛）
- scope: `src/task/CmDecoderv2/` 与正式 run 的只读 loss/进程/checkpoint 诊断

**原因**

step loss 受 batch 组成与扰动影响明显，需使用完整 epoch 的 train aggregate 和 validation 指标判断趋势，并拆分 q、wrist translation、wrist rotation 确认总 loss 的下降来源。

**验证**

- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)：epoch 1→7 的 train loss `2.059890→1.507454`，val loss `1.785547→1.424412`，两者在每个已完成 epoch 均下降。
- 总 loss 的主要下降来自 wrist translation：train `1.794571→1.249410`，val `1.623184→1.263725`；q 与 wrist rotation 分项目前主要呈平台/小幅改善。
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt) 与 [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt) 均为 epoch 7 / step 44807，`best_metric=1.424412033150659`。
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)：torchrun 与 3 个 rank 仍在运行；GPU 0/1/2 查询时利用率约 `69%/57%/69%`，无退出迹象。
- 当前 step loss 在约 `0.87–2.63` 间抖动是 batch-level 波动，不能据此要求逐 step 单调下降；科研结论需等待更多 validation 和最终 MANO→Inspire 可视化。

## 2026-09-06 11:57:01 +0800 — V1.1.1 使用冻结 OICM best.pt 启动 CmDecoderv2 三卡正式训练

- activity_id: `ACT-20260906-115701-CMDECODERV2-FORMAL-TRAIN-STARTED`
- timestamp: `2026-09-06 11:57:01 +0800`
- modification_version: `V1.1.1`
- type: `operation / experiment`
- change_level: `L3 + L2`（按已定稿 v1.1 plan 运行；锁定 OICM checkpoint gate 并启动长任务）
- approval: `user-approved`
- approval_basis: 用户明确要求“先停止训练吧，然后用 bestpt 直接开始训练 CmDecoderv2。照样用三卡”。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留旧 CmDecoder、ObjectInteractionCm 及既有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（正式训练已启动；终态前不作收敛或跨 embodiment 效果结论）
- scope: 固定 Dexplore RL decoder view、Temporal-D2 `K=4`、q/wrist 输出和正式 train/val；GPU 0/1/2，DDP，per-device batch 8，global batch 24，50 epochs（320050 total steps）。
- frozen_oicm_checkpoint: `outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt`
- frozen_oicm_checkpoint_sha256: `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`

**命令与产物（运行中）**

- command: `CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --distributed`
- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/)（`RUNNING`, PENDING）
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/config.json)（`RUNNING`, PENDING）
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/run_manifest.json)（`RUNNING`, PENDING）
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)（`RUNNING`, PENDING）
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)（`RUNNING`, PENDING）
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt)（`RUNNING`, PENDING）
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt)（`RUNNING`, PENDING）

**启动证据**

- launcher 与 3 个 rank 已正常启动；`train_setup` 报告 `device=cuda:0`、`world_size=3`、`global_batch=24`、`total_steps=320050`。
- 本条目仅记录运行启动；正式训练终态、last step/epoch、best metric 和实际 checkpoint 路径待运行结束或用户要求停止后补记。

**文件**

- `src/task/CmDecoderv2/` — 本 Task 的实现、配置、数据工具、测试和文档；本次只更新正式配置中的 OICM checkpoint SHA gate，未改动模型代码或数据 view。

**原因**

OICM 续训已按用户要求停止，必须把静态 decoder 配置绑定到已冻结的 `best.pt`，再使用 approved Temporal-D2、RL-only train/val 和三卡 DDP 进行正式训练，避免运行中动态追随 checkpoint。

**验证**

- `sha256sum outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt` 与配置 gate 一致：`fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。
- 启动命令已创建 run manifest/config/metadata/metrics/train.log；3 个 rank 正常运行，GPU 0/1/2 利用率已上升。
- 前 800 steps 已写入 metrics，loss 与 q/wrist 子损失均为 finite；这是工程运行证据，不代表 decoder 已收敛或 MANO→Inspire 定性效果成立。

## 2026-09-06 10:56:47 +0800 — V1.1.1 Temporal-D2 实现、完整 decoder view 与 smoke

- activity_id: `cmdecoderv2-v1.1.1-20260906-102433-implementation`
- timestamp: `2026-09-06 10:56:47 +0800`
- modification_version: `V1.1.1`
- type: `architecture / code / data / experiment / operation / documentation`
- change_level: `L3 + L2`
- approval: `user-approved`
- approval_basis: 用户确认 v1.1 默认方案、要求首版直接使用 Temporal-D2，并明确回复“可以，你直接开始实现吧”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（旧 CmDecoder、ObjectInteractionCm 和既有实验改动均保留且未纳入本 Task）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅工程实现、数据合同和 smoke；正式训练尚未开始，不构成收敛或跨 embodiment 效果结论）
- scope: 新建独立 CmDecoderv2；实现 Dexplore RL decoder view、`K=4` Temporal-D2、q/wrist loss、
  BaseRunner 训练入口与 MANO-only 定性 rollout。未修改 `src/base/`、旧 `src/task/CmDecoder/`、
  ObjectInteractionCm 代码/cache/checkpoint、原始 GRAB/Dexplore 数据、既有 split 或运行进程。

**设计与文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 新增 `CmDecoderv2: V1.1.1` Task 指针。
- [Task package](../../) 与 [config.py](../../config.py) — 建立独立 Python Task 包和默认静态合同；不从旧
  CmDecoder 导入兼容实现。
- [指导 v1.1](../指导/v1.1.md) 与 [最终计划 v1.1](../plan/v1.1.md) — 冻结研究目标、数据语义、Temporal-D2、
  q/wrist、扰动和分阶段闸门。
- [model.py](../../model.py) — 冻结并 strict-load OICM；把 `K x 16` Cm/anchor/time token set 作为 memory，
  当前 global/link/future-step 作为 query；不添加 slot-ID embedding，输出四步 q residual 与 wrist 相对 `SE(3)`。
- [kinematics.py](../../kinematics.py) — 固化 Dexplore native→URDF、6 个独立 q、mimic、`hand_base_link`
  wrist FK 和 18 个 object-frame link query。
- [dataset.py](../../dataset.py) 与 [decoder view builder](../../tools/data/build_dexplore_view.py) — 每个
  `K=4` Cm window 使用 `K+1=5` 个连续 30 Hz cache frame；RL train/val 提供 q/wrist GT，MANO test 不含 GT。
- [Task-local tools](../../tools/) — 包含独立 tools/data Python 包与 decoder view 入口。
- [runner.py](../../runner.py)、[train.py](../../train.py) 与 [正式配置](../../configs/active/dexplore_rl_v1_1_1.yaml) —
  实现 q Smooth-L1、translation x100 Smooth-L1、SO(3) geodesic、`0.8^(h-1)` horizon 权重和 per-horizon/identity 指标。
- [smoke 配置](../../configs/active/dexplore_rl_v1_1_1_smoke.yaml) 与 [Task tests](../../tests/) — smoke
  关闭 perturbation；正式配置使用已批准的 q/translation/rotation Gaussian perturbation。
- [visualize.py](../../visualize.py) — MANO source Cm 的 receding-horizon rollout 只执行 `h=1`，显式记录
  初始 Inspire state，导出 q/wrist/点云、PNG，并可选 MP4；不读取 paired Inspire test GT。
- [Task 文档入口](../README.md)、[架构记录](architecture_log.md)、[实验记录](experiment_log.md) 与
  [可迁移事实](repo_memory.md) — 记录新 Task 的事实和证据导航。
- [资产合同](../../assets/README.md) — Task-local Inspire 右手 URDF 由本机软链接提供，URDF SHA256 写入 view。

**原因**

旧 CmDecoder 的 HRDexDB/point-flow 兼容路径与本研究的 Dexplore RL q/wrist 语义不同。独立 Task 可以固定
MANO-source Cm → Inspire target 的数据边界，并用未来 Cm window + 当前 Inspire state 直接检验
Temporal-D2 与 receding-horizon 假设，同时避免把同一 parent 的 MANO/Inspire pair 泄露进 decoder 监督。

**数据运行**

- run_id: `cmdecoderv2-view-pilot-20260906-105616`
- run_status: `COMPLETED`
- command: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.tools.data.build_dexplore_view --mode pilot --output data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot --window-size 4 --resume --activity-id cmdecoderv2-v1.1.1-20260906-102433-implementation`
- evidence: [pilot view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/)、
  [manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/manifest.json)、
  [index.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/index.json)、
  [run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/run_manifest.json)。
- result: RL train/val 各 1 sequence、MANO-only test 1 sequence，供静态检查和 smoke。

- run_id: `cmdecoderv2-view-full-20260906-105602`
- run_status: `COMPLETED`
- command: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.tools.data.build_dexplore_view --mode full --output data/processed_data/cm_decoder_v2/dexplore_rl_v1_1 --window-size 4 --resume --activity-id cmdecoderv2-v1.1.1-20260906-102433-implementation`
- evidence: [完整 view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/)、
  [manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/manifest.json)、
  [index.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/index.json)、
  [run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/run_manifest.json)。
- result: RL train `503` sequence / `153602` window，RL val `62` / `19730` window，MANO-only test
  `125` sequence；565 个 RL sidecar 全部存在且 shape 正确，test entry 中 Inspire GT 字段计数为 `0`。

**训练 smoke**

- final run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619`
- run_status: `COMPLETED`
- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1_smoke.yaml`
- last_step: `2`; last_epoch: `1`; best_metric: `val/loss=1.4703673253`
- evidence: [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/)、
  [config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/config.json)、
  [run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/metrics.jsonl)、
  [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/train.log)、
  [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/best.pt)、
  [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/latest.pt)。
- result: 两步 optimizer 与完整 pilot val 完成；真实 Cm shape 为 `[B,4,16,32]`，OICM 保持
  `eval/no-grad`，decoder 可反向；metadata 记录 OICM SHA256
  `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。

实现过程中另产生四个终态为 `COMPLETED` 的预备 smoke，分别用于发现 provenance 缺项和验证修正；不作为
最终证据，但保留可审计产物：

- `..._102937`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/checkpoints/best.pt)。
- `..._104051`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/checkpoints/best.pt)。
- `..._104215`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/checkpoints/best.pt)。
- `..._104811`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/checkpoints/best.pt)。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile $(find src/task/CmDecoderv2 -name '*.py' -type f | sort)`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`；覆盖
  mimic/FK、`K+1` window、输出 shape、slot permutation invariance 与 identity rotation finite gradient。
- 单卡真实 checkpoint forward/backward：`pred_q_delta [1,4,6]`、translation/rotvec `[1,4,3]`；
  `oicm_training=False`、`oicm_grad_count=0`，有效 contact window 的四帧 `cm_sample_valid=True`。
- 完整 view 审计：所有 RL window arrays 为 `[T-4,5]`，所有 sidecar 链接存在，MANO test 不含 q/wrist GT。
- 正式数据加载：503 个 train sequence / 153602 windows 可构造；perturbation 同 epoch 可复现、跨 epoch
  变化，浮点张量全部 finite。batch=8（OICM flatten batch=32）单卡 backward 峰值 allocated `1.39 GiB`、
  reserved `2.57 GiB`。
- 正式 checkpoint 闸门负向测试：`LOCK_AFTER_OICM_TERMINAL` 未替换时按预期拒绝建模，避免误用动态 best。
- `git diff --check`：通过；`audit_diff.py --worktree --scope-prefix src/task/CmDecoderv2
  --scope-prefix docs/current_versions.yaml --check-links`：通过，覆盖 `24` 个变更路径和 `60` 个本地链接。

**未启动与回滚**

- 正式 CmDecoderv2 训练未启动：OICM 续训仍在 0/1/2 卡运行，当前 `best.pt` 只能作为 smoke 输入；必须等
  OICM 终态后冻结 checkpoint SHA256，再进入 Gate C。MANO rollout 代码已实现但遵循计划，需在正式 decoder
  训练后才执行 Gate D。
- 回滚入口为隔离/移走本 Task 目录、`data/processed_data/cm_decoder_v2/` 新 view 和
  `outputs/cmdecoderv2/` 新 smoke；无需也不得改动或删除旧 CmDecoder、OICM cache/checkpoint、原始数据或运行。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。Task 级测试不在根 `pytest.ini` 的默认
  `testpaths` 中，因此按目录规范显式传入 `src/task/CmDecoderv2/tests`；无需改变共享 pytest 配置。
