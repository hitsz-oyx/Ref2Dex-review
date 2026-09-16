# CmResidual 实验记录

- scope: `task:CmResidual`
- related: [任务入口](../README.md)、[V1.5 最终计划](../plan/V1.5.md)、[V1.4 最终计划](../plan/V1.4.md)、[V1.3 最终计划](../plan/V1.3.md)、[活动记录](activity_log.md)

## 2026-09-16 — V1.8 safe residual 20-epoch preservation stability

- modification_version: `V1.8`
- operation_category: `experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- terminal_activity_id: `ACT-20260916-120009-CMRESIDUAL-V18-E20`
- run_ids: `cmresidual_v18_b0_envmax_20260916_114134`、`cmresidual_v18_t10_retry_20260916_115331`、`cmresidual_v18_e10_20260916_115633`、`cmresidual_v18_t20_20260916_115828`、`cmresidual_v18_e20_20260916_120009`
- run_status: 全部 `COMPLETED`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`（dirty worktree 为已批准的 V1.8 实现）
- initial_checkpoint: T10 为 `null`；T20 明确从合格 T10 epoch-10 checkpoint 恢复
- final_checkpoint: [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth)
- last_step / last_epoch: `40960 / 20`
- best_metric: E20 preservation gate `lift_ratio=0.9733328208`、`contact_ratio=1.6835971901`
- conclusion: `SUPPORTED`（单 seed、单场景、20-epoch preservation stability）；抓取改善、泛化、Cm 因果和统计显著性 `INCONCLUSIVE`

**完整结果**

| stage | 状态 | mean-env-max lift (m) | mean contact | 相对 B0 lift/contact | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | COMPLETED | 0.085096 | 0.193631 | 1.000000 / 1.000000 | baseline usable |
| T10 | COMPLETED | — | — | — | epoch 10 checkpoint contract pass |
| E10 | COMPLETED | 0.085426 | 0.225536 | 1.003871 / 1.164776 | gate pass |
| T20 | COMPLETED | — | — | — | epoch 20 checkpoint contract pass |
| E20 | COMPLETED | 0.082827 | 0.325996 | 0.973333 / 1.683597 | gate pass |

- T10/T20 保持 1442-D no-Cm、18D action、零 mean 初始、固定 `sigma=0.1`；两个 checkpoint 的
  model/optimizer/action 均 finite，reload action max diff 均为 `0.0`。
- T10 epoch 10 training residual RMS `0.105227`；T20 epoch 20 为 `0.107345`。E10/E20 使用 deterministic
  mean action，不以 stochastic training rollout 指标代替 preservation gate。
- E10/E20 的 `mean_env_max_lift_m` 均高于 B0 的 80% 阈值 `0.0680769563 m`，mean contact occupancy
  均高于阈值 `0.1549046345`。这支持“短程 residual 学习未显著破坏 base 行为”，不支持“抓取显著改善”。
- GPU PhysX 即使 seed 固定仍不承诺 bitwise determinism；当前只有单次 B0、单 seed 和单 airplane 场景，
  不能给出统计显著性或泛化结论。本版本关闭 OI-Cm，也不能评价 Cm 贡献。

**实现偏差与处置**

- 首个 T10 因错误覆盖根级 `max_iterations` 而实际执行 20 epochs，终态为 `FAILED / INVALID_IMPLEMENTATION`，
  其所有 checkpoint 均未复用。修复为权威 `train.params.config.max_epochs` 并增加启动前校验后，从零重跑。
- 合格 T10 首次离线验证未切换 `model.eval()`，RunningMeanStd 更新造成重复 action 假差 `1.04e-5`；不放宽
  `1e-6` 阈值，修复 eval mode 后对同一 checkpoint 重验为 `0.0`，无需重训。

**证据**

- B0：[manifest](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/metrics.jsonl)
- T10：[manifest](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/metrics.jsonl)、[validation](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/checkpoint_validation.json)、[log](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/train.log)
- E10：[manifest](../../../../../outputs/CmResidual/cmresidual_v18_e10_20260916_115633/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v18_e10_20260916_115633/metrics.jsonl)
- T20：[manifest](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/metrics.jsonl)、[validation](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/checkpoint_validation.json)、[log](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/train.log)
- E20：[manifest](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/metrics.jsonl)、[log](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/eval.log)

## 2026-09-16 — V1.8 B0 mean-env-max-lift 基线

- modification_version: `V1.8`
- operation_category: `experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- activity_id: `ACT-20260916-114134-CMRESIDUAL-V18-B0-ENVMAX`
- run_id: `cmresidual_v18_b0_envmax_20260916_114134`
- run_status: `COMPLETED`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`（dirty worktree 为已批准的 V1.8 实现）
- initial_checkpoint / checkpoint: `null`（全零 residual，不训练）
- last_step / last_epoch: `367 / null`
- best_metric: `mean_env_max_lift_m=0.0850961953`
- output: [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134)
- conclusion: `SUPPORTED`（运行、zero-residual parity 与正 preservation baseline）；科研结论 `INCONCLUSIVE`

**结果与后续 gate**

- 新指标对每个环境从 `0 m` 起累计完整 367 步内的最大 object lift，再对 64 个环境等权平均；它跨
  success reset 保留已达到的 lift，不受后续失败/下落阶段把 batch/time mean 拉成负值的影响。
- `mean_env_max_lift_m=0.085096`、mean contact occupancy `0.193631`、mean tip distance `0.237304 m`、
  max success fraction `0.25`，`max_residual_target_delta=0`；367 行 observation/target 均 finite。
- E10/E20 的 80% preservation 阈值固定为 `mean_env_max_lift_m>=0.0680769563` 且
  `mean_contact_occupancy>=0.1549046345`。整段 batch/time `mean_lift_m=-0.509807` 仅保留为诊断量。
- B0 可用只支持进入后续工程 gate，不证明 residual 改善抓取；本次未启动 T10。

证据：[run manifest](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json)、
[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/metrics.jsonl)、
[eval.log](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/eval.log)、
[resolved config](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/config.json)。

## 2026-09-16 — V1.8 B0 同协议零残差基线

- modification_version: `V1.8`
- operation_category: `experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- activity_id: `ACT-20260916-113318-CMRESIDUAL-V18-B0-RETRY`
- run_id: `cmresidual_v18_b0_retry_20260916_113318`
- run_status: `COMPLETED`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`（dirty worktree 为已批准的 V1.8 实现）
- initial_checkpoint / checkpoint: `null`（全零 residual，不训练）
- last_step / last_epoch: `367 / null`
- best_metric: `mean_lift_m=-0.4904631897`
- output: [outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318)
- conclusion: `SUPPORTED`（运行、协议与 zero-residual parity）；`INVALID_IMPLEMENTATION`（planned lift-ratio baseline）；科研结论 `INCONCLUSIVE`

**假设与协议**

在启动 residual PPO 前，以 GPU5、64 env、seed42、367 steps、`terminateOnSuccess=true`、no-Cm 和全零
residual 建立 B0，并锁定 DExplore/reference/source SHA、reward 与 residual scale，供 E10/E20 的 deterministic
mean-action rollout 使用。首个同协议 run 因 NumPy 1.24/Isaac Gym `np.float` 兼容错误在 step 0 失败；限定于
Task-local 进程入口的 alias 修复通过 16 个测试后，新 run_id 重试完成。

**结果与停止解释**

- 367 行指标完整且 finite，`max_residual_target_delta=0`，支持 zero-residual wiring/parity。
- `mean_lift_m=-0.490463`、`max_lift_m=0.069457`、mean contact occupancy `0.206335`、mean tip distance
  `0.175315 m`、max success fraction `0.359375`；累计 done 数 `302`，367 步中 129 步发生 reset。
- 只有 18/367 个全局 step 的 batch mean lift 为正，最后 20 步 mean lift 约 `-0.572256 m`。当前
  `terminateOnSuccess=true` 会重置成功环境，而未成功环境继续下落，使整段 batch/time mean lift 为负。
- plan 要求 E10/E20 的 mean lift 和 mean contact occupancy 都不低于 B0 的 80%；负的 baseline lift 不能形成
  有方向一致、可解释的 preservation ratio，现有 `preservation_gate` 也明确拒绝非正 baseline。因此停止在 B0，
  不启动 T10；不能由本次结果判断 residual stability、抓取改善或 Cm 因果。

证据：[run manifest](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/run_manifest.json)、
[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/metrics.jsonl)、
[eval.log](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/eval.log)、
[resolved config](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/config.json)。

## 2026-09-16 — V1.7.1 正式 PPO 训练外部中断与部分结果

- modification_version: `V1.7.1`
- operation_category: `experiment`、`diagnostic`、`operation`、`documentation`
- approval: `user-approved`
- activity_id: `ACT-20260916-104238-CMRESIDUAL-V171-PPO-TERMINAL`
- run_id: `cmresidual_ppo_formal_v17_20260915_2345`
- run_status: `FAILED`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`（运行启动时）
- initial_checkpoint: `null`（随机初始化 residual PPO）
- last_step / last_epoch: `335872 / 165`
- best_metric: epoch 50 checkpoint eligible `rewards/iter=5.883934020996094`
- best_checkpoint: [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/CmResidual.pth](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/CmResidual.pth)
- latest_checkpoint: [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/last_CmResidual_ep_100_rew_-6470.568.pth](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/last_CmResidual_ep_100_rew_-6470.568.pth)
- reference: [data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/manifest.json](../../../../../data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/manifest.json)
- output: [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/)
- conclusion: `INVALID_IMPLEMENTATION`（计划预算和终态验证未完成）；科研结论 `INCONCLUSIVE`

**终止与证据完整性**

训练于 `2026-09-15T23:29:00+08:00` 启动，TensorBoard 最后写入 epoch 165（`2026-09-16T10:23:47+08:00`）。
承载 runner 的外部执行会话随后以 `exit_code=-1` 终止，runner 与训练子进程均消失；没有 Python traceback、CUDA OOM
或内核 OOM 证据。由于 runner 未能执行收尾，V1.7.1 从原始 TensorBoard 导出 165 行部分
[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/metrics.jsonl)，并将
[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/run_manifest.json)
终态化。epoch 165 后没有 checkpoint；最新定期 checkpoint 为 epoch 100，且未完成计划要求的终态独立策略评估。

**部分结果与边界**

- V1.4 zero-residual gate：mean/max lift `0.0798409 / 0.2123542 m`、mean tip distance `0.048995 m`、
  mean contact occupancy `0.480790`，末步 `success_fraction=0.75`；只支持单场景 base 抓取/抬升工程行为。
- V1.7 epoch 165：`lift_mean=-0.545820 m`、`tip_distance_mean=1.049141 m`、`contact_occupancy=0.165625`、
  `success_fraction=0`、`residual_rms=0.852116`、`residual_saturation_ratio=0.009549`。
- 训练从 epoch 1 起 residual RMS 已为 `0.731104`。epoch 50/100 checkpoint 的高斯策略平均标准差分别约
  `1.0051 / 0.9984`，说明训练 rollout 持续承受接近单位标准差的随机 residual，而不是从 zero residual 邻域缓慢修正。
- 上述幅度支持“V1.7 随机 residual 训练 rollout 相对 zero-residual 工程 gate 明显退化”的诊断观察；但两者的
  CPU/GPU、env 数、成功重置与评估协议不同，且只有一个 seed，不能声称统计显著，不能把退化归因于 residual
  方法或冻结 OI-Cm。需要同协议 zero/no-Cm/Cm ablation 和 deterministic checkpoint evaluation 才能判断因果。

工程 smoke 与此前 PPO wiring `SUPPORTED` 结论不变；本次未完成训练预算、checkpoint 终态验证或独立抓取评估，
因此 residual 增益、收敛、Cm 贡献和普遍抓取结论均保持 `INCONCLUSIVE`。

## 2026-09-15 — V1.6 corrected reference 训练准入

- modification_version: `V1.6`
- operation_category: `data`、`diagnostic`
- approval: `user-approved`
- activity_id: `ACT-20260915-231754-CMRESIDUAL-V16-REFERENCE-ELIGIBLE`
- run_id: `cmresidual_reference_v2_build_20260915_231227`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（reference/data contract）；正式 PPO 科学结论为 `INCONCLUSIVE`

source tensor 文档明确声明 `205:206` 为 object contact flag。使用固定-wrist、constrained-q6 corrected reference、同一 URDF 和确定性表面采样执行 V1.0 D1 准入：367 帧中 296 个 raw-contact 帧，295 个满足 hand-object minimum distance `<=20 mm`，比例 `0.9966216216`。新 v2 manifest 因此为 `training_eligible=true`；旧 v1 保留为不可训练的历史诊断 artifact。证据：[v2 manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/manifest.json)、[run manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/run_manifest.json)。

该结果只证明 reference 具备正式训练准入，不证明 tracker/residual 的收敛、抓取成功率或 Cm 增益。正式训练需使用 v2 path 且保持 `allowIneligibleFor=""`。

## 2026-09-15 — V1.5.4 PPO wiring pilot 通过

- modification_version: `V1.5.4`
- operation_category: `experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- activity_id: `ACT-20260915-215500-CMRESIDUAL-V154-PPO-SUPPORTED`
- run_id: `cmresidual_ppo_pilot_v154_20260915_214302`
- run_status: `COMPLETED`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- initial_checkpoint: `null`；checkpoint: [outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/rlgames/nn/last_CmResidualPilot_ep_2_rew__5.84_.pth](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/rlgames/nn/last_CmResidualPilot_ep_2_rew__5.84_.pth)
- last_step / last_epoch: `4096 / 2`
- best_metric: deterministic reload action max abs diff `0.0`

**假设与结果**

移除错误的 `num_subscenes=1` override 后，canonical `num_subscenes=4` 在 GPU PhysX 中被自动归一为单场景；同一 GPU5、seed42、64 env × 2 updates pilot 应完成 wiring smoke。[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/config_comparison.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/config_comparison.json) 确认只发生预期的 subscene 与 buffer 配置差异。

- 1-env prepare smoke 成功；随后完整 pilot 正常越过 PhysX 初始化并完成两次 update。
- 2 epochs、episode 已完成；checkpoint、model/optimizer/normalizer/action 均 finite；独立 CPU 重载 action 差 `0.0`。
- 工程结论：`SUPPORTED`（PPO wiring）；科研结论：`INCONCLUSIVE`，不能据两次 update 判断 residual 效果或收敛。

证据：[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/run_manifest.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/train.log)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/checkpoint_validation.json)。

## 2026-09-15 — V1.5.4 GPU prepare_sim 二分诊断

- modification_version: `V1.5.3`
- operation_category: `diagnostic`、`operation`
- activity_id: `ACT-20260915-213309-CMRESIDUAL-V154-PREPARE-DIAGNOSTIC`
- run_id: `cmresidual_prepare_probe_v154_20260915_2120`
- run_status: `COMPLETED`
- 假设：当前崩溃来自 actor 资产组合或 GPU PhysX prepare 配置。
- 证据：hand、hand+table、hand+object、full 均在同一 `prepare_sim` 边界 SIGSEGV；使用 `num_subscenes=4` 后越过该边界，而原 wrapper 使用 `num_subscenes=1`。
- 结论：`SUPPORTED`（`num_subscenes=1` 是高可信触发条件）；`INCONCLUSIVE`（删除 override 后完整 64 env × 2 updates 是否通过，尚未运行）。
- 证据入口：[outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/run_manifest.json)、[outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/subscenes4_probe.log](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/subscenes4_probe.log)、[outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand.log](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand.log)。

## 2026-09-15 — V1.5.3 同卡缩小 PhysX buffer 复跑

- modification_version: `V1.5.3`
- operation_category: `experiment`、`operation`、`diagnostic`
- approval: `user-approved`（[src/task/CmResidual/docs/plan/V1.5.md](../plan/V1.5.md)）
- activity_id: `ACT-20260915-210957-CMRESIDUAL-V153-PPO-FAILED`
- run_id: `cmresidual_ppo_pilot_v153_20260915_210542`
- run_status: `FAILED`；终态入口为 [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md)
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- initial_checkpoint: `null`（随机初始化 residual；frozen teacher/OI-Cm 与 V1.5.2 相同）
- last_step / last_epoch / best_metric / checkpoint: 均为 `null`，没有训练曲线、event 或 checkpoint

**假设与对照**

假设仅将小规模 pilot 的 PhysX `max_gpu_contact_pairs/default_buffer_size_multiplier` 从 `41943040/25.0`
降为 `8388608/5.0`，就足以让 V1.5.2 失败的 GPU5 pilot 完成固定 2 updates。冻结 seed 42、64 env、horizon 32、
minibatch 2048、解释器、全部输入 SHA、PPO/reward/action/reference 和其余 resolved config；只复跑一次。
[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/config_comparison.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/config_comparison.json) 确认只有这两项容量发生配置差异。

**结果与解释**

- 运行 `2026-09-15T21:05:42+08:00` 启动，`2026-09-15T21:05:54+08:00` 以同样的 `SIGSEGV (-11)` 失败；
  [outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/hydra/runs/CmResidualPilot_15-21-05-47/config.yaml](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/hydra/runs/CmResidualPilot_15-21-05-47/config.yaml) 证实子进程确实收到新 buffer 值。
- 日志仍止于 `GPU Pipeline: enabled`，无任何 update 完成证据。[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/metrics.jsonl) 只有失败状态行；
  它不是训练 metric。没有 checkpoint 或 TensorBoard event；退出后无残留进程。
- `REFUTED`：两项指定 buffer 缩小足以修复启动的假设。此反证不排除更广泛的 GPU 内存、PhysX 或初始化问题。
- `INVALID_IMPLEMENTATION`：Gate B 未完成，不能声称 PPO wiring 可用。
- `INCONCLUSIVE`：精确原生崩溃位置、residual 学习、收敛和抓取改善。现有日志不足以锁定失败调用；
  应先获取阶段定位与原生栈，再决定代码修正或设备对照，本轮未启动新的诊断仿真。

**证据**

[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542)；[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/run_manifest.json)；[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/config.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/config.json)；[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/train.log)；
[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/terminal_checks.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/terminal_checks.json)。此前 Gate A 的工程 `SUPPORTED` 结论保持原样。

## 2026-09-15 — V1.5 非零 residual gate 与 PPO wiring pilot

- modification_version: `V1.5.2`
- operation_category: `experiment`、`operation`、`diagnostic`
- approval: `user-approved`（[V1.5 最终计划](../plan/V1.5.md)）
- activity_id: `ACT-20260915-204914-CMRESIDUAL-V151-NONZERO-SUPPORTED`、`ACT-20260915-205520-CMRESIDUAL-V152-PPO-FAILED`
- run_id: `cmresidual_nonzero_v15_rerun_20260915_204817`、`cmresidual_ppo_pilot_v15_20260915_204915`
- run_status: Gate A `COMPLETED`；Gate B `FAILED`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- frozen inputs: DExplore/OI-Cm checkpoints、legacy source tensor 与 corrected reference 的路径和 SHA256 见各自
  `run_manifest.json`
- initial_checkpoint: `null`（residual PPO 随机初始化）
- checkpoint: `null`（Gate B 在训练更新前失败）
- last_step: Gate A `32`；Gate B `null`
- last_epoch: Gate A `null`；Gate B `null`
- best_metric: Gate A `final_signed_response_separation=0.3832877874`（工程响应量）；Gate B `null`

**假设与冻结条件**

V1.5 先用固定正负 residual 检查 simulator-side physical target 合同，再用 64 env × 2 updates 验证 PPO wiring。
保持 V1.4 frozen DExplore base、18D action schema、reward、termination、residual scale、模型宽度、checkpoint 和数据
不变。corrected reference 继续为 `training_eligible=false`；诊断与 pilot 分别显式使用 `diagnostic` 和
`ppo_pilot` 限定准入，不把 legacy teacher source 重新声明为 corrected training GT。

**Gate A 结果**

- 修订后运行使用 CPU PhysX、seed 42、4 env × 32 steps，zero/zero/+0.25/-0.25 固定 residual；requested、
  ignored mimic、zero target、runtime mimic、joint limit、saturation 与 indexed-reset isolation 最大误差均为 0。
- 正负环境 applied delta 约为 `0.05`，末步 DOF signed response separation=`0.3832877874`；所有 observation、
  reward、target 和 object pose finite，无 reset 或 simulator error。
- 工程结论为 `SUPPORTED`。证据：[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/metrics.jsonl)、
  [eval.log](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/eval.log)。

**Gate B 结果与停止条件**

- GPU5 隔离后仅暴露进程内 `cuda:0`；子进程完成 Isaac Gym/gymtorch 加载和 GPU PhysX 创建提示后，以
  `SIGSEGV (-11)` 退出，没有进入 `Started to train`，也没有 epoch、checkpoint 或 TensorBoard event。
- 本次 `run_status=FAILED`、Gate B=`INVALID_IMPLEMENTATION`。证据：[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/run_manifest.json)、
  [config.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/config.json)、
  [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/metrics.jsonl)、
  [train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/train.log)。
- 当前最高可信根因假设是小规模 pilot 沿用了面向大规模训练的 PhysX buffer
  `41943040 / 25.0`；DExplore 自己只在小批量 export 路径降到 `8388608 / 5.0`，源码说明前者可能在
  24-GB 卡触发异步 illegal access。由于计划规定 gate 失败即停止，本轮未通过复跑验证该假设。

**结论**

- `SUPPORTED`（工程）：固定非零 residual 能按 18D/mimic/joint-limit 合同进入 simulator physical target。
- `INVALID_IMPLEMENTATION`（Gate B）：当前 pilot 配置未完成任何 PPO update，不能宣称训练 wiring 可用。
- `INCONCLUSIVE`（科研）：没有产生可评估策略；residual 增益、收敛和抓取改善均无证据。

## 2026-09-15 — V1.4.2 DExplore parity 与 zero-residual gate

- modification_version: `V1.4.2`
- operation_category: `experiment`、`operation`
- approval: `user-approved`（[V1.4 最终计划](../plan/V1.4.md)）
- activity_id: `ACT-20260915-191707-CMRESIDUAL-V142-ZERO-FINAL`
- run_id: `cmresidual_zero_v14_final_20260915_191707`
- run_status: `COMPLETED`
- base_commit: `b1dac1c7955c9b960481279425f9e4e2e2269a98`
- seed / budget: `42` / `4 envs × 367 control steps`，CPU PhysX
- frozen inputs: DExplore `inspire.pth`、OI-Cm `best.pt`、legacy source tensor 与 corrected reference 的路径和 SHA256 见
  [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/run_manifest.json)
- initial_checkpoint: `null`（评估不恢复 CmResidual PPO）
- checkpoint: `null`（本次不构造或训练 PPO）
- last_step: `367`
- last_epoch: `null`
- best_metric: `max_lift_m=0.2123541832`（gate 行为观测量，不是独立抓取 benchmark）

**假设与冻结条件**

本 gate 只检验：按发布源码重建 DExplore actor、721D observation、legacy reference/reset、simulator 参数和
base target 后，严格全零 residual 能否退化为可运行的 frozen DExplore baseline。corrected reference 继续保持
`training_eligible=false`；reward 数值、OI-Cm、residual 权限、模型权重与 checkpoint 均冻结，并关闭
success-triggered reset 以完整覆盖 367 帧。不检验 Cm residual 的可学习性，也不启动 PPO。

**实现诊断与中间失败**

- 首次正式运行 `cmresidual_zero_v14_20260915_184916` 虽完成 367 步、finite/reset/zero-target 条件，但在
  source 第 16 个接触帧没有近物或接触，判为 `INVALID_IMPLEMENTATION`；证据保留在
  [失败运行 manifest](../../../../../outputs/CmResidual/cmresidual_zero_v14_20260915_184916/run_manifest.json)。
- 对未经修改的 DExplore 首帧 state/reference 做真实输入对照，定位到 provider 错误解释源码
  `body_pos[..., 6:]`：末维只有 3，因此该切片实际为空且应为 no-op。错误实现却沿 body 维对 10 个关键点
  取 remainder，生成最高 `188.4956 m/s` 的伪 reference 速度并破坏 object-surface IG。
- 修正后 20 步单环境 [reference smoke](../../../../../outputs/CmResidual/cmresidual_zero_v14_reference_smoke_20260915_191608/run_manifest.json)
  在第 16 步得到 tip distance=`0.045783 m`、实测 contact occupancy=`0.6`，随后才执行最终正式 gate。

**最终结果**

- 初始 native q/wrist/object/table 最大误差分别为 `0`、`2.09e-7 m`、`1.30e-7 m`、`7.45e-8 m`；
  367 步无 reset，observation、target 与 object pose 全程 finite。
- `residual_target_delta=0`、`target_saturation_ratio=0`。第 16 步 tip distance=`0.045808 m`、实测/reference
  contact occupancy=`0.65/0.4`；全程平均/最大 contact occupancy=`0.480790/0.75`，最小 tip distance=
  `0.037769 m`。
- `success_fraction` 从第 34 步出现，最大为 `1.0`、末步为 `0.75`；由于 gate 禁用 success reset、没有完成
  episode，manifest 内 legacy `final_success_rate=0` 不代表逐环境成功状态为零。
- 完整证据：[config.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/config.json)、
  [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/metrics.jsonl)、
  [eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/eval.log)。

**结论**

- `SUPPORTED`（工程）：真实 source 对照、reset、zero-target、finite、接触行为和正常进程终态共同支持
  “不加 residual 时，当前实现可正常执行 frozen DExplore baseline”。
- `INCONCLUSIVE`（科研）：mean/max lift=`0.079841/0.212354 m` 只是同一 gate 内的行为观测，不是独立
  benchmark；Cm residual 增益、PPO 收敛和抓取效果尚未检验。

按 [V1.4 最终计划](../plan/V1.4.md) 的边界，本轮没有启动 PPO，也没有修改 corrected reference、reward、
checkpoint 或外部 DExplore 源码。

## 2026-09-15 — V1.3.2 zero-residual gate

- modification_version: `V1.3.2`
- operation_category: `experiment`、`operation`
- approval: `user-approved`
- run_id: `cmresidual_zero_v13_20260915_164405`
- run_status: `FAILED`
- base_commit: `95d5aea25a8d75306982d4266d1040911fd25efa`
- seed / budget: `42` / `4 envs × 367 control steps`，CPU PhysX
- initial_checkpoint: DExplore `inspire.pth` 与冻结 OI-Cm `best.pt`，确切路径及 SHA256 见
  [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/run_manifest.json)
- checkpoint: `null`（本次不构造或训练 PPO）
- last_step: `367`
- best_metric: `max_lift_m=0.0885452628`（仅为 rollout 观测值，不是有效抓取成绩）

**问题与冻结条件**

本 gate 检查冻结 DExplore 在严格全零 residual 下能否保持 base-only physical target 映射、完成有限的
Isaac Gym rollout，并产生可解释的接触与物体行为。保持 `s1/airplane_lift` reference、30 Hz、reward、
OI-Cm tokens/anchors/effect、DExplore checkpoint 和全部模型参数不变。reference manifest 明确为
`training_eligible=false`，本次只允许诊断性评估。

**结果**

- 367 行逐步 [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/metrics.jsonl)
  已完整写入；所有 observation 与 physical target finite。
- `residual_action_abs_max=0`、`residual_target_delta_max=0`，三组 physical residual RMS 均为 0，
  `target_saturation_ratio=0`。这支持 zero-residual wiring 精确退化到 frozen base target 的局部工程不变量。
- mean / max lift 为 `0.0697425 / 0.0885453 m`，累计 reset 数为 4，最终内部 success rate 为 1.0；但
  contact occupancy 全程为 0，mean tip distance 为 `1.421636 m`。无手部接触时的 apparent lift/success
  不能解释为抓取，反而说明初始化、reference 对齐或物体/地面动力学仍存在阻断性问题。
- rollout 步数完成后，Isaac Gym `destroy_sim` 超过 7 分钟仍未返回并持续异常占用 CPU；进程最终以
  `SIGTERM` 停止。终态 manifest 由停止操作补记为 `FAILED`，不把完整 metrics 等同于正常退出。

**结论**

- `SUPPORTED`：单测与 rollout 都支持“zero residual 不改变 DExplore physical target”，且本次状态张量 finite。
- `INVALID_IMPLEMENTATION`：完整 gate 未满足正常退出和合理接触行为；不能据 apparent lift/success 宣称
  DExplore 抓取有效。
- `INCONCLUSIVE`：Cm residual 的可学习性、PPO 收敛、抓取效果及科研假设均未检验。

按 [V1.3 最终计划](../plan/V1.3.md) 的停止条件，本次没有启动 PPO。下一步需先经用户确认，将初始化/reference
对齐与 `destroy_sim` 清理分别作为诊断范围，再决定是否修正和复跑 gate。运行日志见
[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/eval.log)。
## 2026-09-16 — V1.9.1 critic-only Cm 配对 wiring smoke

- modification_version: `V1.9.1`
- operation_category: `experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- activity_ids: `ACT-20260916-151209-CMRESIDUAL-V191-CONTROL-SMOKE`、`ACT-20260916-152152-CMRESIDUAL-V191-CRITICCM-SMOKE`
- run_ids: `cmresidual_v19_control_smoke_20260916_151209`、`cmresidual_v19_critic_cm_smoke_20260916_152152`
- run_status: 两侧均 `COMPLETED`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`（dirty worktree 为已批准的 V1.9.1 实现；无关 ObjectInteractionCm 差异未触碰）
- initial_checkpoint: 两侧均为 `null`（seed42 从零初始化）
- last_step / last_epoch: 两侧均为 `4096 / 2`
- conclusion: `SUPPORTED`（两侧各自 GPU wiring、finite 与 checkpoint reload）；`INVALID_IMPLEMENTATION`（严格配对 stochastic RNG）；Cm utility、抓取改善与统计显著性 `INCONCLUSIVE`

**假设与协议**

Gate II 只检查 matched control（actor/critic 1442-D）与 critic-Cm（actor 1442-D、critic 2005-D）能否在相同 GPU5、64 env、seed42、2 epochs、零 mean、固定 `sigma=0.1` 和冻结 OI-Cm 下完成训练与 checkpoint 重载。smoke 不承担效果判断；只有严格配对合同成立后才允许 T10/E10。

**结果**

| variant | epoch-2 c_loss | epoch-2 KL | success fraction | residual RMS | checkpoint reload diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 3.538330 | 0.017957 | 0.06250 | 0.105352 | 0.0 |
| critic-Cm | 4.212568 | 0.199856 | 0.09375 | 0.105254 | 0.0 |

两侧 model、optimizer、action、observation 和 checkpoint 均 finite，sigma 保持冻结且约为 `0.1`，因此各自 wiring 为工程 `SUPPORTED`。证据：control [manifest](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/metrics.jsonl)、[validation](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/checkpoint_validation.json)、[log](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/train.log)；critic-Cm [manifest](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/metrics.jsonl)、[validation](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/checkpoint_validation.json)、[log](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/train.log)。

**配对失效与结论边界**

静态 preflight 证明两侧 actor 参数、初始 mean/sigma 和 Cm 后缀 action 不变性完全一致；但 critic replacement 的输入宽度不同，构造线性层时消耗的 PyTorch RNG draws 数不同。同 seed 构造后下一组 `torch.rand(8)` 最大绝对差为 `0.6465547085`，对应首 epoch success fraction 已为 `0.15625 / 0.203125`。因此首轮 stochastic action/trajectory 不能视为严格配对，两个 smoke 的 loss、success 或 reward 差异不得解释为 Cm 效果。

按 V1.9 停止条件，Gate II 后停止，未启动 B0/T10/E10。正式比较前需修复并验证 post-build RNG parity 与首轮 action-noise parity，再用新 run_id 复跑 Gate II；当前 Cm utility 结论保持 `INCONCLUSIVE`。
## 2026-09-16 — V1.9.2 RNG parity 修复与 Gate II 复跑

- modification_version: `V1.9.2`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- activity_ids: `ACT-20260916-153913-CMRESIDUAL-V192-CONTROL-SMOKE`、`ACT-20260916-154900-CMRESIDUAL-V192-CRITICCM-SMOKE`
- run_ids: `cmresidual_v19_control_smoke_rngfix_20260916_153913`、`cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900`
- run_status: 两侧均 `COMPLETED`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`（dirty worktree 为已批准的 V1.9.2 实现）
- initial_checkpoint: 两侧均为 `null`
- last_step / last_epoch: 两侧均为 `4096 / 2`
- conclusion: `SUPPORTED`（post-build RNG/首轮 stochastic action 静态 parity、两侧 GPU wiring 与 checkpoint）；分进程 GPU trajectory bitwise parity、Cm utility 和抓取改善 `INCONCLUSIVE`

**修复与协议**

V1.9.1 失败原因为两种 critic replacement 宽度消耗不同数量的 CPU RNG draws。V1.9.2 在两侧都按固定 `[1442, 2005]` 顺序构造 critic 候选，再选择各自的 1442-D 或 2005-D 层；不改变 actor、实际 critic 输入、OI-Cm、reward、数据、seed 或 PPO 超参数。静态硬 gate 新增 post-build CPU RNG equality 和首轮 stochastic action exact equality。

**结果**

| variant | epoch-1 success | epoch-1 residual RMS | epoch-2 c_loss | epoch-2 KL | reload diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.18750 | 0.1032118 | 3.707792 | 0.028788 | 0.0 |
| critic-Cm | 0.15625 | 0.1032118 | 3.724522 | 0.024047 | 0.0 |

两侧 manifest 均记录 `post_build_cpu_rng_equal=true`、`stochastic_action_max_abs_diff=0.0`，并完成 2 epochs、finite 和 checkpoint 重载。因此 Gate II 的工程 wiring 与静态配对合同为 `SUPPORTED`。然而独立 GPU 进程的 epoch-1 success fraction 仍未逐值一致；当前证据不能区分未记录的 CUDA sampling 差异与 GPU PhysX 非严格确定性，不能把后续 loss/success 差异解释为 Cm 效果。

证据：control [manifest](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/metrics.jsonl)、[validation](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/checkpoint_validation.json)、[log](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/train.log)；critic-Cm [manifest](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/metrics.jsonl)、[validation](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/checkpoint_validation.json)、[log](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/train.log)。

本结果只允许进入下一次经用户确认的 B0/T10/E10 seed42 探索性实验；单 seed 结果仍须标记 `INCONCLUSIVE`，正式因果或显著性结论需要后续多 seed 方案。
