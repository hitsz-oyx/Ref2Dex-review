# CmResidual 实验记录

- scope: `task:CmResidual`
- related: [任务入口](../README.md)、[V1.5 最终计划](../plan/V1.5.md)、[V1.4 最终计划](../plan/V1.4.md)、[V1.3 最终计划](../plan/V1.3.md)、[活动记录](activity_log.md)

## 2026-09-17 — V1.14.5 冻结 Cmv2 executed-transition CmBuffer smoke

- modification_version: `V1.14.5`
- operation_category: `code`、`diagnostic`、`operation`
- approval: `user-approved`（[V1.14 最终计划](../plan/V1.14.md) 的 V1.14.5 修订）
- terminal_activity_id: `ACT-20260917-224000-CMRESIDUAL-V1145-CMBUFFER`
- run_id: `cmresidual_v1145_cmbuffer_smoke_20260917_2240`
- run_status: `COMPLETED`；GPU5/1 env/seed42，`last_step=1`。
- conclusion: frozen action evaluator 与无点云 buffer 的工程接线 `SUPPORTED`；Cmv2 prediction quality、候选排序、是否应微调与任何 PPO 效果均 `INCONCLUSIVE`。

PPO observation 仍为 68-D，actor/critic/reward/optimizer 不接收 Cmv2 输出。pre-physics 对实际执行 residual 的
nominal `q -> PD target -> FK -> hand sweep` 做 Cmv2 prediction；post-physics 仅把真实 object local effect 补为
label。生成 shard 不包含 point/normals/flow，而含 `q/dq` pre/post、object pose pre/post、residual/reference/PD
target、authority、reference index、predicted/reference/actual delta xi。manifest 固定 Cmv2 checkpoint/model schema、
reference、URDF/mesh SHA、control dt 和 residual scale，供离线重建。

证据：[运行目录](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/run_manifest.json)、[summary](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/summary.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/config.json)、[CmBuffer manifest](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/cm_buffer/rank_000/manifest.json) 与 [shard](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/cm_buffer/rank_000/chunk_000000.npz)。

## 2026-09-17 — V1.14.4 局部可行动作映射后的 64-step PPO

- modification_version: `V1.14.4`
- operation_category: `code`、`diagnostic`、`experiment`、`operation`
- approval: `user-approved`（[V1.14 最终计划](../plan/V1.14.md) 的 V1.14.3 后局部修订）
- terminal_activity_id: `ACT-20260917-221500-CMRESIDUAL-V1144-FEASIBLE`
- run_id: `cmresidual_v1144_feasible_authority_smoke_20260917_2215`、`cmresidual_v1144_reftrack_feasible_w64_20260917_2215`
- run_status: 两个运行均 `COMPLETED`；PPO `last_step / last_epoch = 40960 / 10`。
- conclusion: 局部 action-boundary、PPO saturation gate 与 checkpoint reload 的工程协议 `SUPPORTED`；reference tracking、residual utility 与抓取效果 `INCONCLUSIVE`。

**协议与结果**

保持 V1.14 的 18-D action、`0.015 m / 0.20 rad / 0.08 rad` configured authority、reference、reward、mimic、PPO 超参数和 `5%` 阈值不变。只对 absolute reference target 已无同方向 margin 的分量，将实际 authority 限为 source/mimic 共同可行余量；zero action 仍逐值返回 reference target。GPU5/1 env smoke 先执行 3 步 zero action、后执行 16 步固定 `±0.25` action：19 步均 finite，target saturation 最大 `0`；zero action 的 authority-limited ratio 为 `0`，nonzero 阶段最高 `0.1111111`，表明边界处理被显式记录而非 clamp。

随后 GPU5、64 env、seed42、random 64-step window、horizon64 的既定 2+8 epoch PPO 正常完成。epoch 1–10 的 residual saturation 全为 `0`，不触发 `>0.05` 停止门；epoch-10 checkpoint 的 model、optimizer、action 均 finite，deterministic reload action 最大差 `0`。这只说明该训练预算不再受 target saturation 阻断；10 个 epoch、单 seed 和未做独立评估不支持对 reward、tracking 或抓取收益作科研结论。

**证据**

- [单环境 smoke 目录](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/run_manifest.json)、[summary](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/summary.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/metrics.jsonl)、[日志](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/smoke.log)。
- [PPO 目录](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/metrics.jsonl)、[train.log](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/checkpoint_validation.json)、[epoch-10 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/continuation/CmResidualGrabReferenceTransition_continuation/nn/last_CmResidualGrabReferenceTransitionPPO_ep_10_rew__-2736.07_.pth)。

## 2026-09-17 — V1.13.4 GRAB retargeted base 探索性 PPO

- modification_version: `V1.13.4`
- operation_category: `experiment`、`operation`
- approval: `user-approved`（明确批准 [V1.13 PPO 替代方案](../plan/V1.13.md) 越过原追踪门禁）
- terminal_activity_id: `ACT-20260917-211422-CMRESIDUAL-V1134`
- run_id: `cmresidual_v1134_grab_ppo_20260917_2115`
- run_status: `FAILED`（epoch 10 结束后饱和门禁失败）
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`（dirty worktree 包含先前 V1.13 及用户改动）
- initial_checkpoint: `null`；last_step / last_epoch: `20480 / 10`；best_metric/best_checkpoint: 不适用（无完成 episode）；latest_checkpoint: epoch 10，路径见下。
- conclusion: 探索协议 `INVALID_IMPLEMENTATION`；PPO 接线及 checkpoint 可重载的工程证据局部 `SUPPORTED`；科研效果 `INCONCLUSIVE`。

**假设与协议**

检验在直接 GRAB 重定向 base、400/40 腕部 PD、原 18-D residual/68-D observation/reward、固定 reference clock 下，单轨迹 PPO 能否正常更新，并在 10 epochs 预算内获得可解释的训练反馈。GPU5、64 env、seed42、horizon32、minibatch2048，先运行 2 epochs，再从保存的 checkpoint 续至 epoch 10。旧追踪门禁未通过；本次无物体零 residual Gate、无 Cmv2、无多 seed，不作为效果对照。

**结果与解释**

- 2+8 epochs 均实际执行，共 `20480` env-steps；10 行逐 epoch 指标 finite。epoch 1/2 的 residual 饱和率为 `0/0`；epoch 10 上升至 `0.107638888`，超过预设 `0.10` 的工程停止阈值。epoch 10 residual RMS=`0.1144237`，success fraction=`0`。
- 每环境只经历 `10×32=320` 步，而完整 episode 为 `366` 步。训练日志明确报告没有环境完成过 episode；因此 `last_*_rew_-inf.pth` 中的 `-inf` 是缺少 episode return 的记录，不是负无穷的物理 reward 或策略表现。success fraction 的零值亦不能当作完整轨迹失败率。
- epoch-10 checkpoint 的模型、optimizer、action 都有限；固定 sigma 约 `0.1`，确定性加载两次输出动作最大差 `0`。这仅支持训练接线可运行；无法判断收敛、抓取提升或 residual/Cmv2 效果。训练未追加预算。

**证据**

- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/)、[config.json](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/metrics.jsonl)、[train.log](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/train.log)、[checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/checkpoint_validation.json)。
- [epoch-2 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/stage2/CmResidualGrabRetargetedPPO_stage2/nn/last_CmResidualGrabRetargetedPPO_ep_2_rew_-inf.pth)、[epoch-10 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/stage10/CmResidualGrabRetargetedPPO_stage10/nn/last_CmResidualGrabRetargetedPPO_ep_10_rew_-inf.pth)、[活动记录](activity_log.md)。

## 2026-09-16 — V1.10.1 deterministic A-E10 / B-E10 配对评估

- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- approval: `user-approved`（[V1.10 最终计划](../plan/V1.10.md)）
- terminal_activity_id: `ACT-20260916-205028-CMRESIDUAL-V1101-CONTROL-E10`、`ACT-20260916-213422-CMRESIDUAL-V1101-CRITIC-CM-E10`
- run_id: `cmresidual_v110_control_e10_20260916_205028`、`cmresidual_v110_critic_cm_e10_20260916_213422`
- run_status: A/B 均为 `COMPLETED`（`367/367` steps）
- base_commit: `759e732d9c359c883c49c2b912009e357ed74bc0`（仅有运行记录及无关 ObjectInteractionCm 工作树改动）
- initial_checkpoint: A/B 分别为各自 epoch-10 checkpoint；SHA256 见下方证据，未交叉加载
- last_step / last_epoch: A/B 均为 `367 / 10`
- best_metric: A/B 的 `mean_env_max_lift_m` 分别为 `0.0849745274 / 0.0850966126 m`
- conclusion: `SUPPORTED`（两侧 preservation 工程 gate）；critic-only Cm utility `INCONCLUSIVE`

**假设与冻结条件**

比较仅问：在 V1.10 的同一 2005-D environment observation、1442-D actor prefix、seed42、GPU5、64 env、
367 steps、terminate-on-success 和 deterministic mean-action 协议下，B critic 额外读取冻结的 OI-Cm 563-D
上下文后，相对 A 是否出现可复核的单 seed 行为差异。A/B 与 fresh [B0 manifest](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json)
的 `protocol` 完全相同；各自读取对应 T10 epoch-10 checkpoint，输入 SHA、reward、residual scale 与 B0 相同。
两侧评估均为 367 行，observation/target finite 全为 true。训练、评估均不改变模型、数据或环境合同。

**deterministic 评估结果**

| 指标 | B0（zero residual） | A control | B critic-Cm | B−A | B 相对 A |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean env max lift（m） | 0.084895 | 0.084975 | 0.085097 | +0.000122 | +0.14% |
| mean contact occupancy | 0.225852 | 0.315250 | 0.197420 | −0.117830 | −37.38% |
| max success fraction | 0.31250 | 0.18750 | 0.34375 | +0.15625 | +83.33% |
| final success rate | 1.00000 | 1.00000 | 1.00000 | 0 | 0% |
| completed episode count | 328 | 357 | 294 | −63 | −17.65% |
| mean completed episode return | 6.634963 | 6.944557 | 6.968346 | +0.023789 | +0.34% |

- A 相对 B0 的 lift/contact preservation ratio=`1.000942 / 1.395830`；B=`1.002380 / 0.874114`。
  两侧均超过预注册的 `0.8 / 0.8` 门槛，因此只支持“10-epoch 训练后未明显破坏该 base 行为”的工程判断。
- B 的 contact occupancy 明显低于 A，而 lift 几乎相同；max success fraction 较高，但 completed episode
  count 更低、tip distance mean 更高（A/B=`0.154581 / 0.220549 m`）。方向混合，不能据此宣称 B 改善抓取。
- completed episode count 两侧均大于零，因此 mean completed episode return 可解释；其 B−A 差值仅
  `+0.023789`。final success rate 是最后一步的运行指标，两侧同为 `1.0`，不代表全轨迹成功率相同。

**T10 学习动态诊断**

以下为同一训练 `metrics.jsonl` 的 epoch 1→10 值；只能描述优化过程，不直接验证 advantage 更稳定或 Cm utility。

| 指标 | A control | B critic-Cm |
| --- | ---: | ---: |
| actor loss | −0.013783→−0.014009 | −0.014678→−0.017789 |
| critic loss | 1.173944→0.065594 | 1.313784→0.067126 |
| entropy | −15.905638→−15.905638 | −15.905638→−15.905638 |
| KL | 0.003009→0.008693 | 0.002035→0.010509 |
| learning rate | 0.000675→0.001013 | 0.000675→0.001519 |
| residual RMS | 0.103212→0.103786 | 0.103212→0.102356 |
| saturation ratio | 0→0.002604 | 0→0.001736 |
| success fraction | 0.187500→0.015625 | 0.046875→0 |

**结论边界**

两侧运行与 preservation gate 为工程 `SUPPORTED`。A/B 只做了一个 seed、一个 airplane 场景、固定顺序的
独立 GPU PhysX rollout；尽管启动静态 RNG/action parity 合同通过，轨迹并非逐步 bitwise 配对。
指标方向混合，未估计跨 seed 方差或统计显著性，故 critic-only Cm utility 维持 `INCONCLUSIVE`。
本轮按最终计划停在 E10，不执行 T20、多 seed 或额外调参；这些结果也不评价 DexYCB 泛化。

**证据**

- A：[运行目录](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028)、[manifest](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/metrics.jsonl)、[eval log](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/eval.log)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth)（SHA256 `fdcf7e00f3d8f250686c4fb30d3527577db5eb9fbfe2e6db39be29f8b6009278`）
- B：[运行目录](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422)、[manifest](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/metrics.jsonl)、[eval log](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/eval.log)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth)（SHA256 `1a12b955cd0cefcecfde3560df235beb0408ec7340784fa18250359a54776cb1`）

## 2026-09-16 — V1.10.1 paired A-control / B-critic-Cm T10

- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- approval: `user-approved`（[V1.10 最终计划](../plan/V1.10.md)）
- terminal_activity_id: `ACT-20260916-183934-CMRESIDUAL-V1101-CONTROL-T10`、`ACT-20260916-192729-CMRESIDUAL-V1101-CRITIC-CM-T10`
- run_id: `cmresidual_v110_control_t10_20260916_183934`、`cmresidual_v110_critic_cm_t10_20260916_192729`
- run_status: A/B 均为 `COMPLETED`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`（dirty worktree 为已批准 V1.10/V1.11 实现）
- initial_checkpoint: A/B 均为 `null`（从零初始化，未加载 smoke、V1.8 或对侧 checkpoint）
- last_step / last_epoch: A/B 均为 `20480 / 10`
- best_metric: A/B 的 `checkpoint_reload_action_max_abs_diff` 均为 `0`
- conclusion: `SUPPORTED`（配对 T10 工程门禁）；Cm utility `INCONCLUSIVE`

**协议与结果**

A/B 固定 GPU5、64 env、seed42、horizon32、minibatch2048，均构造 2005-D environment observation，actor
只读相同 1442-D base prefix；A critic 读取 1442-D，B critic 读取完整 2005-D。两侧 matched actor state、
suffix/cross-variant mean、sigma、post-build CPU RNG 和首个 stochastic action parity 全部通过。模型、optimizer、
deterministic action 均 finite，sigma min/max=`0.099999994` 且冻结，checkpoint reload diff 均为 `0`。

- A epoch-10：critic loss=`0.0655944`、KL=`0.00869283`、success=`0.015625`、residual RMS=`0.103786`。
- B epoch-10：critic loss=`0.0671258`、KL=`0.0105094`、success=`0`、residual RMS=`0.102356`。
- 这些训练诊断量不等于 deterministic preservation 或 Cm 效果证据；截至本 T10 终态记录时 A-E10/B-E10
  尚未运行，不能据此比较最终抓取行为，也不能从单 seed 推断统计显著性。后续 E10 结果见上方记录。

**证据**

- A：[manifest](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/run_manifest.json)、
  [metrics](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/metrics.jsonl)、
  [validation](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/checkpoint_validation.json)、
  [log](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/train.log)、
  [checkpoint](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth)
- B：[manifest](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/run_manifest.json)、
  [metrics](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/metrics.jsonl)、
  [validation](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/checkpoint_validation.json)、
  [log](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/train.log)、
  [checkpoint](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth)

## 2026-09-16 — V1.11.2 DexYCB reset 修复后 GPU5 base-only Gate

- modification_version: `V1.11.2`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- terminal_activity_id: `ACT-20260916-182728-CMRESIDUAL-V1112-DEXYCB-GPU5`
- run_id: `cmresidual_v1112_dexycb_base_gpu5_20260916_182728`
- run_status: `COMPLETED`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`（dirty worktree 为已批准 V1.10/V1.11 实现）
- initial_checkpoint: frozen DExplore `inspire.pth`，SHA 见 run manifest
- checkpoint: `null`
- last_step / last_epoch: `72 / null`
- best_metric: `min_tip_distance_m=0.2998754382`（描述量，不是成功指标）
- conclusion: `SUPPORTED`（工程 Gate C）；固定单轨迹成功抓取 `REFUTED`；DexYCB 泛化 `INCONCLUSIVE`

**协议与修复**

V1.11.1 的 tensor reset 在 actor 创建后才写 q，导致首次 rigid-body tensor 仍是零 DOF pose。V1.11.2 仅对
DexYCB config opt-in actor-creation q/dq/target 初始化，wrist reset error 降为 `1.17e-07 m`。GPU6 被外部任务
占用、V1.10 B0 已结束后，用户明确批准同协议迁移到空闲 GPU5；序列、物体、reference/source、checkpoint、
seed42、4 env、72 steps 和 strict zero residual 均不变。

**结果**

- 运行正常退出；72 行 observation/target/object pose finite，max residual target delta=`0`，初始 native q/
  wrist/object/table error=`0 / 1.17e-07 m / 1.11e-16 m / 0`，工程 Gate C 为 `SUPPORTED`。
- mean/min tip distance=`1.056172/0.299875 m`；实测接触仅 12/72 steps，mean/max occupancy=
  `0.015972/0.20`，而 reference mean occupancy=`0.636111`。
- mean/max lift=`0.022967/0.139027 m`，但最大 lift 出现在 step 8 且 contact occupancy=`0`；接触首次出现于
  step 11，此后 lift 持续下降。该抬升不能归因于手部抓取。
- final step tip distance=`2.994933 m`、root/tip reference error=`2.579618/2.513221 m`、contact=`0`。
  因此 frozen checkpoint 在这条固定 DexYCB trajectory 上没有成功跟踪或抓取；单轨迹证据不能推出所有
  DexYCB 物体都无法抓取，也不评价 residual/Cm。

证据：[run manifest](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/run_manifest.json)、
[config](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/config.json)、
[metrics](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/metrics.jsonl)、
[log](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/eval.log)。

## 2026-09-16 — V1.10.1 fresh B0-Cm-path baseline

- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- approval: `user-approved`
- activity_id: `ACT-20260916-170651-CMRESIDUAL-V1101-B0`
- run_id: `cmresidual_v110_b0_cm_path_20260916_170651`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 17:48:16 +0800`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- last_step / last_epoch: `367 / null`
- checkpoint: `null`（严格 zero residual baseline）
- best_metric: `mean_env_max_lift_m=0.0848945305`
- conclusion: `SUPPORTED`（zero residual parity 与正 preservation baseline）；Cm utility `INCONCLUSIVE`

GPU5、64 env、seed42、2005-D Cm 环境路径完成 367 steps；mean contact occupancy=`0.2258515`、
mean tip distance=`0.1705366 m`、max residual target delta=`0`。运行未被 V1.11 的 GPU6 工作停止或修改。
证据：[manifest](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json)、
[metrics](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/metrics.jsonl)、
[log](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/eval.log)。该 baseline 只允许进入
V1.10 后续 preservation gate，不证明 Cm 改善或统计显著性。

## 2026-09-16 — V1.11.1 DexYCB frozen-DExplore base-only Gate

- modification_version: `V1.11.1`
- operation_category: `data`、`experiment`、`operation`、`diagnostic`
- approval: `user-approved`
- terminal_activity_id: `ACT-20260916-174041-CMRESIDUAL-V1111-DEXYCB`
- run_id: `cmresidual_v111_dexycb_base_20260916_174041`
- run_status: `COMPLETED`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`（dirty worktree 为已批准 V1.10/V1.11 实现）
- initial_checkpoint: frozen DExplore `inspire.pth`，SHA 见 run manifest
- checkpoint: `null`
- last_step / last_epoch: `72 / null`
- conclusion: `INVALID_IMPLEMENTATION`（reset wrist/FK 对齐）；DexYCB base 泛化 `INCONCLUSIVE`

**假设与协议**

固定使用首个合格 right sequence `subject-10/20201022_110806`、抓取物体 `002_master_chef_can`，将同一
DexYCB→sim SE(3) 应用于 object/MANO，固定 canonical fingertip identity retarget 到 Inspire，并在独立 YCB
物理资产中以 frozen DExplore、OI-Cm disabled、严格零 residual、GPU6、seed42、4 env × 72 steps 评估。
lift 仅描述，不沿用 airplane `0.08 m` 成功阈值。

**离线结果**

Gate A/B 为工程 `SUPPORTED`：72 帧 reference 与 `[74,598]` padded source finite，retarget fingertip RMS
`32.386 mm`，相对 neutral `47.579 mm` 改善 `31.933%`；joint/q velocity/q acceleration/坐标回代通过，
相关测试 `32 passed`。证据见 [reference manifest](../../../../../data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806/manifest.json)。

**物理结果与失效解释**

运行正常退出并写满 72 行；zero residual target delta 始终为 `0`，observation/target/object pose 全程 finite。
但初始 native q/object/table 对齐通过时，wrist body 对离线 FK 仍有 `0.486826 m` 误差，违反 reset hard gate。
因此 manifest 正确标记 `COMPLETED / INVALID_IMPLEMENTATION`；mean/max lift、tip distance、contact 和 tracking
数值仅是无效实现下的诊断量，不可用于回答 frozen DExplore 在 DexYCB 上是否抓取或泛化。证据：
[manifest](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041/run_manifest.json)、
[metrics](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041/metrics.jsonl)、
[log](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041/eval.log)。

按 [V1.11 最终计划](../plan/V1.11.md) 的失败停止条件，本轮未改变轨迹、物体、阈值或 checkpoint 重跑，
未训练 residual/Cm，也未影响仍在 GPU5 运行的 V1.10 B0。后续若要修复 reset 同步并复跑，需重新确认范围。

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
## 2026-09-17 — V1.13.1 GRAB 确定性 base 无接触追踪 Gate

- modification_version: `V1.13.1`
- operation_category: `experiment`、`diagnostic`
- activity_id: `ACT-20260917-203900-CMRESIDUAL-V1131`
- run_id: `cmresidual_v113_grab_nominal_20260917_2039`
- run_status: `COMPLETED`
- initial_checkpoint: `null`（base 不加载 DExplore；residual 全零）
- last_step: `366`
- conclusion: 工程接线 `SUPPORTED`；无接触追踪是否足以进入交互 PPO 为 `INCONCLUSIVE`。

**假设与协议**

GRAB 现成重定向 `reference_tracking_v2/s1_airplane_lift` 提供 367 帧、30 Hz、train 且 `training_eligible=true` 的腕部世界位姿与手指 native q。将物体从 reset pose 沿 x 移出 10 m，仅检查绝对 PD 目标与零 residual 的手部追踪。单环境 CPU、seed 42、固定参考时钟，step 1–366 对应 reference index 1–366；该操作不回答真实物体接触或抓取效果。

**结果**

运行 [目录](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/run_manifest.json)、[逐步指标](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/metrics.jsonl) 与 [日志](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/eval.log) 记录了 366 步。全程 finite，hand contact occupancy 和 residual saturation 均为零。手腕位置误差平均 `0.01120 m`、p95 `0.02924 m`、最大 `0.06276 m`；指尖误差平均 `0.01395 m`、p95 `0.03775 m`、最大 `0.07101 m`；手指 q 绝对误差平均 `0.00855 rad`。按 step 对齐原 GRAB 源接触标记选取 296 帧，指尖误差平均 `0.01286 m`、p95 `0.03703 m`、最大 `0.04245 m`。此前同段文字的接触帧统计误将 step 1–366 对齐到 reference 0–365，已按 V1.13.2 诊断纠正。

绝对目标与零 residual 的工程接线成立；接触帧高分位误差大于 Cmv2 `0.02 m` interaction radius，不能由此宣称名义 PD 足够精准，也不能开始 PPO 或将误差归因为物体交互。下一步应先诊断参考时间对齐与控制器滞后，再按计划协商是否调整低层控制或训练门禁。更早的 [首次运行](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2036/) 在终止步读取了自动 reset 后的 native q，指尖/手腕 info 未受影响，但末步手指 q 误差无效；本条只采用修复终止步读取后的第二次运行。
## 2026-09-17 — V1.13.2 固定时钟无接触腕部滞后诊断

- modification_version: `V1.13.2`
- operation_category: `diagnostic`、`experiment`、`operation`
- activity_id: `ACT-20260917-204549-CMRESIDUAL-V1132`
- run_id: `cmresidual_v1132_grab_lag_20260917_2046`
- run_status: `COMPLETED`
- last_step: `366`；initial_checkpoint: `null`；best_metric/checkpoint: 不适用。
- conclusion: 固定时钟下约两帧腕部位置滞后的诊断证据 `SUPPORTED`；增益修正效果、交互训练资格与科研效果 `INCONCLUSIVE`。

与 V1.13.1 完全相同的 GRAB train reference、CPU/1 env、seed42、物体移出接触、零 residual 和 366 步协议，只在逐步指标中增记实际/参考腕部世界位置。证据：[运行目录](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/)、[配置](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/config.json)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/metrics.jsonl)、[日志](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/eval.log)。先前同协议运行 [目录](../../../../../outputs/CmResidual/cmresidual_v113_grab_lag_20260917_2043/) 的 manifest 误记 `V1.13.1`，不作为规范证据；两次逐步指标 SHA256 一致，旧目录只作审计保留。

用 `step=1..366` 对齐 reference index 1..366，并按原 GRAB 源 tensor 同 index 的接触标记选出 296 帧。同帧腕部误差 mean/p95=`0.01000/0.02762 m`；实际腕部位置与两帧前参考位置比较的 mean/p95=`0.00264/0.00740 m`，在测试的 lag `-3..+6` 中最小。腕部同帧误差与参考相邻帧位移的 Pearson 相关系数约 `0.898`。接触标记帧中，腕部误差超过 `0.015 m` 的比例 `27.03%`，指尖误差超过 `0.020 m` 的比例 `26.69%`。这些是滞后诊断与门禁风险，不证明提高 PD 增益必然有效，也不证明 residual PPO 可训练。V1.13.1 原记录的接触帧统计存在一帧对齐错误，已在原条目标明修正值。
## 2026-09-17 — V1.13.3 GPU5 腕部 PD 增益无接触对照

- modification_version: `V1.13.3`
- operation_category: `experiment`、`operation`、`diagnostic`
- activity_id: `ACT-20260917-205100-CMRESIDUAL-V1133`
- run_ids: `cmresidual_v1133_gpu5_kp200_kd20_20260917_2051`、`cmresidual_v1133_gpu5_kp300_kd30_20260917_2056`、`cmresidual_v1133_gpu5_kp400_kd40_20260917_2057`
- run_status: 三组均 `COMPLETED`，每组 `last_step=366`；初始 checkpoint、best metric/checkpoint 均不适用。
- conclusion: 三组都未满足预注册无接触追踪 Gate，对“三组任一可直接达到此门禁”为 `REFUTED`；训练可行性、速度前馈效果、抓取与 Cm 效果仍为 `INCONCLUSIVE`。

**协议**

同一 GRAB train 参考、GPU5、GPU pipeline、seed42、1 env、366 步、物体移出接触、零 residual、canonical `num_subscenes=4` 与相同 PhysX buffer。仅腕部六 DOF position drive 的 stiffness/damping 分别取 `200/20`、`300/30`、`400/40`；手指 drive、reference clock、base target、reward 与其他物理设置不变。每组在独立进程与运行目录完成，不用此前 CPU 结果计算增益收益。依据原 GRAB 接触标记按 step 1–366 对齐后统计 296 帧，预设门禁为腕部位置误差 p95 `≤0.015 m`、指尖误差 p95 `≤0.020 m`，且 finite、零 residual target 差异、无接触。

| 腕部 stiffness/damping | 接触帧腕部 p95 | 接触帧指尖 p95 | 腕部超过 1.5 cm | 指尖超过 2 cm | 最佳腕部位置对齐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `200/20` | `0.027618 m` | `0.037040 m` | `27.03%` | `26.69%` | 滞后 2 帧 |
| `300/30` | `0.026923 m` | `0.036556 m` | `25.68%` | `26.35%` | 滞后 2 帧 |
| `400/40` | `0.026527 m` | `0.036484 m` | `24.66%` | `25.34%` | 滞后 2 帧 |

三组均 366 行、有限、手部接触 occupancy 为零、零 residual target 差异约 `1.19e-07` 的浮点精度。增益上升后误差仅小幅下降，两项 p95 仍超门禁，且两帧滞后未消失。因此按 V1.13 最终计划停止搜索，没有继续加大增益、启动有物体 Gate 或 PPO。该单序列、单 seed 结果只否定预定三组参数在该门禁下的通过性，不否定其他低层控制方案。

**证据**

- `200/20`：[目录](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/)、[配置](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/config.json)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/metrics.jsonl)、[日志](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/eval.log)。
- `300/30`：[目录](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/)、[配置](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/config.json)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/metrics.jsonl)、[日志](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/eval.log)。
- `400/40`：[目录](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/)、[配置](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/config.json)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/metrics.jsonl)、[日志](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/eval.log)。

## 2026-09-17 — V1.14.1 GRAB 真实物体 zero-residual reference 基线

- modification_version: `V1.14.1`
- activity_id: `ACT-20260917-213945-CMRESIDUAL-V1141-PHYSICAL-ZERO`
- run_id: `cmresidual_v1141_grab_physical_zero_20260917_2139`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（object transition 指标与物理工程接线）；reference tracking、residual utility 和抓取效果 `INCONCLUSIVE`。

**协议与结果**

固定 GRAB training-eligible reference、400/40 wrist drive、GPU5/1 env/seed42、真实 object、zero residual 和
366 steps；`terminateOnSuccess=false`。每个执行后状态严格与 `reference_index=step` 对齐；同时记录 actual/ref
world object pose、SO(3) rotation error，以及 local `T[t-1]^-1 T[t]` transition error。运行 [目录](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/)、[日志](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/eval.log) 证明全程 finite、正常退出且 zero target delta 最大 `1.19e-07`。

object position error mean/p95/max=`0.047775/0.164753/0.221184 m`，rotation error=`0.268102/0.543927/0.599684 rad`；一步 transition translation error=`0.004155/0.015792/0.020499 m`，rotation error=`0.018703/0.051554/0.170197 rad`。因此 deterministic base 在本真实物体 rollout 中并没有精确复现 object pose trajectory；该单 env/seed 基线不评价 residual 改善或抓取，且不能由工程正常结束推断科研效果。它只满足 V1.14 的实现前置工程门，允许进入冻结的 reference-tracking reward/window 实现。

## 2026-09-17 — V1.14.2 64-step reference-transition PPO smoke

- modification_version: `V1.14.2`
- activity_id: `ACT-20260917-214656-CMRESIDUAL-V1142-W64-PPO`
- run_id: `cmresidual_v1142_reftrack_w64_20260917_2147`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（预注册 saturation Gate）；reset/reward/checkpoint 工程接线局部 `SUPPORTED`；科研效果 `INCONCLUSIVE`。

固定 GRAB reference、random-uniform start `[0,302]`、64-step window、GPU5/64 env/seed42/horizon64。reward 固定为 object pose/transition tracking 的 `0.02 m/0.05 rad` 归一化负加权和与 action penalty；Cmv2 未接入。2 epochs/8192 env-steps 完成，epoch-2 checkpoint finite 且可重载，但 saturation 在 epoch 1/2 为 `5.3819%/4.8611%`；按 `>5%` 停止条件不续跑 epoch 3–10，也不进入 128/366。证据：[manifest](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/metrics.jsonl)、[log](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/train.log)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/smoke/CmResidualGrabReferenceTransition_smoke/nn/last_CmResidualGrabReferenceTransitionPPO_ep_2_rew__-1168.13_.pth)。

tracking 曲线（epoch 1/2）position=`0.066881/0.181973 m`、rotation=`0.535478/0.701575 rad`、transition translation=`0.003643/0.006629 m`、transition rotation=`0.017725/0.041125 rad`。这不足以证明新 reward 改善或反驳该研究假设：本轮按工程饱和门终止，且只有单 seed/两 epochs。下一步若要改阈值、action scale、reward 权重、控制器、PPO 超参数或继续预算，必须新计划和用户批准，不能在 V1.14 中追调。

## 2026-09-18 — V1.15 Cmv2 actor wiring smoke

- modification_version: `V1.15`
- operation_category: `experiment`、`operation`
- activity_id: `ACT-20260918-110000-CMRESIDUAL-V115-ACTOR`
- run_id: `cmresidual_v115_cmv2_actor_smoke_20260918_1100`
- run_status: `COMPLETED`
- last_step/last_epoch: `8` / `1`；checkpoint reload action diff=`0.0`。
- conclusion: implementation/wiring `SUPPORTED`；attention utility、effect accuracy、PPO tracking、Cmv2 微调资格与抓取效果 `INCONCLUSIVE`。

冻结 Cmv2 只从 action 前的 current state 和 zero residual nominal controller/FK sweep 产生 16×40 token、predicted/reference/error local effects；PPO actor transport 为 726-D、attention 后为 214-D，critic 始终只读 68-D base prefix。GPU5/1 env/seed42、horizon/minibatch=8、1 epoch 实际完成；checkpoint、optimizer 和动作有限，sigma=`0.1`、residual saturation=`0`。由于仅 8 steps 且无 episode 终结，reward 为 `-inf`，不能用它比较策略质量。CmBuffer 生成 8 个 finite shard，且不含 point/normals/flow。证据：[运行目录](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/metrics.jsonl)、[log](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/checkpoint_validation.json)、[CmBuffer manifest](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/cm_buffer/rank_000/manifest.json)。

## 2026-09-18 — V1.16 GPU nominal Cmv2 / transition-only wiring smoke

- modification_version: V1.16
- operation_category: experiment、operation
- activity_id: ACT-20260918-003000-CMRESIDUAL-V116-GPU
- run_id: cmresidual_v116_gpu_nominal_smoke_20260918_0030
- run_status: COMPLETED
- last_step/last_epoch: 8 / 1；checkpoint reload action diff=0.0。
- conclusion: GPU nominal path、transition-only buffer 与 base68 RMS implementation SUPPORTED；DDP capacity、attention utility、PPO tracking 和抓取效果 INCONCLUSIVE。

固定 reference/SHA、GPU5/1 env/seed42、horizon/minibatch=8、1 epoch。actor 由实际 GPU hand link poses 和 next reference GPU link poses 构造 nominal sweep；训练日志无 legacy CPU FK warning。checkpoint、optimizer 和 action 均 finite，sigma=0.1、residual saturation=0。transition-only v2 buffer 有 8 个 finite sample，精确含 16 个 pre/action/reference/post/actual-effect 字段，不含 executed-action prediction 或任何 point/normal/flow。该短 smoke 在 episode 结束前达到 epoch cap，reward=-inf 不构成策略质量证据。三卡 capacity 未启动，因为 GPU2 当前约 23 GiB 已占用。证据：[运行目录](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/metrics.jsonl)、[log](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/checkpoint_validation.json)、[v2 buffer manifest](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/cm_buffer/rank_000/manifest.json)。

## 2026-09-18 — V1.16.4 GPU0/1/3 DDP 3×128 capacity probe

- modification_version: V1.16.4
- activity_id: ACT-20260918-010932-CMRESIDUAL-V1164-MEMPEAK
- run_id: cmresidual_v1164_ddp_3x128_mempeak_20260918_010932
- run_status: COMPLETED
- conclusion: SUPPORTED（3×128 DDP 工程 capacity）；3×256 capacity、长期 PPO 训练与科研效果 INCONCLUSIVE。

固定 pinned GRAB reference/SHA、seed42、physical GPU0/1/3、3 rank×128 env、horizon32、1 epoch。完成 12288 total steps；三个 rank 的 transition-only v2 buffer 各写 4096 sample，checkpoint/optimizer/action finite、deterministic reload diff=0。0.5 秒采样的物理显存峰值为 `16440/14137/15358 MiB`，总 throughput 约 1069 FPS。最满 GPU0 尚余约 8.1 GiB；该单点不支持安全外推至 256 env/rank，且 epoch 内没有完整 episode、reward=-inf，不能用于效果比较。证据：[运行目录](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/metrics.jsonl)、[log](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/checkpoint_validation.json)、[rank 0 buffer](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/cm_buffer/rank_000/manifest.json)、[rank 1 buffer](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/cm_buffer/rank_001/manifest.json)、[rank 2 buffer](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/cm_buffer/rank_002/manifest.json)。
