# CmDecoderv2 实验记录

## 2026-09-13 — V1.1.16 并行离线消融与 GPU contact 能力核验

- modification_version: `V1.1.16`；operation_category: diagnostic / experiment / operation；approval: user-approved。
- 计划：[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)；状态唯一入口：[src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md)。
- B1 主训练继续在 GPU7 运行；本次 GPU1 固定 epoch3 checkpoint，GPU0 单独执行 instrument-only
  contact probe，CPU 核验 D 输入来源；没有改变训练、GT、physics 或主 run checkpoint。

### B1/C0/Cs 离线诊断

- 固定 checkpoint：`step_000012969_epoch_000003.pt`，SHA256 记录在本次 manifest；不是读取不断变化的 best.pt。
- 全部 30 val sequences、4254 windows，active horizon 共16260，active h1共4065；C0置零raw F7，
  Cs以seed42对每个window取独立anchor permutation，K内共享，anchor geometry不动。

| 条件 | active sample-horizon point EPE/mm | h1 EPE/mm | 相对B1的预测腕平移变化/mm |
| --- | ---: | ---: | ---: |
| B1 | 31.96618123 | 13.64633166 | 0 |
| C0 | 31.96618122 | 13.64633171 | 0.00001262 |
| Cs | 31.96618118 | 13.64633168 | 0.00000064 |

- q预测变化：C0 `4.23e-8 rad`，Cs `2.05e-9 rad`。该中间checkpoint对F7干预几乎不敏感，
  但不能据此否定F7 representation；需要区分训练/实现/信息合同，并以终态统一权重复核。
- 本次用全体active sample-horizon的micro汇总；训练日志用batch汇总，因此31.966与此前32.054
  不能被解释为性能提升。结果只是offline sensitivity，未发生物理rollout。
- conclusion: `SUPPORTED`（本checkpoint输出对这些干预近乎不变）；`INCONCLUSIVE`（Field增量控制价值）。
- 证据：[outputs/cmdecoderv2/field_conditions_epoch3_v116_20260913_234700/run_manifest.json](../../../../../outputs/cmdecoderv2/field_conditions_epoch3_v116_20260913_234700/run_manifest.json)、[outputs/cmdecoderv2/field_conditions_epoch3_v116_20260913_234700/evaluation.json](../../../../../outputs/cmdecoderv2/field_conditions_epoch3_v116_20260913_234700/evaluation.json)。

### 真实 GPU contact probe

- 使用contact50 source、64env、原GPU pipeline，连续4个控制步后查询env0；Gym明确返回
  `GymGetEnvRigidContacts cannot be used with the GPU pipeline after simulation starts`，同时返回空数组。
- 先前静态hasattr检查不足以通过此Gate；现有tracker的空接触结果不能作为有效指标。按final plan
  暂停依赖pairwise指标的正式物理Gate，不用net force替代、不切CPU physics。
- conclusion: `REFUTED`（此API可用于当前GPU pipeline的假设）；`INVALID_IMPLEMENTATION`（现有tracker作为正式接触评估器）；完整Gate仍`INCONCLUSIVE`。
- 证据：[outputs/cmdecoderv2/field_contact_gpu_probe_v116_20260913_234820/capability.json](../../../../../outputs/cmdecoderv2/field_contact_gpu_probe_v116_20260913_234820/capability.json)、[outputs/cmdecoderv2/field_contact_gpu_probe_v116_20260913_234820/train.log](../../../../../outputs/cmdecoderv2/field_contact_gpu_probe_v116_20260913_234820/train.log)。

### D 数据来源核验

- 284/284原始GRAB序列包含24D hand_pose、orient/transl和存在的subject template；parent与actual
  Inspire raw frame ids逐位一致。此前“MANO-H数据不足”的表述过强，准确状态是尚未完成D接线。
- 现有GRABSeqData在缺显式betas时采用zero beta + subject template；parent wrist原点是MANO
  joint0，不能直接拿raw transl代替。mesh/template parity和D相同object-geometry输入仍待实现，未开训。
- conclusion: `SUPPORTED`（来源可用、帧对齐）；`INCONCLUSIVE`（几何等价及D控制效果）。
- 证据：[src/task/CmDecoderv2/research/field_realizer_gate/output/mano_h_availability_v116_20260913_235000/availability.json](../../research/field_realizer_gate/output/mano_h_availability_v116_20260913_235000/availability.json)。

## 2026-09-13 - V1.1.15 接触起点残差抬升 pilot

- 假设：把 episode 初始化切到原参考轨迹第50帧、避开远距离接近阶段后，冻结 Cm base 上的残差策略是否能改善物体抬升。
- 固定：`s1/airplane_lift` train 单序列、decoder epoch14 best、OICM/奖励/动作参数化/PPO 超参不变、64env、seed42、100更新/204800 samples。source 从第50帧截取378个窗口，valid比例0.8756614；不提供未来 Inspire 目标动作。
- source run：[contact50 source](../../../../../outputs/cmdecoderv2/rl_online_source_airplane_contact50_v15_20260913/manifest.json)；gate run：[contact50 gate](../../../../../outputs/cmdecoderv2/rl_online_gate64_contact50_v15_20260913/gate.json)。

| 64个同初态 episode | 零残差 | epoch100残差，独立重载 |
| --- | ---: | ---: |
| 平均累计回报 | -258.1657 | -341.9249 |
| 平均最大抬升 | 2.4400 mm | 2.7103 mm |
| 最大抬升 | 2.4401 mm | 15.5768 mm |
| 瞬时抬升>8cm成功率 | 0/64 | 0/64 |

- 训练输出：[contact50 PPO](../../../../../outputs/cmdecoderv2/rl_online_ppo_contact50_v15_20260913/)、[metrics](../../../../../outputs/cmdecoderv2/rl_online_ppo_contact50_v15_20260913/metrics.jsonl)、[train log](../../../../../outputs/cmdecoderv2/rl_online_ppo_contact50_v15_20260913/train.log)、[最终checkpoint](../../../../../outputs/cmdecoderv2/rl_online_ppo_contact50_v15_20260913/nn/last_rl_online_ppo_contact50_v15_20260913_ep_100_rew_-2719.5344.pth)。重载结果：[evaluation](../../../../../outputs/cmdecoderv2/rl_online_eval_contact50_v15_20260913/evaluation.json)。
- 解释：gate说明接触起点和在线闭环接口有效；100更新没有达到8cm抬升，故“接触起点已解决抬升”被否定。残差策略的最大单次抬升略高于base，但平均值仅高0.27mm，尚不足以支持Cm带来有效抓持控制的结论。
- conclusion: `SUPPORTED`（source截取、物理gate、PPO保存/恢复）；`REFUTED`（本预算内已学会抬升）；`INCONCLUSIVE`（更长训练、奖励设计、接触动力学和Cm实际贡献）。不把该pilot扩展为泛化结果。

## 2026-09-13 - V1.1.15 在线冻结 base 与单序列残差 PPO pilot

- modification_version: V1.1.15；operation_category: code / experiment / operation；approval: user-approved；[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)。
- 假设：在真实airplane/table与实际状态反馈下，冻结Cm base上的12维残差短训能否提高单序列抬升成功率。只检查train `s1/airplane_lift`，不是held-out测试。
- 固定：decoder epoch14 SHA `0814bdabcbdf484d90c6855a50e3bfebc1053b4d085ebde152b15f65aa8c4494`；OICM SHA `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`；K4、30Hz、frame0固定实际native18/object/table初始化、64env、seed42、原奖励/8cm瞬时lift阈值。仅PPO参数更新，100迭代/204800样本，GPU7/graspenv，约123.14秒含训练后回放。
- 训练run_id: `rl_online_ppo_airplane_v15_20260913`；run_status: COMPLETED。CPU9测试、4/64env完整物理gate通过；正式状态和失败尝试见唯一时间线 [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md)。
- 工程证据：Gym/native按名称映射；15mm腕残差产生15.006mm实际位移；64env FK最大位置差1.1341e-6m，query差1.2517e-6；reset隔离和冻结权重逐位检查通过。物理状态不是6维耦合参考重建。

| 64个同初态episode | 零残差 | epoch100残差（重新加载） |
| --- | ---: | ---: |
| 平均原始累计回报 | -1317.6615 | -912.3494 |
| 平均最大抬升/mm | 1.18053 | 1.18042 |
| 瞬时抬升>8cm成功率 | 0/64 | 0/64 |

- 训练滚动episode回报从epoch25约-1409.16升到epoch94约-755.38；这是PPO探索策略的滚动训练统计，不是验证集指标，也不能与表中确定性回放直接混用。按该统计保存的best为epoch94/frame192512；最终epoch100/frame204800。所有保存policy模型tensor有限。
- epoch100内存模型回放回报-912.4229；独立进程重新加载相同checkpoint回报-912.3494，均0/64成功。只证明恢复/完整回放可运行和近似一致，不声称逐位确定性。
- source artifact的frame0窗口4个Cm均invalid，第一个至少1个valid的窗口为50、全4个valid为53（0-based）；原初始化腕部距物体中心1.5238m。在线provider延续原core前向，不添加未批准的hold/oracle接近轨迹；invalid窗口的raw输出不能解释为有物理意义的Cm。
- 首个env非末帧轨迹中，腕物中心距离均值从零残差1.531m降至残差1.061m，最小仍0.643m；这一观察与回报改善主要来自接近项相符，但不是所有env的接触率统计。不能据短训无lift判定Cm不可用，也不能据回报上升声称学会抓取。
- conclusion: SUPPORTED（在线冻结base、物理接口、PPO更新和保存/恢复）；REFUTED（本次100更新策略已学会抬升）；INCONCLUSIVE（接触起点后的残差可学性、Cm跨手和泛化）。64个相同初态env不是64种独立泛化条件。
- 下一步建议仅讨论：先确认并设置原轨迹的有效接触起点，隔离接近与抓持/抬升，再做同预算base/残差对照；不立即把本次frame0条件扩大长训，也不修改Cm主干。该初始化变更尚未执行。
- 证据：[outputs/cmdecoderv2/rl_online_gate64_terminal_v15_20260913/gate.json](../../../../../outputs/cmdecoderv2/rl_online_gate64_terminal_v15_20260913/gate.json)、[outputs/cmdecoderv2/rl_online_ppo_airplane_v15_20260913](../../../../../outputs/cmdecoderv2/rl_online_ppo_airplane_v15_20260913/)、[metrics.jsonl](../../../../../outputs/cmdecoderv2/rl_online_ppo_airplane_v15_20260913/metrics.jsonl)、[training_result.json](../../../../../outputs/cmdecoderv2/rl_online_ppo_airplane_v15_20260913/training_result.json)、[重新加载评估](../../../../../outputs/cmdecoderv2/rl_online_eval_reload_airplane_v15_20260913_b/evaluation.json)。

## 2026-09-13 — V1.1.14 IsaacGymEnvs CmResidual wiring smoke

- modification_version: `V1.1.14`；category: code / operation / diagnostic；approval: user-approved；计划见 [plan/v1.1.md](../plan/v1.1.md)。
- run_id: `cm_residual_isaac_smoke_20260913_001902`；run_status: `COMPLETED`；外部输出：`/home2/wyy/oyx_ws/IsaacGymEnvs/runs/CmResidual_13-00-18-30/nn/last_CmResidual_ep_1_rew_-inf.pth`。
- 固定接口：`airplane_lift`；动作 `Δq6 + Δwrist_SE(3)` 共12维；冻结 decoder/OICM/base；实际 native18 q/dq、base、wrist/object/tip 观测共71维。
- 结果：IsaacGymEnvs 任务注册、18-DOF Inspire asset、GPU PhysX、reset/step、actor forward 和单次 PPO epoch 均通过；Ref2Dex 定向测试17项通过。单epoch没有完整终止episode，`rew=-inf`仅是框架输出，不是策略效果。
- provider 边界：本次只运行 `reference_frozen`；`decoder` 模式在 Cm-bank adapter 缺失时显式失败。root wrist 直接目标设置仅为 wiring smoke，尚未证明真实闭环动力学。
- conclusion: `SUPPORTED`（工程框架接入）；`INCONCLUSIVE`（残差策略收敛、抓取成功、CmDecoder 在线闭环）。正式RL训练待 Cm-bank adapter 和物理状态控制接口完成后再启动。

## 2026-09-12 — V1.1.14 batch8三卡几何监督5epoch初步训练

- modification_version: V1.1.14；category: experiment / operation；approval: user-approved；[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)。
- run_id: `cm_decoder_v2_coupled_geometric_batch8_20260912_225225`；run_status: COMPLETED；8315步/5epoch，40分24秒；最佳epoch5，val/loss=0.009018792。
- 假设/范围：标签FK合同通过后，扩大batch仍可稳定训练；初步观察耦合几何参考解码。不是batch2/8学习效果随机对照，也不是跨手验证。冻结OICM SHA、split、K4/10135点/2cm/KNN32不变；新初始化decoder，每卡8/global24、GPU2/4/5、seed42、FP32、workers0、lr3e-4/cosine5epoch、perturb_train=false，不使用前述无效smoke权重。
- full FK run_id `coupled_fk_gate_v14_20260912_224950`，run_status COMPLETED；285条/72935帧通过。新旧view内容相同，版本锁要求生成新轻量view；原44GiB几何缓存复用。

| epoch | val/loss | val h1表面EPE(mm) | val h1腕平移(mm) |
|---|---:|---:|---:|
| 1 | 0.00943987 | 11.4056 | 12.7611 |
| 2 | 0.00947883 | 11.3262 | 12.7342 |
| 3 | 0.00910965 | 10.9544 | 12.3610 |
| 4 | 0.00906360 | 10.8358 | 12.2591 |
| 5 | 0.00901879 | 10.7753 | 12.1966 |

- 最终四horizon平均表面EPE25.9471mm。h1手指q MAE0.00779017rad，高于identity的0.00618874rad；腕旋转2.57239deg，高于identity的2.47760deg；腕平移12.1966mm低于identity的15.6140mm。只描述同一验证集/同一Runner聚合下的观察，不把全手指标下降解释为每个手指学好或Cm条件因果有效。
- 训练稳定，无OOM或非有限；83段性能窗口均值266.802ms/step，数据等待占46.51%，完整epoch含验证约8.1分钟。同设置50epoch约6.7小时，仅是计算耗时外推，不保证收敛。未改数据加载并行度、损失权重或加入扰动。
- conclusion: SUPPORTED（正确点对应下batch8/global24三卡训练可运行）；INCONCLUSIVE（完整运动解码充分性、Cm跨手、实际状态微调及物理RL成功）。后几轮收益较小且q/旋转未超过identity，不能仅凭此权重宣称RL base已经合格，也不据此否定Cm。
- 限制：单seed、仅几何Inspire参考到耦合Inspire标签、同域val选择checkpoint、无MANO测试/实际反馈/物理rollout；现有指标是Runner按batch/mask的聚合，不作frame micro或sequence macro解释；没有把原始参考适配残差与学习误差相减。
- 权重验收：strict load全key匹配，全部权重有限，60个OICM tensor相对原checkpoint逐位未变；best/latest的step/epoch与metrics一致。best SHA `1739175b789e22158b67d4c2f7b653f32f519dbba5e31f1bb4713027dac134b6`。
- 输出：[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/run_manifest.json)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/metrics.jsonl)、[outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/checkpoints/best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_coupled_geometric_batch8_20260912_225225/checkpoints/best.pt)。命令、日志、轻量view和验收manifest见唯一终态：[src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md)。

## 2026-09-12 — V1.1.14 采样配置更正与全量耦合标签验收

- modification_version: `V1.1.14`；category: diagnostic / experiment；approval: user-approved；[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)。
- 更正V1.1.13耦合训练4次短跑的解释：配置缺少`surface_sampling=v1_3_cache`，默认legacy点对应与target不同，故训练指标/权重为`INVALID_IMPLEMENTATION`。保留原始记录和运行目录，不resume，不把该实现错误归因于Cm。运行结束不等于数据/模型一致性闸门通过。
- 新配置直接选择正确采样器。run_id `coupled_fk_gate_20260912_224800`，run_status `COMPLETED`；285条train/val共72935帧完整10135点逐点比较，mean EPE `0.000016003mm`，max `0.000491738mm`；所有完整手指关节限位与独立mimic方程检查通过。此验收覆盖训练/验证参考，不包括原actual12维FK适配或MANO test。
- conclusion: `SUPPORTED`（耦合参考标签与配置预测FK一致）；`INCONCLUSIVE`（decoder学习效果、跨手泛化、物理成功）。[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/run_manifest.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/run_manifest.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/verification.json](../../research/dexplore_contract_audit/output/coupled_fk_gate_20260912_224800/verification.json)。
- 同时澄清V1.1.13拟合summary：18.9055mm指尖与1.6071mm表面是序列等权均值，p95 34.3876/2.7154mm是**序列均值的p95**，不是全帧误差p95。适配误差不能与学习误差直接相减。
- 20-step旧短跑包含全套validation且未超过性能预热20步，其总时间不支持50epoch需要1–2天的估算，也不支持batch4/8稳定吞吐比较；正式速度以新配置warmup后训练perf为准。
- 新batch8无扰动初步训练与后续实际状态适配分开，不声称已完成闭环训练。唯一运行状态：[src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md)。

## 2026-09-12 — V1.1.13 六维耦合几何重定向、缓存与三卡阶段一短程训练

- experiment_id: `coupled-geometric-stage1-v1.1.13`; modification_version: `V1.1.13`; category: data / experiment / operation。
- approval: user-approved；[执行计划](../plan/v1.1.md)。目标是保持6个独立手指控制量，按RL mimic规则展开到完整native姿态；实际观测FK使用完整18维native状态。
- 参考拟合 run_id: `coupled_geometric_v1_full_20260912_194544`，run_status `COMPLETED`，660条/166337帧，5021.06秒。
- 适配误差：原始12维几何到当前固定耦合基线指尖均值35.1269mm；优化后的耦合参考相对原始指尖均值18.9055mm、p95 34.3876mm，表面均值1.6071mm、p95 2.7154mm；fitted mimic最大残差`5.96e-8 rad`。这是参考适配误差，不能与模型学习误差做数值相减。
- [拟合run manifest](../../../../../data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/run_manifest.json)、[拟合summary](../../../../../data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/summary.json)、[独立验收](../../../../../data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/independent_validation.json)。
- 训练缓存 run_status `COMPLETED`：285条 train/val（255/30）、72935帧；与保留的63条MANO test合并为新 view。缓存 builder 的三 worker和finalize均通过 shape/finite/KNN/frame validation。[cache manifest](../../../../../data/processed_data/coupled_geometric_cache_v1_20260912/run_manifest.json)、[view manifest](../../../../../data/processed_data/cm_decoder_v2/coupled_geometric_v1_20260912/run_manifest.json)。
- 三卡阶段一 smoke 10 steps及100-step短程训练均完成，使用冻结 OICM v1.3 checkpoint SHA `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`、KNN32/2cm/10135点。100-step run val loss `0.0141249`，val point-flow EPE `36.391mm`，h1 `16.6984mm`；输出：[100-step run](../../../../../outputs/cmdecoderv2/cm_decoder_v2_20260912_212609/)。
- conclusion: `SUPPORTED`（6维耦合参考和完整native观测接口、目标侧几何缓存可以一致运行）；`INCONCLUSIVE`（Cm是否收敛、跨手迁移、二阶段MANO→actual微调和物理RL成功）。100-step不是正式长训，不据此报告研究效果。
- 限制：本轮尚未生成MANO参考→actual Inspire状态的第二阶段专用paired cache，也未接入IsaacGymEnvs rollout；当前只证明阶段一数据链路和三卡训练 smoke 可运行。

## 2026-09-12 — V1.1.13 DExplore几何轨迹与6维decoder状态不等价

- experiment_id: dexplore-contract-audit-v1.1.13；modification_version: V1.1.13；category: diagnostic / experiment。
- approval: user-approved；[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md)。
- run_id: contract_full_20260912_191500；run_status: COMPLETED；实际2026-09-12 19:05:19 +0800启动，134.11秒。
- base_commit: 5cf7eaaee2cff914d73ccc98aac390bd67cf8f3a；dirty=true；无checkpoint加载、训练或策略rollout。
- conclusion: SUPPORTED（现有6维mimic状态与原始几何/实际状态存在差异）；REFUTED（几何轨迹已满足现有decoder固定耦合的前提）；INCONCLUSIVE（两阶段训练收益、Cm能力与物理任务成功）。

**协议与来源**

检查660条同名geometric/actual，合计166337帧。完整float32张量有限、形状一致，除`198:205`和`373:391`外逐位不变；原canonical human/object文件全部存在且帧数匹配。
这只确认文件身份和时长，不代表MANO mesh的坐标/表面对应已验证；该步骤明确`not_checked`，未构建正式paired训练cache。
既有归属train509/val58/test63保持，只对其余30条记录`unassigned`；审计manifest为`diagnostic_only=true, training_eligible=false`。

外部`convert_grab.py`的`setup_retargeting`将18个joint作为优化变量并显式`ignore_mimic_joint=True`。
现有decoder仅保留`[6,8,10,12,14,15]`，从动关节按`[1.05,1.05,1.05,1.05,.6,.8]`重建。
DExplore同一关系用于PD target，但实际DOF位置不保证严格满足target耦合；**观测状态维度与可控动作维度不能混同**。
独立Isaac Gym资产加载确认18个DOF的名称顺序和现有native映射一致；只加载资产，physics_steps=0，不证明仿真控制稳定。

全帧统计从动关节相对耦合关系的绝对偏差。每条每源确定性等距抽8帧（每源5280帧），比较原始18-DOF FK与保留独立6维、限位后mimic展开的FK；保持wrist完全相同。
表面使用现有V1.3 seed2024、10135点对应，指尖使用5个URDF tip link等权。
另算有界线性mimic模型的关节空间最小二乘；其RMSE是这个关节空间问题的最优值，不能叫最优几何误差或物理可达性下界。

| 指标 | 几何重定向 | RL实际状态 |
|---|---:|---:|
| 从动关节耦合绝对偏差均值（全帧，deg） | 68.7413 | 7.9929 |
| 至少一个从动关节偏差>1deg的帧比例 | 100% | 76.8933% |
| 当前6维重建的手指关节RMSE均值（deg） | 59.0586 | 7.1078 |
| 关节空间最小二乘最优RMSE均值（deg） | 51.7703 | 5.1848 |
| 原状态到6维重建的全手表面EPE均值（mm，抽样） | 2.3702 | 0.2674 |
| 原状态到6维重建的指尖EPE均值（mm，抽样） | 35.1815 | 5.0975 |
| 原状态到6维重建的指尖EPE p95（mm，抽样帧） | 56.0135 | 19.4209 |

表面/指尖误差不是训练后的误差，也不是对所有6维预测优化后的下界。它们量化现有状态压缩实际丢失的形状；不能直接宣称几何预训练无任何价值或当前Cm因此失效。
每源每序列恰好8个FK样本，所以抽样micro mean与序列macro mean相同；关节全帧micro/macro分开保存在summary。

**判断与下一步审批边界**

当前几何数据不能不加说明地作为同一6维可执行状态的GT。直接长训会混合参考解码与输出表示不匹配的误差，先暂停正式cache/三卡训练。
建议先在本Task建立遵守现有控制耦合的几何重定向数据（重新优化，不直接将旧q裁剪后当GT）；实际状态观测保留12个手指关节，动作仍保持6个独立控制量与wrist控制。
这些是待用户确认的GT/状态/动作合同调整，本轮未实施，也未扩展decoder输出或改动外部生成器。下一步还需明确状态参考经PD执行与教师PD动作监督的区别，不将导出的实际q误称为原策略action。

**复核与证据**

- 10项Task定向测试通过。独立NumPy FK核验全部10560个抽样状态；surface EPE最大差1.43e-6mm，tip EPE最大差1.78e-13mm；16个torch表面复核样本最大逐点差0.000140mm。
- 从源张量重查所有保存q，使用`np.linalg.lstsq`独立核验闭式关节投影；检查的summary最大差3.59e-7。全部输入与受保护代码/资产stat和SHA不变。
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/summary.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/summary.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/paired_manifest.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/paired_manifest.json)。
- [src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/verification.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/verification.json)、[src/task/CmDecoderv2/research/dexplore_contract_audit/output/contract_full_20260912_191500/gym_asset_check.json](../../research/dexplore_contract_audit/output/contract_full_20260912_191500/gym_asset_check.json)。
- 唯一终态入口：[src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md)。

## 2026-09-12 — V1.1.12 轨迹质量标记：参考跟踪与实际跨手兼容不能合并成同一准入条件

- experiment_id: trajectory-quality-gate-v1.1.12；modification_version: V1.1.12；approval: user-approved（用户在质量准入建议后回复“继续”）；[最终计划](../plan/v1.1.md)。
- run_id: quality_val_20260912_155200；run_status: COMPLETED；实际2026-09-12 15:51:22–15:51:28 +0800，6.06秒。
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607；dirty=true；无模型加载、训练或仿真。
- conclusion: SUPPORTED（既定参考跟踪标记与既定跨手几何兼容性选出不同样本）；INCONCLUSIVE（这些阈值作为物理成功/正式训练准入标准的有效性）。

**协议**

固定4254个val窗口，其中target四步有效且运动的窗口3017。复用V1.1.10原始MANO任意偏移的全部严格候选边，基线53个运动接收窗口/10 parent/18个双流不重叠贪心配对。
质量标记只依赖参考与实际物体轨迹，不利用未来手q/wrist或decoder误差。effect标记要求K4每一步actual→几何参考的逐点flow EPE<=max(1mm,25%×较大RMS)。
pose20标记要求K+1五帧全部中心误差<=20mm且旋转<=15deg；固定pose40=40mm/30deg、pose80=80mm/60deg作敏感性检查；combined取effect与pose交集。
pose阈值在运行前写入计划，是工程诊断值，不是经任务成功率验证的标准，也没有根据结果择优放宽。

所有候选交集重新选择供体和重算不重叠贪心，未直接过滤旧贪心列表。清单明确diagnostic_only=true、split=val，不写正式训练index。

**结果**

| 标记条件 | 通过标记的运动窗口/3017 | 仍可跨手配对的接收窗口 | parent | 不重叠贪心配对 |
|---|---:|---:|---:|---:|
| 基线（不加跟踪标记） | 3017 | 53 | 10 | 18 |
| effect | 20 | 13 | 4 | 5 |
| pose20 | 47 | 0 | 0 | 0 |
| combined20 | 0 | 0 | 0 | 0 |
| pose40 | 216 | 12 | 4 | 4 |
| combined40 | 8 | 8 | 2 | 2 |
| pose80 | 1470 | 35 | 7 | 12 |
| combined80 | 16 | 12 | 4 | 4 |

原53个几何兼容运动窗口中，40个未达原参考effect跟踪门槛，53个均未达pose20；其中仍有13个同时满足effect跟踪与跨手兼容。
这里“兼容”只指前述逐步物体效果/接触区域条件，不等于物理成功或跨手唯一正确动作。

**路线调整**

V1.1.11指出实际执行没有忠实保持原参考，仍成立；但不能由此推导“跟好原参考”是Cm跨手正对的必要条件。原参考跟踪质量检查生成器，实际效果/接触兼容性检查两段动作能否成为候选正对，二者目标不同。
object-centric动作可以发生在不同world位姿；直接用绝对参考pose误差硬过滤，会额外要求任务没有定义的绝对轨迹接近，并排除原几何兼容候选。新工具因此只输出标记与检查队列，不自动应用于正式训练。
下一步应回到训练划分，按真实执行的效果与接触组织跨手正对，保留跟踪质量为独立标签/分层变量；评估Cm一致性训练时隔离validation，不能把本轮53个val候选送入训练。
本轮没有证明53个候选足以训练，也没有证明更宽pose阈值有效；不以扩大筛选规模代替数据或物理验证。

**实现与证据**

- 可复用入口 `research/trajectory_quality_gate/gates.py::quality_flags`，区分4步effect和5帧pose，拒绝非有限/负值/错误shape；失败位1/2/4分别是effect/position/rotation。
- 每窗口指标含独立cross_hand_compatible和质量标记；CSV检查队列按有效运动窗口中effect失败比例降序、失败数降序、id排序，只是检查顺序，不代表修复收益排序。首项包括s6/toothpaste_lift、s9/flashlight_on_1、s8/pyramidmedium_inspect_1。
- 候选清单含mode、best/贪心类别、原始MANO与目标Inspire的parent、cache/raw起点及原因位，共178行；不同条件/类别重复同一配对，不能当178个独立样本。
- 4项定向测试通过；独立复核全部4254个窗口标记和178条候选，效果归一化比值最大差4.78e-13，旧输入/代码/产物摘要与stat保持不变。
- [运行目录](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/)、[manifest](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/run_manifest.json)、[config](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/config.json)、[metadata](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/metadata.json)
- [逐窗口指标](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/window_metrics.jsonl)、[候选清单](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/candidate_manifest.jsonl)、[检查队列CSV](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/repair_queue.csv)
- [分层指标](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/metrics.jsonl)、[summary](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/summary.json)、[coverage图](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/quality_coverage.png)、[独立核验](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/independent_verification.json)、[run.log](../../research/trajectory_quality_gate/output/quality_val_20260912_155200/run.log)

## 2026-09-12 — V1.1.11 物体轨迹来源审计：主要效果差出现在RL实际执行段

- experiment_id: object-tracking-audit-v1.1.11；modification_version: V1.1.11；approval: user-approved（用户回复“继续”）；[最终计划](../plan/v1.1.md)。
- run_id: tracking_val_20260912_154240；run_status: COMPLETED；实际2026-09-12 15:42:29–15:42:53 +0800，约24秒。
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607；dirty=true。CPU只读诊断，无decoder、训练或仿真。
- conclusion: SUPPORTED（已有RL导出与几何参考存在显著物体跟踪差异）；REFUTED（本协议下前半选择的固定lag能改善后半效果误差）；INCONCLUSIVE（Cm整体跨手能力及物理任务成功）。

**来源链与坐标合同**

对前两轮相同30个val parent的7927帧核验：几何参考tensor与RL导出在物体pose `[198:205]`、native q `[373:391]` 以外逐元素完全一致；缓存actual pose可由RL tensor重放，raw frame步长4、frame_time=raw/120。
RL exporter逐次记录实际object root state与DOF状态，保留参考/contact字段。参考标签未被actual替换并不意味着真实接触已通过核验。

外部 `dexplore/data_processing/prepare_grab.py` 在生成object angles时显式使用原旋转的inverse；`convert_grab.py` 对骨盆、地面和object相对位移再次转换。
native几何参考的旋转矩阵转置后，与parent rotation的最大元素差为9.34e-7；这一关系可追溯，不能只因cache转换包含 `.T` 就判定新bug。
上游逆旋转对物理资产的合理性未在此轮仿真验证，不能把来源一致性升级成完整物理坐标正确性证明。

本实验显式区分两类量：几何参考→RL实际的中心位置/旋转角误差采用相同native XYZW约定；与前两轮比较的逐点object flow继续采用既有cache转置约定。
原parent到几何参考有world偏移且包含小幅时变分量，去每序列均值偏移后的残差均值4.6627 mm、中位数1.4757 mm。固定world平移不改变局部endpoint flow，不能直接把绝对world位置偏差当动作效果误差。

**定量结果**

全7927帧、30 parent的native参考→实际跟踪：

| 指标 | frame mean | frame median | frame P95 | parent macro mean |
|---|---:|---:|---:|---:|
| 物体中心位置误差 mm | 123.9094 | 41.4962 | 498.8197 | 133.9448 |
| 旋转误差 deg | 53.5425 | 33.0451 | 173.8581 | 49.2279 |

初始帧中心误差最大0.00192 mm、旋转误差最大2.96e-6 deg。导出从几何参考初始状态开始，之后出现跟踪偏离；不是所有序列一开始就有统一world错位。
8条parent的平均中心误差超过100 mm：s3/pyramidsmall_pass_1(103.5)、s4/duck_pass_1(622.5)、s6/hand_inspect_1(166.2)、s6/scissors_use_2(251.3)、s8/cubelarge_inspect_1(193.2)、s8/eyeglasses_pass_1(908.6)、s9/hammer_use_1(228.7)、s9/hammer_use_2(736.2)。这些是轨迹误差，不据此自动标注掉落或具体物理失效原因。

从target四步有效窗口覆盖的transition中去重，再取actual object RMS>=1 mm，得到2983个运动transition。与V1.1.10的3017个运动窗口分母不同。

| 既有cache合同下的逐步4096点效果差 | frame mean mm | median mm | parent macro mm |
|---|---:|---:|---:|
| parent → 几何参考 | 0.4801 | 0.0840 | 0.6233 |
| 几何参考 → RL实际 | 14.7682 | 9.1728 | 16.7992 |
| parent → RL实际 | 14.9160 | 9.1586 | 17.0175 |

三段EPE不是可加分解。可观察到parent与几何参考的运动效果接近，而几何参考与实际轨迹的差明显更大，支持优先检查/改善实际轨迹跟踪。
去除每序列actual-reference均值偏移后的平均欧氏位置残差131.1013 mm；均值偏移最小化的是平方误差，不保证降低平均欧氏距离，此数不能解释成“最优刚体校准反而恶化”。

**固定lag检查**

lag=-15..15，负值表示actual[t]对应reference[t+lag]即实际滞后；所有lag使用相同支持，排除边界样本变化。
每序列按时间前半选择lag，后半测试；1498个去重运动transition、30 parent，baseline EPE=14.9919 mm，选lag后15.8312 mm，配对增加0.8393 mm，parent聚类bootstrap2000次CI95=[0.1303,1.8224]；parent macro增加1.2997 mm。
30个parent中只有9个后半段得到改善；选择lag分散，10个为-1、6个为0，其余跨多个值并包括边界。
即使在全支持上取oracle最优lag，平均效果差也只由14.7682降至14.2641 mm，约3.4%。该oracle不可当独立验证结果。
结论仅针对固定lag与当前运动支持；不否定可变时间重参数化等其他方法，也不能定位控制器、接触、资产或奖励中的具体根因。

**对后续Cm工作的影响与限制**

当前直接以“同parent、同时间”的MANO与Inspire作等价正对，会混入真实物体效果差。优先建立带物体跟踪质量准入的跨手演示数据，再比较Cm一致性训练；不能用transported oracle隐式注入目标运动来掩盖这一缺口。
本轮保留了输入、历史结果和坐标合同，没有修改cache、剔除正式样本、重训模型或改外部dexplore代码。
frame_time只核验了当前缓存与原始采样一致；历史导出没有逐帧模拟器时间戳，此轮未独立锁定当时controlFrequencyInv配置，故30Hz按既有数据合同解释。

**验证与入口**

- 4项定向测试通过；独立四元数测地距离检查7927帧，最大差2.96e-6 deg；1080项独立刚体pointflow检查最大差4.55e-13 mm；lag选择与heldout均值复现。
- 所有新tensor、外部源码与旧输入/产物SHA256及stat保持不变；输出约2.78 MiB。smoke 2序列1096帧，6.21秒，只证明工程实现。
- [运行目录](../../research/object_tracking_audit/output/tracking_val_20260912_154240/)、[manifest](../../research/object_tracking_audit/output/tracking_val_20260912_154240/run_manifest.json)、[config](../../research/object_tracking_audit/output/tracking_val_20260912_154240/config.json)、[metadata与外部源码SHA](../../research/object_tracking_audit/output/tracking_val_20260912_154240/metadata.json)
- [metrics.jsonl](../../research/object_tracking_audit/output/tracking_val_20260912_154240/metrics.jsonl)、[summary](../../research/object_tracking_audit/output/tracking_val_20260912_154240/summary.json)、[tracking.png](../../research/object_tracking_audit/output/tracking_val_20260912_154240/tracking.png)
- [独立核验](../../research/object_tracking_audit/output/tracking_val_20260912_154240/independent_verification.json)、[run.log](../../research/object_tracking_audit/output/tracking_val_20260912_154240/run.log)、[活动终态](activity_log.md)

## 2026-09-12 — V1.1.10 跨时间配对覆盖：偏移有帮助，但不足以补齐运动配对

- experiment_id: cross-hand-pair-coverage-v1.1.10；modification_version: V1.1.10；operation_category: diagnostic / experiment / operation。
- approval: user-approved（用户在优先补齐同效果/接触跨手配对的建议后回复“继续”）；[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) final。
- run_id: pair_coverage_val_20260912_153150；run_status: COMPLETED；实际2026-09-12 15:30:52–15:31:06 +0800，13.88秒。
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607；dirty=true；没有加载 decoder、更新权重或运行训练。
- conclusion: **REFUTED（当前候选库中，仅允许时间偏移即可达到100严格运动窗口/10 parent的充足性命题）**；总体跨手迁移仍 INCONCLUSIVE。自动运行摘要保留后者，不把覆盖诊断当作迁移效果实验。

**协议与保护边界**

候选固定为 V1.1.9 的4254个 val active-only窗口；接收手四步有效3874，其中平均 actual object RMS>=1 mm的运动窗口3017。
同一 parent 内穷举原始 MANO 的四步有效源起点，原始 object trajectory、K4/30 Hz、空间坐标和所有效果/接触门槛保持不变。
不使用 transported，不放宽相对/绝对效果差，不做时间缩放或跨 parent 搜索，不按 latent、decoder误差或目标未来q/wrist选样本。
逐点平均欧氏效果差是最终判据；均值运动向量差仅作 Jensen 必要条件预筛，随后对全部候选执行精确4096点计算。

选择满足严格条件且平均归一化效果差最小的源；平局依次选较小时间差、较早源起点。
供体可复用，因此分别统计接收覆盖、唯一选中源、最大一对一窗口匹配容量，以及两流各自K+1帧都不重叠的确定性贪心配对数。
后者是保守下界，不是最优解，也不意味着这些片段是独立物理试验。所有结果只属于该val候选库，不回流训练。

**严格运动覆盖**

| 源时间范围（30 Hz） | 接收窗口 | parent | 唯一选中源 | 一对一窗口容量 | 双流不重叠贪心配对 |
|---|---:|---:|---:|---:|---:|
| 同步 | 16 | 4 | 16 | 16 | 6 |
| ±5帧 | 40 | 9 | 31 | 35 | 12 |
| ±15帧 | 48 | 10 | 38 | 43 | 14 |
| ±30帧 | 48 | 10 | 38 | 43 | 14 |
| 同parent任意偏移 | 53 | 10 | 43 | 48 | 18 |
| 只允许绝对偏移>=5帧 | 16 | 5 | 15 | 16 | 10 |
| 只允许绝对偏移>=20帧 | 6 | 1 | 6 | 6 | 5 |

“偏移>=5帧”限制单个源/目标片段的相对时间；最后一列限制不同配对在每条流内的时间区间复用，两者含义不同。
±15帧已覆盖48/53，扩大到±30帧没有新增运动接收窗口；远时匹配很少。任意偏移的严格运动覆盖为53/3017=1.76%。
即使只检查效果、暂不检查接触，也只有76个运动接收窗口能匹配，仍低于100；这把当前库的主要缺口指向动作效果对应，而不仅是接触门槛。

不加运动条件时，任意偏移产生20096条严格边、覆盖651个接收窗口/16 parent，但只有90条边属于运动接收窗口。
其中18299条（91.1%）来自s1/torussmall_lift，其严格运动接收数为0。不能把两万条边当作两万个动作正对：静止片段和供体复用会严重放大数据量。

53个运动接收窗口来自10 parent、8个物体，动作标签主要为pass（7 parent），其余为lift、see、on各1。
逐parent覆盖/贪心数为：waterbottle_lift(s1)7/6、binoculars_pass(s10)7/2、stamp_pass(s10)3/1、
pyramidlarge_pass(s2)1/1、alarmclock_see(s3)2/1、stapler_pass(s3)8/1、apple_pass(s7)14/3、
pyramidlarge_pass(s7)2/1、waterbottle_pass(s7)4/1、flashlight_on(s9)5/1。

**对 Cm 路线的影响**

允许小幅时间偏移确实提高配对可用性，支持保留异步匹配机制；但当前库不足以支持预设规模的严格运动迁移诊断，更不宜直接把大量静止匹配送入跨手一致性训练。
下一步优先定位/补齐目标手实际物体运动与源参考的对应：按共同物体动作目标构造跨手数据，控制轨迹跟踪误差，再检查接触与可达性。
若只扩展现有库，还需另行核验未进入当前active-only bank的窗口、跨parent同物体匹配或时间尺度；本实验没有否定这些可能性，也没有证明Cm结构本身无效。
当前53个优选运动配对和18个不重叠片段可作后续诊断锚点；不能作为充分的训练集或最终独立测试集。

**工程验证与产物**

- 5项定向测试通过：Jensen预筛对完整穷举、无效源/接收排除、四步接触、供体复用和端点重叠。
- V1.1.9同步effect/strict对角线逐窗口完全复现，85严格窗口/16严格运动窗口不变；对角线描述最大数值差7.44e-6。
- 独立使用归档pose与canonical点构造刚体位移，对1749个局部网格候选穷举，边存在性完全一致；保存描述最大差1.33e-5。
- 所有保存边门槛、选中供体最优排序、parent/有效性、双流不重叠区间和summary复核通过；旧输入/代码/产物摘要及stat保持不变。
- GPU3 peak allocated199.9 MiB，输出约5.20 MiB；两parent smoke仅工程证据（15:29:49–15:29:54 COMPLETED）。正式运行前仅补充独立复核脚本，并把manifest last_step改为实际active-only评估数。
- [运行目录](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/)、[manifest](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/run_manifest.json)、[config](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/config.json)、[metadata](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/metadata.json)
- [全部严格候选边](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/candidate_pairs.npz)、[选择索引](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/selected_pairs.jsonl)。target/source为V1.1.9 bank全局行号；每parent [descriptors](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/descriptors/) 的global_rows/starts映射到30 Hz起点。
- [逐parent metrics](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/metrics.jsonl)、[summary](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/summary.json)、[coverage.png](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/coverage.png)
- [独立核验](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/independent_verification.json)、[run.log](../../research/cross_hand_pair_coverage/output/pair_coverage_val_20260912_153150/run.log)、[活动终态](activity_log.md)

## 2026-09-12 — V1.1.9 同步 MANO→Inspire 冻结 Cm 交换

- experiment_id: cross-hand-cm-swap-v1.1.9；modification_version: V1.1.9。
- approval: user-approved（用户连续回复“继续”）；[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) 已 final。
- run_id: cross_hand_val_20260912_151850；run_status: COMPLETED；2026-09-12 15:18:08–15:21:28 +0800。
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607；worktree_dirty=true。
- 冻结 checkpoint 与 V1.1.8 完全相同：decoder SHA256 `e60f0e954c062d15e7c2fa217e1bebcb0a4b0be8795d1b5729771ee1f8f5fef7`，OICM SHA256 `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`；无训练或 checkpoint 选择。
- conclusion: **INCONCLUSIVE（总体跨手迁移）**；观察到同步 MANO 相对错时条件的局部兼容性信号，严格运动覆盖不足。

**假设与协议**

H：当源/目标的物体效果和接触区域相近时，原始 MANO 完整 Cm 可以在固定 Inspire 当前状态下支持参考动作解码。
30 条 Inspire val、4254 active-only 窗口，K4、stride1、30 Hz、右手、object_pose_t、KNN32/2 cm，源 MANO2048、目标 Inspire10135 点。
使用 V1.3 原表面 sampler/seed2024，从原 parent mesh 与 raw timestamp 重建 MANO；原始 parent object pose 保持不变。
每个 transition 的手点、法向与两端 flow 都先转到其起点物体坐标，再编码；完整替换 tokens/anchor position/normal。
目标当前 q/wrist/link features 不变，未来目标手点/q/wrist 只用于误差计算。

`mano_transported` 将原 MANO 的物体相对几何重放到目标实际 object pose，显式注入目标物体运动，属于 oracle。
正式 producer 的 `dex_data[:,198:205]` 是实际物体 pose；旧正式 MANO cache 也含这类搬运。因此不能把 transported 的结果当作独立迁移证据。
两源所有对齐帧的 canonical object4096 逐点一致性最大误差 2.46e-7 m，没有拟合额外刚体变换。

两源四步均有效后，每步 object endpoint flow EPE <= max(1 mm,0.25×两源较大 RMS)；进一步要求四步接触物体点 IoU>=0.25、centroid 距离<=20 mm；运动子集平均 actual RMS>=1 mm。
物体未来轨迹只作离线效果筛选及 oracle 对照，不输入原始 MANO 条件。门槛没有按结果调整。
错时源在同 parent 内抽取，起点间隔>=20 帧、四步有效、seed42。所有差异在共同 recipient 上配对；bootstrap2000 次，以 parent 聚类。

**覆盖与结果**

| 子集 | 窗口 | parent | correct EPE mm | identity | mano_sync | mano_shift | mano_transported |
|---|---:|---:|---:|---:|---:|---:|---:|
| 两源四步有效 | 3665 | 30 | 6.6180 | 13.2766 | 15.5397 | 23.2145* | 10.6531 |
| 再匹配物体效果 | 176 | 17 | 5.1410 | 8.1242 | 8.4433 | 17.2948 | 8.2750 |
| 再匹配接触区域 | 85 | 9 | 4.7513 | 9.9045 | 5.4492 | 17.7010 | 5.0848 |
| 严格运动子集 | 16 | 4 | 8.7499 | 46.7284 | 10.8513 | 54.0171 | 9.0612 |

数值为 frame micro；*错时有效窗口为3650，其同步 MANO 配对基线为15.5694 mm，不能直接用3665行平均做差。各条件 parent macro、q MAE、wrist translation/rotation、输出点变化和区间完整保存在 summary。

- 有效共同3650窗口，同步优于错时7.6451 mm，parent cluster CI95=[4.6160,11.6464]；这支持源时间信息影响目标输出，但未控制两源物体效果相等。
- 有效3665窗口，同步对 identity 的改善为 -2.2631 mm，CI95=[-6.1117,1.2326]；不能声称直接无筛选替换已带来总体改善。
- 严格85窗口，同步相对 correct 增加0.6980 mm，CI95=[0.1939,2.2326]；相对 correct 预测点变化3.7469 mm；对 identity 改善4.4553 mm，但 CI95=[-2.4486,29.4832] 跨零。
- 严格运动16窗口，同步对 identity 改善35.8771 mm，对错时改善43.1658 mm；相对 correct 增加2.1014 mm、预测点变化6.4274 mm。**仅4个聚类，不能以其区间作总体显著性论证。**
- 16窗口分别来自 s1/waterbottle_lift(1)、s3/stapler_pass_1(8)、s7/apple_pass_1(6)、s7/waterbottle_pass_1(1)；14/16集中在两条序列，且窗口重叠。
- 两源有效窗口的平均每步效果差中位数8.0708 mm，P90=26.7838 mm。同步时间不是同一动作效果；effect 门槛只保留176/3665=4.80%，严格运动只保留16/3665=0.44%。

**结论与下一步依据**

预设总体结论要求>=100严格运动窗口且>=10 parent，本次16/4未达标，故总体结论 INCONCLUSIVE。
局部匹配样本有可复用信号，不能把未匹配样本的退化直接解释成 Cm 只编码手型，也不能用 oracle 搬运改善证明跨手成功。
本轮是单步 reference compatibility；目标参考并非跨手唯一正确动作，没有执行接触力、碰撞、物体任务成功率或物理闭环评估；val 曾参与 checkpoint 选择，也不是独立 test。

下一步优先补足“同物体效果 + 相近接触”的真实跨手运动配对：先在现有序列只读检查允许时间偏移时的配对覆盖和 parent 分布，保持效果/接触门槛，避免先扩大模型或直接施加同 timestamp 的 latent 对齐。
未来训练用的正对必须由 train 构造并单独冻结验证划分；本轮 val 诊断配对不能被直接回流为训练样本。充足配对后再比较 effect-conditioned Cm 一致性目标与现有手流重建目标。

**工程验证与证据**

- 5 项定向测试通过；正式 Inspire source_window 回放差0；MANO缓存表面回放差0；correct完整 Cm 最大回放差4.14e-6，q/wrist回放差1.19e-7。
- 全条件×30 parent 共150次独立 NumPy FK 检查，手 EPE/输出变化最大差1.68e-5 mm；归档 donor、分层覆盖、paired summary 可重现；参数/buffer、输入/code摘要及文件stat前后不变。
- 完整运行200.53秒，GPU3 peak allocated5459.5 MiB，输出约75.9 MiB；两序列 smoke 仅证明工程实现可用。
- [src/task/CmDecoderv2/research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/)
- [run_manifest.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/run_manifest.json)、[config.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/config.json)、[metadata.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/metadata.json)
- [metrics.jsonl](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/metrics.jsonl)、[summary.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/summary.json)、[comparison.png](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/comparison.png)
- [independent_verification.json](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/independent_verification.json)、[run.log](../../research/cross_hand_cm_swap/output/cross_hand_val_20260912_151850/run.log)、[活动终态](activity_log.md)

## 2026-09-12 — V1.1.8 冻结 Cm 条件依赖与 16 步递归诊断

- experiment_id: cm-condition-dependence-v1.1.8
- modification_version: V1.1.8；operation_category: diagnostic / experiment / operation
- approval: user-approved（用户回复“按照你的想法继续”）；计划：[src/task/CmDecoderv2/docs/plan/v1.1.md](../plan/v1.1.md) 。
- run_id: cm_dependence_val_20260912_145016；run_status: COMPLETED；终态见 [src/task/CmDecoderv2/docs/logs/activity_log.md](activity_log.md) 。
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607；worktree_dirty=true，代码 SHA256 由 metadata 锁定。
- decoder: 固定 full10135 best，epoch=5、step=7240、原 best val/loss=0.003993955866854461；OICM V1.3 冻结。

**假设与协议**

H1：在同一个目标 Inspire 当前状态下，正确的完整 Cm 比另一真实动作的 Cm 更能恢复参考运动。
H2：这种作用是否可以维持到 16 步手状态递归？本轮没有改变手来源，不能回答跨手等价或迁移。

保持现有 K=4、30 Hz、stride=1、object_pose_t、右手、KNN32/2 cm/unique-hand、10135 点 FK 和无扰动
val loader。完整 30 条 Inspire val 共 4254 active-only 窗口，其中 h1 有效 4061、四帧均有效 3874。
原始几何/输入 hand flow 经冻结 OICM 得到完整 Cm bank，后续只将 tokens/anchors 与当前手状态/link
queries 交给 core。未来 target 手点/q/wrist 仅用于误差，没有输入 decoder，也没有使用 GT 选择 donor。

这里 source hand-flow 按参考解码任务的定义合法包含未来源动作；同源实验的源参考和目标动作来自同一
Inspire 序列。“GT 不输入”指没有额外传入未来 target q/wrist/手点的捷径，不是声称不利用未来参考运动。
因此这是 reference-conditioned reconstruction/realization 诊断，不是不看未来的在线动作预测。

所有 donor 都在接收方同一序列内，起点相差至少 20 帧，完整 K=4 Cm 全有效；一次交换整个 window，
保留 tokens/anchor/时间对应。随机交换覆盖 3848 帧/30 序列；严格匹配覆盖 1302 帧/21 序列，约占
h1 有效样本的 32.06%。严格匹配按计划固定当前 wrist/q/active fraction 与 hand-flow RMS 条件，并要求
输入动作均值向量有变化，不根据预测误差调阈值。它仍是有限条件匹配，不等于所有物理状态完全一致。

**单步结果：每行只比较该对照自己的共同接收样本**

| 对照 | 样本/序列 | 正确 Cm EPE / mm | 对照 EPE / mm | 对照增加 / mm | 增加的 95% sequence CI |
| --- | ---: | ---: | ---: | ---: | --- |
| 保持当前手 identity | 4061/30 | 7.0968 | 13.6469 | 6.5501 | [3.7909, 9.8326] |
| 完整 Cm 置零（OOD） | 4061/30 | 7.0968 | 21.2582 | 14.1613 | [11.7423, 16.4101] |
| 真实窗口随机交换 | 3848/30 | 6.7863 | 23.2002 | 16.4139 | [12.3724, 21.5001] |
| 严格匹配后交换 | 1302/21 | 5.3292 | 24.3919 | 19.0628 | [12.3278, 30.5853] |

CI 使用 2000 次 paired sequence-cluster bootstrap，donor 和 recipient 位于同一 cluster。
sequence-macro 的 random/matched 交换惩罚分别为 18.9452 / 25.5138 mm，方向一致；两类交换覆盖率与
接收样本不同，不能用 24.39 与 23.20 比较两种对照的难度，更不能将 matched 较大误差解释为因果增强。

交换相对原预测的全表面点变化均值为 21.8483 / 23.1559 mm；严格匹配 donor 的平均 wrist 位移差
13.21 mm、旋转差 9.39°、q RMS 差 0.0668 rad、active fraction 差 0.0221，输入平均 flow 向量差
24.45 mm。因匹配允许一定状态差异，惩罚不能全归因于某一个 disentangled motion 维度。

在 identity EPE >=2 mm 的子集（3221 帧/30 序列）中，正确/identity EPE 为 7.6762/16.8500 mm，
相对改善 9.1738 mm；random/matched 交换惩罚分别为 18.6342/20.1699 mm，CI 仍为正。
该子集按已声明阈值定义，identity 本身包含固定 mimic/FK 表示约束，不等同于纯 native cache 点位移阈值。

**短时递归**

每个条件从相同 GT 当前状态 handoff，后续仅反馈自己的预测状态，clamp finger q、右乘 wrist delta。
每序列最多 3 个确定性分散起点，要求后续 16 个完整 source window 全有效；共 83 个起点/30 序列。
随机 donor 具有连续 16 步，覆盖 78 起点/27 序列；严格匹配仅 17 起点/10 序列，不外推到所有状态。
物体 pose 每步仍来自当前参考帧，不能把此结果称为物理闭环或接触成功率。

| 执行步数 | 正确 Cm EPE / mm（83 起点） | identity / mm（相同起点） |
| --- | ---: | ---: |
| 1 | 8.5667 | 15.5855 |
| 4 | 20.1408 | 42.0166 |
| 8 | 31.6613 | 62.1715 |
| 16 | 53.1931 | 118.9288 |

第 16 步 random swap：184.1595 mm，同接收方 correct=50.6555 mm，惩罚 133.5040 mm，
95% CI [90.3943,182.4695]。matched swap：125.2776 mm，同接收方 correct=36.4542 mm，
惩罚 88.8234 mm，CI [51.1880,127.5756]。正确 Cm 的有用性延续到短递归，但仍有较大状态误差。

**新发现：native 从动关节与 decoder 固定 mimic 的不一致**

首次 smoke 将“6 维 GT q + 固定 mimic FK 必须等于 native cache”作为断言，因 0.577569 mm 最大坐标差
退出；这项诊断断言的假设不成立。cache producer 使用原始 18 维 simulator q，含实际从动关节状态；
decoder 将 6 个独立 finger q 按 1.05、0.6、0.8 比例展开并夹紧独立关节。真实从动关节可能偏离比例。
现有 pointflow 测试主要覆盖初始化帧，不能保证动作中两者相等。

没有改模型、GT 或误差定义；改用 full-native q 重放 cache 检查采样顺序与坐标。在每 batch 一行的
抽查中，native replay 最大坐标差为 0；现有 core 重放差也为 0。所有窗口的 6 维 GT 重建 EPE
均值/中位数/p95/max 为 0.6074/0.4907/1.4958/2.5348 mm；h1 有效集均值 0.6305 mm。
38.74% 的窗口下一帧至少一个独立 q 超限 >1e-5 rad；从动关节相对固定比例的逐样本 RMS 均值 0.3308 rad。
局部点最大坐标差可达 75.75 mm，但全表面均值较小，不应把二者混为同一个误差指标。

这些数值是代入 GT 六维状态后的几何重建差异，不是优化所有可用自由度得到的不可约下界。
按三角不等式，在同一 h1 有效集上仅把 native GT 替换成该 reduced GT，平均 EPE 的变化至多约
0.6305 mm，不能由此解释目前全部约 7.10 mm 误差；本轮并未替换 GT。

**结论与下一项实验**

- SUPPORTED：当前冻结 decoder 在已评估 Inspire val 条件下利用完整 Cm，正确 Cm 对单步和短递归
  均有任务相关收益。不能再把“完全忽略 Cm、只靠当前状态”作为这些样本的默认解释。
- SUPPORTED：存在 native simulator 与固定 mimic/限位 FK 的表示差异，以及短递归累计状态误差。
- INCONCLUSIVE：哪部分 token/anchor 分别负责动作，跨手可互换性、未见手型泛化、长期稳定与物理闭环成功。
- 下一项优先推进受控 MANO→Inspire Cm 交换：固定目标手起点/短 horizon，审查源参考与目标状态的
  物体 frame、实际 effect、接触可达性。保留本轮同源上界和错误 Cm 对照，优先测交换增量；
  暂不因 source classifier 可分性就加入对抗对齐，也不立即重训 state-only baseline。
- 生成 summary/manifest 保持保守的自动 `INCONCLUSIVE` 标签；上述分命题人工解释以本实验记录为准。
  工程 smoke 与正式科研结果分开，未修改前序实验结论。

**验证与产物**

- 13 个定向/相关模型与运动学测试通过；正式运行约 104.9 秒、peak allocated 2745.3 MiB、输出约 49.1 MiB。
- 17912 teacher 指标行、5504 rollout 指标行；donor 与配对 summary 独立重算一致。
- 独立 NumPy FK 对 15 个分条件样本复算，EPE 与归档最大差 0.0000078251 mm；
  20 个输入/代码 SHA256、450 个 geometry/q/wrist 文件 stat 和模型参数/缓冲区摘要保持不变。
- 输出目录：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016)
- 配置：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/config.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/config.json)
- Manifest：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run_manifest.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run_manifest.json)
- 单步指标：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/metrics.jsonl](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/metrics.jsonl)
- 递归指标：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/rollout_metrics.jsonl](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/rollout_metrics.jsonl)
- 统计：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence_summary.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence_summary.json)
- 图：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence.png](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/dependence.png)
- 独立复核：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/verification.json](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/verification.json)
- 运行日志：[src/task/CmDecoderv2/research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run.log](../../research/cm_condition_dependence/output/cm_dependence_val_20260912_145016/run.log)

## 2026-09-11 — V1.3 full10135 从 GT 接触帧开始的纯 Inspire recursive rollout/effect 诊断

- experiment_id: `cmdecoderv2-inspire-rollout-effect-v13-contact-start-v1.1.7`
- activity_id: [`cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021`
- run_status: `COMPLETED`
- modification_version: `V1.1.7`
- operation_category: `diagnostic / experiment / operation`
- importance: `diagnostic`
- pinned: `false`

**假设与固定合同**

针对此前从 frame 0 启动导致的分布外问题，改用 GT Inspire `10135` 点到完整 object pool 的首个
`<=20 mm` 合法帧作为 recursive handoff。只在起点输入一次 GT Inspire state，之后每步反馈 decoder
预测的 Inspire state；真实 object flow 仅作当前 `t -> t+1` 的 display-only 对照，不输入 decoder/OICM，
也不反馈 rollout。V1.3 保持 `unique_knn_edges`、`K=32`、2 cm validity、临时 source KNN 和冻结 OICM。

**运行**

- command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021 --run-id cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021 --knn-batch-size 4`
- base_commit: `aba8a3650714b62823f996d1f1fde3d383b82fbf`；run 启动时 `worktree_dirty=false`。
- decoder checkpoint: [best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)，SHA256 `e60f0e954c062d15e7c2fa217e1bebcb0a4b0be8795d1b5729771ee1f8f5fef7`。
- frozen OICM checkpoint: [best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)，SHA256 `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。

**结果**

- 自动起点解析为 0-based sequence frame `45`（source frame `180`），GT hand-object 最近距离
  `0.564 mm`；完整 run 生成 `323` 个 transition，实际 sequence frame 为 `45..367`。
- GT 接触段为 sequence frame `45..317` 的 `273` 帧；在该段内 `sample_valid=true` 为 `273/273`。
  GT 接触段的递归 hand position EPE mean=`87.909 mm`，hand-flow EPE mean=`2.363 mm`。
- GT 接触段内，Cm/OICM 对实际 object flow 的 display-only `pred-GT effect EPE`
  mean/median/p90/max=`3.545/3.053/4.437/47.174 mm`；effective effect RMS mean=`5.566 mm`，
  GT effect RMS mean=`3.611 mm`。
- 按预测 rollout hand 与 object 的距离，预测 hand `<=20 mm` 的帧为 `285/323`，OICM
  `sample_valid` 同为 `285/323=88.24%`；sequence frame `330..367` 的 `38` 帧无效并归零
  effective effect。全程 hand position EPE mean/max=`246.452/1517.173 mm`。
- 后段 GT object flow 接近零，因此 invalid 帧上的零 effective effect 与 GT 的低 EPE 不计作
  Cm/OICM 预测能力；详细 GT/预测分段和里程碑见 `diagnostic_tables.json`。

**结论边界**

- `SUPPORTED`（工程协议）: 接触起点自动选择、一次 GT handoff、后续纯预测递归、V1.3 KNN/2 cm
  validity 和 display-only 对照均按合同执行。
- `INCONCLUSIVE`（科研效果）: 接触段可以得到有效 object-effect 预测，EPE 均值 `3.545 mm`；
  但递归 hand position 已出现显著累计漂移，并在后段离开 2 cm 有效区。该单序列结果不足以
  证明长期 rollout 正确或跨 embodiment 泛化成立。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/effect_summary.json)
- [diagnostic_tables.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/diagnostic_tables.json)
- [source_knn_indices.npy](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-contact-full-20260911-001021/source_knn_indices.npy)
- [最终计划](../plan/v1.1.md)

## 2026-09-10 — V1.3 full10135 纯 Inspire recursive rollout/effect 诊断

- experiment_id: `cmdecoderv2-inspire-rollout-effect-v13-full10135-v1.1.7`
- activity_id: [`cmdecoderv2-inspire-effect-v13-full-20260910-175216`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-v13-full-20260910-175216`
- run_status: `COMPLETED`
- modification_version: `V1.1.7`
- operation_category: `diagnostic / experiment / operation`
- importance: `diagnostic`
- pinned: `false`

**假设与固定合同**

复现纯 Inspire `s1/mouse_lift` 递归 rollout self-effect 诊断，但换用 V1.3 full10135 decoder `best.pt` 和冻结 OICM V1.3 `best.pt`。初始 frame 0 用 GT Inspire state handoff，之后递归反馈 decoder 预测 state；真实 object flow 仅作为逐步 `t -> t+1` display-only 对照，不输入 decoder/OICM，也不反馈 rollout。V1.3 `unique_knn_edges` 使用 run 目录内临时 source KNN32；effect path 每步对预测 hand `10135` 点重算 selected object KNN32，并按 2 cm validity 交给 OICM。

**运行**

- command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-full-20260910-175216 --run-id cmdecoderv2-inspire-effect-v13-full-20260910-175216 --knn-batch-size 4`
- base_commit: `6aa2c13b53b4748b9d8205edc459e61463ddeeb1`；run 启动时 `worktree_dirty=false`。
- decoder checkpoint: [best.pt](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)，SHA256 `e60f0e954c062d15e7c2fa217e1bebcb0a4b0be8795d1b5729771ee1f8f5fef7`。
- frozen OICM checkpoint: [best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)，SHA256 `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。

**结果**

- 368 个 transition 全部完成；2 cm 阈值下 rollout predicted hand 到 object 最近距离 min/median/max=`298.091/480.982/1245.904 mm`，368/368 均远于 2 cm。
- OICM `sample_valid` 全为 false，`oicm_valid_ratio=0.0`，有效 object-flow RMS mean/max=`0/0 mm`；raw dummy path effect RMS mean=`27.709 mm`，但无物理语义。
- 递归 self-rollout hand EPE mean/median/max=`640.851/580.975/1192.516 mm`；future hand EPE mean=`643.875 mm`；hand-flow EPE mean/median/max=`14.610/4.927/89.366 mm`。
- display-only GT object effect RMS mean/max=`2.693/55.968 mm`；零有效 prediction 对 display-only GT 的 EPE mean=`2.672 mm`。
- 同一 V1.3 `10135` surface sampling 的 GT Inspire hand-object 距离审计：`273/371` 帧 `<=20mm`，首个 `<=20mm` step 为 `46`，GT 距离 min/median/max=`0.040/0.341/1245.904 mm`。

**结论边界**

- `REFUTED`：该 V1.3 full10135 decoder `best.pt` 在 `s1/mouse_lift` 的递归 Inspire rollout 中，没有复现 GT 轨迹的接触段；hand state 漂移主导结果。
- `SUPPORTED`（工程协议）：V1.3 临时 source KNN、per-step predicted-hand KNN、2 cm OICM validity gate 和 display-only object flow 对照均按计划完成。
- `INCONCLUSIVE`（object-effect 预测质量）：因为 predicted hand 全程不进入 2 cm 有效区，本 run 无法评价接触时 Cm/OICM 对实际 object flow 的预测质量；`2.672 mm` EPE 主要是零有效输出与 display-only GT flow 的差异。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/effect_summary.json)
- [source_knn_indices.npy](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/source_knn_indices.npy)
- [diagnostic_tables.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-full-20260910-175216/diagnostic_tables.json)
- [最终计划](../plan/v1.1.md)

## 2026-09-10 — V1.3 OICM + Inspire 全点监督 decoder 正式训练

- experiment_id: `cmdecoderv2-temporal-d2-oicm-v1.3-full10135-v1.1.7`
- activity_id: [`cmdecoderv2-v1.1.7-formal-20260910-163524`](activity_log.md)
- run_id: `cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812`
- run_status: `COMPLETED`
- modification_version: `V1.1.7`
- operation_category: `architecture / code / data / experiment / operation`
- importance: `primary`
- pinned: `true`

**假设与固定合同**

验证最终 OICM V1.3 的 `K=32`、2 cm unique-KNN edge stream 能否在不回退到 1538 点 hand stream 的情况下，接入 Temporal-D2 decoder，并使用完整 Inspire-F1 surface `10135` 点进行 point-flow supervision。2 cm 只用于 KNN edge validity 和 `active_only` frame gating；point-flow loss 对全部 `10135` 点计算，不使用旧的 point-level 2 cm mask。

- frozen OICM: [best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)，SHA256 `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。
- data view: [dexplore_rl_v1_3_full10135](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/)，train/val/test=`255/30/63` sequences，view train/val windows=`63988/7807`；启用 `active_only` 后实际 train/val windows=`34746/4254`。
- supervision: target `knn_hand_points_world.npy`，`10135` points，all points=`true`，point mask=`none`；hand stream=`unique_knn_edges`，`K=32`，radius=`0.02 m`，distance runtime recompute，batch-max dynamic padding。
- split: `inspire_rl` train/val only；MANO test 保持 qualitative-only，不参与训练或正式定量 test。
- decoder: Temporal-D2，`K=4`，30 Hz，active-only，state perturbation，50 epochs，从头训练；frozen OICM 保持 eval/no-grad。

**运行**

- command: `CUDA_VISIBLE_DEVICES=0,2,3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --distributed`
- device: physical GPU `0,2,3`；world size `3`；per-device batch `8`；global batch `24`；seed `42`。
- total: `72400` steps / `50` epochs / approximately `03:52:52`。

**结果**

- 首个 validation：`val/loss=0.0093770677`，point-flow EPE=`26.4114 mm`。
- checkpoint selection best：epoch `5` / step `7240`，`val/loss=0.0039939559`，point-flow EPE=`12.8782 mm`，h1=`7.1633 mm`。
- total point-flow EPE 最好：epoch `27` / step `39096`，`12.6071 mm`；这不是当前 best checkpoint 的选择指标。
- final epoch `50` / step `72400`：`val/loss=0.0041478906`，point-flow EPE=`12.7156 mm`，h1=`9.4626 mm`，wrist translation=`11.8189 mm`，wrist rotation=`6.1306 deg`。
- train final epoch：loss=`0.0019386694`，point-flow EPE=`7.6044 mm`。

**结论边界**

- `SUPPORTED`（工程协议）：V1.3 frozen OICM、unique KNN edge input、dynamic padding、完整 `10135` 点 point-flow target、frozen OICM no-grad 和 50-epoch train/val 均已正常完成；`metrics.jsonl` 共记录 50 个 train epoch 和 50 个 val epoch，全部数值 finite。
- `INCONCLUSIVE`（科研效果）：validation 在 epoch 5 取得 checkpoint selection 的最低 loss，后续在约 `0.00415` 附近平台化；该 run 只包含 Inspire RL train/val，没有 MANO→Inspire 定量 test，因此不能仅凭这次训练宣称跨 embodiment decoder 效果成立。

**证据**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/)
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/run_manifest.json)
- [config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/config.json)
- [metadata.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metadata.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/latest.pt)
- [最终计划](../plan/v1.1.md)
## 2026-09-08 — teacher-forced effect 误差归因诊断

- experiment_id: `cmdecoderv2-teacherforced-effect-attribution-v1.1.4`
- activity_id: [`cmdecoderv2-teacherforced-effect-attribution-20260908-202626`](activity_log.md)
- run_id: `cmdecoderv2-teacherforced-effect-attribution-20260908-202626`
- run_status: `COMPLETED`
- modification_version: `V1.1.4`
- operation_category: `diagnostic / operation`
- sequence/checkpoint: `s1/mouse_lift`，修正数据 decoder best + 冻结 OICM V1.2.5；逐帧只读复核，无新持久化输出。

**问题定义**

viewer 的 `教师强制` 是“GT 当前 Inspire state → decoder 预测下一 state → 预测 hand flow → OICM”，并非 GT hand flow 直通 OICM。需要用 GT hand flow→OICM 对照拆分 decoder 与 OICM 误差。

**结果**

- 近物体且 OICM 有效的 `275` 帧中，GT object effect `<=0.1 mm` 的 `126` 帧：teacher-forced 预测 effect RMS 均值/中位数/最大值 `7.317/7.293/10.120 mm`，GT effect RMS 均值 `0.048 mm`，prediction-GT EPE `7.286 mm`。
- 这 `126` 帧的 teacher-forced decoder hand-flow EPE 均值 `3.850 mm`，GT hand-flow RMS 均值 `1.179 mm`。
- 同一 `126` 帧将真实 GT hand flow 直接送入同一 OICM：预测 effect RMS 均值/中位数/最大值 `0.706/0.688/1.786 mm`，prediction-GT EPE `0.660 mm`。

**结论边界**

- `SUPPORTED`：6–7 mm 现象主要由 decoder 单步 hand-state/hand-flow 误差引起；GT hand flow 下 OICM 没有同量级固有偏置。
- `INCONCLUSIVE`：这不能单独证明 decoder 的所有 hand-state 误差来源，也不能外推到所有序列；teacher-forced 只排除了递归漂移。

**证据入口**

- [活动记录](activity_log.md)
- [完整 effect 运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)

## 2026-09-08 — 修正数据 best.pt 的中文 Inspire effect viewer（完整重跑）

- experiment_id: `cmdecoderv2-inspire-rollout-effect-viewer-v1.2.5`
- activity_id: [`cmdecoderv2-inspire-effect-v1.2.5-best-20260908-194900`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-20260908-195042`
- run_status: `RUNNING`（368 帧 effect 已计算；Viser `8104` 保持运行）
- modification_version: `V1.1.4`
- operation_category: `code / diagnostic / experiment / operation / documentation`
- decoder/OICM: 修正数据 decoder best epoch 24 / step 36120 与冻结 OICM V1.2.5 best；checkpoint SHA256 见 activity。
- sequence: `s1/mouse_lift`，368 个递归 rollout transition；GT object flow 只在当前帧独立计算，display-only。
- controls: 中文右侧控件支持任意合法帧 handoff、`教师强制`、`递归Rollout`、播放/停止、跳帧、`预测+GT`/`仅预测`/`仅GT` 和 raw/effective prediction 切换。

**结果**

- 远于 `50 mm`：12 帧全部 `sample_valid=false`，effective effect RMS `0 mm`。
- 近于或等于 `50 mm`：356 帧全部有效，effective effect RMS 均值 `4.528 mm`，中位数 `1.647 mm`，最大 `52.341 mm`。
- 全局 raw/effective effect RMS 均值 `4.998/4.380 mm`；GT effect RMS 均值 `2.693 mm`；预测-GT EPE 均值 `2.512 mm`；OICM valid ratio `0.9674`。

**结论边界**

- `SUPPORTED` 子结论：该 held-out 序列远距离 hand flow 仍被 OICM validity gate 屏蔽为零有效 object effect。
- `INCONCLUSIVE` 总结：单序列可视化和 display-only EPE 不能证明 decoder 整体动作质量或 MANO→Inspire 泛化。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect_summary.json)

## 2026-09-08 — 修正数据 best.pt 的 Inspire rollout self-effect 诊断

- experiment_id: `cmdecoderv2-inspire-rollout-effect-oicm-v1.2.5-best`
- activity_id: [`cmdecoderv2-inspire-effect-v1.2.5-best-20260908-125300`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-20260908-151014`
- run_status: `STOPPED`（368-step effect 已完成；8104 Viser 后被会话超时终止）
- modification_version: `V1.1.4`（复用既有诊断协议）
- operation_category: `diagnostic / experiment / operation`
- decoder: 修正数据正式 run 的 epoch 24 / step 36120 `best.pt`，SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`
- OICM: V1.2.5 `best.pt`，SHA256 `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`
- sequence: `s1/mouse_lift`，368 个纯 Inspire recursive rollout transition
- contract: 每步 rollout hand point flow 重新 forward OICM，保存 raw/effective `pred_obj_flow`；GT object flow 仅 display-only，不进入模型

**结果**

- 距离 `>50 mm`：12/12 帧无效，effective object-flow RMS 均值和最大值均为 `0 mm`。
- 距离 `<=50 mm`：356/356 帧有效，effective object-flow RMS 均值 `4.528 mm`、中位数 `1.647 mm`、最大值 `52.341 mm`。
- 全局 OICM valid ratio `0.9674`，effective effect RMS 均值 `4.380 mm`。

**结论边界**

- `SUPPORTED`：本 held-out 序列中，修正数据 decoder rollout 的远距离 Inspire motion 仍被 OICM 5 cm validity gate 正确屏蔽为零有效 object effect。
- `INCONCLUSIVE`：单序列诊断不能证明 decoder 整体动作或 MANO→Inspire 泛化已经正确；无效帧的 raw dummy head 输出也没有物理语义。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect_summary.json)

## 2026-09-08 — 修正 DExplore RL 数据上的 Temporal-D2 正式训练

- experiment_id: `cmdecoderv2-temporal-d2-oicm-v1.2.5`
- activity_id: [`cmdecoderv2-decoder-v1.2.5-20260908-101402`](activity_log.md)
- run_id: `cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402`
- run_status: `COMPLETED`
- modification_version: `V1.1.6`
- operation_category: `data / experiment / operation`
- input: 修正 DExplore RL 物体轨迹 decoder view + OICM V1.2.5 `best.pt`；decoder 从头训练

**结果**

- 50 epochs / `75250` steps，耗时 `02:32:18`。
- best epoch 24 / step 36120：`val/loss=0.0040070958`、`val/hand/point_flow_epe_mm=12.4907`、h1=`9.0049 mm`。
- final epoch 50：`val/loss=0.0041402271`、`val/hand/point_flow_epe_mm=12.5936`。

**结论边界**

- `SUPPORTED`：冻结 OICM V1.2.5 的正式 decoder 训练、验证和 best checkpoint 选择完整完成。
- `INCONCLUSIVE`：RL-Inspire validation 收敛不等于 MANO→Inspire recursive rollout 的跨 embodiment 效果成立，需单独可视化/诊断。

**证据**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt)

## 2026-09-10 — V1.3 中间 best.pt MANO/Inspire Cm 来源分类诊断

- experiment_id: `cmdecoderv2-cm-source-classifier-v13-v1.1.6`
- activity_id: [`cmdecoderv2-cm-source-classifier-v13-20260910-110919`](activity_log.md)
- run_id: `cmdecoderv2-cm-source-classifier-20260910-111037`
- run_status: `COMPLETED`
- modification_version: `V1.1.6`（分类诊断 Task）；上游冻结 OICM 为 `V1.3`
- operation_category: `diagnostic / experiment / operation`
- hypothesis: 若 V1.3 Cm 保留明显的手来源/embodiment 信息，只用 `cm_tokens` pooling 的小分类器也能在未见 sequence 上区分 MANO 与 Inspire。
- source: 冻结 OICM V1.3 训练过程中的中间 `best.pt`；标签 `mano=0`、`inspire_rl=1`。
- split: 使用 V1.3 现有 train sequence 训练分类器，val sequence 作为 `val-held-out`；正式 test 不使用；不做 frame-level train/test mixing。
- feature: 主输入为 V1.3 `unique_knn_edges` 路径输出的 `cm_tokens` permutation-invariant mean/max/std pooling；anchor pooling 为 object-side control。
- input: V1.3 cache 的 `K=32`、2 cm 半径、MANO `2048` 点、Inspire-F1 `10135` 点；有效 KNN 边去重后按 batch max 动态 padding。

**结果**

- 共同 object 类别 `11` 个；train `60 MANO + 60 Inspire` sequences，val-held-out `11 MANO + 11 Inspire` sequences；每条 sequence 抽取 `8` 个 transition，共 `1136` 个样本。
- 全部样本（train `960`，val-held-out `176`，两类各 `88`）：Cm val-held-out accuracy/balanced accuracy `57.95%`，F1 `0.4714`，AUROC `0.6475`；anchor control accuracy `57.39%`，AUROC `0.6058`。
- `sample_valid=true` 子集（train 可用 `443`、val-held-out 可用 `90`；平衡后训练 `208+208`、held-out `44+44`）：Cm val-held-out accuracy/balanced accuracy `73.86%`，F1 `0.7473`，AUROC `0.7531`；confusion matrix `[[31, 13], [10, 34]]`。
- 同一有效子集的 anchor control accuracy `62.50%`、AUROC `0.6689`，Cm 高于 anchor control。

**结论边界**

- `SUPPORTED`：在当前中间 OICM V1.3 checkpoint 和 `val-held-out` 口径下，Cm 保留了弱到中等的可识别 MANO/Inspire source/embodiment 信号。
- `INCONCLUSIVE`：该结果弱于旧 V1.2.1 分类实验，且不能单独证明是纯静态手型信息；运动幅度、接触状态和 source-domain 差异仍可能造成可分性。

**证据**

- [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/)
- [run manifest](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/run_manifest.json)
- [features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/features.npz)
- [metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/metrics.json)
- 中间 OICM checkpoint SHA256：`3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。

## 2026-09-07 — V1.1.6 MANO/Inspire Cm 来源分类诊断

- experiment_id: `cmdecoderv2-cm-source-classifier-v1.1.6`
- activity_id: [`cmdecoderv2-cm-source-classifier-final-20260907-214000`](activity_log.md)
- run_id: `cmdecoderv2-cm-source-classifier-20260907-213748`
- run_status: `COMPLETED`
- modification_version: `V1.1.6`
- operation_category: `diagnostic / experiment / operation`
- hypothesis: 若 Cm 保留明显的手来源/embodiment 信息，只用 Cm token pooling 的小分类器也能在 held-out sequence 上区分 MANO 与 Inspire。
- source: 冻结 ObjectInteractionCm V1.2.1 best checkpoint；标签 `mano=0`、`inspire_rl=1`。
- split: 使用现有 ObjectInteractionCm train/val sequence split；每个 split 按共同 object 类别平衡两种 source 的 sequence 数；不做 frame-level train/test mixing。
- feature: 主输入为 `cm_tokens` 的 permutation-invariant mean/max/std pooling；anchor pooling 为 object-side control。

**结果**

- 全部样本：Cm test accuracy `74.48%`、balanced accuracy `74.48%`、AUROC `0.8467`；anchor control AUROC `0.6263`。
- 只保留 `sample_valid=true` 的有效 Cm：Cm test accuracy `97.52%`、balanced accuracy `97.52%`、F1 `0.9752`、AUROC `0.9958`；anchor control accuracy `62.81%`、AUROC `0.6702`。
- 有效 Cm 测试 confusion matrix 为 `[[118, 3], [3, 118]]`，两类各 121 个平衡样本。

**结论边界**

- `SUPPORTED`：当前 Cm 包含明显可识别的 MANO/Inspire source/embodiment signal，且该信号在 object anchor control 之上仍然很强。
- `INCONCLUSIVE`：该实验不能单独证明 Cm 编码的是纯静态手型；运动幅度、接触状态和 source-domain 统计也可能贡献分类结果。

**证据**

- [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/)
- [run_manifest.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/run_manifest.json)
- [features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/features.npz)
- [metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/metrics.json)

## 2026-09-07 — V1.1.5 MANO rollout self-effect 对照诊断

- experiment_id: `cmdecoderv2-mano-rollout-effect-v1.1.5`
- activity_id: [`cmdecoderv2-mano-effect-full-20260907-203300`](activity_log.md)
- run_id: `cmdecoderv2-mano-effect-20260907-203231`
- run_status: `RUNNING`（348 步 effect 计算已完成；8103 Viser 仍运行）
- modification_version: `V1.1.5`
- operation_category: `diagnostic / experiment / operation`
- checkpoint: CmDecoderv2 best epoch 3 / step 10296；冻结当前锁定的 OICM `best.pt`
- sequence: `s1/camera_takepicture_3_Retake`；正式 MANO qualitative-only source，352 帧，348 个 rollout transition
- rollout_contract: handoff frame 0 使用 `q=0 + MANO wrist` 初始化 Inspire，之后递归反馈 decoder 预测 Inspire state
- input_contract: 当前 GRAB/MANO object geometry/pose、当前 MANO-source Cm window、当前 rollout Inspire geometry；不读取 paired Inspire GT、未来 object pose/flow 或 GT object flow
- output_contract: 保存 raw `pred_obj_flow` 与 `sample_valid` 屏蔽后的 effective effect；无效 sample 的 raw 输出仅作 dummy computational path 审计

**结果**

- 距离 `>50 mm`：258 帧全部 `sample_valid=false`，effective effect 为 0；raw effect RMS 均值 `12.535 mm`。
- 距离 `<=50 mm`：90 帧中 89 帧有效；effective effect RMS 均值 `9.194 mm`，最大 `41.302 mm`。
- 唯一无效的近距离帧为 frame `258`，距离约 `47.42 mm`、sampled active count `0`，符合随机采样未命中 candidate 的情况。

**结论边界**

- `SUPPORTED`：MANO-source rollout 也呈现和纯 Inspire rollout 一致的 OICM 5 cm validity gate；远距离 rollout hand flow 没有合同有效的 object effect。
- `INCONCLUSIVE`：MANO→Inspire decoder 的整体动作是否正确、以及远处动作是否由 decoder 产生了错误状态，仍需通过手部 rollout 几何和其它指标单独判断。

**证据**

- [运行目录](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/)
- [run_manifest.json](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/run_manifest.json)
- [effect.npz](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect.npz)
- [effect_summary.json](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect_summary.json)

## 2026-09-07 — V1.1.4 rollout self-effect OICM 旁路诊断

- experiment_id: `cmdecoderv2-inspire-rollout-effect-v1.1.4`
- activity_id: [`cmdecoderv2-inspire-effect-full-20260907-171315`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-20260907-171349`
- run_status: `RUNNING`（348 步 effect 计算已完成；8102 Viser 仍运行）
- modification_version: `V1.1.4`
- operation_category: `diagnostic / experiment / operation`
- checkpoint: CmDecoderv2 best epoch 3 / step 10296；冻结当前锁定的 OICM `best.pt`
- sequence: `s1/camera_takepicture_3_Retake`；352 帧，348 个 `state_t -> state_{t+1}` transition
- input_contract: 当前 object geometry/pose、当前 rollout Inspire geometry、当前 rollout hand point flow；不读取未来 object pose/flow，不使用 GT object flow
- output_contract: 保存 raw `pred_obj_flow` 与 `sample_valid` 屏蔽后的 effective effect；无效 sample 的 raw 输出仅作 dummy computational path 审计

**结果**

- 距离 `>50 mm`：122 帧全部 `sample_valid=false`，effective effect 为 0；raw effect RMS 均值 `12.533 mm` 不具物理语义。
- 距离 `<=50 mm`：226 帧中 225 帧有效；effective effect RMS 均值 `7.959 mm`，最大 `39.277 mm`。
- 唯一无效的近距离帧为 frame `127`，最近距离 `48.29 mm`、sampled active count `0`，符合 1024 点随机采样未命中 candidate 的情况。

**结论边界**

- `SUPPORTED`：本序列中 OICM 的 5 cm validity gate 能把远离物体的 rollout hand flow 屏蔽为无有效 object effect。
- `INCONCLUSIVE`：该旁路诊断不能单独判断 decoder 是否已经学会正确的远距离动作，或 MANO→Inspire rollout 的整体动作质量。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect_summary.json)

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
# 2026-09-13 - V1.1.14 MANO source→actual Inspire 微调（运行中）

- hypothesis: 在冻结 OICM 和阶段一 decoder 初始化下，用 MANO parent geometry 作为 source、Dexplore actual Inspire 状态/几何作为 target，可学习跨手 source-to-actual 的 decoder 适配。
- protocol: paired view `mano_actual_finetune_v1_1_14_eligible_20260913`，train=254、val=30，排除 `s2/mug_drink_2`（缺 MANO provenance）；6维独立输出，实际 target 保持完整 native 状态语义；三卡 `0,1,3`，per-device batch8/global24，20 epochs，未 resume optimizer/runner state。
- initialization: [initial.pt](../../../../outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_20260913/checkpoints/initial.pt) copied from completed stage-one best; OICM checkpoint SHA `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。
- run_id: `cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251`; run_status: `RUNNING`；step 200/31940 时 train h1 point-flow EPE=6.04841mm、loss=0.00434146。该快照不是终态，也不代表跨手或RL成功。
- evidence: paired source/target smoke 通过；17项定向 pytest 通过；正式三进程均完成 setup 并正常加载初始化权重。
- conclusion: `SUPPORTED`（实现和运行合同）；`INCONCLUSIVE`（微调收益、泛化、物理闭环）。终态指标、best/latest checkpoint 和独立评估待运行完成后补记。
# 2026-09-13 - V1.1.14 当前 best.pt 的残差 RL 接口 smoke

- protocol: 当前 MANO→actual 微调 best.pt（SHA `420ba5798bb668cc96bfbb266bf999bb17ef920b247410afc860245eca276201`）在 `s1/airplane_lift` paired view 上生成 `teacher_forced_one_step` decoder bank；IsaacGymEnvs 以显式 `decoder_bank` provider 读取该 bank，4 env、单 PPO epoch、GPU4。
- result: decoder bank 432 帧/覆盖428帧；RL task 成功加载 bank、18-DOF Inspire 资产，action `(12,)`、observation `(71,)`，PPO 单迭代 exit_code=0。
- output: 外部 `/home2/wyy/oyx_ws/IsaacGymEnvs/runs/CmResidual_13-10-13-11/`；Ref2Dex bank 在 `outputs/cmdecoderv2/rl_decoder_bank_s1_airplane_best_20260913/`。
- conclusion: `SUPPORTED_FOR_INTERFACE_SMOKE_ONLY`；`INCONCLUSIVE`（在线 decoder 闭环、残差策略训练、物理抓取）。由于单 epoch 未完成 episode，`rew=-inf` 不作策略指标。

## 2026-09-14 — V1.1.16 B1 终态及 F7 敏感性诊断

- experiment_id: `cmdecoderv2-field-realizer-gate-v1.1.16`
- modification_version: `V1.1.16`
- hypothesis: 在冻结训练合同下，object-indexed F7 是否被 Inspire FieldRealizer 用作动作条件，并具备可检验的增量控制价值。
- B1 training: `cm_decoder_v2_field_realizer_v1_1_16_20260913_224249`，20 epochs / 86460 steps，GPU7，run_status=`COMPLETED`；best epoch19/step82137，val loss `0.0124190375`，point-flow EPE `32.0424307 mm`；latest epoch20 EPE `32.0400469 mm`。
- final offline conditions: `field_conditions_best_v116_20260914_010500`，best checkpoint SHA256 `caaf37736d0bbcb72bf8087dcc6e06b2dcc12c391cb5cfaa9caed0182f490dc7`，4254 val windows（active horizon 16260，h1 4065）。B1 EPE `31.951362 mm`；C0（F7=0）`31.951363 mm`；Cs（anchor shuffle）`31.951362 mm`。相对 B1 的 q 变化分别为 `2.72e-5 rad` 和 `6.36e-7 rad`，说明当前 Realizer 对 F7 几乎不敏感。
- conclusion: `SUPPORTED`（B1 训练、验证、终态 checkpoint 和离线敏感性运行完成）；`INCONCLUSIVE`（不能据此判定 F7 在物理控制中无价值，也不能完成 A/R/B1/D 的 representation attribution）。
- evidence: [B1 结果报告](../../research/field_realizer_gate/results/V1.1.16_B1_progress_20260913.md)、[B1 run_manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_field_realizer_v1_1_16_20260913_224249/run_manifest.json)、[终态 conditions run_manifest](../../../../../outputs/cmdecoderv2/field_conditions_best_v116_20260914_010500/run_manifest.json)、[终态 evaluation](../../../../../outputs/cmdecoderv2/field_conditions_best_v116_20260914_010500/evaluation.json)。

- limitation: contact50 正式 physics Gate 尚未完成。GPU pipeline 下 `GymGetEnvRigidContacts` 在 simulation start 后被 Isaac Gym 明确拒绝，返回空 structured contacts；因此不能把 net contact force 或空 pairwise 输出当作 hand-object 接触结论。D 目前仅完成 284 条 MANO-H source cache，尚未完成满足 geometry parity 的正式训练。
