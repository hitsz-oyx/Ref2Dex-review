# ObjectInteractionCm 实验记录

- scope: task:ObjectInteractionCm
- related: [任务入口](../README.md)、[V1.1 执行计划](../plan/V1.1.md)、[V1.1 架构](../architecture/V1.1.md)、[活动记录](activity_log.md)

## 2026-09-12 — V1.3.2 无学习基线、刚性与误差份额诊断

- modification_version: `V1.3.2`；operation_category: `diagnostic / experiment / operation`。
- run_id: `mechanism_v1_3_2_val_20260912_131550`；run_status: `COMPLETED`，终态见 [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)。
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`；approval: `user-approved`（自主探索、继续）。
- 沿用 [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md) 第 8 节；不启动网络训练。
- 同一 58 条 val、7670 有效样本、28/30 序列、原 frozen checkpoint 预测；CPU 约 49.1 秒，约 11.54 MiB 输出。

**目的与可解释范围**

上一轮“低接触失效，所以应优先加 null”只依据条件误差，没有量化整体误差贡献。本次先检查该解释的
覆盖范围，再用无需学习的强基线判断复杂模型提供了多少增益。只重放原 Dataset，并逐样本核对相同
object points、GT、frame IDs、stride=2 与有效 unique-hand 点数；训练/数据/模型合同不变。

无学习对照只使用当前物体点和模型原本可见的 hand points/hand flow：

1. `mean_hand`：有效 unique-hand flow 向量的均值，作为所有物体点的平移。
2. `rigid_hand`：SciPy Rotation 对 current hand → current+hand-flow 拟合 SE(3)，作用于物体点。
   假设接触手与物体近似无滑移；少于 3 点、中心化 rank<2 或拟合不唯一时退回平均平移，共 25 帧。
3. `projected`：把当前 Cm 预测的物体点位移拟合成刚体变换，不使用 object GT。
4. `gt_rigid_residual`：仅用 GT 检查物体刚性，明确是 oracle 数据一致性检查，不是可部署基线。

**完整有效集结果**

| Object EPE / mm | MANO | Inspire RL |
| --- | ---: | ---: |
| 零流 | 29.9620 | 24.7855 |
| 平均手位移，无学习 | 8.4420 | 7.1632 |
| 手点刚体拟合，无学习 | 7.7724 | 7.9028 |
| 当前完整 Cm 模型 | 7.2907 | 5.7426 |
| 当前预测作刚体投影 | 7.2842 | 5.7326 |

Cm 相比平均手位移改善 `1.1513 / 1.4206 mm`，paired sequence-cluster 95% CI 分别为
`[0.5037,1.8441] / [0.8449,2.1415] mm`。这是真实平均增益，不能说 Cm 无效；但平均手位移
已取得完整模型相对零流改善的 `94.92% / 92.54%`，只与零流比较明显高估了复杂模型的必要性。
该百分比是误差差值比例，不是解释方差、信息量或因果贡献比例。

Cm 相比手刚体拟合改善 `0.4817 / 2.1602 mm`，CI 为
`[-1.6594,2.3808] / [1.5647,2.9861] mm`；MANO 上尚无明确超过这个无学习基线的证据。
这不等于两者等价；基线亦非适用于所有交互的物理模型，尤其手部形变或滑移时。

**修正低接触结论**

| active fraction <0.1 | MANO | Inspire RL |
| --- | ---: | ---: |
| 样本占比 | 6.65% | 4.93% |
| 总 EPE 份额 | 12.49% | 13.89% |
| 总 squared-error 份额 | 15.73% | 28.43% |
| 该组全部改为零流，全体 EPE 改善 / mm | 0.1023 | 0.1768 |
| 即使该组完美预测，全体 EPE 改善上限 / mm | 0.9108 | 0.7975 |

**因此撤回上一轮“总体效果差主要来自接触稀疏、应立即优先 null 分支”的判断。**
稀疏接触是局部 failure mode，但不能解释大部分 EPE；后验零流 gate 不是已训练 null 模型，
这些数字也不能证明所有 null-aware 架构均无价值，只说明目前没有足够收益依据将其排第一。
实际上在低接触组，平均手位移 EPE `23.20 / 30.27 mm`，Cm 为 `13.70 / 16.19 mm`，
Cm 相比简单运动随动的优势反而更大。“比零流差”与“没有学习有用信息”不能混为一谈。

误差最高约 1% 的样本贡献总 squared error 的 `57.08% / 53.08%`；最高 5% 贡献
`84.34% / 81.50%`，但其总 EPE 份额仅 `35.58% / 37.31%`。应区分 RMSE 与 EPE 的长尾问题，
不能把所有大误差都归因于低接触，也不能按这些事后 GT 排序的帧做部署 gate。

**架构与数据检查**

- GT 刚体拟合平均残差 `0.0000626 / 0.0000653 mm`：本次检查没有发现非刚性/点对应不一致；
  不能由此断言物理仿真、力学对应、接触标注或源分布全部正确。
- 预测的非刚体残差约 `0.4867 / 0.3261 mm`，但投影后 EPE 仅改善 `0.0065 / 0.0100 mm`。
  对现有输出作 SE(3) 投影不足以解释/消除数 mm 主误差；不等于 SE(3) 训练参数化一定无效。
- [src/task/ObjectInteractionCm/runner.py](../../runner.py) 的 `slot/effective_count` 根据
  `cm_slot_weights.mean(dim=-1)` 计算。每个 slot 的权重已沿 object points 归一化到和为 1，
  所以该量几乎必然得到 `S=16`，不是 slot 使用充分或未塌缩的证据。只定位未改核心指标。
- 归档 Cm 跨 slot 的中心化谱有效秩约 `5.57 / 5.53`，跨 slot 标准差/整体 RMS 约 `0.371 / 0.373`，
  不是所有 tokens 数值相同。谱秩不等于语义 slot 数或信息量，不据此宣称“只有 5 个有效 slots”。
- 训练末期 epoch=275，train object EPE=8.1995 mm，val=6.6839 mm；前者 stride=1..10、
  后者 stride=2，不能用这个差值判断欠拟合或过拟合。最佳 checkpoint 仍是 epoch=180。

**结论与研究取舍**

- `SUPPORTED`：相同输入与有效样本下，完整模型相对平均手位移有约 1–1.4 mm 增益；此前零流是弱基线。
- `REFUTED`：低接触样本贡献了大部分总 EPE，以及 `slot/effective_count=16` 可证明充分 slot 利用率的解释。
- `INCONCLUSIVE`：Cm bottleneck 是否必要、是否限制效果、geometry-only 性能、未见手型泛化及表征不变性。
- 暂不优先大规模 source-heldout、null 分支或刚体输出改造。下一项最有区分力的学习对照是
  **相同输入和局部交互 encoder，绕过 Cm 压缩直接预测 effect**，同时统一 train/val 评估 stride。
  若明显优于当前模型，定位瓶颈/解码路径；若同样差，优先调查条件信息、监督和分布，不能盲目扩充 slots。
  geometry-only 可保留为辅助条件消融，但它不能单独定位 Cm 压缩损失；本轮没有训练该模型。

**产物与复核**

- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run_manifest.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run_manifest.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/config.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/config.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metadata.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metadata.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run.log](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run.log)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json)

原输入/预测 archive、index/scale/checkpoint SHA256 和几何文件 stat 前后相同；全部 7670 样本
与原运行精确对应。另以独立 NumPy SVD 公式复核两源 12 个真实样本，刚体基线 EPE 与 SciPy
实现最大差 `2.79e-13 mm`；smoke 的 6 行与 full 对应行完全相同；全部分源均值独立重算通过。

## 2026-09-12 — V1.3.1 同一冻结 decoder 的跨源 object-effect 诊断

- modification_version: `V1.3.1`
- operation_category: `diagnostic / experiment / operation`
- run_id: `cross_source_effect_v1_3_1_val_20260912_112900`
- run_status: `COMPLETED`；终态以 [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) 的 `ACT-20260912-113510-OICM-CROSS-SOURCE-END` 为准。
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`；dirty code 以 metadata 的代码 SHA256 锁定。
- approval: `user-approved`；[src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md) 第 7 节。
- initial_checkpoint: [outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)
- 权重 SHA256：`3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`；epoch=180、step=132480、best equal-source EPE=6.5166713276 mm。

**问题与冻结条件**

H1：混合训练的同一个冻结模型能否分别从两源 Cm 重建各自 object flow，并在平均误差上优于零流？
H2：控制部分任务难度后，能否证明不同手型 Cm 表达等价的 object effect？本实验没有同交互配对，
因此 H2 只能作为有限的源差距诊断，不能完成严格等价检验。

全部 encoder、slot、object/hand decoder 均 eval + inference_mode，参数 requires_grad=false，
无 optimizer/训练。object decoder 输入为 raw object points/normals、Cm tokens、动态 anchors，
不是 token-only。两源均参与过训练，val 也参与 checkpoint 选择，不能称为 unseen-hand 测试。
checkpoint 自身配置恢复；右手、30 Hz、stride=2、object_pose_t、米、4096/1024 点、KNN32、
2 cm、unique valid-edge hand stream、动态 padding 和 train-only scales 保持不变。

58 条 val（MANO 28 / Inspire RL 30）；seed=42、dataset epoch=0，复用现有 fixed-stride 行范围，
每序列 T-2，共 14622 行。关闭本诊断 loader 的 active_only 只为统计覆盖率；主误差只统计
full-pool active 且 sampled-valid 的 7670 行。GT 从未送入模型，仅用于指标和事后离线分箱。

**完整验证集结果**

EPE 是物体点三维误差范数的均值；下表 EPE/RMSE 均为 mm，主行使用 frame micro。

| 指标 | MANO | Inspire RL |
| --- | ---: | ---: |
| 完整行数 / full-pool active / sampled-valid | 6755 / 3613 / 3610 | 7867 / 4065 / 4060 |
| full-pool active 中 sampled-valid 比例 | 99.917% | 99.877% |
| 冻结完整 Cm EPE | 7.290740 | 5.742603 |
| EPE 95% sequence-cluster CI | [5.6385, 8.9609] | [4.1401, 7.6198] |
| 冻结完整 Cm RMSE | 16.126230 | 12.902363 |
| 零流 EPE | 29.962009 | 24.785472 |
| 相比零流 EPE 降低 | 75.667% | 76.831% |
| hand-flow 空间置换 EPE | 8.882639 | 6.232415 |
| 置换 EPE 增量 95% CI | [1.2207, 2.0060] | [0.2711, 0.7541] |
| 零-token 保留 anchors EPE | 47.044641 | 43.269641 |
| 序列等权 EPE | 6.768255 | 7.203126 |
| hand-flow RMS 均值 | 32.502318 | 26.945168 |
| object-flow RMS 均值 | 30.156014 | 24.984936 |
| full-pool active fraction 均值 | 0.308063 | 0.474365 |

equal-source EPE=6.5166713434 mm，与 checkpoint best metric 相差约 1.6e-8 mm。
frame-micro 源差 MANO minus Inspire=+1.548137 mm，95% CI=[-1.0364, 3.9388]。
两源任务难度、接触比例、序列长度不同；切换到序列等权时源排序也改变，不能将原始均值差
归因于手型，更不能把 CI 跨零解释为等价。

**预定匹配结果**

exact object name + hand RMS / GT object RMS 的固定倍增分箱 + full-pool active fraction 分箱，
每 stratum 两源等量无放回抽样，seed=42。没有按结果修改分箱、剔除失败案例或调参。
共 11 类物体、115 个共同 stratum，两源各 458 行，分别来自 14 / 12 条序列，仅占各自
有效样本的 12.687% / 11.281%。匹配的 CI 条件于已选择样本，按 source 内序列重采样 2000 次。

| 指标 | MANO | Inspire RL |
| --- | ---: | ---: |
| 完整 Cm EPE / RMSE | 5.735400 / 8.236668 | 6.817376 / 11.798896 |
| EPE 95% CI | [4.6750, 6.8967] | [3.5582, 10.3771] |
| 零流 EPE | 30.735341 | 31.036106 |
| hand-flow 空间置换 EPE | 7.659373 | 7.779821 |
| 零-token EPE | 45.698639 | 46.081183 |
| 序列等权 EPE | 5.697255 | 9.641658 |
| hand / object RMS 均值 | 31.475611 / 30.924032 | 32.341697 / 31.395894 |
| active fraction 均值 | 0.373257 | 0.396776 |

matched 源差=-1.081977 mm，95% CI=[-4.7626, 2.3963]；源 EPE ratio=0.8413，CI=[0.5332, 1.6353]。
这是粗难度匹配，不是相同物体状态、接触构型、动作方向、动力学条件或相同目标 effect 的配对。
部分序列只保留极少帧，例如 Inspire hammer_use_2 的 4 帧 EPE=46.47 mm、stamp 的 1 帧 EPE=13.40 mm；
序列等权匹配结果对这类小样本序列敏感，不应只报告较接近的 frame micro。

**反例与解释边界**

完整有效集上，MANO 1/28 条序列（s8/mouse_lift）、Inspire 6/30 条序列
（s1/torussmall_lift、s2/toothbrush_lift、s3/pyramidsmall_pass_1、s6/hand_inspect_1、
s8/cubelarge_inspect_1、s9/hammer_use_2）平均 EPE 不优于零流。平均有效不等于所有交互均有效。

空间置换只破坏有效 unique-hand 点的 flow/位置对应，保留向量集合与几何/KNN；
近似刚体平移不会被完全破坏。两源正的置换 penalty 支持利用部分空间运动对应信息，
不证明模型只靠运动而非几何。零-token 保留原动态 anchors，且属于 OOD 消融；
它支持当前 decoder 对 token 敏感，不能替代训练过的 geometry-only baseline 或 token 充分性证明。

**结论**

- `SUPPORTED`：在这份 mixed-source、val-selected checkpoint 的有效验证样本上，两源完整 Cm
  经同一个 decoder 的平均重建均显著优于零流；相对零流增益的 sequence-cluster CI 均大于 0。
- `SUPPORTED`：工程有限值、decoder 精确重放、权重和输入不变、完整行覆盖、独立 NPZ 数值复算。
- `INCONCLUSIVE`：跨手型表征不变性、同 effect 编码等价、相近预测的等价性、未见手型泛化。
  两源训练可共享 decoder 是必要线索，但模型仍可能在内部保留 source-specific 编码。
- 下一阶段需要另行批准：双向 source-heldout encoder/decoder 训练与跨源测试；如要直接回答
  “同 effect 不同手型是否一致”，还需同物体初态/目标 effect 的配对数据与预定等价容差。

**产物与复现**

- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run_manifest.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run_manifest.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/config.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/config.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metadata.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metadata.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metrics.jsonl](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run.log](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run.log)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_summary.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_summary.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/matched_selection.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/matched_selection.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_comparison.png](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_comparison.png)

确切命令见 activity；默认同代码运行使用新的 run-id，不覆盖目录。最终运行约 149.3 秒/870.15 MiB。
首轮 manifest 将 `split: val` 误判为文件路径，本地改名 `evaluation_partition` 后复跑；两轮
metrics.jsonl 与 matched_selection.json 逐字节相同，统计/覆盖/工程检查也完全相同。
独立 NumPy 复算全部 58 个 NPZ，最大 Torch/NumPy EPE 差 0.00002693 mm；9 项 Task 测试通过。

## 2026-09-12 — V1.3.1 接触稀疏性后验诊断

此节保留当时观察；“应优先 null/低接触分支”的解释已被上方 V1.3.2 的误差份额与强基线检查修正。

基于同一冻结运行的逐样本指标进行只读分箱分析，没有重新选择 checkpoint，也没有把后验阈值当成模型结果。

- 低 active fraction 是主要 failure mode：MANO `<0.1` 时完整 Cm EPE 高于 zero-flow，Inspire RL 在
  `<0.1`、`0.01–0.05`、`0.05–0.1` 等区间同样高于 zero-flow；相反，接触比例 `>=0.25` 时两源均有明显增益。
- MANO / Inspire 的中位 EPE 为 `4.01 / 2.49 mm`，但 p99 为 `69.62 / 59.46 mm`。所以“平均效果差”
  更准确的表述是长尾鲁棒性差，而不是所有样本都没有 object-effect 信号。
- active `<0.1` 的样本只占 MANO `6.65%`、Inspire `4.93%`，但其预测经常比 zero-flow 更差。
  当前 V1.3 只跳过完全无有效 interaction 的样本；低接触但非空样本仍然进入 decoder，可能导致伪确定性输出。
- 这暴露出比跨手型更优先的架构问题：需要让模型能够表达 no-effect/低置信度，或至少估计接触充分性，
  而不是把所有 `sample_valid=true` 都当作同等可靠的 object effect。

**下一步决策**

先做一个受控的 low-contact/null-aware 训练对照，并与 geometry-only control 同时建立。对照必须使用相同
split、采样、预算和 checkpoint 选择规则；报告接触分层、长尾 p95/p99、zero-flow gain 和序列 cluster CI。
若 Full 相对 geometry-only 没有稳定增益，则停止跨手型表征路线；若增益只在高接触区存在，则先重新定义
Cm 的适用域和置信度输出，再做 source-heldout。配对同-effect 数据仍是等价性问题的最终验证，不是当前优先项。

## 2026-09-10 — V1.3 unique-KNN-hand 全量训练（进行中）

- run_id: `object_interaction_cm_dexplore_rl_v1_3_20260910_020856`
- run_status: `RUNNING`
- modification_version: `V1.3`
- operation_category: `experiment / operation`
- base_commit: `158e0f34068e260d7077a94b4457799b7fca2f31`
- seed: `42`
- initial_checkpoint: `null`（from scratch）
- data index: [V1.3 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json)
- scale manifest: [V1.3 unique-KNN scale](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3_unique_knn.json)
- split: train `509`（MANO `254` / Inspire-F1 `255`）、val `58`（`28` / `30`）、test `63` MANO-only；沿用 V1.2.5 assignment。
- training: 右手、`object_pose_t`、30 Hz、object pool/sample `4096/1024`、KNN `K=32`、interaction/hand supervision radius `2 cm`、MANO/Inspire KNN hand `2048/10135`、unique valid-edge hand supervision、batch-max dynamic padding、每卡 batch `32`、global batch `96`、目标 `202300` steps、物理 GPU `0,2,3`。

**假设与变量**

在不改变 split、GT、坐标系和 object sampling 语义的前提下，使用采样 object 点的有效 KNN 边构造
interaction；将这些边中的 hand ID 去重后，每个 hand 点只进入一次 hand decoder/loss。动态 padding
只到当前 batch 的最大 unique hand count，不将完整 Inspire `10135` 点固定送入每个 batch。

**运行入口**

- [V1.3 指导](../指导/V1.3.md)
- [V1.3 执行计划](../plan/V1.3.md)
- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/)
- [配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/config.json)
- [运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/run_manifest.json)
- [逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/metrics.jsonl)
- [训练日志](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/train.log)
- [最佳 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt) 与 [最近 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/latest.pt) — 已生成；epoch `1` 至 `4` 阶段 checkpoint 已生成。

**当前状态与结论边界**

- 已完成 `step=144000`、`epoch=196`；训练进程仍为 `RUNNING`，最新吞吐约 `653.63 samples/s`，ETA 约 `2.38 h`。
- equal-source object validation 最佳为 `6.516671 mm`（epoch 180 / step 132480）；最近验证为
  `6.714598 mm`（epoch 195 / step 143520）。最佳之后连续 15 次验证未刷新 best，最近窗口均值
  `6.700502 mm`，当前判断为 object 指标进入平台期，但正式训练尚未完成。
- GRAB 最近/最佳 object EPE 为 `7.658820 / 7.239350 mm`；Inspire-F1 为
  `5.770376 / 5.701175 mm`；unique-KNN hand 最近为 `1.921042 mm`（当前最佳）。
- 当前仅支持工程启动和 forward/backward 运行正常；正式科研结论保持 `INCONCLUSIVE`，待完整训练、
  validation 和 checkpoint 结果后再更新。

## 2026-09-07 — V1.2.5 修正 DExplore 实际物体轨迹、KNN=16 的 Cm 重训（已完成）

- run_id: `object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606`
- run_status: `COMPLETED`
- modification_version: `V1.2.5`
- operation_category: `experiment / operation / data`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- seed: `42`
- initial_checkpoint: `null`（from scratch）
- data index: `data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json`
- scale manifest: `data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/scales_train_v1_2_5.json`
- split: train `509`（MANO 254 / RL-Inspire 255）、val `58`（28 / 30）、test `63` MANO-only；630 个 parent sequence 全部互斥。
- training: 实际 simulated object trajectory、右手 1538 点、object pool/sample `4096/1024`、`D=128,C=32,S=16,K=16`，global batch 96，GPU 0/1/2，最大 202300 steps。

**假设与变量**

修正后的 RL-Inspire hand/object 联合轨迹能够为 ObjectInteractionCm 提供一致的 object-flow GT；将局部
KNN 从 8 改为 16 后，从随机初始化学习新的 object-centric Cm。其余 stride、source probability、loss、
优化器和 checkpoint 选择规则沿用 V1.2.3。旧错误轨迹 checkpoint 不参与初始化或对照选择。

**运行入口**

- [V1.2.5 执行计划](../plan/V1.2.5.md)
- [修正 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)、[index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json) 与 [KNN=16 scale](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/scales_train_v1_2_5.json)
- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/)、[配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/config.json)、[运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/run_manifest.json)、[逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/metrics.jsonl) 与 [训练日志](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/train.log)
- [当前 best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/best.pt) 与 [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/latest.pt)

**结果与结论边界**

- 三卡 2-step smoke 正常完成；正式 run 在 262 个 epoch 后达到 `202300/202300` steps，自然结束，耗时
  `06:30:21`。全程 loss/gradient finite，未发现 traceback、NaN、OOM 或 NCCL failure。
- 按两 source 等权 `val/obj/flow_epe_mm` 选择的 best 位于 epoch 143 / step 110682：总指标
  `6.422504 mm`，MANO `7.243497 mm`，RL-Inspire `5.601512 mm`；对应 source hand-flow EPE 为
  MANO `2.809946 mm`、RL-Inspire `1.776661 mm`。
- 最终 epoch 262 / step 202300 的等权 object EPE 为 `6.706175 mm`（MANO `7.733837 mm`、RL-Inspire
  `5.678513 mm`），略差于 best；后续消费必须显式使用 `best.pt`，不能把 `latest.pt` 当作最佳模型。
- 结论 `SUPPORTED`：修正实际物体轨迹、KNN=16 的 Cm 训练链路稳定完成并生成有效 checkpoint。
  该结论不证明 CmDecoderV2 或 rollout 已改善；下游效果仍为 `INCONCLUSIVE`，需使用本轮 best checkpoint
  重新训练/评估 decoder。

## 2026-09-05 — V1.2.3 Dexplore RL/MANO right-hand Cm mixed training（已停止）

- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `STOPPED`
- modification_version: `V1.2.3`
- operation_category: `experiment / operation / data`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- seed: `42`
- initial_checkpoint: `null`（from scratch）
- data index: `data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json`
- scale manifest: `data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json`
- split: train `1004`（MANO 501 / RL-Inspire 503）、val `126`（64 / 62）、test `125` MANO-only；同一 parent sequence 不跨 variant。
- stride: 两种 source 均为 30 Hz、stride `1..10`；val/test 固定 stride 2。
- training: 右手 1538 点，object pool 4096 / sample 1024，`D=128,C=32,S=16,K=8`，source probability `0.5/0.5`，每卡 batch 32、global batch 96、最大 202300 steps，GPU 0/1/2。

**假设与边界**

新 Dexplore RL-Inspire 与 MANO variant 在不共享 parent sequence 的前提下混合训练，能够学习不依赖 source/手形态 ID 的 object-centric Cm。当前只证明训练工程链路；CmDecoderV2 和 test 可视化不在本 run 内，科研结论保持 `INCONCLUSIVE` 直到正式训练终态及后续 decoder 验证。

**运行入口**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)
- [配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)
- [运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)
- [逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)
- [训练日志](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [最佳 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt) — epoch 46 / step 81420，equal-source object EPE `9.470077 mm`。
- [最近 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt) — epoch 65 / step 115050，包含 optimizer/scheduler/scaler，可恢复。

**结果与终态**

- run 在 step 115900 / epoch 66 非正常停止，只完成 202300 计划 steps 的约 57.3%；最后完整 checkpoint/validation 为 step 115050 / epoch 65。
- 最佳 validation 位于 epoch 46 / step 81420：equal-source object EPE `9.470077 mm`，MANO `6.301521 mm`，RL-Inspire `12.638633 mm`；对应 hand EPE 为 MANO `4.170636 mm`、RL-Inspire `3.261057 mm`。
- 最后完整 validation 的 equal-source object EPE 为 `9.574271 mm`；训练到中断前 loss/gradient finite，未发现 NaN、Python traceback、CUDA OOM 或 kernel OOM 证据。
- 进程在日志无终止标记的情况下消失，准确外部信号未知；因此 `run_status=STOPPED`，科研结论保持 `INCONCLUSIVE`。该中途 checkpoint 可用于恢复，但不能当作完成的 202300-step 正式结果。

## 2026-09-03 — ObjectInteractionCm best 的 GRAB/Inspire-F1 t-SNE

- run_id: `objectinteractioncm_tsne_best_20260903`
- checkpoint: `outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt`（epoch 17 / step 193953，`object_pose_t`）
- split: test；GRAB stride `1..10`，Inspire-F1 偶数 stride `2..20`；各 source 每 stride 等量采样，最终 520/source、1040 条 pooled 样本。
- 产物：[output/research/objectinteractioncm_tsne_best_20260903/](../../../../../output/research/objectinteractioncm_tsne_best_20260903/)
- pooled 原空间 dataset silhouette=`0.1835287`；stride 着色在主体区域内明显交叠，而 dataset 着色仍可见 source-specific 区域。该结果是探索性表征诊断，不能替代下游泛化评估；结论状态 `SUPPORTED`（运行和产物有效）。

## 2026-09-03 — V1.1.3 GRAB + Inspire-F1 available-hand union 全量训练

- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- modification_version: V1.1.3
- operation_category: experiment / operation
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- seed: 42
- initial_checkpoint: null
- conclusion: INCONCLUSIVE

**假设与变量**

在 V1.1 冻结合同下，单一 `16×32` object-centric Cm 可以从所有真实可用 hand points 的运动中稳定学习 object/hand flow。GRAB 与 Inspire-F1 按 `0.5/0.5` source probability 混训；GRAB stride 为 `1..10`，Inspire-F1 stride 为偶数 `2..20`；object pool 4096 点、运行时采样 1024 点；不输入 source/side ID 或 GT object flow。

**运行入口**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/)
- [配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)
- [运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)
- [终态摘要](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json)
- [逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl)
- [最佳 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt)

**结果**

训练在 18 个 epoch 后达到 202300 steps 并正常结束，无 NaN/Inf 或未处理异常。按两 source 等权 object-flow EPE 选择的最佳 checkpoint 位于 epoch 17 / step 193953：

| 指标 | best epoch 17 | final epoch 18 |
| --- | ---: | ---: |
| equal-source object EPE (mm) | 4.089469 | 4.097444 |
| GRAB object EPE (mm) | 5.823295 | 5.851712 |
| Inspire-F1 object EPE (mm) | 2.355643 | 2.343177 |
| GRAB hand EPE, 3 cm mask (mm) | 2.729673 | 2.728593 |
| Inspire-F1 hand EPE, 3 cm mask (mm) | 0.777010 | 0.774676 |

**结论边界**

该运行支持“实现能够稳定完成混合训练并生成可用 checkpoint”这一工程判断。由于尚未运行 held-out test、旧 Cm/OI-Cm 对照或下游任务验证，不能判断 V1.1 表征是否优于基线或具有预期泛化能力，科研结论标记为 `INCONCLUSIVE`。
