# correspondence_ptv3_v2 现有结果全方位对比

- generated_at: 2026-08-23
- scope: `src/task/correspondence_ptv3_v2/result/` 中截至当前的全部完成结果
- metric direction: QFL / MAE 越低越好；`pseudo_recovery_brier` 越高越好

## 1. 执行摘要

当前结果不存在一个在所有目标上都占优的单一 checkpoint，主要结论如下：

1. **跨数据域泛化**：在两套 ARCTIC object-only 域外评测中，纯 GRAB 总体优于 GRAB+ContactPose；OakInk-only 明显落后。加入 ContactPose 没有在当前 checkpoint 和协议下带来稳定 ARCTIC 增益。
2. **协议 E 的 10 mm hand perturb**：H80 best 的 hand-only 最终 QFL / recovery 最好；H50 best 的三条件等权 perturbed QFL 最好。历史 GRAB+ContactPose 两域版的最终 QFL 和 Balanced recovery 均明显落后。
3. **recovery 不能单独代表实际退化**：noPCA 在 10 mm hand-only 的 `pseudo_recovery_brier` 为 0.2295，ΔQFL 仅 +0.00002624；相对恢复、绝对退化和最终 QFL 必须并列报告。
4. **旧协议 D 的 hand 结果失效**：旧 evaluator 继承 checkpoint 的训练门控概率，H80/noPCA 与 H50 实际使用了不同的手扰动覆盖率；object-only 不受影响，hand-only / hand+object 标记为 `INVALID_IMPLEMENTATION`。
5. **当前三域 mix 尚无最终结论**：正在训练的 GRAB / ContactPose / OakInk 各 1/3 baseline 不属于本汇总中的历史 GRAB+ContactPose mixed；它尚未形成可写入 `result/` 的最终评测结果。

## 2. 评测协议地图

绝对数值只能在同一行协议内横向比较。不同子集的帧分布、changed-edge 构成、MANO 字段和 checkpoint 选择口径均可能显著改变 QFL 与 recovery。

### 2.1 统一报告口径

本文使用 `pseudo_recovery_brier` 作为唯一总体 recovery 主指标。它在所有显著 changed edges（默认 `|y_clean-y_pseudo| > 0.05`）上合并 fake-contact 与 missed-contact，并按伪几何平方误差自然加权：

```text
pseudo_recovery_brier = (E_pseudo - E_model) / E_pseudo
```

- `1`：完全恢复到 clean target；
- `0`：不优于直接信任扰动几何；
- `< 0`：比伪几何 baseline 更差。

`pseudo_fake_contact_recovery_*`、`pseudo_missed_contact_recovery_*` 和 `pseudo_recovery_projection` 只作为失败模式诊断，不参与本文主排名。绝对预测质量仍由 perturbed random QFL 补充，因为相对 recovery 高不等于最终 QFL 一定低。

| 协议 | 数据规模 | subject / object | hand noise | object noise | 主要回答的问题 | 证据强度 |
| --- | ---: | --- | --- | --- | --- | --- |
| A：ARCTIC 20-file | 20 文件 / 12,389 帧 | 小子集 | 关闭 | 10° / 10 mm | OakInk、GRAB、GRAB+ContactPose 谁的 object-only 域外泛化更好 | 方向性 |
| B：V1 分层 ARCTIC | 179 文件 / 93,968 帧 | 5 subjects / 11 objects | 关闭 | 10° / 10 mm | pure GRAB 与 GRAB+ContactPose 的 micro 和 object-macro 对比 | 当前最强域外证据 |
| C：旧 min11 object-only | 11 文件 / 约 4,175 帧 | s01 / 11 objects | 关闭 | 10° / 10 mm | noPCA 与 H80/O20 latest 的快速 object-only 筛查 | 方向性 |
| D：MANO min11 三条件 | 11 文件 / 4,175 帧 | s01 / 11 objects | axis-angle45，5 mm RMS | 可选 10° / 10 mm | 历史三条件；当前仅 object-only 可用 | hand/joint 失效 |
| E：MANO min11 10 mm 三条件 | 11 文件 / 4,175 帧 | s01 / 11 objects | axis-angle45，10 mm RMS，概率 1.0 | 可选 10° / 10 mm | 五条 checkpoint 的最终质量、ΔQFL 与相对 recovery | 当前有效机制筛查 |

### 2.2 为什么不把所有数字合并排名

- 协议 A/B 的 hand 输入是 stored clean hand points，不测试 hand perturb。
- 协议 D/E 使用重新生成且带 MANO 的 Stage 3；axis-angle45 与 GRAB PCA24 仅在几何 RMS 上标定，不是同参数分布。协议 D 的旧 hand/joint 结果另有门控不一致问题。
- 协议 B 覆盖 5 个 subject 和 93,968 帧；min11 只含 s01，且每个物体一个文件。
- `best.pt` 与 `latest.pt` 的选择目标不同；训练 global batch、optimizer step 和样本曝光量也并不统一。
- recovery 依赖 changed-edge 样本组成，跨子集的绝对值不能直接解释成模型退化或提升。

### 2.3 协议 A：ARCTIC 20-file 三模型快速域外对比

**数据口径**

- root：`tmp/arctic_eval_subset_20260817`；
- 20 个文件、12,389 帧；这是历史小子集，不是全量或 object/subject 严格均衡抽样；
- ARCTIC 不参与三个 checkpoint 的训练，用作 held-out domain。

**模型口径**

- OakInk-only：`best.pt`，step 140940 / epoch 10；
- pure GRAB compact：`latest.pt`，step 154670 / epoch 69；
- GRAB+ContactPose：`latest.pt`，step 154670 / epoch 54。

**输入与扰动口径**

- hand 始终使用落盘的 `stored_clean_hand_points`；MANO reconstruction 和 hand perturb 均关闭；
- runtime object resampling 关闭；
- clean stream 不施加输入扰动；
- perturbed stream 对 object 施加统一协议的 rotation std 10°、translation std 10 mm，概率 1.0；hand 保持 clean；
- 三个模型使用相同文件、batch、采样与扰动 seed。

**聚合与用途**

- 当前结果是整个子集上的 frame-weighted micro；没有 object-macro；
- 主看 clean/perturbed random QFL 和 perturbed `pseudo_recovery_brier`；
- 只能回答“这三个现有 checkpoint 在该小子集谁更好”，不能把差异严格归因于训练数据域，因为模型初始化、batch、loss 和训练预算并不统一。

### 2.4 协议 B：V1 确定性分层 ARCTIC

**数据口径**

- runtime root：`/tmp/arctic_eval_stratified_v1`；
- 来源 Stage 3：`/mnt/ugreen_nas/storage/oyx_storage/processed_data_backup/arctic/arctic_initonly_4096_hand_root_v2`；
- evaluator 实际读取 179 个 NPZ、179 条序列、93,968 帧，覆盖 s01/s02/s04/s05/s06 和全部 11 个 object；
- 对每个 `(subject, object)` 优先选择字典序最小的 `grab` 与 `use` paired sequence；不存在某 action 时选择该组合下字典序最小的实际序列；保留实际存在的左右手文件；
- 通过软链接构建子集，不复制或修改 NAS 数据。

**模型口径**

- pure GRAB compact 与 GRAB+ContactPose 使用和协议 A 相同的两条 `latest.pt`；
- 不包含 OakInk，因为本协议针对 V1 的 pure-vs-two-domain 问题。

**输入与扰动口径**

- 与协议 A 相同：stored clean hand、关闭 MANO/hand perturb/runtime resampling；
- clean stream 无扰动；perturbed stream 仅有 object 10°/10 mm，概率 1.0；
- batch size 16、num_workers 0、同一确定性采样和 evaluator。

**聚合与用途**

- micro：按全体帧/有效边全局聚合，保留自然帧频率；
- object-macro：11 个 object 分别运行相同 evaluator，再对 11 个类别算术平均，每个 object 等权；
- 这是当前最可靠的 pure GRAB vs GRAB+ContactPose ARCTIC 域外证据，但仍非全量 ARCTIC，且两条训练预算不等价。

### 2.5 协议 C：旧 Stage 3 min11 object-only 筛查

**数据口径**

- source：`/mnt/ugreen_nas/storage/oyx_storage/processed_data_backup/arctic/arctic_initonly_4096_hand_root_v2`；
- runtime root：`/tmp/arctic_eval_min11_v1`；
- 从 V1 清单中为每个 object 取一个确定性的首文件，共 11 文件、约 4,175 帧；全部来自 s01，覆盖 11/11 objects；
- 使用旧的已处理 Stage 3，不从 raw MANO 重新生成，因此该协议没有可用于 hand-noise 评测的完整 MANO schema。

**模型口径**

- noPCA `latest.pt`：step 317800；
- H80/O20 5 mm `latest.pt`：step 454000。

**输入与扰动口径**

- hand 使用 stored clean points，关闭 MANO/PCA hand perturb；
- runtime object resampling 关闭；
- clean stream 无扰动；perturbed stream 仅 object 10°/10 mm，概率 1.0。

**聚合与用途**

- 在约 4,175 帧上做 frame-weighted micro，不做 object-macro；
- 只回答 noPCA 与 H80/O20 的 object-only 快速方向，不回答 hand-only 或 compound robustness；
- 协议 D 已在相同物体/文件清单上补齐 MANO 和三条件，因此机制判断优先看协议 D。

### 2.6 协议 D：重建 MANO min11 三条件机制评测

**数据口径**

- raw source：`/mnt/ugreen_nas/storage/Ref2Dex_storage/arctic/data/arctic_data/data/raw_seqs`；
- Stage 3：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1`；
- 复用协议 C 的 11 个确定性 s01 左手文件与 11 个 object，共 4,175 帧；
- 从 raw `.mano.npy` 重新生成，包含 axis-angle45 pose、global orient、translation 和 betas；`mano_use_pca=false`；
- `ketchup` 经过 frame filtering 后仅 15 帧，因此该集合不是 object-balanced benchmark。

**模型 / checkpoint 口径**

- 第一组：noPCA latest（step 317800）vs H80/O20 latest（step 454000）；
- 第二组：H80/O20 best（step 399520 / epoch 44）vs H50/O50 best（step 236067 / epoch 13）；
- 四者使用同一 evaluator 和数据，但 latest/best 选择、global batch、step 和样本曝光量不统一，只能做行为与机制对照。

**三种 perturbed 条件**

| condition | hand 输入 | object 输入 | 回答的问题 |
| --- | --- | --- | --- |
| `object_only` | clean | 10°/10 mm perturb | 能否消除 object pose error 制造的伪接触变化 |
| `hand_only` | axis-angle45，5 mm RMS perturb | clean | 是否学会 hand pose noise recovery |
| `hand_object` | axis-angle45，5 mm RMS perturb | 10°/10 mm perturb | 对未在 exclusive recipe 中出现的 compound corruption 是否仍能恢复 |

hand noise 复用 ARCTIC axis-angle45 的 9 mm 几何标定并乘 `5/9` 缩放到目标 5 mm RMS；这只保证几何尺度对齐，不代表与 GRAB PCA24 noise 同分布。`hand_object` 评测时显式关闭 exclusive gate，使两种噪声同时生效。

**clean / perturbed 与聚合口径**

- 每个 condition 都生成一个 clean stream 和一个对应的 perturbed stream；三种 condition 的 clean 数值应仅有确定性采样/浮点级差异；
- `pseudo_recovery_brier` 只在 perturbed stream 的显著 changed edges 上定义，clean stream changed-edge fraction 为 0，不解读 clean recovery；
- batch size 16、num_workers 0、runtime object resampling 关闭；
- 所有 4,175 帧做 frame-weighted micro，不做 object-macro；
- 该协议适合诊断 corruption exposure 和 failure mode，不足以代表跨 subject、全量 ARCTIC 或严格 matched-budget 的最终排名。

### 2.7 协议 E：10 mm、固定 100% hand perturb

**数据与 checkpoint 口径**

- 数据、文件清单、帧数和前四个 checkpoint 与协议 D 完全相同；另加入历史 GRAB+ContactPose 两域 mixed latest.pt；
- object-only 没有 hand noise，因此与协议 D 数学等价，复用既有有效结果；
- hand-only / hand+object 将 axis-angle45 几何目标提高到 10 mm RMS，并运行全部五个 checkpoint。

**评估不变量**

- hand 条件显式设置 `hand_perturb_prob=1.0`，禁止继承 checkpoint 的训练 exposure；
- object 条件仍为 10° rotation std / 10 mm translation std，概率 1.0；
- batch size 16、num_workers 0、runtime object resampling 关闭；
- 五个 checkpoint 的 hand-only changed-edge fraction 均为 0.003200，hand+object 均为 0.004462，作为输入一致性核验。

**新增绝对退化口径**

```text
ΔQFL = perturbed random QFL - condition-matched clean random QFL
```

协议 E 联合报告 perturbed QFL、ΔQFL 和 `pseudo_recovery_brier`。它用于区分“最终预测差”“相对恢复差”和“小扰动分母放大 recovery”三种情况；仍然只做 min11 frame-micro，不承担最终 benchmark。

## 3. Checkpoint 与训练口径索引

| 简称 | 训练数据 / corruption | checkpoint 口径 | step / epoch | 关键预算差异 |
| --- | --- | --- | ---: | --- |
| OakInk-only | OakInk；无 hand perturb | best.pt | 140940 / 10 | 单卡 batch 16；runtime object resampling 开启 |
| pure GRAB compact | GRAB；无 hand perturb | latest.pt | 154670 / 69 | 双卡 global batch 128；旧 compact 协议 |
| GRAB+ContactPose | 两域联合；无 hand perturb | latest.pt | 154670 / 54 | 单卡 batch 64；与 pure GRAB 样本预算不等价 |
| noPCA | GRAB；H0/O100 近似口径 | latest.pt | 317800 / 未统一 | runtime sampling 路线；未完成与 5 mm 的 matched budget |
| 5 mm H80/O20 | 80% hand-only / 20% object-only / 0% compound | latest.pt | 454000 / 50 | 双卡 global batch 32 |
| 5 mm H80/O20 | 同上 | best.pt | 399520 / 44 | best 由 GRAB clean QFL 选择 |
| 5 mm H50/O50 | 50% hand-only / 50% object-only / 0% compound | best.pt | 236067 / 13 | 单卡 global batch 16；整条 454k run 实际约 25 数据 epoch |

> 注意：这里的 “mixed” 有两种不同含义。历史结果中的 mixed 指 GRAB+ContactPose 两域联合；当前训练中的 mixed 指 GRAB+ContactPose+OakInk 三域各 1/3，二者不可混为同一实验。

## 4. 协议 A：OakInk / pure GRAB / GRAB+ContactPose

数据为 ARCTIC 20-file 子集，12,389 帧，只施加 object 10°/10 mm perturb。

| 指标 | OakInk | pure GRAB | GRAB+ContactPose | 排名 |
| --- | ---: | ---: | ---: | --- |
| clean random QFL ↓ | 0.0004892 | **0.0001431** | 0.0001669 | GRAB > GRAB+CP > OakInk |
| perturbed random QFL ↓ | 0.0007355 | **0.0002136** | 0.0002670 | GRAB > GRAB+CP > OakInk |
| clean random MAE ↓ | 0.026147 | **0.002711** | 0.004029 | GRAB > GRAB+CP > OakInk |
| perturbed random MAE ↓ | 0.026676 | **0.003699** | 0.005797 | GRAB > GRAB+CP > OakInk |
| `pseudo_recovery_brier` ↑ | 0.5361 | **0.8383** | 0.8000 | GRAB > GRAB+CP > OakInk |

相对 GRAB+ContactPose，pure GRAB 的 clean QFL 低约 14.3%、perturbed QFL 低约 20.0%，`pseudo_recovery_brier` 高 0.0383。OakInk 的 clean/perturbed QFL 均约为 pure GRAB 的 3.4 倍，说明当前 OakInk-only checkpoint 的 ARCTIC 域外 correspondence 明显较弱。

**协议内结论**：pure GRAB 全指标占优；ContactPose 没有带来当前 ARCTIC 子集上的正增益；OakInk-only 只能提供有限 object recovery。

## 5. 协议 B：V1 分层 ARCTIC pure GRAB vs GRAB+ContactPose

这是当前覆盖最完整的域外结果：93,968 帧、5 subjects、11 objects，同时报告 frame-weighted micro 和 object 等权 macro。

### 5.1 Micro

| 指标 | pure GRAB | GRAB+ContactPose | 更优者 |
| --- | ---: | ---: | --- |
| clean random QFL ↓ | 0.00026570 | **0.00026452** | GRAB+CP（仅低 0.44%） |
| clean random MAE ↓ | **0.003752** | 0.004786 | pure GRAB |
| clean contact QFL ↓ | 0.005734 | **0.005557** | GRAB+CP |
| perturbed random QFL ↓ | **0.00033781** | 0.00036213 | pure GRAB（低 6.7%） |
| perturbed random MAE ↓ | **0.004607** | 0.006102 | pure GRAB |
| perturbed contact QFL ↓ | **0.007560** | 0.007859 | pure GRAB |
| `pseudo_recovery_brier` ↑ | **0.7981** | 0.7775 | pure GRAB（+0.0206） |

### 5.2 Object-macro

| 指标 | pure GRAB | GRAB+ContactPose | 更优者 |
| --- | ---: | ---: | --- |
| clean random QFL ↓ | 0.00027894 | **0.00027497** | GRAB+CP |
| clean random MAE ↓ | **0.003879** | 0.004895 | pure GRAB |
| perturbed random QFL ↓ | **0.00035004** | 0.00037329 | pure GRAB（低 6.2%） |
| perturbed random MAE ↓ | **0.004732** | 0.006204 | pure GRAB |
| `pseudo_recovery_brier` ↑ | **0.8102** | 0.7880 | pure GRAB（+0.0222） |

**协议内结论**：GRAB+ContactPose 在 clean QFL 上有极小优势，但 pure GRAB 在 perturbed correspondence、MAE 和 recovery 上稳定更好；micro 与 object-macro 方向一致，差异不是由单一高频 object 主导。应把结论表述为“pure GRAB 的鲁棒域外表现略优”，而不是“所有 clean 指标均优”。

## 6. 协议 C：旧 min11 object-only 快速筛查

| 指标 | noPCA latest | H80/O20 latest | 更优者 |
| --- | ---: | ---: | --- |
| clean random QFL ↓ | 0.0007073 | **0.0001345** | H80/O20 |
| clean random MAE ↓ | 0.018176 | **0.008157** | H80/O20 |
| perturbed random QFL ↓ | **0.0008409** | 0.0009378 | noPCA |
| perturbed random MAE ↓ | 0.018949 | **0.011547** | H80/O20 |
| `pseudo_recovery_brier` ↑ | **0.5760** | 0.3404 | noPCA |

**协议内结论**：H80/O20 的 clean 拟合明显更好；noPCA 的 object recovery 明显更好。该旧 Stage 3 快速筛查已被协议 D 的 MANO 三条件评测扩展，不应单独承担 hand robustness 结论。

## 7. 协议 D：MANO min11 三条件总表

> **实现失效说明**：旧 evaluator 没有固定 `hand_perturb_prob=1.0`，导致 noPCA/H80 与 H50 分别继承 0.8 和 0.5 的训练门控概率。7.2、7.3 的输入不一致，标记为 `INVALID_IMPLEMENTATION`；7.1 object-only 不受影响。有效的统一 100% 手扰动比较见协议 E。

### 7.1 Object-only

| 指标 | noPCA latest | H80 latest | H80 best | H50 best |
| --- | ---: | ---: | ---: | ---: |
| clean random QFL ↓ | 0.0002265 | **0.0000392** | **0.0000375** | 0.0000860 |
| perturbed random QFL ↓ | 0.0002819 | 0.0003012 | 0.0003039 | **0.0002591** |
| `pseudo_recovery_brier` ↑ | **0.5355** | 0.2929 | 0.2787 | 0.4119 |

### 7.2 Hand-only

以下数字仅保留为历史追溯，不参与结论或排名。

| 指标 | noPCA latest | H80 latest | H80 best | H50 best |
| --- | ---: | ---: | ---: | ---: |
| perturbed random QFL ↓ | 0.0002322 | 0.0000516 | **0.0000505** | 0.0000952 |
| `pseudo_recovery_brier` ↑ | -0.5066 | 0.2112 | **0.2122** | -0.0074 |

### 7.3 Hand+object

以下数字仅保留为历史追溯，不参与结论或排名。

| 指标 | noPCA latest | H80 latest | H80 best | H50 best |
| --- | ---: | ---: | ---: | ---: |
| perturbed random QFL ↓ | 0.0002862 | 0.0003169 | 0.0003199 | **0.0002640** |
| `pseudo_recovery_brier` ↑ | **0.5356** | 0.2813 | 0.2693 | 0.4116 |

### 7.4 当前可保留结论

只保留 object-only 结论：H50 best 的 perturbed QFL 最低，noPCA latest 的相对 recovery 最高。旧 hand / joint 排序不再作为 exposure trade-off 证据。

## 8. 协议 E：10 mm、100% hand perturb

协议 E 沿用同一 MANO min11 数据，将 hand noise 提高到 10 mm RMS，并在 evaluator 中固定手扰动概率为 1.0。object-only 复用协议 D 的有效结果；hand-only 和 hand+object 全部重跑。完整分项见 [`arctic_min11_mano_protocol_e_10mm_compare_20260823.md`](arctic_min11_mano_protocol_e_10mm_compare_20260823.md)。

| checkpoint | Balanced perturbed QFL ↓ | Balanced ΔQFL ↓ | Balanced recovery Brier ↑ |
| --- | ---: | ---: | ---: |
| noPCA latest | 0.00027900 | **+0.00005226** | **0.4339** |
| H80 latest | 0.00025684 | +0.00021759 | 0.3053 |
| H80 best | 0.00025872 | +0.00022123 | 0.2981 |
| H50 best | **0.00023450** | +0.00014843 | 0.3755 |
| GRAB+ContactPose latest | 0.00097292 | **-0.00000792** | 0.1672 |

历史 GRAB+ContactPose 的 Balanced ΔQFL 为负，但其 clean / perturbed QFL 均约为 0.001，且 Balanced recovery 最低；这表示噪声让一个高误差模型的 QFL 偶然略降，不代表最好鲁棒性。原四条新路线中，noPCA 的 ΔQFL 最小但最终 QFL 较高；H80 best 的 hand-only 最终质量最好；H50 best 的三条件等权最终质量最好。recovery、ΔQFL 和 perturbed QFL 必须联合解读。

## 9. 跨协议一致性与冲突

### 9.1 一致结论

- pure GRAB 在 ARCTIC object perturb 域外 `pseudo_recovery_brier` 上稳定优于当前 GRAB+ContactPose checkpoint；协议 A 与 B 一致。
- object-only 的排序与训练中 object exposure 的方向一致，但旧协议 D 的 hand/joint 结果因门控不一致不能用于同一论证。
- 协议 E 显示 H80 的 10 mm hand-only 最终 QFL / recovery 最好，H50 的三条件等权 perturbed QFL 最好。
- clean correspondence 与 perturb recovery 不是同一目标，多个实验都出现 clean 更好但 recovery 更差的情况。

### 9.2 表面冲突及解释

- 协议 B 中 GRAB+ContactPose 的 clean QFL 略优，但 MAE 与多数 perturbed/recovery 指标仍由 pure GRAB 占优；这属于指标侧重点不同，不应简化为“mixed clean 全面更好”。
- min11 的 recovery 绝对值显著低于较大 ARCTIC 子集，主要受 subject、序列和 changed-edge 分布影响，不能跨协议直接判定模型退化。
- H50/O50 best 出现在 epoch 13，是因为单卡保持 454k step 后整条 run 实际只有约 25 数据 epoch，并且 best 只按 clean GRAB QFL 选择；后期 perturbed/recovery 指标仍继续改善。

## 10. 按目标选择当前候选

| 目标 | 当前候选 | 依据 | 主要风险 |
| --- | --- | --- | --- |
| ARCTIC object-only 域外泛化 | pure GRAB compact | 两套外部子集均优于 GRAB+ContactPose / OakInk | 旧 compact 协议，和新 v2.1 recipe 不完全同源 |
| 5 mm clean correspondence | H80/O20 best | MANO min11 clean QFL 最低 | object/joint recovery 弱 |
| 10 mm hand-only 最终质量 | H80/O20 best | 协议 E hand-only QFL 0.00010732、recovery 0.3621 | object-only 最终 QFL 较弱 |
| 10 mm 三条件等权最终质量 | H50/O50 best | Balanced perturbed QFL 0.00023450 最低 | 训练预算不匹配；仅 s01 |
| 原四条新路线中的最小实际退化量 | noPCA latest | Balanced ΔQFL +0.00005226 最低 | clean 基线差，最终 QFL 较高 |

当前证据不足以指定“全局最佳模型”。如果最终目标同时要求 clean、hand-only、object-only 和 compound，下一步应优先做 matched-budget 的 H60/O40 或 H70/O30，而不是直接把 H50/O50 或 H80/O20 宣布为最终方案。

## 11. 科研限制与待补实验

1. **统一预算**：H80/O20 是双卡 global batch 32；H50/O50 是单卡 global batch 16。应按相同 global sample exposure 或相同 global batch 重跑。
2. **统一 checkpoint 选择**：当前 noPCA 只有 latest 口径参与三条件表；应补同一选择指标下的 best.pt。
3. **扩大 MANO 评测**：min11 只含 s01。应扩展至多 subject，并保留 object-macro 汇总。
4. **加入 compound 训练对照**：当前 H80/O20 和 H50/O50 都没有 compound 样本；需要单独比较允许 hand+object 同时扰动的 recipe。
5. **确定多目标选模规则**：现有 best 只按 GRAB clean QFL，无法代表三域或 robustness 最优。应预先定义多域、多条件 checkpoint selection。
6. **修复后的 5 mm 对照**：若仍需比较 5 mm，应使用已固定 100% 手扰动概率的 evaluator 重跑，不能复用旧协议 D hand/joint 数字。
7. **等待三域 mix 完成**：当前 GRAB / ContactPose / OakInk 各 1/3 run 尚在训练，完成后应按协议 B 和 E 分别做 object-only 域外评测与三条件机制评测。

## 12. 原始结果索引

| 文件 | 内容 |
| --- | --- |
| [`arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md`](arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md) | 协议 A：OakInk / GRAB / GRAB+ContactPose |
| [`arctic_grab_grabcontactpose_compare_20260820_090000.md`](arctic_grab_grabcontactpose_compare_20260820_090000.md) | 协议 B：V1 分层 ARCTIC micro / object-macro |
| [`arctic_min11_v1_compare_20260821.md`](arctic_min11_v1_compare_20260821.md) | 协议 C：旧 min11 object-only 快速筛查 |
| [`arctic_min11_mano_v1_compare_20260821.md`](arctic_min11_mano_v1_compare_20260821.md) | 协议 D：noPCA / H80 latest 三条件 |
| [`arctic_min11_mano_h80_h50_bestpt_compare_20260823.md`](arctic_min11_mano_h80_h50_bestpt_compare_20260823.md) | 协议 D：H80 / H50 best.pt 三条件 |
| [`arctic_min11_mano_protocol_e_10mm_compare_20260823.md`](arctic_min11_mano_protocol_e_10mm_compare_20260823.md) | 协议 E：统一 100% 的 10 mm hand 三条件与 ΔQFL |
| [`arctic_min11_v1_manifest.json`](arctic_min11_v1_manifest.json) | 旧 min11 数据清单 |
| [`arctic_min11_mano_v1_manifest.json`](arctic_min11_mano_v1_manifest.json) | MANO min11 数据清单与 noise schema |

## 13. 最终结论

现有证据最清楚地揭示了两个问题：训练域选择影响 ARCTIC object-only 域外泛化；相对 recovery、绝对 ΔQFL 与最终 perturbed QFL 描述的是不同目标。协议 E 中 H80 偏 hand-only 最终质量，H50 的三条件等权最终质量最好，noPCA 的实际退化量最小但基础误差较高。下一阶段应在统一预算、统一 checkpoint 选择和更大的多 subject MANO 数据上联合报告这三类指标。
