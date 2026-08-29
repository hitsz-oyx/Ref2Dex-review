# Cm 长程训练结果对比

- scope: task:Cm
- snapshot: 2026-08-22 09:40 CST
- source: `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/docs/logs/experiment_log.md`、历史配置与各 run 的 `config.json`、`metrics.jsonl`
- comparison rule: 只让至少完成 20 个完整 epoch 的 run 进入主表；80-step 吞吐测试、300-step smoke/短训和约 2--3 epoch 的 10k pilot 不参与主比较

## 结论摘要

| Run | 当前结论 | 收敛判断 | 主要问题 |
| --- | --- | --- | --- |
| Mixed GRAB+ARCTIC，no-gate，C=256 | 当前三条 object-v2 run 中 mixed validation EPE 最低；仍在训练 | 基本收敛但尚未完全平台化；epoch 37 仍刷新 best | 使用旧 GRAB cache，不能作为修复后数据的正式结果 |
| GRAB-only，gate，C=64，无 warm-up | 50 epoch 已完成；GRAB 相对 EPE 最好 | 已收敛，epoch 48 与 50 几乎持平 | hard gate 几乎退化为单 slot；使用旧 GRAB cache |
| GRAB-only，gate，C=64，有 warm-up | warm-up 没有长期阻止单 slot collapse | 尚未完成；epoch 29 后暂未继续改善 | epoch 30 时仍约 1.09 个 active slot；使用旧 GRAB cache |
| 旧 Stage4 GRAB，no-gate，C=256 | 20 epoch 完成，EPE 12.366 mm | 最后一次验证仍改善，预算结束时尚有缓慢下降 | 旧数据/缩放/评估合同，只能与同组 gate 直接比较 |
| 旧 Stage4 GRAB，gate，C=256 | 20 epoch 完成，EPE 12.399 mm | 最后一次验证仍改善，整体与 no-gate 持平 | gate 把 stride-5 effective branches 从 10.04 压到 2.13，但没有改善 EPE |
| 旧 Stage4 GRAB 单手筛选，no-gate，C=256 | 旧数据长训中 EPE 最低，为 10.398 mm | 20 epoch 完成 | 数据经过 dominant-hand manifest 筛选，不能与未筛选 run 直接归因比较 |
| 旧 Stage4 GRAB 单手筛选，gate，C=256 | EPE 10.954 mm，落后同数据 no-gate | 20 epoch 完成 | gate 完全退化到约 1 个 effective branch |

当前数据只支持优化行为比较，不支持把差异归因于某一个变量：mixed run 同时改变了数据集、`cm_dim` 和 gate；两条 GRAB gate run 才能较干净地比较 warm-up。跨 mixed 与 GRAB-only 比较时应优先看 relative EPE / zero-flow improvement，不能只按绝对毫米 EPE 排名。

旧 Stage4 组与 object-v2 组的数据字段、采样方式、缩放、validation stride 聚合均不同。旧组的 `val/mean_stride_epe_mm` 聚合 stride 1--10，object-v2 主表只聚合 stride 1/5/10；因此两个大组之间不能按 mean EPE 直接排名。旧组内部两对 gate/no-gate 使用相同数据合同，可以进行直接对照。

此外，上述三条 object-v2 长程 run 都从旧的 `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2` 读取 GRAB 数据。该 cache 的 GRAB hand geometry 受 subject-template 路径错误影响，可能回退到平均 MANO。修复版 GRAB cache 已生成，但尚无对应的正式 mixed / GRAB-only 长训。因此这三条结果只能作为旧数据链路诊断和后续重训 baseline。

## 实验变量

| 变量 | Mixed GRAB+ARCTIC | GRAB gate+cm64 | GRAB gate+cm64 warm-up |
| --- | ---: | ---: | ---: |
| 数据范围 | GRAB + ARCTIC | GRAB | GRAB |
| cache 版本 | 旧 mixed object-v2 | 旧 GRAB object-v2 | 旧 GRAB object-v2 |
| `cm_dim` | 256 | 64 | 64 |
| Cm slot 数 | 16 | 16 | 16 |
| time condition | 是 | 是 | 是 |
| slot gate | 否 | 是 | 是 |
| gate threshold | 不适用 | 0.85 | 0.85（epoch 6--10 ramp） |
| gate warm-up | 否 | 否 | 前 5 epoch 全 slot，后 5 epoch ramp |
| slot count loss weight | 0.001，但 no-gate 下不形成稀疏 routing | 0.001 | 目标 0.001，随 gate ramp |
| flow RMS calibration | 0.0732755 m | 0.0931932 m | 0.0931932 m |
| target scale | 13.6471 | 10.7304 | 10.7304 |
| train seed | 42 | 42 | 42 |
| per-device batch | 48 | 48 | 48 |
| GPU / global batch | 3 / 144 | 2 / 96 | 2 / 96 |
| learning rate | 3e-4 cosine | 3e-4 cosine | 3e-4 cosine |
| weight decay | 1e-4 | 1e-4 | 1e-4 |
| grad clip norm | 1.0 | 1.0 | 1.0 |
| AMP | 关闭 | 关闭 | 关闭 |
| 预算 | 50 epoch / 184650 step | 50 epoch / 134900 step | 50 epoch / 约 134900 step |
| 快照进度 | epoch 38，step 136500 | epoch 50，step 134900，完成 | epoch 31，step 83638 |

## 最佳验证结果

主指标均来自各自 `best.pt` 对应的 validation epoch。EPE 单位为 mm，relative EPE 与 zero-flow improvement 为无量纲。

| 指标 | Mixed GRAB+ARCTIC | GRAB gate+cm64 | GRAB gate+cm64 warm-up |
| --- | ---: | ---: | ---: |
| best epoch / step | 37 / 135562 | 48 / 129504 | 29 / 78242 |
| train EPE（最近完整 epoch） | 10.723 | 11.351 | 11.919 |
| val mean-stride EPE ↓ | **11.853** | 12.855 | 13.363 |
| val relative EPE ↓ | 0.3526 | **0.2637** | 0.2826 |
| zero-flow improvement ↑ | 64.74% | **73.63%** | 71.74% |
| prediction/GT norm ratio | 0.864 | 0.925 | 0.920 |
| stride-1 EPE ↓ | **2.963** | 3.105 | 3.526 |
| stride-5 EPE ↓ | **11.126** | 11.961 | 12.412 |
| stride-10 EPE ↓ | **21.469** | 23.498 | 24.151 |
| stride-5 hard active slots | 16.000（no-gate） | 1.021 | 1.079 |
| stride-5 effective branches | 12.710 | 1.014 | 1.075 |
| stride-5 global top-1 usage | 0.127 | 0.990 | 0.967 |
| stride-5 action shuffle EPE | 23.936 | 24.920 | 25.162 |
| stride-5 action reverse EPE | 65.977 | 97.632 | 97.121 |

## 旧 Stage4 GRAB、C=256 长训变量

| 变量 | 原始缩放 no-gate | 原始缩放 gate | 校准缩放 no-gate | 单手筛选 no-gate | 单手筛选 gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 数据 root | 旧 Stage4 GRAB | 旧 Stage4 GRAB | 旧 Stage4 GRAB | 旧 Stage4 GRAB + manifest | 旧 Stage4 GRAB + manifest |
| `cm_dim` / slots | 256 / 16 | 256 / 16 | 256 / 16 | 256 / 16 | 256 / 16 |
| time condition | 是 | 是 | 是 | 是 | 是 |
| slot gate | 否 | 是 | 否 | 否 | 是 |
| gate regularization | 无 | count/confidence 权重均为 0 | 无 | 无 | count=0.001，confidence=0.1 |
| geometry scale | 100 | 100 | 15 | 15 | 15 |
| hand-flow scale | 100 | 100 | 17.7196 | 17.7196 | 17.7196 |
| object target scale | 100 | 100 | 10.7892 | 10.7892 | 10.7892 |
| grad clip norm | 1 | 1 | 5 | 5 | 5 |
| train seed | 42 | 42 | 42 | 42 | 42 |
| GPU / per-device batch / global batch | 2 / 8 / 16 | 2 / 8 / 16 | 2 / 8 / 16 | 2 / 8 / 16 | 2 / 8 / 16 |
| validation strides | 1--10 | 1--10 | 1--10 | 1--10 | 1--10 |
| 预算 | 20 epoch / 328920 step | 20 epoch / 328920 step | 20 epoch / 328920 step | 20 epoch / 226780 step | 20 epoch / 226780 step |

## 旧 Stage4 GRAB、C=256 最佳验证结果

五条 run 的 best 都出现在 epoch 20，即最后一次 validation。

| 指标 | 原始缩放 no-gate | 原始缩放 gate | 校准缩放 no-gate | 单手筛选 no-gate | 单手筛选 gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| best step | 328920 | 328920 | 328920 | 226780 | 226780 |
| train EPE | 9.952 | 10.137 | 10.099 | **8.309** | 9.718 |
| val mean-stride EPE ↓ | 12.366 | 12.399 | 12.508 | **10.398** | 10.954 |
| val relative EPE ↓ | 0.2453 | 0.2462 | 0.2491 | **0.1902** | 0.2002 |
| zero-flow improvement ↑ | 75.47% | 75.38% | 75.09% | **80.98%** | 79.98% |
| prediction/GT norm ratio | 0.929 | 0.932 | 0.930 | 0.964 | 0.967 |
| stride-1 EPE ↓ | 2.962 | 2.976 | 3.063 | **2.637** | 2.795 |
| stride-5 EPE ↓ | 11.418 | 11.509 | 11.574 | **9.508** | 9.909 |
| stride-10 EPE ↓ | 21.729 | 21.910 | 21.808 | **18.451** | 19.750 |
| stride-5 hard active slots | 16.000 | 2.227 | 16.000 | 16.000 | 1.000 |
| stride-5 effective branches | 10.040 | 2.129 | 12.343 | 12.830 | 1.000 |
| stride-5 global top-1 usage | 0.167 | 0.412 | 0.159 | 0.143 | 1.000 |
| grad-clipped fraction（末 epoch） | 100.0% | 99.98% | 2.15% | 2.05% | 38.86% |

### 旧数据 gate/no-gate 直接对照

| 对照 | EPE 变化（gate - no-gate） | routing 变化 | 结论 |
| --- | ---: | --- | --- |
| 原始缩放、全量 GRAB | +0.0324 mm（+0.26%） | stride-5 effective branches 10.040 → 2.129 | gate 显著稀疏化，但效果与 no-gate 基本持平且略差 |
| 校准缩放、单手筛选 GRAB | +0.5563 mm（+5.35%） | stride-5 effective branches 12.830 → 1.000 | gate 发生单 slot collapse，并显著弱于同数据 no-gate |

原始缩放两条 run 的末 epoch 几乎每步都触发 `grad_clip_norm=1`，说明其尺度设置导致强烈 clipping。校准缩放后 no-gate 的 clipping fraction 降至约 2%，所以“原始缩放 vs 校准缩放”的小幅 EPE 差异不能只看最终 EPE，还应结合优化条件解释。

### 对结果的解释

| 比较问题 | 观察 | 判断 |
| --- | --- | --- |
| Mixed 是否在收敛 | val EPE 从 epoch 1 的 29.416 mm 降至 epoch 37 的 11.853 mm；最近 10 epoch 仍以约 0.044 mm/epoch 缓慢下降 | 已进入收敛尾段，尚未严格平台化 |
| 无 warm-up gate 是否收敛 | epoch 48 best 为 12.8546 mm，epoch 50 为 12.8585 mm | 已平台化，继续同 schedule 收益很小 |
| warm-up 是否改善 GRAB gate EPE | 当前 best 13.363 mm，高于无 warm-up 的 12.855 mm | 截至 epoch 31 未显示改善，但训练尚未完成 |
| warm-up 是否避免 slot collapse | warm-up run 在 stride 5 仅约 1.08 个 effective branch，top-1 usage 约 0.967 | 没有；warm-up 后仍退化为近单 slot routing |
| 模型是否使用 hand action | 三条 run 的 shuffle/reverse EPE 均显著高于正常 EPE | 有明显 action dependence，不像 zero-flow 或忽略 action 的退化解 |
| mixed 与 GRAB gate 谁更好 | mixed 的绝对 EPE 更低，但 mixed 的 GT 尺度和数据组成不同；GRAB gate 的 relative EPE 更低 | 不能直接据此判定 gate、C=64 或 mixed data 的因果优劣 |

## 数据集分项：Mixed best

Mixed run 在 best epoch 37 的分数据集结果如下。这里是相同 mixed 模型、相同 checkpoint 内的分项，可以用于观察两套数据是否有一侧明显失效。

| Dataset | stride 1 EPE | stride 5 EPE | stride 10 EPE | 三个 stride 简单平均 |
| --- | ---: | ---: | ---: | ---: |
| GRAB | 2.892 | 10.673 | 20.623 | 11.396 |
| ARCTIC | 3.035 | 11.588 | 22.330 | 12.318 |

两套数据都产生稳定的非零预测，ARCTIC 的简单平均 EPE 比 GRAB 高约 0.92 mm；没有出现某一数据集完全未学习的迹象。该表仍受旧 GRAB cache 语义问题影响。

## 纳入主比较的路径

### Mixed GRAB+ARCTIC，no-gate，C=256

| 类型 | 绝对路径 |
| --- | --- |
| 配置 | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/active/object_v2_grab_arctic.yaml` |
| cache root | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2` |
| split | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2/splits_seed42/splits.json` |
| run 目录 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_arctic_20260819_165241` |
| 完整配置快照 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_arctic_20260819_165241/config.json` |
| metrics | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_arctic_20260819_165241/metrics.jsonl` |
| best checkpoint | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt` |
| latest checkpoint | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/latest.pt` |
| stdout log | `/home2/wyy/oyx_ws/Ref2Dex/output/exp/cm_v121/cm_v121_mixed_3gpu_bs48_50ep_resume_20260819_195242.log` |

### GRAB-only，gate，C=64，无 warm-up

| 类型 | 绝对路径 |
| --- | --- |
| 配置 | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/active/object_v2_grab_gate_cm64.yaml` |
| cache root | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2/grab` |
| split | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2/grab/splits_seed42/splits.json` |
| run 目录 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_20260819_234926` |
| 完整配置快照 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_20260819_234926/config.json` |
| metrics | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_20260819_234926/metrics.jsonl` |
| best checkpoint | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_20260819_234926/checkpoints/best.pt` |
| latest checkpoint | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_20260819_234926/checkpoints/latest.pt` |
| stdout log（初始） | `/home2/wyy/oyx_ws/Ref2Dex/output/exp/cm_v121/cm_v121_grab_gate_cm64_2gpu_bs48_50ep_20260819_234923.log` |
| stdout log（迁移续训） | `/home2/wyy/oyx_ws/Ref2Dex/output/exp/cm_v121/cm_v121_grab_gate_cm64_2gpu_bs48_50ep_gpu67_resume_20260820_090438.log` |

### GRAB-only，gate，C=64，有 warm-up

| 类型 | 绝对路径 |
| --- | --- |
| 配置 | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/active/object_v2_grab_gate_cm64_warmup.yaml` |
| cache root | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2/grab` |
| split | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2/grab/splits_seed42/splits.json` |
| run 目录 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_warmup_20260820_111406` |
| 完整配置快照 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_warmup_20260820_111406/config.json` |
| metrics | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_warmup_20260820_111406/metrics.jsonl` |
| best checkpoint | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_warmup_20260820_111406/checkpoints/best.pt` |
| latest checkpoint | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_object_v2_grab_gate_cm64_warmup_20260820_111406/checkpoints/latest.pt` |
| stdout log | `/home2/wyy/oyx_ws/Ref2Dex/output/exp/cm_v121/cm_v121_grab_gate_cm64_warmup_2gpu_bs48_50ep_gpu01_20260820_111403.log` |

## 旧 Stage4 GRAB、C=256 路径

| Run | 配置 | run / metrics / checkpoint |
| --- | --- | --- |
| 原始缩放 no-gate | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/archive/sequence_full_grab_16slot_20e_fresh.yaml` | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_20e_fresh_20260805_203822`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_20e_fresh_20260805_203822/metrics.jsonl`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_20e_fresh_20260805_203822/checkpoints/best.pt` |
| 原始缩放 gate | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/archive/sequence_full_grab_16slot_gate_20e_fresh.yaml` | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_gate_20e_fresh_20260805_234939`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_gate_20e_fresh_20260805_234939/metrics.jsonl`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_gate_20e_fresh_20260805_234939/checkpoints/best.pt` |
| 校准缩放 no-gate | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/archive/sequence_full_grab_16slot_calibrated_scale_clip5_20e.yaml` | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_20e_20260806_220346`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_20e_20260806_220346/metrics.jsonl`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_20e_20260806_220346/checkpoints/best.pt` |
| 单手筛选 no-gate | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/archive/sequence_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_20e.yaml` | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_20e_20260807_005752`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_20e_20260807_005752/metrics.jsonl`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_20e_20260807_005752/checkpoints/best.pt` |
| 单手筛选 gate | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/configs/archive/sequence_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_gate_regularized_20e.yaml` | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_gate_regularized_20e_20260807_142936`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_gate_regularized_20e_20260807_142936/metrics.jsonl`<br>`/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_clip5_single_hand_v1_gate_regularized_20e_20260807_142936/checkpoints/best.pt` |

| 公共输入 | 绝对路径 |
| --- | --- |
| 旧 Stage4 root | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/stage4/data` |
| 固定 split | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/stage4/splits/full_grab_v1/split.json` |
| dominant-hand manifest | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/stage4/dominant_hand_full_v1/dominant_hand.jsonl` |

## 未纳入主比较的结果

| 结果 | 实际预算 | 排除理由 | 路径 |
| --- | ---: | --- | --- |
| mixed seed 42/43/44 | 各 300 step | 仅验证链路，cosine LR 已在极短预算内衰减 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_mixed_seed{42,43,44}_20260819_094543` |
| GRAB-only seed 42/43/44 | 各 300 step | 不足一个有代表性的长程训练阶段 | 见下方精确路径 |
| ARCTIC-only seed 42/43/44 | 各 300 step | 不足一个有代表性的长程训练阶段 | 见下方精确路径 |
| mixed batch-24 full pilot | 10000 step，约 2 epoch | 用户已确认它只是 pilot，且已被当前 50-epoch mixed run 覆盖 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_mixed_bs24_3gpu_full_20260819_131714` |
| mixed batch sweep | 各 80 step | 只用于吞吐选型，不是效果实验 | `/home2/wyy/oyx_ws/Ref2Dex/output/exp/cm_v121_throughput/mixed_bs*_3gpu_trainonly.log` |
| 旧 Stage4 初始 no-gate run | 15 个完整 epoch | 低于 20-epoch 门槛，且已被 fresh 20-epoch no-gate 覆盖 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_20260805_000407` |
| 旧 calibrated-scale 中断 run | 11100 step / epoch 1 | 没有 validation，已被 clip5 20-epoch run 覆盖 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_16slot_calibrated_scale_20e_20260806_213650` |
| Scene Cache V1.1 full run | 10 epoch | 低于 20-epoch 门槛，且属于另一数据合同 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_full_grab_scene_v1_1_16slot_10epoch_20260818_001836` |

由于 GRAB-only 和 ARCTIC-only 的 no-gate + C=256 路线目前只有 300-step 结果，本文不为它们填写正式效果排名。当前仍缺少基于修复版 cache、相同模型和相同 epoch 预算的 mixed / GRAB-only / ARCTIC-only 三方公平对照。

### 300-step run 精确路径

| Dataset | Seed | 绝对路径 |
| --- | ---: | --- |
| mixed | 42 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_mixed_seed42_20260819_094543` |
| mixed | 43 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_mixed_seed43_20260819_094543` |
| mixed | 44 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_mixed_seed44_20260819_094543` |
| GRAB | 42 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_grab_seed42_20260819_103530` |
| GRAB | 43 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_grab_seed43_20260819_103530` |
| GRAB | 44 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_grab_seed44_20260819_114552` |
| ARCTIC | 42 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_arctic_seed42_20260819_103530` |
| ARCTIC | 43 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_arctic_seed43_20260819_114552` |
| ARCTIC | 44 | `/home2/wyy/oyx_ws/Ref2Dex/outputs/cm/cm_v121_arctic_seed44_20260819_114552` |

## 修复版 cache 与后续正式实验路径

| 类型 | 绝对路径 |
| --- | --- |
| 修复版 GRAB cache | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2_subject_template_20260820/grab` |
| 修复版 GRAB split | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2_subject_template_20260820/grab/splits_seed42/splits.json` |
| 修复版 GRAB statistics | `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/cm_object_v2_subject_template_20260820/grab/object_v2_statistics.json` |
| 当前实验事实源 | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/docs/logs/experiment_log.md` |
| cache 修复记录 | `/home2/wyy/oyx_ws/Ref2Dex/src/task/Cm/docs/logs/modification_log.md` |

下一轮正式比较需要先把修复版 GRAB 与现有 ARCTIC 组成新的联合 root，并重新生成联合 fixed split 与 train-only calibration。不能只替换目录后继续旧 checkpoint，否则数据几何、candidate mask 和 target scale 语义仍不一致。
