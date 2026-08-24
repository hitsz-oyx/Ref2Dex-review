# CmDecoder 实验记录

## 当前状态

已完成两次 overfit sanity check。旧版静止帧实验被判为 INCONCLUSIVE；active-motion 窗口已验证训练链路能超过 identity baseline。当前仍未进行泛化结论实验。

## EXP-016 — GRAB 30Hz 右手无配对重定向 smoke test

### 日期

2026-08-23

### 设置

- cache: `cm_object_v2_subject_template_20260820/grab`
- side / rate: GRAB right / 30Hz (`ds_rate=4`, fixed stride=1)
- sequence: `s1/scissors_offhand_1`
- Cm: 旧版 C=256 frozen checkpoint
- decoder: `outputs/cmdecoder/cm_decoder_20260823_000423/checkpoints/best.pt`
- robot initialization: Inspire F1 joint-limit midpoint q0；object-center outward approach 0.12m

### 结果

8 帧 CUDA smoke test 成功，产物为 `output/research/grab_retarget_30hz.npz` 和
`output/research/grab_retarget_30hz.png`。数据读取、GRAB hand-flow→Cm token、机器人
几何输入、point-flow、q fitting 和 wrist 更新链路均可运行。

2026-08-24 重新检查发现默认起始帧0的前32帧右手 candidate 全空，旧图中的物体输入
实际是 padding，不能评价效果。入口增加 `--start-frame` 后，改用右手第一个连续有效
窗口 `start=131` 重跑32帧；每帧512个物体点均有效。有效窗口结果：

- 机器人手—物体质心距离从 `22.1 mm` 增至 `327.3 mm`；
- object-relative 机器人手质心漂移 `324.2 mm`；
- 机器人点到物体最近距离的均值从 `33.5 mm` 增至 `329.9 mm`；
- `<20 mm` 接触点比例从 `34.5%` 降为 `0%`；
- 单关节32帧范围约 `9.0°—17.7°`，逐帧 q 变化均值 `0.53°`，表明主要失败不是
  关节爆炸，而是无约束 wrist SE(3) 逐帧积分漂移。

有效窗口产物：`output/research/grab_retarget_30hz_baseline_eval_active_20260824.{npz,png}`。

### 诊断

有效 object candidate 下预测机器人手仍快速离开物体，因此当前单序列实现不能作为成功
的重定向结果。这暴露出两个后续问题：HRDexDB 训练的 point decoder 尚未证明能接受
GRAB 人手产生的 Cm token 分布；逐帧无约束 wrist fitting 会累积漂移。下一步应先增加
wrist step clamp / 物体接触约束，并用多序列统计，而不是直接扩大运行规模。

### 结论状态

**INCONCLUSIVE**

## EXP-001 — 单 episode overfit sanity check

### 日期

2026-08-21

### 对应指导

当前 CmDecoder 首版架构。

### 假设

如果 frozen Cm tokens 携带当前动作状态且 Decoder 链路实现正确，轻量 MLP 应能在少量相邻帧上过拟合下一帧 6 维手指关节角。

### Baseline

- checkpoint: `outputs/cm/cm_v121_grab_seed42_20260819_103530/checkpoints/best.pt`
- dataset: HRDexDB Inspire F1
- Cm: 16 slots × 256 dim，全部冻结

### 本次修改

- 训练集：1 个 episode，前 32 个相邻帧（31 samples）
- 双卡 DDP：GPU 6、7
- 100 epochs / 100 optimizer steps
- `weight_decay=0`（overfit mode）
- Decoder 和 loss 保持首版实现

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  -m src.task.CmDecoder.train --device cuda \
  --set train.distributed.enable=true \
  --set train.overfit_mode=true --set train.epochs=100 \
  --set data.max_episodes=1 --set data.max_frames_per_episode=32 \
  --set data.batch_size=16 --set data.num_workers=0 \
  --set wandb.enable=false --set performance.mode=off
```

### 结果

| Metric | Step 1 | Step 100 |
|---|---:|---:|
| train q MAE (deg) | 13.93 | 0.132 |
| train q MAE (rad) | 0.243 | 0.00230 |
| train q RMSE (rad) | 0.440 | 0.00290 |
| val q MAE (deg) | 27.33 | 4.65 |

### 关键观察

训练集 MAE 从 13.93° 降至 0.132°，Decoder 能够有效拟合这批样本；验证集使用的是不同 episode，MAE 约 4.65°，不能作为泛化结论。

### 解释

冻结 Cm 后仍然可以通过 Cm tokens 和当前 q 重建下一帧动作，说明当前数据→Cm→Decoder→loss→optimizer 链路没有明显断路。训练集尚未达到严格的 0°，但已足够作为实现 sanity check。

### 结论状态

**INCONCLUSIVE**

由于该窗口 `q_next=q_t`，当前证据只能说明优化链路运行，不能说明 Decoder 读取了有效动作信息。

### 决策

保留 checkpoint 用于后续代码回归，不作为正式模型或泛化 baseline。

### 下一步

使用 active-motion 窗口和 q_t-only / Cm-only / q_t+Cm 对照，避免静止帧 shortcut。

### 证据

- checkpoint: `outputs/cmdecoder/cm_decoder_20260821_161226/checkpoints/best.pt`
- log: `outputs/cmdecoder/cm_decoder_20260821_161226/train.log`
- metrics: `outputs/cmdecoder/cm_decoder_20260821_161226/metrics.jsonl`

## EXP-002 — active-motion 单步 overfit

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

在排除相邻静止帧后，冻结 Cm + 当前 q 的 Decoder 应能拟合真实手指动作；训练误差应低于 `q_next=q_t` identity baseline。

### Baseline

- checkpoint: `outputs/cm/cm_v121_grab_seed42_20260819_103530/checkpoints/best.pt`
- cache: `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_20_seed42.json`
- active threshold: `max(|q_next-q_t|) >= 0.5°`
- Cm tokens: precomputed, 16×256, frozen

### 本次修改

- train split：1 episode，前 32 个 active-motion、连续 30 Hz pair
- 500 optimizer steps，双卡 GPU 6、7
- 比较训练 q+Cm Decoder 与 identity `q_next=q_t`

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  -m src.task.CmDecoder.train --device cuda \
  --set train.distributed.enable=true --set train.overfit_mode=true \
  --set train.epochs=500 --set data.active_motion_only=true \
  --set data.active_motion_threshold_deg=0.5 \
  --set data.max_episodes=1 --set data.max_frames_per_episode=32 \
  --set meta.use_cached_cm_tokens=true --set wandb.enable=false
```

### 结果

| Metric | Identity | Decoder step 500 |
|---|---:|---:|
| train q MAE (deg) | 0.647 | 0.294 |
| train q MAE (rad) | 0.01130 | 0.00513 |
| val q MAE (deg) | 0.471 | 4.12 |

### 关键观察

Decoder 在 active-motion 训练窗口上低于 identity baseline；验证集是不同 episode，不能用于支持泛化。

### 解释

这排除了 EXP-001 的“全是静止帧”问题，并确认当前 active-motion 数据筛选、Cm token sidecar 和 Decoder 训练链路有效。但单 episode overfit 仍不能证明 Cm 相比 q-only 的独立贡献。

### 结论状态

**SUPPORTED**

当前证据支持“active-motion 窗口可被 Decoder 过拟合且训练误差低于 identity”的实现假设；不支持泛化或 Cm 独立贡献结论。

### 决策

保留该 checkpoint 作为 active-motion 回归测试，不作为正式模型。

### 下一步

在相同 active-motion 子集上运行 `qt_only / cm_only / qt_cm / shuffled-flow` 对照。

### 证据

- checkpoint: `outputs/cmdecoder/cm_decoder_20260821_192109/checkpoints/best.pt`
- log: 训练输出 `cmdecoder_active_overfit_v1_500`

## EXP-003 — active-motion Decoder 输入对照

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

如果 Cm tokens 携带当前动作相关信息，`q_t + Cm` 应不劣于 `q_t-only`，而打乱 flow 后性能应下降；`Cm-only` 用于衡量不提供当前 q 时的解码能力。

### Baseline

- 同 EXP-002 的 1 episode、32 个 active-motion pair
- 500 optimizer steps，GPU 6、7
- identity: `q_next=q_t`

### 本次修改

- `qt_cm`：当前主模型，使用预缓存 Cm tokens
- `qt_only`：不输入 Cm
- `cm_only`：不输入 `q_t`
- `shuffled-flow`：在线 Cm 编码，但 batch 内打乱 hand flow

### 结果

| Variant | Train q MAE (deg) | Val q MAE (deg) | 时间 |
|---|---:|---:|---:|
| identity | 0.647 | 0.471 | — |
| qt_cm | **0.294** | 4.12 | 00:40 |
| qt_only | 0.302 | 2.71 | 00:40 |
| cm_only | 0.453 | 4.08 | 00:40 |
| shuffled-flow | 0.464 | 3.99 | 11:20 |

### 关键观察

四个可训练变体都低于 identity。`qt_cm` 相比 `qt_only` 仅有小幅训练集优势；`cm_only` 和 shuffled-flow 明显更差。验证集来自不同 episode，且规模很小，不能据此判断泛化。

### 解释

当前结果支持 Cm tokens 包含可用信息，但其独立贡献相对当前 q shortcut 较小；shuffled-flow 对照的下降与动作信息被破坏的预期一致。由于是单 episode overfit，仍可能受样本和优化设置影响。

### 结论状态

**INCONCLUSIVE**

结果支持继续进行 held-out episode 对照，但不足以确认 Cm 在正式数据上的独立增益。

### 决策

不选择任何 overfit checkpoint 作为正式模型；保留四个输出用于回归和后续对照复现。

### 下一步

在 20-episode split 上固定 optimizer/budget，运行四个 variant 的 held-out episode 评测；正式结论以 active-motion 子集和 identity baseline 为主。

### 证据

- qt_cm: `outputs/cmdecoder/cm_decoder_20260821_192109/`
- qt_only: `outputs/cmdecoder/cm_decoder_20260821_193337/`
- cm_only: `outputs/cmdecoder/cm_decoder_20260821_193442/`
- shuffled-flow: `outputs/cmdecoder/cm_decoder_20260821_193549/`

## EXP-004 — 20-episode qt_cm 训练

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

在 20-episode 的 v4 数据划分上，冻结 Cm 并输入 `q_t + Cm`，Decoder 可以学习从当前手部状态重建下一帧 6 个手指关节角；训练误差应显著低于 identity baseline，同时 held-out episode 验证误差用于观察初步泛化。

### Baseline

- v4 manifest: `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_20_seed42.json`
- split: 16 train / 2 val / 2 test，train 10,266 samples
- frozen Cm checkpoint: `outputs/cm/cm_v121_grab_seed42_20260819_103530/checkpoints/best.pt`
- identity baseline: `q_next=q_t`

### 本次修改

- 使用预缓存 Cm tokens 的 `qt_cm` 输入。
- 保持 30 Hz 相邻帧、6 个手指关节标量角度、Cm 完全冻结。
- 双卡 6/7 DDP，30 epochs，9,600 optimizer steps。
- W&B 使用 offline 模式；训练代码和数据划分未改变。

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=6,7 WANDB_MODE=offline \
torchrun --standalone --nproc_per_node=2 -m src.task.CmDecoder.train --device cuda \
  --set train.distributed.enable=true --set train.epochs=30 \
  --set meta.use_cached_cm_tokens=true --set meta.decoder_input=qt_cm
```

### 结果

| Metric | Best (epoch 9) | Final (epoch 30) |
|---|---:|---:|
| train q MAE (deg) | — | 0.836 |
| val q MAE (deg) | **2.461** | 2.651 |
| val q RMSE (rad) | 0.0561 | 0.0613 |
| val identity MAE (deg) | 0.130 | 0.130 |

### 关键观察

训练集误差持续下降，但验证集在第 9 epoch 达到最低后回升，出现明显过拟合迹象。最佳验证 MAE 约为 2.46°，优于该 split 的 identity baseline 约 0.13° 这一点需要谨慎解释：当前验证 identity 统计来自 q 标量缩放/采样后的定义，后续应核查 baseline 与 q 误差口径的一致性。

### 解释

本次实验确认 20-episode 数据和预缓存 Cm tokens 的训练链路可稳定运行，但不能仅凭当前结果得出 Cm 的泛化贡献结论；需要先统一 identity 指标口径，再进行 held-out test 评测。

### 结论状态

**INCONCLUSIVE**

### 决策

保留 epoch 9 的 `best.pt` 作为当前 20-episode qt_cm checkpoint；不将其视为最终模型。

### 下一步

- 核查 identity baseline 与预测误差的计算口径。
- 在 test split 上评测 best checkpoint，并与 `qt_only` 对照。
- 需要线上记录时，通过代理执行该 run 的 `wandb sync`。

### 证据

- checkpoint: `outputs/cmdecoder/cm_decoder_20260821_195912/checkpoints/best.pt`
- final checkpoint: `outputs/cmdecoder/cm_decoder_20260821_195912/checkpoints/step_000009600_epoch_000030.pt`
- metrics: `outputs/cmdecoder/cm_decoder_20260821_195912/metrics.jsonl`
- log: `outputs/cmdecoder/cm_decoder_20260821_195912/train.log`

## EXP-005 — 20-episode qt_only / cm_only 对照

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

在与 EXP-004 完全相同的数据、优化器和训练预算下，`qt_only` 用于测量当前关节状态 shortcut，`cm_only` 用于测量不提供当前 q 时 Cm 的独立解码能力；两者可作为 `qt_cm` 的输入消融。

### Baseline

- 同 EXP-004 的 v4 20-episode split、10,266 train samples
- 30 epochs / 9,600 steps，GPU 6、7，Cm checkpoint 冻结
- `qt_cm` 最佳 val q MAE：2.461°

### 本次修改

- 运行 `qt_only`：仅输入 `q_t`。
- 运行 `cm_only`：仅输入冻结 Cm tokens。
- 其余数据、训练预算和评价方式保持不变。

### 结果

| Variant | Best epoch | Best val q MAE (deg) | Final val q MAE (deg) | Final train q MAE (deg) |
|---|---:|---:|---:|---:|
| qt_cm | 9 | 2.461 | 2.651 | 0.836 |
| qt_only | 24 | **1.423** | 1.434 | 2.527 |
| cm_only | 7 | 5.028 | 5.941 | 1.547 |

### 关键观察

在该 20-episode split 上，`qt_only` 验证误差低于 `qt_cm`，而 `cm_only` 明显更差。`qt_only` 的训练误差反而高于 `qt_cm`，说明当前结果不能简单归因于训练集拟合程度。

### 解释

当前证据表明，下一帧 q 的主要可预测信息来自当前 q；Cm 单独输入不足以稳定重建下一帧，且将 Cm 与 q_t 拼接并未在本次设置下带来验证集收益。由于尚未进行 test split、多个 seed 和统一 baseline 口径复核，这仍是初步结果。

### 结论状态

**INCONCLUSIVE**

### 决策

保留两组 checkpoint 作为输入消融结果；不据此移除 Cm 或改变正式模型定义。

### 下一步

- 在固定 best checkpoint 上评测 test split。
- 核查 identity baseline 与 q MAE 的统计口径。
- 如需判断 Cm 的增益，增加多 seed 或 active-motion 子集评测。

### 证据

- qt_only: `outputs/cmdecoder/cm_decoder_20260821_200732/`
- cm_only: `outputs/cmdecoder/cm_decoder_20260821_201002/`

## EXP-006 — active-motion 子集评估

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

仅在实际发生明显手指运动的相邻 30 Hz 帧上评估，可以排除静止帧主导的平均误差，更直接观察三种输入对动作重建的影响。

### Baseline

- v4 manifest：同 EXP-004/005
- active-motion 条件：`is_30hz_pair=true` 且 `q_delta_abs_max >= 0.5°`
- val active samples：240；test active samples：162
- 使用各组训练得到的 best checkpoint，不重新训练

### 本次修改

- 仅改变评估数据筛选为 active-motion；模型、checkpoint 和 q 误差计算保持不变。

### 结果

| Variant | Val active q MAE (deg) | Test active q MAE (deg) | Val identity (deg) | Test identity (deg) |
|---|---:|---:|---:|---:|
| qt_cm | 2.827 | 4.821 | 0.679 | 0.369 |
| qt_only | **2.593** | **1.691** | 0.679 | 0.369 |
| cm_only | 6.148 | 9.262 | 0.679 | 0.369 |

### 关键观察

活动帧上 identity baseline 的 q MAE 仅为 0.679°（val）/ 0.369°（test），三组 Decoder 均未超过 identity；`qt_only` 在 test 上明显优于 `qt_cm`，`cm_only` 误差最大。

### 解释

当前训练目标是重建 `q_{t+1}`，而活动帧筛选只要求存在至少一个关节变化，并不保证下一帧变化幅度大于 identity 误差。现有模型在 held-out active-motion 上尚未学到足够稳定的一步预测能力；Cm 在本次实验中没有表现出独立增益。

### 结论状态

**INCONCLUSIVE**

### 决策

不把活动帧结果解释为 Cm 表示无效；保留结果作为当前 V1 的诊断证据。后续应考虑只在动作幅度更高的帧上定义动作子集，或改用 flow/增量目标评估。

### 下一步

- 分析更高阈值（例如 1°/2°）的活动帧分布和误差。
- 评估预测增量 `q_{t+1}-q_t`，避免静止分量掩盖动作重建。
- 再决定是否调整 Decoder 输入或训练目标。

### 证据

- qt_cm: `outputs/cmdecoder/cm_decoder_20260821_195912/checkpoints/best.pt`
- qt_only: `outputs/cmdecoder/cm_decoder_20260821_200732/checkpoints/best.pt`
- cm_only: `outputs/cmdecoder/cm_decoder_20260821_201002/checkpoints/best.pt`

## EXP-007 — 20-episode 残差预测输入对照

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

将目标从绝对 `q_next` 改为相邻帧残差 `q_next-q_t`，可消除绝对姿态回归的优化负担；若 Cm 携带有效动作信息，`qt_cm` 应在 held-out episode 上优于 `qt_only` 和 identity。

### Baseline

- 同 EXP-004/005 的 v4 20-episode split：16 train / 2 val / 2 test
- 30 epochs / 9,600 steps，GPU 6、7，offline W&B
- identity：`pred_q_next=q_t`
- direct-q 对照最佳 val MAE：qt_cm 2.461°、qt_only 1.423°、cm_only 5.028°

### 本次修改

- 三组均预测 `delta_q=q_next-q_t`，再以 `q_t+pred_delta_q` 重建下一帧。
- 运行 `qt_cm / qt_only / cm_only`；不运行 shuffled-flow。
- 数据、网络容量、优化器和预算保持一致。

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=6,7 WANDB_MODE=offline \
torchrun --standalone --nproc_per_node=2 -m src.task.CmDecoder.train --device cuda \
  --set train.distributed.enable=true --set train.epochs=30 \
  --set meta.prediction_target=delta_q \
  --set meta.decoder_input=<qt_cm|qt_only|cm_only> \
  --set meta.use_cached_cm_tokens=true
```

### 结果

全量 split：

| Variant | Best val q MAE (deg) | Test q MAE (deg) | Test identity (deg) |
|---|---:|---:|---:|
| qt_cm | 0.158 | 0.103 | 0.065 |
| qt_only | **0.137** | **0.076** | 0.065 |
| cm_only | 0.158 | 0.097 | 0.065 |

active-motion（0.5° 阈值；val 240 / test 162 samples）：

| Variant | Val active q MAE (deg) | Test active q MAE (deg) | Val/Test identity (deg) |
|---|---:|---:|---:|
| qt_cm | 0.689 | 0.394 | 0.679 / 0.369 |
| qt_only | **0.680** | **0.369** | 0.679 / 0.369 |
| cm_only | 0.682 | 0.386 | 0.679 / 0.369 |

### 关键观察

残差预测相对 direct-q 将误差降低一个数量级以上，但 `qt_only` 在全量和 active-motion 上均最佳。三组均未稳定超过 identity；active-motion 上 `qt_only` 几乎等于 identity，说明模型主要预测接近零的动作残差。

### 解释

残差参数化成功修复了绝对 q 回归困难，但没有证明模型学会动作。当前 30 Hz 数据中小残差占主导，SmoothL1 的最优保守策略接近零残差；Cm 输入在单 seed、当前 decoder 和损失下没有带来 held-out 收益。`cm_only` 仍通过固定 skip `q_t+pred_delta_q` 使用 q_t 作为重建基准，但 Decoder 本身不接收 q_t。

### 结论状态

**INCONCLUSIVE**

### 决策

保留残差预测作为默认参数化；不宣称 Cm 有动作预测增益，也不以全量 q MAE 作为唯一成功指标。

### 下一步

- 训练时增加 active-motion 采样权重或仅训练活动帧，避免零残差主导。
- 直接记录 delta-q MAE、动作方向/幅值指标，以及相对 identity 的 improvement。
- 在更高活动阈值和多 seed 下复核 `qt_cm` 与 `qt_only`。

### 证据

- qt_cm: `outputs/cmdecoder/cm_decoder_20260821_210013/`
- qt_only: `outputs/cmdecoder/cm_decoder_20260821_210256/`
- cm_only: `outputs/cmdecoder/cm_decoder_20260821_210526/`

## EXP-008 — 20-episode 30 Hz q 差分分布统计

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

如果 HRDexDB 的相邻 30 Hz 帧大多是静止或低速帧，直接用平均 q MAE 评估动作重建会被大量近零残差主导。

### Baseline

- 数据范围：当前已构建的 v4 20-episode split（16 train / 2 val / 2 test）
- 筛选：`is_30hz_pair=true`，共 13,039 pairs
- q 差分：`abs(q_next-q_t)`，单位转换为角度

### 结果

所有 split 合并后的每帧最大关节变化 `max_j |Δq_j|`：

| 分位数 | 变化角度 |
|---|---:|
| P25 | 0° |
| P50 | 0.074° |
| P75 | 0.262° |
| P90 | 0.690° |
| P95 | 1.206° |
| P99 | 3.158° |
| P100 | 6.131° |

帧比例：

| 条件 | 比例 |
|---|---:|
| `max|Δq| >= 0.1°` | 39.27% |
| `max|Δq| >= 0.25°` | 25.55% |
| `max|Δq| >= 0.5°` | 14.82% |
| `max|Δq| >= 1°` | 6.50% |
| `max|Δq| >= 2°` | 2.32% |
| `max|Δq| >= 5°` | 0.12% |

各关节绝对差分均值（deg）为 `[0.1317, 0.0246, 0.0885, 0.1072, 0.1246, 0.0971]`；六个关节的中位数均为 0°，说明大量帧对应关节没有可观测变化。

### 关键观察

当前 20-episode 数据中约 85.18% 的相邻帧最大关节变化小于 0.5°，约 93.50% 小于 1°，因此零残差/identity 是非常强的 baseline；但仍有约 6.5% 的帧变化至少 1°，不能把整个数据集视为静止。

`source_frame_delta` 在 13,039 个 pair 中全部为 1；`delta_time_s` 中位数为 29.999 ms，但范围为 0.263 ms–105.794 ms，说明 `is_30hz_pair` 当前表达的是相邻源帧映射，不是严格的时间间隔阈值。

### 解释

残差训练出现接近 identity 的结果与数据分布一致。后续动作建模需要显式重采样/筛选更高幅度帧，或者使用按动作幅度加权的训练与评价，避免近零残差淹没有效动作样本。

### 结论状态

**SUPPORTED**

### 下一步

- 统计完整 HRDexDB（当前只覆盖已缓存的 20 episode）。
- 比较 0.5°、1°、2° 阈值下的训练/评测样本数和模型表现。
- 核查并决定是否将 `delta_time_s` 纳入严格 30 ms 过滤。

### 证据

- cache manifest: `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_20_seed42.json`
- task sidecars: `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/episodes/*/task/`

## EXP-009 — 多时间间隔 q 差分统计

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

将预测间隔从 30 Hz 的 1 帧扩大到更长时间，可以提高相邻目标帧的动作幅度，减少 near-zero residual 对训练和评价的支配。

### Baseline

- 数据范围：v4 20-episode geometry cache 的 `q_full`
- 仅保留 source frame id 差值等于 stride 的配对
- 30 Hz stride=1 作为基准

### 结果

按 `q_t → q_{t+stride}` 统计每帧最大关节变化 `max_j |Δq_j|`：

| 目标频率 | stride | pairs | P50 (deg) | P90 (deg) | `>=0.5°` | `>=1°` | `>=2°` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 Hz | 1 | 13,039 | 0.074 | 0.690 | 14.82% | 6.50% | 2.32% |
| 15 Hz | 2 | 12,780 | 0.100 | 1.339 | 24.53% | 13.90% | 6.30% |
| 10 Hz | 3 | 12,541 | 0.139 | 1.997 | 30.56% | 19.33% | 9.96% |
| 6 Hz | 5 | 12,079 | 0.280 | 3.328 | 38.86% | 26.40% | 16.23% |
| 5 Hz | 6 | 11,852 | 0.300 | 4.117 | 42.04% | 29.22% | 18.76% |
| 3 Hz | 10 | 10,948 | **0.505** | **6.819** | **51.04%** | **38.18%** | **26.12%** |
| 2 Hz | 15 | 9,866 | 0.800 | 10.221 | 58.11% | 46.43% | 34.15% |
| 1 Hz | 30 | 7,285 | 1.921 | 17.281 | 71.94% | 62.91% | 49.33% |

3 Hz 的各关节平均绝对差分为 `[1.277°, 0.181°, 0.900°, 1.091°, 1.286°, 0.979°]`。

### 关键观察

3 Hz 相比 30 Hz，P50 最大关节变化约增大 6.8 倍，超过 0.5° 的 pair 从 14.82% 增加到 51.04%；因此 3 Hz 确实能显著减弱 zero-residual 问题。更低频率继续增大动作幅度，但 pair 数量减少。

### 解释

当前 cache 已保存完整 `geometry/q_full` 和 `frame_time`，因此上述统计不需要重新读取原始 mesh。它还表明多时间尺度需要显式记录 stride/真实 `Δt`，不能继续把所有样本都标成单一 30 Hz。

### 结论状态

**SUPPORTED**

### 下一步

- 若进入训练，建议先做 3 Hz 单尺度 baseline：生成 stride=10 的 hand flow、`q_t/q_next` 和真实 `delta_time_s`。
- Decoder 可将 `Δt` 作为显式标量输入；Cm 的时间条件同步使用该 `Δt`。
- 再比较 30 Hz、3 Hz 和混合时间间隔，而不是直接把不同 stride 混在同一标签定义中。

### 证据

- geometry cache: `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/episodes/*/geometry/q_full.npy`

## EXP-010 — 3 Hz（stride=10）残差预测三组对照

### 日期

2026-08-21

### 对应指导

`docs/指导/V1.md`

### 假设

3 Hz 的 `q_t → q_{t+10}` 具有更大的动作残差，能够减弱 30 Hz near-zero residual 的影响；若 Cm 对动作预测有贡献，`qt_cm` 应优于 `qt_only`。

### Baseline

- 独立 3 Hz cache：stride=10，真实 `delta_time_s≈0.3 s`
- split：16 train / 2 val / 2 test
- train/val/test pairs：8,480 / 1,346 / 1,122
- 30 epochs，训练步数因数据量为 7,950
- identity：`q_next=q_t`

### 本次修改

- 从 v4 geometry layer 派生 3 Hz hand flow、q pair、真实 dt 和 source stride。
- 三组均使用残差目标：`qt_cm / qt_only / cm_only`。
- 预缓存 3 Hz Cm tokens；不运行 shuffled-flow。

### 结果

全量 split：

| Variant | Best val q MAE (deg) | Test q MAE (deg) | Test identity (deg) |
|---|---:|---:|---:|
| qt_cm | 1.370 | 0.737 | 0.648 |
| qt_only | **1.329** | **0.699** | 0.648 |
| cm_only | 1.376 | 0.759 | 0.648 |

高动作子集（`max|Δq|>=0.5°`；val 557 / test 511）：

| Variant | Val q MAE (deg) | Test q MAE (deg) | Val/Test identity (deg) |
|---|---:|---:|---:|
| qt_cm | 3.108 | 1.449 | 3.094 / 1.389 |
| qt_only | **3.096** | **1.409** | 3.094 / 1.389 |
| cm_only | 3.115 | 1.462 | 3.094 / 1.389 |

### 关键观察

3 Hz 将 identity 的 val MAE 从 30 Hz 的约 0.13° 提高到 1.30°，动作信号明显增强；但三组模型仍只接近 identity，`qt_only` 略好于 `qt_cm`，Cm 没有显现独立增益。高动作子集上三组仍未明显超过 identity。

### 解释

时间间隔增大成功解决了“所有标签几乎为零”的数据分布问题，但仅改变 horizon 不足以让当前 Decoder 学到可泛化的动作残差。可能需要将 `delta_time_s` 显式输入 Decoder、按动作幅度重采样，或改为直接优化 flow/增量方向。

### 结论状态

**INCONCLUSIVE**

### 决策

保留 3 Hz cache 和三组 checkpoint 作为后续多时间尺度 baseline；暂不宣称 Cm 有效或无效。

### 下一步

- 将 `delta_time_s` 作为 Decoder 显式条件，比较 30 Hz/3 Hz/混合 horizon。
- 评估 delta-q MAE、方向准确率和相对 identity improvement。
- 在 3 Hz 上尝试 active-motion 加权训练。

### 证据

- 3 Hz cache：`data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/selection_20_seed42.json`
- qt_cm: `outputs/cmdecoder/cm_decoder_20260821_214719/`
- qt_only: `outputs/cmdecoder/cm_decoder_20260821_214933/`
- cm_only: `outputs/cmdecoder/cm_decoder_20260821_215143/`

## EXP-011 — object-v2 GRAB+ARCTIC Cm 的20-episode 3 Hz三组对照

### 日期

2026-08-22

### 假设

训练更充分且覆盖 object-v2 GRAB+ARCTIC 的 Cm checkpoint 应比早期 GRAB checkpoint 提供更可迁移的动作表示，使 `qt_cm` 或 `cm_only` 相对 `qt_only`/identity 获益。

### Baseline 与控制变量

- 沿用 EXP-010 的3 Hz（stride=10）20-episode manifest：16 train / 2 val / 2 test，8,480 / 1,346 / 1,122 pairs。
- frozen Cm：`outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt`。
- `qt_cm / qt_only / cm_only` 均为 residual prediction、seed=42、30 epochs、全局 batch 32、7,950 optimizer steps。
- identity：`pred_q_next=q_t`；高动作阈值为 `max|Δq|>=0.5°`。
- 只重建20个 episode 的 Cm token sidecar，没有导出全量 cache。

### 结果

全量 split：

| Variant | Best epoch | Best val q MAE (deg) | Test q MAE (deg) | Test identity (deg) |
|---|---:|---:|---:|---:|
| qt_cm | 11 | 1.228 | 0.826 | 0.648 |
| qt_only | 30 | 1.331 | **0.695** | 0.648 |
| cm_only | 10 | **1.200** | 0.773 | 0.648 |

高动作子集（val 557 / test 511）：

| Variant | Val q MAE (deg) | Test q MAE (deg) | Val/Test identity (deg) |
|---|---:|---:|---:|
| qt_cm | **2.293** | 1.423 | 3.094 / 1.389 |
| qt_only | 3.096 | 1.404 | 3.094 / 1.389 |
| cm_only | 2.299 | **1.342** | 3.094 / 1.389 |

相对旧 EXP-010，新 Cm 将 `qt_cm/cm_only` 的最佳 val MAE 分别改善 `0.142°/0.176°`；高动作 val 分别改善约 `0.815°/0.816°`。高动作 test 的 `cm_only` 从 `1.462°` 改善到 `1.342°`，并首次优于 identity `0.047°`。但全量 test 上 `qt_cm/cm_only` 分别比旧实验退化 `0.089°/0.014°`，且仍不如 `qt_only`。

### 实现有效性说明

第一次并行启动时，自动输出目录按秒命名导致 `qt_cm/qt_only` 同目录混写；同时单卡 batch 16 产生15,900 steps，与旧 EXP-010 的双卡全局 batch 32不一致。该批输出 `cm_decoder_20260822_{125826,125827}` 标记为 **INVALID_IMPLEMENTATION**，不参与上表。随后改为单卡 batch 32、错开目录，三组均完成7,950 steps并独立复评。

### 结论状态

**INCONCLUSIVE**

### 解释与决策

新 Cm 确实改变了可解码信息：验证集和高动作 `cm_only` test 均出现明显改善，不能再简单认为 Cm 完全没有动作信号。但增益未稳定传递到 `qt_cm` 的 held-out test，全量 test 仍由不使用 Cm 的 `qt_only` 最佳。当前 test 只有2个 episode，且并非 object-disjoint，因此不足以支持稳定泛化结论。

暂不导出全量 cache。后续扩大数据时按用户指定采用 object-disjoint split，再判断新 Cm 的跨物体贡献。

### 证据

- qt_cm: `outputs/cmdecoder/cm_decoder_20260822_130259/`
- qt_only: `outputs/cmdecoder/cm_decoder_20260822_130301/`
- cm_only: `outputs/cmdecoder/cm_decoder_20260822_130305/`
- 临时评估日志：`/tmp/cmdecoder_objectv2_3hz_eval_20260822/`

## EXP-012 — 全量 object-disjoint 3 Hz qt_cm

### 日期

2026-08-22

### 假设

扩大到跨89个互斥物体的 object-disjoint 数据后，新 object-v2 GRAB+ARCTIC Cm 的动作信息若具有跨物体泛化能力，`qt_cm` 应在 held-out object 的 full/high-motion 指标上稳定改善 identity，并为后续 `qt_only/cm_only` 全量消融提供主模型基线。

### Baseline 与配置

- 3 Hz（stride=10）object-disjoint manifest：568 episodes、284,414 pairs。
- train/val/test：448/58/62 episodes，70/10/9 objects，228,977/28,391/27,046 pairs。
- frozen Cm：`outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt`。
- `qt_cm` residual prediction，seed=42，两卡 DDP，每卡 batch 16、全局 batch 32，30 epochs / 214,650 steps。
- 使用预缓存 normal-flow Cm tokens；checkpoint SHA256 为 `64737bf4453b4ed37de105ea4c17a5e235e63c8bf2df7338c4c88a9d0817ebb5`。

### 结果

训练完成30 epochs / 214,650 steps，用时44分08秒。best checkpoint 出现在 epoch 1（step 7,155）；最终 epoch 的 val MAE 回升到 `0.945°`，表明继续训练没有改善泛化。

| 子集 | Val q MAE (deg) | Val identity | Test q MAE (deg) | Test identity |
|---|---:|---:|---:|---:|
| full | **0.882** | 0.933 | **1.231** | 1.298 |
| high-motion (`>=0.5°`) | **1.485** | 1.638 | **1.931** | 2.092 |

相对 identity，full val/test 分别改善约 `5.4%/5.2%`，high-motion val/test 分别改善约 `9.4%/7.7%`。四个 held-out 口径方向一致。

### 关键观察

- object-disjoint 扩大数据后，`qt_cm` 首次在 full 与 high-motion 的 val/test 上全部超过 identity。
- best epoch=1，后续训练虽然继续降低 train MAE，但没有改善验证泛化；当前30 epochs预算明显过长。
- 本实验只有 `qt_cm`，尚不能区分收益来自当前 q 的跨物体统计、Cm tokens，还是两者组合。

### 结论状态

**SUPPORTED**：支持“全量 object-disjoint 设置下，当前 `qt_cm` 主模型优于 identity”。不支持“增益可独立归因于 Cm”，该结论需要同 split 的 `qt_only/cm_only`。

### 下一步

- 运行全量 `qt_only/cm_only`，保持相同 split、seed 和全局 batch。
- 后续采用 early stopping，或先将预算缩短到约5 epochs。
- 在三组 best checkpoint 上统一报告 full/high-motion test。

### 证据

- cache: `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/selection_576_object_disjoint_seed42.json`
- output: `outputs/cmdecoder/cm_decoder_20260822_143843/`
- best: `outputs/cmdecoder/cm_decoder_20260822_143843/checkpoints/best.pt`

## EXP-013 — 全量 object-disjoint 3 Hz 三组归因对照

### 日期

2026-08-22

### 目的与控制变量

在 EXP-012 的同一全量 object-disjoint split 上补齐 `qt_only/cm_only`，判断 `qt_cm` 的跨物体增益来自当前关节状态、Cm token，还是二者组合。

- cache、split、frozen Cm、seed、residual prediction 与 EXP-012 完全相同。
- `qt_only/cm_only` 均使用两卡 DDP、每卡 batch 16、全局 batch 32。
- 根据 EXP-012 的 epoch 1 最优现象，将两组预算缩短为 5 epochs / 35,775 steps。
- 模型选择只依据 full validation q MAE；最终统一评估各自 `best.pt` 的 full/high-motion val/test。

### 结果

| Variant | Best epoch | Full val | Full test | High-motion val | High-motion test |
|---|---:|---:|---:|---:|---:|
| identity | — | 0.933 | 1.298 | 1.638 | 2.092 |
| qt_cm | 1 | **0.882** | **1.231** | **1.485** | 1.931 |
| qt_only | 5 | 0.965 | 1.328 | 1.636 | 2.082 |
| cm_only | 1 | 0.907 | 1.238 | 1.492 | **1.918** |

单位均为 q MAE（deg），高动作阈值为 `max|Δq|>=0.5°`。

- `qt_only` 在 full val/test 分别比 identity 差约 `3.4%/2.3%`；高动作仅有约 `0.1%/0.5%` 的微弱改善。
- `cm_only` 在 full val/test 分别优于 identity 约 `2.8%/4.6%`，高动作 val/test 分别改善约 `8.9%/8.3%`，四个 held-out 口径方向一致。
- `qt_cm` 与 `cm_only` 非常接近：`qt_cm` 在 full val/test 和 high-motion val 分别领先 `0.025°/0.008°/0.008°`，`cm_only` 在 high-motion test 领先 `0.013°`。
- `cm_only` 的 best epoch=1，epoch 2—5 的 full val 为 `0.936/0.948/0.927/0.915°`，均未超过 epoch 1；5 epoch 预算足以识别当前最优区间。

### 结论状态

**SUPPORTED**：支持“全量 object-disjoint 的跨物体改善主要由 Cm token 提供”。`cm_only` 在四个 held-out 口径上均稳定超过 identity，而 `qt_only` 在 full val/test 退化；`qt_cm` 相比 `cm_only` 的差异很小且方向不完全一致，当前没有证据证明显式加入 `q_t` 能稳定进一步改善泛化。

### 后续建议

- 后续优先围绕 `cm_only/qt_cm` 做多 seed 或 decoder 容量/正则化验证，不再将 `qt_only` 作为主要候选。
- 当前模型选择集中在 epoch 1，后续训练应加入 early stopping，5 epochs 可作为初始上限。
- 若要声称 `qt_cm` 优于 `cm_only`，需多 seed 验证；当前 `0.008°` 量级差异不足以作该结论。

### 证据

- qt_cm: `outputs/cmdecoder/cm_decoder_20260822_143843/`
- qt_only: `outputs/cmdecoder/cm_decoder_20260822_152742/`
- cm_only: `outputs/cmdecoder/cm_decoder_20260822_153543/`

## EXP-014 — wrist-aware baseline 全量三组对照

### 日期

2026-08-22

### 目的与控制变量

在 EXP-013 的同一 object-disjoint 3 Hz split 上，将 baseline 输出扩为 `6维手指 delta-q + 当前腕到目标腕的相对 SE(3)`，比较 `qt_cm / qt_only / cm_only` 三种输入。

- manifest：`selection_576_object_disjoint_seed42.json`，568 episodes、284,414 pairs。
- frozen Cm 与 token sidecar 沿用 SHA256 `64737bf4453b4ed37de105ea4c17a5e235e63c8bf2df7338c4c88a9d0817ebb5`。
- 相对腕监督为 `inverse(wrist_t) @ wrist_next`，平移单位米、旋转为 rotvec；未来 arm FK 不参与。
- seed=42，单卡 batch 32，全局 batch 32，3 epochs / 21,465 steps；三组只改变 `decoder_input`。
- 选模指标为 full `val/loss`，最终需同时报告 q MAE、wrist translation EPE 与 wrist rotation geodesic error。

### 运行状态

三组于 2026-08-22 20:16 CST 启动，GPU 0 运行 `qt_cm/qt_only`，GPU 5 运行 `cm_only`；均完成21,465 steps，无 OOM，单组用时约5分38秒至5分53秒。

### 结果

`qt_cm/cm_only` 的 best 均在 epoch 1，`qt_only` best 在 epoch 3。下表均来自各自 `best.pt`；q 为 MAE（deg），wrist translation 为 EPE（mm），wrist rotation 为 SO(3) 测地误差（deg）。identity wrist 指零相对平移与单位旋转。

Full split：

| Variant | Best epoch | q val/test | Wrist trans val/test | Wrist rot val/test |
|---|---:|---:|---:|---:|
| identity | — | 0.933 / 1.298 | **10.376 / 10.693** | **2.115 / 2.288** |
| qt_cm | 1 | **0.936 / 1.267** | 11.212 / 11.426 | 2.321 / 2.456 |
| qt_only | 3 | 0.974 / 1.333 | 10.615 / 10.877 | 2.176 / 2.331 |
| cm_only | 1 | 0.980 / 1.321 | 11.040 / 11.262 | 2.267 / 2.449 |

High-motion（`max|delta-q|>=0.5°`）：

| Variant | q val/test | Wrist trans val/test | Wrist rot val/test |
|---|---:|---:|---:|
| identity | 1.638 / 2.092 | 13.709 / 13.686 | 2.986 / 3.128 |
| qt_cm | **1.528 / 1.967** | 13.989 / 13.857 | 3.072 / 3.176 |
| qt_only | 1.640 / 2.086 | **13.520 / 13.537** | **2.979 / 3.120** |
| cm_only | 1.583 / 2.016 | 13.785 / 13.691 | 3.030 / 3.192 |

Full validation 总 loss 为 `qt_cm=0.328862`、`qt_only=0.323572`、`cm_only=0.324775`。当前 translation loss 约 `0.30`，而 q/rotation loss 各约 `0.01`，所以等权相加后的 checkpoint 选模事实上主要由 wrist translation 决定。

### 关键观察

- `qt_cm` 在 full test 和 high-motion val/test 的 q 指标三组最好，高动作 q 相对 identity 改善约6%—7%，Cm 对手指动作仍有可解码信息。
- 三组 wrist 在 full val/test 均没有超过零腕运动 baseline；`qt_only` 仅在 high-motion wrist 上有约0.1—1.2%的微小改善，不能证明模型学会了可泛化腕部运动。
- `qt_only` 因腕部误差相对较小而获得最低 full val 总 loss，但其 q 最差；当前总 loss/选模权重不适合直接代表“整体手运动质量”。

### 结论状态

**INCONCLUSIVE**：3-epoch 结果支持 Cm 改善高动作手指 q，但不支持当前 baseline 已学会腕部相对运动。正式与逐点模型比较前，应先规范化三类 loss 权重或采用分项选模，并分析 wrist target 分布/可预测性。

### 证据

- qt_cm: `outputs/cmdecoder/cm_decoder_wrist_qt_cm_20260822_201608/`
- qt_only: `outputs/cmdecoder/cm_decoder_wrist_qt_only_20260822_201608/`
- cm_only: `outputs/cmdecoder/cm_decoder_wrist_cm_only_20260822_201607/`

## EXP-015 — wrist-aware baseline 纯点 loss 三组对照

### 日期

2026-08-22

### 假设与控制变量

将 EXP-014 中量纲不一致的 q/translation/rotation 参数 loss 替换为统一的整手固定对应点误差，可能更直接地监督整体手运动，并避免总 loss 被 translation 分项主导。

- 数据、split、Cm/token、seed、三种 `decoder_input`、全局 batch 32和3-epoch预算均与 EXP-014 相同。
- 输出仍为6维 finger delta-q、3维相对 wrist translation 和3维相对 wrist rotvec。
- 唯一反传目标为1538个固定对应点的米制 Smooth-L1，beta=`0.01 m`；原 q/translation/rotation loss 权重均为0但继续记录。
- checkpoint 按 `val/hand_points/epe_mm` 选取。

### 实现有效性

- GT q+wrist 重建 cache target point 的平均/最大 EPE 为 `0.000007/0.000170 mm`。
- q、translation、rotvec 均从 point loss 获得有限非零梯度。
- 100倍点缩放 smoke 每步触发梯度裁剪，已废弃；正式配置使用原始米制点，8-step smoke 梯度范数约0.312且不裁剪。

### 运行状态

三组于 2026-08-22 20:39 CST 在 GPU 0/5 启动，epoch 1 后发现3 Hz v1 cache 的目标点坐标错误，已主动停止全部进程。

### 结果

无有效科研结果。启动期 `identity_hand/epe_mm≈0.36 mm`，但独立 wrist identity translation EPE 约 `10.4 mm`；代码核验确认 `build_horizon_cache.py` 使用了 `_to_frame(hand[target], wrist[target])`，而联合运动监督必须使用 `_to_frame(hand[target], wrist[current])`。因此当前 point target 已消除 wrist 运动，继续训练会错误地推动预测 wrist 接近单位变换。

### 结论状态

**INVALID_IMPLEMENTATION**：输入监督不包含任务要求的腕部运动，不能用于评价纯点 loss 假设。

### 证据

- qt_cm: `outputs/cmdecoder/cm_decoder_wrist_pointloss_qt_cm_20260822_203951/`
- qt_only: `outputs/cmdecoder/cm_decoder_wrist_pointloss_qt_only_20260822_203951/`
- cm_only: `outputs/cmdecoder/cm_decoder_wrist_pointloss_cm_only_20260822_203951/`
- stable smoke: `outputs/cmdecoder/cm_decoder_wrist_pure_point_meters_smoke_20260822_203820/`

上述 smoke 来自近静止 episode，只验证数值反传，不足以验证全量 target 坐标语义。

### 修正后 v2 复跑

保留 v1 以复现历史实验，新建3 Hz v2：目标点改为在当前腕坐标系表达，并由 manifest `hand_flow_frame=current_wrist` 与 dataset fail-fast 共同约束。v2 仍为568个 episode、284,414 pairs，object-disjoint split 与 frozen Cm checkpoint 均不变；568个 token sidecar 重导后全量扫描0错误。

三组均训练3 epochs / 21,465 steps，无 OOM、无梯度裁剪：

| Variant | Best epoch | 用时 |
|---|---:|---:|
| qt_cm | 3 | 52:05 |
| qt_only | 2 | 52:35 |
| cm_only | 3 | 48:39 |

直接参数 loss 的 EXP-014 单组仅约5分38秒至5分53秒。纯点 loss 每 step 需要点到 link 反绑、预测 q 可微 FK、1538点腕部变换及其反传，实测约 `115—140 ms/step`，而原实现约 `15 ms/step`；GPU 0 上 `qt_cm/qt_only` 并发还会互相争用。

Full split（点与 wrist translation 单位 mm；q 与 wrist rotation 单位 deg）：

| Variant | Point EPE val/test | q val/test | Wrist trans val/test | Wrist rot val/test |
|---|---:|---:|---:|---:|
| identity | 12.118 / 12.697 | **0.933 / 1.298** | 10.376 / 10.693 | 2.115 / 2.288 |
| qt_cm | **3.412 / 3.801** | 1.239 / 1.593 | **3.772 / 4.030** | **1.697 / 1.908** |
| qt_only | 12.360 / 12.887 | 1.095 / 1.437 | 10.552 / 10.842 | 2.235 / 2.376 |
| cm_only | 3.467 / 3.838 | 1.222 / 1.575 | 3.776 / 4.051 | 1.701 / 1.889 |

High-motion（`max|delta-q|>=0.5°`）：

| Variant | Point EPE val/test | q val/test | Wrist trans val/test | Wrist rot val/test |
|---|---:|---:|---:|---:|
| identity | 16.175 / 16.323 | **1.638 / 2.092** | 13.709 / 13.686 | 2.986 / 3.128 |
| qt_cm | **3.988 / 4.494** | 1.768 / 2.234 | **4.444 / 4.764** | **2.158 / 2.441** |
| qt_only | 15.953 / 16.153 | 1.719 / 2.145 | 13.536 / 13.561 | 2.995 / 3.131 |
| cm_only | 4.052 / 4.514 | 1.757 / 2.217 | 4.489 / 4.786 | 2.181 / 2.436 |

### 修正后结论状态

**SUPPORTED（几何目标）**：纯点监督使 `qt_cm` full test 点 EPE 相对 identity 改善约70%，wrist translation 改善约62%，wrist rotation 改善约17%，且 high-motion 方向一致。Cm token 是主要信息来源；`qt_only` 基本不能预测整体手运动，`qt_cm/cm_only` 差异很小且无稳定归因证据。

**REFUTED（参数可辨识性）**：仅最小化对应点误差不能保证恢复更准确的 q。`qt_cm` full/high-motion test q MAE 均差于 identity，说明 wrist 与关节自由度可用不同参数组合产生接近的点几何；若下游需要准确 q，应恢复一个小权重 q 辅助约束或其他参数先验。

### v2 证据

- cache: `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v2/selection_576_object_disjoint_seed42.json`
- qt_cm: `outputs/cmdecoder/cm_decoder_wrist_v2_pointloss_qt_cm_20260822_215718/`
- qt_only: `outputs/cmdecoder/cm_decoder_wrist_v2_pointloss_qt_only_20260822_215718/`
- cm_only: `outputs/cmdecoder/cm_decoder_wrist_v2_pointloss_cm_only_20260822_215718/`

## EXP-016 — 旧版 Cm 下逐点 flow 新架构十 epoch训练

### 日期

2026-08-23

### 对应指导

当前 CmDecoder 逐点 flow 架构记录（见 `architecture_log.md`）。

### 假设

在相同旧版 frozen Cm C=256 token、相同 v2 object-disjoint 3 Hz 数据和固定对应点监督下，逐点 slot-routing flow decoder 直接预测整手对应点运动，可能比整体 q+wrist baseline 更好地拟合几何运动；q fitting 只作为训练后处理，不改变表示学习。

### Baseline

- frozen Cm checkpoint: `outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt`
- cache: `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v2/selection_576_object_disjoint_seed42.json`
- 对照: EXP-015 v2 `qt_cm` wrist-aware pure-point baseline

### 本次修改

- 使用 `src/task/CmDecoder/point_config.py` 的 `CmPointFlowModel`
- 训练预算从5 epoch改为10 epoch
- 使用已迁移的 v2 cache；静态 point-to-link/local-point 字段已具备，但逐点网络训练路径不读取 q 或该 binding，binding 优化主要服务 wrist baseline 和后处理
- 其余数据 split、Cm checkpoint、batch、flow loss 保持不变

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=5 \
  /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.CmDecoder.train \
  --config src.task.CmDecoder.point_config:Config \
  --set train.epochs=10 \
  --set train.description=cmdecoder_point_flow_old_cm_10ep \
  --set wandb.enable=false
```

### 结果

训练已完成 10 epoch / 143,110 steps。验证集 `val/hand_flow/epe_mm` 最好为 `3.3256 mm`
（epoch 9，step 128,799），最终 epoch 10 为 `3.3297 mm`；zero-flow 验证基线为
`12.1176 mm`，相对改善约 `72.5%`。输出：`outputs/cmdecoder/cm_decoder_20260823_000423/`。

随后使用 best checkpoint 做 held-out point-flow 评估：

| 口径 | 样本数 | hand-flow EPE | zero-flow EPE |
|---|---:|---:|---:|
| full val | 28,391 | 3.325 mm | 12.118 mm |
| full test | 27,046 | 3.726 mm | 12.697 mm |
| high-motion val | 15,905 | 3.725 mm | 16.175 mm |
| high-motion test | 16,618 | 4.192 mm | 16.323 mm |

point-flow → q/wrist fitting 先做每个 split 256 样本、100 步的 pilot。full val/test 的
拟合后整手点 EPE 为 `3.182/3.501 mm`，high-motion val/test 为 `3.200/3.447 mm`；
腕平移误差约 `3.53—3.64 mm`，旋转误差约 `1.16—2.10°`。q MAE 在 full test 和
high-motion test 为 `1.68°/2.22°`，未稳定优于 identity 的 `1.48°/2.10°`。

### 结论状态

INCONCLUSIVE（point-flow 的 full/high-motion val/test 已完成；q/wrist fitting 目前仍是
每个 split 256 样本的 pilot，尚未完成全量 fitting）。

### 下一步

- 完成10 epoch后按 `val/hand_flow/epe_mm` 选择 best checkpoint；
- 如需正式结论，运行 point-flow → q/wrist fitting 的全量 full/high-motion val/test；
- 与 EXP-015 的 `qt_cm` point EPE、wrist EPE 和 q MAE 对比。
