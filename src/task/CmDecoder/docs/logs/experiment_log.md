# CmDecoder 实验记录

## 当前状态

已完成两次 overfit sanity check。旧版静止帧实验被判为 INCONCLUSIVE；active-motion 窗口已验证训练链路能超过 identity baseline。当前仍未进行泛化结论实验。

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
