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
