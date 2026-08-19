# Cm 实验记录

## 当前研究状态

V1.2 object-only GRAB + ARCTIC 的 full-data Stage4、object-v2 cache、B=4 sampling bank 和 E1 统计已经完成，共得到 662557 个有效 sample。V1.2.1 已补上 sequence 固定 split、train-only calibration、`mp.Value` worker epoch 同步、LRU cache 和 no-gate + time condition 的 mixed / GRAB-only / ARCTIC-only 小规模短训；当前结论仍是 INCONCLUSIVE，300-step 预算更像实现与稳定性 gate。

## 历史证据索引

历史实验原文保留在 [`../../research/log.md`](../../research/log.md)，其中包含 V1.2 smoke、Scene Cache V1/V1.1、DenseToken parity 和 V1.1.2 两层 cache 路线的假设、结果及后续决策。本文档作为规范入口，维护当前状态和正式 EXP 记录。

### 当前结论

- V1.2 implementation gate：`SUPPORTED`（真实 smoke 级别）。
- V1.2.1 mixed data-only short training：`INCONCLUSIVE`（实现有效，但 300-step 级别未形成明确效果结论）。
- V1.2.1 single-dataset short training：`INCONCLUSIVE`（GRAB-only / ARCTIC-only 300-step 级别也未形成明确效果结论）。
- V1.1.1 downstream cache parity：`SUPPORTED`；feature 逐元素差异归因于 spconv 非确定性，不作为科学反证。
- V1.1.2 全量 DenseToken bank：`INVALID_IMPLEMENTATION`/路线撤回，因资源成本过高而停止，不用于效果结论。

## EXP-002 — V1.2.1 fixed split + no-gate/time mixed short training

### 日期

2026-08-19

### 对应指导

`docs/指导/V1.2.1.md`

### 假设

只改数据链路，并把模型保持在稳定的 no-gate + time condition 语义下，配合固定 sequence split 和 train-only flow calibration，可以先把 V1.2.1 的实现问题收敛到可重复的训练基线，再判断 mixed GRAB/ARCTIC 数据是否会带来早期优化信号。

### Baseline

- commit: `ccb76ca`
- config: `src/task/Cm/configs/object_v2_grab_arctic.yaml`
- checkpoint: 无；本次只做短训验证

### 本次修改

- 增加 sequence 固定 split 生成器；
- `_MmapSequenceDataset` 改为 `mp.Value` 共享 epoch；
- object-v2 cache 加入 LRU 打开数上限；
- train-only flow calibration 接入 object-v2 mixed root；
- mixed config 固定为 `use_time_condition=true`、`use_slot_gate=false`；
- runner 增加联合 root 的 object-v2 识别。

### 实现审查

Verdict: PASS

关键检查：
- train/val/test 按 sequence 互斥；
- dataloader 能稳定读取联合 `grab/` + `arctic/` root；
- 2-step smoke 与 3-seed 短训均无 NaN / 崩溃 / 数据错误；
- no-gate + time condition 与当前配置一致。

### 实验命令

```bash
python -m src.task.Cm.train \
  --config src/task/Cm/configs/object_v2_grab_arctic.yaml \
  --set train.max_steps=300 \
  --set train.seed=42
```

### 结果

| Seed | Final train/mean_stride_epe_mm | Final train/val_mean_stride_epe_mm | Final train/relative_epe | Final train/zero_flow_improvement |
| --- | ---: | ---: | ---: | ---: |
| 42 | 28.0734 mm | 28.0734 mm | 1.00372 | -0.003715 |
| 43 | 27.5051 mm | 27.5051 mm | 1.00405 | -0.0040464 |
| 44 | 28.0194 mm | 28.0194 mm | 1.00533 | -0.0053268 |

证据文件：
- `output/exp/cm_v121/cm_v121_mixed_seed42.stdout.log`
- `output/exp/cm_v121/cm_v121_mixed_seed43.stdout.log`
- `output/exp/cm_v121/cm_v121_mixed_seed44.stdout.log`
- `outputs/cm/cm_v121_mixed_seed42_20260819_094543/metrics.jsonl`
- `outputs/cm/cm_v121_mixed_seed43_20260819_094543/metrics.jsonl`
- `outputs/cm/cm_v121_mixed_seed44_20260819_094543/metrics.jsonl`

### 关键观察

- 数据链路与训练链路都能闭环，说明 V1.2.1 的实现修正是有效的；
- 三个 seed 的 300-step 结果都没有给出明显的优化信号，zero-flow 对比略差于当前预测；
- 这个 budget 更像实现 gate，而不是足够强的科学判定。

### 解释

fixed split 和 train-only calibration 解决的是可复现性与统计口径问题；它们让实验可比，但并不会自动提升指标。300 step 训练太短，且 cosine schedule 已明显衰减，当前结果不足以判断 mixed data-only 假设是否成立。

### 结论状态

INCONCLUSIVE

### 决策

不把当前 300-step checkpoint 作为候选最佳模型；保留配置与日志，继续做更有区分度的 GRAB-only、ARCTIC-only、mixed 对照。

### 下一步

按 V1.2.1 指导继续跑同预算的 GRAB-only / ARCTIC-only / mixed 对照，并在需要时提高训练步数再比较。

### 证据

- commit: `HEAD`
- config: `src/task/Cm/configs/object_v2_grab_arctic.yaml`
- train log: `output/exp/cm_v121/`
- metrics: `outputs/cm/cm_v121_mixed_seed*/metrics.jsonl`

## EXP-003 — V1.2.1 grab-only / arctic-only small-scale comparison

### 日期

2026-08-19

### 对应指导

`docs/指导/V1.2.1.md`

### 假设

如果 V1.2.1 的固定 split 与 train-only calibration 足够稳定，那么把 mixed 数据拆成 GRAB-only / ARCTIC-only 后，至少应在 300-step 小预算上看到比 zero-flow 更一致的优化趋势；否则说明这个预算只够验证链路，不足以验证数据假设。

### Baseline

- commit: `cb7ae88`
- config: `src/task/Cm/configs/object_v2_grab_only.yaml` / `src/task/Cm/configs/object_v2_arctic_only.yaml`
- checkpoint: 无；只做短训对照

### 本次修改

- 新增 GRAB-only 与 ARCTIC-only object-v2 配置；
- 修正 object-v2 flow calibration 的 sequence 计数口径；
- 复用各自独立的固定 split + train-only calibration；
- 维持 `use_time_condition=true`、`use_slot_gate=false` 不变。

### 实现审查

Verdict: PASS

关键检查：
- 单数据集 split 可读；
- loader 能稳定返回 train/val/test；
- calibration metadata 与 config scale 一致；
- 300-step run 无 NaN / 崩溃 / 数据错误。

### 实验命令

```bash
python -m src.task.Cm.train \
  --config src/task/Cm/configs/object_v2_grab_only.yaml \
  --set train.max_steps=300 \
  --set train.seed=42
```

### 结果

| Run | val/mean_stride_epe_mm | val/mean_stride_relative_epe | val/zero_flow_improvement | 备注 |
| --- | ---: | ---: | ---: | --- |
| GRAB seed42 | 51.3116 | 1.00292 | -0.00292 | 已完成 |
| GRAB seed43 | 51.3039 | 1.00245 | -0.00245 | 已完成 |
| ARCTIC seed42 | 23.5727 | 1.00296 | -0.00296 | 已完成 |

### 关键观察

- GRAB-only 和 ARCTIC-only 都能正常收敛到稳定的 300-step 轨迹，但都没有明显优于 zero-flow；
- GRAB 和 ARCTIC 的尺度差异仍然显著，说明 train-only calibration 是必要的，但仅靠校准不能让短预算立刻出现正向信号；
- 这批结果和 mixed 小预算一起看，仍然更像是“实现可用”而不是“科学假设已证实”。

### 解释

固定 split 与 train-only calibration 已经把可复现性问题收住了；剩下的瓶颈是预算太短，cosine lr 也已经衰减到零，模型还没进入能分辨数据差异的区间。

### 结论状态

INCONCLUSIVE

### 决策

不再继续用更多 300-step seed 去堆重复证据；如果后面要进一步判断数据假设，应把预算加长，而不是只加 seed。

### 下一步

先基于当前 completed runs 更新文档和提交，再考虑是否把预算提高到更能区分 mixed / single-dataset 的级别。

### 证据

- `output/exp/cm_v121/cm_v121_grab_seed42.stdout.log`
- `output/exp/cm_v121/cm_v121_grab_seed43.stdout.log`
- `output/exp/cm_v121/cm_v121_arctic_seed42.stdout.log`
- `outputs/cm/cm_v121_grab_seed42_20260819_103530/metrics.jsonl`
- `outputs/cm/cm_v121_grab_seed43_20260819_103530/metrics.jsonl`
- `outputs/cm/cm_v121_arctic_seed42_20260819_103530/metrics.jsonl`

## EXP-001 — V1.2 full-data Stage4/cache 与 E1 统计

### 日期

2026-08-18

### 对应指导

`docs/指导/V1.2.md`

### 假设

复用现有 GRAB/ARCTIC Stage4，转换为统一 object-v2 mmap/ragged cache 后，可以在不改变 Cm 模型和 loss 的前提下获得足够的跨数据集 transition，用于决定混合采样和 flow calibration。

### Baseline

- code: 当前 V1.2 implementation（commit `48e3b16`）
- config: `configs/object_v2_grab_arctic.yaml`
- checkpoint: 无；本 EXP 只做数据与 cache gate

### 本次修改

- 生成 GRAB/ARCTIC 全量 Stage4 object-only cache；
- 转换为 `cm_object_v2` mmap/ragged cache；
- 构建 B=4、512 点 sampling bank；
- 对两套 cache 分别及联合计算 E1 统计。

### 实验命令

```bash
python -m process.GRAB.stage4_cm ... --ds-rate 4 --num-obj-points 4096
python -m process.ARCTIC.stage4_cm ... --ds-rate 1 --num-obj-points 4096
python -m process.common.object_cache_v2 ...
python -m src.task.Cm.build_object_sampling_bank ... --bank-size 4 --num-points 512
python -m src.task.Cm.compute_object_v2_stats ... --min-stride 1 --max-stride 10
```

### 结果

| Dataset | sequences | frames | valid samples | flow RMS |
| --- | ---: | ---: | ---: | ---: |
| GRAB | 1335 source / 1028 newly written | 643090 | 328309 | 53.20 mm |
| ARCTIC | 301 | 436546 | 334248 | 23.55 mm |
| Combined | 1636 cache sequences | — | 662557 | 41.01 mm（按两侧统计合并） |

证据文件：`data/processed_data/cm_object_v2/object_v2_statistics.json`、`grab_statistics.json`、`arctic_statistics.json`。cache 约 `110 GB`，sampling bank 已生成。

### 关键观察

- GRAB 与 ARCTIC 有效 sample 数接近，`sqrt(N)` 初始采样比例约 `1:1`；
- flow RMS 相差约 `2.26×`（GRAB 更大），不能直接假设两个 dataset 的梯度分布一致；
- 首次联合统计命令暴露了 combined root 兼容问题，已修复 `CmObjectV2Dataset` 对嵌套 `grab/`、`arctic/` root 的读取。

### 解释

当前证据支持使用近似均衡的 dataset sampler 作为 E2 初始方案，但不支持 dataset-specific normalization 或最终 sampler 决策。当前统计是 E1 诊断统计，不等同于严格 train-only calibration metadata。

### 实现审查

Verdict: PASS（cache/E1 gate）；正式训练 calibration gate 尚未完成。

### 结论状态

`SUPPORTED`（仅支持 V1.2 数据链路可运行和 E1 统计，不代表模型效果假设已验证）。

### 决策

保留 full-data object-v2 cache；下一步先补 train-only calibration、固定 sequence split 和 GRAB-only/ARCTIC-only/mixed 短训配置。

### 下一步

实现 object-v2 的 train-only calibration 与可复现 split，然后执行三组 5k--10k step 短训。
