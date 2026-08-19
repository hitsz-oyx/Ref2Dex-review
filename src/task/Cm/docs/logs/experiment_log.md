# Cm 实验记录

## 当前研究状态

V1.2 object-only GRAB + ARCTIC 的 full-data Stage4、object-v2 cache、B=4 sampling bank 和 E1 统计已经完成，共得到 662557 个有效 sample。GRAB/ARCTIC sample 数接近，但 flow RMS 相差约 2.26×；下一步是固定 split、train-only calibration 与 GRAB-only/ARCTIC-only/mixed 短训。

## 历史证据索引

历史实验原文保留在 [`../../research/log.md`](../../research/log.md)，其中包含 V1.2 smoke、Scene Cache V1/V1.1、DenseToken parity 和 V1.1.2 两层 cache 路线的假设、结果及后续决策。本文档作为规范入口，维护当前状态和正式 EXP 记录。

### 当前结论

- V1.2 implementation gate：`SUPPORTED`（真实 smoke 级别）。
- V1.1.1 downstream cache parity：`SUPPORTED`；feature 逐元素差异归因于 spconv 非确定性，不作为科学反证。
- V1.1.2 全量 DenseToken bank：`INVALID_IMPLEMENTATION`/路线撤回，因资源成本过高而停止，不用于效果结论。

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
