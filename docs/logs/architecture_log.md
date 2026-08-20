# Ref2Dex 架构记录

## 当前结构

Ref2Dex 按研究 Task 组织实现：共享数据处理与训练基础设施位于 `process/`、`src/base/`，Task 实现位于 `src/task/<Task>/`，测试位于 `tests/`。根级入口是 [`docs/项目总览.md`](../项目总览.md)；各 Task 的事实、架构、实验和决策在其自身 `docs/` 下维护。

## 共享数据流

原始数据经 dataset adapter 和 Stage 4 预处理生成时序 cache，Task Dataset 按 sequence 级 train/val/test 划分读取 cache，模型与 Runner 负责训练和评估。跨帧点流监督必须依赖稳定 point correspondence；统计量和 calibration 只能由 train split 生成。

## Cm 关系

Cm 是独立 Task：当前主路线为 V1.2 object-only GRAB + ARCTIC。Stage4 → object mmap/ragged candidate → sampling bank → 单侧 hand/object Dataset → frozen DenseToken + hard-gated Slot Attention Cm → point-flow evaluator。Scene Cache V1.1.2 保留作历史对照，正式路径使用在线 Frozen DenseToken。

GRAB Stage4 的 raw adapter 同时负责 sequence 和 subject MANO asset 解析：传入 root 可能是 `dataset/GRAB` 或 `dataset/GRAB/data`，但 `v_template` 必须从实际 raw asset 根解析。Cm/V2 正式生成默认拒绝缺失 subject template，防止平均 MANO 手型污染 hand geometry 和 candidate mask。
