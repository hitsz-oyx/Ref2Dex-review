# Cm 任务仓库常识

## 任务目标

从当前 target object、当前单手状态和未来 hand point flow 学习 `C_m`，预测目标物体短时 point flow。

## 当前状态

主路线是 V1.2.1 object-only GRAB + ARCTIC；左右手独立采样，`dataset_id` 仅用于诊断。Scene Cache V1.1.2 为历史对照，正式训练在线运行 Frozen DenseToken。full-data Stage4、object-v2 cache、sampling bank、E1 统计、sequence 固定 split、train-only calibration 和 no-gate + time condition 的 mixed / GRAB-only / ARCTIC-only 300-step 短训都已完成；随后做了 train-only 吞吐 benchmark：GRAB 1/2/3 GPU 约 73.4 / 143.1 / 209.9 samples/s，ARCTIC 3 GPU 约 209.2 samples/s；mixed 3 GPU batch-size sweep 在 8/16/24/32/48/64 上得到约 207.1 / 282.6 / 329.3 / 343.5 / 366.7 / 357.1 samples/s。当前 mixed 正式入口按 3 GPU + batch_size=48、50 epochs/184650 steps 运行。用户追加的 GRAB gate+cm64 候选在共享 GPU 1、5 上按 batch 48、50 epochs/134900 steps 运行；共享初始吞吐约 111 samples/s，不得与独占卡 benchmark 直接比较。

## 文档索引

- Pipeline/接口：[`architecture_log.md`](architecture_log.md)
- 实验证据：[`experiment_log.md`](experiment_log.md)
- 自主决策：[`decision_log.md`](decision_log.md)
- 修改记录：[`modification_log.md`](modification_log.md)
- 版本指导：[`../指导/`](../指导/)
- 兼容性/历史说明：[`../项目说明.md`](../项目说明.md)、[`../Cm框架.md`](../Cm框架.md)、[`../Pipeline.md`](../Pipeline.md)
- 当前研究指导：`../指导/V1.2.1.md`

## 主要入口

`process/GRAB/stage4_cm.py`、`process/ARCTIC/stage4_cm.py`、`process/common/object_cache_v2.py`、`src/task/Cm/build_object_v2_splits.py`、`src/task/Cm/dataset_object_v2.py`、`src/task/Cm/compute_flow_scale.py`、`src/task/Cm/model.py`、`src/task/Cm/runner.py`、`src/task/Cm/train.py`、`src/task/Cm/eval.py`。
