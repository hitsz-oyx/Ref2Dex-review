# Cm 任务仓库常识

## 任务目标

从当前 target object、当前单手状态和未来 hand point flow 学习 `C_m`，预测目标物体短时 point flow。

## 当前状态

主路线是 V1.2 object-only GRAB + ARCTIC；左右手独立采样，`dataset_id` 仅用于诊断。Scene Cache V1.1.2 为历史对照，正式训练在线运行 Frozen DenseToken。full-data Stage4、object-v2 cache、sampling bank 和 E1 统计已经完成；固定 split、train-only calibration 与三组短训尚未完成。

## 文档索引

- Pipeline/接口：[`architecture_log.md`](architecture_log.md)
- 实验证据：[`experiment_log.md`](experiment_log.md)
- 自主决策：[`decision_log.md`](decision_log.md)
- 修改记录：[`modification_log.md`](modification_log.md)
- 版本指导：[`../指导/`](../指导/)
- 兼容性/历史说明：[`../项目说明.md`](../项目说明.md)、[`../Cm框架.md`](../Cm框架.md)、[`../Pipeline.md`](../Pipeline.md)

## 主要入口

`process/GRAB/stage4_cm.py`、`process/ARCTIC/stage4_cm.py`、`process/common/object_cache_v2.py`、`src/task/Cm/dataset_object_v2.py`、`src/task/Cm/model.py`、`src/task/Cm/runner.py`、`src/task/Cm/train.py`、`src/task/Cm/eval.py`。
