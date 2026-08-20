# Cm 任务仓库常识

## 任务目标

从当前 target object、当前单手状态和未来 hand point flow 学习 `C_m`，预测目标物体短时 point flow。

## 当前状态

主路线是 V1.2.1 object-only GRAB + ARCTIC；左右手独立采样，`dataset_id` 仅用于诊断。Scene Cache V1.1.2 为历史对照，正式训练在线运行 Frozen DenseToken。full-data Stage4、object-v2 cache、sampling bank、E1 统计、sequence 固定 split、train-only calibration 和 no-gate + time condition 的 mixed / GRAB-only / ARCTIC-only 300-step 短训都已完成；随后做了 train-only 吞吐 benchmark：GRAB 1/2/3 GPU 约 73.4 / 143.1 / 209.9 samples/s，ARCTIC 3 GPU 约 209.2 samples/s；mixed 3 GPU batch-size sweep 在 8/16/24/32/48/64 上得到约 207.1 / 282.6 / 329.3 / 343.5 / 366.7 / 357.1 samples/s。当前 mixed 正式入口按 3 GPU + batch_size=48、50 epochs/184650 steps 运行。用户追加的 GRAB gate+cm64 候选先在 GPU 1、5 共享运行，后从安全 checkpoint step 16188 迁移到空闲 GPU 6、7；共享阶段约 111 samples/s，不得与独占卡 benchmark 直接比较，迁移后首段约 270 samples/s。

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

## V2 数据链路关键约定

- GRAB 原始根目录允许传 `dataset/GRAB` 或 `dataset/GRAB/data`；必须先解析出直接包含 `subject/*.npz` 的 sequence root。`rhand/lhand` 中的 `vtemp` 是相对路径（例如 `tools/subject_meshes/male/s1_rhand.ply`），解析时要相对于 raw asset 根（通常为 `dataset/GRAB/data`），不能简单拼到 `dataset/GRAB`。
- 正式 Cm/V2 Stage4 默认要求每条 GRAB 手都加载 subject-specific `v_template`；缺失时应失败而不是静默回退平均 MANO。只有调试/兼容旧数据时显式传 `--allow-default-mano` 才允许回退。
- Stage4 输出的 `meta.json` 必须记录 `source_grab_root`、`resolved_sequence_root` 和 `subject_vtemplate_policy`。这三项用于确认 cache 是否由正确的 raw layout 和手型语义生成。
- 当前 `cm_stage4_v2` / `cm_object_v2` cache 是在上述路径修复前生成的，GRAB 手点和 candidate mask 可能使用平均 MANO；在正式比较或长训前必须重建对应 GRAB cache。不要直接覆盖旧 cache，先写到新输出目录并保留旧产物用于追溯。
- V2 的其他合同保持不变：4 倍下采样（30 Hz）、4096 object pool、当前帧 5 cm candidate、object-only mmap/ragged 转换、B=4 sampling bank、sequence fixed split、train-only calibration。
