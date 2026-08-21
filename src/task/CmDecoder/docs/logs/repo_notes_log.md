# CmDecoder 任务仓库常识

## 任务目标

冻结 GRAB-only 的 16-slot Cm checkpoint，将 HRDexDB Inspire F1 相邻 30 Hz 手部动作编码为 Cm，并重建下一帧 6 维手指关节角。

## 当前状态

已完成训练入口、frozen-Cm wrapper 和 HRDexDB 数据集初版；当前数据位于 `/home2/wyy/oyx_ws/HRDexDB/v0`。2026-08-20 已通过临时代理完成 LFS 内容拉取，591 组 Inspire F1 hand q/C2R 可读取；单 episode 30 Hz 几何 smoke 和 checkpoint 冻结检查通过。

## 当前数据 cache

- builder: `python -m src.task.CmDecoder.build_cache --episodes 50 --workers 8`
- manifest: `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_20_seed42.json`
- schema: `cmdecoder_layered_v4` + `cmdecoder_cm_tokens_v1`
- 规模: 20 episodes（16 train / 2 val / 2 test），10,266 / 1,475 / 1,298 samples；geometry/task 约 1.5 GB，含 Cm tokens 约 2.0 GB
- Dataset 使用 mmap lazy load；禁止在 DDP 训练初始化阶段现场构建 cache。
- active-motion sanity: `active_motion_threshold_deg=0.5`，仍为相邻 30 Hz 单步，不改变正式任务定义。

## 文档索引

- Pipeline：`architecture_log.md`
- 修改记录：`modification_log.md`
- 自主决策：`decision_log.md`
