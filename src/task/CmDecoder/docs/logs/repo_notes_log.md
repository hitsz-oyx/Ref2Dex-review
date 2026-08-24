# CmDecoder 任务仓库常识

## 任务目标

冻结 16-slot Cm checkpoint，将 HRDexDB Inspire F1 手部动作编码为 Cm，并重建未来帧的6维手指关节角；当前同时保留30 Hz相邻帧与3 Hz（stride=10）诊断设置。

## 当前状态

已完成训练入口、frozen-Cm wrapper 和 HRDexDB 数据集初版；当前数据位于 `/home2/wyy/oyx_ws/HRDexDB/v0`。2026-08-22 已补齐非视频运动资产：591 组 arm position/time、591 组 v1 compact pose、555 组 v2 compact pose、576 组 C2R；按当前几何合同最多可形成 576 个完整 episode。视频未参与本次下载。当前 builder 尚未读取根级 compact pose NPZ，扩展 cache 前需优先支持 v2、缺失时回退 v1。

## 当前数据 cache

- builder: `python -m src.task.CmDecoder.build_cache --episodes 50 --workers 8`
- manifest: `data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_20_seed42.json`
- schema: `cmdecoder_layered_v4` + `cmdecoder_cm_tokens_v1`
- 规模: 20 episodes（16 train / 2 val / 2 test），10,266 / 1,475 / 1,298 samples；geometry/task 约 1.5 GB，含 Cm tokens 约 2.0 GB
- Dataset 使用 mmap lazy load；禁止在 DDP 训练初始化阶段现场构建 cache。
- active-motion sanity: `active_motion_threshold_deg=0.5`，仍为相邻 30 Hz 单步，不改变正式任务定义。
- 3 Hz manifest: `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/selection_20_seed42.json`；其20个 Cm token sidecar 当前绑定 object-v2 GRAB+ARCTIC checkpoint，详见 [`memory_log.md`](memory_log.md)。
- 后续扩大数据时采用 object-disjoint split；当前20-episode manifest仍是历史 episode-random split。
- 全量 object-disjoint 3 Hz manifest: `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/selection_576_object_disjoint_seed42.json`；568 episodes、284,414 pairs，当前 frozen-Cm token 已全部导出。

## 文档索引

- Pipeline：`architecture_log.md`
- 修改记录：`modification_log.md`
- 自主决策：`decision_log.md`
