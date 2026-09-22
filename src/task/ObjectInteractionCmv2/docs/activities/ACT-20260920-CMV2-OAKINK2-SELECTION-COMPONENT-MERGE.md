# OakInk2 时间线 gap 的 selection 根因追溯

- timestamp: 2026-09-20 +0000
- activity_id: ACT-20260920-CMV2-OAKINK2-SELECTION-COMPONENT-MERGE
- work_version: V1.11.1
- mode: inspect
- change_level: L0
- branch: cmv2
- git_commit: cb48d8f
- scope: 只读追溯 OakInk2 MANO 最大点流对应的时间线断点。

## Root cause

selection index 中目标记录的 official timeline position 在 2254 后直接从 2254 跳至 2282；相应 `selected_frame_ids` 从 9025 跳至 9137。官方 timeline 的 position 2255--2281 存在，并非原始录制缺帧；它们未进入 selected 列表。

该记录包含两个独立的 contact component：`8801--9025` 与 `9137--9301`。`segment_motion_then_contact` 正确识别这两个 component，但将它们的帧取集合并集并排序为单一 `selected_frame_ids`；`_selection_row` 因而将不连续列表作为一个 selection row。MANO producer 逐项复制 `selected_frame_ids`/`selected_timeline_positions` 到 cache；其验证仅要求 `source_frame_id` 严格递增，未要求相邻 source/timeline frame 连续，故没有阻止两个 component 在 cache 索引中相邻。

## Impact and boundary

这是 selection-to-cache 的 transition 连续性缺口，不是官方原始帧缺失、MANO correspondence 或模型预测问题。对目前仅使用 GRAB/ARCTIC 的二域训练无影响。把 component 拆为独立序列，或在 loader 中过滤 timeline gap transition，会改变 OakInk2 sequence/transition/split 语义，属于 L2；本次未改 selection、cache、loader 或训练。

## Verification and rollback

核对 selection index 的 `selected_timeline_positions`、`contact_components`、`selected_components` 及 MANO producer 的写入/验证逻辑。回滚仅删除本 Activity 与索引条目。
