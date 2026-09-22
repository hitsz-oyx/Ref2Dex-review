# OakInk2 MANO cache 点流最大值时序断点诊断

- timestamp: 2026-09-20 +0000
- activity_id: ACT-20260920-CMV2-OAKINK2-MANO-TIMELINE-GAP
- work_version: V1.11.1
- mode: inspect
- change_level: L0
- branch: cmv2
- git_commit: 8570070
- scope: 对 OakInk2 MANO 全量点流统计中的最大值做只读定位。

## Finding

stride 1/2/3 的最大点流均来自同一序列 `oakink2/scene_01__A003++seq__de088ef0fd274c39348d__2023-04-18-10-11-07/0353`、右手 point 2411、cache frame 56。该点从 frame 56 至 57 的位移为 201.2795 mm，但相邻 cache 索引帧的 `frame_time` 从 75.133331 s 跳至 76.066666 s（0.933334 s），`source_frame_id` 从 9025 跳至 9137；不是正常的 1/30 s transition。

全量 377,742 个相邻 transition 中，11 个的 `frame_time` 间隔超过 `1/30 + 1e-4` s，5 个超过 0.1 s，仅 1 个超过 0.5 s；中位数为 0.0333328 s、P99 为 0.0333405 s、最大值为 0.9333344 s。

## Interpretation

最大点流由 source timeline gap 被保留在连续 cache 索引中造成；它不是 MANO 点对应错误，也不是模型预测量。此类 gap 仅占 transition 的约 0.0029%，对全量 point-micro 均值影响很小，但若 OakInk2 后续加入训练或正式评估，应先确认是否以 timeline gap 切分/过滤 transition。这将改变 transition/split 语义，需另行 L2 审批，故本次不修改 cache 或 loader。

## Verification and rollback

全量只读扫描定位 sequence、point、相邻 frame time 与 source ID；临时 `/tmp` 日志已移除。回滚仅删除本 Activity 与索引条目。
