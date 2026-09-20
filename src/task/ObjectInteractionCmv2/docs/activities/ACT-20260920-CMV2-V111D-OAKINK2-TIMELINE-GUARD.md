# V1.11d OakInk2 transition 时间连续性保护

- timestamp: 2026-09-20 +0000
- activity_id: ACT-20260920-CMV2-V111D-OAKINK2-TIMELINE-GUARD
- work_version: V1.11.1
- mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认按 `frame_time` loader continuity guard 实施。
- branch: cmv2
- base_commit: a45381c
- scope: OakInk2 loader row 构建的短 stride 时间连续性保护；不重导 cache。

## Change

`ThreeDomainTransitions` 继续用既有稳定 seed 选择每个样本的实际 stride。仅当 OakInk2 的 `frame_time[t + stride] - frame_time[t]` 在 `1e-4 s` 绝对容差内等于 `stride / effective_fps` 时保留该 row；并暴露 `dropped_timeline_transitions`。GRAB/ARCTIC 不进入这条新 guard。

## Verification

- `py_compile src/task/ObjectInteractionCmv2/multi_domain.py`：通过。
- `pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_4_three_domain.py src/task/ObjectInteractionCmv2/tests/test_v1_4_4_two_domain_split.py`：9 passed。合成单 gap 测试确认 stride 1/2/3 分别只过滤 1/2/3 个跨 gap row。
- 对完成态 OakInk2 MANO cache 的全量 `frame_time` 只读核对：stride 1/2/3 分别过滤 `11/18/25` 个 transition，保留 `377731/375875/374019` 个；不写 cache。

## Protected and rollback

selection、所有原始/派生 cache、split、坐标、GT、KNN、模型、loss、checkpoint 和 GRAB/ARCTIC row 集合均未修改。反转本次 loader、测试、V1.11d 计划与本 Activity 的提交即可恢复原 OakInk2 transition 集合；不需要删除任何数据。
