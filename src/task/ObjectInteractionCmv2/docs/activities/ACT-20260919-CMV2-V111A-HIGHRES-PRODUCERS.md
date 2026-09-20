# V1.11a 高分辨率 cache producer 恢复准备

- timestamp: 2026-09-19 16:25:00 +0000
- activity_id: ACT-20260919-CMV2-V111A-HIGHRES-PRODUCERS
- work_version: V1.11.1
- mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认恢复 OakInk2 MANO、GRAB Inspire 与 ARCTIC Inspire，并明确要求高分辨率 cache。
- branch: cmv2
- base_commit: 2377c45

## Changes

- 新增 [V1.11a 最终计划](../plan/V1.11a.md)。
- OakInk2 MANO 与 Stage4 Inspire producer 的新写 geometry/run/index/cache manifest 从历史 `modification_version` 迁移为 `work_version: V1.11.1`。
- 未改动 schema 名、数组 contract、MANO 4096 点、Inspire 20270 点、KNN32、坐标、split 或 retargeting。

## Verification

- 两个 producer 的 Python syntax compile 与 `--help` CLI 均通过。
- `git diff --check` 通过。

## Rollback

反转本次代码提交可恢复历史 manifest 字段；不触及 data/cache 实体。后续运行仅以 `--resume` 复用已完成原子段。
