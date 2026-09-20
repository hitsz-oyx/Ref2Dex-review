# V1.11f Inspire 并行调度

- timestamp: `2026-09-20T06:50:00Z`
- activity_id: `ACT-20260920-CMV2-V111F-INSPIRE-PARALLEL`
- Task: `ObjectInteractionCmv2`
- work_version：`V1.11.1`
- base_commit：`b9acd32b0bc7aa7aca73e1a6b3f0a512cbf8dadb`
- branch：`cmv2`
- mode：`run`
- change_level：`L3`
- approval：`user-approved`；用户确认将 Inspire 处理并行化。
- scope：GPU2 保持 GRAB；修改当前 launcher 使其不启动同一输出根的 ARCTIC；GPU1 独立托管 ARCTIC。
- protected：研究语义、输入、split、坐标/单位、点数、KNN32、已完成 cache、OakInk2 cache、旧 queue 和其他进程。
- research semantics：unchanged
- run_id：`cmv2_v111e_grab_inspire_resume_20260920T055000Z`；`cmv2_v111f_arctic_inspire_parallel_20260920T065000Z`
- run_status：`UNKNOWN`（并行服务预检期间 GRAB 已完成，原服务已先启动 ARCTIC；GPU1 未启动第二个 producer）
- conclusion：`INCONCLUSIVE`；运行中不构成科学效果结论。

## 启动证据

- GRAB service：`ref2dex-cmv2-inspire-v111e.service`，GPU2，继续写既有 GRAB output root。
- ARCTIC service：`ref2dex-cmv2-arctic-v111f.service`，GPU1，独立写 `arctic_inspire_20260920T055000Z`。
- 两个 service 的 launcher、日志、TMPDIR 和 status 均写入 NAS；不依赖当前 Codex shell。
- 两个服务不共享目标 output root，避免 parallel producer 双写；GRAB launcher 的 ARCTIC 分支由并行模式显式跳过。

## 实际结果

- 预检发现 GPU2 的 GRAB 已于 `2026-09-20T06:59:15Z` 完成 `1335/1335`；原 `v111e` launcher 随即在 GPU2 启动 ARCTIC，故不存在可安全并行的 GRAB 工作段。
- 为避免同一 ARCTIC output root 双写，GPU1 的 `ref2dex-cmv2-arctic-v111f.service` 未启动；GPU1 运行目录只保留预检 launcher/config，不产生 cache。
- 当前实际 ARCTIC run 为 `cmv2_v111e_arctic_inspire_full_20260920T055000Z`，GPU2，`17/301` sequence、`12169` frame、`failures=[]`；并行请求不改变实际数据语义。
