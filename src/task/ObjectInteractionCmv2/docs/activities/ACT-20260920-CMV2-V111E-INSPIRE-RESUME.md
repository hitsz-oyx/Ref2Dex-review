# V1.11e Inspire 续跑

- timestamp: `2026-09-20T06:00:18Z`
- activity_id: `ACT-20260920-CMV2-V111E-INSPIRE-RESUME`
- Task: `ObjectInteractionCmv2`
- work_version: `V1.11.1`
- base_commit: `b9acd32b0bc7aa7aca73e1a6b3f0a512cbf8dadb`
- branch: `cmv2`
- mode: `run`
- change_level: `L3`
- approval: `user-approved`；用户同意执行 `plan/V1.11e.md`。
- scope: 将已核验的空 `bowl_lift.partial` 原样改名隔离；以 user systemd 托管 GRAB Inspire 恢复，GRAB 成功后串行导出 ARCTIC Inspire；不启动 OakInk2 job。
- protected: 已完成 geometry、旧 queue、旧 manifest、输入、代码语义、其他 GPU 进程和本地未提交修改。
- research semantics: unchanged
- run_id: `cmv2_v111e_grab_inspire_resume_20260920T055000Z`; `cmv2_v111e_arctic_inspire_full_20260920T055000Z`
- run_status: `RUNNING`（user systemd unit 已启动，producer manifest 已写入）
- conclusion: `INCONCLUSIVE`；运行尚未终态，不构成科学效果结论。

## 固定合同

- 输入 index：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json`。
- GRAB source/output：`oicm_v1_4_raw/grab_mano_30hz` → `oicm_v1_4_5_highres_full/grab_inspire_20260918T022000Z`。
- ARCTIC source/output：`oicm_v1_4_raw/arctic_mano_30hz` → `oicm_v1_4_5_highres_full/arctic_inspire_20260920T055000Z`。
- `cuda:2`、`--knn-frame-batch 2`、`--knn-object-chunk 512`、`--resume`；每侧 `10,135` Inspire 点，双侧 `20,270`，KNN32。
- launcher、stdout/stderr、TMPDIR 和状态写入 NAS 的 `data/processed_data/oicmv2_inspire_resume/v1/<run_id>/`；不依赖当前 Codex shell。

## 验证与回滚

- 启动前核对 GPU2 至少 20 GiB 空闲、NAS 可写、残留目录只有空 `geometry/`、隔离目标不存在、无同类进程。
- 启动后核对 systemd user unit、MainPID、launcher status、producer manifest 更新时间和失败列表。
- GRAB 未返回 0 时不启动 ARCTIC；两域都只有 producer `COMPLETED` 且 index/cache manifest 完整才接受完成。
- 停止本次 unit 即可回滚运行；保留 NAS 日志、manifest、geometry 和隔离残留，不删除数据。

## 启动证据

- unit：`ref2dex-cmv2-inspire-v111e.service`；invocation：`1722ca27fc43439293ba1fcab55c4f44`；MainPID：`3429046`；cgroup 为 user systemd service，不属于当前 Codex shell。
- 启动时 GPU2 空闲约 `48,509 MiB`，NAS 可用约 `51 TiB`；没有同类旧 producer/queue 进程。
- 空残留已从 `bowl_lift.partial` 原样改名为 `bowl_lift.partial.interrupted_20260920T055000Z`，隔离目录仅含空 `geometry/` 子目录。
- 启动后 GRAB manifest `cmv2_v111e_grab_inspire_resume_20260920T055000Z` 已写入，最近核验为 `304/1335` sequence、`99,288` frame、`failures=[]`；这包含 `--resume` 的既有段验证，不等于新增导出数量。
