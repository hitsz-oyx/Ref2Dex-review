# Inspire 导出中断与 NAS / 本地容量诊断

- timestamp: `2026-09-20T05:41:37Z`
- activity_id: `ACT-20260920-CMV2-INSPIRE-EXPORT-INTERRUPTION`
- Task: `ObjectInteractionCmv2`
- work_version: `V1.11.1`
- base_commit: `b9acd32b0bc7aa7aca73e1a6b3f0a512cbf8dadb`
- branch: `cmv2`
- mode: `inspect`；记录阶段为 `change / L0`
- approval: `L0-auto`；用户要求排查反复停止、磁盘容量和 NAS 写入位置，不包含重启或清理授权。
- scope: 核对两次 GRAB Inspire 启动命令、进程、manifest、日志、挂载和容量；只增加本 Activity 及索引。
- research semantics: unchanged
- protected: 既有暂存修改、代码、配置、原始数据、所有 cache / partial / manifest、checkpoint、其他用户进程和 viewer 均不修改。
- run_id: `cmv2_v145_grab_inspire_full_20260918T022000Z`；`cmv2_v111a_grab_inspire_resume_knnfb2_20260920T024809Z`
- run_status: `UNKNOWN`（两次进程均已不存在、未完成；具体退出原因未知；磁盘上的旧 `STARTED` / `RUNNING` 状态保留，不回写）。
- scientific conclusion: `INCONCLUSIVE`；本次是运行诊断，不产生模型效果结论。

## 容量与实际路径

- `data/processed_data` 解析到 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data`，对应 NFS 挂载 `192.168.110.40:/volume2/ugreen_nas`。
- 两次 GRAB 输出均为 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/grab_inspire_20260918T022000Z`；原队列日志也位于 NAS。
- 本次 `df -hT`：NAS 总量约 `71 TiB`、已用 `20 TiB`、可用 `51 TiB`、使用率 `29%`；NAS inode 使用率约 `1%`。
- 本地 `/`、仓库和 `/tmp` 共用 btrfs 系统盘：总量约 `3.7 TiB`、可用约 `2.9 GiB`、显示 `100%`。`btrfs filesystem usage -b /` 显示 data 使用率 `99.92%`，估计空闲约 `3,018,231,808 bytes`；部分详细信息需 root，未提权。
- 当前本地容量危险，但这是当前快照，不是中断时容量证据。不能据此认定磁盘满导致退出，也不能以 NAS 空闲排除所有本地临时文件或会话日志风险。

## 两次中断的共同证据

| 项目 | 原队列 | 最近续跑 |
| --- | --- | --- |
| 启动 UTC | `2026-09-18 02:19:04` | `2026-09-20 02:48:09` |
| 启动方式 | `exec_command` 内直接运行 Python queue，经 `tee` 写 launcher log | `exec_command` 内直接运行 Python producer，stdout 连接会话工具 |
| producer / queue | producer PID `897162`，queue PID `897032` | producer PID `3277041` |
| 最后 manifest 写入 UTC | `2026-09-18 04:23:57.090921154` | `2026-09-20 02:53:25.218613284` |
| 完成计数 | `972/1335` sequence，`302724` frame | 本次已重新验证 `213/1335` sequence，`75137` frame |
| 会话实例边界 | 原 Codex PID `882545` 的本地日志结束于 `04:23:49`，后续会话文件开始于 `04:24:32` | 原 Codex PID `2817643` 日志结束于 `02:53:23`；新实例 PID `3281546` 从 `02:53:31` 开始 |

两个实际启动命令均没有独立 supervisor / `tmux` / `systemd` / 脱离会话的启动封装。两次停写时间都与对应会话实例切换接近，因此**会话生命周期导致整个长任务中断是目前最强的解释，但仍是推断**；没有退出码或信号记录，不能断言是特定信号、用户手动关闭、工具清理还是 OOM。

- 原队列 `grab_inspire_20270_knn32.log` 最后一条是第 `972` 条正常完成，未找到 `Traceback`、`No space`、`Killed`、`FAILED` 或其他 error 行；`queue.launcher.log` 为零字节。
- 原 `queue_state.json` 停留在 `RUNNING`，没有 `exit_code`。按 [queue 实现](../../tools/data/run_cache_queue_v1_4_4.py)，若子进程退出且 queue 正常接管，应写 `FAILED` 与退出码；当前形态更符合 queue 本身也未能完成退出处理，但不是独立的死亡原因证明。
- [producer 实现](../../tools/data/build_stage4_inspire_highres_v1_4.py) 在 `--resume` 时先读取并验证已完成 geometry，再更新 NAS 上的 run manifest。最近停在第 213 条，尚未超过原有 972 条；不能将 213 解释为磁盘仅保留 213 条或新导出 213 条。
- 代码可见的本地临时写入是 converter 初始化时的临时 URDF；主要数组、`.partial` 和 manifest 均在 NAS。最近中断仍在已有 cache 的验证阶段，不存在将大量新数组导出到系统盘的证据。
- 当前磁盘确有 `972` 个 geometry manifest，仍缺 `363` 条及全量最终 index / cache manifest。当前 queue 和 Inspire producer 均无存活进程。
- `dmesg`、`/var/log/kern.log.1` 和 `/var/log/syslog` 读取被拒绝，`journalctl -k` 无完整系统日志可见性；不能排除历史 OOM、内核 I/O 错误或其他外部终止。主机启动时间为 `2026-08-11 02:48:55`，不是近期整机重启。

## 证据入口与命令

- 原启动及完整参数：`/home/wbcd/workspace/oyx_ws/.codex_oyx/sessions/2026/09/18/rollout-2026-09-18T02-08-18-01a0b245-90ce-7f53-abd7-5d03910d965b.jsonl` 中 `2026-09-18T02:19:04.583Z` 的工具调用。
- 最近启动及完整参数：`/home/wbcd/workspace/oyx_ws/.codex_oyx_frj/sessions/2026/09/18/rollout-2026-09-18T04-24-32-01a0b2c2-49f2-7870-b63c-cb765f7fd195.jsonl` 中 `2026-09-20T02:48:09.322Z` 的工具调用。
- 原命令入口：`python -u -m src.task.ObjectInteractionCmv2.tools.data.run_cache_queue_v1_4_4 --queue <queue-root>/queue.json --state-root <queue-root> --poll-seconds 0`，经 `tee` 写 launcher log。
- 最近命令入口：`python -u -m src.task.ObjectInteractionCmv2.tools.data.build_stage4_inspire_highres_v1_4 --domain grab --device cuda:2 --knn-frame-batch 2 --knn-object-chunk 512 --resume`，完整 source / output / dex / run-id 参数见上述原始调用。
- queue-root：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_queue_20260918T022000Z`，含 `queue.json`、`queue_state.json`、`queue.launcher.log`、`grab_inspire_20270_knn32.log`。
- 两个 run manifest 为上述输出根下 `run_manifest_<run_id>.json`；无训练 step / epoch、best metric、checkpoint、metrics.jsonl 或 train.log。
- 会话实例时间来自两个对应 Codex home 下的 `logs_2.sqlite`，以 SQLite `mode=ro` 查询 `logs` 的 `ts`、`process_uuid`、`target`；不修改数据库。

## 验证、后续与回滚

- verification: 已交叉检查 `df` / `findmnt` / btrfs、进程、manifest、geometry 数量、日志、历史启动调用与只读 SQLite 实例边界；`git diff --check` 与 `git diff --cached --check` 通过。用 `/home/wbcd/miniconda3/envs/openpi/bin/python tools/verify.py --changed` 执行统一入口：`9 passed, 1 failed`，失败为既有 `test_tracked_task_sources_and_configs_use_work_version`，部分 Task 源码 / 配置仍含旧版本字段；本次未修改这些文件，未发现本次文档链接错误，不声称 VERIFY PASS。
- next step: 若用户确认恢复，先固定独立持久托管方案、NAS stdout/stderr、退出码 / 信号记录及本地空间预检，再按 [V1.11a](../plan/V1.11a.md) 核对提交、输入和 cache 后续跑；不能简单把旧 manifest 的 `STARTED` 当作任务仍在运行。
- 本次不运行清理、不重启或停止任务，不修改 run manifest；无需数据回滚。文档回滚仅移除本 Activity 和新增索引行，保留所有既有暂存内容。
