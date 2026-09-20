# 五组混合双卡训练准备与 OakInk2 语义门禁

- timestamp：`2026-09-20T08:26:06+00:00`
- activity_id：`ACT-20260920-CMV2-V111G-MIXED-DDP-GATE`
- Task / work_version：`ObjectInteractionCmv2` / `V1.11.1`
- base_commit：`b9acd32b0bc7aa7aca73e1a6b3f0a512cbf8dadb`；branch：`cmv2`；无新 commit。
- scope / impact：Task-only 五组 split、adapter、DDP入口与验证，L2/L3。
- approval：用户连续确认五组、MANO/Inspire与组内比例、GPU1+3、每卡64、16epochs、旧best模型权重初始化、stride1..3、分组验证及OakInk2 10% holdout；后续已确认 [V1.11h FINAL](../plan/V1.11h.md)。
- run_status：本节记录的初始整对象 smoke `FAILED`；后续状态与部件 adapter、通过的 smoke 及正式 run 见文末“Smoke 与正式 run 记录”。
- scientific conclusion：混合训练效果 `INCONCLUSIVE`；OakInk2 全部按单刚体接入的假设不成立。

## 已完成与保护

新增 `mixed_training.py`、`train_mixed_articulated_ddp.py`、`configs/active/mixed_articulated_v1_11g_ddp.yaml` 和定向测试。
独立源码快照、配置/input SHA256、systemd user service、NAS TMPDIR/日志/输出、strict模型权重加载与新optimizer已实现。
源码快照包含 base commit 之外未提交改动，不能把运行描述为纯 HEAD 实现。
已有14个 staged 路径未撤销/覆盖；新增文档索引增量保留于工作区。未改模型、共享 `src/base`、导出器、原始数据/cache/best或指导。
GPU0无关进程、GPU2的 `ref2dex-cmv2-inspire-v111e.service` 保持运行且未改动；ARCTIC Inspire未加入五组。

## 固定 split

run_id：`cmv2_v111g_split_20260920`；run_status：`COMPLETED`。
目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicmv2_mixed_training/v1/cmv2_v111g_split_20260920`。
入口：同目录 `run_manifest.json`、`config.json`、`index.json`、`cache_manifest.json`。
命令：`python -m src.task.ObjectInteractionCmv2.mixed_training --config src/task/ObjectInteractionCmv2/configs/active/mixed_articulated_v1_11g_ddp.yaml --run-id cmv2_v111g_split_20260920`。

| 组 | train片段 | val片段 | test片段 |
| --- | ---: | ---: | ---: |
| GRAB MANO | 1068 | 134 | 133 |
| ARCTIC MANO | 266 | 35 | 0 |
| OakInk2 MANO | 1625 | 224 | 0 |
| GRAB Inspire | 1068 | 134 | 133 |
| OakInk2 Inspire | 1625 | 224 | 0 |

OakInk2共472个原始录制，固定48个录制（10%向上取整）进入val；同录制的segment和两种手型共组。
GRAB/ARCTIC原split未改，原cache的物理split未改，test不参与模型选择。

## 运行与故障证据

初始化 checkpoint：`/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v111_articulated_random_init_stride1to3_formal_20260919T075300Z/best.pt`。
读取到旧epoch9、step72594；仅加载模型，新optimizer状态数为0。

- 准备失败 run：`cmv2_v111g_mixed_ddp_smoke_20260920T082156Z`，Python3.8的 `Path.parents` 不支持切片；在launch前失败，已修复为list切片，保留FAILED manifest。
- 实际 smoke run：`cmv2_v111g_mixed_ddp_smoke_20260920T082221Z`。
- 目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v111g_mixed_ddp_smoke_20260920T082221Z`。
- service：`ref2dex-cmv2-v111g-mixed-ddp-smoke-20260920T082221Z.service`，终态failed，退出码1。
- 命令：`python -m src.task.ObjectInteractionCmv2.train_mixed_articulated_ddp launch --config src/task/ObjectInteractionCmv2/configs/active/mixed_articulated_v1_11g_ddp.yaml --run-id cmv2_v111g_mixed_ddp_smoke_20260920T082221Z --smoke-steps 2`；实际systemd完整命令见manifest。
- 最后step0 / epoch1；best metric为null；**未生成latest.pt或best.pt**。
- 日志入口：同目录 `run_manifest.json`、`config.json`、`metrics.jsonl`、`train.log`、`service.log`、`rank0_error.json`、`rank1_error.json`。
- 拦截：首批样本读取时OakInk2 rigid pose/point replay超过0.2mm，未执行第一个训练forward/backward；不能宣称batch64显存/DDP梯度同步/完整15组validation已验证。

## 原因与只读核查

同目录 `oakink2_rigid_replay_diagnostic.json` 包含全1849×2份manifest部件计数与固定连续帧对诊断。
每种手型：1413单部件片段/250585帧；433双部件片段/128518帧；3三部件片段/488帧。
两种手型对应segment的部件身份列表完全一致。

示例：`oakink2/scene_01__A001++seq__07bb164dc3d3873d6389__2023-04-27-20-45-29/0000`；raw帧1006→1018，stride3，时间差0.09999943s，符合连续性保护。
4096点中首部件2073点，另一部件2023点。按缓存首部件参考位姿重建，首部件max约0.000062mm，另一部件max约3.580073mm。
分别刚体拟合，两部件max均低于0.00005mm；MANO/Inspire结果相同。此拟合只作诊断，未写作训练GT。

producer `_sample_object_pool` 将各部件独立位姿作用后的world points合并，返回的reference pose为 `poses[:,0]`；manifest明确标注 `single_root_articulated_world_points_with_reference_pose`。
因此“完整点云+单个刚体FK”不满足当前门禁，是本次适配假设错误；不能据此声称数据损坏、原始数据伪造或Inspire重定向错误。

## 验证与回滚

- `graspenv` Python3.8：初次新增及V1.5定向测试17 passed；补充非刚体replay拒绝回归后，Task测试集47 passed。
- `git diff --check`通过。
- 新增/相关7份文档的本地链接审计通过；原14个staged路径保留，未新增暂存或提交。
- `openpi`运行 `python tools/verify.py --changed`：9 passed / 1 failed；既有 `test_tracked_task_sources_and_configs_use_work_version` 因已跟踪文件保留历史版本字段失败，非本次新增字段；不声称VERIFY PASS。
- 正式训练启动门禁未通过，不自动忽略436片段、不关闭检查、不改变研究语义强行训练。
- 失败服务已退出，无需停其它服务。保留split/诊断/日志供复核；仅移除本次新文件或文档增量即可回到旧入口，不能覆盖用户staged内容。

## V1.11h 后续执行（进行中）

- timestamp：`2026-09-20T09:19:00+00:00`；approval：用户已确认按真实刚性部件训练，`V1.11h` 已定稿为 `FINAL`。
- 适配器 run_id：`cmv2_v111h_oakink2_parts_20260920T091403Z`；NAS 根目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicmv2_oakink2_parts/v1/cmv2_v111h_oakink2_parts_20260920T091403Z`。
- 该 adapter 已 `COMPLETED`，使用过 8 个 CPU worker 的独立 user systemd 服务 `ref2dex-cmv2-v111h-oakink2-parts-20260920T091403Z.service`；目标 1849 个 segment，引用既有 MANO/Inspire 手流，不覆盖原 cache。
- 已完成记录均通过 raw component SE(3)、Stage3/raw pose 和 MANO/Inspire canonical pool replay 门禁；当前失败数为 0。适配器完成后先做 GPU1+GPU3、每卡64的真实双卡 smoke，再决定正式训练。

## Smoke 与正式 run 记录

- 部件 adapter：`cmv2_v111h_oakink2_parts_20260920T091403Z` 已于 `2026-09-20T09:53:35+00:00` 完成；1849 个 segment、2288 个部件、`bad_count=0`。
- smoke：`cmv2_v111h_mixed_ddp_smoke_20260920T104846Z` 为 `COMPLETED`；2 steps、每卡64、GPU1+3，15 组验证完成，checkpoint roundtrip 与 rank 参数一致性通过。
- 首次正式 run：`cmv2_v111h_mixed_ddp_formal_20260920T105004Z` 在全量 memmap 加载阶段因 `OSError(24, Too many open files)` 失败，未执行训练 step、未产生 checkpoint；NAS 空间与 GPU 显存均不是故障原因。
- 修复：仅将 Task-local systemd `LimitNOFILE` 从 `65536` 提升到 `262144`，保持研究变量不变；`cmv2_v111h_mixed_ddp_formal_20260920T111159Z_nofile262k` 曾进入训练至 step160，后按用户要求受控停止，manifest 为 `STOPPED`。
- 该旧 run 于 `2026-09-20T11:32:17+00:00` 收到 `SIGTERM`，未覆盖旧 checkpoint；GPU1/3 上既有进程未触碰。
- 修复后的 Task 测试集为 `48 passed`，`git diff --check` 通过；formal run 的 source snapshot 已包含 `LimitNOFILE=262144`。

## V1.11i 资源切换（已批准，待 smoke）

- 用户于 `2026-09-20` 要求停止 GPU1+GPU3，改用物理 GPU0+GPU2；该资源切换已写入 [V1.11i FINAL](../plan/V1.11i.md)，不覆盖 V1.11g/h。
- 新配置：`configs/active/mixed_articulated_v1_11i_ddp.yaml`，显式 `physical_gpus=[0,2]`、`LimitNOFILE=262144`；数据、split、batch、模型、checkpoint 和步数保持不变。
- 启动前观测 GPU0 约 39 GiB free、GPU2 约 48 GiB free；GPU0 上既有进程保留，GPU2 当前无 compute process。下一步先在 0+2 运行真实 2-step smoke。
- `cmv2_v111i_mixed_ddp_smoke_20260920T114127Z` 已 `COMPLETED`：GPU0+GPU2、每卡64、2 steps、15 组验证、checkpoint roundtrip 与 rank parity 均通过；正式 run 可在同一配置下启动。
- 新正式 run：`cmv2_v111i_mixed_ddp_formal_20260920T114242Z` 已在 `2026-09-20T11:44:36+00:00` 进入训练；manifest 为 `RUNNING`，`steps_per_epoch=11706`、`planned_steps=187296`、`physical_gpus=[0,2]`。截至 step60 无错误，单卡峰值 allocated 约 `7.49 GiB`、reserved 约 `8.76 GiB`；当前约 `0.67 s/step`，仅作运行时间估计。
- 已越过 checkpoint 闸门 `step=200`，`latest.pt` 已写入 NAS，run 仍 `RUNNING`、无 error；checkpoint 写入耗时会计入后续实际 ETA。
