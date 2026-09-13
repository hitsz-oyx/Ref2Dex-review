# correspondence_ptv3_v2 活动记录

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-09-06
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [架构](architecture_log.md)、[仓库记忆](repo_memory.md)、[实验](experiment_log.md)、[历史修改](modification_log.md)、[当前指导](../指导/V2.md)

## 2026-09-01 21:12:26 +0800 — V1.2.15 活动记录切换

- activity_id: ACT-20260901-211226-CORRESPONDENCE
- timestamp: 2026-09-01 21:12:26 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 根级治理方案获用户直接批准；本 Task 仅迁移日志入口，不改变 correspondence 研究语义
- skills_used: research-change-control, research-experiment-workflow
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: root/governance mirror; task:correspondence_ptv3_v2 活动日志入口；不涉及模型、数据、GT、坐标、split、checkpoint 或训练变量

**文件**
- [`activity_log.md`](activity_log.md) — 从切换点开始作为本 Task 活动时间线，保留原 `status_log.md` 历史内容。
- [`repo_memory.md`](repo_memory.md) — 更新活动入口链接。
- [`architecture_log.md`](architecture_log.md) — 收窄架构记录头部为更新时间。
- [`../../config.py`](../../config.py) — 将当前运行元数据改为 `modification_version`。
- [`../../../../../docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 链接根级当前版本指针。

**原因**
统一本 Task 的运行反馈、产物导航和治理事件记录；历史 `modification_log.md` 继续只读保留。

**验证**
- 根级文档/版本指针校验和全量回归：`311 passed, 3 skipped`；本次仅为工程治理迁移，`conclusion: N/A`。

## 当前操作 — V1.2.12：撤出 Task-local Component/data/registry（2026-09-01）

- 已删除 `src/task/correspondence_ptv3_v2/components/`、`data/`、`registry/` 以及根级 `components/ref2dex/correspondence_ptv3_v2` 兼容入口。
- 配置不再声明 Component 清单；数据和 cache 继续直接使用根级 `data/`、`dataset/` 与 `third_party/`，校准 JSON 仍属于 Task 的研究配置输入。
- 真实数据、cache、checkpoint、output 和既有评估产物未移动、删除或改写。
- 计划：[`docs/plan/V1.md`](../plan/V1.md)（final）；历史 Component 校准记录保留，不作为当前入口。

## 历史操作 — V1.2.1：组件声明已校准（2026-08-31）

- 历史记录：Task 级 component 曾作为只读任务入口，组件版本从 `2.1.0` 校准为 `2.1.1`；该入口已在 V1.2.12 撤出。
- manifest 现在与 live `StaticHOCPTv3V2.forward()` 的预测端口一致：random cross-edge、contact-aux cross-edge，以及可选 dense hand-contact head。
- 本次只修改声明清单、Task 选择和合同测试；训练代码、数据、GT、坐标系和 checkpoint 未改变。
- 证据：`researchctl check-task-config`、manifest entrypoint check 和定向测试均通过。

## 2026-09-03 22:15:00 +0800 — 合并前迁移本地状态快照

- activity_id: ACT-20260903-221500-CORRESPONDENCE-MERGE
- type: merge / documentation / operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求以远端 `oyx` 为主合并，并将本地 status 内容迁移到 activity 记录
- branch: oyx
- scope: task:correspondence_ptv3_v2 activity；本地 `status_log.md` 已按远端治理结构移除

**本地状态摘要**

- OakInk2 object-centered Stage3、object-disjoint train/val 软链接以及 ARCTIC/HRDexDB 四手型 object-frame cache 均已完成。
- 七域训练配置包含 GRAB、ARCTIC、HRDexDB human/Inspire DFTP/Inspire F1/Allegro V5 和 OakInk2，采用 object-centered、512 点、4:4:2 扰动和每 5000 step 保存。
- 训练已修复 worker 导入冲突并在 GPU 2/3 正常运行；最近状态约 step 143,700 / 21,030,400，checkpoint 为 step 140,000，日志 ETA 约 4441 小时。
- data-wait ratio 约 `0.00027`，当前主要耗时来自模型计算和总 step 数，而非数据等待。

**验证**

- OakInk2 train/val manifest：3425/1317 文件、106 个对象；训练帧约 672.97 万。
- object-frame cache manifest 已完成；机器人重建 fallback 已通过 `xarm_inspire_f1_right.urdf` 解析 smoke。

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-09-01
- last_verified: 2026-09-01
- related: [架构](architecture_log.md) / [仓库记忆](repo_memory.md) / [实验](experiment_log.md) / [修改](modification_log.md) / [当前指导](../指导/V1.md)

## 当前状态

- 当前阶段 / 指导: feature 分支内容已合入 `oyx` 并通过回归测试；ARCTIC 外部评估仍遵循 V1。
- 当前进行中: 当前未发现本 Task 训练或 ARCTIC Stage 2/3 导出进程。三域 mixed 曾从 step 22748 恢复，但目前只保留 checkpoint/日志状态；HOCap subject_1 Stage 3 和首次纯 GRAB 外部测试已完成。
- 最近可靠结论: 协议 E 已用统一 100% 的 10 mm hand perturb 评估 noPCA latest、H80 latest/best、H50 best 和历史 GRAB+ContactPose 两域 mixed latest。H80 best 的 hand-only 最终 QFL / recovery 最好；H50 best 的三条件等权 perturbed QFL 最低；历史两域 mixed 的 Balanced perturbed QFL 约 0.000973、recovery 0.1672，明显落后。旧协议 D 的 hand-only / hand+object 因 evaluator 继承训练门控概率而标记为 `INVALID_IMPLEMENTATION`，object-only 不受影响。三域 mixed 续训 W&B run ID 为 `d5yvor4z`。HOCap subject_1 已用纯 GRAB noPCA latest 跑完 28 文件 / 23,896 帧：clean random QFL 0.00016607，10°/10 mm object perturb 后 0.00020392，recovery Brier 0.5267。
- 阻塞 / 风险: 协议 E 仍仅覆盖 s01 min11，且 checkpoint 选择、global batch 与样本曝光量不统一。旧 run 日志 step 27980 后中断，但最近完整 checkpoint 仅到 step 22748，因此 5232 step 未保留为模型状态。全量 ARCTIC Stage 2/3 完成状态仍需核验。HOCap 当前只覆盖 subject_1，接触距离由几何派生且结果为 micro 聚合，不等同于正式多主体 benchmark。
- 下一步: 核验全量 ARCTIC Stage 2/3 产物，决定是否按原配置恢复 mixed 训练。HOCap 仍只作外部测试，不加入训练。
- 证据与相关文档: 合并后全量 pytest `282 passed, 3 skipped`；[EXP-012](experiment_log.md#exp-012--纯-grab-nopca-在-hocap-subject_1-的外部测试)；[EXP-011](experiment_log.md#exp-011--协议-e10-mm-mano-min11-三条件评估)；HOCap 结果 `output/research/hocap_subject1_grab_nopca_20260824/full_object_only.json`；[协议 E 结果](../../result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md)。

## 2026-09-03 23:49:07 +0800 — 停止七域 object-centered 训练

- activity_id: ACT-20260903-234907-CORRESPONDENCE-STOP
- timestamp: 2026-09-03 23:49:07 +0800
- modification_version: V1.2.12
- type: training / operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求先停止当前训练
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: a7c5b04f7f6797c78e232c2d0425f3a9694f7078
- worktree_dirty_before_record: false
- run_id: correspondence_ptv3_v2_grab_arctic_hrdexdb_oakink2_object_centered_20260902_065152
- run_status: STOPPED
- tmux_session: expanded_object_centered_train_20260902_rerun
- devices: CUDA_VISIBLE_DEVICES=2,3

**终止与恢复证据**

- 向目标 tmux 会话发送一次 `SIGINT`，两个 DDP rank 及 DataLoader worker 均已退出；未影响其他 tmux 会话或 GPU 任务。
- 停止前日志最后完整训练记录约为 step 156240 / epoch 1。
- 最新可恢复 checkpoint：`outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_grab_arctic_hrdexdb_oakink2_object_centered_20260902_065152/checkpoints/step_000155000_epoch_000001.pt`；`latest.pt` 与其同时生成。
- 运行日志：`/mnt/ugreen_nas/storage/Ref2Dex_storage/logs/expanded_object_centered_train_20260902_rerun.log`。
- conclusion: N/A（本条仅记录运行终止，不形成科研结论）。

## 2026-09-04 00:05:00 +0800 — V2 配置与 mask 契约完成 smoke

- activity_id: ACT-20260904-000500-CORRESPONDENCE-V2-SMOKE
- timestamp: 2026-09-04 00:05:00 +0800
- modification_version: V1.2.16
- type: implementation / validation
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认六项 V2 训练语义，并允许在改动过大时新建版本；本次保持原 Task 接口兼容
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: a7c5b04f7f6797c78e232c2d0425f3a9694f7078
- run_status: READY_FOR_TRAINING
- config: `configs/mixed_grab_hrdexdb_object_centered_5cm.yaml`

**证据**

- 配置成功加载；五个 NAS 域均可索引，训练样本计数为 GRAB 295243、HRDexDB human 73025、Inspire DFTP 280897、Inspire F1 179911、Allegro V5 126630。
- weighted sampler 解析为 `[0.5, 0.125, 0.125, 0.125, 0.125]`，batch size 40 的 local quotas 为 `[20, 5, 5, 5, 5]`。
- 真实首 batch shape 为 points `(40, 2562, 3)`、point-valid `(40, 2562)`、hand-valid `(40, 1538)`、object points `1024`；首批手点有效数示例为 932–1223。
- correspondence 定向测试：`59 passed`；全仓库集合测试受环境缺少 `viser`/`pytorch3d` 阻塞，非本次改动引入。
- conclusion: smoke 通过，尚未启动完整训练。

## 2026-09-04 00:11:36 +0800 — 启动 V2 正式训练

- activity_id: ACT-20260904-001136-CORRESPONDENCE-V2-TRAIN
- timestamp: 2026-09-04 00:11:36 +0800
- modification_version: V1.2.16
- type: training / operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认 V2 六项配置并明确要求继续训练
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: a7c5b04f7f6797c78e232c2d0425f3a9694f7078
- run_status: RUNNING
- run_id: correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136
- tmux_session: grab_hrdexdb_5cm_object_centered_train_20260904
- devices: CUDA_VISIBLE_DEVICES=2,3
- output_dir: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136`
- log: `/mnt/ugreen_nas/storage/Ref2Dex_storage/logs/grab_hrdexdb_5cm_object_centered_train_20260904.log`
- train_setup: per-device batch 40、global batch 80、total steps 1404500、warmup 42135、每 5000 step 保存

**启动前证据**

- 1-step smoke 已完成并清理残留 smoke 进程；正式训练仅占用 GPU 2/3。
- 正式日志已完成 W&B 初始化和 MANO layer 初始化，两个 DDP rank 显存约 43 GB，未见 OOM 或异常退出。

## 2026-09-04 16:49:23 +0800 — contact-aux 分支隔离 benchmark

- activity_id: ACT-20260904-164923-CORRESPONDENCE-CONTACT-AUX-BENCH
- timestamp: 2026-09-04 16:49:23 +0800
- modification_version: V1.2.16
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求在不改变正式训练的前提下测量关闭 contact-aux 的收益；仅使用 GPU 1 隔离合成 benchmark
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: a7c5b04f7f6797c78e232c2d0425f3a9694f7078
- worktree_dirty: true
- run_id: correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136
- run_status: RUNNING

**benchmark 证据**

- B=40、1024 object、1538 hand 的 `_build_contact_supervision_edges`：`20.322 ms`；空分支：`0.037 ms`；可避免 `20.285 ms`（该子阶段 99.8%）。
- 合成 B=8 的 StaticHOCPTv3V2 forward：带 80 条 contact-aux edge 为 `108.135 ms`，空 contact edge 为 `104.321 ms`，可避免 `3.814 ms`（forward 3.5%）。该比例不直接等同于完整训练 step，但方向与当前日志一致。
- 正式训练未停止、未改动；benchmark 使用 GPU 1，正式训练继续使用 GPU 2/3。
- conclusion: N/A（隔离性能诊断，不形成科研效果结论）。

## 2026-09-06 21:24:45 +0800 — 停止 V2 GRAB+HRDexDB 训练

- activity_id: ACT-20260906-212445-CORRESPONDENCE-V2-STOP
- timestamp: 2026-09-06 21:24:45 +0800
- modification_version: V1.2.16
- type: operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求停止当前训练
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: a7c5b04f7f6797c78e232c2d0425f3a9694f7078
- worktree_dirty: true
- run_id: correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136
- run_status: STOPPED
- tmux_session: grab_hrdexdb_5cm_object_centered_train_20260904
- scope: 仅停止该 run 的 DDP 进程和 tmux 会话；不修改 checkpoint、输出、配置或其他 GPU 进程
- command: `CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 src/task/correspondence_ptv3_v2/train.py --config src/task/correspondence_ptv3_v2/configs/mixed_grab_hrdexdb_object_centered_5cm.yaml`
- output: [运行目录](../../../../../outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136)；[run manifest](../../../../../outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136/run_manifest.json)；[最新 checkpoint](../../../../../outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136/checkpoints/step_000420000_epoch_000015.pt)

**原因**

按用户要求停止当前 V2 GRAB+HRDexDB 训练，并保留最近完整 checkpoint 供后续恢复或评估。

**验证**

- 正常发送一次 `SIGINT`，两个 DDP rank、DataLoader worker 和目标 tmux 会话均已退出；GPU 2/3 均恢复为 0 MiB、0% utilization。
- 停止前最后训练记录为 step 421350 / epoch 15，处于 epoch 边界验证阶段。
- 最新完整可恢复状态为 `step_000420000_epoch_000015.pt`，`latest.pt` 与其同时生成；停止后的约 1350 step 未形成 checkpoint。
- 终止日志：`/mnt/ugreen_nas/storage/Ref2Dex_storage/logs/grab_hrdexdb_5cm_object_centered_train_20260904.log`。
- conclusion: INCONCLUSIVE（人工停止；本次操作不解释模型效果）。

## 2026-09-04 15:42:43 +0800 — V2 训练耗时与 cache 诊断

- activity_id: ACT-20260904-154243-CORRESPONDENCE-V2-PERF-DIAG
- timestamp: 2026-09-04 15:42:43 +0800
- modification_version: V1.2.16
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户请求分析训练耗时和可预处理 cache；本次仅读检查，不改变运行变量
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: a7c5b04f7f6797c78e232c2d0425f3a9694f7078
- worktree_dirty: true
- run_id: correspondence_ptv3_v2_grab_hrdexdb_object_centered_5cm_20260903_161136
- run_status: RUNNING

**诊断证据**

- 最近 400 个训练 step 平均 `561.27 ms/step`，`data_wait` 平均 `0.3373 ms`，占比 `0.000601`；GPU 2/3 利用率约 95–100%。
- 当前约 step 93980；完整训练共 1404500 step，按当前计算速度仅优化器 step 就约 219 小时。每 5000 step checkpoint 间隔约 46–47 分钟。
- 每 epoch 28090 step；epoch 结束还会对 10 个 domain validation loader 做评估，边界日志显示单个 epoch 额外约十几分钟。
- 已有 uncompressed array cache 约 213 GB（GRAB 44G，HRDexDB 四域合计 169G）；NAS 挂载剩余约 52 TB。新增 cache 不会明显改善当前 step 速度。

**结论边界**

- 主要热点是模型中的 object-hand 全距离 `torch.cdist`（1024×1538）以及即使权重为 0 仍构造的 contact-aux 监督 `torch.cdist`；MANO/FK 扰动是次级候选。5 cm mask、1024 点随机索引和 NAS 读取均不是当前 data-wait 瓶颈。
- 建议优先做独立 benchmark 后再修改：关闭 contact-aux 分支、降低 validation 频率或改为按 step；不要预计算随机扰动结果，以免改变 4:4:2 数据语义。
- conclusion: N/A（性能诊断，不形成科研效果结论）。

## 2026-09-13 11:21:49 +0800 — 远端 oyx 合并并保留 V2 研究入口

- activity_id: ACT-20260913-112149-CORRESPONDENCE-MERGE
- timestamp: 2026-09-13 11:21:49 +0800
- modification_version: V1.2.16
- type: operation / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认远端优先但不删除本地 correspondence V2 版本
- branch: oyx
- base_commit: 0fbfcf5465a66bc07f582db91e355f3f1ee21d3b
- merge_target: origin/oyx @ 2fc7b921ddcc78103efb3dcc9958ba1c282cbcac
- run_status: COMPLETED

**结果**

- 保留本地 5cm object-centered 配置、V2 指导/计划、校准文件及研究脚本；远端新增内容按远端版本合入。
- 合并无未解决冲突；训练变量、数据和 checkpoint 未启动或改写。

**验证**

- correspondence V2 文件均可从合并结果访问；无冲突标记。
