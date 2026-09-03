# correspondence_ptv3_v2 活动记录

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-09-01
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [架构](architecture_log.md)、[仓库记忆](repo_memory.md)、[实验](experiment_log.md)、[历史修改](modification_log.md)、[当前指导](../指导/V1.md)

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
