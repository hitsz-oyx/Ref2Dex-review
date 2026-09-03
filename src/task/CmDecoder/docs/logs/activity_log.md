# CmDecoder 活动记录

- scope: task:CmDecoder
- last_updated: 2026-09-02
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [架构](architecture_log.md)、[实验](experiment_log.md)、[接手记忆](repo_memory.md)、[历史修改](modification_log.md)

## 2026-09-01 21:12:26 +0800 — V1.2.15 活动记录切换

- activity_id: ACT-20260901-211226-CMDECODER
- timestamp: 2026-09-01 21:12:26 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 根级治理方案获用户直接批准；本 Task 仅迁移日志入口，不改变 CmDecoder 研究语义
- skills_used: research-change-control, research-experiment-workflow
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: root/governance mirror; task:CmDecoder 活动日志入口；不涉及模型、数据、GT、坐标、split、checkpoint 或训练变量

**文件**
- [`activity_log.md`](activity_log.md) — 从切换点开始作为 CmDecoder 活动时间线，保留原 `status_log.md` 历史内容。
- [`repo_memory.md`](repo_memory.md)、[`experiment_log.md`](experiment_log.md) — 更新活动入口链接。
- [`architecture_log.md`](architecture_log.md) — 收窄架构记录头部为更新时间。
- [`../../config.py`](../../config.py)、[`../../current_cm_point_config.py`](../../current_cm_point_config.py) — 将当前运行元数据改为 `modification_version`。
- [`../../../../../docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 链接根级当前版本指针。

**原因**
统一 CmDecoder 的运行反馈、产物导航和治理事件记录；历史 `modification_log.md` 继续只读保留。

**验证**
- 根级文档/版本指针校验和全量回归：`311 passed, 3 skipped`；本次仅为工程治理迁移，`conclusion: N/A`。

## 2026-09-02 10:29:34 +0800 — 当前 best.pt 的 object-pose Inspire rollout

- activity_id: `ACT-20260902-102934-CMDECODER-ROLLOUT`
- timestamp: 2026-09-02 10:29:34 +0800
- modification_version: V1.1
- type: evaluation / rollout
- change_level: L1（新增只读评估入口）
- approval: user-approved（用户要求使用当前 best.pt 运行 Inspire rollout）
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- base_commit: `d9cecd0889087fbe7bff6ba85751e269cee8aac7`
- worktree_dirty: true（仓库治理迁移及 runtime rollout 入口未提交）
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`
- scope: 当前 CmDecoder best checkpoint 的 Inspire-F1 test episode rollout；不停止或修改训练进程

**输入与产物**

- checkpoint: [outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt)
- episode: `inspire_f1/bamboo_basket/5`（test split）
- cache/坐标：`hrdexdb_inspire_f1_object_pose_t_20260901`，`object_pose_t`，固定 stride=2
- 评估入口： [src/task/CmDecoder/runtime_rollout.py](../../runtime_rollout.py)
- 输出目录： [output/research/](../../../../../output/research/)
- 轨迹： [output/research/inspire_runtime_rollout_cmdecoder_best_20260902.npz](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_20260902.npz) 、 [output/research/inspire_runtime_rollout_cmdecoder_best_20260902.png](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_20260902.png)
- 清单： [output/research/inspire_runtime_rollout_cmdecoder_best_20260902.run_manifest.json](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_20260902.run_manifest.json) 、 [output/research/inspire_runtime_rollout_cmdecoder_best_20260902.summary.json](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_20260902.summary.json)

**结果**

- 32 个连续 pair；point EPE mean/final=`84.406/146.855 mm`。
- wrist 平移 EPE mean/final=`57.401/113.304 mm`；旋转误差 mean/final=`22.366/37.247°`。
- q MAE mean/final=`6.797/7.645°`。
- 这是 ground-truth Inspire hand-flow 输入 Cm 的 teacher-forced action-conditioned rollout，只测试 Decoder 状态反馈，不代表 autonomous Cm action prediction。

**验证**

- 使用 GPU7 完成；输出文件可重新加载，PNG 已生成。
- runtime evaluator `py_compile` 和模块导入通过；训练中的 GPU0/1/2 未停止或改动。
- 结果显示误差随闭环步数明显累积，当前证据不足以支持稳定 rollout，结论标记为 `INCONCLUSIVE`。

## 2026-09-02 10:36:00 +0800 — 修正 rollout evaluator 后复评

- activity_id: `ACT-20260902-103600-CMDECODER-ROLLOUT-FIX`
- timestamp: 2026-09-02 10:36:00 +0800
- modification_version: V1.1
- type: evaluation / implementation correction
- change_level: L1
- approval: user-approved（用户要求使用当前 best.pt 运行 rollout）
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- base_commit: `d9cecd0889087fbe7bff6ba85751e269cee8aac7`
- worktree_dirty: true
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`
- scope: 修正模型权重加载后的同一 test episode runtime rollout

**修正**

- 发现上一轮 runtime evaluator 漏掉 Decoder `payload["model"]` 的 `load_state_dict`，上一轮 `84.406/146.855 mm` 结果实际使用随机初始化 Decoder，标记为 `INVALID_IMPLEMENTATION`，不纳入科研结论。
- 已补充权重加载并以新文件名重跑；训练 GPU0/1/2 未停止或修改。

**修正后结果**

- checkpoint： [outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt) （step 46780 / epoch 10）；episode=`inspire_f1/bamboo_basket/5`（test）；`object_pose_t`；stride=2；32 pair；GPU7。
- point EPE mean/final=`14.669/22.807 mm`，wrist 平移 EPE mean/final=`18.305/18.895 mm`，q MAE mean/final=`7.930/10.666°`。
- 修正后仍可观察到闭环误差累积，但幅度远小于上一轮随机初始化结果；结论仍为 `INCONCLUSIVE`，需要多 episode 复评。
- 修正后产物： [output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.npz](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.npz) 、 [output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.png](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.png) 、 [output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.run_manifest.json](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.run_manifest.json) 、 [output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.summary.json](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.summary.json) 。

## 2026-09-02 11:09:00 +0800 — GRAB stride=2 到 Inspire-F1 跨手型 rollout

- activity_id: `ACT-20260902-110900-CMDECODER-GRAB-INSPIRE`
- timestamp: 2026-09-02 11:09:00 +0800
- modification_version: V1.1
- type: evaluation / cross-hand rollout
- change_level: L1（任务内只读评估入口）
- approval: user-approved（用户确认沿用历史 GRAB 轨迹与 object-pose/stride=2 设置）
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- base_commit: `d9cecd0889087fbe7bff6ba85751e269cee8aac7`
- worktree_dirty: true
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`
- scope: GRAB right-hand action token → Inspire-F1 point-flow/q-wrist rollout；不停止或修改 GPU0/1/2 训练

**设置与产物**

- GRAB：`s1/scissors_offhand_1/right`，start frame=`131`，stride=2，32 steps（15 Hz action source）。
- Inspire：joint-limit midpoint + 物体中心外侧 `0.12 m` approach pose；当前 CmDecoder best step=`46780` / epoch=`10`。
- 坐标：GRAB token 与 Inspire decoder 均按 `object_pose_t`；q-fit steps=`40`；GPU7。
- 入口： [src/task/CmDecoder/grab_runtime_rollout.py](../../grab_runtime_rollout.py) 。
- checkpoint： [outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt) 。
- 输出目录： [output/research/](../../../../../output/research/) 。
- 产物： [output/research/grab_runtime_rollout_cmdecoder_best_20260902.npz](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_20260902.npz) 、 [output/research/grab_runtime_rollout_cmdecoder_best_20260902.png](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_20260902.png) 、 [output/research/grab_runtime_rollout_cmdecoder_best_20260902.run_manifest.json](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_20260902.run_manifest.json) 、 [output/research/grab_runtime_rollout_cmdecoder_best_20260902.summary.json](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_20260902.summary.json) 。

**结果**

- Inspire robot/object centroid distance：初始/最终=`38.822/264.293 mm`，均值=`212.433 mm`，最大=`372.609 mm`。
- GRAB source/object centroid distance：初始/最终=`130.053/59.834 mm`，均值=`74.048 mm`。
- Inspire centroid cumulative displacement=`650.358 mm`，GRAB source=`391.940 mm`；平均单步分别 `20.324/12.248 mm`。
- 结果显示跨手型闭环明显偏离物体关系；无 Inspire GT，因此不报告目标 EPE，结论为 `INCONCLUSIVE`。

## 2026-09-02 11:21:00 +0800 — GRAB 30 Hz（stride=1）到 Inspire-F1 跨手型 rollout

- activity_id: `ACT-20260902-112100-CMDECODER-GRAB-INSPIRE-30HZ`
- timestamp: 2026-09-02 11:21:00 +0800
- modification_version: V1.1
- type: evaluation / cross-hand rollout / OOD source rate
- change_level: L1（任务内只读评估入口）
- approval: user-approved（用户明确要求 GRAB 不间隔、使用 30 Hz）
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- base_commit: `d9cecd0889087fbe7bff6ba85751e269cee8aac7`
- worktree_dirty: true
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`
- scope: GRAB right-hand action token（stride=1）→ Inspire-F1 point-flow/q-wrist rollout；不停止或修改 GPU0/1/2 训练

**设置与结果**

- `s1/scissors_offhand_1/right`，start=`131`，32 steps，30 Hz；object_pose_t；GPU7；当前 best step=`46780` / epoch=`10`。
- robot/object centroid distance 初始/最终/均值/最大=`27.644/316.077/99.922/316.077 mm`；GRAB source/object 初始/最终/均值=`130.053/65.808/87.817 mm`。
- robot/GRAB centroid 累计位移=`377.780/114.285 mm`，平均单步（含首帧零位移）=`11.806/3.571 mm`。
- stride=1 明确标注为 OOD：当前 Decoder 训练使用偶数 stride `2..20`；结果仅说明链路可执行，不能作为 30 Hz 泛化结论。

**产物**

- 入口： [src/task/CmDecoder/grab_runtime_rollout.py](../../grab_runtime_rollout.py) 。
- checkpoint： [outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt) 。
- 输出目录： [output/research/](../../../../../output/research/) 。
- [output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.npz](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.npz)
- [output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.png](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.png)
- [output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.run_manifest.json](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.run_manifest.json)
- [output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.summary.json](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_20260902.summary.json)

**原因**

按用户要求补充 stride=1 / 30 Hz 的 OOD 运行，并与训练使用的偶数 stride 明确区分，避免把链路可执行误写成泛化成立。

**验证**

- 运行终态为 `COMPLETED`；NPZ、PNG、run manifest 和 summary 均已生成，结论保持 `INCONCLUSIVE`。
- 本次导航修正只改变活动文档中的显示文本和相对链接，不改写运行产物、指标或科研结论。

## 2026-09-02 11:46:00 +0800 — 5 cm 起点与 GRAB 质心对齐初始化 rollout

- activity_id: `ACT-20260902-114600-CMDECODER-GRAB-INSPIRE-30HZ-ALIGNED`
- timestamp: 2026-09-02 11:46:00 +0800
- run_id: `grab_runtime_rollout_cmdecoder_best_30hz_aligned_20260902`
- modification_version: V1.1
- type: code_change + experiment_run / cross-hand rollout
- change_level: L2（改变 rollout 起点与坐标初始化语义）
- approval: user-approved（用户确认使用 5 cm 起点并将 Inspire 初始化到 GRAB 大致同一位置）
- approval_basis: 本轮用户明确回复“可以”
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- base_commit: `d9cecd0889087fbe7bff6ba85751e269cee8aac7`
- worktree_dirty: true
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`
- scope: 自动 5 cm 起点选择 + Inspire 中性手质心对齐 + 30 Hz GRAB→Inspire rollout；不停止或修改 GPU0/1/2 训练

**文件**

- `src/task/CmDecoder/grab_runtime_rollout.py`：新增 5 cm surface proximity 起点搜索；保留物体朝向，平移中性 Inspire 手部质心到 GRAB 起始质心；新增 pre-update 初始化距离诊断。

**原因**

- 排除从 GRAB 序列开头的远距离接近阶段，并使 Inspire 初始手部位置与 GRAB 起始状态一致，隔离初始化偏差对跨手型 rollout 的影响。

**验证**

- 自动选择 frame=`131`，起点最近表面距离=`43.174 mm`（frame130=`56.831 mm`）；初始化前 robot/object 质心距离=`130.053 mm`，与 GRAB 同帧一致。
- 30 Hz、32 steps 运行正常，无 OOM/NaN/异常退出；GPU7。
- 产物：[NPZ](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_aligned_20260902.npz)、[PNG](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_aligned_20260902.png)、[run manifest](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_aligned_20260902.run_manifest.json)、[summary](../../../../../output/research/grab_runtime_rollout_cmdecoder_best_30hz_aligned_20260902.summary.json)。

## 2026-09-02 12:00:00 +0800 — 物体表面采样密度 PLY 预览

- activity_id: `ACT-20260902-120000-CMDECODER-OBJECT-SAMPLING-PREVIEW`
- timestamp: 2026-09-02 12:00:00 +0800
- modification_version: V1.1
- type: data visualization / sampling preview
- change_level: L0（只读采样与可视化，不修改训练 cache）
- approval: user-approved（用户要求查看 512/1024/2048 点稀疏度）
- approval_basis: 用户明确要求生成 PLY 预览
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- base_commit: `d9cecd0889087fbe7bff6ba85751e269cee8aac7`
- worktree_dirty: true
- run_status: `COMPLETED`
- conclusion: `N/A`
- scope: GRAB 与 HRDexDB 各 5 个物体样本，表面池随机采样 512/1024/2048 点；输出仅写入工作目录 `tmp/object_sampling_preview/`

**文件**

- `tmp/generate_object_sampling_preview.py` — 可复现 PLY 生成脚本。
- `tmp/object_sampling_preview/manifest.json` — 样本、来源、随机种子和输出映射。
- `tmp/object_sampling_preview/{grab,hrdexdb}/*.ply` — 共 30 个 PLY（10 个对象 × 3 种采样数）。

**原因**

- 直观看当前 512 点表面采样相对于 1024/2048 点的稀疏程度；每个对象使用同一随机排列的前缀，便于密度对比。

**验证**

- 生成命令：`python tmp/generate_object_sampling_preview.py`；共生成 30 个 ASCII PLY。
- GRAB/HRDexDB 均从各自 4096 点表面池的 frame 0 采样；未改动任何 cache、checkpoint 或训练进程。

## 2026-09-02 12:30:00 +0800 — 近距离手/物体彩色 PLY 预览

- activity_id: `ACT-20260902-123000-CMDECODER-HAND-OBJECT-PLY`
- timestamp: 2026-09-02 12:30:00 +0800
- modification_version: V1.1
- type: data visualization / sampling preview
- change_level: L0（只读搜索、采样与可视化）
- approval: user-approved（用户要求寻找近距离帧并同时渲染手和物体）
- approval_basis: 用户明确要求不同颜色显示
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- base_commit: `d9cecd0889087fbe7bff6ba85751e269cee8aac7`
- worktree_dirty: true
- run_status: `COMPLETED`
- conclusion: `N/A`
- scope: GRAB 与 HRDexDB 各 5 个物体样本；每个选择手/物表面最近距离最小的帧，输出手+物体彩色 512/1024/2048 点 PLY 到 `tmp/`

**文件**

- [`tmp/generate_hand_object_preview.py`](../../../../../tmp/generate_hand_object_preview.py) — 近距离帧搜索与彩色 PLY 生成脚本。
- [`tmp/object_sampling_preview/close_frames/manifest.json`](../../../../../tmp/object_sampling_preview/close_frames/manifest.json) — 记录样本来源、选中帧、最近表面距离、颜色和输出映射。
- `tmp/object_sampling_preview/close_frames/{grab,hrdexdb}/*.ply` — 共 30 个合并 PLY；手为红色 `(232,70,70)`，物体为蓝色 `(65,125,235)`。

**原因**

- 在手和物体确实接近的时刻观察 512/1024/2048 点物体采样的空间稀疏度，同时保留 1538 个手部表面点作为参照。

**验证**

- 运行 `python tmp/generate_hand_object_preview.py` 成功；生成 30 个 PLY，顶点头信息和实际行数一致。
- GRAB 选中帧最近距离为 `0.018–0.080 mm`；HRDexDB 为 `0.064–0.319 mm`；未修改训练 cache、checkpoint 或训练进程。

## 2026-09-01 已停止 Cm，CmDecoder V1.1 计划待确认

- 当前 Cm 新训练 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222` 已按用户要求安全停止；其 `best.pt` 保留作为下一版 CmDecoder 的候选 frozen Cm checkpoint。
- 已新增草案计划：[`docs/plan/V1.1.md`](../plan/V1.1.md)。计划默认使用 Inspire-F1 object-disjoint v4 geometry、`object_pose_t` task cache、偶数 stride `{2,...,20}` 和 `CmPointFlowModel`；旧 hand-root/v2 task cache 与旧 token sidecar 禁止复用。
- 在计划定稿前不修改 CmDecoder 代码、不重建 cache、不启动 Decoder 长时训练；旧 cache、checkpoint 和输出均未覆盖。

## 2026-09-01 CmDecoder V1.1 已启动

- V1.1 计划已获用户确认并定稿；已新增 object-pose cache view、运行时偶数 stride Dataset 和当前 Cm 配置。
- 新训练：`outputs/cmdecoder/cm_decoder_20260901_151052`，GPU0/1/2、global batch=48、`CmPointFlowModel`、在线 frozen Cm token、初始 Cm 为 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222/checkpoints/best.pt`。
- 启动 smoke 已通过：step 100 hand-flow EPE=`8.136 mm`、zero-flow=`10.734 mm`，三卡显存/利用率正常；当前未见 OOM/NaN/NCCL 错误。
- 由于在线 token 与 10 个动态 stride 的输入语义，本 run 不复用旧 C=256 或 hand-root token sidecar。

## 当前操作 — V1.2.12：撤出 Task-local Component/data/registry（2026-09-01）

- 已删除 `src/task/CmDecoder/components/`、`data/`、`registry/` 以及根级 `components/ref2dex/cmdecoder_pointflow` 兼容入口。
- 配置不再声明 Component 清单；数据和 cache 继续直接使用根级 `data/`、`dataset/` 与 `outputs/`。
- 真实数据、cache、checkpoint、output 和运行中的 decoder 进程未移动、删除或停止。
- 计划：[`docs/plan/V1.md`](../plan/V1.md)（final）；历史 Component 修改记录保留，不作为当前入口。

- scope: task:CmDecoder
- last_updated: 2026-09-01
- last_verified: 2026-09-01
- related: [架构](architecture_log.md), [实验](experiment_log.md), [接手记忆](repo_memory.md)

## 当前状态

- 当前阶段 / 指导: wrist-aware baseline 保持相对 wrist SE(3)+6维手指 q 输出；已在修正腕部语义的3 Hz v2 cache 上完成纯固定对应点 loss 三组对照。逐手点 Cm flow → 联合 wrist+q fitting 路线也已实现。
- 历史 EXP-022 已停止：输出 `outputs/cmdecoder/cm_decoder_20260829_225518`；停止前约 step `86500/143110` / epoch `19`，无 OOM/NaN/NCCL 错误。EXP-023 rollout 已通过 Viser 在 `http://localhost:8096` 提供播放。
- 历史短暂启动的 decoder run `outputs/cmdecoder/cm_decoder_20260831_165400` 已停止；仅推进到 step `300` / epoch `1`，未生成 checkpoint。停止原因是发现其 v2 task cache 坐标与当前 Cm checkpoint 不一致，不能作为有效实验结果。
- 最近可靠结论: EXP-022 validation hand-flow EPE 从 epoch 1 的 `2.291 mm` 降至 epoch 15 的当前最佳 `1.425 mm`（step `71550`），相对 zero-flow=`12.118 mm` 改善约 `88.2%`。同 best 的 Inspire action-conditioned EXP-023 在固定 train episode 上 point EPE=`7.325/13.450 mm`，EXP-025 在 held-out test episode `bamboo_basket/5` 上为 `6.896/11.415 mm`（mean/final），两者均未超过 `20 mm`；test 结果支持短程状态反馈具有一定泛化，但仍有误差累积。本次 EXP-024 改为 GRAB hand-flow→当前 Cm tokens→Inspire F1，无 Inspire 侧 Cm 输入；32 帧 robot centroid-object 距离由 `27.9 mm` 漂至 `471.5 mm`，机器人质心累计位移 `297.1 mm`，说明跨手型无配对重定向仍明显发散。
- 阻塞 / 风险: v1 仅保留历史复现，不可用于 wrist-aware 点监督；旧版 Cm、EXP-018 Inspire checkpoint 与本次混合 Cm 的效果不能混作同一实验结论。当前三卡 decoder 与 Cm 不再并行，之前的 GPU1 慢卡竞争已解除；新的 global batch48 相比旧 global batch16 增加每次 optimizer step 的样本量，学习率仍保持 `3e-4`，需单独解释收敛速度与指标可比性。GRAB→Inspire F1 单序列有效窗口重定向仍出现约 `324 mm` object-relative wrist 漂移，当前无约束逐帧 fitting 不可作为可用重定向方案。
- 下一步: 先从当前 v4 geometry cache 重建与 `object_pose_t` 一致的 Inspire-F1 decoder task cache，再重新启动逐点 hand-flow decoder；重建前不复用旧 v2 的 wrist-frame task arrays 或 C=256 token sidecar。
- 证据与相关文档: [接手记忆](repo_memory.md), [实验记录](experiment_log.md)
## 2026-08-30 cache 语义更新

- 用户确认取消 5cm object candidate 查询；新 builder 保留兼容文件名但写入全 object surface pool 的全真 mask。
- 新 Inspire-F1 cache 正在会话中重建：`data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_20260830`。

## 2026-08-30 重新计算 Inspire-F1 5cm mask

- 用户要求恢复帧级 5cm candidate 过滤，但保留新的完整 object surface pool 与运行时 512 点随机采样。
- 对 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4` 的 576/576 episode 已完成 GPU 计算；每个 episode 写入独立的 `geometry/obj_candidate_mask_5cm_recomputed.npy`，未覆盖训练当前读取的全真 `obj_candidate_mask_5cm.npy`。
- 全量统计：362,027 帧、mask 有效点总数 349,129,312，占 object-pool 点数 `23.5443%`。按 manifest：train 455 episode / 285,956 transitions 中 160,179 个有至少一个 5cm candidate（56.015%）；val 为 58.529%，test 为 59.248%。
- 当前 Cm 混合训练仍在 GPU 0/1/2 运行，尚未切换到 sidecar mask，避免在运行中改变 dataset 行空间。

## 2026-09-03 09:13:06 +0800 — CmDecoder V1.2 全 stride 运行启动

- activity_id: `ACT-20260903-091306-CMDECODER-V12-FULL`
- timestamp: 2026-09-03 09:13:06 +0800
- modification_version: V1.2
- type: operation / experiment
- change_level: L3（冻结编码器迁移与长时训练）+ L2（模型接口、pair schema）
- approval: user-approved（用户确认条件重建、随机初始化、十个 stride 全展开、停止旧 run、三卡 global batch48、首轮不存 token cache）
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `7b63dc50896392023c5c0f33c7d508cbe4436b50`
- worktree_dirty: true（保留用户既有改动；本次新增 V1.2 Task-local 文件）
- stopped_previous_run: CmDecoder 旧 run 的 torchrun PID `4148810` 已发送 SIGTERM，子进程已退出；旧输出保留。
- run_id: `cm_decoder_20260903_091220`
- run_status: `RUNNING`
- command: `CUDA_VISIBLE_DEVICES=0,1,2 setsid nohup /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoder.train --config src.task.CmDecoder.current_object_interaction_cm_v1_2_config:Config --set train.distributed.enable=true --set train.description=cmdecoder_v1_2_inspire_all_stride_expanded_full --set wandb.enable=false`
- scope: Inspire-F1 object-disjoint 455/58/63 episodes；全部 1538 手点监督；object_pose_t；1024/4096 物体点；stride `{2,4,6,8,10,12,14,16,18,20}` 全展开；ObjectInteractionCm 全冻结。
- current evidence: 三卡 DDP 短跑 `cm_decoder_20260903_091053` 10 steps 通过；全量 run 启动时 `total_steps=1,403,370`、per-device batch=16、global batch=48；约 step 1200 时无 OOM/NaN/NCCL，吞吐约 1,680 samples/s，ETA 约 11.1 h。

**产物**

- [V1.2 指导](../指导/V1.2.md)
- [V1.2 最终计划](../plan/V1.2.md)
- [pair index manifest](../../../../../data/processed_data/cm_decoder/cmdecoder_inspire_f1_object_pose_t_v1_2/manifest.json)
- [pair index run manifest](../../../../../data/processed_data/cm_decoder/cmdecoder_inspire_f1_object_pose_t_v1_2/run_manifest.json)
- [全量运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/)
- [全量 run manifest](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/run_manifest.json)
- [全量 metrics](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/metrics.jsonl)
- [全量 train log](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/train.log)
- [三卡短跑目录](../../../../../outputs/cmdecoder/cm_decoder_20260903_091053/)

**原因**

- 用户确认用 ObjectInteractionCm 冻结 Cm 训练 Inspire-F1 Decoder；为避免旧 CmDecoder 继续占用三卡，按授权停止旧 torchrun，并将十个偶数 stride 全部展开训练。
- 现有 geometry cache 已满足 1538 手点和 4096 物体 pool 合同，仅需新增可审计的 pair/index manifest；首轮不生成易错配的 Cm-token sidecar。

**文件**

- `src/task/CmDecoder/dataset.py` — 1024/4096 物体采样、全 stride 展开和 smoke 截断参数。
- `src/task/CmDecoder/object_interaction_cm_model.py` — 冻结 ObjectInteractionCm adapter 与随机初始化 1538 点 Decoder。
- `src/task/CmDecoder/current_object_interaction_cm_v1_2_config.py` — V1.2 配置。
- `src/task/CmDecoder/build_pair_index_v1_2.py` — pair/index builder。
- `src/task/CmDecoder/docs/指导/V1.2.md`、`src/task/CmDecoder/docs/plan/V1.2.md` — 同后缀指导与最终计划。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 正式实验登记；`src/task/CmDecoder/docs/logs/modification_log.md` 保留既有历史变更（未覆盖）。

**验证**

- `python -m py_compile src/task/CmDecoder/dataset.py src/task/CmDecoder/object_interaction_cm_model.py src/task/CmDecoder/build_pair_index_v1_2.py src/task/CmDecoder/current_object_interaction_cm_v1_2_config.py` 通过。
- pair builder 生成 train/val/test=`2,245,405/277,764/263,722`，十个 stride 均有记录，manifest 绑定 source/checkpoint SHA256。
- 单卡 smoke `cm_decoder_20260903_090924` 通过；三卡 DDP smoke `cm_decoder_20260903_091053` 10 steps 通过，world size=3、global batch=48、仅 Decoder 有梯度。
- 全量运行已启动并持续写入 train log/metrics；当前无 OOM、NaN 或 NCCL 错误，终态链接保持 PENDING 语义仅适用于尚未生成的 checkpoint。

**验证与结论**

- 单卡 smoke：模型输出 `[B,1538,3]`，Cm token `[B,16,32]`，所有输出 finite；仅新 Decoder 参数产生梯度。
- 三卡 DDP smoke：world size=3、global batch=48、10 steps 完成，无 OOM/NaN/NCCL；该证据仅证明工程可运行，不代表科研效果。
- 当前全量实验状态为 `INCONCLUSIVE`（RUNNING），待终态 test 指标和 manifest 审计后再判断。

## 2026-09-03 10:09:40 +0800 — CmDecoder V1.2 训练状态诊断

- activity_id: `ACT-20260903-100940-CMDECODER-V12-STATUS`
- timestamp: 2026-09-03 10:09:40 +0800
- modification_version: V1.2
- type: diagnostic
- change_level: L0（只读进程、日志、指标和 GPU 状态）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `7b63dc50896392023c5c0f33c7d508cbe4436b50`
- scope: `src/task/CmDecoder/`、`data/processed_data/cm_decoder/cmdecoder_inspire_f1_object_pose_t_v1_2/`
- run_id: `cm_decoder_20260903_091220`
- run_status: `RUNNING`

**原因**

- 用户要求核查 V1.2 全量训练进度、验证指标、收敛迹象和资源状态。

**验证**

- torchrun parent PID `4034733` 及 3 个 rank 仍在运行；GPU 0/1/2 利用率约 94%，未见训练进程 OOM/NaN/NCCL 报错。
- 最新完整 epoch 为 epoch 2 / step `93,558`（总步数 `1,403,370`，约 6.66%）；检查时 epoch 2 验证已完成，epoch 3 已开始（约 step `95,000`）。
- epoch 1 验证 all-stride 聚合 hand-flow EPE=`7.46595 mm`，zero-flow=`12.18257 mm`，相对 zero-flow 改善约 38.7%；当前最佳 checkpoint 仍为 epoch 1（配置每 10 epoch 保存一次普通 checkpoint）。
- epoch 2 验证 all-stride 聚合 hand-flow EPE=`7.53461 mm`，zero-flow=`12.18257 mm`，较 epoch 1 略差，尚未刷新 best。
- epoch 2 训练聚合 hand-flow EPE=`7.36823 mm`，zero-flow=`11.85406 mm`；训练仍在下降，但验证尚不足以判断已收敛。

**产物**

- `src/task/CmDecoder/docs/logs/modification_log.md` — 保留既有历史修改记录，本次未改写。
- [全量运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/)
- [逐步指标](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/metrics.jsonl)
- [训练日志](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/train.log)
- [当前最佳 checkpoint](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/checkpoints/best.pt)

## 2026-09-03 11:37:30 +0800 — CmDecoder V1.2 收敛状态复核

- activity_id: `ACT-20260903-113730-CMDECODER-V12-CONVERGENCE`
- timestamp: 2026-09-03 11:37:30 +0800
- modification_version: V1.2
- type: diagnostic
- change_level: L0（只读进程、metrics、checkpoint、GPU 与错误日志）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `52f47ae`
- scope: `outputs/cmdecoder/cm_decoder_20260903_091220/`
- run_id: `cm_decoder_20260903_091220`
- run_status: `RUNNING`
- command: `ps`、`metrics.jsonl`、`train.log`、checkpoint metadata、`nvidia-smi` 只读检查

**原因**

- 用户要求确认当前训练进度、是否已经收敛，以及验证曲线是否足以支持停止训练。

**验证**

- torchrun parent PID `4034733` 与 3 个 rank 仍在运行；GPU0/1/2 仍有训练负载，未发现 OOM、NaN、Inf、NCCL 或 traceback。
- 最新读取到 step `280674`（epoch 6 完成），约为总计划 `1,403,370` steps 的 `20.0%`；仍有约 80% 训练预算未执行，epoch 6 validation 尚未写入。
- validation all-stride hand-flow EPE：epoch 1 `7.46595 mm`、epoch 2 `7.53461 mm`、epoch 3 `7.59809 mm`、epoch 4 `7.52755 mm`、epoch 5 `7.41799 mm`；epoch 5 刷新当前 best，较 epoch 1 仅改善约 `0.048 mm / 0.64%`，且曲线仍有波动，尚无足够 epoch 判断平台。
- train epoch EPE 从 epoch 1 `7.58562 mm` 降至 epoch 5 `7.20424 mm`，epoch 6 为 `7.17160 mm`，说明优化仍在进行；当前 best checkpoint 为 epoch 5 / step `233895`，普通保存间隔为每 10 epoch。
- 当前判断：**尚未收敛，继续训练**。现有证据为 `INCONCLUSIVE`，不能据此提前停止或宣称终态收敛。

**产物**

- `src/task/CmDecoder/docs/logs/modification_log.md` — 保留既有历史修改记录，本次未改写。
- [全量运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/)
- [metrics](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/metrics.jsonl)
- [train log](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/train.log)
- [当前 best checkpoint](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/checkpoints/best.pt)

**回滚**

- 本条只增加诊断记录；未修改训练、模型、配置、cache、checkpoint 或任何进程。

## 2026-09-03 19:02:30 +0800 — CmDecoder V1.2 全量训练按用户要求停止

- activity_id: `ACT-20260903-190230-CMDECODER-V12-STOP`
- timestamp: `2026-09-03 19:02:30 +0800`
- modification_version: V1.2.1
- type: operation / diagnostic
- change_level: L3（停止既有长时训练进程并释放 GPU0/1/2）
- approval: user-approved
- approval_basis: 用户明确要求停止当前 GPU0/1/2 的 decoder 训练并将三张卡转给 ObjectInteractionCm
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `52f47ae7bf688e652aef0fc6b49e2ee2eea8d868`
- worktree_dirty: true
- scope: [CmDecoder Task](../../)；只停止指定 `cm_decoder_20260903_091220`，不修改代码、cache 或 checkpoint
- run_id: `cm_decoder_20260903_091220`
- run_status: STOPPED
- conclusion: INCONCLUSIVE（提前停止，不能作为完整实验结论）

**文件**

- [CmDecoder 全量运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/) — 保留已有 manifest、metrics、日志和 checkpoint。
- [CmDecoder Task](../../) — 记录停止操作；实现和数据文件未修改。

**原因**

释放 GPU0/1/2 给用户指定的 ObjectInteractionCm 三卡训练。目标 torchrun parent `4034733` 及 ranks `4034841/4034843/4034844` 均与 `cmdecoder_v1_2_inspire_all_stride_expanded_full` 命令一致，确认后发送 SIGTERM；GPU0 上的 InteractionTransfer viewer 进程未停止。

**验证**

- 停止后未再发现 `src.task.CmDecoder.train` 或对应 torchrun 进程。
- `nvidia-smi`：GPU1/2 已降至约 5 MiB；GPU0 仅保留三个 InteractionTransfer viewer 的约 2.6 GiB 显存，不属于 decoder。
- 运行目录 checkpoint 未删除；最后日志可见约 step `973100`，既有产物保持可复核。

**回滚**

该训练未被删除；如需继续，可从其已有 checkpoint 按原配置显式 resume。停止操作本身不可恢复进程状态，但不影响已有产物。
