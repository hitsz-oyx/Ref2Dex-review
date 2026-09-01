# CmDecoder 修改记录

## 2026-09-01 — 提交 Cm/CmDecoder 表面采样与 object-pose 训练链路

- change_level: L3（训练/cache 迁移与长时任务）+ L2（采样、坐标系、GT、cache/schema）
- approval: user-approved（用户确认 V1.1 计划并要求提交相关更改）
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/modular-component-runtime`
- post-commit: 已提交（以 Git 历史为准）
- scope: `process/GRAB`、`process/common`、`src/task/Cm`、`src/task/CmDecoder` 及定向测试

**文件**

- `process/GRAB/build_cm_split.py` — 支持 scene cache 的序列发现和 manifest 路径。
- `process/GRAB/raw.py` — 传递手部 mesh 顶点与面。
- `process/GRAB/stage4_cm.py` — 写入物体姿态和手部 mesh 字段。
- `process/GRAB/stage4_cm_scene.py` — 使用完整物体表面池并写入物体姿态。
- `process/common/object_cache_v2.py` — 转换 cache 时保留物体姿态及手部 mesh 字段。
- `src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml` — 对齐当前 object-pose、surface512 和 Inspire-F1 stride 配置。
- `src/task/Cm/configs/active/hrdexdb_inspire_f1_finetune_cm64_additive.yaml` — 对齐 HRDexDB Inspire-F1 finetune 数据合同。
- `src/task/Cm/dataset/cache_schema.py` — 扩展 cache schema 字段合同。
- `src/task/Cm/dataset/hrdexdb.py` — 支持姿态坐标、表面池和数据筛选。
- `src/task/Cm/dataset/object_v2.py` — 支持新的物体表面与姿态字段。
- `src/task/Cm/dataset/scene.py` — 支持 scene cache 新字段和候选语义。
- `src/task/Cm/dataset/stage4.py` — 适配 stage4 表面采样与坐标字段。
- `src/task/Cm/dataset/surface_sampling.py` — 提供物体/手部表面采样实现。
- `src/task/CmDecoder/build_cache.py`、`dataset.py` — 使用运行时表面采样、`object_pose_t` 和偶数 stride pair。
- `src/task/CmDecoder/current_cm_point_config.py`、`prepare_object_pose_cache.py`、`recompute_candidate_mask.py` — 当前 Cm 初始化配置、cache view 和可复现 mask 工具。
- `src/task/CmDecoder/docs/plan/V1.1.md` 及 `docs/logs/*.md` — 记录最终计划、决策、实验、状态和实际修改。
- `tests/test_cm_surface_sampling.py` — 覆盖表面采样行为。

**原因**

将 Cm 与 CmDecoder 的数据合同统一到物体姿态坐标系，支持物体表面 512 点采样和 Inspire-F1 偶数 stride，并保留可审计的 cache/训练入口；本次提交不包含生成的 cache、checkpoint、output 或运行进程状态。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_cm_surface_sampling.py tests/test_run_manifest.py`：5 passed。
- `git diff --cached --check`：通过。
- `audit_diff.py --log src/task/CmDecoder/docs/logs/modification_log.md --staged`：提交前重新运行并通过。

## 2026-09-01 — Cm 停止与 CmDecoder V1.1 计划草案

- change_level: L3（长时训练与大型 cache）+ L2（cache/schema、GT 和坐标系）
- approval: pending（用户确认停止 Cm；新 Decoder 计划待定稿）
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: working tree
- post-commit: 未提交
- scope: `task:CmDecoder` 计划和状态记录；不修改代码、cache 或训练配置

**文件**
- `src/task/CmDecoder/docs/plan/V1.1.md` — 新增使用当前 Cm `best.pt` 训练 point-flow Decoder 的 cache、object_pose_t 坐标合同、stride、token sidecar、验证和回滚计划。
- `src/task/CmDecoder/docs/logs/status_log.md` — 记录 Cm 已停止及计划待确认状态。

**原因**
用户要求停止当前 Cm，并在处理 cache 与坐标系后使用该版本训练 CmDecoder；现有旧 hand-root/v2 task cache 与当前 Cm 的 `object_pose_t` 合同不兼容。

**验证**
- 已确认旧 Cm torchrun 及其 rank 进程退出。
- 已读取当前 CmDecoder V1 指导、V1 plan、架构/决策/实验日志、v4 geometry manifest 和当前 Cm 配置。
- 仅新增文档；未重建 cache、未修改代码、未启动 Decoder 训练。

## 2026-09-01 — CmDecoder V1.1 object_pose_t cache 与训练启动

- change_level: L3（长时三卡训练）+ L2（cache/schema、坐标系和采样合同）
- approval: user-approved（用户确认 V1.1 计划并授权执行）
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: working tree
- post-commit: 未提交
- scope: `task:CmDecoder` runtime pair cache、配置、训练进程和日志

**文件与产物**
- `src/task/CmDecoder/dataset.py` — `RandomHorizonGeometryDataset` 增加 `object_pose_t`、偶数 stride、全 stride eval 和 stride 记录；修正训练 stride 分布偏置。
- `src/task/CmDecoder/prepare_object_pose_cache.py` — 新增轻量 object-pose cache manifest/symlink 生成器，避免复制 v4 geometry。
- `src/task/CmDecoder/current_cm_point_config.py` — 新增当前 Cm C=64 point-flow Decoder 配置。
- `data/processed_data/cm_decoder/hrdexdb_inspire_f1_object_pose_t_20260901/` — 新 cache view（manifest + geometry symlink，不复制大型数组）。
- `outputs/cmdecoder/cm_decoder_20260901_151052/` — 新三卡 Decoder 训练输出。

**原因**
旧 CmDecoder v2 task cache 的 hand-flow 在 current-wrist frame，而当前 Cm checkpoint 使用 `object_pose_t`；本次统一坐标合同并按当前 Cm 的偶数 stride 分布训练。

**验证**
- cache manifest：576 episodes，object-disjoint train/val/test=`455/58/63`。
- loader：train/val/test=`224551/277764/263722`，train stride 分布近似均衡，val/test 覆盖 `{2,4,6,8,10,12,14,16,18,20}`。
- 单 batch CUDA forward/backward：输出 `[2,1538,3]`，finite，梯度存在，Cm 参数冻结。
- 正式训练启动：world size=3，step 100 hand-flow EPE=`8.136 mm`，zero-flow=`10.734 mm`；无 OOM/NaN/NCCL 错误。

## 2026-09-01 — V1.2.12 撤出 Task-local Component/data/registry

- change_level: L3（破坏性目录治理与数据入口迁移）
- approval: user-approved（用户明确确认彻底删除 Component、Task data/registry，并使用根空间）
- skills_used: `research-change-control`
- branch: `feature/modular-component-runtime`
- version: `V1.2.12`（plan: `docs/plan/V1.md`, final）
- category: `governance`、`operation`、`documentation`
- post-commit: 未提交；真实数据、cache、checkpoint、output 和运行进程未触碰
- scope: `src/task/CmDecoder/components/`、`data/`、`registry/`、配置和当前目录说明

**文件**

- 删除 Task-local Component 清单、数据软链接、路径 registry 及根级兼容软链接。
- `config.py` 移除 Component 选择并更新细分版本；新增 V1 执行计划。
- 状态、记忆和根/Task 文档同步当前入口，历史实验记录保留。
- `src/task/CmDecoder/docs/logs/repo_memory.md` — 记录 CmDecoder 数据入口统一回到根空间。
- `src/task/CmDecoder/docs/logs/architecture_log.md` — 标记当前入口边界并区分历史架构描述。

**验证**

- 入口扫描无残留；CmDecoder 配置导入成功。
- 全量 pytest 与 `git diff --check` 通过。

## 2026-08-30 — 完成 held-out Inspire test rollout

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验产物与文档；change_level: L0；approval: auto

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 更新 test rollout 证据与泛化边界。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-025，记录 test episode、协议、指标和结论。

**原因**

响应用户要求将当前 best 的 Inspire rollout 改到 test split；使用 `inspire_f1/bamboo_basket/5`，不影响三卡训练。

**验证**

manifest 核对该 episode 属于 test；32 个有效 pair 成功生成 NPZ/PNG，point EPE mean/final=`6.896/11.415 mm`，无运行错误。

## 2026-08-30 — 核对 Inspire rollout 序列的 split 归属

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验文档；change_level: L0；approval: auto

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md`、`experiment_log.md` — 明确 EXP-023 使用的 `inspire_f1/apple/2` 属于 train split，限制其泛化解释。

**原因**

响应用户询问 rollout 使用 train 还是 test；通过读取 `selection_576_seed42.json` 的 `splits` 字段核对，`apple/2` 在 train 列表中。

**验证**

manifest split 计数为 train/val/test=`455/58/63`，`inspire_f1/apple/2` 命中 train，未命中 val/test。

## 2026-08-30 — 完成 GRAB→Inspire 无配对重定向诊断

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验产物与文档；change_level: L0；approval: auto

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 同步跨手型重定向诊断结论。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-024，记录 source hand-flow、target robot、结果和边界。

**原因**

响应用户澄清：保持当前 Cm/decoder checkpoint，改用 GRAB 数据产生 Cm tokens 驱动 Inspire F1，而不是使用 Inspire 数据产生的 Cm。

**验证**

已有 `grab_retarget.py` 完成 32 帧 CUDA 运行并生成 NPZ/PNG；无异常退出。机器人 centroid-object 距离末帧 `471.5 mm`，显示当前序列明显漂移。

## 2026-08-30 — 启动 EXP-023 rollout Viser 可视化

- branch: working tree
- post-commit: 未提交
- scope: task 内部可视化服务；change_level: L0；approval: auto

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 记录当前 Viser 播放地址。

**原因**

响应用户要求直接查看当前 best 的 rollout；使用已有 `inspire_rollout.py --load-trajectory --serve`，不重新计算轨迹、不影响三卡训练。

**验证**

Viser 已监听 `*:8096`，控制台输出 `Inspire rollout viewer: http://localhost:8096`。

## 2026-08-30 — 完成 EXP-022 best 的 action-conditioned rollout

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验产物与文档；change_level: L0；approval: auto

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 同步 rollout 结论与下一步。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-023，记录命令、协议、指标、对比和结论。

**原因**

响应用户要求，用 EXP-022 当前 `best.pt` 检查 rollout 效果；保持三卡训练不停止，评估放在物理 GPU3。

**验证**

rollout 生成 31 个有效 pair，trajectory/PNG 均成功写出；point EPE mean/final=`7.325/13.450 mm`，wrist=`4.830/9.574 mm`，无运行错误。

## 2026-08-30 — 同步 EXP-022 epoch 16 验证与当前进度

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验文档；change_level: L0；approval: auto

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 更新 step、epoch、吞吐、ETA、最佳 checkpoint 与风险判断。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 记录 epoch 1–16 validation 曲线和当前最佳证据。

**改动原因**

响应训练状态检查；仅同步已有运行证据，不修改训练进程、配置或科研合同。

**验证**

进程、三卡利用率、`train.log`、`metrics.jsonl` 和 checkpoint payload 已交叉核对；当前 step 约 `79200`，best 为 epoch 15 / step `71550`、`1.4249 mm`，未发现训练错误。

## 2026-08-29 — 启动三卡 global batch 48、固定总 step 的 CmDecoder

- branch: working tree
- post-commit: 未提交
- scope: task 内部长时实验；change_level: L3；approval: user-approved

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 更新三卡 run 状态和 global batch/step 合同。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-022，记录命令、snapshot、预算和早期 smoke 证据。
- `src/task/CmDecoder/docs/logs/decision_log.md` — 记录保持总 step、扩宽 epoch 的选择及样本暴露量影响。

**改动原因**

按用户确认停止单卡试跑并在 GPU0/1/2 启动 decoder；显式打开 distributed，设置 global batch48、`max_steps=143110`、30 epoch。首次启动因 distributed 开关保护退出，补齐 `train.distributed.enable=true` 后启动成功。

**验证**

`train_setup` 已报告 `world_size=3 per_device_batch=16 global_batch=48 total_steps=143110`；step500 正常记录，无 OOM/NaN。

## 2026-08-29 — 记录三卡 Cm 停止后的 decoder 待迁移状态

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验文档；change_level: L3；approval: user-approved

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 更新单卡 decoder 当前 step，并记录 Cm 已停止及三卡迁移待确认项。

**改动原因**

用户要求停止 Cm 并将 GPU0/1/2 改用于 CmDecoder；当前 decoder 尚无 epoch checkpoint，故在训练预算和续训语义确认前保持单卡进程运行，避免丢弃已完成的约 `11500` steps。

## 2026-08-29 — 启动混合 Cm 驱动的单卡 CmDecoder 实验

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验；change_level: L3；approval: user-approved

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 更新当前运行、资源占用与下一步。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-021，记录 checkpoint、命令、配置、产物和早期证据。
- `src/task/CmDecoder/docs/logs/decision_log.md` — 记录单卡并行与启动时固定 Cm snapshot 的实现选择。

**改动原因**

响应用户确认：不停止现有 Cm，使用当前混合 Cm `latest.pt`，按上一版 CmDecoder 合同在单卡 GPU1 启动新训练。启动 smoke 已验证 `world_size=1`、总步数 `143110`，训练进程与日志持续更新；未修改模型源码或基线配置。

## 2026-08-27 — 同步新版 Inspire decoder 的 epoch 1 状态

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验文档

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md`、`experiment_log.md` — 更新新版 Inspire F1 point-flow decoder 至约 step `23800`，补充 epoch 1 validation 与当前收敛判断。

**改动原因**

响应训练状态检查；当前新版 decoder 已明显优于 zero-flow，但仅有一个 validation 点，继续训练后再判断是否达到旧 decoder 的最终水平。

## 2026-08-28 — 同步新版 Inspire decoder 的 epoch 5 验证状态

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验文档

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md`、`experiment_log.md` — 更新新版 Inspire F1 point-flow decoder 至约 step `80000`，补充 epoch 1--5 validation 曲线和当前最佳 checkpoint。

**改动原因**

响应训练状态检查；新版 decoder validation EPE 持续下降至 `5.201 mm`，仍在训练中，暂不判定收敛。

## 2026-08-27 — 统一单帧与 rollout 可视化界面并补充图例

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/viewer.py` — 将 rollout 轨迹接入原有 viewer 控件；固定显示模式下拉框、Pair/Rollout step、mesh/点云开关和 Play/Stop。未传 `--rollout-trajectory` 时 rollout 模式保持灰色禁用；传入轨迹后可在同一界面切换单帧与闭环模式，并自动从轨迹 episode 恢复单帧数据。增加蓝/绿/橙/灰/品红颜色及 mesh、点云、flow 语义图例，并补充两种启动方式说明；CLI 明确 pair 始终使用 checkpoint/cache 现场推理。
- `src/task/CmDecoder/docs/logs/architecture_log.md` — 同步统一可视化模式、mesh/点云控件和颜色语义。
- `src/task/CmDecoder/docs/logs/status_log.md` — 更新统一 viewer 的当前状态。

**改动原因**

避免根据输入文件改变基础 UI，保留此前单帧 mesh 与采样点功能，同时让 rollout 成为显式、可检查的数据模式。

**验证**

- `graspenv` 下 `viewer.py` 与 `inspire_rollout.py` 编译通过。
- 使用 `output/research/inspire_rollout_cmdecoder_apple2_30hz_32.npz` 启动统一 viewer，Viser 正常监听 `http://localhost:8096`。

## 2026-08-27 — 将 rollout 播放入口并回单帧 viewer

- branch: working tree
- post-commit: 未提交
- scope: task 内部可视化入口

**文件**

- `src/task/CmDecoder/viewer.py` — 保持原有 `--object/--scene` 单帧 viewer 命令不变；新增可选 `--rollout-trajectory`，使用同一个 Viser 入口播放已有 rollout NPZ。

**改动原因**

避免单帧和 rollout 需要记忆两个独立脚本；rollout 是显式模式，默认行为仍是原来的逐 pair teacher-forced viewer。

## 2026-08-27 — 增加 Inspire action-conditioned rollout 评估与可视化

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验入口与可视化

**文件**

- `src/task/CmDecoder/inspire_rollout.py` — 新增 Inspire 同手型 action-conditioned closed-loop rollout；输出逐步 hand points、mesh、q、wrist、接触比例、NPZ 和诊断 PNG，并支持从 NPZ 启动 Viser 播放。
- `src/task/CmDecoder/viewer.py` — 增加 30 Hz v4 manifest/cache override、物体世界法向读取、丢帧 pair 过滤和动作幅度字段，保证 rollout 与 teacher-forced viewer 使用一致的有效 pair。
- `docs/logs/experiment_log.md` — 记录 EXP-020 的协议、定量结果和结论。
- `docs/logs/status_log.md` — 同步 Inspire rollout 发散状态和复跑计划。

**改动原因**

用户要求增加类似 rollout 的可视化并立即运行 Inspire 自身结果。当前实现明确区分 GT action-conditioned closed-loop 与完全自主 Cm policy，避免把前者误称为 autonomous rollout。

## 2026-08-27 — 记录新版 Cm 驱动 decoder 训练启动

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验文档

**文件**

- `docs/logs/experiment_log.md` — 新增 EXP-018 的运行假设、checkpoint、数据合同和初始指标。
- `docs/logs/status_log.md` — 更新当前进行中的训练、GPU、输出目录和下一步。
- `docs/logs/decision_log.md` — 记录禁用旧 token cache、在线重算 Cm 的决定及预计时长影响。

**改动原因**

按用户要求启动新版 `Cm` checkpoint 的 Inspire F1 point-flow decoder，并保留可复现实验合同与 checkpoint 绑定约束。

## 2026-08-24 — 拆分 CmDecoder 仓库记忆与机器记忆

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部 / 全局文档规范联动

**文件**

- `repo_memory.md` — 由旧 `memory_log.md` 和 `repo_notes_log.md` 提炼可迁移的 HRDexDB 入口、cache 版本和 checkpoint 绑定事实。
- `machine_memory.md` — 保存当前服务器的 HRDexDB 绝对路径和本机资产计数，并由 Git 忽略。
- `status_log.md`、`architecture_log.md` — 更新接手记忆链接。
- `repo_notes_log.md` — 删除。

**改动原因**

遵循新的递归 memory 规范，使 CmDecoder 的公共数据合同使用仓库相对路径。

## 2026-08-24 — 修正 GRAB 重定向起始窗口并记录有效窗口诊断

- branch: working tree
- post-commit: 未提交
- scope: task 内部实验入口与文档

**文件**

- `src/task/CmDecoder/grab_retarget.py` — 新增 `--start-frame`，所有几何、human sample 和初始化均从指定帧开始，并在 NPZ 中记录起始帧。
- `src/task/CmDecoder/docs/logs/architecture_log.md` — 补充有效 candidate 窗口约束。
- `src/task/CmDecoder/docs/logs/{experiment,status}_log.md` — 记录默认空 candidate 窗口无效，以及 start=131 有效窗口仍发生 wrist 漂移的结果。

**改动原因**

默认序列前32帧右手没有物体 candidate，旧重定向图实际使用 padding 物体点；必须选择有效窗口后才能诊断当前 baseline 的真实重定向行为。

## 2026-08-24 — 完成 point-flow baseline held-out 评估记录

- branch: working tree
- post-commit: 未提交
- scope: task 内部文档

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 同步 full/high-motion point-flow 评估已完成、fitting 仍为 pilot 的状态。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 记录 baseline 的 full/high-motion val/test 以及 256 样本 q/wrist fitting pilot。

**改动原因**

用户要求评估 `cm_decoder_20260823_000423` baseline；评估确认 point-flow 泛化明显优于 zero-flow，但 q 分解仍存在 identity shortcut/不可辨识风险。

## 2026-08-24 — 同步 CmDecoder baseline 完成状态

- branch: working tree
- post-commit: 未提交
- scope: task 内部文档

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md` — 将旧版 Cm point-flow baseline 和随机 horizon pilot 标记为已完成，并记录当前验证结果与后续评估项。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 补充 baseline 的10 epoch训练结果。

**改动原因**

核验 `outputs/cmdecoder/cm_decoder_20260823_000423/` 后确认训练已经结束；原状态记录仍将该实验和随机 horizon pilot 标为进行中。

## 2026-08-23 — 切换 HRDexDB 规范数据路径

- branch: working tree
- post-commit: 未提交
- scope: task 内部 / 跨 task 数据路径

**文件**

- `src/task/CmDecoder/config.py`、`build_cache.py`、`grab_retarget.py`、`migrate_point_bindings.py` — 默认 HRDexDB 数据与 URDF 路径改为仓库 `dataset/HRDexDB/`。
- `src/task/CmDecoder/docs/logs/{status,memory,modification}_log.md` — 同步在途 cache 与规范路径。

**改动原因**

用户要求将 Cm 使用的 HRDexDB 非视频数据整理进 Ref2Dex；旧路径保留 symlink，使已经运行的四手型 cache builder 不丢进度。

## 2026-08-23 — 优化 HRDexDB candidate mask 构建

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部 / 跨 task 共享 cache

**文件**

- `src/task/CmDecoder/build_cache.py` — 使用 `scipy.spatial.cKDTree` 做逐帧 5 cm 邻域查询，替代全量 object-pool/hand 距离张量。

**改动原因**

保持 candidate mask 语义不变，同时降低 4096×1538 距离计算的 CPU 和峰值内存；单 episode 探针约从 192.6 s 降至 37.4 s。

## 2026-08-25 — 评估 geometry-only/no-time decoder 的中途 rollout

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部

**文件 / 产物**

- `output/research/arctic_mano_cm64_geometry_only_no_time_step26000_s01_box_use_01_left.{npz,png}` — step 26000 decoder 在固定 ARCTIC 32步窗口上的 teacher-forced 与 autoregressive rollout 结果。
- `src/task/CmDecoder/docs/logs/status_log.md`、`experiment_log.md` — 记录中途 checkpoint 的定量结果、适用边界和最终复评要求。

**改动原因**

在训练尚未结束时先检验 geometry-only/no-time Cm 是否已经改善 decoder 的闭环稳定性，同时使用不可变 step checkpoint 避免并行训练覆盖 `best.pt`。

**验证**

- checkpoint: step 26000 / epoch 7，GRAB val EPE=`3.477 mm`；
- ARCTIC teacher-forced EPE=`2.954 mm`；rollout mean/final EPE=`73.672/102.922 mm`；
- 评估正常完成，训练进程未中断。

## 2026-08-24 — 增加 ARCTIC 状态扰动纠偏诊断

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/CmDecoder/research/arctic_perturbation.py` — 固定未扰动 `cm_tokens`，对 ARCTIC 当前 MANO 点施加物体方向、远离物体方向和随机方向的 5/10/20 mm 平移，统计下一帧 EPE 与纠偏投影。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 记录 EXP-018 的设置、结果和结论，并补充 step 22000 best 的 teacher-forced/rollout 复评。
- `src/task/CmDecoder/docs/logs/status_log.md` — 更新 decoder 训练进度和扰动诊断状态。

**改动原因**

验证 DenseToken 间接提供的当前手—物空间条件是否能在状态偏离真实轨迹时产生纠偏流，而不是只在 teacher-forced 状态上取得较低单步误差。

**验证**

- `graspenv` GPU 诊断成功，结果写入 `output/research/arctic_mano_cm64_perturbation.json`。
- 5 mm 偏移存在弱纠偏，10–20 mm 偏移下纠偏快速减弱；step 22000 best 的 ARCTIC teacher-forced EPE 为 `2.398 mm`；训练模型和数据未被修改。

## 2026-08-25 — 启动 geometry-only/no-time C=64 Cm 的 MANO decoder 对照

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部 / 跨 task checkpoint 对照

**文件 / 运行入口**

- `output/exp/cmdecoder_grab_mano_cm64_geometry_only_no_time.log` — 新 decoder 训练日志。
- `outputs/cmdecoder/cmdecoder_grab_mano_pointflow_cm64_geometry_only_no_time_20260825_103711/` — 新 run 输出。
- `src/task/CmDecoder/docs/logs/status_log.md`、`experiment_log.md` — 记录当前 run 和对照假设。

**改动原因**

在保持 decoder、GRAB 数据、训练预算和评估口径不变的前提下，仅替换为 `use_object_context=false`、`use_time_condition=false` 的 C=64 Cm，检验 token 语义是否影响 rollout 稳定性。

**验证**

- Cm checkpoint 配置核验通过：C=64、geometry-only、no-time；
- decoder 初始化与 step 100 训练通过，当前无 OOM/NaN。

## 2026-08-25 — 更新 geometry-only/no-time decoder 对照进度

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/status_log.md`、`experiment_log.md` — 更新新 run 至 step 2000 / epoch 1 及首个验证结果。

**改动原因**

记录对照训练的第一阶段证据，避免将早期验证误判为最终 Cm 结构结论。

## 2026-08-24 — 增加 GRAB 训练、ARCTIC MANO 跨域评估入口

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部实验与配置

**文件**

- `src/task/CmDecoder/dataset_object_v2.py` — 按 combined object-v2 split 过滤 GRAB/ARCTIC sequence，提供 MANO point-flow decoder 的 train/val/test loader。
- `src/task/CmDecoder/dataset.py` — 接入 `object_v2_filter` 数据入口，保留 HRDexDB legacy/cache loader 不变。
- `src/task/CmDecoder/mano_grab_point_config.py` — 新增冻结 mixed C=64 Cm、GRAB-only 10 epoch point-flow decoder 配置。
- `src/task/CmDecoder/arctic_mano_eval.py` — 新增 ARCTIC teacher-forced 单步与 autoregressive rollout 评估、NPZ/PNG 导出。
- `src/task/CmDecoder/docs/logs/{status,experiment,decision,modification}_log.md` — 记录实验定义、当前运行和实现选择。

**改动原因**

用户要求排除 Inspire 形态因素：在 GRAB MANO 上训练 decoder，再用完全不读取 GRAB 帧的 ARCTIC MANO 轨迹测试跨域泛化。

**验证 / 状态**

- object-v2 loader：GRAB train/val/test=`1067/134/134` sequences，samples=`262513/33718/31567`；
- decoder forward smoke 通过，输出 `[B,1538,3]`；
- ARCTIC evaluator 使用历史 point-flow checkpoint 完成 2-frame smoke，生成 `output/research/arctic_mano_eval_smoke.{npz,png}`；
- 正式 GRAB decoder 已在 GPU6 启动，运行产物不纳入版本管理。

## 2026-08-24 — 调整 GRAB decoder 吞吐与 validation 频率

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部训练配置

**文件**

- `src/task/CmDecoder/mano_grab_point_config.py` — per-device/global batch 从16调为64，改为每2000 step validation；其余数据、模型和监督不变。

**改动原因**

首个 batch16 run 的稳定吞吐约60 samples/s，10 epoch预计耗时过长；GPU6显存余量充足，增大 batch 可更快获得首个 best checkpoint并执行ARCTIC评估。

**验证**

- 重启后 `train_setup` 报告 `per_device_batch=64/global_batch=64/total_steps=41020`，GPU6约占5.2GB，未发生OOM。
- 现有 CmDecoder 回归测试：`12 passed`；新增 loader、forward 和 ARCTIC 2-frame evaluator smoke 均通过。

## 2026-08-24 — 运行 C=64 GRAB decoder 的 ARCTIC 双模式评估

- branch: `oyx`
- post-commit: 未提交
- scope: task 内部实验产物与日志

**文件 / 产物**

- `output/research/arctic_mano_cm64_best_s01_box_use_01_left.npz`、`.png` — 使用 GRAB decoder best checkpoint 的 ARCTIC teacher-forced/rollout 结果。
- `src/task/CmDecoder/docs/logs/status_log.md`、`experiment_log.md` — 记录初步定量结果与 rollout 发散。

**验证**

- teacher-forced 单步 EPE `2.717 mm`，zero-flow `4.545 mm`；
- 32 帧 rollout 平均/末帧 EPE `113.957/193.094 mm`；
- 训练进程继续运行，未因本次评估中断。

## 2026-08-23 — 扩展 HRDexDB 多手型 layered cache builder

- branch: working tree
- post-commit: HEAD
- 范围: task 内部（依赖外部 HRDexDB 原始数据）

**文件**

- `src/task/CmDecoder/build_cache.py` — 按 episode 自动适配 MANO、Allegro-V5、Inspire-DFTP、Inspire-F1；统一读取对象 pose/mesh；保留机器人原始 q 维度并记录 `robot_type/q_semantics`。
- `src/task/CmDecoder/dataset.py` — 混合手型 point-flow 读取时将非 Inspire q 规范为六维零占位，避免 Allegro 16 维与 MANO 不可用 q 破坏 batch collate。
- `src/task/CmDecoder/docs/logs/{architecture,status,memory,modification}_log.md`、`docs/logs/memory_log.md` — 同步数据合同、状态、路径和修改记录。

**验证**

- 使用 `/home2/wyy/oyx_ws/HRDexDB/v0_nonvideo` 完成 robot smoke：Inspire-DFTP、Inspire-F1、Allegro-V5 各成功生成 `[T,1538,3]` 手点与对象点/法向。
- MANO smoke 成功生成固定 1538 face-center 点、JSON wrist pose 与对应 hand flow。
- 混合 `RandomHorizonGeometryDataset` 的 DataLoader batch shape 通过，`q_t/q_next` 均为 `[B,6]`；当前 flat point decoder 不使用 q。

## 2026-08-23 — 增加随机时间间隔点流 pilot

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/dataset.py` — 新增逐帧 geometry reader，每个当前帧稳定伪随机选择 `stride=1..10`，在线构造当前 wrist frame 的 hand/object flow，不复制 horizon task cache。
- `src/task/CmDecoder/random_horizon_config.py` — 新增 3 epoch pilot，关闭 Cm token sidecar，在线运行 frozen DenseToken/Cm head。

**改动原因**

验证固定 30Hz near-zero flow 是否是 decoder 迁移不佳的主要原因；本 pilot 不引入幅度分层、不改变 decoder 或 loss。

**验证 / 状态**

- 使用 `graspenv` 启动 3 GPU 训练；`fastwam` 缺少 `addict/spconv`，未用于正式实验。
- 训练输出：`outputs/cmdecoder/cm_decoder_flat_point_random_horizon_20260823_145753/`。

## 2026-08-22 — 完成全量 qt_cm 训练与 held-out 评估

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `outputs/cmdecoder/cm_decoder_20260822_143843/` — 完成30 epochs / 214,650 steps，生成 epoch 1 best checkpoint。
- `src/task/CmDecoder/docs/logs/{experiment,status,modification}_log.md` — 写入 full/high-motion val/test 结果及当前结论。

**改动原因**

训练自然完成后核验最佳 checkpoint，并补齐 object-disjoint held-out 评价，回答当前训练状态。

**验证**

- 训练正常退出，用时44分08秒；best 为 epoch 1 / step 7,155。
- best checkpoint 在 full 和 high-motion 的 val/test 四个口径上均优于各自 identity。
- 结论仅支持 `qt_cm` 主模型有效；在全量 `qt_only/cm_only` 完成前不归因于 Cm。

## 2026-08-22 — 导出全量 object-disjoint 3 Hz cache 并启动 qt_cm

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/build_cache.py` — 支持官方 compact pose v2→v1 fallback、可配置 object-disjoint split 及 split object 清单。
- `src/task/CmDecoder/build_horizon_cache.py` — 3 Hz 派生遇到零连续 pair 时排除 episode，并同步过滤 splits/objects。
- `tests/test_cmdecoder_build_cache.py` — 覆盖 compact 数字帧序、object-disjoint 不相交和零 pair 排除。
- `data/processed_data/cm_decoder/hrdexdb_inspire_f1/` — 576-episode 30 Hz geometry/task cache，约44 GB（被 gitignore 忽略）。
- `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/` — 568-episode 3 Hz task/token cache，约23 GB（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260822_143843/` — 全量 object-disjoint 3 Hz `qt_cm` 训练输出（运行中，被 gitignore 忽略）。
- `src/task/CmDecoder/docs/logs/{architecture,experiment,memory,repo_notes,status,modification}_log.md` — 同步数据合同、实验定义和运行状态。

**改动原因**

用户要求导出全量 cache，并基于此前确认的新 Cm checkpoint、3 Hz horizon 和 object-disjoint 语义启动一版 `qt_cm` 训练。

**验证**

- 单元测试3项通过；v1 fallback 端到端 FK/mesh/cache smoke 通过。
- 576个 geometry manifest 完整，pose source 为539个 compact v2 + 37个 compact v1。
- 3 Hz保留568个 episode、284,414 pairs；train/val/test object 交集为空。
- 568个 token 的 checkpoint hash 唯一且 shape 检查0错误。
- 两卡 DDP 已完成初始化并进入 epoch 1：global batch 32、total steps 214,650。

## 2026-08-22 — 完成新 Cm 的20-episode 3 Hz三组对照

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/episodes/*/cm` — 仅重建20个 episode 的 Cm token sidecar，绑定 object-v2 GRAB+ARCTIC best checkpoint。
- `outputs/cmdecoder/cm_decoder_20260822_{130259,130301,130305}/` — `qt_cm / qt_only / cm_only` 的有效30-epoch训练输出（被 gitignore 忽略）。
- `src/task/CmDecoder/docs/logs/{architecture,experiment,memory,repo_notes,status,modification}_log.md` — 记录新实验、cache checkpoint 绑定及后续 object-disjoint 约束。

**改动原因**

用户要求暂不导出全量 cache，先在现有20-episode 3 Hz设置上使用新 Cm checkpoint运行三组输入对照，并指定后续扩大数据采用 object-disjoint split。

**验证**

- 20/20 token manifest 的 checkpoint SHA256 一致；三组均完成30 epochs / 7,950 steps。
- 对各自 `best.pt` 完成全量 val/test 与 `max|Δq|>=0.5°` 高动作子集评估。
- 首次并行输出目录冲突且全局 batch 不一致的两项输出已在 EXP-011 标记为 `INVALID_IMPLEMENTATION`，不纳入结论。

## 2026-08-22 — 补齐 HRDexDB 非视频运动资产并核验完整性

- branch: working tree
- post-commit: HEAD
- 范围: 跨 task / 外部数据

**文件**

- `/home2/wyy/oyx_ws/HRDexDB/v0` — 新增1968个白名单文件，补齐 Inspire F1 arm 与 compact v1/v2 pose 等 CmDecoder 所需资产；未新增视频。
- `src/task/CmDecoder/docs/logs/{memory_log,status_log,repo_notes_log,modification_log}.md` — 记录下载结果、可用规模和 builder 的 compact-pose 待办。
- `docs/logs/{memory_log,repo_notes_log,modification_log}.md` — 同步仓库级外部数据事实。

**改动原因**

扩大 episode 前核查发现旧下载显式排除了 arm，且本地缺少大多数 object pose。改用官方 Hub、7897代理和精确白名单下载，避免视频与全仓库递归同步。

**验证**

- 1968/1968 文件成功，0失败；arm position/time 591/591，加载与 shape 检查0错误。
- compact v1/v2 pose 为591/555组；完整模态交集576组，其中539组优先使用v2、37组回退v1。
- Inspire F1 MP4 数量保持4119，未因本次下载增加。

## 2026-08-21 — 按 V1 修正时间/动作语义并加入 token cache 与对照接口

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/build_cache.py` — v4 cache，显式 source frame mapping、真实 delta time、连续 30 Hz mask、active-motion 字段、implementation fingerprint。
- `src/task/CmDecoder/dataset.py` — active-motion/30 Hz 筛选、episode/frame overrides、真实 delta_time、lazy Cm token sidecar。
- `src/task/CmDecoder/build_cm_cache.py` — GPU 预提取 frozen Cm tokens 并以 checkpoint/task hash 校验。
- `src/task/CmDecoder/model.py` — `qt_cm / qt_only / cm_only`、shuffled-flow、q scale 支持。
- `src/task/CmDecoder/runner.py` — scaled target loss 与 identity MAE。
- `src/task/CmDecoder/config.py` — v4 manifest、active-motion、decoder input 和 token cache 配置。
- `src/task/CmDecoder/docs/logs/{architecture_log,repo_notes_log,experiment_log}.md` — 同步 V1 研究状态和证据。

**改动原因**

按 `docs/指导/V1.md` 修正静止帧 overfit 误判、时间语义和 cache 可复现性，并为 Cm 独立贡献对照准备接口。

## 2026-08-21 — 完成 active-motion Decoder 输入对照

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-003，记录 qt_cm、qt_only、cm_only、shuffled-flow 对照。
- `outputs/cmdecoder/cm_decoder_20260821_{193337,193442,193549}/` — 对照训练输出（被 gitignore 忽略）。

**改动原因**

用户要求先运行 V1 中的 overfit 对照，检查 Cm 相对当前 q shortcut 的独立作用。

**结果摘要**

`qt_cm=0.294°`，`qt_only=0.302°`，`cm_only=0.453°`，`shuffled-flow=0.464°`，identity=`0.647°`；当前结论为 INCONCLUSIVE。

## 2026-08-21 — 改为分层增量 cache 并构建 50 episode 子集

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/build_cache.py` — 物体分层选集、episode 多进程 builder、geometry/task 分层、source SHA256 自动失效。
- `src/task/CmDecoder/dataset.py` — 新增 v3 cache 的 mmap lazy Dataset，训练阶段不再执行 FK。
- `src/task/CmDecoder/config.py` — 默认指向 50-episode v3 manifest。
- `src/task/CmDecoder/docs/logs/{architecture_log,repo_notes_log,decision_log}.md` — 同步 cache 架构、路径和自主选择。

**改动原因**

用户要求停止全量旧 cache，支持未来增量字段，并先构建 50 个 episode 观察小规模训练。

**验证**

- 50 episodes：40 train / 5 val / 5 test；32,766 samples；3.6 GB。
- lazy DataLoader batch shape 与 frozen-Cm GPU 前向通过。
- 重跑 builder 能根据 source SHA256 判定 cache 命中。

## 2026-08-20 — 创建 frozen-Cm Inspire F1 动作重建首版

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/config.py` — 训练、数据和 checkpoint 配置。
- `src/task/CmDecoder/model.py` — 加载并冻结 Cm checkpoint，接入 q 解码 MLP。
- `src/task/CmDecoder/dataset.py` — HRDexDB Inspire F1 相邻帧、URDF mesh surface sampling 和 Cm 输入构造。
- `src/task/CmDecoder/runner.py` — 复用 BaseRunner 的训练和 q 角度指标。
- `src/task/CmDecoder/train.py` — 训练入口。
- `src/task/CmDecoder/docs/logs/*` — 任务架构、决策和修改记录。

**改动原因**

实现用户确认的 frozen-Cm action-conditioned reconstruction：当前 Inspire F1 手部 q 和相邻帧 hand mesh flow 编码到 Cm，Decoder 重建下一帧 6 维手指 q。

**实现状态**

代码已通过 Python compile 检查；随后完成真实单 episode/30 帧 Dataset smoke：得到 29 个相邻 pair，hand/object shape 分别为 `[1538,3]` / `[512,3]`，hand flow 与 q delta 均出现非零变化。选定 checkpoint 成功加载为 16×256 Cm，Cm trainable parameter 为 0，Decoder trainable parameter 为 2,241,810。

### 实现备注

- HRDexDB 的 NumPy hand/time 数组为 object dtype，读取时显式允许 pickle 后立即转为数值数组；
- 动态导入 HRDexDB dataclass helper 时先注册 `sys.modules`，兼容 Python 3.8；
- Inspire F1 URDF 的前 6 个 qpos 为 arm、后 6 个为 hand；旧版曾错误地将 arm 固定为零，见下方修正记录；
- 本次 smoke 只验证实现链路，不产生科研结论，不新增 EXP。

## 2026-08-20 — 增加 Inspire F1 Viser 几何查看器

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/viewer.py` — 单 episode/frame slider、播放、object samples、hand mesh、1538 hand samples 和 flow segment 可视化。

**改动原因**

参考 `InteractionDynamics/viewer_v2` 和 `viewer_gty`，在训练前检查 HRDexDB 坐标系、URDF hand surface sampling、object pose 和 hand flow correspondence。

**验证**

`--help` 与 Python compile 通过；`apple/2` 场景可启动 Viser 服务并监听 `8095` 端口。

## 2026-08-21 — 修正 arm FK、C2R 世界坐标和腕部 hand-root 数据链路

- branch: working tree
- post-commit: HEAD
- 范围: task 内部（依赖 HRDexDB arm q 数据）

**文件**

- `src/task/CmDecoder/dataset.py` — 读取真实 arm q 做完整 FK；将 robot-base 几何经 `C2R` 转到 HRDexDB world，再以当前腕部 `base_link` 作为 Cm 的 hand-root frame；缓存 key 更新为新坐标版本。
- `src/task/CmDecoder/viewer.py` — 统一使用 world 坐标显示，真实 arm q 参与腕部位姿，但只渲染手部 mesh、采样点和 flow，不渲染 arm。

**改动原因**

修复用户发现的腕部不动和手物体坐标不一致问题。手部 q 不含 arm 是 Decoder 语义，不应被解释为 FK 时把 arm 置零。

**验证**

- 已下载并验证 HRDexDB Inspire F1 arm `position.npy/time.npy` 为实际 NumPy 数组；`apple/2` arm 六维范围均非零。
- Dataset、viewer 通过 compile；`apple/2` Viser 在 world 坐标启动成功。
- 训练输入仍为 hand-root 当前帧坐标，Decoder target 仍只有 6 个手指关节角。

## 2026-08-21 — 修正 robot 与物体轨迹的起始时间错位

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/dataset.py` — 优先使用 HRDexDB `raw/timestamps/timestamp.npy` 对 arm+hand q 重采样，再和 object pose 序列对齐。
- `src/task/CmDecoder/viewer.py` — viewer 使用同一视频时间轴对齐 q 与物体 pose。

**改动原因**

HRDexDB 的 robot 流比视频/物体流早约 2.65 秒。原实现把两个流的第 0 帧直接配对，会造成明显的时序穿模；现在以视频时间戳作为共同时间轴。

**验证**

`apple/2` 现在 viewer 帧数为 501（与 pose 数一致），arm q 仍保持非零运动；代码编译和真实数据 smoke 均通过。

## 2026-08-21 — 补充 CmDecoder 张量架构文档

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/architecture_log.md` — 以张量 shape、坐标系、冻结边界和 loss 简洁记录当前架构。

**改动原因**

用户要求明确记录 CmDecoder 的张量架构，便于训练和后续复现实验。

## 2026-08-21 — 完成单 episode frozen-Cm overfit sanity check

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 记录 EXP-001 的命令、结果、解释和证据位置。
- `outputs/cmdecoder/cm_decoder_20260821_161226/` — overfit 训练输出（被 gitignore 忽略）。

**改动原因**

用户要求先做一版过拟合训练，验证冻结 Cm 到 Decoder 的训练链路。

## 2026-08-21 — Viewer 增加采样点 Poisson mesh 与尺寸控制

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/viewer.py` — 增加 GT/Reconstructed/Both/Hidden 手部 mesh 模式；以 1538 个手部点和法向做 Open3D Poisson 重建；增加 hand/object point size、flow line width、flow length scale 滑块。

**改动原因**

用于直观检查 Cm 输入的手部采样点能否较好复现原始 URDF GT mesh，同时让不同尺度的点云和 flow 更易观察。

**验证**

- `apple/2` 第 200 帧产生 7646 顶点、15220 三角面；首次建拓扑约 0.93 秒，后续帧 KNN 变形约 0.004 秒。
- viewer 在 `8098` 端口成功启动；物体点数保持 512，手部点数保持 1538。

## 2026-08-21 — 完成 20-episode qt_cm 离线训练

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-004，记录 20-episode qt_cm 训练配置、结果和后续判断。
- `outputs/cmdecoder/cm_decoder_20260821_195912/` — 训练 checkpoint、metrics 和离线 W&B 日志（被 gitignore 忽略）。

**改动原因**

用户要求在 20 个 episode 上运行 qt_cm 版本训练；W&B 在线证书异常，因此本次采用 offline 模式完成训练。

**对应指导**

`docs/指导/V1.md`

**训练状态**

30 epochs / 9,600 steps 已完成；best checkpoint 位于 `outputs/cmdecoder/cm_decoder_20260821_195912/checkpoints/best.pt`。

## 2026-08-21 — 完成 qt_only 与 cm_only 对照训练

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-005，记录两组 20-episode 输入消融。
- `outputs/cmdecoder/cm_decoder_20260821_200732/` — qt_only 训练输出（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260821_201002/` — cm_only 训练输出（被 gitignore 忽略）。

**改动原因**

用户要求继续运行除 shuffled-flow 外的另外两组对照；两组均沿用 EXP-004 的训练预算和数据划分。

**对应指导**

`docs/指导/V1.md`

**训练状态**

两组均完成 30 epochs / 9,600 steps；最佳验证 q MAE 分别为 qt_only 1.423°、cm_only 5.028°。

## 2026-08-21 — 完成 active-motion 子集评估

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-006，记录活动帧验证/测试结果。

**改动原因**

用户要求单独评估活动帧上的三组模型效果；采用现有 `0.5°` 阈值和 30 Hz 相邻帧定义，不改动模型或训练结果。

**对应指导**

`docs/指导/V1.md`

**评估状态**

完成 val 240 samples、test 162 samples 的 active-motion 评估；qt_only test q MAE 1.691°，qt_cm 4.821°，cm_only 9.262°。

## 2026-08-21 — Decoder 改为预测关节残差

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/config.py` — 新增默认 `prediction_target=delta_q`。
- `src/task/CmDecoder/model.py` — Decoder 输出 `pred_delta_q`，通过 `q_t + pred_delta_q` 重建下一帧 q。
- `src/task/CmDecoder/runner.py` — SmoothL1 监督改为 `q_next-q_t`，重建 q 指标保持不变。
- `src/task/CmDecoder/docs/logs/architecture_log.md` — 同步残差预测张量流。
- `src/task/CmDecoder/docs/logs/decision_log.md` — 记录旧 direct-q checkpoint 兼容策略。

**改动原因**

用户要求预测动作残差而非直接预测绝对关节角，以便模型聚焦相邻 30 Hz 帧的关节变化。

**对应指导**

`docs/指导/V1.md`

## 2026-08-21 — 完成 20-episode 残差预测对照

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-007，记录三组残差训练及全量/活动帧评估。
- `outputs/cmdecoder/cm_decoder_20260821_210013/` — residual qt_cm 输出（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260821_210256/` — residual qt_only 输出（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260821_210526/` — residual cm_only 输出（被 gitignore 忽略）。

**改动原因**

用户要求在 20 个 episode 上比较残差预测版本；保持数据和训练预算一致，并额外复核 active-motion 子集。

**对应指导**

`docs/指导/V1.md`

**实验状态**

三组训练与全量 test、active-motion val/test 评估均完成；qt_only 最佳，但未稳定优于 identity。

## 2026-08-21 — 统计 20-episode 30 Hz q 差分分布

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-008，记录 13,039 个相邻 pair 的 q 差分和时间间隔统计。

**改动原因**

用户要求确认 30 ms 相邻 GT 帧的 q 是否普遍只变化很小；本次只读统计 cache，不改变训练代码或数据。

**对应指导**

`docs/指导/V1.md`

**统计状态**

约 85.18% 的 pair 满足最大关节变化小于 0.5°，约 6.50% 达到至少 1°；dt 中位数约 30 ms，但存在 105.8 ms 最大值。

## 2026-08-21 — 统计多时间间隔 q 差分

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-009，记录 stride 1/2/3/5/6/10/15/30 的 q 差分统计。

**改动原因**

用户要求确认改用 3 Hz 等更长时间间隔后相邻 GT q 的变化幅度；本次从已有 geometry cache 的 `q_full` 只读统计，不改变训练代码。

**对应指导**

`docs/指导/V1.md`

**统计状态**

3 Hz（stride=10）共 10,948 pairs，最大关节变化 P50=0.505°、P90=6.819°，超过 0.5° 的比例为 51.04%。

## 2026-08-21 — 完成 3 Hz 三组残差对照训练

- branch: working tree
- post-commit: HEAD
- 范围: task 内部

**文件**

- `src/task/CmDecoder/build_horizon_cache.py` — 从 v4 geometry 派生 stride=10 的 3 Hz task cache。
- `src/task/CmDecoder/build_cm_cache.py` — 支持 `--set`，为派生 cache 预计算 Cm tokens。
- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-010，记录 3 Hz 三组训练和评估。
- `data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/` — 3 Hz cache（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260821_214719/` — 3 Hz qt_cm 输出（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260821_214933/` — 3 Hz qt_only 输出（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260821_215143/` — 3 Hz cm_only 输出（被 gitignore 忽略）。

**改动原因**

用户要求将时间间隔改为 3 Hz，并继续运行 qt_cm、qt_only、cm_only 三组对照。

**对应指导**

`docs/指导/V1.md`

**实验状态**

3 Hz cache、Cm token sidecar、三组 30 epoch 训练以及全量/高动作 val/test 评估均完成；qt_only 最佳但未超过 identity。

## 2026-08-22 — 完成全量 object-disjoint 3 Hz 归因对照

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-013，记录 `qt_only/cm_only` 5 epoch 训练及三组统一评估结果。
- `src/task/CmDecoder/docs/logs/status_log.md` — 更新为全量三组对照已完成，并记录当前可靠结论与风险。
- `outputs/cmdecoder/cm_decoder_20260822_152742/` — 全量 object-disjoint `qt_only` 5 epoch 输出（被 gitignore 忽略）。
- `outputs/cmdecoder/cm_decoder_20260822_153543/` — 全量 object-disjoint `cm_only` 5 epoch 输出（被 gitignore 忽略）。

**改动原因**

用户要求其余两组各运行 5 epochs，以归因 EXP-012 的 `qt_cm` 改善来源。

**实验状态**

两组均完成 35,775 steps，并以各自 full validation 最优 checkpoint 完成 full/high-motion val/test 评估。结果支持跨物体增益主要来自 Cm token；`qt_cm` 与 `cm_only` 当前近似持平。

## 2026-08-22 — viewer 加入 Decoder 预测手叠加对比

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/viewer.py` — 默认加载全量 `qt_cm` best checkpoint，允许 CLI 覆盖；按 checkpoint cache pair 推理，并增加当前/GT/预测三张手 mesh 的独立 checkbox 与对比状态栏。
- `tests/test_cmdecoder_viewer.py` — 覆盖 horizon pair 映射和固定 arm、替换手指 q 的组合逻辑。
- `src/task/CmDecoder/docs/logs/{architecture,status,decision,modification}_log.md` — 同步可视化入口、坐标约定和当前状态。

**改动原因**

用户要求 viewer 可添加 CmDecoder 预测的优化后手，默认使用 `qt_cm` 且 checkpoint 可通过运行参数指定；当前手、GT 手和预测手可独立开关并适合透明叠加。

**验证**

- `tests/test_cmdecoder_viewer.py` 与 `tests/test_cmdecoder_build_cache.py` 共6项通过。
- 默认 checkpoint 在 GPU 上成功读取 `inspire_f1/apple/2` 的329个3 Hz pair及匹配 token cache，生成三张 FK mesh 和预测 q。
- Viser 在 `127.0.0.1:8097` 完成服务、GUI 和首帧 mesh 初始化，无运行时错误；smoke test 后由 `timeout` 正常结束。

## 2026-08-22 — 新增逐手点 Cm flow 与 q fitting 模型

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/point_model.py` — 新增不读取 q 的逐手点 slot-routing flow decoder，使用当前手点/法向、frozen DenseToken `z_hand`/contact 和 Cm tokens。
- `src/task/CmDecoder/q_optimizer.py` — 新增 Inspire F1 fixed-correspondence 可微 FK 与从 `q_t` 初始化的 bounded q fitting。
- `src/task/CmDecoder/point_config.py` — 新增全量 object-disjoint 3 Hz 新模型配置，保留原 Config/模型作为 baseline。
- `src/task/CmDecoder/runner.py` — 增加 hand-flow Smooth-L1、point EPE/RMSE 和 zero-flow 指标分支。
- `src/task/CmDecoder/viewer.py` — 按 checkpoint 动态加载 baseline 或逐点模型；逐点输出经 q fitting 后进入原三手叠加视图。
- `tests/test_cmdecoder_q_optimizer.py` — 覆盖关节旋转、q 初值优化、误差下降与 joint bound。
- `src/task/CmDecoder/docs/logs/{architecture,status,decision,modification}_log.md` — 同步新模型合同、工程选择和当前状态。

**改动原因**

用户要求保留整体 q 回归为 baseline，新增不以 q 为网络输入的逐手点 Cm decoder；训练预测对应点 flow，推理时再从 `q_t` 初始化优化得到6维手指 q，且优化不参与训练。

**验证**

- 8项 CmDecoder cache/viewer/q-optimizer 测试通过，`git diff --check` 通过。
- 真实 cache 点在 `q_t` 反绑再 FK 后平均/最大误差约 `0.000004/0.00003 mm`。
- 真实高运动 batch 的逐点模型 forward/backward shape 正确、梯度有限；trainable 参数257,156。
- `outputs/cmdecoder/cm_decoder_20260822_193946/` 完成1 episode × 16 pairs、8 steps 的端到端训练/验证/checkpoint smoke。
- 新模型 best checkpoint 已通过 viewer 的 `pred_hand_flow → q fitting → predicted mesh` 分支。

## 2026-08-22 — 为 baseline 与逐点路线补全相对腕部运动

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/dataset.py` — 从 geometry sidecar 只读派生 horizon pair 的相对 wrist 平移与 rotvec 监督，不重写 task/token cache。
- `src/task/CmDecoder/model.py`、`config.py`、`wrist_baseline_config.py` — 新增 wrist-aware 12维 baseline 输出、loss 配置及独立5-epoch object-disjoint 3 Hz 训练入口，同时兼容历史6维 checkpoint。
- `src/task/CmDecoder/runner.py` — 增加 wrist 平移/旋转 loss、mm EPE 和旋转测地角指标。
- `src/task/CmDecoder/q_optimizer.py`、`point_config.py` — 将后处理扩为从 `q_t` 和单位 wrist 变换初始化的联合 wrist SE(3)+q 拟合。
- `src/task/CmDecoder/viewer.py` — GT 与预测手应用相对 wrist 变换；未来 arm FK 不参与目标手重建。
- `tests/test_cmdecoder_q_optimizer.py`、`tests/test_cmdecoder_viewer.py` — 增加 rotvec 数值稳定性、联合拟合回归及当前 arm 仅作坐标框架的测试/命名约定。
- `src/task/CmDecoder/docs/logs/{architecture,status,decision,modification}_log.md` — 记录统一的腕部运动语义、兼容边界和当前状态。

**改动原因**

用户确认 baseline 使用方案 A 直接预测腕部，且两条路线均预测当前腕到目标腕的相对运动，不再用未来 arm FK 求目标腕。

**验证**

- wrist-aware baseline 完成1 episode × 16 pairs、8 steps 的训练/验证/checkpoint smoke：`outputs/cmdecoder/cm_decoder_20260822_195837/`。
- 真实高运动样本联合拟合100步后 point EPE 从约 `2.270 mm` 降至 `0.859 mm`，手指 identity MAE 从 `5.734°` 降至 `1.668°`。
- 历史6维 baseline、新12维 baseline 和逐点 checkpoint 均通过 viewer 模型分支兼容 smoke。

## 2026-08-22 — 完成 wrist-aware baseline 三组3-epoch对照

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/docs/logs/experiment_log.md` — 新增 EXP-014，记录三组配置、best checkpoint 的 full/high-motion q 与 wrist 指标、identity wrist 对照及结论。
- `src/task/CmDecoder/docs/logs/status_log.md` — 将三组训练更新为已完成，并记录当前 loss 权重风险与下一步。
- `src/task/CmDecoder/docs/logs/modification_log.md` — 记录本次实验文档更新。

**改动原因**

用户要求使用当前剩余 GPU，将 wrist-aware baseline 的 `qt_cm / qt_only / cm_only` 三组在全量 object-disjoint 3 Hz 数据上各训练3 epochs。

**验证**

- 三组均完成21,465 steps，无 OOM；`qt_cm/cm_only` best epoch=1，`qt_only` best epoch=3。
- 三个 `best.pt` 均完成 full/high-motion val/test 统一评估，并补算零相对腕运动 baseline。
- 结果表与 checkpoint 路径见 EXP-014。

## 2026-08-22 — 将 wrist-aware baseline 改为纯固定对应点 loss

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/config.py`、`wrist_baseline_config.py` — 增加 point/parameter loss 权重与米制 Smooth-L1 配置；wrist baseline 默认只启用 point loss并按 point EPE 选模。
- `src/task/CmDecoder/q_optimizer.py` — 抽出从预测 q、相对 wrist 和当前点绑定进行可微目标点重建的公共函数。
- `src/task/CmDecoder/runner.py` — baseline 分支加入固定对应点重建、纯点 loss、point EPE/RMSE 与 identity-hand 指标；原参数 loss 保留并由权重控制。
- `tests/test_cmdecoder_q_optimizer.py` — 覆盖 point loss 到 q、translation、rotvec 的有限非零梯度。
- `src/task/CmDecoder/docs/logs/{architecture,status,experiment,decision,modification}_log.md` — 同步训练目标、数值选择、EXP-015 和运行状态。

**改动原因**

用户要求 baseline 输出保持 q 与相对位姿，但改用采样点几何误差反传；先试纯点 loss，原参数 loss 保留且权重置0。

**验证**

- 10项 CmDecoder q-optimizer/viewer/cache 测试通过，`git diff --check` 通过。
- GT 参数重建真实 cache target point 的平均/最大误差为 `0.000007/0.000170 mm`。
- 原始米制 point loss 的真实8-step smoke 完成训练、验证和 checkpoint，梯度范数约0.312且未触发裁剪。

### 后续有效性修正

全量训练 epoch 1 暴露3 Hz v1 cache 的 `hand_flow` 使用目标腕坐标系、丢失 wrist 刚体运动。三组进程已停止，EXP-015 标记为 `INVALID_IMPLEMENTATION`；近静止 smoke 的 GT 点重建结论不得外推到全量数据。纯点 loss 代码本身保留，等待版本化 cache 修复后再验证。

## 2026-08-22 — 修复 wrist-aware cache 并完成纯点 loss v2 三组复跑

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/build_horizon_cache.py` — 将目标手点统一变换到当前腕坐标系，增加显式输出版本与 v2 hand-flow 语义元数据。
- `src/task/CmDecoder/dataset.py`、`point_config.py` — wrist-aware 训练切换至 v2 manifest，并按 `required_hand_flow_frame=current_wrist` fail-fast。
- `tests/test_cmdecoder_horizon_cache.py` — 增加目标点必须保留当前腕到目标腕运动的回归测试。
- `src/task/CmDecoder/docs/logs/{architecture,status,memory,experiment,modification}_log.md` — 同步版本边界、cache 路径、训练性能和 EXP-015 修正结果。

**改动原因**

v1 目标点坐标系会消除腕部运动；用户要求保留 v1，创建语义正确的 v2 cache，重导全量 token 并重跑纯点 loss 三组实验。

**验证**

- v2 共568个 episode、284,414 pairs；task/token schema、shape、dtype、checkpoint/hash 与 hand-flow 语义全量扫描0错误。
- GT q+wrist 重建 target point 的抽样平均/最大 EPE 约 `0.000007/0.000226 mm`。
- 11项 CmDecoder 相关测试通过，`git diff --check` 通过。
- 三组均完成3 epochs / 21,465 steps，无 OOM、无梯度裁剪；best checkpoint 已完成 full/high-motion val/test 统一评估，结果见 EXP-015。

## 2026-08-23 — 预计算点绑定并向量化 FK 点变换

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/build_cache.py`、`build_horizon_cache.py` — 在 layered v4/v2 cache 中保存静态 `hand_point_link_index` 与 `hand_points_local`。
- `src/task/CmDecoder/migrate_point_bindings.py` — 复用已有全量 geometry，为576个 v4 episode 和568个 v2 episode补写绑定，不重算逐帧几何。
- `src/task/CmDecoder/dataset.py` — 可选加载静态绑定，并处理 DataLoader collate 后的 batch 维度。
- `src/task/CmDecoder/q_optimizer.py`、`runner.py` — 跳过当前 q 反绑，按全 URDF link index gather 后批量变换1538个点。
- `src/task/CmDecoder/config.py`、`wrist_baseline_config.py` — 增加 `use_cached_point_bindings` 开关，wrist baseline 默认启用。
- `tests/test_cmdecoder_q_optimizer.py`、`tests/test_cmdecoder_build_cache.py` — 增加 cache binding 接口覆盖。

**改动原因**

用户确认先按旧版 Cm 做逐点新架构10 epoch训练；在训练前消除纯点 baseline 中每 batch 的当前 q FK/逆变换和逐 link 点变换开销，同时保持固定 correspondence 和 loss 语义不变。

**验证**

- 12项 CmDecoder 测试通过。
- 单 episode真实 cache 新旧重建最大差约 `1.2e-7 m`，平均误差 `4.2e-5 mm`。
- CPU batch=32 点重建约 `2.7×` 加速。
- 旧版 Cm、v2 cache 的逐点新架构10 epoch训练已启动：`outputs/cmdecoder/cm_decoder_20260823_000423/`。

## 2026-08-23 — 增加 30Hz GRAB 右手到 Inspire F1 重定向 smoke test

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/grab_retarget.py` — 读取 subject-template GRAB 右手 30Hz cache，使用 GRAB hand flow 生成 Cm token，以物体中心外侧 12cm 接近位姿和关节限位中点 q0 初始化 Inspire F1，执行 point-flow→q/wrist fitting，并导出 NPZ/PNG。
- `src/task/CmDecoder/q_optimizer.py` — 为静态采样点增加 hand-root 法向 FK 变换接口，供 decoder 接收机器人当前几何。

**改动原因**

验证“GRAB 参考轨迹生成 Cm、机器人当前状态送入 decoder”的无配对重定向路径，不把 GRAB 人手点云误当作 Inspire F1 的真实 q 监督。

**验证**

- `graspenv` CPU/GPU 依赖检查完成；CPU 由于 spconv implicit-gemm 仅支持 CUDA，预期失败。
- CUDA 单帧 smoke test 成功，输出 `/tmp/grab_rt.npz` 与 `/tmp/grab_rt.png`。

## 2026-08-23 — 增加 30Hz Cm-flat 逐点 baseline

- branch: working tree
- post-commit: HEAD
- scope: task 内部

**文件**

- `src/task/CmDecoder/flat_point_model.py` — 将每个当前手点 xyz 与 flatten 后的 `[16,256]` Cm token 拼接，经共享 MLP 预测该点 3D flow。
- `src/task/CmDecoder/flat_point_config.py` — 使用 HRDexDB v4 30Hz 相邻帧和缓存 Cm token，配置10 epoch训练。

**改动原因**

建立不含 DenseToken/法向/slot edge routing 的快速结构 baseline，隔离“Cm-flat + 当前点坐标”对点流预测的贡献。

**验证 / 状态**

- 模型和配置可加载，训练参数量约95M（其中 frozen Cm 不更新，trainable MLP 约2.2M）。
- 30Hz v4 token sidecar 仅预先覆盖部分 episode；576 episode 的全量 token 预计算已在3张 GPU 上运行，完成后再启动正式训练。
## 2026-08-23 — 优化 HRDexDB candidate mask 构建

- branch: 当前工作分支
- post-commit: 未提交
- scope: task 内部 / 跨 task 共享 cache

**文件**

- `src/task/CmDecoder/build_cache.py` — 使用 `scipy.spatial.cKDTree` 做逐帧 5 cm 邻域查询，替代全量 object-pool/hand 距离张量。

**改动原因**

保持 candidate mask 语义不变，同时降低 4096×1538 距离计算的 CPU 和峰值内存；单 episode 探针约从 192.6 s 降至 37.4 s。
## 2026-08-30 — 物体表面采样 cache 移除 5cm 查询

- branch: `feature/modular-component-runtime`
- scope: HRDexDB CmDecoder cache builder/loader
- change_level: L2（用户已确认）
- approval: 用户确认物体表面随机采样 512 点，不再按 5cm 邻域过滤

**修改与验证**

- `src/task/CmDecoder/build_cache.py` 直接将完整 4096 点 object surface pool 标为有效，保留旧 mask 文件名以维持 manifest/loader 兼容。
- 新 cache 根目录为 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_20260830`。
- 相关回归测试通过；导出任务已重新启动。
## 2026-08-30 — Inspire-F1 逐帧 mesh/FK 优化与 CUDA 批量变换

- branch: `feature/modular-component-runtime`
- scope: HRDexDB cache builder
- change_level: L2（用户已确认）
- approval: 用户要求继续优化并启用 GPU

**修改**

- `src/task/CmDecoder/build_cache.py`：每帧只调用一次 `compute_link_transforms`，直接变换 1538 个已采样局部表面点；物体表面点和法线也支持批量 CUDA 变换。移除逐帧百万级完整机器人 mesh 构造路径。
- builder 新增 `--device cpu|cuda[:index]`，cache manifest 记录 `transform_device`。

**验证**

- 代表性 episode：旧路径约 700–1,500 秒；优化后完整 builder CPU 5.11 秒、CUDA 4.44 秒。
- CPU/CUDA sampled-point 输出最大差 `5.96e-8`，与旧缓存手点最大差约 `2.09e-7`。
- 相关测试通过，当前全量 Inspire-F1 使用 `--workers 8 --device cuda:0` 导出。

## 2026-08-30 — 重新计算 Inspire-F1 5cm candidate mask sidecar

- change_level: L2（恢复数据帧过滤判据，新增 cache sidecar）
- approval: user-approved
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- post-commit: 未提交
- scope: Task:CmDecoder geometry cache / Cm 数据筛选

**文件与产物**

- `src/task/CmDecoder/recompute_candidate_mask.py` — 新增可复现的 CPU/GPU 逐帧 5cm 查询工具，支持批量 `torch.cdist` 和原子 sidecar 写入。
- `data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/episodes/*/geometry/obj_candidate_mask_5cm_recomputed.npy` — 576 个 episode 的独立重算 mask（生成数据，不纳入版本控制）。
- `src/task/CmDecoder/docs/logs/status_log.md` — 记录进度、统计和当前未切换状态。

**原因**

恢复旧版“没有任何 5cm object candidate 的 current frame 不进入训练”的帧级规则，同时不改变新 cache 的 4096 点 object surface pool 和运行时 512 点随机采样。sidecar 与当前全真兼容文件分离，避免训练进程读到半成品或发生行空间变化。

**验证**

- 命令：`CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python src/task/CmDecoder/recompute_candidate_mask.py --root data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4 --device cuda:0 --frame-batch 32`。
- 结果：576/576 episode、362,027 帧全部生成；输出 mask 均为 `[T,4096] bool`。全量 point-level 有效率 `0.235443`；train/val/test transition-level 有效帧率分别为 `56.015%/58.529%/59.248%`。
- GPU7 计算耗时约 2.2 分钟；当前 GPU0/1/2 的 Cm 混合训练进程保持运行，未被此任务停止或修改。

## 2026-08-31 — 切换到当前 Cm 的 Inspire-F1 逐点解码器训练

- change_level: L3（停止三卡长时任务并启动新的三卡训练）
- approval: user-approved
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- post-commit: 未提交
- scope: Task:Cm 与 Task:CmDecoder 训练进程切换

**文件与运行**

- 停止当前 `src.task.Cm.src.train` 三卡进程，保留其 `best.pt/latest.pt`；
- 预检使用当前 Cm `best.pt`、`CmPointFlowModel` 和 Inspire-F1 object-disjoint v2 cache，前向/反向通过，输出手流形状 `[B,1538,3]`，可训练参数 `21,284`；
- 启动 `outputs/cmdecoder/cm_decoder_20260831_165400`，冻结 Cm checkpoint 为 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/checkpoints/best.pt`，`meta.use_cached_cm_tokens=false`，GPU0/1/2、global batch=48、30 epoch/143110 steps。

**验证**

- 旧 Cm torchrun 与 rank 已退出；
- 新 decoder step `100` 已写入训练日志，初始 hand-flow EPE=`7.201 mm`、zero-flow=`10.572 mm`；无 OOM/NaN/NCCL 错误。

## 2026-08-31 — 因坐标系不一致停止短暂 decoder run

- change_level: L3（停止不满足坐标合同的长时任务，未改动数据）
- approval: research-safety stop，依据用户已确认的坐标约束
- skills_used: research-change-control, research-experiment-workflow
- branch: `feature/modular-component-runtime`
- post-commit: 未提交
- scope: Task:CmDecoder 训练有效性审计

**发现与处置**

- `outputs/cmdecoder/cm_decoder_20260831_165400` 使用的 `hrdexdb_inspire_f1_3hz/v2` task cache 由 `build_horizon_cache.py` 生成，`hand_points`、`obj_points` 和 `hand_flow` 均在 current-wrist frame；其 manifest 明确 `hand_flow_frame=current_wrist`、`horizon_stride=10`。
- 当前冻结 Cm `best.pt` 的 checkpoint metadata 为 `coordinate_frame=object_pose_t`；Cm PointFlow 前向虽然可运行，但坐标语义不一致。
- 该 run 仅到 step `300` / epoch `1`，未写入 checkpoint，已安全停止；其 loss/EPE 不纳入实验结论。

**后续边界**

- 需要从 `hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4` 的 world geometry 重新生成 object-pose task cache（并明确 stride/hand-flow frame），通过坐标合同与 loader/model smoke 后才能重启训练；旧 v2 task arrays 和 C=256 token sidecar 不复用。
