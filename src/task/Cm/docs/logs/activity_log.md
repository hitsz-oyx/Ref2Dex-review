# Cm 活动记录

- scope: task:Cm
- last_updated: 2026-09-01
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- current_plan: [`src/task/Cm/docs/plan/V1.2.md`](../plan/V1.2.md)（final）
- related: [架构记录](architecture_log.md)、[接手记忆](repo_memory.md)、[实验记录](experiment_log.md)、[训练结果对比](../../results/object_v2_training_comparison.md)

## 2026-09-01 21:12:26 +0800 — V1.2.15 活动记录切换

- activity_id: ACT-20260901-211226-CM
- timestamp: 2026-09-01 21:12:26 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 根级治理方案获用户直接批准；本 Task 仅迁移日志入口，不改变 Cm 研究语义
- skills_used: research-change-control, research-experiment-workflow
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: root/governance mirror; task:Cm 日志入口和文档导航；不涉及模型、数据、GT、坐标、split、checkpoint 或训练变量

**文件**
- [`activity_log.md`](activity_log.md) — 从切换点开始作为 Cm 的活动时间线，保留原 `status_log.md` 历史内容。
- [`../README.md`](../README.md)、[`architecture_log.md`](architecture_log.md)、[`experiment_log.md`](experiment_log.md) — 更新当前入口和记录职责说明。
- [`repo_memory.md`](repo_memory.md)、[`decision_log.md`](decision_log.md) — 更新活动入口链接。
- [`../../research/README.md`](../../research/README.md) — 更新实验产物和 activity 入口说明。
- [`../../src/config.py`](../../src/config.py)、[`../../configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml`](../../configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml)、[`../../research/dense_cache_v1_1_1/experiment.yaml`](../../research/dense_cache_v1_1_1/experiment.yaml)、[`../../research/hand_flow_decoder/experiment.yaml`](../../research/hand_flow_decoder/experiment.yaml)、[`../../research/tsne_slots/experiment.yaml`](../../research/tsne_slots/experiment.yaml)、[`../../research/tsne_slots/run.py`](../../research/tsne_slots/run.py) — 将当前运行/实验元数据改为 `modification_version`。
- [`../../../../../docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 链接根级当前版本指针。

**原因**
让 Cm 的启动、运行、停止、产物和文档修改都有单一事件入口；Task 不复制根项目总览，机器事实改由 `machine_memory.md` 承担。

**验证**
- 根级文档/版本指针校验和全量回归：`311 passed, 3 skipped`；本次仅为工程治理迁移，`conclusion: N/A`。

## V1.2.11 — Cm 资产入口下沉到 Task（2026-08-31）

- category: `governance`、`operation`
- 状态: Cm 的 DenseToken 资产入口改为 `src/task/Cm/assets/`；根 `assets` 仅保留被忽略的兼容软链接。
- 保护边界: 未移动大型 checkpoint；真实文件仍位于 `src/task/Cm/densetoken_ckpt`，Task-local `checkpoints/densetoken` 只负责兼容指向。
- 配置: `src/task/Cm/src/config.py` 已改用 Task-local canonical 路径；当前运行中的 Cm 进程未停止。
- 验证: 新旧入口均可解析到同一 `best.pt`，配置导入和定向 loader 回归通过。

## V1.2.10 — 取消 Cm 路径索引（2026-08-31）

- category: `governance`、`data`、`documentation`
- 状态: 已删除 `src/task/Cm/registry/` 和 `src/task/Cm/data/` 下的路径入口；Cm 配置直接声明外部数据/cache/asset 路径。
- 保护边界: 未移动或删除任何外部数据、cache、asset、checkpoint、output 或 outputs。
- 验证: Cm 配置和 loader 回归、全量 pytest 通过。

## V1.2.9 — 撤销隔离试验线（2026-08-31）

- category: `governance`、`code`、`documentation`
- 状态: 已删除 Cm 隔离试验目录、Cm Task 的清单/适配器入口和活动文档中的失效引用；Cm 的模型、数据、cache、训练配置与历史运行产物保留。
- 保护边界: 未删除 `outputs/`、`output/`、cache、checkpoint 或研究产物；历史日志保留原始条目，不重写审计历史。
- 验证: AGENTS/Skill 文本扫描无相关术语；Cm 配置和 loader 定向回归通过。

## 当前状态

- 当前阶段 / 指导: V1.2.1 object-only Cm；正在从受平均 MANO template 影响的旧 GRAB ObjectV2 切换到 subject-template 修复版数据。
- 当前进行中: 用户于 2026-08-29 要求停止新的 GRAB/Inspire-F1 手流重建混合实验；GPU 0/1/2 的 torchrun 已停止。最近完整 checkpoint 为 epoch 29 / step `124410`，位于 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324/checkpoints/latest.pt`；未完成的 epoch 30 不作为完整 checkpoint。CmDecoder 已切换为 GPU0/1/2 三卡训练，Cm 不再占用这三张卡。
- 最近可靠结论: mixed C=256 在 epoch 45--48 的 val mean stride EPE 约 `11.81 mm`，mixed C=64 在 epoch 35--40 约 `12.53--12.64 mm`，均无继续下降趋势；停止时分别保留最近完整 epoch 48 与 epoch 40 checkpoint。修复版 DexYCB C=256 评测平均 `14.92 mm`、相对 zero-flow 改善 `68.27%`。
- 阻塞 / 风险: 当前 best object `val/mean_stride_epe_mm=8.4257 mm`（epoch 21），best hand `val/mean_stride_hand_epe_mm=2.8225 mm`；GRAB object/hand=`13.2589/4.3258 mm`，Inspire-F1 object/hand=`3.5925/1.3193 mm`。两源均未出现明显 hand loss 失衡，但 object 曲线仍有小幅波动，尚不能宣称收敛。CmDecoder 占用 GPU1 后成为 DDP 慢卡，当前 step time 约 `650 ms`（此前约 `300--313 ms`），预计剩余时间相应增加；显存仍无压力。
- 下一步: 保留 epoch 29 / step `124410` 作为当前可恢复锚点；若后续恢复 Cm，从该 `latest.pt` 续训。EXP-023 正式 t-SNE 已生成，包括自然 stride 1--10 dataset 图；matched `[4,6] mm` 仅有 16 条/source，需扩大区间或按分位数匹配后再作 domain/magnitude 判断。旧 slot-wise 产物保留在 `output/research/cm_tsne_inspire_f1_latest_stride5/`。
- HRDexDB 微调入口: `configs/active/hrdexdb_finetune_cm64.yaml`；全量非视频原始数据已迁移到 `dataset/HRDexDB/v0_nonvideo` 并由根 `.gitignore` 排除。共享 geometry cache 允许保留，但 HRDexDB Cm 微调强制 `data.use_dense_cache=false`，DenseToken 输出不得预缓存，必须在线参与反向传播。已完成 loader/builder 框架修正：HRDexDB geometry 需要 4096 稳定物体池、5cm candidate mask，Cm 在线采样 512 点；验证/测试按 source×stride=1/5/10 分开。全量 cache 已导出并生成正式 object-disjoint manifest。
- 证据与相关文档: [EXP-010](experiment_log.md#exp-010--dexycb-subject-10-修复版正式评估)、[EXP-009](experiment_log.md#exp-009--dexycb-subject-10-首次评估无效性诊断)、[EXP-008](experiment_log.md#exp-008--subject-template-修复版-mixed-c64-长训)、[EXP-007](experiment_log.md#exp-007--subject-template-修复版-mixed-正式长训)。
- 2026-08-30 单帧改造已完成代码与回归验证：新 cache 将保存手 mesh/object pose，Dataset 在线随机表面采样并使用 object-pose_t；Inspire-F1 source 的 train/val/test stride 固定为 2。GRAB 与 Inspire-F1 cache 已在新目录后台重建，旧 cache 未覆盖；当前 GRAB 约 1074 条序列、Inspire-F1 约 16 个 episode 已完成，任务仍在运行。

## 新版训练入口

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> \
  /home2/wyy/miniconda3/envs/graspenv/bin/torchrun \
  --standalone --nproc_per_node=<num_gpus> \
  -m src.task.Cm.src.train \
  --config src/task/Cm/configs/active/object_v2_grab_arctic_subject_template_20260820.yaml \
  --distributed
```

当前正式 run 使用 `CUDA_VISIBLE_DEVICES=2,3`、`--set data.batch_size=72 --set data.val_batch_size=72 --set train.resume=<latest.pt>` 续训，输出目录仍为 `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_20260822_125835`。

C=64 严格容量对照使用 `src/task/Cm/configs/active/object_v2_grab_gate_cm64_geometry_only_no_time.yaml`，在 GPU 1/2 以 DDP 启动，per-device batch 32、global batch 64，输出目录为 `outputs/cm/cm_object_v2_grab_gate_cm64_geometry_only_no_time_20260824_155338`。
## 2026-08-30 当前 cache 重建状态

- 用户确认取消 5cm 物体候选过滤，改为完整 manipulated-object surface pool 上的 512 点随机采样；旧 cache 保留不动。
- 旧导出进程已停止。新 GRAB cache：`data/processed_data/cm_object_v2_surface512_object_pose_20260830`；新 Inspire-F1 cache：`data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_20260830`。
- 新任务使用持久在线程会话运行；初始检查 GRAB 已写入 32 个序列，Inspire-F1 尚在加载/准备阶段。
## 2026-08-30 优化版 cache 导出

- Inspire-F1 builder 已切换为每帧一次 FK、1538 点直接变换，并使用 `cuda:0` 批量处理；全量任务当前约 94/576 episode。
- GRAB cache 在原 surface512 输出根上继续补齐，使用 `cuda:7`，当前约 650+ 序列。
- 优化版单 episode 完整构建约 4–5 秒核心变换；预计两项 cache 约 30–60 分钟完成。
## 2026-08-30 导出完成状态

- Inspire-F1 surface512/object-pose cache 已完成 576/576 episode，manifest 与 loader smoke 均通过。
- GRAB surface512/object-pose cache 已物化 1255 个序列；另有 80 个序列因 table/environment 在序列内移动、当前 Scene V1 仅支持 `static_world` 而被拒绝。已完成序列的 candidate 索引均为完整 manipulated-object pool（每帧 4096 个）。
- 当前没有残留 cache 导出进程；旧 cache 未覆盖。若要纳入这 80 个动态环境序列，需要另行采用 dynamic environment 或 object-only fallback 语义。

## 2026-08-30 Cm 混合训练已启动

- 用户确认停止 GPU 0/1/2 上旧的 `src.task.CmDecoder.train`，并启动当前单帧 Cm 版本；旧 CmDecoder 进程已发送 SIGTERM 并确认退出，未影响同机的 rollout/viewer 进程。
- 当前运行：`outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_185013`，命令使用 `CUDA_VISIBLE_DEVICES=0,1,2`、world size=3、per-device batch=32、global batch=96。
- 数据：GRAB train 1004 序列 + Inspire-F1 train 455 episode；`source_probabilities={grab:0.5,inspire_f1:0.5}`。当前 cache 使用 object-pose_t、object surface pool 512 点随机采样、手表面 1538 点随机采样；Inspire-F1 stride 固定为 2；不使用 5cm 距离过滤和时间条件。
- 初始权重：`outputs/cm/cm_hrdexdb_inspire_f1_decoder_only_resume_20260827_011451/checkpoints/latest.pt`；仅加载模型权重，optimizer/scheduler/epoch/step 从零开始。
- 启动验证：三个 Cm rank 已完成 checkpoint 初始化并处于运行态；GPU 1/2 已有高利用率，GPU 0 rank 处于 DDP 同步/数据阶段，当前未见异常退出。

## 2026-08-30 Inspire-F1 5cm mask 已备妥

- 已在 GPU7 对 576 个 Inspire-F1 geometry episode 完成逐帧 5cm mask 重算，结果保存为 `obj_candidate_mask_5cm_recomputed.npy` sidecar。
- 当前运行仍使用全真兼容 mask；切换到重算 mask 需要先安全停止当前混合训练，再更新 loader 的 mask 文件入口并重新启动，避免运行中改变样本行空间。

## 2026-08-30 已切换到 5cm 帧过滤训练

- 已安全停止上一轮使用全真 mask 的 Cm 混合训练，并保留其输出目录与 checkpoint。
- 当前运行：`outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401`，GPU 0/1/2、world size=3、global batch=96。
- GRAB 读取 `candidate_active_5cm.npy` 帧活动 sidecar；Inspire-F1 读取 `obj_candidate_mask_5cm_recomputed.npy`。完整 object surface pool 与运行时 512 点随机采样保持不变。
- loader smoke 统计 train 行数为 `409,435`（GRAB 约 249,256 + Inspire-F1 约 160,179），新 run 的 `total_steps=213,250`，相比未过滤版本 `457,200` 明显缩短。
- 启动日志已完成初始 checkpoint 加载并进入 step 100，当前早期 ETA 约 25 小时；该 ETA 会在预热后继续校正。

## 2026-08-31 已按用户要求停止 Cm 混合训练

- 当前 5cm 过滤版 Cm 训练已在 epoch 35 完成后由用户要求安全停止，保留 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/checkpoints/{best.pt,latest.pt}`；停止前最佳验证点为 epoch 32 / step `136480`。
- 三个 Cm rank 与 torchrun 均已退出，未发现残留 `src.task.Cm.src.train` 进程；不影响其他非训练进程。

## 2026-08-31 Cm 续训切换 Inspire-F1 偶数 stride 2--20

- 当前运行从上述 `latest.pt` 续训，仍使用 GPU0/1/2、global batch=96；GRAB 保持随机 stride `1..10`。
- Inspire-F1 训练 stride 改为运行时随机采样 `{2,4,6,8,10,12,14,16,18,20}`；验证和测试均建立这 10 个固定 stride loader。v4 geometry cache 未重采样，仍按逐帧 world geometry 在线取 future frame。
- 当前进程已恢复到 step `153800` / epoch `37`，新的 stride 配置已通过 loader smoke；启动与续训日志暂未见 OOM/NaN/NCCL 错误。

## 2026-08-30 研究目录与资产入口迁移

- `research/tsne_slots/` 已建立为独立实验包，包含 `run.py`、`README.md`、`experiment.yaml` 和被忽略的 `output/`；旧 `research/tsne_slots.py` 保留软链接，`python -m src.task.Cm.research.tsne_slots` 兼容入口通过。
- `dense_cache_v1_1_1/` 和 `hand_flow_decoder/` 已补齐实验 README、`experiment.yaml` 和独立 `output/` 目录；`research/` 根目录不再新增散脚本。
- `assets/checkpoints/densetoken` 已指向旧 `src/task/Cm/densetoken_ckpt`，Cm 默认配置改用新入口；旧文件未移动，现有训练不受影响。
- 根级 `output/research/` 和 `Cm/results/` 保留历史内容，不新增；新的研究产物使用实验内 `output/<run_id>/`。

## 2026-08-31 以 best.pt 重新初始化 Cm 新训练

- 已按用户确认停止上一轮从 `latest.pt` 续训的 Cm 进程。
- 新 run 使用 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/checkpoints/best.pt` 作为 `train.init_checkpoint`，不设置 `train.resume`；启动日志确认只加载模型权重，并重置 optimizer、scheduler、epoch 和 step。
- 新输出目录为 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222`，运行于 GPU0/1/2，GRAB+Inspire-F1 与 Inspire 偶数 stride `{2,...,20}` 配置保持不变。

## 2026-09-01 当前 best 的 stride 语义修正版 t-SNE

- 使用当前训练 run 的 `checkpoints/best.pt`（epoch 10 / step 42630，`object_pose_t`）在 test 集重画 Cm t-SNE。
- 可视化 stride 按数据语义统计：GRAB 为 `1..10`；Inspire-F1 的基础数据 stride 为 2，因此使用偶数 `2..20`。每边每 stride 52 条，共 520/source。
- 产物位于 `output/research/cm_tsne_current_best_stride_grab1_10_inspire2_20_20260901/`，正式运行成功；训练中的 GPU0/1/2 进程未停止。
- 最近可靠结论与定量结果见 [EXP-025](experiment_log.md#exp-025--当前-best-的-source-specific-stride-t-sne)。
