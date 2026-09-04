# CmDecoder 活动记录

- scope: task:CmDecoder
- last_updated: 2026-09-04
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [架构](architecture_log.md)、[实验](experiment_log.md)、[接手记忆](repo_memory.md)、[历史修改](modification_log.md)

## 2026-09-04 20:27:56 +0800 — GRAB 外部 Cm 在 Inspire Decoder 的首步偏移检查（纠正）

- activity_id: `ACT-20260904-202756-CMDECODER-GRABCM-FIRSTSTEP`
- timestamp: 2026-09-04 20:27:56 +0800
- modification_version: V1.2.12.5
- type: operation / diagnostic
- change_level: L0（只读使用既有 checkpoint 与 cache 推理；未修改模型、配置、数据或训练进程）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 只检查 GRAB 外部 Cm 输入当前 Inspire 点流 Decoder 后的首步位移；不使用 Inspire 下一帧 GT、不做配对 EPE、不做 q/FK 或闭环 rollout
- run_id: `grabcm_inspire_firststep_20260904_202739`
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（偏移行为已确认；不涉及配对精度结论）

**原因**

- 用户指出前一条把无配对的 GRAB→Inspire 迁移错误地与 Inspire 下一帧 GT 比 EPE；本条按“首步是否立即偏移”重新执行，前一条 EPE 仅保留为历史误解记录，不作为本问题结论。

**验证**

- 从 GRAB `s1/apple_lift` frame `47` 用最新 ObjectInteractionCm `best.pt` 生成单个 Cm `[1,16,32]`（`sample_valid=true`，采样交互物体点 `106`）。
- 将该 Cm 送入当前 CmDecoder `best.pt`，在 6 个 Inspire `apple/{0..5}` 各自首个 5cm-active 当前手状态上直接解码一步；输入只包含当前 `[1538,3]` 手点和法向。
- `apple/2` 主样本：第一步每点位移平均 `32.2004 mm`、median=`31.9707 mm`、p95=`35.3529 mm`，1538/1538（100%）点位移超过 `20 mm`；手点质心整体平移 `32.1730 mm`，平移向量约 `[-27.96,-8.12,-13.68] mm`。
- 主样本去除整体平移后的形变仅 `1.7545 mm`（p95=`3.3221 mm`），说明首步主要是**整只手的突发平移**，不是随机散点爆炸。其余 5 个 apple Inspire 状态的平均位移为 `31.67–34.42 mm`，均为 100% 点超过 `20 mm`。
- 主样本手物体最近距离从 `47.26 mm` 变为 `30.34 mm`；这只是几何后果，不代表动作正确。全程没有使用目标下一帧或任何配对标签。

**结论**

- 是的，GRAB 外部 Cm 输入当前 Inspire Decoder 后，**第一步就发生了明显偏移**；而且偏移高度一致，表现为约 `32 mm` 的整体手平移。
- 该现象证明当前 Decoder 没有把 GRAB Cm 当作“无动作/小扰动”处理，而是立即解释成一个强动作条件；但由于没有 GRAB↔Inspire 配对动作，不能说这个平移方向是否正确，只能说跨域输入的首步响应过大。

**证据入口**

- [运行目录](../../../../../outputs/cmdecoder/grabcm_inspire_firststep_20260904_202739/)
- [first_step_metrics.json](../../../../../outputs/cmdecoder/grabcm_inspire_firststep_20260904_202739/first_step_metrics.json)
- [first_step_evidence.npz](../../../../../outputs/cmdecoder/grabcm_inspire_firststep_20260904_202739/first_step_evidence.npz)
- [first_step_visual.png](../../../../../outputs/cmdecoder/grabcm_inspire_firststep_20260904_202739/first_step_visual.png)
- [run_manifest.json](../../../../../outputs/cmdecoder/grabcm_inspire_firststep_20260904_202739/run_manifest.json)

**边界**

- 本条只新增诊断产物和活动记录；未修改代码、配置、cache、checkpoint，未停止或改变 GPU0/1/2 上的训练。

## 2026-09-04 19:43:26 +0800 — GRAB 轨迹 Cm → Inspire 手点 teacher-forced 解码

- activity_id: `ACT-20260904-194326-CMDECODER-GRABCM-INSPIRE-POINT`
- timestamp: 2026-09-04 19:43:26 +0800
- modification_version: V1.2.12.5
- type: operation / diagnostic
- change_level: L0（只读使用既有 checkpoint 与 cache 推理；未修改模型、配置、数据或训练进程）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 从 GRAB `s1/apple_lift` 轨迹逐帧提取 ObjectInteractionCm，再将外部 Cm 逐帧送入 Inspire `apple/2` 的当前点流 Decoder；不做 q/FK、不将 GRAB 手点作为 Inspire 当前状态
- run_id: `grabcm_to_inspire_teacherforce_20260904_194250`
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（不同轨迹的跨域动作迁移诊断，不是配对监督精度）

**原因**

- 用户澄清需要测试“GRAB 轨迹 → 最新 ObjectInteractionCm best.pt 生成 Cm → 最新 CmDecoder best.pt 在 Inspire 手上解码”的真实路径；本条修正了此前误用历史 q-CmDecoder 的评估。

**验证**

- 源序列 `s1/apple_lift`，自动选取首个 5cm 交互段起点 frame `47`；目标序列 `inspire_f1/apple/2`，起点 frame `121`；两边均 stride=`2`，共 32 个 source-Cm/target-state 配对。
- ObjectInteractionCm 使用 `best.pt`（epoch `52` / step `202300`，Cm `[32,16,32]`）；CmDecoder 使用 `best.pt`（epoch `9` / step `421011`），直接调用其 `SharedHandFlowDecoder`，输入 Inspire GT 当前 `[1538,3]` 手点和法向。
- 源 Cm 样本全部有效（32/32）；每帧采样交互物体点数 `106–976`，均值见 metrics。
- 全部 49,216 个手点：预测 flow EPE=`8.2153 mm`（median=`6.5787 mm`），而同一 Inspire GT flow 的 zero-flow 基线仅=`2.6597 mm`。
- 5cm 交互样本中的 503 个近物体手点：EPE=`9.3086 mm`，zero-flow=`3.3165 mm`；预测 flow 平均幅度和逐帧结果见 `metrics.json`。
- 补充控制：同一 Inspire 当前状态由 ObjectInteractionCm 重新生成 Cm 时，EPE=`1.4327 mm`（近物体=`0.6785 mm`）；只替换为 GRAB Cm token、保留 Inspire anchor 时 EPE=`8.2241 mm`。因此本段主要问题在跨域 GRAB token，而不是 anchor 坐标。
- 该结果说明直接把 GRAB 外部 Cm 送入当前 Inspire Decoder 在此段未优于 zero-flow；由于 source/target 是不同轨迹、且物体实例未配对，不能据此判定 Cm 表征或 Decoder 本身失败。

**证据入口**

- [运行目录](../../../../../outputs/cmdecoder/grabcm_to_inspire_teacherforce_20260904_194250/)
- [run_manifest.json](../../../../../outputs/cmdecoder/grabcm_to_inspire_teacherforce_20260904_194250/run_manifest.json)
- [metrics.json](../../../../../outputs/cmdecoder/grabcm_to_inspire_teacherforce_20260904_194250/metrics.json)
- [control_metrics.json](../../../../../outputs/cmdecoder/grabcm_to_inspire_teacherforce_20260904_194250/control_metrics.json)
- [evidence.npz](../../../../../outputs/cmdecoder/grabcm_to_inspire_teacherforce_20260904_194250/evidence.npz)
- [GRAB ObjectInteractionCm best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt)
- [CmDecoder best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/checkpoints/best.pt)

**边界**

- 本条只生成诊断输出并记录活动；未改动 `src/`、训练配置、cache、checkpoint，也未干扰 GPU0/1/2 的长训。EPE 与 zero-flow 的比较只作该跨轨迹传输段的 sanity check，不能当作标准 val/test 指标。

## 2026-09-04 17:06:43 +0800 — GRAB Cm + q-CmDecoder teacher-forced 评估

- activity_id: `ACT-20260904-170643-CMDECODER-GRABCM-TEACHERFORCE`
- timestamp: 2026-09-04 17:06:43 +0800
- modification_version: V1.2.12.5
- type: operation / diagnostic
- change_level: L0（只读评估既有 checkpoint 与 cache；未修改模型、配置、数据或训练进程）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 使用已有 GRAB-Cm q-CmDecoder 基线，在 object-disjoint Inspire-F1 3 Hz cache 上做 teacher-forced `q_t → q_next` 评估；不干扰 GPU0/1/2 长训
- run_id: `grab_cm_teacherforce_20260904_170431`
- run_status: COMPLETED
- checkpoint: `outputs/cmdecoder/cm_decoder_20260822_143843/checkpoints/best.pt`（step7155/epoch1）
- cm_checkpoint: `outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt`（16×256，cache token SHA256 匹配）
- conclusion: INCONCLUSIVE（单步 teacher-forcing 可用，但该结果是 q 解码基线，不能直接等同于当前 point-flow Decoder 效果）

**原因**

- 用户希望检查原始 GRAB-Cm 表征送入 Decoder 后的 teacher-forced 效果，以和当前 ObjectInteractionCm 长训的单步表现做参照。

**验证**

- 运行命令：`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python - <<'PY' ... CmDecoderRunner(mode="eval") ... PY`；使用已有 `cm_decoder_20260822_143843/best.pt`，关闭 DDP，仅在物理 GPU3 推理。
- 数据：`selection_576_object_disjoint_seed42.json` 的 v1 cache；分别评估 val/test，Cm token cache 使用的 checkpoint hash 与指定 GRAB-Cm 一致。
- teacher-forced val：q MAE=`0.8824°`，identity（直接使用 `q_t`）=`0.9332°`，相对改善约 `5.4%`。
- teacher-forced test：q MAE=`1.2307°`，identity=`1.2982°`，相对改善约 `5.2%`；q RMSE=`0.03282 rad`。
- 评估耗时约 `28.0 s`（val）+ `28.1 s`（test），未见运行错误；当前 GPU0/1/2 长训未被停止或修改。

**证据入口**

- [评估运行目录](../../../../../outputs/cmdecoder/grab_cm_teacherforce_20260904_170431/)
- [run_manifest.json](../../../../../outputs/cmdecoder/grab_cm_teacherforce_20260904_170431/run_manifest.json)
- [config.json](../../../../../outputs/cmdecoder/grab_cm_teacherforce_20260904_170431/config.json)
- [teacherforce_metrics.json](../../../../../outputs/cmdecoder/grab_cm_teacherforce_20260904_170431/teacherforce_metrics.json)
- [输入 Decoder checkpoint](../../../../../outputs/cmdecoder/cm_decoder_20260822_143843/checkpoints/best.pt)
- [输入 GRAB Cm checkpoint](../../../../../outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt)

**边界**

- 该评估的 Decoder 是历史 `CmDecoderModel`（`qt_cm`，输出 q），不是当前 ObjectInteractionCm 的 1538 点 point-flow Decoder；因此只能说明 GRAB-Cm 在既有 q 解码任务上相对 identity 有小幅增益，不能直接与当前 `7.x mm` point-flow EPE 横向比较。

## 2026-09-04 16:41:59 +0800 — CmDecoder 长训状态核查

- activity_id: `ACT-20260904-164159-CMDECODER-STATUS`
- timestamp: 2026-09-04 16:41:59 +0800
- modification_version: V1.2.12.5
- type: diagnostic
- change_level: L0（只读查询既有长训；未改变进程、代码、配置、数据或 checkpoint）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 查询 `cm_decoder_20260904_114238` 的实时进程、epoch/step、validation 曲线、吞吐、GPU 和 checkpoint
- run_id: `cm_decoder_20260904_114238`
- run_status: RUNNING
- last_step: `280900`（查询时）
- last_epoch: `7`（epoch6 验证已完成，epoch7 训练进行中）
- best_metric: `7.637870638383664 mm`（`val/hand_flow/epe_mm`，epoch4）
- conclusion: INCONCLUSIVE（训练链路健康，但验证曲线尚未显示稳定收敛）

**原因**

- 用户询问当前 CmDecoder 长训状态，需要确认任务是否仍在运行、最新 epoch/step、当前 best 和预计完成时间。

**验证**

- torchrun 与 3 个训练 rank 仍存活，GPU0/1/2 利用率约 `89%/90%/91%`，未发现 OOM、NCCL、NaN 或 Inf。
- 已完成到 `epoch6`：`step=280674`；epoch6 train EPE=`7.5967 mm`、loss=`0.0020173`，随后 validation EPE=`7.7254 mm`、loss=`0.00214723`；当前已进入 epoch7（查询时 `step=280900`）。
- validation EPE 为 epoch1=`7.8656`、epoch2=`8.0648`、epoch3=`8.6717`、epoch4=`7.6379`、epoch5=`7.8849`、epoch6=`7.7254 mm`；epoch4 仍为 best，epoch6 未刷新 checkpoint。相对固定 zero-flow validation 基线 `12.1826 mm`，模型有改善，但曲线有波动，暂不能宣称收敛。
- 当前总计划 `1,403,370` steps，已完成约 `20.0%`；最近吞吐约 `960 global samples/s`、单 step 约 `50 ms`、data wait 约 `10%`，日志 ETA 约 `15.6 h`，按当前速度预计 2026-09-05 早晨结束（验证耗时可能带来小幅偏移）。

**证据入口**

- [运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/)
- [metrics.jsonl](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/train.log)
- [当前 best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/checkpoints/best.pt)

**边界**

- 本次仅做状态诊断并追加活动记录；未停止、恢复、调参或修改代码/配置，也未改变现有 checkpoint。

## 2026-09-04 16:16:24 +0800 — Inspire rollout 耗时归因诊断

- activity_id: `ACT-20260904-161624-CMDECODER-ROLLOUT-TIMING`
- timestamp: 2026-09-04 16:16:24 +0800
- modification_version: V1.2.12.5
- type: diagnostic
- change_level: L0（只读代码、运行清单、文件时间和既有活动记录）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 比较当前 best.pt 的 direct point rollout、当前 q/FK 尝试和历史成功的 q/FK rollout；不启动新运行、不修改代码或训练进程
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（已定位主要耗时段，但当前 q/FK 路径的异常慢仍需单帧 profiler 复核）

**原因**

- 用户询问 q/FK rollout 的长耗时究竟来自点流解码还是每步物理投影；需要基于既有运行证据拆分两段计算，且不把未完成的 q/FK 尝试误报为模型结果。

**验证**

- 当前 q/FK 尝试约运行 4 小时仍停留在 `optimize_q_from_hand_points`，未产生轨迹；该函数默认每帧 100 次可微 FK/反向传播，32 帧对应约 3,232 次优化迭代。
- 同一当前 best 的 direct point rollout 未执行 q/FK，32 步输出目录时间约为 15:52–16:00；该结果包含模型加载、推理、保存和绘图，不能当作严格 kernel benchmark，但未出现 q/FK 阶段的长时间停滞。
- 需要纠正历史时间归因：10:25:00–10:29:10 的 4 分 10 秒运行后来发现漏加载 Decoder `payload["model"]`，属于 `INVALID_IMPLEMENTATION`，不是已修正结果；修正活动从 10:36 记录开始，修正后的 NPZ 在 10:53:29 写出、manifest 在 10:55:10 写出，故可确认修正 run 至少约 17 分钟（manifest 未记录 `started_at`）。因此当前 4 小时仍明显异常，但不能再用 4 分 10 秒作为有效 q/FK 基线。
- 代码检查显示 Decoder 每帧仅执行一次无梯度前向；q/FK 每个优化步执行 1538 点 FK、腕部 SE(3) 变换和反向传播，理论上是主耗时段。当前 adapter 在兼容调用中若先计算 Cm tokens、再调用仍会重新计算 Cm 的 `forward`，会增加前向开销，但预计不是 4 小时量级的首要原因。

**证据入口**

- [当前 direct point rollout](../../research/inspire_hand_rollout/output/cmdecoder_best_epoch4_20260904_155200/)
- [当前 q/FK 实现](../../q_optimizer.py)
- [当前 runtime rollout 入口](../../runtime_rollout.py)
- [历史 q/FK rollout manifest](../../../../../output/research/inspire_runtime_rollout_cmdecoder_best_fixed_20260902.run_manifest.json)

## 2026-09-04 16:01:05 +0800 — CmDecoder best.pt Inspire 手点闭环 rollout

- activity_id: `ACT-20260904-160105-CMDECODER-INSPIRE-POINT-ROLLOUT`
- timestamp: 2026-09-04 16:01:05 +0800
- modification_version: V1.2.12.5
- type: diagnostic / operation
- change_level: L0（只读推理与可视化；未修改模型、配置、数据或长训进程）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 使用 CmDecoder 当前 best.pt（epoch4/step187116）在 Inspire-F1 `bamboo_basket/5`、stride=2、32 步上执行 1538 点直接闭环；冻结 Cm 使用记录的 GT hand-flow 和 GT source hand state，decoder 输入使用上一时刻预测手点；不做 q/FK 拟合
- run_id: `cmdecoder_best_epoch4_20260904_155200`
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（该单 episode 诊断显示明显闭环漂移，但不能替代全量 rollout 或改变训练结论）

**原因**

- 用户要求实际拿 best.pt 检查 Inspire 手本身的自编码 rollout。先尝试了既有 q/FK rollout 入口，但该入口每帧 100 次全手点优化，运行约 4 小时仍无结果，随后停止并改为不含 q/FK 的点空间闭环，以快速隔离 decoder/state feedback。

**验证**

- q/FK 尝试在 `optimize_q_from_hand_points` 阶段手动停止，未生成轨迹，未影响 GPU0/1/2 长训；失败原因是该诊断路径计算量不适合短检查。
- 点空间 rollout 已完成：`episode=inspire_f1/bamboo_basket/5`、自动起点 `start_pair=68`、stride=`2`、32 steps（source frame 73→136）。
- teacher-forced（每步使用 GT 当前手点）point EPE 均值/末步为 `5.0475 / 1.4517 mm`；closed-loop（反馈预测手点）为 `83.9538 / 139.9299 mm`，误差从首步 `5.2613 mm` 增长到 step16 `102.66 mm`、末步 `139.93 mm`。
- 该 episode 的 20mm contact ratio 预测和 GT 全程均为 `0`，因此接触率在本条诊断中没有区分度。
- 使用的 checkpoint SHA256=`adf3def857f80f6b1950257404a40080c18e6839a881883a0b4ad48d44cb1249`；当前长训未被停止或修改。
- 产物：[rollout output](../../research/inspire_hand_rollout/output/cmdecoder_best_epoch4_20260904_155200/)、[point_rollout.npz](../../research/inspire_hand_rollout/output/cmdecoder_best_epoch4_20260904_155200/point_rollout.npz)、[point_rollout.png](../../research/inspire_hand_rollout/output/cmdecoder_best_epoch4_20260904_155200/point_rollout.png)、[run_manifest.json](../../research/inspire_hand_rollout/output/cmdecoder_best_epoch4_20260904_155200/run_manifest.json)。

**边界**

- 这是 object_pose_t 下的点空间自编码闭环；预测 normals 在 rollout 中保持初始 source normals，且没有 q/FK 投影，所以闭环漂移包含 state/normals 近似误差。teacher-forced 与 closed-loop 的巨大差距仍说明当前 decoder 对输入状态偏移敏感，下一步应在多个 episode/stride 上复核，再决定是否修改 decoder 或 rollout 状态更新。

## 2026-09-04 15:30:23 +0800 — CmDecoder 长训 epoch4 验证进展

- activity_id: `ACT-20260904-153023-CMDECODER-EPOCH4-STATUS`
- timestamp: 2026-09-04 15:30:23 +0800
- modification_version: V1.2.12.4
- type: diagnostic / operation
- change_level: L0（只读查询既有长训；未改变进程、代码、配置、数据或 checkpoint）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 查询 `cm_decoder_20260904_114238` 的 epoch4 validation、当前 epoch5 训练、吞吐、GPU 和 checkpoint
- run_id: `cm_decoder_20260904_114238`
- run_status: RUNNING
- last_step: `221000`（查询时）
- last_epoch: `5`（epoch5 训练进行中）
- best_metric: `7.637870638383664 mm`（`val/hand_flow/epe_mm`，epoch4）
- conclusion: INCONCLUSIVE（epoch4 已改善并更新 best，但长训仍未完成）

**原因**

- 用户询问当前长训状态；需确认前一轮 validation 是否结束、best 是否更新以及此前的验证上升趋势是否持续。

**验证**

- epoch4 validation 已完成：`val/hand_flow/epe_mm=7.6379 mm`、`val/loss=0.00213745`，优于 epoch1=`7.8656 mm`、epoch2=`8.0648 mm`、epoch3=`8.6717 mm`；因此 best.pt 已更新为 epoch4。
- 当前进入 epoch5，约 `step=221000 / 1,403,370`；训练 rank 和 GPU0/1/2 均正常运行，无 OOM、NCCL、NaN 或 Inf。
- 稳定吞吐约 `950–970 global samples/s`，最近窗口约 `960 samples/s`，ETA 约 `16.4 小时`。
- 产物入口：[运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/)、[metrics.jsonl](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/train.log)、[best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/checkpoints/best.pt)。

**边界**

- epoch4 的单点改善不能证明已经收敛；继续观察 epoch5–8 的 validation 曲线，暂不调整训练或停止运行。

## 2026-09-04 15:00:24 +0800 — CmDecoder 长训进度核查

- activity_id: `ACT-20260904-150024-CMDECODER-STATUS`
- timestamp: 2026-09-04 15:00:24 +0800
- modification_version: V1.2.12.3
- type: diagnostic / operation
- change_level: L0（只读查询既有长训状态；未改变进程、代码、配置、数据或 checkpoint）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 查询 `cm_decoder_20260904_114238` 的进程、训练/验证日志、metrics、checkpoint 和 GPU0/1/2 状态
- run_id: `cm_decoder_20260904_114238`
- run_status: RUNNING
- last_step: `187116`
- last_epoch: `4`（epoch4 train 已完成，validation 进行中）
- conclusion: INCONCLUSIVE（训练链路正常，但验证趋势尚未完成长训判断）

**原因**

- 用户询问“现在怎么样”，需要确认长训是否仍在运行、是否产生 epoch/checkpoint，以及当前吞吐和验证指标是否异常。

**验证**

- 进程仍存活：torchrun 与三个 rank 均处于运行态，GPU0/1/2 利用率约 `90%/91%/90%`；未发现 OOM、NCCL、NaN 或 Inf。
- 已完成 epoch：`epoch1 step46779`、`epoch2 step93558`、`epoch3 step140337`；当前 epoch4 train 完成于 `step187116`，validation 尚未写入。
- 已完成 validation 的 hand-flow EPE：epoch1=`7.8656 mm`、epoch2=`8.0648 mm`、epoch3=`8.6717 mm`；当前 best checkpoint 仍为 epoch1，验证指标暂时呈上升趋势，不能宣称收敛或最终效果成立。
- 稳定吞吐约 `950–970 global samples/s`，最近 step187100 为 `966.6 samples/s`，当前 ETA 约 `16.8 小时`。
- 产物入口：[运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/)、[metrics.jsonl](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/train.log)、[当前 best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/checkpoints/best.pt)。

**边界**

- 本次只读核查未停止或调整长训；epoch4 validation 完成后再根据曲线判断是否需要早停、调参或保留当前 best。

## 2026-09-04 11:43:52 +0800 — 使用最终 best Cm 启动 CmDecoder 长训

- activity_id: `ACT-20260904-114352-CMDECODER-LONGTRAIN-BESTCM`
- timestamp: 2026-09-04 11:43:52 +0800
- modification_version: V1.2.12.2
- type: operation
- change_level: L3（用户明确批准的长时多卡训练；不改变研究变量或数据合同）
- approval: user-approved
- approval_basis: 用户明确要求“开始长训 CmDecoder，用 best 的 Cm”
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动及 KNN 实现 diff）
- scope: GPU0/1/2 DDP，从头训练当前 V1.2.1 CmDecoder；每卡 batch=16、global batch=48、AMP=false、all-stride、30 epochs，冻结最终 ObjectInteractionCm `best.pt`；不启用尚未实现的邻域 cache
- run_id: `cm_decoder_20260904_114238`
- run_status: RUNNING
- last_step: `1100`（记录时）
- last_epoch: `1`（记录时，epoch 尚未完成）
- conclusion: INCONCLUSIVE（训练进行中，尚无收敛或效果结论）

**原因**

- 先前短训已确认前缀 KNN 路径数值 parity 和吞吐收益；本次按用户确认使用最终 Cm best 开始正式 CmDecoder 长训，保持其它 V1.2.1 数据与损失口径不变。

**验证**

- 启动命令：`CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoder.train --config src.task.CmDecoder.current_object_interaction_cm_v1_2_config:Config --set meta.cm_checkpoint=outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt --set train.distributed.enable=true --set train.description=V1.2.1_longtrain_bestcm_knn012 --set performance.warmup_steps=20`
- 实际运行目录：[outputs/cmdecoder/cm_decoder_20260904_114238](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/)，配置/manifest 已生成；输入 checkpoint：[best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt)。
- 当前 `total_steps=1,403,370`，step 100–1100 稳定吞吐约 `923–972 global samples/s`，step 1100 为 `962.0 samples/s`，初始 ETA 约 `19.4–20.3 h`。
- 当前日志：[metrics.jsonl](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoder/cm_decoder_20260904_114238/train.log)。尚未生成 checkpoint，终态时再补充最佳/最近 checkpoint 链接。

**运行边界**

- 该条保持 `run_status=RUNNING`；后续查询/停止/完成均在本活动时间线追加终态记录。若需停止，使用 torchrun 会话 `53918`，不影响 GPU3/4–7 上既有任务。

## 2026-09-04 11:27:03 +0800 — GPU0/1/2 KNN 邻域短训吞吐

- activity_id: `ACT-20260904-112703-CMDECODER-KNN-012-THROUGHPUT`
- timestamp: 2026-09-04 11:27:03 +0800
- modification_version: V1.2.12.1
- type: operation / diagnostic
- change_level: L1 代码优化后的短训运行；不改变训练目标、数据 split、batch contract 或 radius/K
- approval: user-requested
- approval_basis: 用户要求先修改邻域实现并用 GPU0/1/2 测吞吐
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动和本次局部代码 diff）
- scope: 使用 ObjectInteractionCm V1.2.1 最终 `best.pt`，CmDecoder 每卡 batch=16、global batch=48、AMP=false，`max_steps=200`；只测训练吞吐，末尾 validation 阶段手动停止，避免把全量验证混入短训
- run_id: `cm_decoder_20260904_112313`
- run_status: STOPPED（训练 step 已完成 200；末尾全量 validation 按测量目的手动 SIGINT）
- last_step: `200`
- last_epoch: `1`（train epoch 完成；validation 未完成）
- conclusion: SUPPORTED（工程 smoke 和吞吐证据成立；不是 CmDecoder 收敛或效果结论）

**原因**

- 用户希望验证“只搜索有效手点”对 CmDecoder 训练吞吐的实际影响；因此先采用不带持久化 cache 的在线 KNN，避免把 cache 命中率、采样 key 和 kernel 改动混在一次测量中。

**命令与输入**

- 命令：`CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoder.train --config src.task.CmDecoder.current_object_interaction_cm_v1_2_config:Config --set meta.cm_checkpoint=outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt --set train.distributed.enable=true --set train.max_steps=200 --set train.epochs=1 --set train.log_every_steps=20 --set train.eval_every_steps=none --set train.save_every_steps=none --set train.save_every_epochs=none --set train.output_dir=outputs/cmdecoder/cm_decoder_20260904_120000_knn012_smoke --set train.description=knn_prefix_smoke --set performance.warmup_steps=20`
- 实际运行目录由 BaseRunner 生成：[outputs/cmdecoder/cm_decoder_20260904_112313](../../../../../outputs/cmdecoder/cm_decoder_20260904_112313/)。命令中的 `train.output_dir` 不被当前 runner 的训练命名逻辑采用，未覆盖其他目录。
- 输入 Cm：[ObjectInteractionCm best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt)。

**验证**

- 稳定窗口（step 160/180/200）全局 `samples_per_s=931.323 / 940.528 / 943.164`，对应 `step_ms=51.54 / 51.04 / 50.89`；全局 batch=48，因此约 `943 samples/s`。
- step 40–140 的窗口为 `515 / 489 / 359 / 680 / 902 / 662 samples/s`，属于 CUDA、worker 和文件冷启动/抖动，不能作为稳定值。
- step 200 的训练 epoch 汇总：hand-flow EPE=`10.4071 mm`，loss=`0.00296888`；这些数只用于确认 forward/backward 正常，不用于效果比较。
- `metrics.jsonl`、`train.log`、`config.json`、`metadata.json`、`run_manifest.json` 均已生成；未生成 checkpoint（按命令关闭保存，且 validation 收尾被中止）。

**解释边界与回滚**

- 与此前单卡 GPU3 的约 `270–273 samples/s` 不能直接作严格因果比较，因为卡数、GPU 负载和 DDP 都不同；本次只能确认 0/1/2 上新路径可稳定运行并达到约 `943 global samples/s`。
- 本条运行产物保留作审计证据；代码回滚见 ObjectInteractionCm 活动 `ACT-20260904-112703-OICM-KNN-PREFIX-OPT`。

## 2026-09-04 11:10:42 +0800 — ball query 与邻域 cache 可行性诊断

- activity_id: `ACT-20260904-111042-CMDECODER-BALLQUERY-CACHE-FEASIBILITY`
- timestamp: 2026-09-04 11:10:42 +0800
- modification_version: V1.2.12
- type: diagnostic
- change_level: L0（只读检查算子依赖、邻域语义和采样稳定性）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 评估 PyTorch/PyTorch3D ball query、精确最近 8 点语义和 CmDecoder 邻域 cache 的可行性；不修改代码、配置、数据、cache、checkpoint 或运行状态
- run_status: STOPPED（当前 CmDecoder 无训练进程；本次未改变）
- conclusion: INCONCLUSIVE（依赖和 cache key 条件满足，但实际 kernel 加速与磁盘/吞吐收益尚未 benchmark）

**原因**

- 用户希望先查 5cm 内候选，再取最近 8 点，并预先缓存邻域，避免每个 epoch 重复搜索。本次核对可用算子及不同 Task 的采样是否允许静态复用。

**验证**

- 当前环境有 `torch 2.4.1+cu121`、`pytorch3d 0.7.9` 和 CUDA 扩展；没有 `torch_cluster`。PyTorch3D `ball_query` 可返回固定上限邻域，但官方语义是不保证返回邻居按距离最近排序，因此不能直接以 `K=8` 代替“半径内最近 8 点”。
- “全局最近 8 点后再做半径过滤”与“半径过滤后再取最近 8 点”在无距离并列时集合等价；可优先评估 PyTorch3D `knn_points(K=8)` + 半径 mask，避免 Python 层构造完整距离矩阵。若必须严格使用 ball-first，需要 voxel/hash 候选或较大 K 后再排序，不能把 `ball_query(K=8)` 当作精确实现。
- CmDecoder 的 `RandomHorizonGeometryDataset` 对每个 `(geometry, frame)` 使用稳定 object sampling seed；邻域只依赖当前 frame 几何、1024 object sampling、1538 hand geometry、半径和 K，与 stride 无关，因此可按唯一当前 frame 缓存，十个 stride 共享一份邻域。
- ObjectInteractionCm 训练 Dataset 的 seed 包含 `_epoch`，object 1024 采样会随 epoch 变化；同一邻域 cache 不能直接用于该训练，除非固定采样或改变其增强语义。本计划建议先限定 CmDecoder。

**产物**

- [CmDecoder dataset](../../dataset.py)、[CmDecoder adapter](../../object_interaction_cm_model.py)、[ObjectInteractionCm dataset](../../../ObjectInteractionCm/dataset.py)

**回滚**

- 本条仅追加诊断记录；未修改实现或任何运行产物。

## 2026-09-04 10:46:56 +0800 — 核对 cdist 是否只计算有效手点

- activity_id: `ACT-20260904-104656-CMDECODER-CDIST-MASK-ORDER`
- timestamp: 2026-09-04 10:46:56 +0800
- modification_version: V1.2.12
- type: diagnostic
- change_level: L0（只读源码和算子形状顺序）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 核对 CmDecoder adapter 的 1538→3076 padding、`hand_valid_mask` 和 ObjectInteractionCm `cdist/topk` 执行顺序；不修改代码、配置、数据、cache、checkpoint 或运行状态
- run_id: `cm_decoder_20260904_001753`
- run_status: STOPPED（本次未改变）
- conclusion: SUPPORTED（源码和最小算子复现一致表明：mask 只屏蔽 cdist 输出及后续邻居，不减少 cdist 的输入规模）

**原因**

- 用户追问当前实现是否已经让 `cdist` 只处理有效的 1538 个手点；本次核对 padding、mask 和距离算子的实际执行顺序。

**验证**

- adapter 在 `object_interaction_cm_model.py` 先将 hand `[B,1538,3]` 补成 `[B,3076,3]`，再将前 1538 个位置设为 valid、后 1538 个位置设为 invalid。
- `LocalHandInteraction.forward()` 的顺序是 `torch.cdist(object_points, hand_points)`，随后才执行 `distances.masked_fill(~hand_valid_mask[:,None,:], inf)`，然后对结果 `topk`；因此计算上 cdist 仍是 `[B,1024,3076]`，只有 1538 个邻居在 mask 后有效。
- 最小 torch 复现确认：输入距离输出为 `[1,1024,3076]`；mask 后后 1538 列变为 `inf`，但这些列已经参与 cdist 计算。

**结论解释**

- “交互参与的有效点”是 1538 个；“cdist 实际计算的点”是 3076 个。此前将这两件事混称为“只对有效点 cdist”是不准确的。
- 这条结论只修正计算量分析，不改变当前训练结果或研究结论。若要真正只计算 1538 点，需要在调用 `cdist` 前 compact/裁剪 hand tensor，并对 checkpoint 输出做 parity 验证。

**产物**

- [CmDecoder adapter](../../object_interaction_cm_model.py)、[ObjectInteractionCm model](../../../ObjectInteractionCm/model.py)

**回滚**

- 本条仅追加诊断记录；未修改实现或任何运行产物。

## 2026-09-04 10:34:04 +0800 — CmDecoder 可用性与 DenseToken 速度差异诊断

- activity_id: `ACT-20260904-103404-CMDECODER-AVAILABILITY-SPEED-DIAGNOSTIC`
- timestamp: 2026-09-04 10:34:04 +0800
- modification_version: V1.2.12
- type: diagnostic
- change_level: L0（只读核对 checkpoint、历史 metrics、模型结构、训练步数和 GPU 状态）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 比较当前 V1.2.1 CmDecoder、旧 V1.1 ObjectInteractionCm Decoder 和历史 DenseToken CmDecoder 的可用性、吞吐与总训练预算；不修改代码、配置、数据、cache、checkpoint 或运行状态
- run_id: `cm_decoder_20260904_001753`、`cm_decoder_20260903_091220`、`cm_decoder_20260901_151052`
- run_status: STOPPED / STOPPED / STOPPED（本次未改变）
- conclusion: INCONCLUSIVE（当前 checkpoint 可加载且 smoke 合同成立，但训练提前停止、无完整 test 结果；速度原因已由多项可观测差异解释，尚未做隔离 kernel benchmark）

**原因**

- 用户询问当前 CmDecoder 是否可用，并质疑“去掉 DenseToken、只略微扩大参数量”却变慢的原因。本次区分参数量、单 step 计算、硬件资源和总训练步数，避免把不同代际运行混作一个基线。

**产物与证据**

- [当前 CmDecoder 运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/)、[best checkpoint](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/checkpoints/best.pt)、[metrics](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/metrics.jsonl)、[train log](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/train.log)
- [旧 V1.1 ObjectInteractionCm 运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/)、[历史 DenseToken 运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260901_151052/)
- [CmDecoder adapter](../../object_interaction_cm_model.py)、[当前 ObjectInteractionCm model](../../../ObjectInteractionCm/model.py)、[历史 DenseToken decoder](../../point_model.py)

**验证**

- 当前 `best.pt` 能被 adapter 实例化并加载；checkpoint 为 epoch `1` / step `35084`，包含 `70` 个 state keys，其中 `60` 个为内嵌 `cm.*` 权重。当前长跑最后记录为 step `70600` / epoch `3`，因此它可用于工程 smoke、离线可视化和条件重建诊断，但不能称为已完成或已验证的最终 Decoder。当前 best validation hand-flow EPE=`7.932646 mm`，zero-flow=`12.182689 mm`。
- 历史 DenseToken Decoder (`src.task.CmDecoder.point_model.CmPointFlowModel`) 的真实日志吞吐约 `146–148 samples/s`（三卡 global batch=`48`，`step_ms≈325 ms`），其总预算约 `140340` steps；它使用 `512` 个 object 点、`1538` 个 hand 点和在线 DenseToken。当前 V1.2.1 单卡 GPU3 为 `270–273 samples/s`（`step_ms≈235 ms`），所以按全局 samples/s 当前并没有比该 DenseToken 运行更慢。
- 用户感受到的“很快”主要对应紧接着的旧 V1.1 ObjectInteractionCm 运行：三卡 DDP、专用 GPU0/1/2、global batch=`48`，吞吐约 `1650–1680 samples/s`。当前改为共享中的单卡 GPU3（另有约 `7.4 GiB` 足球任务），按卡归一化后仍约慢 `2x`；当前 GPU3 不是空卡因素已经确认。
- 真正的模型变化是 V1.1 legacy Cm（processing dim=`32`，Cm 参数=`45256`）→ V1.2.1（processing dim=`128`，Cm 参数=`449382`，约 `9.9x`），而外部可训练 Decoder 两次均为 `47043` 参数。新模型增加双路 local interaction、128 维 fusion/SlotAttention（3 iterations）、projection 以及更宽的 object/hand decoders；冻结只关闭梯度，不关闭 forward 计算。
- 当前 adapter 每 batch 调用完整 `self.cm(cm_batch)`。CmDecoder 只取 `cm_tokens/anchors`，但 ObjectInteractionCm forward 仍执行 object decoder 和 hand decoder；这部分是当前路径的可去除计算。
- 当前 local interaction 对 `[B,1024,3076]` 做 `torch.cdist` 和 `topk`；单样本先形成约 `3.15M` 距离。历史 DenseToken 入口是 `[B,512,1538]`，约 `0.787M` 距离，几何距离工作量约为 `4x`；当前还在这些边和 slot 上使用 128 维而非旧 32 维。该几何 all-pairs kernel 不能用参数量直接类比 DenseToken 的 PTv3/MLP 路径。
- 总训练时间还受到数据展开影响：DenseToken 历史运行约 `140340` steps；当前 all-stride V1.2.1 为 `1052520` steps，约 `7.5x`。因此即使单 step 不变，完整 run 也会显著变长。
- 当前 `data_wait_ratio≈2%–3%`，旧快跑约 `13%`；这说明 DataLoader 不是主要解释。未发现当前运行的 OOM、NaN、Inf 或 NCCL traceback。

**结论解释**

- “可用”分两层：工程上可用（checkpoint 可加载、输出 `[B,1538,3]`、冻结 Cm 与新 Decoder wiring 正常）；科研上尚不可用作最终模型（只完成约两轮完整 validation，训练被中断，且该任务是编码器输入包含 GT hand-flow 的条件重建，不是未知动作预测）。
- “去掉 DenseToken 后变慢”不是单一参数量效应。当前主要成本依次是：V1.2.1 的 128 维和更宽 head、1024×3076 的全 pairwise 几何交互、无用的 frozen object/hand decoder forward、单卡共享资源，以及 all-stride 带来的 7.5 倍训练 steps。DenseToken 运行本身在历史日志中并不是 1650 samples/s 的快基线。

**验证命令**

- `torch.load`/adapter 实例化与参数分组统计；读取两次运行的 `config.json`、`metrics.jsonl`、`train.log`；`nvidia-smi` 与 `ps`；源码行级核对 `torch.cdist`、完整 Cm forward 和 DenseToken 输入点数。

**回滚**

- 本条仅追加诊断记录；未修改模型、配置、数据、cache、checkpoint 或任何训练进程。

## 2026-09-04 10:26:59 +0800 — V1.2 CmDecoder 冻结 Cm 与最终 Cm best 性能差异

- activity_id: `ACT-20260904-102659-CMDECODER-V12-CM-CHECKPOINT-COMPARE`
- timestamp: 2026-09-04 10:26:59 +0800
- modification_version: V1.2.12
- type: diagnostic
- change_level: L0（只读比较既有 Cm 验证指标与 checkpoint provenance）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 比较 CmDecoder 实际冻结的 ObjectInteractionCm epoch 33 snapshot 与训练最终 epoch 52 best；不修改代码、配置、数据、cache、checkpoint 或运行状态
- run_id: `cm_decoder_20260904_001753`；`object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806`
- run_status: COMPLETED / STOPPED（历史运行状态；本次未改变）
- conclusion: INCONCLUSIVE（性能差异已量化，但未用最终 Cm 重新训练/评估 CmDecoder）

**原因**

- 用户询问训练 Decoder 时冻结的 Cm 与 Cm 最终 best.pt 的性能差距；本次从 ObjectInteractionCm `metrics.jsonl` 对齐 epoch 33/52，并核对 CmDecoder manifest 中冻结 checkpoint 的 step、epoch 与 SHA256。

**产物与证据**

- [CmDecoder 运行 manifest](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/run_manifest.json)、[CmDecoder config](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/config.json)
- [ObjectInteractionCm metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[ObjectInteractionCm checkpoints](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/)

**验证**

- CmDecoder 实际加载的是 epoch 33 / step `133105` 的 snapshot，manifest 记录 SHA256=`1dfd892771907a6a26aee54103faebba249f55699c11e4383f8a23fc201d4021`；当前同一路径 `best.pt` 已被 Cm 训练结束时覆盖为 epoch 52，当前 SHA256=`1fe07f87f34dbdf2424a0de6e16e97d2eff83dc9a5302a642a893c07bb3e1243`。
- CmDecoder 的 epoch 1 `best.pt` 仍内嵌 60 个 `cm.*` 张量，可作为当时冻结 Cm 权重的恢复入口；但原始 epoch 33 外部 checkpoint 文件已不在当前 `outputs` 中，无法再用外部文件做逐 tensor equality 复核。
- 总体 stride=2 validation：object EPE `3.750058→3.649306 mm`（下降 `0.100752 mm`，`2.69%`）；hand 3 cm EPE `2.685803→2.404122 mm`（下降 `0.281681 mm`，`10.49%`）；loss `0.003239385→0.003072887`（下降 `5.14%`）。
- GRAB：object `5.178715→5.073422 mm`（`2.03%`），hand `3.067312→2.736167 mm`（`10.80%`）。Inspire-F1：object `2.321402→2.225191 mm`（`4.14%`），hand `0.748531→0.718022 mm`（`4.08%`）。
- interaction 邻居数、采样 miss ratio 等输入统计在两个 epoch 相同；差异来自模型参数更新，而不是验证采样合同变化。
- 目前没有“同一 Decoder 分别接 epoch33 与 epoch52 Cm”的受控结果，因此不能把上述 Cm EPE 差异直接等同为 CmDecoder hand-flow EPE 的改善幅度。

## 2026-09-04 10:09:35 +0800 — V1.2 CmDecoder 训练吞吐差异诊断

- activity_id: `ACT-20260904-100935-CMDECODER-V12-THROUGHPUT-DIAGNOSTIC`
- timestamp: 2026-09-04 10:09:35 +0800
- modification_version: V1.2.12
- type: diagnostic
- change_level: L0（只读比较历史运行、checkpoint 配置、模型结构与性能日志）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 比较 `cm_decoder_20260903_091220` 与 `cm_decoder_20260904_001753` 的训练吞吐；不修改代码、配置、数据、cache、checkpoint 或运行状态
- run_id: `cm_decoder_20260903_091220`、`cm_decoder_20260904_001753`
- run_status: COMPLETED / STOPPED（历史运行状态；本次未改变）
- conclusion: INCONCLUSIVE（已定位性能原因，但未做隔离硬件 benchmark）

**原因**

- 用户询问本次 CmDecoder 比之前慢的原因，以及是否引入了额外计算；本次只读核对两个运行的 config、checkpoint 架构和 `perf/*` 日志。

**产物与证据**

- [旧 V1.2 运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/)、[旧 config](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/config.json)、[旧 metrics](../../../../../outputs/cmdecoder/cm_decoder_20260903_091220/metrics.jsonl)
- [本次运行目录](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/)、[本次 config](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/config.json)、[本次 metrics](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/metrics.jsonl)
- [CmDecoder adapter](../../object_interaction_cm_model.py)、[ObjectInteractionCm model](../../../ObjectInteractionCm/model.py)、[CmDecoder V1.2 指导](../指导/V1.2.md)、[CmDecoder V1.2 计划](../plan/V1.2.md)

**验证**

- 旧 V1.2 运行使用 3 张 RTX 3090 的 DDP（每卡 batch=16，global batch=48），稳定吞吐约 `1,650–1,680 samples/s`；本次使用单张 GPU3、batch=64，吞吐约 `270–273 samples/s`。总吞吐约慢 `6.1x`；按卡归一化后仍慢约 `2.0x`。
- 两次 Decoder 的数据合同基本一致：同一 pair manifest、1024 object 点、1538 Inspire 手点、十个 stride、`use_cached_cm_tokens=false`、无 AMP/compile；本次不是因为新增 stride 或新增训练样本，且当前 epoch steps 反而因 batch=64 减少。
- 真正变化是冻结 Cm checkpoint：旧运行加载 V1.1 legacy Cm（processing dim=32，Cm 参数 `45,256`）；本次加载 V1.2.1 Cm（processing dim=128，Cm 参数 `449,382`，约 `9.9x`）。新路径增加 delayed-compression 的 128 维 local interaction 双分支、fusion norm、128 维 SlotAttention、cm projection，以及更宽的 object/hand decoders。
- CmDecoder 每个 batch 仍在线执行完整 frozen Cm：`ObjectInteractionCmHandFlowAutoencoder.forward()` 调用 `self.cm(cm_batch)`；由于未生成 token sidecar，冻结不等于免计算。新 Cm 的 object/hand decoder 也会在这一步被完整 forward，尽管 CmDecoder 最终只取 tokens/anchors。
- 当前单卡 GPU3 还与约 `7.4 GiB` 显存的其他任务共享；旧三卡运行时 GPU0/1/2 为 Decoder 专用。该资源差异解释归一化后约 2 倍的剩余差距。当前/旧 `data_wait_ratio` 约 `2%–3%` / `13%`，因此 DataLoader 不是主要瓶颈。
- `ObjectInteractionCmHandFlowAutoencoder` 的可训练 Decoder 参数两次均为 `47,043`，所以慢点不在新 Decoder head，而主要在在线执行的新版 frozen Cm 和单卡共享资源。

## 2026-09-04 08:40:26 +0800 — V1.2 CmDecoder 长跑状态核查（进程已停止）

- activity_id: `ACT-20260904-084026-CMDECODER-V12-STATUS`
- timestamp: 2026-09-04 08:40:26 +0800
- modification_version: V1.2.12
- type: diagnostic / operation
- change_level: L0（只读进程、日志、metrics、checkpoint 与 GPU 状态）
- approval: user-requested
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: `cm_decoder_20260904_001753` 长跑状态核查；不修改代码、配置、数据、cache 或 checkpoint
- run_id: `cm_decoder_20260904_001753`
- run_status: STOPPED（进程已消失；非正常完成）
- last_step: `70600`
- last_epoch: `3`（epoch 3 进行中，未完成）
- best_metric: `7.932646 mm`（epoch 1 validation hand-flow EPE）
- best_checkpoint: [best.pt](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/checkpoints/best.pt)（step `35084` / epoch `1`）
- latest_checkpoint: [latest.pt](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/checkpoints/latest.pt)（step `35084` / epoch `1`）
- conclusion: INCONCLUSIVE（仅完成两轮完整 validation，第三轮中途停止，不能判断收敛）

**原因**

- 用户询问当前训练状态；本次仅核查现有运行进程、日志、metrics、checkpoint 与 GPU 状态，不启动新训练、不修改研究变量。

**当前证据**

- [metrics.jsonl](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/metrics.jsonl) 与 [train.log](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/train.log) 最后更新时间为 `2026-09-04 05:24:43 +0800`，最后记录为 step `70600` / epoch `3`。
- epoch 1 validation hand-flow EPE=`7.932646 mm`，epoch 2=`8.371129 mm`；zero-flow baseline=`12.182689 mm`。epoch 1 相对 zero-flow 改善约 `34.89%`，epoch 2 改善约 `31.29%`，epoch 2 未刷新 best。
- 当前没有 `src.task.CmDecoder.train` 进程，GPU3 仅保留其他进程；未在 train.log、dmesg 或 kernel journal 中发现 traceback、OOM、NCCL、NaN 或 Inf。因此只能确认运行已中断，无法确认外部停止原因，不能将其解释为收敛或正常结束。
- [run_manifest.json](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/run_manifest.json) 与运行目录保留；如需继续，应显式从已有 checkpoint resume，并单独登记新的运行状态。

**验证**

- 只读检查命令：`ps`、`tail`、`metrics.jsonl` 解析、checkpoint metadata、`nvidia-smi`、`dmesg`/`journalctl`。
- 未停止、重启或修改其他 GPU 上的用户进程；未删除任何运行产物。

## 2026-09-04 00:20:13 +0800 — V1.2 当前 ObjectInteractionCm best 驱动 CmDecoder 长跑

- activity_id: ACT-20260904-002013-CMDECODER-V12-CURRENT-OI-BEST-START
- timestamp: 2026-09-04 00:20:13 +0800
- modification_version: V1.2.12
- type: experiment / operation
- change_level: L3（用户批准的冻结 checkpoint 变体与 GPU3 长时训练）
- approval: user-approved
- approval_basis: 用户明确要求评估后使用当前 best Cm 训练一版 CmDecoder，指定 GPU3、允许增大 batch，并先短跑测时后持续运行
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 6ca6509ab1f8853767d11724877dd8aba5914b63
- worktree_dirty: true（保留既有 ObjectInteractionCm 活动记录及仓库用户改动）
- scope: CmDecoder V1.2 Inspire-F1 object-disjoint all-stride conditional hand-flow reconstruction；冻结当前 ObjectInteractionCm best checkpoint，Decoder 随机初始化，只更新新 hand-flow decoder；不修改 `src/base`、geometry cache、pair index、旧运行或 GPU0/1/2 训练
- run_id: cm_decoder_20260904_001753
- run_status: RUNNING
- command: `CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.CmDecoder.train --config src.task.CmDecoder.current_object_interaction_cm_v1_2_config:Config --set meta.cm_checkpoint=outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt --set train.device=cuda --set data.batch_size=64 --set data.val_batch_size=64 --set train.epochs=30 --set train.max_steps=none --set data.max_episodes=none --set data.max_frames_per_episode=none --set data.num_workers=4 --set data.persistent_workers=true --set data.drop_last=true --set train.log_every_steps=100 --set train.eval_every_epochs=1 --set train.save_every_epochs=10 --set train.description=CmDecoder_V1.2_full_current_OI_best_GPU3_batch64`
- conclusion: INCONCLUSIVE（训练进行中；短跑仅为工程与吞吐证据）

**文件与证据**

- [长跑目录](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/)、[config.json](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/config.json)、[run_manifest.json](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/run_manifest.json) — 正式运行配置、数据合同和运行入口；manifest 已补充冻结 Cm checkpoint SHA256。
- [metrics.jsonl](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/train.log) — 当前 step、吞吐、loss 和 ETA。
- [当前 OI best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt) — step `133105`、epoch `33`、best object metric `3.750058 mm`、SHA256 `1dfd892771907a6a26aee54103faebba249f55699c11e4383f8a23fc201d4021`。
- [batch=64 timing smoke](../../../../../outputs/cmdecoder/cm_decoder_20260904_001447/metrics.jsonl)、[batch=128 timing smoke](../../../../../outputs/cmdecoder/cm_decoder_20260904_001605/metrics.jsonl) — 短跑资源估计，不作为科研结论。
- [checkpoints 目录](../../../../../outputs/cmdecoder/cm_decoder_20260904_001753/checkpoints/) — PENDING：首个验证/保存 checkpoint 尚未生成。

**原因**

用户要求以刚评估后的当前 OI best 作为冻结 Cm 编码器，训练新的 Inspire-F1 全点 Decoder。batch=64 的全量实测约 `270 samples/s`，高于 batch=128 短跑的有效吞吐且显存余量更稳，因此选择 batch=64；全量 30 epochs 预计约 `69 h`，六小时左右应可完成约 2–3 个 epoch。

**验证**

- batch=64 smoke：20 steps / 10 s，输出 `[B,1538,3]`，无 NaN/OOM；batch=128 smoke：8 steps / 9 s，同样通过但吞吐未提升。
- 长跑当前 step `500`、epoch `1`，`perf/samples_per_s≈273`、`ETA≈68.5 h`；GPU3 总显存约 `15.2/24 GiB`（CmDecoder 约 `7.8 GiB` 加既有足球任务约 `7.4 GiB`），无 OOM。
- GPU0/1/2 的 ObjectInteractionCm 三卡训练进程仍存活；GPU4–7 的既有任务未停止或抢占。
- `git diff --check` 和 `audit_diff.py --check-links` 将在本次交接后执行；本条只追加活动记录和运行产物，不修改代码或研究变量。

**回滚**

停止 `cm_decoder_20260904_001753` 即可终止本次长跑；删除/隔离该新运行目录即可回滚运行产物。当前 OI checkpoint、旧 CmDecoder 运行和所有 cache 均保持不变。

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
