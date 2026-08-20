# Modification log

## 2026-08-20 — 重新拉起 GRAB 5 mm no-PCA runtime 训练

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614` — 从 `latest.pt` 重新恢复同一训练 run。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录训练已在 tmux `grab5mm_no_pca_runtime_resume_20260820` 中重新拉起。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次恢复动作。

**改动原因**

上一轮续跑因本地根分区满盘而中断。NAS 软链释放空间后，按原 checkpoint 和原配置重新拉起同一训练 run。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 训练已重新启动并进入 DDP 恢复流程。
- 剩余: 继续观察是否能稳定跑到下一个 checkpoint。

**影响范围**

仅 task 内部训练进程状态，不改变模型、数据或评测协议。

## 2026-08-20 — 将 GRAB 大体积数据切到 NAS 软链

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `dataset/GRAB/data` — 改为指向 NAS 上 `GRAB/data/GRAB` 的软链。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 记录 GRAB 数据根的 NAS 路径。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录训练中断原因是根分区满盘，并注明已释放空间。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次修改。

**改动原因**

本地根分区已满，GRAB 的 `data` 目录是当前最大的可迁移只读数据。NAS 上已有对应数据，因此将本地入口切换到 NAS 软链，释放本地空间并保持代码路径不变。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 软链已切换完成，本地空间已释放。
- 剩余: 如需，后续继续把其他只读大数据目录也切到 NAS。

**影响范围**

仅 task 内部路径与文档，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 迁移 processed_data 到 NAS 并补结果配置摘要

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `data/processed_data` — 迁移到 NAS 上的 `processed_data` 目录软链。
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` — 补充 OakInk / GRAB / GRAB+ContactPose 的训练配置摘要。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 记录 task 级 processed_data NAS 路径。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次迁移和结果文档更新。

**改动原因**

响应用户要求，将本地 processed_data 迁到 NAS，减少本地盘占用，并把当前三模型对比结果中的训练配置补齐，便于以后直接核对协议差异。

**影响范围**

仅 task 内部路径与文档，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 完成三模型 ARCTIC 子集对比并修正结果口径

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` — 写入三模型 ARCTIC 子集对比表，并修正 recovery 高优标注。
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.json` — 保存结构化汇总结果。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 将当前状态改为已完成，并新增 EXP-005。
- `src/task/correspondence_ptv3_v2/research/log.md` — 修正 ARCTIC 结果解读中关于 recovery 方向的旧表述。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次修改。

**改动原因**

响应用户要求，完成 OakInk / GRAB / GRAB+ContactPose 在同一 ARCTIC 子集上的对比汇总，并把旧文档里和实际指标不一致的 recovery 口径修正为任务架构定义下的正确方向。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 三模型对比结果已落盘，相关实验和研究日志已同步。
- 剩余: 无。

**影响范围**

仅 task 内部文档与结果摘要，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 记录三模型 ARCTIC 对比的当前接续点

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录 OakInk、纯 GRAB、GRAB+ContactPose 在统一 ARCTIC 子集上的当前对比目标、协议、已有结果和待完成产物。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档同步。

**改动原因**

响应用户要求，先把当前正在进行的三模型 ARCTIC 对比写入任务文档，避免后续中断后丢失评测口径和接续点。已有纯 GRAB / GRAB+ContactPose 结果将复用，OakInk 已按相同协议完成子集评测，下一步生成时间戳结果表。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/架构.md`

**单次修改进度**

- 当前: 对比定义和评测协议已记录，OakInk 子集评测已完成。
- 剩余: 汇总三方 JSON/历史 JSON，生成 `src/task/correspondence_ptv3_v2/result/` 下的时间戳 Markdown/JSON 结果。
- 续接点: 使用 `/tmp/oakink_arctic_eval_20260820.json`，并读取已有 `arctic_pure_20260817_132348.json`、`arctic_mixed_20260817_132348.json`。

**影响范围**

仅 task 内部文档和研究结果汇总，不改变训练代码、checkpoint 或评测协议。

## 2026-08-20 — 续跑 GRAB 5 mm no-PCA runtime 训练

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 更新任务入口状态，说明 GRAB no-PCA runtime run 已进入续跑。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新当前状态：原进程停在日志 step 36320 / epoch 4，已从 `latest.pt` 的 step 27240 / epoch 3 续跑。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次训练续接。

**改动原因**

用户要求继续训练当前 GRAB “5 mm 配置 + runtime sampling + no-PCA hand perturb” run。原进程已停止且无 Python traceback；`latest.pt` 是最近完整 checkpoint，因此从该 checkpoint 恢复。

训练续接信息：

- tmux: `grab5mm_no_pca_runtime_resume_20260820`
- checkpoint: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614/checkpoints/latest.pt`
- resume step / epoch: `27240` / `3`
- W&B run id: `zeqrhxyk`

**影响范围**

仅 task 内部训练进程与文档状态；不改变模型、数据、配置或评测协议。

## 2026-08-19 — 新增 5 mm no-PCA runtime 配置

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_no_pca_runtime_ddp2.yaml` — 以 5 mm 配置为基线，仅关闭 `apply_hand_perturb`，保留 runtime object resampling。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 更新任务入口状态，说明正在启动新的 GRAB no-PCA runtime run。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新当前状态，记录正在启动新的 no-PCA runtime 训练。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录把 5 mm 作为 no-PCA runtime 新 run 的基线、只关闭 hand perturb 的选择。

**改动原因**

用户要求训练一版“无 PCA 扰动，但用 runtime 采样，配置和 5 mm 一致”的 GRAB 训练。

训练已通过 `torchrun` 拉起，当前 run 目录为 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614`。

**影响范围**

仅 task 内部配置与实验记忆，不改变现有训练代码或评测协议。

## 2026-08-19 — 切换后续 no-PCA 默认对照并补数据集口径说明

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 新增后续 no-PCA 对照默认使用 mid-Aug compact repro 的决策。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 将当前状态中的 no-PCA 默认口径改为 mid-Aug compact repro。

**改动原因**

用户明确要求以后不再使用 `finetune_day_cosine_restart` 的续训口径作为默认 no-PCA 对照，而改用 8 月中旬训练的那条结果更接近的 no-PCA run。

**影响范围**

仅 task 内部文档口径，不改变训练、验证或模型代码。

## 2026-08-19 — 续接 OakInk 无手扰动训练并补齐 baseline 决策记录

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 更新 OakInk 正式训练当前状态：原进程日志停在 step 255300，已从 `latest.pt` 的 step 211410 续跑。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 补充 `finetune_day_cosine_restart` 作为 no-PCA canonical 参考的口径选择。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新与训练续接。

**改动原因**

继续 01a01514-d636-7d50-8a45-4eff274a7f03 会话中断后的任务：完成 GRAB 5 mm 与 canonical no-PCA baseline 对比的文档收尾，并发现 OakInk 单卡训练不在进程列表中后，按原 50 epoch 需求从 `latest.pt` 续跑。`latest.pt` 对应 step 211410，因此日志中 step 211410 到 255300 的未 checkpoint 训练进度不会被保留为模型状态。

**影响范围**

仅 task 内部。训练续接使用原 run 目录和原配置，只额外设置 `train.resume=.../checkpoints/latest.pt`；不改变模型、数据或评测协议。

## 2026-08-19 — 记录 GRAB 5 mm 与 `finetune_day_cosine_restart` 复现对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加 GRAB 5 mm PCA 互斥扰动版与 `finetune_day_cosine_restart` / 复现 run 的当前 best.pt 对比实验记录，并更新当前状态。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录 `finetune_day_cosine_restart` 作为 canonical no-PCA 参考的选择依据。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户澄清后的比较对象，对当前 5 mm 版本与几天前那条 no-PCA 复现链路做同口径对比，并将结果沉淀为可追溯实验记录。

**影响范围**

仅补充 task 实验记忆，不改变代码、训练配置或评测协议。

## 2026-08-19 — 记录 GRAB 5 mm 与旧 no-PCA legacy run 的方向性对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加 GRAB 5 mm PCA 互斥扰动版与旧 no-PCA legacy GRAB run 的方向性对比实验记录，并更新当前状态。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户要求，对当前 5 mm 加 PCA 扰动版本与之前 no-PCA 版本做进一步比较，并明确这次对比的协议差异与适用范围。

**影响范围**

仅补充 task 实验记忆，不改变代码、训练配置或评测协议。

## 2026-08-19 — 记录 GRAB 5 mm 与两条 9 mm best.pt 的当前指标对比

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加 GRAB 5 mm PCA 互斥扰动版与两条 9 mm baseline 的当前 best.pt 对比实验记录，并更新当前状态。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户要求，对未训练完的 GRAB 5 mm 加 PCA 扰动版本与此前 9 mm 版本做当前 best.pt 性能对比，并将结果沉淀为可追溯实验记录。

**影响范围**

仅补充 task 实验记忆，不改变代码、训练配置或评测协议。

## 2026-08-19 — 记录 OakInk/GRAB 训练吞吐与四数据集规模核验

- branch: `feature/hand-pca-perturbation`
- post-commit: 未提交（基于 `d5510e0`）
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 追加运行进度、cache/吞吐诊断及 GRAB、OakInk、ContactPose、ARCTIC 的统一帧样本统计。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次文档更新。

**改动原因**

响应用户对 OakInk 无 PCA 训练耗时、cache 影响和四数据集帧数/训练样本关系的追问。统计为只读核验，不改变实验定义、训练配置或代码。

**影响范围**

仅补充 task 实验记忆，供后续评估训练预算与数据集组合时引用。

## 2026-08-18 — 支持 OakInk 无 MANO 手部扰动但保留 runtime sampling

- branch: `feature/hand-pca-perturbation`
- post-commit: 未提交（基于 `d5510e0`）
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/runner.py` — 加载并校验存储手点 proxy；无 MANO 路径使用该 proxy 完成 full-pool near/global 采样。
- `src/task/correspondence_ptv3_v2/config.py` — 新增 `stored_hand_proxy_indices_path`。
- `src/task/correspondence_ptv3_v2/calibration/oakink_stored_hand_fps256.json` — OakInk 存储手点 FPS-256 版本化索引。
- `src/task/correspondence_ptv3_v2/configs/oakink_50ep_no_hand_perturb_runtime.yaml` — OakInk 无手部扰动候选训练配置。
- `src/task/correspondence_ptv3_v2/research/oakink_conversion/validate_wrist_centering.py` — OakInk 多视角 wrist/camera/world 几何验证脚本。
- `README.md` 和 `docs/架构.md` — 同步路径、坐标系和无 MANO runtime 约定。

**改动原因**

用户选择方案 B：不重建 MANO、不加 PCA 手部扰动，但仍保留 runtime object resampling。

**验证**

- OakInk 8 个多视角组、24 个 view pair：只减 wrist 的手点跨视角 RMS 平均 163.14 mm；用 `cam_extr^{-1}` 后为 0.000080 mm；接触距离跨视角 MAE 为 0.000035 mm。
- stored proxy 与 MANO proxy 的 top-1024 near-pool Jaccard 平均 0.948。
- 无 MANO 模型目录下 dataset/runtime smoke、PTv3 forward、loss 和反向传播均通过。
