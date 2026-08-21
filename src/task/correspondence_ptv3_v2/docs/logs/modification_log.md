# Modification log

## 2026-08-21 — 固定 GRAB H50/O50 互斥扰动配置并核对三域混训中断

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_h50_o50_exclusive_ddp2.yaml` — 新增 H50/O50 互斥扰动训练配置。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录三域混训的实际停止证据及 H50/O50 实验入口。

**改动原因**

用户确认采用方案 A：在保持 5 mm hand noise 和既有 object pose perturbation 的前提下，严格互斥地以 50% hand-only、50% object-only 训练，禁止 hand+object compound 样本。

同时复核已停止的三域混训目录：日志最后写到 step 27980，无 Python traceback、CUDA OOM、kernel OOM 或 checkpoint 临时文件；现有 checkpoint 只到 step 22748，支持“外部停止/会话结束发生在保存周期之间”的判断。

补充证据：W&B 本地日志 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_20260820_141453/wandb/run-20260820_141505-div3m72u/files/output.log` 同样以 step 27980 结束，未出现终止原因。

**对应指导**（如有）

当前三域混训方案；ARCTIC 外部评估遵循 `docs/指导/V1.md`。

**影响范围**

仅 task 内部配置和实验记录；未修改三域 loader 或旧 checkpoint。

## 2026-08-21 — 导出 ARCTIC MANO min11 并扩展三条件评估入口

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件 / 数据**

- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage2/arctic_min11_mano_v1` — 从 raw ARCTIC MANO 生成 11 个 Stage 2 文件。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1` — 生成 11 个带 MANO 的 Stage 3 `.npz`，共 4175 帧。
- `src/task/correspondence_ptv3_v2/config.py` — 增加 validation object perturb 开关，保留默认行为。
- `src/task/correspondence_ptv3_v2/dataset.py` — validation perturbed loader 可按 condition 关闭 object perturb。
- `src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py` — 增加 `object_only` / `hand_only` / `hand_object` 条件和 ARCTIC axis-angle45 5mm calibration。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_v1_manifest.json` — 固化 NAS 数据清单和 MANO schema。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_v1_compare_20260821.md` — 记录三条件结果。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md`、`decision_log.md`、`repo_notes_log.md` — 同步实验、决策和路径。

**改动原因**

响应用户要求先导出部分含 MANO 的 ARCTIC 数据，补齐 hand-only / hand+object 公平评估；不覆盖旧无 MANO缓存或全量数据。

**验证**

- 11/11 文件均含 `mano_pose (T,45)`、`mano_global_orient`、`mano_transl`、`mano_betas`。
- 全部标记 `mano_pose_repr=axis_angle`、`mano_use_pca=false`。
- evaluator `py_compile` 通过；三条件六个 JSON 结果均生成。

**影响范围**

仅 task 内部数据、评估入口和实验记录；默认 object-only evaluator 语义保持兼容。

## 2026-08-21 — 修正 min11 recovery 结果方向并确认互斥暴露差异

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_compare_20260821.md` — 将 recovery 指标按“越高越好”修正为 noPCA 优于 5mm，并补充 object exposure 解释。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 修正 EXP-008 的结论和下一步。

**改动原因**

复核 `pseudo_recovery_brier = (E_pseudo - E_model) / E_pseudo` 后，发现前一版结果表错误地把 5mm 的较低 recovery 标成了更优。代码指标方向本身正确；差异主要与 5mm 互斥门控的约 20% object-only 暴露、noPCA 的 100% object-only 暴露有关。

**影响范围**

仅修正科研记录，不修改模型、训练代码或评测实现。

## 2026-08-21 — 新增 ARCTIC min11 快速评估结果记录

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_manifest.json` — 固化覆盖 11 个物体的最小评估文件清单。
- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_compare_20260821.md` — 记录 noPCA / 5mm 的统一 object-only 协议、结果和限制。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 新增 EXP-008，记录方向性证据和后续 matched evaluation。

**改动原因**

响应用户“先评估、控制规模但覆盖类别”的要求；使用 11 个物体各一个文件的快速筛查，避免全量 ARCTIC 评测耗时过长。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

**影响范围**

仅 task 内部评测结果和科研记录；未修改模型、loss、训练代码或 evaluator。

## 2026-08-20 — 新增 GRAB/ContactPose/OakInk 等比例从头混训入口

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/dataset.py` — 新增 `DomainConcatDataset` 和 `DomainBalancedSampler`；支持按域独立切分 train/val，并在每个 batch 固定 1/3 域比例；`use_mano_reconstruction=false` 时不再把 GRAB 的 MANO 字段混入无 MANO 域的 batch。
- `src/task/correspondence_ptv3_v2/config.py` — 增加 task 级 `data.domain_paths` 配置字段。
- `src/task/correspondence_ptv3_v2/configs/mixed_grab_contactpose_oakink_equal.yaml` — 新增三域从随机初始化开始的 baseline 配置；显式关闭 runtime object resampling、MANO/hand perturb 和 scale augmentation，保留 object pose perturbation。
- `src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md` — 记录三域 loader、采样和 validation 数据流。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录等比例 sampler、batch size 48 和独立 validation loader 的自主选择。

**改动原因**

落实用户确认的三域从头训练协议：GRAB、ContactPose、OakInk 各 1/3，使用 domain sampling，不开启 runtime object resampling，不加入 scale augmentation。

**对应指导**

当前三域混训方案；V1 的 ARCTIC 外部评估指导仍作为启动前评测依据。

**验证**

- 配置成功加载；
- 实际 NAS 数据 Dataset smoke 成功，三个域 train/val 均构造完成；
- 首个 batch 为 `16 grab + 16 contactpose + 16 oakink`；
- batch 不含 runtime full-pool 字段，且无 MANO schema collate 冲突；
- `py_compile` 通过。

## 2026-08-21 — 记录并停止 no-PCA runtime 续训

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- 范围: task 内部实验状态

**文件**

- `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614` — 保留 no-PCA runtime run 的日志和 checkpoint，停止前 step 331300 / epoch 37，最新保存 checkpoint 为 step 317800 / epoch 35。
- `src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md` — 记录 run 路径、停止状态及后续 object-perturb exposure 对照方向。

**改动原因**

用户要求保留 no-PCA 版本路径并停止当前训练，转向验证 5 mm exclusive hand/object perturbation 是否降低了 object perturb 暴露比例。

**影响范围**

仅 task 内部实验进程和记录；未修改模型或数据。

## 2026-08-20 — 完成 V1 ARCTIC 分层外部评估与 object-macro 汇总

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/research/arctic_v1/build_stratified_subset.py` — 构造覆盖全部 object / subject / action 的确定性分层软链子集。
- `src/task/correspondence_ptv3_v2/research/arctic_v1/build_object_subsets.py` — 为每个 object 构造单独的软链子集，便于按 object 汇总 macro。
- `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.md` — 写入 V1 ARCTIC micro / object-macro 对比结果。
- `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.json` — 保存 V1 汇总 JSON。

**改动原因**

用户要求按照 V1 指导实现：在不训练 ARCTIC 的前提下，用 held-out ARCTIC 评估纯 GRAB 与 GRAB+ContactPose。由于现有 evaluator 只输出全局 micro，需要额外构造 object 子集来计算 object-macro。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

**影响范围**

仅 task 内部实验、结果和研究脚本，不改变训练代码、模型、loss 或 evaluator 协议。

## 2026-08-20 — 新增 ARCTIC 全量/分层外部评估 V1 指导

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`
- 范围: task 内部

**文件**

- `src/task/correspondence_ptv3_v2/docs/指导/V1.md` — 规定现有纯 GRAB 与 GRAB+ContactPose checkpoint 的全量 ARCTIC 或全类别分层子集评测流程。
- `src/task/correspondence_ptv3_v2/docs/logs/repo_notes_log.md` — 将 V1 与 ARCTIC 外部评估更新为当前任务入口和研究近况。
- `src/task/correspondence_ptv3_v2/docs/logs/decision_log.md` — 记录时间受限时采用 subject/object/action 确定性分层的自主选择。
- `src/task/correspondence_ptv3_v2/docs/logs/modification_log.md` — 记录本次指导文档新增。

**改动原因**

响应用户要求，将启动三数据集混训前首先要完成的 ARCTIC 外部评估固化为可执行指导，修正旧 20 文件子集仅含 `box_*` 的类别偏置，并明确统一协议、结果统计和后续决策门。

**对应指导**

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

**影响范围**

仅 task 内部研究指导与记录，不修改代码、checkpoint、训练进程或评测实现。

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
