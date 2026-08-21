# CmDecoder 修改记录

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
