# CmDecoder 修改记录

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
