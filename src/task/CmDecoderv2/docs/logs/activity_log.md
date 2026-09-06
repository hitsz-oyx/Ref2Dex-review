# CmDecoderv2 活动记录

- scope: `src/task/CmDecoderv2/`
- last_updated: 2026-09-06
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)

## 2026-09-06 17:28:42 +0800 — V1.1.1 修复 handoff 帧 Viser 回调崩溃并重启 GPU3 viewer

- activity_id: `cmdecoderv2-mano-grab-vis-20260906-172800`
- timestamp: `2026-09-06 17:28:42 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic / code / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户报告 viewer 中断；根据 traceback 修复 Task-local UI 对 handoff 初始状态的错误处理并重新启动同一 viewer。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 CmDecoder、ObjectInteractionCm、CmDecoderv2 及训练改动）
- run_id: `cmdecoderv2-mano-grab-vis-20260906-172842`
- run_status: `RUNNING`
- conclusion: `SUPPORTED`（已定位并修复 handoff UI 的工程错误；MANO→Inspire 科研效果仍为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/`；未修改正式 decoder 训练、GPU0/1/2 训练进程、数据 cache、split 或 checkpoint。

**文件**

- [visualize_mano.py](../../visualize_mano.py) — handoff 帧的 Inspire 初始状态 `cm_valid=None` 时显示 initialized 文本，不再执行 `len(None)`。
- `docs/current_versions.yaml` — 保留既有 `CmDecoderv2: V1.1.1` 指针；本次不改变版本值。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172842/run_manifest.json) — 修复后 GPU3 viewer 的运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172842/trajectory.npz) — 修复后运行的 GT 和 rollout 槽位。

**原因**

旧运行在切换 rollout 后推进 frame 时崩溃。traceback 显示 handoff 初始状态的 `cm_valid=None` 被 UI 当成数组执行 `len(valid)`；这是状态语义正常、显示分支错误，不是模型或 parent cache 数据错误。

**验证**

- 原始 traceback 已保存于旧 Viser session：`TypeError: len() of unsized object`，位置为 `visualize_mano.py` 的 Cm-valid 文本分支。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_mano.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- 修复后 GPU3 viewer 已在 `http://localhost:8098` 启动，HTTP 200；新的 run manifest 已生成，formal decoder training GPU0/1/2 未被中断。
- `git diff --check` 与 activity link audit：通过。

**保护边界与回滚**

- 旧崩溃运行 `cmdecoderv2-mano-grab-vis-20260906-172219` 的输出保留作诊断证据；修复后运行使用新的 run_id，不覆盖旧输出。
- 回滚入口为移除本次 handoff 分支改动或隔离新 viewer 输出；不删除原始 GRAB parent cache、decoder checkpoint 或正式训练输出。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 17:22:19 +0800 — V1.1.1 新增 MANO/GRAB-source Viser 定性 viewer 并启动 GPU3

- activity_id: `cmdecoderv2-mano-grab-vis-20260906-172000`
- timestamp: `2026-09-06 17:22:19 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确确认最终可视化必须使用原始 GRAB MANO 手和物体轨迹；Dexplore RL 只用于 decoder 训练；允许从任意帧以 q=0 + MANO wrist 切换到 rollout。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 CmDecoder、ObjectInteractionCm、CmDecoderv2 及训练改动）
- run_id: `cmdecoderv2-mano-grab-vis-20260906-172219`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（仅确认数据来源、坐标和 viewer wiring；没有 paired Inspire test GT，不作定量或科学效果结论）
- scope: `src/task/CmDecoderv2/`；未修改正式 decoder 训练、GPU0/1/2 训练进程、Dexplore/GRAB 原始数据、已有 RL-Inspire 诊断 viewer。

**文件**

- [visualize_mano.py](../../visualize_mano.py) — 新增最终定性 viewer：GT playback 渲染原始 GRAB parent cache 的 MANO hand/object；rollout 在当前帧以零 finger q 和 MANO `hand_root_pose_world` 初始化，使用 MANO source Cm window 递归预测 Inspire；GT/pred/both 和点大小滑块均可切换。
- `docs/current_versions.yaml` — 保留既有 `CmDecoderv2: V1.1.1` 指针；本次不改变版本值。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172219/run_manifest.json) — 当前 GPU3 Viser 运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172219/trajectory.npz) — 原始 MANO/GRAB GT、物体轨迹和当前已生成的 rollout 槽位。

**原因**

之前的 `visualize_teacherforce.py` 使用 Dexplore RL-Inspire validation GT，因此不能作为最终 MANO→Inspire 测试。新 viewer 将 Dexplore RL 限定为 decoder 训练来源，最终可视化只使用原始 GRAB parent cache；没有把对应 Inspire 轨迹作为测试标签，避免把不同抓取方式混在比较中。

**验证**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize_mano --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence s1/alarmclock_offhand_1 --port 8098 --fps 8 --activity-id cmdecoderv2-mano-grab-vis-20260906-172000`
- checkpoint SHA256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`。
- viewer: `http://localhost:8098`；GPU3 进程正在运行，formal decoder training 的 GPU0/1/2 进程未停止。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_mano.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- GPU3 real-checkpoint smoke：加载 `s1/alarmclock_offhand_1` 原始 parent cache，GT shapes 为 hand `[271,1538,3]` / object `[271,4096,3]`；rollout handoff frame 0 后递归到 frame 1，预测 Inspire points `[1538,3]`、q `[6]`、wrist `[4,4]`，输出 finite。
- `curl http://localhost:8098/`：HTTP 200；run manifest 与 trajectory 已生成。
- `git diff --check`：通过。

**保护边界与回滚**

- 未修改旧 RL-Inspire teacher-forcing viewer；未修改任何数据 cache、split、checkpoint 或正式训练配置。
- 回滚入口为删除/隔离新增 [visualize_mano.py](../../visualize_mano.py) 和本次 viewer 输出目录；不删除原始 GRAB parent cache 或 decoder checkpoint。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 16:42:31 +0800 — V1.1.1 object pose 非接触运动因果诊断

- activity_id: `ACT-20260906-164231-CMDECODERV2-OBJECT-MOTION-CAUSALITY-DIAGNOSTIC`
- timestamp: `2026-09-06 16:42:31 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户追问“没有接触为什么物体会动”；只读核对 Dexplore native tensor 字段定义、cache builder 和 frame 200 轨迹，不修改数据或训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（确认数据没有“无接触冻结物体”的物理约束；是否按右手接触重构/裁剪物体轨迹仍待研究决定）
- scope: `src/task/CmDecoderv2/`、ObjectInteractionCm cache builder 和 Dexplore export schema 的只读诊断；未修改研究变量。

**原因**

确认 frame 200 无右手几何接触时 object pose 仍变化的直接数据来源，并区分 object trajectory、contact label 与 viewer 渲染行为。

**关键证据**

- Dexplore native tensor 的 `198:201` 是 object position、`201:205` 是 object quaternion、`205` 是 object contact flag；cache builder 分别读取 pose 和 flag，没有用 contact flag 对 pose 做冻结或门控。
- 因此 `contact=0` 只表示该帧的 object contact label 为 0，不代表 `T_object[t+1] == T_object[t]`；数据管线不承诺无接触时物体静止。
- `s1/banana_lift` frame 200 的 native contact flag 为 `0`，右手到物体最近距离约 `32.3 mm`，但 object frame 200→201 仍有约 `1.2 mm` translation 和 `2.57°` rotation；这是原始 object pose trajectory 中的轻微抖动/轨迹变化。
- Dexplore 文档明确指出 object motion 可能由未被当前 Inspire 右手显示的其他手/身体接触驱动；当前 tensor 的 object flag 是 object-level contact，不是本 viewer 右手的独立因果门控。
- 更大的 object motion 出现在 frame 220–230，此前 frame 220–226 仍存在 `<2 cm` 接触，随后 object z 约从 `1.129 m` 上升到 `1.335 m`，说明该段是接触后的真实 object trajectory，而非 Viser 自动生成。

**证据入口**

- [Dexplore export schema](../../../../../../dexplore/data_processing/README_GRAB_INSPIRE_EXPORT.md)
- [cache builder](../../tools/data/build_dexplore_view.py)
- [ObjectInteractionCm RL cache builder](../../../ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读核对 native tensor 字段、builder 字段映射、contact flag、world-frame object pose 和 frame 200 hand-object 距离；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:38:21 +0800 — V1.1.1 Viser 世界坐标与 frame 200 物体运动诊断

- activity_id: `ACT-20260906-163821-CMDECODERV2-VISER-COORD-FRAME200-DIAGNOSTIC`
- timestamp: `2026-09-06 16:38:21 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 Viser 使用的坐标系，并指出 frame 200 GT 未接触但物体仍在动；只读核对 viewer 数组索引、world/object pose 和 frame 200 的 hand-object 距离。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（当前 world-frame 物体运动与数组一致；是否要改成 object-centric viewer 仍待用户决定）
- scope: `src/task/CmDecoderv2/` 与 validation geometry 的只读诊断；未修改 viewer、数据或训练。

**原因**

需要确认 frame 200 的物体运动是否来自 Viser 坐标/索引错误，还是来自 Dexplore geometry 中记录的 object pose trajectory。

**关键证据**

- 当前 Viser 使用 `obj_points_world.npy`、`hand_points_world.npy` 和 `wrist_pose_world.npy` 的 Dexplore world frame（manifest 标记为 `dexplore_native_object_pose_world`），没有把所有帧变换到固定 object-centric frame。
- teacherforced frame `t` 显示：object 使用 frame `t+1`，GT current 使用 frame `t`，GT next 使用 frame `t+1`；因此 frame 200 的场景物体是 object frame 201，绿色 GT next 也是 hand frame 201。
- frame 200 的 hand→object 最近距离约 `32.3 mm`，`<20 mm` 接触比例为 `0%`，所以此帧确实没有 2 cm 级物理接触；但此前 frame 175–195 有多段 `<2 cm` 接触，frame 200 是一次短暂分离后的状态。
- frame 200→201 物体质心移动约 `1.4 mm`，object pose translation 约 `1.2 mm`，旋转约 `2.57°`；因此 world-frame 播放时物体确实会轻微移动，主要是该帧的 object pose/rotation trajectory，不是 viewer 凭空移动。
- 更明显的 object motion 出现在 frame 220–230：此区间 hand 在 frame 220–226 仍有 `<2 cm` 接触，之后 object pose z 从约 `1.129 m` 上升到 `1.335 m`；这与前一段接触后的物体运动一致。

**证据入口**

- [teacherforce viewer](../../visualize_teacherforce.py)
- [geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读比较 world-frame object/hand 数组、object pose 帧间增量和 frame 200 的近邻距离；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:29:24 +0800 — V1.1.1 validation GT 预接触运动诊断

- activity_id: `ACT-20260906-162924-CMDECODERV2-GT-PRECONTACT-DIAGNOSTIC`
- timestamp: `2026-09-06 16:29:24 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户反馈 validation Viser 中 GT 在接触物体前已经运动；只读检查 RL q、wrist、hand/object geometry、5 cm candidate mask 和 source contact flag，不修改 viewer、数据或训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（数值上 GT 与 cache/FK 自洽，观察到的是动作的预接触阶段；尚未决定是否新增 grasp-only crop）
- scope: `src/task/CmDecoderv2/` 与 `data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/sequences/val/s1_banana_lift/` 的只读诊断；未改变研究变量。

**原因**

需要确认“接触前运动”是数据包含 reach/approach 动作，还是 q/wrist、hand geometry、object pose 或 frame index 错位。

**关键证据**

- `s1/banana_lift` 为 30 Hz、578 帧；物体基本静止，手从 frame 0 就在接近，hand-object 最近距离从约 `1316 mm` 降到 frame 50 的 `72 mm`。
- 按生成的 5 cm candidate mask，首个 active frame 为 `53`；按实际最近距离，首个 `<20 mm` 接触帧为 `54`；Dexplore source contact flag 首个为 `56`。因此前约 `1.8 s` 是正常 reach/approach 段，不是 hand 静止等待接触。
- frame 0→1 的 GT 手指 q 变化约 `24.18°`（6 维 q 的 L2），腕部平移约 `40.76 mm`、旋转约 `3.39°`；这与“动作一开始就运动”一致。
- `geometry/manifest.json` 标明 RL hand geometry 由 Dexplore native q 经 Inspire URDF FK 生成，decoder view 的 wrist pose 也由同一 q/FK 生成；当前证据未发现 GT frame index 或 coordinate-frame 错位。
- source flag 与几何 5 cm mask 不完全同帧：source flag 从 frame 56 开始，几何 mask 从 frame 53 开始；这是 5 cm candidate 与 Dexplore contact label 的语义差异，不等于 GT 错位。

**证据入口**

- [geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)
- [decoder view manifest](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/sequences/val/s1_banana_lift/manifest.json)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读计算 hand/object 最近距离、候选 active、source contact flag、GT q/wrist 帧间增量和 object/hand centroid 轨迹；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:20:49 +0800 — V1.1.1 Viser 点云显示与 teacherforced/rollout 切换

- activity_id: `ACT-20260906-162100-CMDECODERV2-TEACHERFORCE-TOGGLE`
- timestamp: `2026-09-06 16:20:49 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户确认暂不使用 mesh，要求点云大小滑块、GT/pred/both 显示切换以及 teacherforced/rollout 可随时切换；采用已说明的 GT handoff 语义。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、旧 rollout 和用户改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser server 使用 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（交互可视化 wiring 已通过 smoke；不构成 decoder 科研效果结论）
- scope: `src/task/CmDecoderv2/`（更新 `visualize_teacherforce.py`）与独立 teacher-forcing Viser 输出目录；未修改训练变量、checkpoint 或正式训练。

**原因**

需要在同一 validation sequence 中直接比较 teacher forcing 与 state-feedback rollout，并能按帧选择 GT、预测或两者，同时调整点云可见性。

**实现与运行**

- Viser 新增 `Execution mode`：`teacherforced` / `rollout`；切到 rollout 时以当前帧 GT state 初始化，之后按预测 state 递归，跳帧会从该帧 GT state 重新 handoff；切回 teacherforced 时自动回到 GT state。
- 新增 `Hand display`：`GT` / `pred` / `both`；新增 `Point size (m)` 滑块（`0.001–0.020 m`）。当前仍只渲染点云，不加载 mesh。
- [teacherforce.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)、[run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/run_manifest.json)。Viser 地址：`http://localhost:8097`。

**验证**

- 8-frame Viser smoke 和无服务器 rollout handoff smoke 均通过；完整 validation sequence `s1/banana_lift` 574 帧预计算完成。
- `python3 -m py_compile src/task/CmDecoderv2/visualize_teacherforce.py`、`git diff --check` 和 Viser HTTP 探活通过。
- 正式 decoder 训练 GPU0/1/2 未停止；本次可视化只使用 GPU3。

## 2026-09-06 15:58:41 +0800 — V1.1.1 RL-Inspire validation teacher-forcing Viser 可视化

- activity_id: `ACT-20260906-161100-CMDECODERV2-TEACHERFORCE-VISER`
- timestamp: `2026-09-06 15:58:41 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户要求先不看 closed-loop rollout，改用 teacher forcing 检查 decoder 表现并使用 Viser 可视化；默认使用 RL-Inspire validation，因为 MANO test 没有对应 Inspire GT。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、rollout 输出和用户改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-155735`
- run_status: `RUNNING`（Viser server 仍在 GPU3 / port 8097 提供交互式查看）
- conclusion: `INCONCLUSIVE`（teacher-forced wiring 和定性/诊断指标已生成；不等同于最终科研结论）
- scope: `src/task/CmDecoderv2/`（新增 `visualize_teacherforce.py`）与独立 validation teacher-forcing 输出目录；不修改训练变量、checkpoint 或 MANO rollout。

**原因**

需要隔离 closed-loop state feedback drift：每一帧使用 RL-Inspire validation 的真实当前 q/wrist 和真实 Cm window，只预测 h=1，不把预测状态反馈到下一帧；Viser 同时显示 GT 当前手、GT 下一帧手和 decoder 预测。

**实现与运行**

- 新增 [visualize_teacherforce.py](../../visualize_teacherforce.py)，只接受 `variant=inspire_rl` 的 validation entry，使用 frozen OICM + CmDecoderv2 checkpoint，输出 teacher-forced h=1 轨迹并启动 Viser。
- 使用 GPU3（进程内 `cuda:0`），序列 `s1/banana_lift`，574 个完整 window/frame；Viser 地址为 `http://localhost:8097`。
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-155735/run_manifest.json)、[teacherforce.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-155735/teacherforce.npz)。

**诊断指标**

- 全部帧：q MAE `1.367°`，wrist translation `13.98 mm`，wrist rotation `3.38°`，hand nearest-neighbor EPE `8.40 mm`。
- 当前 Cm 有效帧（379/574，`66.0%`）：q MAE `1.289°`，wrist translation `5.15 mm`，wrist rotation `2.47°`，hand EPE `5.50 mm`。
- 当前 Cm 无效帧：wrist translation `31.14 mm`，hand EPE `14.03 mm`；该差异是 teacher-forcing 诊断中必须单独报告的 mask 分层结果。

**验证**

- 8-frame smoke 已完成；checkpoint load、GT state/window 对齐、forward 和 Viser HTTP 均正常。
- `python3 -m py_compile src/task/CmDecoderv2/visualize_teacherforce.py`、`git diff --check` 通过。
- 正式 decoder 训练 GPU0/1/2 未停止；本次可视化仅使用 GPU3。当前 run 仍保持 RUNNING，未覆盖历史 rollout 或 checkpoint。

## 2026-09-06 15:50:00 +0800 — V1.1.1 核对 5 cm 帧屏蔽与 decoder loss mask

- activity_id: `ACT-20260906-155000-CMDECODERV2-5CM-MASK-DIAGNOSTIC`
- timestamp: `2026-09-06 15:50:00 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问训练是否按 5 cm 屏蔽帧；本次只读核对数据集 active mask、OICM 交互半径和 CmDecoderv2 loss，不修改代码、配置或运行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、CmDecoderv2 实现和用户改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（已确认 5 cm 机制的作用层级；当前 decoder 未按该 mask 重新训练）
- scope: `src/task/CmDecoderv2/` 与其引用的 ObjectInteractionCm 配置/代码；未改变研究变量

**原因**

需要区分 OICM 提取 Cm 时的 5 cm 几何筛选、OICM loss 的 sample mask，以及 CmDecoderv2 自身是否按 5 cm 有效帧筛选和屏蔽 decoder loss。

**核对结果**

- ObjectInteractionCm 配置的 `interaction_radius_m=0.05`、`active_only=true`：数据集会根据预计算的 `obj_candidate_mask_5cm` 丢弃当前帧完全没有 5 cm 候选的行；运行时局部交互边也要求距离不超过 5 cm，`sample_valid` 再表示采样到的 object pool 是否仍有有效交互。OICM runner 会用 `sample_valid` 屏蔽自身的 object/hand flow loss。
- CmDecoderv2 配置的 `data.active_only=false`：decoder window 不会因 5 cm candidate mask 在数据集层被丢弃。
- CmDecoderv2 runner 只记录 `cm/valid_frame_ratio`，q、wrist translation、wrist rotation 三项 loss 没有用 `cm_sample_valid` 做 mask。因此当前 decoder 训练并不是“按 5 cm 有效帧训练”；无效 Cm 仍进入 loss。

**证据入口**

- [CmDecoderv2 config](../../config.py)、[CmDecoderv2 dataset](../../dataset.py)、[CmDecoderv2 runner](../../runner.py)
- [ObjectInteractionCm RL config](../../../../../src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml)、[ObjectInteractionCm dataset](../../../../../src/task/ObjectInteractionCm/dataset.py)、[ObjectInteractionCm runner](../../../../../src/task/ObjectInteractionCm/runner.py)
- [formal training metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)

**验证**

- 只读检查配置、数据集行筛选、5 cm 交互半径、`sample_valid` 传播和 decoder loss 计算；未停止 GPU0/1/2 正式训练。

## 2026-09-06 15:33:31 +0800 — V1.1.1 完整 rollout 的“乱动”原因诊断

- activity_id: `ACT-20260906-153331-CMDECODERV2-ROLLOUT-DIAGNOSTIC`
- timestamp: `2026-09-06 15:33:31 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求分析完整 rollout 中 Inspire 手持续乱动的原因；本次仅做只读轨迹、代码、坐标和训练指标核对，不修改训练、checkpoint、数据或可视化代码。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、CmDecoderv2 实现和用户改动）
- run_id: `cmdecoderv2-mano-viz-20260906-151601`；关联正式训练 run：`cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `COMPLETED`（诊断）；正式训练保持运行状态，未被本次检查中断
- conclusion: `INCONCLUSIVE`（已定位实现与坐标问题，但尚未按修复方案重新训练/复测，不能据此下科研效果结论）
- scope: `src/task/CmDecoderv2/` 与独立 rollout/训练产物；未修改研究变量

**原因**

用户反馈完整 rollout 中 Inspire 手持续乱动，需要区分模型本身、无效 Cm 处理、初始化坐标和视频完整性四类因素；本次只做证据核对，不改变正式训练运行。

**关键证据**

- 可视化 rollout 共 267 steps，MP4 完整；`cm_sample_valid` 为 `[267,4]`，仅 504/1068 个 window-frame 位置有效。当前帧在约 0–90、220–266 均无效，但 visualizer 仍无条件执行 `finger += q_delta`、`wrist = wrist @ delta`，因此 dummy Cm 产生的非零预测会持续累积漂移。前 0–90 帧的每步 q 增量中位数约 `0.00965 rad`，腕部平移中位数约 `5.61 mm`；到第 219 帧 q 范数约 `1.548`、腕部相对初始位移约 `0.944 m`。
- CmDecoderv2 runner 当前只记录 `cm/valid_frame_ratio`，没有用 `cm_sample_valid` mask q/translation/rotation loss。正式训练已完成 epoch 的 valid-frame ratio 约 `0.51816`（train）/`0.52366`（val），约 48% 的 frame supervision 来自无效/dummy Cm；因此当前 best checkpoint 不能视作已按有效交互帧严格训练的最终模型。
- 本次 `mano_wrist` 初始化直接读取 parent GRAB cache 的 `hand_root_pose_world.npy`，但 decoder 测试 geometry view 的 object pose 与 parent cache 的 object pose 存在共同平移偏差，frame 0 约为 `[-39.7, -583.3, +17.6] mm`（旋转基本一致）。直接把 parent-world MANO wrist 当作 view-world Inspire base 会造成初始空间错位；应通过 object pose 关系先变换到 view world。
- 完整 MP4 [rollout](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/rollout.mp4) 已由 `ffprobe` 验证为 267 帧、30 FPS、8.9 秒；此前提前结束的版本是显式 `--max-steps 120`，不是当前编码或接触阶段截断。

**证据入口**

- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/trajectory.npz)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/run_manifest.json)
- [visualize.py](../../visualize.py)、[runner.py](../../runner.py)、[model.py](../../model.py)
- [formal training metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)

**建议的修复顺序（本次未执行）**

1. 用 `T_view_obj @ inv(T_parent_obj) @ T_mano_wrist` 把 MANO wrist 初始化转换到 decoder view world，并确认 MANO root 与 Inspire `hand_base_link` 的固定坐标约定。
2. rollout 只在当前帧 `cm_sample_valid[0,0]` 为真时应用 h=1 action；无效帧明确保持状态，不能用 window 内任意未来有效帧替代当前帧判断。
3. 在 CmDecoderv2 训练中对无效当前帧 mask/drop q、translation、rotation supervision，再重新训练并复测；当前正式 run 和 best.pt 仅作为诊断基线保留。

**验证**

- 只读检查 `trajectory.npz` 的有效帧掩码、状态累积和模型输出范数；对 `visualize.py`、`runner.py`、数据 schema 和 object pose 做源码/数值对齐。
- 本条记录写入后运行 `audit_diff.py --check-links` 与 `git diff --check`；本次未停止正式 GPU0/1/2 训练，也未覆盖旧输出。

## 2026-09-06 15:19:06 +0800 — V1.1.1 修正 MANO wrist 初始化并完成完整 sequence 可视化

- activity_id: `ACT-20260906-151000-CMDECODERV2-MANO-VIZ-MANO-WRIST`
- timestamp: `2026-09-06 15:19:06 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户确认“用 MANO wrist 初始化、Inspire 手指 q 保持 0”，并要求完整跑完接触过程；本次不改变训练变量。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、旧可视化和用户改动）
- run_id: `cmdecoderv2-mano-viz-20260906-151601`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（仅为 MANO-only qualitative rollout；无 paired Inspire test GT，不构成定量或科研效果结论）
- scope: `src/task/CmDecoderv2/visualize.py` 与独立 `outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/` 产物；正式训练 GPU0/1/2 未停止
- checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- checkpoint_sha256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`

**文件**

- `src/task/CmDecoderv2/` — 本 Task 现有实现、配置、数据工具、测试和文档均保留；本次实际代码改动集中在 `visualize.py`，其余路径仅由 activity scope 归档。
- [visualize.py](../../visualize.py) — 默认新增 `mano_wrist` 初始化模式；从 MANO parent cache 读取右手 `hand_root_pose_world.npy`，初始化 Inspire wrist；保留 `manual` 模式和 q=0 手指初始化；MP4 改为 ffmpeg 编码并校验文件大小。
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/run_manifest.json)、[MP4](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/rollout.mp4)、[trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/trajectory.npz)、[PNG frames](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/frames/)

**原因**

旧 rollout 显式限制为 `120` steps，而 `s1/alarmclock_offhand_1` 的完整可用长度为 `frame_count-K=267`，因此视频在接触阶段提前结束；旧默认 wrist 也没有对齐 MANO source。按用户确认，新的 rollout 使用第 0 帧 MANO wrist 世界位姿，手指 q 保持全 0，并跑完整 267 steps。

**验证**

- 命令：`CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence-index 0 --max-steps 10000 --initial-state mano_wrist --render-every 1 --write-mp4 --output-root outputs/cmdecoderv2 --activity-id ACT-20260906-151000-CMDECODERV2-MANO-VIZ-MANO-WRIST`
- `py_compile src/task/CmDecoderv2/visualize.py`、`git diff --check`：通过。
- sequence `s1/alarmclock_offhand_1`：267 steps；MP4 经 `ffprobe` 验证为 H.264、267 帧、30 FPS、8.9 秒；checkpoint SHA 在运行前后保持一致。
- 旧正式 decoder 训练 torchrun 仍在 GPU0/1/2 运行；本次只使用 GPU3。
- 首帧、133 帧和末帧已抽查；画面仅作人工观察，MANO→Inspire 的定性质量仍需用户结合完整视频评估。

## 2026-09-06 15:06:22 +0800 — V1.1.1 使用当前 best.pt 完成 GPU3 MANO 定性 rollout

- activity_id: `ACT-20260906-150300-CMDECODERV2-MANO-VIZ`
- timestamp: `2026-09-06 15:06:22 +0800`
- modification_version: `V1.1.1`
- type: `experiment / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户要求“拿 GPU3 用目前的 best.pt 来可视化”；使用独立输出目录，不触碰 0/1/2 正式训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cmdecoderv2-mano-viz-20260906-150326`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（PNG/MP4 仅供人工定性观察，不含 paired Inspire test GT，不作定量或科研效果结论）
- scope: `src/task/CmDecoderv2/` 的可视化入口与独立 `outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/` 产物
- device: `CUDA_VISIBLE_DEVICES=3`（进程内 `cuda:0`）；正式 decoder 训练 GPU 0/1/2 保持运行
- checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- checkpoint_sha256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`

**原因**

在不停止正式三卡训练的情况下，用当前 decoder best checkpoint 对 MANO-only test sequence 做 receding-horizon `h=1` 定性 rollout，供人工检查 MANO source 与 Inspire predicted 的空间关系。

**命令与产物**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence-index 0 --max-steps 120 --render-every 1 --write-mp4 --output-root outputs/cmdecoderv2 --activity-id ACT-20260906-150300-CMDECODERV2-MANO-VIZ`
- sequence: `s1/alarmclock_offhand_1`；120 steps；初始 Inspire state 由脚本默认参数显式记录。
- [运行目录](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/run_manifest.json)
- [MP4 rollout](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/rollout.mp4)
- [PNG frames](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/frames/)
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/trajectory.npz)

**验证**

- 可视化进程在 GPU3 正常完成；manifest 记录 `paired_inspire_test_gt=false`，未读取配对 Inspire GT。
- 120 张 PNG 已生成；最初的 imageio MP4 写出因 `write() got an unexpected keyword argument 'fps'` 失败，随后使用 ffmpeg 将同一批 PNG 转换为有效 H.264 MP4（120 帧、30 FPS、4 秒），并通过 `ffprobe` 验证。
- `best.pt` 在可视化前后 SHA256 保持一致；正式训练进程仍在 GPU0/1/2 运行，未被中断。
- 首帧/中段/末帧已做肉眼抽查；这是人工观察入口，不把当前画面直接解释为 decoder 科研效果成立。

## 2026-09-06 15:01:07 +0800 — V1.1.1 训练中期 loss/validation 状态诊断

- activity_id: `ACT-20260906-150107-CMDECODERV2-LOSS-STATUS-DIAGNOSTIC`
- timestamp: `2026-09-06 15:01:07 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练状态；只读检查进程、metrics、validation 和 checkpoint，不中断或调整训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- last_step: `156100`（epoch 25 进行中）
- last_completed_epoch: `24`
- best_metric: `val/loss=1.3800124928725293`（epoch 13 / step 83213）
- conclusion: `INCONCLUSIVE`（train loss 仍下降，但 validation 在 epoch 13 后平台化并波动，训练尚未结束）
- scope: `src/task/CmDecoderv2/` 与正式 run 的只读 loss/validation/进程/checkpoint 诊断

**原因**

需要区分优化是否仍在下降与泛化 validation 是否继续改善，避免把持续下降的 train loss 误判为模型已收敛。

**验证**

- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)：已完成 epoch 1–24；train loss 从 `2.059890` 降到 `1.159078`，仍在下降。
- validation 最佳为 epoch 13 的 `1.380012`；epoch 14–24 在约 `1.3867–1.4349` 间波动，epoch 24 为 `1.399368`，尚未恢复 epoch 13 最佳。
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt) 为 epoch 13 / step 83213；[latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt) 为 epoch 24 / step 153624。
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)：torchrun 与 3 个 rank 仍在运行；GPU 0/1/2 利用率约 `68%/57%/71%`。
- 当前判断：训练过程没有异常退出，但 validation 已出现早期平台/波动；是否最终收敛需继续观察后续 epochs，当前科研结论仍为 `INCONCLUSIVE`。

## 2026-09-06 12:54:47 +0800 — V1.1.1 正式训练早期 loss 趋势诊断

- activity_id: `ACT-20260906-125447-CMDECODERV2-LOSS-TREND-DIAGNOSTIC`
- timestamp: `2026-09-06 12:54:47 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练状态及 loss 是否持续下降；只读检查进程、metrics 和 checkpoint，不中断或调整训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- last_step: `49300`（查询时 epoch 8 进行中）
- last_completed_epoch: `7`
- best_metric: `val/loss=1.424412033150659`（epoch 7 / step 44807）
- conclusion: `INCONCLUSIVE`（前 7 个 epoch 的 train/val loss 持续下降，但训练仅完成约 15%，尚不能判断最终收敛）
- scope: `src/task/CmDecoderv2/` 与正式 run 的只读 loss/进程/checkpoint 诊断

**原因**

step loss 受 batch 组成与扰动影响明显，需使用完整 epoch 的 train aggregate 和 validation 指标判断趋势，并拆分 q、wrist translation、wrist rotation 确认总 loss 的下降来源。

**验证**

- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)：epoch 1→7 的 train loss `2.059890→1.507454`，val loss `1.785547→1.424412`，两者在每个已完成 epoch 均下降。
- 总 loss 的主要下降来自 wrist translation：train `1.794571→1.249410`，val `1.623184→1.263725`；q 与 wrist rotation 分项目前主要呈平台/小幅改善。
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt) 与 [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt) 均为 epoch 7 / step 44807，`best_metric=1.424412033150659`。
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)：torchrun 与 3 个 rank 仍在运行；GPU 0/1/2 查询时利用率约 `69%/57%/69%`，无退出迹象。
- 当前 step loss 在约 `0.87–2.63` 间抖动是 batch-level 波动，不能据此要求逐 step 单调下降；科研结论需等待更多 validation 和最终 MANO→Inspire 可视化。

## 2026-09-06 11:57:01 +0800 — V1.1.1 使用冻结 OICM best.pt 启动 CmDecoderv2 三卡正式训练

- activity_id: `ACT-20260906-115701-CMDECODERV2-FORMAL-TRAIN-STARTED`
- timestamp: `2026-09-06 11:57:01 +0800`
- modification_version: `V1.1.1`
- type: `operation / experiment`
- change_level: `L3 + L2`（按已定稿 v1.1 plan 运行；锁定 OICM checkpoint gate 并启动长任务）
- approval: `user-approved`
- approval_basis: 用户明确要求“先停止训练吧，然后用 bestpt 直接开始训练 CmDecoderv2。照样用三卡”。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留旧 CmDecoder、ObjectInteractionCm 及既有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（正式训练已启动；终态前不作收敛或跨 embodiment 效果结论）
- scope: 固定 Dexplore RL decoder view、Temporal-D2 `K=4`、q/wrist 输出和正式 train/val；GPU 0/1/2，DDP，per-device batch 8，global batch 24，50 epochs（320050 total steps）。
- frozen_oicm_checkpoint: `outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt`
- frozen_oicm_checkpoint_sha256: `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`

**命令与产物（运行中）**

- command: `CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --distributed`
- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/)（`RUNNING`, PENDING）
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/config.json)（`RUNNING`, PENDING）
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/run_manifest.json)（`RUNNING`, PENDING）
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)（`RUNNING`, PENDING）
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)（`RUNNING`, PENDING）
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt)（`RUNNING`, PENDING）
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt)（`RUNNING`, PENDING）

**启动证据**

- launcher 与 3 个 rank 已正常启动；`train_setup` 报告 `device=cuda:0`、`world_size=3`、`global_batch=24`、`total_steps=320050`。
- 本条目仅记录运行启动；正式训练终态、last step/epoch、best metric 和实际 checkpoint 路径待运行结束或用户要求停止后补记。

**文件**

- `src/task/CmDecoderv2/` — 本 Task 的实现、配置、数据工具、测试和文档；本次只更新正式配置中的 OICM checkpoint SHA gate，未改动模型代码或数据 view。

**原因**

OICM 续训已按用户要求停止，必须把静态 decoder 配置绑定到已冻结的 `best.pt`，再使用 approved Temporal-D2、RL-only train/val 和三卡 DDP 进行正式训练，避免运行中动态追随 checkpoint。

**验证**

- `sha256sum outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt` 与配置 gate 一致：`fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。
- 启动命令已创建 run manifest/config/metadata/metrics/train.log；3 个 rank 正常运行，GPU 0/1/2 利用率已上升。
- 前 800 steps 已写入 metrics，loss 与 q/wrist 子损失均为 finite；这是工程运行证据，不代表 decoder 已收敛或 MANO→Inspire 定性效果成立。

## 2026-09-06 10:56:47 +0800 — V1.1.1 Temporal-D2 实现、完整 decoder view 与 smoke

- activity_id: `cmdecoderv2-v1.1.1-20260906-102433-implementation`
- timestamp: `2026-09-06 10:56:47 +0800`
- modification_version: `V1.1.1`
- type: `architecture / code / data / experiment / operation / documentation`
- change_level: `L3 + L2`
- approval: `user-approved`
- approval_basis: 用户确认 v1.1 默认方案、要求首版直接使用 Temporal-D2，并明确回复“可以，你直接开始实现吧”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（旧 CmDecoder、ObjectInteractionCm 和既有实验改动均保留且未纳入本 Task）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅工程实现、数据合同和 smoke；正式训练尚未开始，不构成收敛或跨 embodiment 效果结论）
- scope: 新建独立 CmDecoderv2；实现 Dexplore RL decoder view、`K=4` Temporal-D2、q/wrist loss、
  BaseRunner 训练入口与 MANO-only 定性 rollout。未修改 `src/base/`、旧 `src/task/CmDecoder/`、
  ObjectInteractionCm 代码/cache/checkpoint、原始 GRAB/Dexplore 数据、既有 split 或运行进程。

**设计与文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 新增 `CmDecoderv2: V1.1.1` Task 指针。
- [Task package](../../) 与 [config.py](../../config.py) — 建立独立 Python Task 包和默认静态合同；不从旧
  CmDecoder 导入兼容实现。
- [指导 v1.1](../指导/v1.1.md) 与 [最终计划 v1.1](../plan/v1.1.md) — 冻结研究目标、数据语义、Temporal-D2、
  q/wrist、扰动和分阶段闸门。
- [model.py](../../model.py) — 冻结并 strict-load OICM；把 `K x 16` Cm/anchor/time token set 作为 memory，
  当前 global/link/future-step 作为 query；不添加 slot-ID embedding，输出四步 q residual 与 wrist 相对 `SE(3)`。
- [kinematics.py](../../kinematics.py) — 固化 Dexplore native→URDF、6 个独立 q、mimic、`hand_base_link`
  wrist FK 和 18 个 object-frame link query。
- [dataset.py](../../dataset.py) 与 [decoder view builder](../../tools/data/build_dexplore_view.py) — 每个
  `K=4` Cm window 使用 `K+1=5` 个连续 30 Hz cache frame；RL train/val 提供 q/wrist GT，MANO test 不含 GT。
- [Task-local tools](../../tools/) — 包含独立 tools/data Python 包与 decoder view 入口。
- [runner.py](../../runner.py)、[train.py](../../train.py) 与 [正式配置](../../configs/active/dexplore_rl_v1_1_1.yaml) —
  实现 q Smooth-L1、translation x100 Smooth-L1、SO(3) geodesic、`0.8^(h-1)` horizon 权重和 per-horizon/identity 指标。
- [smoke 配置](../../configs/active/dexplore_rl_v1_1_1_smoke.yaml) 与 [Task tests](../../tests/) — smoke
  关闭 perturbation；正式配置使用已批准的 q/translation/rotation Gaussian perturbation。
- [visualize.py](../../visualize.py) — MANO source Cm 的 receding-horizon rollout 只执行 `h=1`，显式记录
  初始 Inspire state，导出 q/wrist/点云、PNG，并可选 MP4；不读取 paired Inspire test GT。
- [Task 文档入口](../README.md)、[架构记录](architecture_log.md)、[实验记录](experiment_log.md) 与
  [可迁移事实](repo_memory.md) — 记录新 Task 的事实和证据导航。
- [资产合同](../../assets/README.md) — Task-local Inspire 右手 URDF 由本机软链接提供，URDF SHA256 写入 view。

**原因**

旧 CmDecoder 的 HRDexDB/point-flow 兼容路径与本研究的 Dexplore RL q/wrist 语义不同。独立 Task 可以固定
MANO-source Cm → Inspire target 的数据边界，并用未来 Cm window + 当前 Inspire state 直接检验
Temporal-D2 与 receding-horizon 假设，同时避免把同一 parent 的 MANO/Inspire pair 泄露进 decoder 监督。

**数据运行**

- run_id: `cmdecoderv2-view-pilot-20260906-105616`
- run_status: `COMPLETED`
- command: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.tools.data.build_dexplore_view --mode pilot --output data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot --window-size 4 --resume --activity-id cmdecoderv2-v1.1.1-20260906-102433-implementation`
- evidence: [pilot view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/)、
  [manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/manifest.json)、
  [index.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/index.json)、
  [run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/run_manifest.json)。
- result: RL train/val 各 1 sequence、MANO-only test 1 sequence，供静态检查和 smoke。

- run_id: `cmdecoderv2-view-full-20260906-105602`
- run_status: `COMPLETED`
- command: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.tools.data.build_dexplore_view --mode full --output data/processed_data/cm_decoder_v2/dexplore_rl_v1_1 --window-size 4 --resume --activity-id cmdecoderv2-v1.1.1-20260906-102433-implementation`
- evidence: [完整 view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/)、
  [manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/manifest.json)、
  [index.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/index.json)、
  [run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/run_manifest.json)。
- result: RL train `503` sequence / `153602` window，RL val `62` / `19730` window，MANO-only test
  `125` sequence；565 个 RL sidecar 全部存在且 shape 正确，test entry 中 Inspire GT 字段计数为 `0`。

**训练 smoke**

- final run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619`
- run_status: `COMPLETED`
- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1_smoke.yaml`
- last_step: `2`; last_epoch: `1`; best_metric: `val/loss=1.4703673253`
- evidence: [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/)、
  [config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/config.json)、
  [run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/metrics.jsonl)、
  [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/train.log)、
  [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/best.pt)、
  [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/latest.pt)。
- result: 两步 optimizer 与完整 pilot val 完成；真实 Cm shape 为 `[B,4,16,32]`，OICM 保持
  `eval/no-grad`，decoder 可反向；metadata 记录 OICM SHA256
  `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。

实现过程中另产生四个终态为 `COMPLETED` 的预备 smoke，分别用于发现 provenance 缺项和验证修正；不作为
最终证据，但保留可审计产物：

- `..._102937`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/checkpoints/best.pt)。
- `..._104051`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/checkpoints/best.pt)。
- `..._104215`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/checkpoints/best.pt)。
- `..._104811`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/checkpoints/best.pt)。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile $(find src/task/CmDecoderv2 -name '*.py' -type f | sort)`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`；覆盖
  mimic/FK、`K+1` window、输出 shape、slot permutation invariance 与 identity rotation finite gradient。
- 单卡真实 checkpoint forward/backward：`pred_q_delta [1,4,6]`、translation/rotvec `[1,4,3]`；
  `oicm_training=False`、`oicm_grad_count=0`，有效 contact window 的四帧 `cm_sample_valid=True`。
- 完整 view 审计：所有 RL window arrays 为 `[T-4,5]`，所有 sidecar 链接存在，MANO test 不含 q/wrist GT。
- 正式数据加载：503 个 train sequence / 153602 windows 可构造；perturbation 同 epoch 可复现、跨 epoch
  变化，浮点张量全部 finite。batch=8（OICM flatten batch=32）单卡 backward 峰值 allocated `1.39 GiB`、
  reserved `2.57 GiB`。
- 正式 checkpoint 闸门负向测试：`LOCK_AFTER_OICM_TERMINAL` 未替换时按预期拒绝建模，避免误用动态 best。
- `git diff --check`：通过；`audit_diff.py --worktree --scope-prefix src/task/CmDecoderv2
  --scope-prefix docs/current_versions.yaml --check-links`：通过，覆盖 `24` 个变更路径和 `60` 个本地链接。

**未启动与回滚**

- 正式 CmDecoderv2 训练未启动：OICM 续训仍在 0/1/2 卡运行，当前 `best.pt` 只能作为 smoke 输入；必须等
  OICM 终态后冻结 checkpoint SHA256，再进入 Gate C。MANO rollout 代码已实现但遵循计划，需在正式 decoder
  训练后才执行 Gate D。
- 回滚入口为隔离/移走本 Task 目录、`data/processed_data/cm_decoder_v2/` 新 view 和
  `outputs/cmdecoderv2/` 新 smoke；无需也不得改动或删除旧 CmDecoder、OICM cache/checkpoint、原始数据或运行。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。Task 级测试不在根 `pytest.ini` 的默认
  `testpaths` 中，因此按目录规范显式传入 `src/task/CmDecoderv2/tests`；无需改变共享 pytest 配置。
