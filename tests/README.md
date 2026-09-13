# 测试目录与迁移清单

本文件定义测试的未来归属，并记录当前根目录测试的 legacy 分类。当前阶段不移动、不删除、不改写
根目录下已有的 `test_*.py`；它们仍由现有 `pytest.ini` 收集。新测试不得继续直接新增到根 `tests/`。

## 一、未来目录

| 归属 | 目录 | 适用范围 |
| --- | --- | --- |
| Task 专属 | `src/task/<Task>/tests/` | 只依赖一个 Task 的模型、Dataset、Runner、局部工具和 schema |
| 数据处理 | `process/tests/` | `process/` 下数据集 adapter、Stage 处理和跨 Task 数据格式 |
| 共享基础设施 | `tests/shared/` | `src/base/`、公共工具、配置、Runner、Manifest 和性能合同 |
| 跨 Task 集成 | `tests/integration/` | 两个或以上 Task、Task 与共享层、checkpoint 兼容或端到端边界 |
| 治理 | `tests/governance/` | AGENTS、目录规范、activity、Markdown 链接和版本合同 |

测试所有权按被验证的合同决定，不按“主要 import 了哪个模块”决定。一个测试同时验证多个 Task
或跨越 `src/base` 与 Task 时，放入 `tests/integration/`；不要在多个目录复制同一测试。

## 二、执行规则

- Task 内代码或配置修改：运行对应 Task 的测试；若触及共享接口，再加 `tests/shared/` 中直接相关的测试。
- `src/base/`、公共配置、Manifest 或 Runner 修改：运行 `tests/shared/`，并运行受影响 Task 的最小
  smoke；只有公共合同大范围变化时才做全量回归。
- 跨 Task、数据格式或 checkpoint 兼容修改：运行 `tests/integration/` 及所有受影响 Task 测试。
- 处理流水线修改：运行 `process/tests/` 以及消费该产物的 Task 定向测试。
- 治理文档或 Skill 修改：运行 `tests/governance/` 和 activity 链接审计。
- 根目录 legacy 测试在迁移完成前不作为新测试的放置位置；需要完整回归时仍可运行 `pytest`。

当前阶段不修改 `pytest.ini` 或 `tests/conftest.py`。未来开始物理迁移时，再统一处理测试发现路径、
NumPy 兼容 fixture 和 CI 收集范围，避免迁移一半时出现环境差异。

## 三、当前根目录 legacy 分类

下表只描述未来目标，不改变文件当前路径。文件名均相对于仓库根 `tests/`。

### Task：Cm → `src/task/Cm/tests/`

`test_cm_dominant_hand.py`、`test_cm_flow_scale.py`、`test_cm_hrdexdb.py`、`test_cm_object_v2.py`、
`test_cm_scene.py`、`test_cm_sequence_dataset.py`、`test_cm_slot_attention.py`、`test_cm_structure.py`、
`test_cm_surface_sampling.py`

### Task：CmDecoder → `src/task/CmDecoder/tests/`

`test_cmdecoder_build_cache.py`、`test_cmdecoder_q_optimizer.py`、`test_cmdecoder_viewer.py`

### Task：ObjectInteractionCm → `src/task/ObjectInteractionCm/tests/`

`test_object_interaction_cm.py`

### Task：Actiontoken → `src/task/Actiontoken/tests/`

`test_action_token.py`

### Task：Posetoken → `src/task/Posetoken/tests/`

`test_pose_token_patch_identity.py`

### Task：InteractionDynamics → `src/task/InteractionDynamics/tests/`

`test_goal_interaction_diffusion.py`、`test_interaction_autoencoder.py`、`test_interaction_compression.py`、
`test_interaction_diffusion.py`、`test_interaction_dynamics_action_probe.py`、
`test_interaction_dynamics_correspondence.py`、`test_interaction_dynamics_dataset.py`、
`test_interaction_dynamics_frames.py`、`test_interaction_dynamics_inverse_mano.py`、
`test_interaction_dynamics_model.py`、`test_interaction_dynamics_no_leakage.py`、
`test_interaction_dynamics_runner.py`、`test_interaction_dynamics_v17.py`、
`test_interaction_dynamics_v18.py`、`test_interaction_dynamics_v18_4.py`、
`test_interaction_dynamics_v20.py`、`test_interaction_dynamics_v20_2.py`、
`test_interaction_dynamics_v20_3.py`、`test_interaction_dynamics_v20_4.py`、
`test_interaction_dynamics_v20_5.py`、`test_interaction_dynamics_v20_7.py`、
`test_interaction_dynamics_viewer.py`、`test_interaction_dynamics_viewer_v2.py`、`test_interaction_field.py`、
`test_pretrained_action_adapter_gradient.py`、`test_residual_interaction_diffusion.py`、
`test_residual_interaction_regression.py`、`test_state_interaction_diffusion.py`

### Task：correspondence_ptv3_v2 → `src/task/correspondence_ptv3_v2/tests/`

`test_correspondence_ptv3_v2.py`、`test_correspondence_ptv3_v2_visualize.py`、
`test_hand_noise_profiles.py`、`test_hand_pca_perturbation.py`

### 数据处理 → `process/tests/`

`test_dexycb_raw.py`、`test_hand_root_frame.py`、`test_stage3_mano_reconstruction.py`

### 共享基础设施 → `tests/shared/`

`test_base_runner_max_steps.py`、`test_base_runner_validation.py`、`test_component_registry.py`、
`test_cosine_restart_scheduler.py`、`test_mano_geometry_noise_calibration.py`、
`test_performance_monitor.py`、`test_run_manifest.py`

### 跨 Task 集成 → `tests/integration/`

`test_checkpoint_compat.py`、`test_framework_contracts.py`、`test_metric_stat.py`、
`test_overfit_diagnosis.py`、`test_pose_token_root_invariance.py`

### 治理 → `tests/governance/`

`test_audit_diff.py`

## 四、迁移边界

未来物理迁移某个文件时，应先确认它不再被根目录路径、CI、外部脚本或用户命令依赖；同步更新
`pytest.ini`、相关 import 路径和 activity。迁移本身是 documentation/governance 变更，不代表
测试所覆盖的科研结论发生变化。
