# 在线 Cm 残差控制

入口为 vendor `CmResidualOnline` + `CmResidualPPO`，Task-local wrapper 复用 IsaacGymEnvs
任务注册、Hydra 配置、RLGPUEnv 和 rl_games Runner，仅补运行追溯、物理验收和 JSON 指标。
从 Ref2Dex 根目录执行。Isaac Gym 必须在 torch 之前导入，因此仿真 wrapper 使用脚本直启，
不能使用 `python -m src...`；后者会先执行项目包初始化。

```bash
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=.:third_party/IsaacGymEnvs
python src/task/CmDecoderv2/tools/rl/run_residual.py validate \
  --output outputs/cmdecoderv2/<new_gate_run> --num-envs 64
python src/task/CmDecoderv2/tools/rl/run_residual.py train \
  --output outputs/cmdecoderv2/<new_train_run> --num-envs 64 --iterations 100 \
  --gate outputs/cmdecoderv2/<new_gate_run>/gate.json
python src/task/CmDecoderv2/tools/rl/run_residual.py evaluate \
  --output outputs/cmdecoderv2/<new_eval_run> --num-envs 64 \
  --checkpoint outputs/cmdecoderv2/<train_run>/nn/<policy>.pth \
  --gate outputs/cmdecoderv2/<new_gate_run>/gate.json
```

使用已安装 Isaac Gym 的 `graspenv`。输出目录必须不存在；训练拒绝缺失或代码/SHA 不匹配的 gate。
示例 GPU 编号需按实际空闲资源选择，不停止其他进程。

## 数据与执行边界

- `prepare_online_source.py` 编码固定 paired MANO reference 的 Cm/anchor 窗口，输入 provenance
  在 source manifest 中记录。已知参考采用既有 paired view 的 object-pose-t 合同，不是独立 MANO 泛化测试。
- 在线每步读取仿真完整 native18 与实际 link/object 位姿，原冻结 Temporal-D2 只执行 h1。
  只允许 frame0 实际 Inspire 状态用于 reset；不提供未来 Inspire GT，不使用 teacher-forced action bank。
- 手指控制保持 q6，再按原比例展开目标；观测保留实际12个手指关节。DOF顺序按名称解析，
  不能将URDF XML顺序当作Gym顺序。腕部通过URDF六个腕关节做PD，不逐步移动actor root。
- residual action 为 q6 + wrist translation3/rotvec3，腕残差在base wrist局部右乘。
  真实airplane/table采用Dexplore资产和初始场景；来源在Task-local ignored资产链接与manifest中登记。
- 旧 `CmResidual`/`CmResidualDecoderBank` 配置仅保留历史，不自动升级或回退到离线bank；新任务拒绝旧mode。
- 原奖励和瞬时抬升8cm阈值只衡量单任务pilot，不代表持续抓持、泛化或Cm因果有效性。

计划与唯一终态见 [docs/plan/v1.1.md](../docs/plan/v1.1.md) 和
[docs/logs/activity_log.md](../docs/logs/activity_log.md)。
