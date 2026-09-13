# DExplore 两阶段训练的输入合同审计

在正式构建数据和三卡训练前，核验原始18-DOF轨迹与当前6维手指状态之间的差异。
不投影或覆盖GT，不训练策略，不改变旧split。参考 [src/task/CmDecoderv2/docs/plan/v1.1.md](../../docs/plan/v1.1.md)。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.CmDecoderv2.research.dexplore_contract_audit.run \
  --run-id <unique_run_id>
```

`--smoke`只取按名称排序的前2条，不能代替全量统计。每条轨迹的所有帧检查mimic约束；
默认等距8帧、10135个固定表面点检查FK。分开报告几何/实际、micro/macro、当前状态重建
和关节空间最小二乘。后者不是最优几何误差，也不是物理任务成功率。

`paired_manifest.json`仅供审计：630条保留现有split来源，其余标为`unassigned`，不得作为训练index。
`state_samples.npz`保存原始q、重建q、采样帧和FK误差，供`verify.py`独立复核。

Isaac Gym资产顺序检查必须使用独立文件入口，不能使用`-m`，因为本Task现有包初始化会提前导入torch。
该检查只加载URDF，不执行物理步，也不需要Hydra：

```bash
CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/CmDecoderv2/research/dexplore_contract_audit/asset_check.py \
  --urdf src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf \
  --output <run_dir>/gym_asset_check.json
```
