# Inspire 的 TopoRetarget 接入及几何方案对比

按 [V1.4 final plan §8](../../docs/plan/V1.4.md) 进行的小样本适配。只读加载外部
TopoRetarget-Repro 和 Dexplore；所有定义位于此 research 目录，派生 URDF/config 位于
Task assets，原始 NAS 数据、正式 cache、val/test 和训练配置保持不变。

## 接入范围

- `inspire_adapter.py`：将同一套 Inspire 手指运动学交给通用 Topo `RobotHandModel`，
  显式拆出 wrist SE(3)，导出带名称的 Dexplore 18D 顺序；资产 hash 和修改记入 manifest。
- `prepare.py`：在 graspenv 重建 GRAB MANO/对象，与官方 PCA/ObjectModel 核对，
  将同一输入交给 Dexplore 风格 position solver；输出 source 和 baseline NPZ。
- `run.py`：独立 FK/Jacobian 校验、Topo 原 warm-start/interaction graph/final refinement API，
  使用固定 collision 样本和独立 reference SDF 统计两方法。未改外部求解器或论文权重。

目标是验证 Inspire 是否可接入并形成同目标小样本证据；不能由几帧 smoke 宣称全数据集优胜。
当前输出是研究 NPZ，不是正式 OI-Cm cache；完整上游 CLI workflow 的 Arti-MANO 特化仍保留。

## 明确的适配假设

1. Dexplore 当前右手 converter 忽略 mimic，故两算法共用 **12 独立指关节 + 6 wrist**。
   保存 18 个数不表示真实硬件具有 18 个独立执行器。
2. Topo 内部为 12 指关节及单独 base；不把 6 虚拟 wrist 再放进 finger qpos。
3. 四指缺少独立 DIP，以 PIP-tip 中点作为固定几何锚点。它不是 MANO GT 关节。
4. Dexplore 自带 left URDF 无标准 tips/wrist，不能直接套 right 映射。当前候选 left
   使用明确指定的官方 dex-retargeting 左手；按侧分别报告，同侧两算法共用资产。
5. GRAB 保留原始 native world、原 frame ID 和 120 Hz 短窗口。`fullpose` 使用
   `flat_hand_mean=True`；对象 row-vector `v @ R` 转成 column-vector SE(3) 时转置 R。
6. 默认 refinement 为仓库已有 `scipy_slsqp_active_set_contact_rich_v3_fixed`。
   失败帧继续记录用于诊断，只有 `accepted` 为真的帧算上游接受；不会静默混入训练。
   无穿透不等于接触保持。碰撞统计只覆盖 URDF collision 样本，不能代表完整连续表面无穿透。

## 运行

先用 `prepare.py --help` 查看 GRAB/model/资产路径参数；`--sides right` 使用 Dexplore
右手，`--sides left --left-assets <dex-retargeting>/assets/robots/hands/inspire_hand`
使用官方左手。`--assets` 与 `--output` 必须是新的目录。源重建使用 graspenv。

Topo 要求 Python 3.10+；本机使用 Task assets 下的隔离 venv，复用 wbcd 的 Torch，
仅追加 trimesh/zarr 等依赖，graspenv/PyTorch/CUDA 均未升级。

```bash
<topo-python> run.py --topo-root <TopoRetarget-Repro> \
  --assets <prepared-assets> --input <prepare-output>/source_and_dex.npz \
  --output output/<new-run-id>
```

`--foundation-only` 用于只验证目标模型与 warm-start。真实调用和输入 hash 以各 run 的
`config.json`、`run_manifest.json` 为准；本机运行与终态见 [activity](../../docs/logs/activity_log.md)。

回滚仅移除本研究新增文件及 Task-local 派生资产/隔离环境，不改外部仓库。
