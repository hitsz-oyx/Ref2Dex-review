# V20.7 GenDex GT contact-map baseline

该目录只验证一条静态、无网络的 round-trip：

```text
GT MANO stable grasp → GenDex aligned contact map → 官方 Allegro CMapAdam
```

不使用 CVAE、checkpoint、`sharp_lift` 或多帧轨迹优化。官方代码固定到 GenDexGrasp commit
`29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7`；外部代码和 `data.zip` 放在被忽略的
`output/research/InteractionDynamics/v20_7/vendor/GenDexGrasp`。
官方 32-particle 实现会显式展开完整 object-hand 笛卡尔积并在 24 GB GPU 上 OOM；本目录按
256 个 object points 分块计算完全相同的 energy。1-particle 对齐中 energy 最大误差为 0，
梯度最大误差为 `2.98e-8`。加载官方类时还会把新版 SciPy 已删除的 `as_dcm()` 名称替换为
完全等价的 `as_matrix()`；除此之外不改官方初始化或能量定义。

外部依赖准备（不提交 vendor/data）：

```bash
git clone https://github.com/tengyu-liu/GenDexGrasp.git \
  output/research/InteractionDynamics/v20_7/vendor/GenDexGrasp
cd output/research/InteractionDynamics/v20_7/vendor/GenDexGrasp
git checkout 29fe7efc558b8cc3ce2d8e67c9c22b014adc21c7
gdown --fuzzy 'https://drive.google.com/file/d/1WRV7m9AAfDOFE6Z9InIRJwhhlRUSQzCX/view' -O data.zip
unzip data.zip
conda run -n graspenv pip install urdf-parser-py transforms3d
conda run -n graspenv pip install -e thirdparty/pytorch_kinematics
```

构建单条 alarmclock 对比缓存：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n graspenv \
python -m src.task.InteractionDynamics.research.gendex_gt_baseline.build_compare_cache \
  --v20-cache output/research/InteractionDynamics/v20_6/viewer_cache/trajectory_0000.pt \
  --gendex-root output/research/InteractionDynamics/v20_7/vendor/GenDexGrasp \
  --output output/research/InteractionDynamics/v20_7/alarmclock.pt \
  --particles 32 --steps 100 --seed 0
```

查看 GT MANO、旧 GT-Y Allegro 和 GenDex Allegro：

```bash
conda run -n graspenv \
python -m src.task.InteractionDynamics.research.gendex_gt_baseline.viewer \
  --cache output/research/InteractionDynamics/v20_7/alarmclock.pt --port 8091
```
