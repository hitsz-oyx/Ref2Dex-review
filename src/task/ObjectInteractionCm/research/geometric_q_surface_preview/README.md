# DExplore geometric q 高分辨率表面预览

本实验只读复用已经生成的 DExplore geometric `interaction_hand_inspire.pt`，从 tensor 的
`373:391` 读取 Inspire 18-DoF native q。它不会再次执行 MANO→Inspire 重定向，也不会读取同级
RL rollout。

固定 pilot 为 `s1/airplane_fly_1`：右手 MANO 在完整表面采样 2048 点，右手 Inspire 在 13 个
visual mesh 的完整表面采样 10135 点。两者均使用 `seed=2024`，跨帧保持相同 face/barycentric
correspondence；Inspire mesh-local 表面逐帧只执行一次 FK。原 GRAB MANO 和物体通过 object-local
坐标放入 geometric tensor 的 object pose，保证两种手在同一 world 下比较。

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python \
  -m src.task.ObjectInteractionCm.research.geometric_q_surface_preview.run \
  --output src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/<run_id>
```

输出包含 viewer-only `index.json`、两条 geometry、固定采样 correspondence、`diagnostics.json` 和
`run_manifest.json`。使用 `src.task.ObjectInteractionCm.visualize_grab` 打开该 index，可用累计距离
阈值将手点标红，并用 object-to-hand KNN 将命中的手点标黄。

该 pilot 不覆盖正式 cache，不接入 split/训练；工程检查通过不代表穿透、接触保真或训练收益成立。
