# ARCTIC 手-物体穿模检测阶段性总结

本文基于当前本地检测脚本 `.codex/local_tools/check_arctic_penetration.py`，以及已经生成的 3 组检测结果：

- `outputs/quality_preview/s01_box_grab_01_penetration_summary.json`
- `outputs/quality_preview/s01_capsulemachine_use_01_penetration_phase3_summary.json`
- `outputs/quality_preview/s01_capsulemachine_use_01_penetration_stride1_summary.json`

## 1. 当前脚本的检测方式

当前脚本是本地 ARCTIC 数据质量审计工具，不是官方 ARCTIC 功能。它从 MeshCat world-verts cache 中读取：

- `verts_right`
- `verts_left`
- `verts_object`

然后根据物体顶点数，在 `data/arctic_data/data/meta/object_vtemplates/<object_name>/*.obj` 中寻找匹配的 object mesh，优先选择 `mesh.obj`。

脚本对左右手分别构造查询点，支持：

- `vertices`：只检测 MANO 顶点。
- `face_centers`：只检测 MANO 三角面中心。
- `vertices_and_face_centers`：同时检测 MANO 顶点和三角面中心。

当前主要使用：

- `backend = hybrid`
- `surface_mode = vertices_and_face_centers`
- `eps_mm = 1.0`

判定逻辑是：如果手部查询点位于物体内部，且穿入深度超过 1 mm，则记为穿模点。每帧输出左右手的 inside count、inside ratio、mean/p95/max depth，并在 summary 中汇总总点比例、触发帧比例、最大穿入深度和最坏帧。

脚本的 heuristic quality flag 规则为：

- `OK`：最大深度 < 2 mm，最大 inside ratio < 0.5%，任意穿模帧比例 < 5%。
- `MILD`：最大深度 < 5 mm，最大 inside ratio < 2%，任意穿模帧比例 < 20%。
- 其余为 `SEVERE`。

如果检测可靠性存在问题，例如物体 mesh 非 watertight、inside sign 自动判断不确定，最终 `quality_flag` 会降级为 `CHECK`，同时保留 `heuristic_quality_flag`。

## 2. 当前脚本和数据的关键限制

### 2.1 非 watertight mesh 会导致 inside/outside 结论只能作为 heuristic

`signed_distance` 和 `contains` 这类 inside/outside 检测通常要求物体 mesh 是封闭、法向一致的 watertight mesh。当前结果中：

- `capsulemachine/mesh.obj` 不是 watertight。
- `capsulemachine` 的所有 `phase3` 帧都使用了 hybrid fallback。
- 因此 capsulemachine 的 SDF/inside-outside 结果不能作为严格几何结论，只能作为强提示。

这也是后续需要做 object mesh topology audit，以及不依赖 watertight 的 triangle-triangle surface collision 检测的核心原因。

### 2.2 box 虽然 watertight，但 inside sign 自动检查不确定

`box/mesh.obj` 的 summary 显示：

- `object_is_watertight = true`
- `object_euler_number = -28`
- warning: `inside-sign auto check was inconclusive`

说明 box 的 mesh 可以做封闭体检测，但其拓扑并不简单，且脚本未能用 bounding-box center / outside probe 明确验证 signed distance 的内外符号，只能默认 trimesh 的 inside-positive 约定。因此 box 结果比 capsulemachine 可靠，但仍被标为 `reliability = heuristic`。

### 2.3 当前检测是点采样，不是连续表面碰撞

当前脚本检测 MANO 顶点和/或面中心是否进入物体内部。它不是 triangle-triangle surface collision：

- 可能漏掉两个三角面在采样点之间发生的相交。
- 对开放 mesh 或复杂拓扑的 inside/outside 判断可能产生误差。
- inside ratio 反映的是采样点比例，不等价于真实接触面积比例。

因此当前脚本适合做数据质量筛查和定位可疑帧，不适合作为最终几何证明。

### 2.4 检测结果依赖当前选中的 object mesh

脚本按 cache 中 object 顶点数选择模板 mesh，并优先使用 `mesh.obj`。但 object template 目录下存在多个候选 mesh，分辨率和拓扑可能不同。

示例：

- box:
  - `mesh.obj`: 3947 vertices / 7950 faces
  - `mesh_tex.obj`: 108226 vertices / 200000 faces
  - `top.obj`, `bottom.obj` 也存在
- capsulemachine:
  - `mesh.obj`: 3982 vertices / 7951 faces
  - `mesh_tex.obj`: 60258 vertices / 114087 faces
  - `top.obj`, `bottom.obj` 也存在

因此当前结论严格来说是针对 cache 所匹配的 `mesh.obj`，不是对所有 object candidate mesh 的完整拓扑结论。

### 2.5 box 使用 stride=20，只覆盖采样帧

`s01_box_grab_01` 当前只检测了 732 帧中的 37 帧，stride 为 20。因此：

- 检测到的高比例穿模已经有意义。
- 但帧比例是“采样帧比例”，不能直接等同于全序列逐帧比例。

## 3. 数据集检测结论

### 3.1 `s01_box_grab_01`

检测配置：

- object: `box`
- object mesh: `data/arctic_data/data/meta/object_vtemplates/box/mesh.obj`
- backend: `hybrid`
- surface mode: `vertices_and_face_centers`
- stride: 20
- eps: 1 mm
- total frames: 732
- checked frames: 37

mesh 状态：

- watertight: true
- euler number: -28
- reliability: heuristic
- warning: inside sign 自动检查不确定

核心结果：

- `quality_flag = CHECK`
- `heuristic_quality_flag = SEVERE`
- 任意手穿模帧：29 / 37，占 78.38%
- 右手穿模帧：23 / 37，占 62.16%
- 左手穿模帧：18 / 37，占 48.65%
- 全局最大穿入深度：4.40 mm
- 最大单帧 inside ratio：3.76%
- 右手最坏深度帧：380
- 左手最坏深度帧：320

按采样帧看，穿模主要出现在：

- 60-320
- 360-400
- 440-660

结论：

`s01_box_grab_01` 在抽样帧中存在明显手-物体穿模信号。虽然最大深度未超过 5 mm，但触发帧比例很高，最大 inside ratio 超过 MILD 阈值，因此脚本判为 `SEVERE`。考虑到 box mesh 是 watertight，该序列很可能存在真实的手-物体几何重叠；但由于本次只做 stride=20 抽样，建议后续用 stride=1 复查并结合可视化或 triangle-triangle 检测确认。

### 3.2 `s01_capsulemachine_use_01` phase3

检测配置：

- object: `capsulemachine`
- object mesh: `data/arctic_data/data/meta/object_vtemplates/capsulemachine/mesh.obj`
- backend: `hybrid`
- surface mode: `vertices_and_face_centers`
- stride: 1
- eps: 1 mm
- total frames: 783
- checked frames: 783

mesh 状态：

- watertight: false
- euler number: 5
- reliability: heuristic
- fallback frames: 783
- warning: object mesh is not watertight; hybrid fallback uses union of signed_distance and contains

核心结果：

- `quality_flag = CHECK`
- `heuristic_quality_flag = SEVERE`
- 任意手穿模帧：653 / 783，占 83.40%
- 右手穿模帧：244 / 783，占 31.16%
- 左手穿模帧：550 / 783，占 70.24%
- 全局最大穿入深度：7.78 mm
- 最大单帧 inside ratio：9.72%
- 右手最坏深度帧：619
- 左手最坏深度帧：331

穿模帧分布：

- 45-243
- 252-451
- 457-480
- 507-508
- 510-737

补充统计：

- inside ratio >= 2% 的帧：433 / 783
- inside ratio >= 5% 的帧：53 / 783
- inside ratio >= 8% 的帧：28 / 783
- 最大深度 >= 5 mm 的帧：264 / 783
- 仅右手触发：103 帧
- 仅左手触发：409 帧
- 左右手同时触发：141 帧

结论：

`s01_capsulemachine_use_01` 的穿模信号非常强，并且覆盖序列大部分时间。左手触发帧更多，右手在 380-390 附近出现最高 inside ratio，左手在 frame 331 达到最大深度 7.78 mm。不过由于 `capsulemachine/mesh.obj` 不是 watertight，当前 inside/outside 结果只能作为 heuristic。该序列应被视为高优先级可疑数据，需要用非 watertight 依赖的 surface collision 检测进一步确认。

### 3.3 `s01_capsulemachine_use_01` stride1 旧版结果

检测配置和 phase3 的主要差异：

- surface mode: `vertices`
- 每只手每帧查询点为 778 个，而 phase3 为 2316 个。
- summary 中未包含 phase3 新增的 hybrid backend、frame ratio、fallback frame 等完整字段。

核心结果：

- checked frames: 783
- suspicious frames: 650
- 最大右手深度：5.60 mm
- 最大左手深度：7.78 mm
- 最大右手 inside ratio：10.03%
- 最大左手 inside ratio：5.14%
- 右手最坏深度帧：445
- 左手最坏深度帧：331

穿模帧分布：

- 45-243
- 252-359
- 361-451
- 457-479
- 507-508
- 511-737

左右手分布：

- 仅右手触发：102 帧
- 仅左手触发：411 帧
- 左右手同时触发：137 帧

结论：

旧版 stride1 结果与 phase3 高度一致：suspicious frames 为 650，而 phase3 为 653；主要触发区间也基本相同。这说明 capsulemachine 的异常不是由 face center 采样或 phase3 hybrid 汇总单独引入的。虽然非 watertight mesh 限制了结论强度，但两个版本的一致性表明该序列确实存在稳定的大范围可疑手-物体重叠信号。

## 4. 总体判断

当前检测已经足以支持以下阶段性结论：

1. `s01_box_grab_01` 和 `s01_capsulemachine_use_01` 都存在明显手-物体穿模或相交风险。
2. `s01_capsulemachine_use_01` 的问题更严重，触发帧比例、最大穿入深度和高 inside ratio 帧数都更高。
3. `capsulemachine/mesh.obj` 非 watertight 是当前结论最大的可靠性限制；其结果应标记为 `CHECK / SEVERE / heuristic`，不能当作严格 SDF 结论。
4. `s01_box_grab_01` 的 mesh 是 watertight，但 inside sign 自动检查不确定，且只做了 stride=20 抽样；它的异常很可能真实存在，但仍需要 stride=1 或 surface collision 复核。
5. 当前脚本适合作为数据质量筛查器：能有效发现高风险序列、定位可疑帧、比较左右手问题分布。但最终确认需要 topology audit 和不依赖 watertight 的三角面碰撞检测。

## 5. 建议后续工作

优先级建议如下：

1. 对所有 `object_vtemplates/*/*.obj` 做 topology audit，输出 watertight、boundary edge、nonmanifold edge、degenerate face、duplicate face、connected components 等指标。
2. 实现 triangle-triangle surface collision 检测，避免依赖 object mesh 是否 watertight。
3. 对 `s01_capsulemachine_use_01` 先做小范围 surface collision 复核，例如 frame 331、380-390、619。
4. 对 `s01_box_grab_01` 运行 stride=1 复查，重点关注 frame 260、320、380、440 以及当前抽样触发区间。
5. 报告中继续区分三类结论：
   - `strict geometry confirmed`
   - `heuristic penetration signal`
   - `topology-limited / needs collision confirmation`
