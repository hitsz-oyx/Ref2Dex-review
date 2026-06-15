# preprocess 工具说明

本目录存放 Ref2Dex 数据预处理阶段使用的网格流形性检查、筛选和修复脚本。当前脚本主要面向 OBJ 资产，部分检查脚本也支持 STL/PLY。

推荐处理顺序：

1. 用 `check_manifold.py` 扫描原始网格，先得到整体质量报告。
2. 用 `filter_manifold_objs.py` 将 OBJ 按流形/非流形分流。
3. 对非流形 OBJ 用 Blender 修复脚本处理。
4. 对修复输出再次运行 `check_manifold.py` 验证。

## 环境依赖

### 普通 Python 环境

`check_manifold.py` 和 `filter_manifold_objs.py` 需要普通 Python 环境。建议使用 Python 3.10 或 3.11：

```bash
conda create -n ref2dex-preprocess python=3.10 -y
conda activate ref2dex-preprocess

python -m pip install --upgrade pip
python -m pip install numpy tqdm open3d
```

如果需要从源码安装 PyTorch3D，建议先准备基础编译工具：

```bash
# conda 环境内安装构建工具
conda install -c conda-forge cmake ninja -y

# Linux 还需要系统编译器，例如 Ubuntu:
# sudo apt-get update && sudo apt-get install -y build-essential

# macOS 需要先安装 Xcode Command Line Tools:
# xcode-select --install
```

普通检查/筛选脚本会通过 `pytorch3d.io.load_obj/load_ply` 读取并三角化 OBJ/PLY；其中 `filter_manifold_objs.py` 只处理 OBJ，`check_manifold.py` 处理 OBJ/PLY/STL。因此还需要安装 PyTorch 和 PyTorch3D。PyTorch/PyTorch3D 对 Python、CUDA、平台版本比较敏感，先按当前机器安装匹配的 PyTorch，再安装 PyTorch3D：

```bash
# CPU 示例；如果有 CUDA，请按本机 CUDA 版本换成对应的 PyTorch 安装命令。
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 常见源码安装方式；如果失败，按 PyTorch3D 官方说明匹配 Python/Torch/CUDA 版本。
python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

安装后可用下面的命令快速验证：

```bash
python -c "import numpy, open3d, torch; from pytorch3d.io import load_obj, load_ply; print('python deps ok')"
```

### Blender 环境

`repair_non_manifold_with_blender.py` 和 `repair_non_manifold_with_blender_advanced.py` 必须用 Blender 运行，不能直接用 `python` 运行，因为它们依赖 Blender 内置模块 `bpy`、`bmesh` 和 `mathutils`。

Linux 上通常可以直接使用：

```bash
blender --background --python preprocess/repair_non_manifold_with_blender.py
```

macOS 上如果命令行找不到 `blender`，可以使用完整路径：

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python preprocess/repair_non_manifold_with_blender.py
```

如果本机还没有 Blender，Linux 可用系统包管理器安装，macOS 可安装官网版本或使用 Homebrew Cask：

```bash
brew install --cask blender
```

Blender 修复脚本只使用 Blender 自带模块，不需要在普通 conda 环境里安装 `bpy`。脚本会尝试启用 Print3D 插件；如果当前 Blender 没有可用的 Print3D Make Manifold 操作，会自动跳过该步骤并继续执行其它修复操作。

## 文件用途

### `check_manifold.py`

批量检测网格资产的流形性和基础几何质量。

支持格式：

- `.obj`
- `.stl`
- `.ply`

主要功能：

- 递归扫描一个或多个输入目录。
- OBJ/PLY 使用 PyTorch3D 读取并转为三角面；STL 使用 Open3D 读取。
- 检查 Open3D 指标：`is_watertight`、`is_edge_manifold`、`is_vertex_manifold`、`is_orientable`、`is_self_intersecting`。
- 统计 OBJ 和 ASCII PLY 中的原始非三角面数量。
- 输出 JSON 详细报告和 txt 摘要。
- 支持多进程并行检测。

常用命令：

```bash
python preprocess/check_manifold.py \
  --input-dir get_assets/0_merged_visual_objs \
  --output-dir reports/manifold_check \
  --workers 8
```

多个输入目录：

```bash
python preprocess/check_manifold.py \
  --input-dir get_assets/1_manifold_visual_objs get_assets/2_non_manifold_objs_repaired_advanced \
  --output-dir reports/manifold_check_after_repair \
  --workers 8
```

常用参数：

- `--input-dir`：必填，可接收一个或多个输入目录。
- `--output-dir`：报告输出目录，默认 `reports/manifold_check`。
- `--no-recursive`：只扫描输入目录第一层，不递归子目录。
- `--workers`：并行进程数，`0` 表示自动，默认 `0`。
- `--no-json`：不保存 JSON 报告，只打印汇总。
- `--quiet`：安静模式，只显示汇总信息。

退出码说明：

- 所有网格均为流形且无解析错误时返回 `0`。
- 发现非流形网格或解析错误时返回 `1`。如果只是想生成报告，不希望 shell 流程中断，可以在命令后追加 `|| true`。

输出内容：

- `manifold_check_YYYYmmdd_HHMMSS.json`：完整检测结果。
- `manifold_summary_YYYYmmdd_HHMMSS.txt`：文本摘要。

### `filter_manifold_objs.py`

从输入目录中递归读取 OBJ 文件，按流形检查结果分流到两个输出目录，并生成 CSV 报告。

主要功能：

- 只处理 `.obj` 文件。
- 使用与 `check_manifold.py` 类似的加载、清理和 Open3D 检查逻辑。
- 将通过检查的 OBJ 写入 `manifold_output_root`。
- 将未通过检查或解析失败的 OBJ 写入 `non_manifold_output_root`。
- 输出 `manifold_report.csv`，记录每个 OBJ 的流形状态、失败原因和几何统计。
- 导出时会重写为只包含顶点和三角面的简化 OBJ；如果重写失败，会回退为复制原文件。

常用命令：

```bash
python preprocess/filter_manifold_objs.py \
  --input-root get_assets/0_merged_visual_objs \
  --manifold-output-root get_assets/1_manifold_visual_objs \
  --non-manifold-output-root get_assets/1_non_manifold_visual_objs \
  --report-csv get_assets/1_manifold_visual_objs/manifold_report.csv \
  --workers 10
```

常用参数：

- `--input-root`：输入 OBJ 根目录，默认 `get_assets/0_merged_visual_objs`。
- `--manifold-output-root`：流形 OBJ 输出目录，默认 `get_assets/1_manifold_visual_objs`。
- `--non-manifold-output-root`：非流形 OBJ 输出目录，默认 `get_assets/1_non_manifold_visual_objs`。
- `--report-csv`：CSV 报告路径，默认 `get_assets/1_manifold_visual_objs/manifold_report.csv`。
- `--overwrite`：覆盖已存在的输出 OBJ。
- `--limit`：只处理前 N 个 OBJ，适合 smoke test。
- `--workers`：并行进程数，`1` 为单进程，`<=0` 自动选择，默认 `10`。

### `repair_non_manifold_with_blender.py`

使用 Blender 对非流形 OBJ 进行批量基础修复。

修复流程：

1. 读取 OBJ。
2. 尝试执行 Print3D 的 Make Manifold。
3. 合并重复点。
4. 填洞。
5. 四边形/多边形转三角面。
6. 重新统一法线方向。
7. 导出 OBJ，并删除额外生成的 `.mtl`。

使用方法：

1. 修改脚本顶部配置：

```python
INPUT_ROOT = "/path/to/1_non_manifold_visual_objs"
OUTPUT_ROOT = "/path/to/2_non_manifold_objs_repaired"
FAILED_ROOT = "/path/to/2_non_manifold_objs_failed"

OVERWRITE = False
LIMIT = None
PRESERVE_REGISTRY_LEVEL = True
```

2. 用 Blender 后台模式运行：

```bash
blender --background --python preprocess/repair_non_manifold_with_blender.py
```

适用场景：

- 非流形问题相对简单。
- 希望保留原始形状，不想使用体素重建。
- 先做快速修复，再用 `check_manifold.py` 验证。

### `repair_non_manifold_with_blender_advanced.py`

使用 Blender 对非流形 OBJ 进行更激进的批量修复。它包含基础修复、质量指标检查、体素重建、Decimate 简化和半体素重试流程。

高级修复流程：

1. 导入 OBJ 并记录初始质量指标。
2. 对每个 mesh object 执行基础清理。
3. 用 `bmesh` 检查非流形边、边界/孤立边、自相交、非平面面、连通分量数量。
4. 如果基础清理后仍不合格，执行 voxel remesh 和 decimate。
5. 如果仍不合格且启用 `RETRY_WITH_HALF_VOXEL_ON_FAIL`，用一半 voxel size 再试一次。
6. `final_ok=True` 的结果写入 `OUTPUT_ROOT`。
7. `final_ok=False` 或异常的结果写入 `FAILED_ROOT`，并尽量在 `FAILED_RESULT_ROOT` 保存修复后但仍不合格的结果快照。

使用方法：

1. 修改脚本顶部路径：

```python
INPUT_ROOT = "/path/to/1_non_manifold_visual_objs"
OUTPUT_ROOT = "/path/to/2_non_manifold_objs_repaired_advanced"
FAILED_ROOT = "/path/to/2_non_manifold_objs_failed_advanced"
FAILED_RESULT_ROOT = "/path/to/2_non_manifold_failed_results_advanced"
```

2. 按资产尺度调整高级参数：

```python
ALWAYS_USE_COMPLEX_REPAIR = False
VOXEL_SIZE = 0.001
DECIMATE_RATIO = 0.1
MAX_HOLE_SIDES = 0
RETRY_WITH_HALF_VOXEL_ON_FAIL = True
```

3. 用 Blender 后台模式运行：

```bash
blender --background --python preprocess/repair_non_manifold_with_blender_advanced.py
```

关键参数说明：

- `ALWAYS_USE_COMPLEX_REPAIR`：是否所有文件都强制走 voxel remesh + decimate。默认只在基础清理失败时启用。
- `VOXEL_SIZE`：体素重建尺寸。越小越保细节但越慢，也可能生成更多面。
- `DECIMATE_RATIO`：简化比例。越小面数越少，但细节损失更大。
- `MAX_HOLE_SIDES`：Blender 填洞允许的边数，`0` 表示尽量填所有洞。
- `VERBOSE_PER_FILE`：是否打印每个文件的详细指标。
- `SUPPRESS_BLENDER_OP_LOGS`：是否隐藏 Blender operator 的大量日志。
- `NON_PLANAR_EPSILON`：判定非平面面的容差。
- `RETRY_WITH_HALF_VOXEL_ON_FAIL`：失败后是否用一半体素尺寸再试一次。

适用场景：

- 基础修复仍有非流形边、边界边、自相交或非平面面。
- 可以接受 voxel remesh 带来的形状和拓扑变化。
- 需要保存 `final_ok=False` 的失败结果，便于人工检查。

## 推荐工作流示例

```bash
# 1. 检查原始合并 OBJ。
python preprocess/check_manifold.py \
  --input-dir get_assets/0_merged_visual_objs \
  --output-dir reports/manifold_check_raw \
  --workers 8 || true

# 2. 分流流形/非流形 OBJ。
python preprocess/filter_manifold_objs.py \
  --input-root get_assets/0_merged_visual_objs \
  --manifold-output-root get_assets/1_manifold_visual_objs \
  --non-manifold-output-root get_assets/1_non_manifold_visual_objs \
  --report-csv get_assets/1_manifold_visual_objs/manifold_report.csv \
  --workers 10

# 3. 修改 repair_non_manifold_with_blender_advanced.py 顶部路径后，用 Blender 修复。
blender --background --python preprocess/repair_non_manifold_with_blender_advanced.py

# 4. 检查修复后的输出。
python preprocess/check_manifold.py \
  --input-dir get_assets/2_non_manifold_objs_repaired_advanced \
  --output-dir reports/manifold_check_repaired \
  --workers 8 || true
```

## 注意事项

- 普通 Python 脚本应从仓库根目录运行，或传入绝对路径，避免默认相对路径指向错误位置。
- Blender 修复脚本目前没有命令行参数，路径和修复参数需要在脚本顶部修改。
- `filter_manifold_objs.py` 和 `check_manifold.py` 的 OBJ/PLY 读取依赖 PyTorch3D；如果只安装了 Open3D，脚本仍会在读取 OBJ/PLY 时报错。
- `check_manifold.py` 对 binary PLY 不会精确统计原始非三角面，只会在报告里记录对应的 face check warning。
- Blender 的 voxel remesh 会改变拓扑和细节。对需要严格保持原始几何的资产，先使用基础修复脚本，再考虑高级修复脚本。
- 修复完成后必须重新运行 `check_manifold.py`，不要只根据 Blender 脚本日志判断资产已经可用。
- `PRESERVE_REGISTRY_LEVEL=True` 会尽量保留 `aigen_objs`、`objaverse`、`lightwheel` 等 registry 目录层级，便于和原始资产来源对应。
