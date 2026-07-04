#!/usr/bin/env python3
"""
可视化 ContactOpt 优化输出的目标/最终物体接触图。

功能说明：
    - 从 contactopt_fit 输出的 PKL 文件中加载优化结果
    - 在物体点云上根据接触值渲染颜色（热力图形式）
    - 可叠加显示初始手部顶点（蓝色）和优化后手部顶点（红色）
    - 支持调整颜色映射、点云大小、窗口尺寸等参数

典型用法：
    python contact_target_visualize.py --opti-pkl output.pkl --frame-index 0
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d


def _get_cmap(name: str):
    """
    获取 matplotlib 颜色映射表，兼容新旧版本 API。

    Args:
        name: 颜色映射表名称，如 'turbo', 'viridis', 'jet' 等

    Returns:
        matplotlib Colormap 对象
    """
    import matplotlib

    # matplotlib 3.7+ 使用 colormaps 属性
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]

    # 旧版本使用 cm.get_cmap 函数
    import matplotlib.cm as cm

    return cm.get_cmap(name)


def _load_object_points(payload: dict, frame_index: int) -> np.ndarray:
    """
    从原始 NPZ 文件中加载指定帧的物体点云数据。

    物体点云存储在预处理阶段生成的 NPZ 文件中，这里根据帧 ID
    从中检索对应的点云数据。

    Args:
        payload: PKL 文件加载的数据字典，包含 'source_preprocess_file' 和 'raw_frame_id'
        frame_index: PKL 中的帧索引

    Returns:
        物体点云坐标，形状为 (N, 3)，单位为米
    """
    # 获取预处理 NPZ 文件路径
    npz_path = Path(payload["source_preprocess_file"])
    # 获取该帧对应的原始帧 ID
    raw_id = int(payload["raw_frame_id"][frame_index])

    with np.load(str(npz_path), allow_pickle=True) as data:
        # 兼容不同的键名格式
        raw_key = "raw_frame_id" if "raw_frame_id" in data.files else "frame_id"
        raw_all = np.asarray(data[raw_key])
        # 找到 raw_id 在数组中的位置
        matches = np.where(raw_all == raw_id)[0]
        if matches.size == 0:
            raise ValueError(f"raw_frame_id={raw_id} not found in {npz_path}")

        # 兼容不同的点云键名格式
        obj_key = "obj_points_world" if "obj_points_world" in data.files else "obj_points"
        return np.asarray(data[obj_key][int(matches[0])], dtype=np.float64)


def _make_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    根据点和颜色数组创建 Open3D 点云对象。

    Args:
        points: 点云坐标，形状 (N, 3)
        colors: 每个点对应的 RGB 颜色，形状 (N, 3)，值域 [0, 1]

    Returns:
        Open3D PointCloud 对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def _uniform_point_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """
    创建单色点云，所有点使用相同的颜色。

    用于渲染手部顶点等辅助几何信息。

    Args:
        points: 点云坐标，形状 (N, 3)
        color: RGB 颜色元组，如 (1.0, 0.0, 0.0) 表示红色

    Returns:
        Open3D PointCloud 对象
    """
    # 将单一颜色扩展为与点数量相同的数组
    colors = np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1))
    return _make_point_cloud(points, colors)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        包含所有参数的 Namespace 对象
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # 输入文件路径（必需）
    parser.add_argument("--opti-pkl", required=True,
                        help="ContactOpt 优化输出的 PKL 文件路径")

    # 帧索引：选择可视化哪一帧
    parser.add_argument("--frame-index", type=int, default=0,
                        help="PKL 文件中的帧索引，从 0 开始")

    # 选择可视化哪种接触图
    parser.add_argument(
        "--contact-key",
        default="target_obj_contact",
        choices=["target_obj_contact", "final_obj_contact"],
        help="可视化目标物体接触图还是最终物体接触图"
    )

    # 颜色映射表名称
    parser.add_argument("--cmap", default="turbo",
                        help="Matplotlib 颜色映射表名称，默认 turbo（蓝-红渐变）")

    # 点云渲染大小
    parser.add_argument("--point-size", type=float, default=5.0,
                        help="点云中每个点的大小，默认 5.0")

    # 窗口尺寸
    parser.add_argument("--width", type=int, default=1400,
                        help="可视化窗口宽度，默认 1400 像素")
    parser.add_argument("--height", type=int, default=900,
                        help="可视化窗口高度，默认 900 像素")

    # 控制手部顶点的显示/隐藏
    parser.add_argument("--hide-init-hand", action="store_true",
                        help="隐藏初始手部顶点（蓝色）")
    parser.add_argument("--hide-opt-hand", action="store_true",
                        help="隐藏优化后手部顶点（红色）")

    return parser.parse_args()


def main() -> None:
    """
    主函数：加载数据并启动 3D 可视化。

    流程：
        1. 解析命令行参数
        2. 加载 PKL 文件中的优化结果
        3. 提取物体点云和接触值
        4. 根据接触值计算颜色（热力图）
        5. 创建 Open3D 可视化窗口并显示
    """
    args = parse_args()

    # 解析 PKL 文件路径
    pkl_path = Path(args.opti_pkl).resolve()
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    # 获取帧索引并验证范围
    frame_index = int(args.frame_index)
    frame_count = int(np.asarray(payload["raw_frame_id"]).shape[0])
    if frame_index < 0 or frame_index >= frame_count:
        raise IndexError(f"--frame-index 必须在 [0, {frame_count - 1}] 范围内")

    # 加载物体点云和接触值
    obj_points = _load_object_points(payload, frame_index)
    contact = np.asarray(payload[args.contact_key][frame_index], dtype=np.float64).reshape(-1)

    # 验证点云和接触值数量匹配
    if contact.shape[0] != obj_points.shape[0]:
        raise ValueError(
            f"{args.contact_key} 点数不匹配: contact={contact.shape[0]}, object={obj_points.shape[0]}"
        )

    # 获取颜色映射表，将接触值转换为颜色
    cmap = _get_cmap(args.cmap)
    # 将接触值裁剪到 [0, 1] 范围后映射为颜色
    obj_colors = np.asarray(cmap(np.clip(contact, 0.0, 1.0)))[:, :3]

    # 构建可视化几何体列表
    geoms = [_make_point_cloud(obj_points, obj_colors)]

    # 添加初始手部顶点（蓝色）
    if not args.hide_init_hand:
        geoms.append(_uniform_point_cloud(
            payload["init_hand_verts_world"][frame_index],
            (0.0, 0.15, 1.0)  # 蓝色
        ))

    # 添加优化后手部顶点（红色）
    if not args.hide_opt_hand:
        geoms.append(_uniform_point_cloud(
            payload["opt_hand_verts_world"][frame_index],
            (1.0, 0.0, 0.0)  # 红色
        ))

    # 添加坐标系辅助线（世界坐标系参考）
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08))

    # 获取原始帧 ID 用于显示
    raw_id = int(payload["raw_frame_id"][frame_index])

    # 打印统计信息
    print(f"[contact-vis] PKL文件: {pkl_path}")
    print(f"[contact-vis] 帧索引={frame_index}, 原始帧ID={raw_id}, 接触图类型={args.contact_key}")
    print(
        f"[contact-vis] 接触值统计: "
        f"均值={contact.mean():.6f}, 最大值={contact.max():.6f}, "
        f">0.5比例={(contact > 0.5).mean():.6f}"
    )
    print("[contact-vis] 图例: 物体颜色=接触值热力图; 蓝色=初始手部; 红色=优化后手部")

    # 创建 Open3D 可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"{args.contact_key} raw frame {raw_id}",
        width=int(args.width),
        height=int(args.height),
    )

    # 添加所有几何体到场景
    for geom in geoms:
        vis.add_geometry(geom)

    # 设置渲染选项
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
    render_opt.point_size = float(args.point_size)  # 点大小

    # 运行可视化，窗口关闭后自动退出
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
