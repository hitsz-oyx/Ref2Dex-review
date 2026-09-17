# ObjectInteractionCm 仓库记忆

## MANO 可视化入口

- 用户指定：GRAB、ARCTIC 等 MANO 轨迹的默认人工复核入口统一使用 `src.task.ObjectInteractionCm.visualize_grab`。
- 必须保留累计手物距离阈值着色（红色）与 object-to-hand KNN 着色（黄色）；除非用户另行指定，不使用不含这两项功能的普通 MANO viewer。
- ARCTIC articulated object 以逐帧 world-space object points 参与距离/KNN；没有单刚体 pose 时不得伪造 object mesh 语义。
