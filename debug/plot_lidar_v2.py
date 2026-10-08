import numpy as np
import open3d as o3d

# 載入點雲資料
data = np.load('../dataset/scenario_004/basestations/bs_0/lidar/frame_00080.npy')


# 分離座標和強度
xyz = data[:, :3]  # x, y, z
intensity = data[:, 3]  # i

# 建立 Open3D 點雲物件
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# 將強度值轉換為顏色（灰階或彩色映射）
# 正規化強度值到 0-1 範圍
intensity_normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())

# 選項 A：灰階顯示
colors = np.stack([intensity_normalized] * 3, axis=1)

# 選項 B：使用色彩映射（例如 jet colormap）
import matplotlib.pyplot as plt
cmap = plt.cm.jet
colors = cmap(intensity_normalized)[:, :3]  # 取 RGB，不要 alpha

pcd.colors = o3d.utility.Vector3dVector(colors)

# 視覺化
o3d.visualization.draw_geometries([pcd],
                                  window_name='Point Cloud',
                                  width=800,
                                  height=600,
                                  point_show_normal=False)