import trimesh
import numpy as np

# 讀取 PLY 檔案
mesh = trimesh.load('Plane.ply')

# 取得物件的邊界框（bounding box）
bounds = mesh.bounds  # 回傳 [[min_x, min_y, min_z], [max_x, max_y, max_z]]

# 計算各軸向的範圍
extents = mesh.extents  # 回傳 [x_range, y_range, z_range]

# 計算物件中心點
centroid = mesh.centroid

# 顯示結果
print(f"物件邊界：")
print(f"  最小值 (x, y, z): {bounds[0]}")
print(f"  最大值 (x, y, z): {bounds[1]}")
print(f"\n物件尺寸範圍：")
print(f"  X軸範圍: {extents[0]:.2f}")
print(f"  Y軸範圍: {extents[1]:.2f}")
print(f"  Z軸範圍: {extents[2]:.2f}")
print(f"\n物件中心點: {centroid}")
print(f"\n物件總體積: {mesh.volume:.2f}")
