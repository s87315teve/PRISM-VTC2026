import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 載入資料
data = np.load('../dataset/scenario_004/basestations/bs_0/lidar/frame_00080.npy')
print(data.shape)
xyz = data[:, :3]
intensity = data[:, 3]

# 建立 3D 圖
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 繪製散點圖，用強度值作為顏色
scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], 
                     c=intensity, 
                     cmap='jet',  # 可改為 'viridis', 'hot', 'rainbow' 等
                     s=1,  # 點的大小
                     alpha=0.6)

# 加上色條
plt.colorbar(scatter, label='Intensity')

# 設定標籤
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

plt.show()