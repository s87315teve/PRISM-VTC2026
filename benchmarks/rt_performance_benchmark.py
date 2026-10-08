# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

# Other imports
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import drjit as dr
import mitsuba as mi
import math
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, ITURadioMaterial,\
    Camera, PathSolver, InteractionType, RadioMapSolver, PolarizedAntennaPattern, register_antenna_pattern, register_polarization
no_preview = True # Toggle to False to use the preview widget


scene_name="../my_scene/nycu_carla_v7_add_plane_yf_mirror/nycu_carla_v7_add_plane_yf_mirror.xml"
scene = load_scene(scene_name)


print("set material...")
scene.frequency=28e9
# new_material = ITURadioMaterial("new_material",
#                                 "concrete",
#                                thinkness=0.1)
new_material = sionna.rt.RadioMaterial(f"my_material_{scene.frequency}",
                            relative_permittivity=5.24,
                            conductivity=0.0462*(scene.frequency/1e9)**0.7822,
                            scattering_pattern='lambertian')
for i, obj in enumerate(scene.objects.values()):
    scene.get(obj.name).radio_material=new_material
print("set material success")


# 座標系統設定：
# 原點 (0,0) 為參考中心
# 基地台 (BS) 位於右下角 (1.5, -12)
# 地圖已沿 XY 軸各翻轉一次 (相當於旋轉 180 度)

tx_position=(1.5, -12.5)
# node_positions = {
#     1:  (0, -12.5),    # 緊鄰基地台
#     2:  (0, -11),   # 底部走廊 (原上方)
#     3:  (1.5, -11),     # 垂直向上
#     4:  (2, -2),      # 樓梯旁 (中段)
#     5:  (-3.5, 2.5),      # 樓梯另一側 (中段)
#     6:  (2, 5),     # 頂部角落 (原左下)
#     7:  (3.5, 2),      # 電梯旁
#     8:  (2, 17),    # 頂部走廊
#     9:  (-21, 14),   # 頂部中央
#     10: (-21, -12.5),  # 底部中央 (原上方)
#     11: (-42, -12.5),  # 底部遠端 (原上方)
#     12: (-42, 2.5),    # 最左側 (原右側電梯)
# }



max_depth=3

import random
# 定義範圍
x_min, x_max = -300, 300
y_min, y_max = -300, 300
z_min, z_max = 1, 100

# num_points = 100000

# # 生成 2000 個點
# points = []
# for _ in range(num_points):
#     x = random.uniform(x_min, x_max)
#     y = random.uniform(y_min, y_max)
#     z = random.uniform(z_min, z_max)
#     points.append((x, y, z))

# # 依 x 再依 y 排序
# points_sorted = sorted(points, key=lambda p: (p[0], p[1], p[2]))

# # 轉成字典格式，key 為 1 到 100000
# node_positions = {i+1: points_sorted[i] for i in range(num_points)}

# # 驗證
# print(f"總共生成 {len(node_positions)} 個點")
# print(f"前 5 個點:")
# for i in range(1, 6):
#     print(f"  {i}: {node_positions[i]}")
# print(f"後 5 個點:")
# for i in range(num_points-4, num_points+1):
#     print(f"  {i}: {node_positions[i]}")

# 建立一個 list 來儲存所有數據
data_records = []
default_beamwidth=120
# beam_pattern_list=["beamSetAbf_32A_like", "beamSetAbf_32B_like", "beamSetAbf_32C_like"]
beam_pattern_list=["dipole"]
# steering_angle_deg_dict={90:[np.pi/2, 0, 0], 45:[3*np.pi/4, 0, 0], 0:[np.pi, 0, 0] } # SteeringAngle_deg: 90, 45, 0 對應到不同的orientation

# orient_combos = [[np.pi/2, 0, 0], [3*np.pi/4, 0, 0], [np.pi, 0, 0]] 
# print(steering_angle_deg_dict)

refraction_flag=True
diffraction_flag=True
edge_diffraction_flag=True

import math


# 創建新的子圖
# 初始化索引


import time

# 測試配置
tx_num_list = [1, 2, 4, 8]  # 不同的發射器數量
rx_num_list = [60, 70, 80, 90, 100]  # 不同的接收器數量
num_depth_list = [3, 4, 5, 6, 7, 8]  # 不同的最大深度
num_samples_per_case = 10  # 每種情境測試的次數（隨機位置采樣）
enable_render = False  # 是否啟用渲染

my_cam = Camera(position=[300,-100, 50], look_at=[0,0,0])

# 儲存所有測試結果
test_results = []

# 配置天線陣列
scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="dipole",
    polarization="V"
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    pattern="dipole",
    polarization="V"
)

# 開始測試
total_cases = len(tx_num_list) * len(rx_num_list) * len(num_depth_list)
current_case = 0

for num_tx in tx_num_list:
    for num_rx in rx_num_list:
        for depth in num_depth_list:
            current_case += 1
            print(f"\n{'='*60}")
            print(f"測試情境 {current_case}/{total_cases}: TX={num_tx}, RX={num_rx}, Depth={depth}")
            print(f"{'='*60}")
            
            # 儲存此情境的所有采樣結果
            case_path_times = []
            case_render_times = []
            
            for sample_idx in range(num_samples_per_case):
                print(f"  采樣 {sample_idx + 1}/{num_samples_per_case}...")
                
                # 清除現有的 TX 和 RX
                for i in range(100):  # 清除可能存在的舊設備
                    try:
                        scene.remove(f"tx{i}")
                    except:
                        pass
                    try:
                        scene.remove(f"rx{i}")
                    except:
                        pass
                
                # 隨機生成 TX 位置並添加
                tx_positions = []
                for i in range(num_tx):
                    x = random.uniform(x_min, x_max)
                    y = random.uniform(y_min, y_max)
                    z = random.uniform(z_min, z_max)
                    tx_positions.append((x, y, z))
                    
                    tx = Transmitter(name=f'tx{i}', position=[x, y, z])
                    scene.add(tx)
                
                # 隨機生成 RX 位置並添加
                rx_positions = []
                for j in range(num_rx):
                    x = random.uniform(x_min, x_max)
                    y = random.uniform(y_min, y_max)
                    z = random.uniform(z_min, z_max)
                    rx_positions.append((x, y, z))
                    
                    rx = Receiver(name=f"rx{j}", position=[x, y, z])
                    scene.add(rx)
                
                # 測試路徑求解時間
                path_start_time = time.time()
                paths = PathSolver()(scene,
                                    max_depth=depth,
                                    los=True,
                                    specular_reflection=True,
                                    diffraction=diffraction_flag,
                                    edge_diffraction=edge_diffraction_flag,
                                    refraction=refraction_flag,
                                    synthetic_array=False)
                path_end_time = time.time()
                path_solve_time = path_end_time - path_start_time
                case_path_times.append(path_solve_time)
                
                print(f"    路徑求解時間: {path_solve_time:.4f} 秒")
                
                # 測試渲染時間（如果啟用）
                render_time = 0
                if enable_render:
                    render_start_time = time.time()
                    img = scene.render(camera=my_cam, 
                                     resolution=[640, 480], 
                                     paths=paths, 
                                     num_samples=512)
                    render_end_time = time.time()
                    render_time = render_end_time - render_start_time
                    case_render_times.append(render_time)
                    plt.close()
                    print(f"    渲染時間: {render_time:.4f} 秒")
            
            # 計算此情境的平均時間
            avg_path_time = np.mean(case_path_times)
            std_path_time = np.std(case_path_times)
            
            avg_render_time = np.mean(case_render_times) if case_render_times else 0
            std_render_time = np.std(case_render_times) if case_render_times else 0
            
            # 儲存結果
            result = {
                'num_tx': num_tx,
                'num_rx': num_rx,
                'depth': depth,
                'num_samples': num_samples_per_case,
                'avg_path_time': avg_path_time,
                'std_path_time': std_path_time,
                'avg_render_time': avg_render_time,
                'std_render_time': std_render_time,
                'all_path_times': case_path_times,
                'all_render_times': case_render_times
            }
            test_results.append(result)
            
            # 顯示統計結果
            print(f"\n  統計結果:")
            print(f"    路徑求解平均時間: {avg_path_time:.4f} ± {std_path_time:.4f} 秒")
            if enable_render:
                print(f"    渲染平均時間: {avg_render_time:.4f} ± {std_render_time:.4f} 秒")

# 顯示所有測試結果摘要
print(f"\n{'='*80}")
print("所有測試結果摘要")
print(f"{'='*80}")
print(f"{'TX':>4} {'RX':>5} {'Depth':>6} {'采樣數':>8} {'路徑求解(秒)':>20} {'渲染(秒)':>20}")
print(f"{'-'*80}")
for result in test_results:
    path_str = f"{result['avg_path_time']:.4f} ± {result['std_path_time']:.4f}"
    render_str = f"{result['avg_render_time']:.4f} ± {result['std_render_time']:.4f}" if enable_render else "N/A"
    print(f"{result['num_tx']:>4} {result['num_rx']:>5} {result['depth']:>6} "
          f"{result['num_samples']:>8} {path_str:>20} {render_str:>20}")

# 保存結果到文件
import json
output_filename = f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(test_results, f, indent=2, ensure_ascii=False)
print(f"\n結果已保存到: {output_filename}")
