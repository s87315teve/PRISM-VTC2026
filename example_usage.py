"""
DatasetWriter 使用範例

展示如何在不同程式中使用 DatasetWriter 來寫入各類資料。
實際使用時，這些程式會在不同的 process 中執行。
"""

import numpy as np
import time
from pathlib import Path
from dataset_writer import (
    DatasetWriter,
    TrajectoryData,
    BasestationConfig,
    CSIMetadata,
    ScenarioMetadata,
    generate_readme
)


def simulate_vehicle_program():
    """
    模擬車輛感測器程式
    負責寫入：RGB、Semantic、LiDAR、Radar、Trajectory
    """
    print("\n" + "="*60)
    print("Vehicle Sensor Program Started")
    print("="*60)
    
    # 建立 writer
    writer = DatasetWriter(
        dataset_root="./dataset",
        scenario_id="scenario_001",
        entity_type="vehicle",
        entity_id="veh_0",
        queue_size=100,
        num_workers=2
    )
    
    try:
        # 模擬 10 個 frame 的資料
        for frame_id in range(1, 11):
            print(f"[Vehicle] Processing frame {frame_id}...")
            
            # 模擬 RGB 圖片 (1080p)
            rgb_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            writer.write_rgb(frame_id=frame_id, image=rgb_image)
            
            # 模擬語意分割圖
            semantic_image = np.random.randint(0, 23, (1080, 1920), dtype=np.uint8)
            writer.write_semantic(frame_id=frame_id, image=semantic_image)
            
            # 模擬 LiDAR 點雲 (10000 點)
            lidar_points = np.random.randn(10000, 4).astype(np.float32)
            writer.write_lidar(frame_id=frame_id, points=lidar_points)
            
            # 模擬 Radar 資料
            radar_data = np.random.randn(64, 256).astype(np.float32)
            writer.write_radar(frame_id=frame_id, data=radar_data)
            
            # 寫入軌跡資料
            trajectory = TrajectoryData(
                timestamp=frame_id * 0.1,  # 10 FPS
                x=frame_id * 1.0,
                y=frame_id * 0.5,
                z=0.0,
                roll=0.0,
                pitch=0.0,
                yaw=frame_id * 0.1,
                velocity_x=10.0,
                velocity_y=5.0,
                velocity_z=0.0
            )
            writer.write_trajectory(frame_id=frame_id, trajectory_data=trajectory)
            
            # 模擬處理延遲
            time.sleep(0.05)
            
            # 顯示佇列狀態
            print(f"  Queue size: {writer.get_queue_size()}")
        
        print("[Vehicle] All frames queued, waiting for flush...")
        
    finally:
        writer.close()
        print("[Vehicle] Program finished.")


def simulate_basestation_program():
    """
    模擬基地台感測器程式
    負責寫入：RGB、Semantic、LiDAR、Config
    """
    print("\n" + "="*60)
    print("Basestation Sensor Program Started")
    print("="*60)
    
    writer = DatasetWriter(
        dataset_root="./dataset",
        scenario_id="scenario_001",
        entity_type="basestation",
        entity_id="bs_0",
        queue_size=100,
        num_workers=2
    )
    
    try:
        # 更新基地台設定
        bs_config = BasestationConfig(
            position_x=100.0,
            position_y=50.0,
            position_z=25.0,
            rotation_yaw=45.0,
            num_antennas=64,
            antenna_pattern="directional",
            tx_power_dbm=43.0,
            carrier_frequency_ghz=3.5,
            description="Urban macro base station"
        )
        writer.update_basestation_config(bs_config)
        print("[BS] Config updated")
        
        # 模擬 10 個 frame
        for frame_id in range(1, 11):
            print(f"[BS] Processing frame {frame_id}...")
            
            # 基地台視角的 RGB
            rgb_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            writer.write_rgb(frame_id=frame_id, image=rgb_image)
            
            # 語意分割
            semantic_image = np.random.randint(0, 23, (720, 1280), dtype=np.uint8)
            writer.write_semantic(frame_id=frame_id, image=semantic_image)
            
            # 基地台 LiDAR（可能是固定掃描）
            lidar_points = np.random.randn(50000, 4).astype(np.float32)
            writer.write_lidar(frame_id=frame_id, points=lidar_points)
            
            time.sleep(0.03)
        
    finally:
        writer.close()
        print("[BS] Program finished.")


def simulate_csi_program():
    """
    模擬 CSI 計算程式（Sionna）
    負責寫入：CSI 資料
    """
    print("\n" + "="*60)
    print("CSI Computation Program Started")
    print("="*60)
    
    writer = DatasetWriter(
        dataset_root="./dataset",
        scenario_id="scenario_001",
        entity_type="csi",
        entity_id=None,  # CSI 不需要 entity_id
        queue_size=100,
        num_workers=4  # CSI 資料較多，可以用更多 worker
    )
    
    try:
        # 更新 CSI 元資料
        csi_metadata = CSIMetadata(
            num_subcarriers=128,
            num_tx_antennas=64,
            num_rx_antennas=4,
            carrier_frequency_ghz=3.5,
            bandwidth_mhz=100.0,
            subcarrier_spacing_khz=30.0,
            description="5G NR CSI with MIMO configuration"
        )
        writer.update_csi_metadata(csi_metadata)
        print("[CSI] Metadata updated")
        
        # 定義 TX/RX 組合
        tx_list = ["bs_0", "bs_1"]
        rx_list = ["veh_0", "veh_1"]
        
        # 模擬 10 個 frame
        for frame_id in range(1, 11):
            print(f"[CSI] Processing frame {frame_id}...")
            timestamp = frame_id * 0.1
            
            # 計算所有 TX-RX 組合的 CSI
            for tx_id in tx_list:
                for rx_id in rx_list:
                    # 模擬 CSI 矩陣 (complex)
                    csi_data = (
                        np.random.randn(128, 64, 4) + 
                        1j * np.random.randn(128, 64, 4)
                    ).astype(np.complex64)
                    
                    writer.write_csi(
                        frame_id=frame_id,
                        tx_id=tx_id,
                        rx_id=rx_id,
                        csi_data=csi_data,
                        timestamp=timestamp
                    )
            
            time.sleep(0.02)
        
    finally:
        writer.close()
        print("[CSI] Program finished.")


def simulate_second_vehicle():
    """
    模擬第二輛車的感測器程式
    展示多車輛支援
    """
    print("\n" + "="*60)
    print("Second Vehicle Program Started")
    print("="*60)
    
    writer = DatasetWriter(
        dataset_root="./dataset",
        scenario_id="scenario_001",
        entity_type="vehicle",
        entity_id="veh_1",  # 第二輛車
        queue_size=100
    )
    
    try:
        for frame_id in range(1, 11):
            print(f"[Veh1] Processing frame {frame_id}...")
            
            # 只寫入部分感測器（示範彈性）
            rgb_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            writer.write_rgb(frame_id=frame_id, image=rgb_image)
            
            trajectory = {
                "timestamp": frame_id * 0.1,
                "x": 50.0 + frame_id * 0.8,
                "y": 30.0 + frame_id * 0.3,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 1.57,
                "velocity_x": 8.0,
                "velocity_y": 3.0,
            }
            writer.write_trajectory(frame_id=frame_id, trajectory_data=trajectory)
            
            time.sleep(0.02)
        
    finally:
        writer.close()
        print("[Veh1] Program finished.")


def update_scenario_info():
    """
    更新場景資訊（可在任何程式中執行）
    """
    print("\n" + "="*60)
    print("Updating Scenario Metadata")
    print("="*60)
    
    # 可以用任何 entity_type 來更新場景資訊
    writer = DatasetWriter(
        dataset_root="./dataset",
        scenario_id="scenario_001",
        entity_type="vehicle",
        entity_id="veh_0"
    )
    
    metadata = ScenarioMetadata(
        scenario_id="scenario_001",
        map_name="Town03",
        weather="ClearNoon",
        num_vehicles=2,
        num_basestations=2,
        duration_seconds=1.0,  # 10 frames @ 10 FPS
        fps=10.0,
        description="Urban intersection scenario with two vehicles and two base stations"
    )
    writer.update_scenario_metadata(metadata)
    
    writer.close()
    print("Scenario metadata updated.")


def main():
    """
    主程式：模擬所有程式的執行
    
    注意：實際使用時，這些會在不同的 process 中執行
    這裡只是為了展示而順序執行
    """
    print("\n" + "#"*60)
    print("# DatasetWriter Demo")
    print("# Note: In real usage, these programs run in separate processes")
    print("#"*60)
    
    # 1. 車輛感測器程式
    simulate_vehicle_program()
    
    # 2. 第二輛車
    simulate_second_vehicle()
    
    # 3. 基地台程式
    simulate_basestation_program()
    
    # 4. CSI 計算程式
    simulate_csi_program()
    
    # 5. 更新場景資訊
    update_scenario_info()
    
    # 6. 生成 README（事後執行）
    print("\n" + "="*60)
    print("Generating README.md")
    print("="*60)
    generate_readme("./dataset")
    
    # 顯示產生的目錄結構
    print("\n" + "="*60)
    print("Generated Directory Structure:")
    print("="*60)
    import subprocess
    result = subprocess.run(
        ["find", "./dataset", "-type", "f", "-name", "*.json", "-o", 
         "-type", "f", "-name", "*.csv", "-o",
         "-type", "f", "-name", "*.md"],
        capture_output=True, text=True
    )
    print(result.stdout)
    
    # 統計檔案數量
    result = subprocess.run(
        ["find", "./dataset", "-type", "f"],
        capture_output=True, text=True
    )
    file_count = len(result.stdout.strip().split('\n'))
    print(f"\nTotal files generated: {file_count}")


if __name__ == "__main__":
    main()
