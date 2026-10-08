import numpy as np
import json

class PathGenerator:
    """路徑生成器，根據關鍵點生成位置序列"""
    
    @staticmethod
    def generate_path(waypoints, points_per_segment=None, total_points=None):
        """
        根據關鍵點生成路徑
        
        Args:
            waypoints: 關鍵點列表，每個點為 [x, y, z]
                      例如: [[0,0,0], [10,0,0], [10,10,0], [0,10,0], [0,0,0]]
            points_per_segment: 每段路徑的點數（不包含起點）
            total_points: 總點數（會平均分配到各段）
            
        Returns:
            positions: 位置陣列，shape: [total_points, 3]
        """
        waypoints = np.array(waypoints)
        
        if len(waypoints) < 2:
            raise ValueError("至少需要2個關鍵點")
        
        # 計算每段的點數
        num_segments = len(waypoints) - 1
        
        if points_per_segment is not None:
            points_per_seg = points_per_segment
        elif total_points is not None:
            points_per_seg = max(1, (total_points - 1) // num_segments)
        else:
            points_per_seg = 10  # 預設值
        
        positions = []
        
        for i in range(num_segments):
            start_point = waypoints[i]
            end_point = waypoints[i + 1]
            
            # 生成這一段的位置點（不包含終點，避免重複）
            segment_points = points_per_seg if i < num_segments - 1 else points_per_seg + 1
            
            for j in range(segment_points):
                t = j / points_per_seg  # 插值參數 0 到 1
                if t > 1.0:  # 最後一段包含終點
                    t = 1.0
                
                # 線性插值
                position = start_point + t * (end_point - start_point)
                positions.append(position)
        
        return np.array(positions)

def positions_to_json(position_list, tx_position=None, output_file="wireless_config.json"):
    """
    將位置列表轉換成指定的JSON格式並存檔
    
    Args:
        position_list: 位置陣列，來自PathGenerator.generate_path()
        tx_position: 固定的發射器位置 [x, y, z]，預設為 [0, 0, 50]
        output_file: 輸出檔案名稱
    """
    if tx_position is None:
        tx_position = [0, 0, 50]
    
    json_data = []
    
    for i, rx_pos in enumerate(position_list):
        # 時間步長從1開始
        timestamp = i+1
        
        config = {
            "target": "wireless_env",
            "message_type": "run", 
            "timestamp": timestamp,
            "tx_list": {
                "tx01": {
                    "x_pos": float(tx_position[0]),
                    "y_pos": float(tx_position[1]),
                    "z_pos": float(tx_position[2])
                } 
            },
            "rx_list": {
                "rx01": {
                    "x_pos": float(rx_pos[0]),
                    "y_pos": float(rx_pos[1]),
                    "z_pos": float(rx_pos[2])
                }
            },
            "output_content": [
                "SNR",
                "BER"
            ]
        }
        
        json_data.append(config)
    
    # 存檔
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    print(f"已生成 {len(json_data)} 個配置項目，存檔至 {output_file}")
    return json_data

# 主程式
if __name__ == "__main__":
    # 定義路徑點
    waypoints = [
        [350, 0, 25],      # 起點
        [150, -250, 25],   # 中間點
        [-100, 10, 25],    # 終點
    ]
    
    # 生成300個位置點
    position_list = PathGenerator.generate_path(waypoints, total_points=300)
    
    print(f"生成了 {len(position_list)} 個位置點")
    print(f"第一個位置: [{position_list[0][0]:.1f}, {position_list[0][1]:.1f}, {position_list[0][2]:.1f}]")
    print(f"最後一個位置: [{position_list[-1][0]:.1f}, {position_list[-1][1]:.1f}, {position_list[-1][2]:.1f}]")
    
    # 轉換成JSON格式並存檔
    json_data = positions_to_json(position_list, tx_position=[0, 0, 50], output_file="wireless_config.json")
    
    # 顯示前3個項目作為預覽
    print("\n前3個JSON項目預覽:")
    for i in range(min(3, len(json_data))):
        print(f"\n=== 項目 {i+1} ===")
        print(json.dumps(json_data[i], indent=2, ensure_ascii=False))