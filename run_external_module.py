import os # Configure which GPU
import numpy as np

# For plotting
# %matplotlib inline
# also try %matplotlib widget

import matplotlib.pyplot as plt

import socket
import json
import threading
import time
from typing import Dict, Any, Callable
from utils import NetworkComponent, read_json_array


import numpy as np

class RSUAngleCalculator:
    def __init__(self, rsu_position, reference_rx_position):
        """
        初始化 RSU 角度計算器
        
        Parameters:
        -----------
        rsu_position : list or array
            RSU 的位置 [x, y, z]（只用前兩個）
        reference_rx_position : list or array
            參考 rx 的位置（用來定義 RSU 的正前方）
        """
        self.rsu_pos = np.array(rsu_position[:2])
        
        # 計算 RSU 的正前方向
        ref_pos = np.array(reference_rx_position[:2])
        forward_vector = ref_pos - self.rsu_pos
        self.forward_angle = np.arctan2(forward_vector[1], forward_vector[0])
        
    def calculate_angle(self, rx_position):
        """
        計算 rx 相對於 RSU 正前方的夾角
        
        Parameters:
        -----------
        rx_position : list or array
            rx 的位置 [x, y, z]
            
        Returns:
        --------
        angle_deg : float
            相對角度（度），範圍 -180 到 180
            正值 = 右側，負值 = 左側
        """
        rx_pos = np.array(rx_position[:2])
        
        # 計算從 rsu 指向 rx 的向量角度
        vector_to_rx = rx_pos - self.rsu_pos
        angle_to_rx = np.arctan2(vector_to_rx[1], vector_to_rx[0])
        
        # 計算相對角度
        relative_angle = angle_to_rx - self.forward_angle
        
        # 標準化到 [-π, π]
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle))
        
        return np.degrees(relative_angle)


# 使用範例
rsu_position = [340.0, 80.0, 12.0]
rx_40_position = [292.1495666503906, -45.17396545410156, 0.0]

# 初始化計算器（用 rx_40 定義正前方）
angle_calculator = RSUAngleCalculator(rsu_position, rx_40_position)
# angle = angle_calculator.calculate_angle(rx_40_position)
# print(f"rx_40 相對於 rsu 正前方的夾角: {angle:.2f}°")

# 無線環境模擬器 (Port: 5001)
class ExternalModule(NetworkComponent):
    def __init__(self):
        super().__init__('external_module', 5004)
        self.simulation_params = {}
        
        
        # 啟動網路監聽
        self.start_listening()
        self.register_handler('send_scene_info', self._get_scene_info)
        self.register_handler('beam_control', self._beam_control)
        self.register_handler('reset_start_time', self._reset_start_time)
    
    def _get_scene_info(self, message):
        pass
    def _beam_control(self, message):
        scene_obj_info_list=message["data"]["scene_obj_info"]
        # self.sys_print(f"scene_obj_info_list:\n{scene_obj_info_list}")
        rsu_obj = None
        first_rx=None
        for obj in scene_obj_info_list:
            if obj['id'] == 'rsu':
                rsu_obj = obj
            elif obj['id'].startswith('rx_') and first_rx is None:
                first_rx = obj
        angle = angle_calculator.calculate_angle(first_rx['position'])

        if angle < -22.5:
            target_steering_angle=-30
        elif angle>=-22.5 and angle<-7.5:
            target_steering_angle=-15
        else:
            target_steering_angle=0

        self.sys_print(f"target_steering_angle={target_steering_angle}")
        data={
            "pattern":"elliptical_beam",
            "steering_angle":target_steering_angle
        }


        self.send_message(
            target_component="wireless_env", 
            message_type="update_antenna_pattern", 
            data=data,
            to_scheduler=True
        )

    def _reset_start_time(self, message):
        """ message["data"] example
        {
            "target": "wireless_env",
            "message_type": "set_dataset_writer",
            "start_time" : float in seconds
            
        }
        """
        self.offset_time = self.start_time - message["data"]["start_time"] 
        self.sys_print(f"Reset start time, offset time: {self.offset_time:3f} s")




    

    
  

def main():
    """
    基於UDP+JSON的Wireless Digital Twin系統
    """
    
    # 啟動其他組件
    external_module = ExternalModule()
    external_module.register_component('scheduler', 'localhost', 5000)
    external_module.register_component('wireless_env', 'localhost', 5001)
    external_module.register_component('end_user', 'localhost', 5002)
    external_module.register_component('visualizer', 'localhost', 5003)
    
    
    # 等待網路連接建立
    print(f"[{external_module.component_name}] 等待網路連接建立...")
    time.sleep(1)
    external_module.process_messages()
    
  
   


if __name__ == "__main__":
    main()