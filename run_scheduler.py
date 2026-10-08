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



# 調度器 - 系統核心 (Port: 5000)
class Scheduler(NetworkComponent):
    def __init__(self, mode='non_interactive'):
        super().__init__('scheduler', 5000)
        self.mode = mode
        self.current_timestamp = 0
        self.configs = []

        # 啟動網路監聽
        self.start_listening()

        self.register_handler('route_message', self._route_message)
        self.register_handler('set_start_time', self.set_start_time)
        
    
    def setup_network(self):
        """設定其他組件的網路連接"""
        self.register_component('wireless_env', 'localhost', 5001)
        self.register_component('end_user', 'localhost', 5002)
        self.register_component('visualizer', 'localhost', 5003)
        self.register_component('external_module', 'localhost', 5004)
    
    def load_config(self, config_file):
        """載入配置檔案並廣播給所有組件"""
        try:
            self.configs = read_json_array(config_file)
            # print(f"配置載入完成: {self.configs}")
            
        except Exception as e:
            print(f"載入配置失敗: {e}")


    
    def update_parameters(self, new_params):
        """更新模擬參數"""
        self.current_params.update(new_params)
        print(f"參數已更新: {new_params}")
    
    def _route_message(self, message):
        
        target_addr=self.remote_components[message["to"]]
        json_data = json.dumps(message).encode('utf-8')
        
        try:
            self.socket.sendto(json_data, target_addr)
            print(f"[{message['from']}] 經過 [{self.component_name}]發送訊息到 [{message['to']}]: {message['message_type']}")
        except Exception as e:
            print(f"發送訊息失敗: {e}")
    
    def _handle_message(self, message: Dict):
        """處理接收到的訊息"""
        # print(f"receive msg: \n {message}")
        if message["to"] != self.component_name:
            self.message_handlers["route_message"](message)

        else:
            message_type = message.get("message_type")
            if message_type in self.message_handlers:
                try:
                    self.message_handlers[message_type](message)
                except Exception as e:
                    print(f"[{self.component_name}] 處理訊息 {message_type} 時發生錯誤: {e}")
            else:
                print(f"[{self.component_name}] 未處理的訊息類型: {message_type}")


    
   
    def run_simulation(self):
        """依序執行配置中的所有指令，並在每個指令間加入延遲"""
        for i, current_config in enumerate(self.configs):
            print(f"[scheduler] 執行第 {i+1} 個配置: {current_config['message_type']}")
            
            self.send_message(
                target_component=current_config["target"], 
                message_type=current_config["message_type"], 
                data=current_config,
                to_scheduler=False
            )
    
    def set_start_time(self):
        self.sys_print("start to set start time")
        data={"start_time": self.start_time}
        self.send_message(
            target_component="visualizer", 
            message_type="reset_start_time", 
            data=data,
            to_scheduler=False
        )
        self.send_message(
            target_component="wireless_env", 
            message_type="reset_start_time", 
            data=data,
            to_scheduler=False
        )
        self.send_message(
            target_component="external_module", 
            message_type="reset_start_time", 
            data=data,
            to_scheduler=False
        )
        self.sys_print("start time message sended")
   



def main():
    """
    基於UDP+JSON的Wireless Digital Twin系統
    """

    # === 非互動式模式 ===
    # print("啟動非互動式模擬...")
    config_file="simulation_config_with_carla.json"
    
    
    # 啟動調度器
    scheduler = Scheduler(mode='non_interactive')
    scheduler.setup_network()
    scheduler.set_start_time()
    # 載入配置並開始模擬
    scheduler.load_config(config_file)
    # time.sleep(2)

    # 建立執行緒
    thread_run_simulation = threading.Thread(target=scheduler.run_simulation)

    # 啟動執行緒
    thread_run_simulation.start()
    # scheduler.run_simulation()
    scheduler.process_messages()
    

    
   


if __name__ == "__main__":
    main()
    input("")
