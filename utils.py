import socket
import json
import threading
import time
from typing import Dict, Any, Callable  
from queue import Queue
import numpy as np
# 網路通訊基礎類別
class NetworkComponent:
    def __init__(self, component_name: str, listen_port: int):
        self.component_name = component_name
        self.listen_port = listen_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', listen_port))
        self.running = False
        self.message_handlers = {}
        self.remote_components = {}  # 儲存其他組件的位址
        
        # 簡化：移除複雜的同步機制
        self.message_queue = Queue()
        self.processing = False
        self.process_lock = threading.Lock()
        self.start_time = time.time()
        self.offset_time = 0
    def sys_print(self, log_message: str):
        current_time = time.time() - self.start_time + self.offset_time
        print(f"[{self.component_name} {current_time:.3f}s] {log_message}")
    
    def register_component(self, component_name: str, host: str, port: int):
        """註冊其他組件的網路位址"""
        self.remote_components[component_name] = (host, port)
    
    def send_message(self, target_component: str, message_type: str, data: Dict[Any, Any], to_scheduler: bool = True):
        """發送JSON訊息到目標組件"""
        if target_component not in self.remote_components:
            self.sys_print(f"警告: 未知的目標組件 {target_component}")
            return
        
        message = {
            'from': self.component_name,
            'to': target_component,
            'message_type': message_type,
            'computer_timestamp': time.time(),
            'data': data
        }
        
        
        json_data = json.dumps(message).encode('utf-8')
        
        if to_scheduler:
            target_addr = self.remote_components["scheduler"]
            # self.sys_print(f"target addr: {target_addr}")
            
            try:
                self.socket.sendto(json_data, target_addr)
                self.sys_print(f"經過 [scheduler] 發送訊息到 [{target_component}]: {message_type}")
            except Exception as e:
                self.sys_print(f"發送訊息失敗: {e}")
        else:
            target_addr = self.remote_components[target_component]
            # self.sys_print(f"target addr: {target_addr}")
            try:
                self.socket.sendto(json_data, target_addr)
                self.sys_print(f"發送訊息到 [{target_component}]: {message_type}")
            except Exception as e:
                self.sys_print(f"發送訊息失敗: {e}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """註冊訊息處理函數"""
        self.message_handlers[message_type] = handler
    
    def start_listening(self):
        """開始監聽網路訊息"""
        self.running = True
        
        # 只啟動網路監聽線程
        listen_thread = threading.Thread(target=self._listen_loop)
        listen_thread.daemon = True
        listen_thread.start()
        
        self.sys_print(f"開始監聽 port {self.listen_port}")
    
    def stop_listening(self):
        """停止監聽"""
        self.running = False
        self.socket.close()
    
    def process_messages(self):
        """在主線程中處理訊息 - 不斷從queue中抓取並處理訊息"""
        while self.running:
            try:
                # 等待從queue中取得訊息
                message = self.message_queue.get(timeout=1)
                
                # 標記開始處理
                with self.process_lock:
                    self.processing = True
                
                self.sys_print(f"開始處理訊息: {message['message_type']}")
                
                # 直接處理訊息，不等待完成信號
                self._handle_message(message)
                
                # 標記完成處理
                with self.process_lock:
                    self.processing = False
                
                
                
                # 在訊息間加入小延遲，避免處理過快
                # time.sleep(0.001)
                
            except Exception as e:
                # 處理 Queue.Empty (timeout) 和其他錯誤
                if "Empty" in str(type(e).__name__):
                    # Queue timeout 是正常的，繼續等待
                    continue
                elif self.running:
                    self.sys_print(f"處理訊息錯誤: {e}")
                    # 發生錯誤時也要解除處理狀態
                    with self.process_lock:
                        self.processing = False
    
    def _listen_loop(self):
        """監聽循環 - 只負責接收訊息並放入queue"""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                
                # 直接將完整的JSON訊息放入queue
                self.message_queue.put(message)
                self.sys_print(f"收到訊息，放入queue: {message['message_type']}")
                
            except Exception as e:
                if self.running:
                    self.sys_print(f"接收訊息錯誤: {e}")
    
    def _handle_message(self, message: Dict):
        """處理接收到的訊息"""
        message_type = message.get("message_type")
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type](message)
                self.sys_print(f"完成處理訊息: {message['message_type']}")
            except Exception as e:
                self.sys_print(f"處理訊息 {message_type} 時發生錯誤: {e}")
        else:
            self.sys_print(f"未處理的訊息類型: {message_type}")
    
    def is_processing(self):
        """檢查是否正在處理訊息"""
        with self.process_lock:
            return self.processing
    
    def get_queue_size(self):
        """取得queue中待處理的訊息數量"""
        return self.message_queue.qsize()



def read_json_array(filename):
    """讀取包含多個 JSON 物件的陣列"""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            configs = json.load(file)  # 直接讀取陣列
        
        print(f"找到 {len(configs)} 個配置")
        for i, config in enumerate(configs):
            print(f"配置 {i+1}: {config['message_type']}")
        
        return configs
    except Exception as e:
        print(f"讀取失敗: {e}")
        return None


def carla_image_to_array(carla_image):
    """
    array[:, :, 0] = Blue
    array[:, :, 1] = Green
    array[:, :, 2] = Red
    array[:, :, 3] = Alpha
    """
    # 將CARLA的raw_data轉成numpy陣列
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))  

    # CARLA 是 BGRA，直接取前 3 個通道就是 BGR（OpenCV 格式）
    bgr_array = array[:, :, :3]

    return bgr_array

def carla_to_sionna_orientation(yaw, pitch, roll):
    """
    將CARLA的旋轉轉換為Sionna的方向
    
    Parameters:
    -----------
    yaw : float
        CARLA的yaw角度（度）
    pitch : float
        CARLA的pitch角度（度）
    roll : float
        CARLA的roll角度（度）
    
    Returns:
    --------
    list : [sionna_yaw, sionna_pitch, sionna_roll] in radians
    
    轉換規則（根據實測結果）：
    CARLA: pitch=-20.0, yaw=-110.0, roll=50.0
    Sionna: yaw=-1.9199, pitch=0.3491, roll=-0.8727
    """
    # Sionna的yaw = CARLA的yaw轉弧度
    sionna_yaw = np.radians(yaw)  # -110° → -1.9199 rad
    
    # Sionna的pitch = -CARLA的pitch轉弧度（符號相反）
    sionna_pitch = -np.radians(pitch)  # -(-20°) = 20° → 0.3491 rad
    
    # Sionna的roll = -CARLA的roll轉弧度（符號相反）
    sionna_roll = -np.radians(roll)  # -(50°) = -50° → -0.8727 rad
    
    return [float(sionna_yaw), float(sionna_pitch), float(sionna_roll)]