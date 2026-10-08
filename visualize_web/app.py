# app.py
from flask import Flask, render_template, jsonify, request
import json
import threading
import time
from datetime import datetime
from collections import deque
import base64
import io
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

app = Flask(__name__)

# 分別儲存 CARLA 和 Sionna 的圖片
carla_image_storage = deque(maxlen=2)  # 儲存最近 5 張 CARLA 圖片
sionna_image_storage = deque(maxlen=2)  # 儲存最近 5 張 Sionna 圖片
carla_lock = threading.Lock()
sionna_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

# CARLA 圖片相關 API
@app.route('/api/carla_images', methods=['GET'])
def get_carla_images():
    with carla_lock:
        return jsonify({
            'timestamps': [item['timestamp'] for item in carla_image_storage],
            'images': [item['image_base64'] for item in carla_image_storage],
            'latest_image': carla_image_storage[-1]['image_base64'] if carla_image_storage else None,
            'latest_timestamp': carla_image_storage[-1]['timestamp'] if carla_image_storage else None
        })

@app.route('/api/add_carla_image', methods=['POST'])
def add_carla_image():
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        timestamp = int(data.get('timestamp', int(time.time())))
        
        with carla_lock:
            carla_image_storage.append({
                'timestamp': timestamp,
                'image_base64': image_base64
            })
        
        return jsonify({'status': 'success', 'message': 'CARLA image added successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Sionna 圖片相關 API
@app.route('/api/sionna_images', methods=['GET'])
def get_sionna_images():
    with sionna_lock:
        return jsonify({
            'timestamps': [item['timestamp'] for item in sionna_image_storage],
            'images': [item['image_base64'] for item in sionna_image_storage],
            'latest_image': sionna_image_storage[-1]['image_base64'] if sionna_image_storage else None,
            'latest_timestamp': sionna_image_storage[-1]['timestamp'] if sionna_image_storage else None
        })

@app.route('/api/add_sionna_image', methods=['POST'])
def add_sionna_image():
    try:
        data = request.get_json()
        image_base64 = data['image_base64']
        timestamp = int(data.get('timestamp', int(time.time())))
        
        with sionna_lock:
            sionna_image_storage.append({
                'timestamp': timestamp,
                'image_base64': image_base64
            })
        
        return jsonify({'status': 'success', 'message': 'Sionna image added successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# # 將 matplotlib.figure.Figure 轉換為 base64 字串
# def figure_to_base64(fig):
#     """將 matplotlib.figure.Figure 轉換為 base64 字串"""
#     buffer = io.BytesIO()
#     fig.savefig(buffer, format='png', bbox_inches='tight', dpi=200)
#     buffer.seek(0)
#     image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
#     buffer.close()
#     return image_base64


def figure_to_base64(fig):
    """將 matplotlib.figure.Figure 或 PIL.Image 轉換為 base64 字串"""
    
    # 如果是 matplotlib Figure
    if isinstance(fig, plt.Figure):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return image_base64
    
    # 如果是 PIL Image
    elif isinstance(fig, Image.Image):
        buf = io.BytesIO()
        fig.save(buf, format='PNG')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64
    
    else:
        raise ValueError("輸入必須是 matplotlib.figure.Figure 或 PIL.Image.Image")

def send_figure_to_web(fig, target='sionna', timestamp=None, port=5080):
    """將 matplotlib.figure.Figure 傳送到指定環境
    
    Args:
        fig: matplotlib.figure.Figure 物件
        target: 'carla' 或 'sionna'，決定圖片顯示在左邊還是右邊
        timestamp: 時間步數（整數），如果為 None 則使用當前時間
        port: Flask 應用程式埠號
    """
    try:
        import requests
        
        # 轉換圖片為 base64
        image_base64 = figure_to_base64(fig)
        
        if timestamp is None:
            timestamp = int(time.time())
        
        payload = {
            'image_base64': image_base64,
            'timestamp': timestamp
        }
        
        # 根據目標選擇對應的 API 端點
        if target.lower() == 'carla':
            endpoint = f'http://localhost:{port}/api/add_carla_image'
            env_name = 'CARLA'
        elif target.lower() == 'sionna':
            endpoint = f'http://localhost:{port}/api/add_sionna_image'
            env_name = 'Sionna'
        else:
            raise ValueError("target 必須是 'carla' 或 'sionna'")
            
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            print(f"Successfully sent image to {env_name} (timestamp: {timestamp})")
        else:
            print(f"Failed to send image to {env_name}: {response.text}")
    except Exception as e:
        print(f"Error sending image to {target}: {e}")

# old
def send_to_web(data: float, timestamp: int = None, port=5080):
    """通過 HTTP API 將數據傳送到網站"""
    try:
        payload = {'value': data}
        if timestamp is not None:
            payload['timestamp'] = timestamp
            
        response = requests.post(f'http://localhost:{port}/api/add_data', 
                               json=payload)
        if response.status_code == 200:
            print(f"Successfully sent data: {data} (timestamp: {timestamp})")
        else:
            print(f"Failed to send data: {response.text}")
    except Exception as e:
        print(f"Error sending data: {e}")

# 產生測試圖片的函數
def create_carla_test_figure(timestamp):
    """產生 CARLA 測試圖片"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 模擬車輛路徑數據
    x = np.linspace(0, 10, 100)
    vehicle_path = np.sin(x + timestamp * 0.05) * 2 + 5
    
    ax.plot(x, vehicle_path, 'b-', linewidth=3, label='Vehicle Path')
    ax.fill_between(x, 0, 10, alpha=0.2, color='gray', label='Road')
    ax.set_title(f'CARLA Simulation - Vehicle Tracking (t={timestamp})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Lane Position (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    return fig

def create_sionna_test_figure(timestamp):
    """產生 Sionna 測試圖片"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 模擬信號強度熱力圖
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2)/4) * np.sin(timestamp * 0.1)
    
    im = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax.set_title(f'Sionna RF Coverage Map (t={timestamp})', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    plt.colorbar(im, ax=ax, label='Signal Strength (dBm)')
    
    plt.tight_layout()
    return fig

# 測試函數
def test():
    timestamp = 1
    while True:
        # 每 3 秒生成 CARLA 圖片
        if timestamp % 30 == 0:
            carla_fig = create_carla_test_figure(timestamp)
            send_figure_to_web(carla_fig, target='carla', timestamp=timestamp)
            plt.close(carla_fig)
        
        # 每 5 秒生成 Sionna 圖片
        if timestamp % 50 == 0:
            sionna_fig = create_sionna_test_figure(timestamp)
            send_figure_to_web(sionna_fig, target='sionna', timestamp=timestamp)
            plt.close(sionna_fig)
        
        timestamp += 1
        time.sleep(0.1)

if __name__ == '__main__':
    # 可以取消註解下面這行來測試圖片生成
    # threading.Thread(target=test, daemon=True).start()
    app.run(host="0.0.0.0", port=5080, debug=True, threaded=True)