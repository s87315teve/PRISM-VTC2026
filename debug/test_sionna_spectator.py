

import sionna
from sionna.rt import Camera, load_scene
import mitsuba as mi
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ========== 參數設定 ==========
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
NUM_SAMPLES = 128

MOVE_SPEED = 10.0
ROTATE_SPEED = 5.0

# ========== 載入場景 ==========
scene = sionna.rt.load_scene("../my_scene/nycu_carla_v7_add_plane_yf_mirror/nycu_carla_v7_add_plane_yf_mirror.xml")  # 或使用 sionna.rt.Scene()


# ========== 相機狀態 ==========
cam_pos = [340.0, 80.0, 12.0]
cam_rot = [-1.9199, 0.3491, -0.8727]  # [yaw, pitch, roll] in radians



def render_view():
    """渲染當前視角"""
    pos = [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])]
    rot = [float(cam_rot[0]), float(cam_rot[1]), float(cam_rot[2])]
    cam = Camera(position=pos, orientation=rot)
    fig = scene.render(camera=cam, resolution=[IMAGE_WIDTH, IMAGE_HEIGHT], num_samples=NUM_SAMPLES, fov=120)
    
    # 轉換 Figure 為 numpy array
    # 方法2: 使用 buffer_rgba
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape((h, w, 4))
    plt.close(fig)
    
    # 轉 BGR for OpenCV
    img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    
    # 左右翻轉
    img = cv2.flip(img, 1)  # 1 表示水平翻轉
    return img

# ========== 主迴圈 ==========
print("控制: WASD=移動 QE=上下 方向鍵=旋轉 ESC=退出")

cv2.namedWindow('Sionna View', cv2.WINDOW_NORMAL)

try:
    while True:
        # 渲染
        frame = render_view()
        
        # 顯示資訊
        cv2.putText(frame, f"Pos: [{cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Rot: [{np.degrees(cam_rot[0]):.1f}, {np.degrees(cam_rot[1]):.1f}, {np.degrees(cam_rot[2]):.1f}]",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Sionna View', frame)
        
        # 按鍵控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        
        # 移動 (WASD)
        if key == ord('w'):
            cam_pos[1] += MOVE_SPEED
        elif key == ord('s'):
            cam_pos[1] -= MOVE_SPEED
        elif key == ord('a'):
            cam_pos[0] -= MOVE_SPEED
        elif key == ord('d'):
            cam_pos[0] += MOVE_SPEED
        elif key == ord('q'):
            cam_pos[2] += MOVE_SPEED
        elif key == ord('e'):
            cam_pos[2] -= MOVE_SPEED
        
        # 旋轉 (方向鍵)
        elif key == 82:  # Up
            cam_rot[1] += np.radians(ROTATE_SPEED)
        elif key == 84:  # Down
            cam_rot[1] -= np.radians(ROTATE_SPEED)
        elif key == 81:  # Left
            cam_rot[0] -= np.radians(ROTATE_SPEED)
        elif key == 83:  # Right
            cam_rot[0] += np.radians(ROTATE_SPEED)

finally:
    cv2.destroyAllWindows()