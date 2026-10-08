import carla
import cv2
import numpy as np
import time
import os
import subprocess
from datetime import datetime
from collections import deque

# ========== 參數設定 ==========
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
CAM_FOV = 90
FPS = 30  # 預設30FPS
FRAME_INTERVAL = 1.0 / FPS

MOVE_SPEED = 1.0  # 每次移動的距離
ROTATE_SPEED = 5.0  # 每次旋轉的角度

# ========== FPS 計算相關 ==========
fps_buffer = deque(maxlen=30)  # 保存最近30幀的時間戳記
last_time = time.time()

# ========== 連接CARLA ==========
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 設定固定時間步長
settings = world.get_settings()
# settings.fixed_delta_seconds = FRAME_INTERVAL  # 每frame間隔
# settings.synchronous_mode = True  # 啟動同步模式（建議做即時控制/錄影必開）
# world.apply_settings(settings)

spectator = world.get_spectator()

# ========== 創建攝影機 ==========
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
camera_bp.set_attribute('fov', str(CAM_FOV))

camera_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=spectator)

image_data = {'frame': None}

def camera_callback(image, data_dict):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # 去alpha
    data_dict['frame'] = array
    # data_dict['frame'] = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

camera.listen(lambda image: camera_callback(image, image_data))

# ========== 錄影相關 (ffmpeg pipe方式) ==========
recording = False
ffmpeg_proc = None
output_filename = None

def toggle_recording(size):
    global recording, ffmpeg_proc, output_filename
    if recording:
        # 停止錄影
        recording = False
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            ffmpeg_proc = None
        print(f"停止錄影，已儲存：{output_filename}")
    else:
        # 開始錄影
        video_dir = "video"
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        output_filename = os.path.join(video_dir, f"{ts}.mp4")

        # ffmpeg pipe 命令 (軟體H.264編碼)
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',           # OpenCV預設是BGR
            '-s', f'{size[0]}x{size[1]}',
            '-r', str(FPS),
            '-i', '-',                     # stdin輸入
            '-an',                         # 無音訊
            '-vcodec', 'h264_nvenc',       # NVIDIA硬體加速
            '-pix_fmt', 'yuv420p',
            '-preset', 'fast',
            '-crf', '23',
            output_filename
        ]
        
        # 如要硬體加速 (NVIDIA)，可改成：
        # '-vcodec', 'h264_nvenc', '-preset', 'p3', '-b:v', '5M',

        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        recording = True
        print(f"開始錄影(pipe to ffmpeg)：{output_filename}")

# ========== 畫面與操作說明 ==========
cv2.namedWindow('CARLA Spectator View', cv2.WINDOW_NORMAL)
cv2.resizeWindow('CARLA Spectator View', IMAGE_WIDTH, IMAGE_HEIGHT)

print("="*60)
print("CARLA 觀察者即時監控系統")
print("="*60)
print("控制說明：")
print("  W/S/A/D - 基本移動")
print("  Q/E     - 上下移動")
print("  ↑/↓/←/→ - 視角旋轉")
print("  R       - 錄影開關")
print("  ESC     - 退出程式")
print("="*60)

def print_transform(transform):
    loc = transform.location
    rot = transform.rotation
    print(f"\n位置: x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")
    print(f"角度: pitch={rot.pitch:.2f}°, yaw={rot.yaw:.2f}°, roll={rot.roll:.2f}°")
    print("-"*60)

try:
    while True:
        # 計算 FPS
        current_time = time.time()
        fps_buffer.append(current_time)
        if len(fps_buffer) > 1:
            fps_value = len(fps_buffer) / (fps_buffer[-1] - fps_buffer[0])
        else:
            fps_value = 0.0
        # 等待有資料
        if image_data['frame'] is not None:
            # 取得原始畫面（BGR格式）
            frame_original = image_data['frame'].copy()
            
            # 錄影：使用不含文字的原始畫面
            if recording and ffmpeg_proc:
                try:
                    ffmpeg_proc.stdin.write(frame_original.tobytes())
                except BrokenPipeError:
                    print("ffmpeg pipe 已關閉")
                    recording = False
            
            # 顯示：在副本上添加文字和錄影標記
            frame_display = frame_original.copy()
            transform = spectator.get_transform()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_display, f"X: {transform.location.x:.2f} Y: {transform.location.y:.2f} Z: {transform.location.z:.2f}",
                        (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Pitch: {transform.rotation.pitch:.2f}  Yaw: {transform.rotation.yaw:.2f}  Roll: {transform.rotation.roll:.2f}",
                        (10, 60), font, 0.7, (0, 255, 0), 2)
            # FPS 顯示
            cv2.putText(frame_display, f"FPS: {fps_value:.1f}",
                        (10, 90), font, 0.7, (0, 255, 255), 2)
            if recording:
                cv2.putText(frame_display, 'REC', (IMAGE_WIDTH-80, 40), font, 1.2, (0,0,255), 3)
            
            cv2.imshow('CARLA Spectator View', frame_display)

        # 指令輸入
        key = cv2.waitKey(1)
        key_ascii = key & 0xFF

        if key_ascii == 27:  # ESC
            print("\n退出程式...")
            break
        elif key_ascii == ord('r'):
            toggle_recording((IMAGE_WIDTH, IMAGE_HEIGHT))

        # ---- 觀察者移動 & 旋轉 ----
        transform = spectator.get_transform()
        location = transform.location
        rotation = transform.rotation

        # 前/右方向計算
        yaw_rad = np.radians(rotation.yaw)
        forward_x = np.cos(yaw_rad)
        forward_y = np.sin(yaw_rad)
        right_x = np.cos(yaw_rad + np.pi/2)
        right_y = np.sin(yaw_rad + np.pi/2)
        moved = False

        # wasd & qe
        if key_ascii == ord('w'):
            location.x += forward_x * MOVE_SPEED
            location.y += forward_y * MOVE_SPEED
            moved = True
            print("向前移動")
        elif key_ascii == ord('s'):
            location.x -= forward_x * MOVE_SPEED
            location.y -= forward_y * MOVE_SPEED
            moved = True
            print("向後移動")
        elif key_ascii == ord('a'):
            location.x -= right_x * MOVE_SPEED
            location.y -= right_y * MOVE_SPEED
            moved = True
            print("向左移動")
        elif key_ascii == ord('d'):
            location.x += right_x * MOVE_SPEED
            location.y += right_y * MOVE_SPEED
            moved = True
            print("向右移動")
        elif key_ascii == ord('q'):
            location.z += MOVE_SPEED
            moved = True
            print("向上移動")
        elif key_ascii == ord('e'):
            location.z -= MOVE_SPEED
            moved = True
            print("向下移動")

        # 方向鍵（OpenCV回傳：左81，上82，右83，下84）
        elif key == 82:  # 上箭頭
            rotation.pitch += ROTATE_SPEED
            moved = True
            print("向上旋轉")
        elif key == 84:  # 下箭頭
            rotation.pitch -= ROTATE_SPEED
            moved = True
            print("向下旋轉")
        elif key == 81:  # 左箭頭
            rotation.yaw -= ROTATE_SPEED
            moved = True
            print("向左旋轉")
        elif key == 83:  # 右箭頭
            rotation.yaw += ROTATE_SPEED
            moved = True
            print("向右旋轉")

        if moved:
            new_transform = carla.Transform(location, rotation)
            spectator.set_transform(new_transform)
            print_transform(new_transform)

        # 呼叫tick推進同步世界
        world.tick()

finally:
    # 清理資源
    if recording and ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
    camera.stop()
    camera.destroy()
    cv2.destroyAllWindows()
    # 關閉同步模式，恢復自由步長
    settings = world.get_settings()
    # settings.synchronous_mode = False
    world.apply_settings(settings)
    print("\n資源已清理完成")