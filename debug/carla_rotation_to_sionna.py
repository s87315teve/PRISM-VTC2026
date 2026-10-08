import numpy as np

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
    
    return [sionna_yaw, sionna_pitch, sionna_roll]

# 使用範例
if __name__ == "__main__":
    # CARLA的旋轉
    carla_yaw = -110.0
    carla_pitch = -20.0
    carla_roll = 50.0
    
    print("CARLA原始旋轉:")
    print(f"  Yaw: {carla_yaw}°")
    print(f"  Pitch: {carla_pitch}°")
    print(f"  Roll: {carla_roll}°")
    print()
    
    # 轉換
    sionna_rot = carla_to_sionna_orientation(carla_yaw, carla_pitch, carla_roll)
    
    print("Sionna方向 [yaw, pitch, roll] (弧度):")
    print(f"  Yaw: {sionna_rot[0]:.4f} rad ({np.degrees(sionna_rot[0]):.2f}°)")
    print(f"  Pitch: {sionna_rot[1]:.4f} rad ({np.degrees(sionna_rot[1]):.2f}°)")
    print(f"  Roll: {sionna_rot[2]:.4f} rad ({np.degrees(sionna_rot[2]):.2f}°)")
    print()
    
    # 驗證
    expected_yaw = -1.9199
    expected_pitch = 0.3491
    expected_roll = -0.8727
    
    print("驗證結果:")
    print(f"  Yaw 誤差: {abs(sionna_rot[0] - expected_yaw):.6f} rad")
    print(f"  Pitch 誤差: {abs(sionna_rot[1] - expected_pitch):.6f} rad")
    print(f"  Roll 誤差: {abs(sionna_rot[2] - expected_roll):.6f} rad")