import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def calculate_path_gain(frame_data):
    """
    Calculates the path gain in dB from the amplitude values 'a' in the frame data.
    
    Args:
        frame_data (dict): Dictionary containing the loaded pickle data with key 'a'.
    
    Returns:
        float: Path gain in dB, or None if 'a' is not present or calculation fails.
    """
    if frame_data is None:
        return None
    
    if 'a' not in frame_data:
        print("Warning: 'a' key not found in frame data")
        return None
    
    try:
        amplitudes = np.array(frame_data['a'])
        # Calculate total power: sum of squared magnitudes
        power = np.sum(np.abs(amplitudes)**2)
        
        # Avoid log of zero
        if power <= 0:
            print("Warning: Power is zero or negative, cannot calculate dB")
            return None
        
        # Convert to dB
        gain_db = 10 * np.log10(power)
        return gain_db
    except Exception as e:
        print(f"Error calculating path gain: {e}")
        return None

def read_prism_frames(start_frame, end_frame):
    """
    Reads .pkl files for a specified range of frames from multiple baseline and beamselection scenarios.

    Args:
        start_frame (int): The starting frame number (inclusive).
        end_frame (int): The ending frame number (inclusive).

    Returns:
        dict: A dictionary where keys are frame numbers and values are dictionaries containing
              'baseline1', 'baseline2', 'baseline3', 'beamselection' data and their respective path gains in dB.
    """
    
    # Base directories
    base_dir = r"../dataset"
    
    # 定義多個 baseline 的路徑模板
    baseline_configs = {
        'baseline1': {
            'path': os.path.join(base_dir, "scenario_baseline_turn_28G_beam1_v2", "channel", "frame_{:05d}", "rsu-rx_51", "tx_ant0-rx_ant0_frame{:05d}.pkl"),
            'name': 'Beam 1 only'
        },
        'baseline2': {
            'path': os.path.join(base_dir, "scenario_baseline_turn_28G_beam2_v2", "channel", "frame_{:05d}", "rsu-rx_123", "tx_ant0-rx_ant0_frame{:05d}.pkl"),
            'name': 'Beam 2 only'
        },
        'baseline3': {
            'path': os.path.join(base_dir, "scenario_baseline_turn_28G_beam3_v2", "channel", "frame_{:05d}", "rsu-rx_153", "tx_ant0-rx_ant0_frame{:05d}.pkl"),
            'name': 'Beam 3 only'
        }
    }
    
    beamselection_path_template = os.path.join(base_dir, "scenario_beamselection_turn_28G_v2", "channel", "frame_{:05d}", "rsu-rx_195", "tx_ant0-rx_ant0_frame{:05d}.pkl")
    
    results = {}

    for relative_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        frame_data = {}
        
        # Read all baselines
        for baseline_key, baseline_config in baseline_configs.items():
            baseline_file = baseline_config['path'].format(frame_idx, frame_idx)
            
            if os.path.exists(baseline_file):
                try:
                    with open(baseline_file, 'rb') as f:
                        frame_data[baseline_key] = pickle.load(f)
                        print(f"Loaded {baseline_config['name']} Frame {frame_idx}")
                        
                        # Calculate path gain
                        gain_db = calculate_path_gain(frame_data[baseline_key])
                        frame_data[f'{baseline_key}_gain_db'] = gain_db
                        if gain_db is not None:
                            print(f"  {baseline_config['name']} Path Gain: {gain_db:.2f} dB")
                except Exception as e:
                    print(f"Error reading {baseline_config['name']} Frame {frame_idx}: {e}")
                    frame_data[baseline_key] = None
                    frame_data[f'{baseline_key}_gain_db'] = None
            else:
                print(f"{baseline_config['name']} file missing for Frame {frame_idx}: {baseline_file}")
                frame_data[baseline_key] = None
                frame_data[f'{baseline_key}_gain_db'] = None

        # Read BeamSelection
        beamselection_file = beamselection_path_template.format(frame_idx, frame_idx)
        if os.path.exists(beamselection_file):
            try:
                with open(beamselection_file, 'rb') as f:
                    frame_data['beamselection'] = pickle.load(f)
                    print(f"Loaded BeamSelection Frame {frame_idx}")
                    
                    # Calculate path gain
                    gain_db = calculate_path_gain(frame_data['beamselection'])
                    frame_data['beamselection_gain_db'] = gain_db
                    if gain_db is not None:
                        print(f"  BeamSelection Path Gain: {gain_db:.2f} dB")
            except Exception as e:
                print(f"Error reading BeamSelection Frame {frame_idx}: {e}")
                frame_data['beamselection'] = None
                frame_data['beamselection_gain_db'] = None
        else:
            print(f"BeamSelection file missing for Frame {frame_idx}: {beamselection_file}")
            frame_data['beamselection'] = None
            frame_data['beamselection_gain_db'] = None
            
        results[relative_idx] = frame_data

    return results

def plot_path_gain(data, save_path=None):
    """
    Plots path gain vs frame number for baseline and beamselection scenarios.
    
    Args:
        data (dict): Dictionary returned from read_prism_frames.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    frames = sorted(data.keys())
    baseline_gains = []
    beamselection_gains = []
    
    for frame in frames:
        baseline_gain = data[frame].get('baseline_gain_db')
        beamselection_gain = data[frame].get('beamselection_gain_db')
        
        baseline_gains.append(baseline_gain)
        beamselection_gains.append(beamselection_gain)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot baseline
    plt.plot(frames, baseline_gains, marker='o', linestyle='--', linewidth=2,
         markersize=4, label='Baseline', color='#E8672A')
    
    # Plot beamselection
    plt.plot(frames, beamselection_gains, marker='o', linestyle='-', linewidth=2,
         markersize=4, label='Beam Selection', color='#1B7FA3')
    
    plt.xlabel('Frame', fontsize=12, fontweight='bold')
    plt.ylabel('Path Gain (dB)', fontsize=12, fontweight='bold')
    plt.title('Path Gain vs Frame Number', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def plot_path_gain_averaged(data, fps=10.0, window_size=10, save_path=None):
    """
    Plots averaged path gain vs time (in seconds) with averaging window.
    
    Args:
        data (dict): Dictionary returned from read_prism_frames.
        fps (float): Frames per second (default: 10.0).
        window_size (int): Number of frames to average (default: 10).
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    frames = sorted(data.keys())
    baseline_gains = []
    beamselection_gains = []
    
    for frame in frames:
        baseline_gain = data[frame].get('baseline_gain_db')
        beamselection_gain = data[frame].get('beamselection_gain_db')
        baseline_gains.append(baseline_gain)
        beamselection_gains.append(beamselection_gain)
    
    # Average every window_size frames
    averaged_frames = []
    averaged_baseline = []
    averaged_beamselection = []
    
    for i in range(0, len(frames), window_size):
        window_frames = frames[i:i+window_size]
        window_baseline = [g for g in baseline_gains[i:i+window_size] if g is not None]
        window_beamselection = [g for g in beamselection_gains[i:i+window_size] if g is not None]
        
        if window_baseline and window_beamselection:
            # Use the middle frame of the window for time calculation
            avg_frame = sum(window_frames) / len(window_frames)
            averaged_frames.append(avg_frame)
            averaged_baseline.append(np.mean(window_baseline))
            averaged_beamselection.append(np.mean(window_beamselection))
    
    # Convert frames to time (seconds)
    time_seconds = [f / fps for f in averaged_frames]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot baseline
    plt.plot(time_seconds, averaged_baseline, marker='^', linestyle='--', linewidth=2, 
             markersize=4, label='Baseline (Averaged)', color='#E8672A')
    
    # Plot beamselection
    plt.plot(time_seconds, averaged_beamselection, marker='o', linestyle='-', linewidth=2, 
             markersize=4, label='Beam Selection (Averaged)', color='#1B7FA3')
    
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Path Gain (dB)', fontsize=12, fontweight='bold')
    plt.title(f'Averaged Path Gain vs Time (Window: {window_size} frames)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def plot_path_gain_difference(data, save_path=None):
    """
    Plots the difference in path gain (Beam Selection - Baseline) vs frame number.
    
    Args:
        data (dict): Dictionary returned from read_prism_frames.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    frames = sorted(data.keys())
    gain_differences = []
    
    for frame in frames:
        baseline_gain = data[frame].get('baseline_gain_db')
        beamselection_gain = data[frame].get('beamselection_gain_db')
        
        if baseline_gain is not None and beamselection_gain is not None:
            # Positive value means beam selection is better (higher gain)
            difference = beamselection_gain - baseline_gain
            gain_differences.append(difference)
        else:
            gain_differences.append(None)
    
    # Filter out None values
    valid_frames = [f for f, d in zip(frames, gain_differences) if d is not None]
    valid_differences = [d for d in gain_differences if d is not None]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot difference
    plt.plot(valid_frames, valid_differences, marker='o', linestyle='-', linewidth=2, 
             markersize=4, label='Beam Selection - Baseline', color='#F18F01')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.xlabel('Frame', fontsize=12, fontweight='bold')
    plt.ylabel('Path Gain Difference (dB)', fontsize=12, fontweight='bold')
    plt.title('Path Gain Difference: Beam Selection vs Baseline', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_path_gain_averaged_and_mark_beamselection(data, fps=10.0, window_size=10, save_path=None, xlim=None, ylim=None):
    """
    Plots averaged path gain vs time (in seconds) with averaging window for multiple baselines.
    """
    frames = sorted(data.keys())
    
    # 準備多個 baseline 的資料
    baseline1_gains = []
    baseline2_gains = []
    baseline3_gains = []
    beamselection_gains = []
    
    for frame in frames:
        baseline1_gains.append(data[frame].get('baseline1_gain_db'))
        baseline2_gains.append(data[frame].get('baseline2_gain_db'))
        baseline3_gains.append(data[frame].get('baseline3_gain_db'))
        beamselection_gains.append(data[frame].get('beamselection_gain_db'))
    
    # Average every window_size frames
    averaged_frames = []
    averaged_baseline1 = []
    averaged_baseline2 = []
    averaged_baseline3 = []
    averaged_beamselection = []
    
    for i in range(0, len(frames), window_size):
        window_frames = frames[i:i+window_size]
        window_baseline1 = [g for g in baseline1_gains[i:i+window_size] if g is not None]
        window_baseline2 = [g for g in baseline2_gains[i:i+window_size] if g is not None]
        window_baseline3 = [g for g in baseline3_gains[i:i+window_size] if g is not None]
        window_beamselection = [g for g in beamselection_gains[i:i+window_size] if g is not None]
        
        if window_baseline1 or window_baseline2 or window_baseline3 or window_beamselection:
            avg_frame = sum(window_frames) / len(window_frames)
            averaged_frames.append(avg_frame)
            
            averaged_baseline1.append(np.mean(window_baseline1) if window_baseline1 else None)
            averaged_baseline2.append(np.mean(window_baseline2) if window_baseline2 else None)
            averaged_baseline3.append(np.mean(window_baseline3) if window_baseline3 else None)
            averaged_beamselection.append(np.mean(window_beamselection) if window_beamselection else None)
    
    # Convert frames to time (seconds)
    time_seconds = [f / fps for f in averaged_frames]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot all baselines
    if any(averaged_baseline1):
        plt.plot(time_seconds, averaged_baseline1, marker='^', linestyle='--', linewidth=2, 
                 markersize=4, label='Beam 1 only', color='red')
    
    if any(averaged_baseline2):
        plt.plot(time_seconds, averaged_baseline2, marker='s', linestyle='--', linewidth=2, 
                 markersize=4, label='Beam 2 only', color='green')
    
    if any(averaged_baseline3):
        plt.plot(time_seconds, averaged_baseline3, marker='d', linestyle='--', linewidth=2, 
                 markersize=4, label='Beam 3 only', color='blue')
    
    # Plot beamselection
    if any(averaged_beamselection):
        plt.plot(time_seconds, averaged_beamselection, marker='o', linestyle='-', linewidth=2, 
                 markersize=4, label='External control module', color='purple')

     # 設定座標軸範圍（如果有指定的話）
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # ===== 在 15 秒處畫分隔線 =====
    division_time = 15
    plt.axvline(x=division_time, color='black', linestyle='--', linewidth=2, alpha=0.6)
    
    # 獲取 y 軸範圍以便放置文字
    y_min, y_max = plt.ylim()
    y_mid = (y_min + y_max) / 2  # 中間高度
    
    # 左側文字標註（15秒之前）
    plt.text(division_time - 1, -135, 'Without beam selection', 
            ha='right', va='center', fontsize=14, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', edgecolor='black', alpha=0.9))
    
    # 右側文字標註（15秒之後）
    plt.text(division_time + 1, -135, 'With beam selection', 
            ha='left', va='center', fontsize=14, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', edgecolor='black', alpha=0.9))

    # 標示 beam 切換時間點 (17s 和 23s)
    beam_switch_times = [17, 23]
    text_x, text_y = 10, -98 # 第一個切換點的文字位置
    plt.text(text_x, text_y, f'Defult\nUsing Beam 1', 
        ha='center', va='center', fontsize=14, color='purple', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='purple', alpha=0.8))
    for switch_time in beam_switch_times:
        # 找到最接近切換時間的數據點
        idx = min(range(len(time_seconds)), key=lambda i: abs(time_seconds[i] - switch_time))
        x_pos = time_seconds[idx]
        y_pos = averaged_beamselection[idx]
        
        # 畫短的垂直線（只在數據點附近）
        line_height = 5  # 線的高度範圍 (dB)
        plt.plot([x_pos, x_pos], [y_pos - line_height/2, y_pos + line_height/2], 
                color='#ff4500', linestyle='-', linewidth=2.5, alpha=0.8)
        
        # 手動指定文字位置（你可以調整這些座標）
        if switch_time == 17:
            text_x, text_y = 20.5, -97 # 第一個切換點的文字位置
            plt.text(text_x, text_y, f'Switch to\nBeam 2', 
                ha='center', va='center', fontsize=14, color='purple', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='purple', alpha=0.8))
        else:  # switch_time == 23
            text_x, text_y = 26.5, -97  # 第二個切換點的文字位置
            plt.text(text_x, text_y, f'Switch to\nBeam 3', 
                ha='center', va='center', fontsize=14, color='purple', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='purple', alpha=0.8))
        
        
    
       
    
    plt.xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Path Gain (dB)', fontsize=14, fontweight='bold')
    plt.title(f'Average Path Gain vs Virtual Simulation Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    # Test frames
    data = read_prism_frames(10, 310)
    print("Execution complete.")
    plot_path_gain_averaged_and_mark_beamselection(data, fps=10.0, window_size=10, ylim=(-152, -93))
    # Plot the results
    # plot_path_gain(data)
    # plot_path_gain_averaged(data, fps=10.0, window_size=10)
    # plot_path_gain_difference(data)