import pickle
import traceback
import os # Configure which GPU
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    raise e
              

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMaterial, Camera, \
                        PathSolver, RadioMapSolver, subcarrier_frequencies, \
                        LambertianPattern, DirectivePattern, BackscatteringPattern, \
                        ITURadioMaterial, SceneObject
from sionna.rt.utils import r_hat
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.utils import count_block_errors, ebnodb2no, PlotBER
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
                            time_lag_discrete_time_channel, ApplyOFDMChannel, ApplyTimeChannel, \
                            OFDMChannel, TimeChannel
from sionna.phy import Block
import mitsuba as mi

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

import numpy as np
from PIL import Image

# For plotting
# %matplotlib inline
# also try %matplotlib widget

import matplotlib.pyplot as plt

import socket
import json
import threading
import time
from typing import Dict, Any, Callable
from utils import NetworkComponent, read_json_array, carla_to_sionna_orientation
from commu_modular import LDPC_system
# from visualize_web.app import send_to_web, send_figure_to_web
# from visualize_web.app_v1.app import send_figure_to_web, send_to_web
from visualize_web.app import  send_figure_to_web
from dataset_writer import DatasetWriter
import custom_antenna.custom_antenna_pattern

send_to_web_flag=False
enable_paths=True
enable_render=False

# 無線環境模擬器 (Port: 5001)
class WirelessEnvironment(NetworkComponent):
    def __init__(self):
        super().__init__('wireless_env', 5001)
        self.simulation_params = {}
        
        # 註冊訊息處理器
        self.register_handler('initialize', self._initialize_wireless_env)
        self.register_handler('run', self._run_simulation)
        self.register_handler('save_data', self._save_data)
        self.register_handler('create_object', self._create_object)
        self.register_handler('update_object', self._update_object)
        self.register_handler('spectator_render', self._spectator_render)
        self.register_handler('set_dataset_writer', self._set_dataset_writer)
        self.register_handler('reset_start_time', self._reset_start_time)
        self.register_handler('get_current_info', self._get_current_info)
        self.register_handler('update_antenna_pattern', self._update_antenna_pattern)


        
        # 啟動網路監聽
        self.start_listening()

        self.car_material = ITURadioMaterial("car-material",
                                "metal",
                                thickness=0.01,
                                color=(0.8, 0.1, 0.1))
        self.ego_car_material = ITURadioMaterial("car-material",
                                "metal",
                                thickness=0.01,
                                color=(0.1, 0.8, 0.1))
        

    
    def _initialize_wireless_env(self, message):
        print("Current Working Directory:", os.getcwd())
        print(f"[{self.component_name}] 收到初始化訊息")
        print(f"當前執行緒: {threading.current_thread().name}")
        print(f"是否為主執行緒: {threading.current_thread() == threading.main_thread()}")
        print(f"[{self.component_name}] _initialize_wireless_env 開始執行")
        if "wireless_environment" in message["data"]:
            print("load scene...")
            scene_path = message["data"]["wireless_environment"]["scene_name"]
            print(f"scene_path: {scene_path}")
            self.scene = sionna.rt.load_scene(scene_path)
            # self.scene = sionna.rt.load_scene(sionna.rt.scene.etoile)
            print("load scene success")
            
            print("set material...")
            self.scene.frequency = message["data"]["wireless_environment"]["frequency"]
            new_material = sionna.rt.RadioMaterial(f"my_material_{self.scene.frequency}",
                                        relative_permittivity=5.24,
                                        conductivity=0.0462*(self.scene.frequency/1e9)**0.7822,
                                        scattering_pattern='lambertian')
            for i, obj in enumerate(self.scene.objects.values()):
                self.scene.get(obj.name).radio_material=new_material
            print("set material success")
        else:
            print("\"wireless_environment\" is not in config")
        
        print("set wireless parameters...")
        # PHY層參數
        self.batch_size = message["data"]["wireless_params"]["batch_size"]
        self.num_streams_per_tx = message["data"]["wireless_params"]["num_streams_per_tx"]
        self.num_bits_per_symbol = message["data"]["wireless_params"]["num_bits_per_symbol"]
        self.num_ofdm_symbol = message["data"]["wireless_params"]["num_ofdm_symbol"]
        self.fft_size = message["data"]["wireless_params"]["fft_size"]
        self.coderate = message["data"]["wireless_params"]["coderate"]
        self.tx_power = message["data"]["wireless_params"]["tx_power"] # dBm
        self.noise = message["data"]["wireless_params"]["noise"] # dBm
        self.subcarrier_spacing = message["data"]["wireless_params"]["subcarrier_spacing"]
        self.num_tx = message["data"]["wireless_params"]["num_tx"]
        self.tx_list = message["data"]["wireless_params"]["tx_list"]
        self.rx_list = message["data"]["wireless_params"]["rx_list"]
        
        self.tx_pattern = message["data"]["wireless_params"].get("tx_pattern", "iso")
        self.rx_pattern = message["data"]["wireless_params"].get("rx_pattern", "iso")
        self.tx_num_rows = message["data"]["wireless_params"].get("tx_num_rows", 1)
        self.tx_num_cols = message["data"]["wireless_params"].get("tx_num_cols", 1)
        self.rx_num_rows = message["data"]["wireless_params"].get("rx_num_rows", 1)
        self.rx_num_cols = message["data"]["wireless_params"].get("rx_num_cols", 1)

        self.rg = ResourceGrid(num_ofdm_symbols=self.num_ofdm_symbol,
                  fft_size=self.fft_size,
                  subcarrier_spacing=self.subcarrier_spacing,
                  num_tx=self.num_tx,
                  num_streams_per_tx=self.num_streams_per_tx,
                  cyclic_prefix_length=6,
                  num_guard_carriers=[5,6],
                  dc_null=True,
                  pilot_pattern="kronecker",
                  pilot_ofdm_symbol_indices=[2,11])
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        self.n=int(self.rg.num_data_symbols*self.num_bits_per_symbol)  # total bits
        self.k=int(self.n*self.coderate) # info bits
        self.rg_mapper = sionna.phy.ofdm.ResourceGridMapper(self.rg)
        self.rx_tx_association = np.array([[1]])
        self.stream_management = sionna.phy.mimo.StreamManagement(self.rx_tx_association, self.num_streams_per_tx)
        self.commu_sys=LDPC_system(num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate, k=self.k, n=self.n, 
                                   rg=self.rg, stream_management=self.stream_management)
        print("set wireless parameters success")
        print("set ray tracing object...")
        self.scene.tx_array = self.load_antenna_pattern(num_rows=self.tx_num_rows,
                             num_cols=self.tx_num_cols,
                             pattern=self.tx_pattern)


        self.scene.rx_array = self.load_antenna_pattern(num_rows=self.rx_num_rows,
                             num_cols=self.rx_num_cols,
                             pattern=self.rx_pattern)
        for tx_name in self.tx_list:
            position = [ self.tx_list[tx_name]["x_pos"], self.tx_list[tx_name]["y_pos"], self.tx_list[tx_name]["z_pos"] ]
            orientation=[0, 0, 0]
            if ("yaw_carla" in self.tx_list[tx_name]) and ("pitch_carla" in self.tx_list[tx_name]) and ("roll_carla" in self.tx_list[tx_name]):
                carla_orientation = [ self.tx_list[tx_name]["yaw_carla"], self.tx_list[tx_name]["pitch_carla"], self.tx_list[tx_name]["roll_carla"] ]
                orientation = carla_to_sionna_orientation(carla_orientation[0], carla_orientation[1], carla_orientation[2])
            self.scene.add(sionna.rt.Transmitter(name=tx_name, position=position, orientation=orientation, display_radius=1))
        # for rx_name in self.rx_list:
        #     position = [ self.rx_list[rx_name]["x_pos"], self.rx_list[rx_name]["y_pos"], self.rx_list[rx_name]["z_pos"] ]
        #     self.scene.add(sionna.rt.Receiver(name=rx_name, position=position, display_radius=1))
        self.p_solver = sionna.rt.PathSolver()
        print("set ray tracing object success")

        self.ber_list=[]
        self.snr_list=[]
        self.car_dict={}
        self.create_count=0
        self.enable_dataset_writer=False
        self.addictional_tx_antenna_info=""

    def _get_current_info(self, message):
        data={}
        self.send_message(
                target_component=f"{message['from']}", 
                message_type="current_info", 
                data=data,
                to_scheduler=True
        )
        pass

    def _set_dataset_writer(self, message):
        """ message["data"] example 
        {
            "target": "wireless_env",
            "message_type": "set_dataset_writer",
            "dataset_root":"./dataset",
            "scenario_id":"scenario_001",
            "queue_size":100,
            "num_workers":4
            
        }
        """
        self.enable_dataset_writer=True
        self.dataset_root=message["data"]["dataset_root"]
        self.scenario_id=message["data"]["scenario_id"]
        self.writer_queue_size=message["data"]["queue_size"]
        self.writer_num_workers=message["data"]["num_workers"]
        if self.enable_dataset_writer==True:
            # set camera writer
            self.channel_writer = DatasetWriter(
                dataset_root=self.dataset_root,
                scenario_id=self.scenario_id,
                entity_type="channel",
                queue_size=self.writer_queue_size,
                num_workers=self.writer_num_workers
            )

            self.bs_datawriter = DatasetWriter(
                dataset_root=self.dataset_root,
                scenario_id=self.scenario_id,
                entity_type="basestation",
                entity_id="rsu",
                queue_size=self.writer_queue_size,
                num_workers=self.writer_num_workers
            )

            self.trajectory_datawriter = DatasetWriter(
                dataset_root=self.dataset_root,
                scenario_id=self.scenario_id,
                entity_type="vehicle",
                entity_id="vehicles",
                queue_size=self.writer_queue_size,
                num_workers=self.writer_num_workers
            )

        self.sys_print("Enable dataset writer")
    
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

    def load_antenna_pattern(self, num_rows=1, num_cols=1, pattern="iso"):
        if pattern=="elliptical_beam":
            return PlanarArray(num_rows=num_rows, num_cols=num_cols, 
                    pattern="elliptical_beam",
                    theta_0=90,           # 主波束水平
                    phi_0=0.0,              # 方位角 0°
                    theta_beamwidth=30.0,   # theta 方向窄
                    phi_beamwidth=30.0) 
        else:  
            return PlanarArray(num_rows=num_rows,
                             num_cols=num_cols,
                             pattern=pattern,
                             polarization="V")
    def _update_antenna_pattern(self, message):
        pattern=message["data"]["pattern"]
        
        if pattern=="elliptical_beam":
            steering_angle=message["data"]["steering_angle"]
            num_rows = message["data"].get("num_rows", self.tx_num_rows)
            num_cols = message["data"].get("num_cols", self.tx_num_cols)
            self.scene.tx_array=PlanarArray(num_rows=num_rows, num_cols=num_cols, 
                    pattern="elliptical_beam",
                    theta_0=90,           # 主波束水平
                    phi_0=steering_angle,  # 方位角 
                    theta_beamwidth=30.0,   # theta 方向窄
                    phi_beamwidth=30.0) 
            self.addictional_tx_antenna_info=f"steering_angle={steering_angle}"


    def save_paths_info(self, timestamp, scene, paths):
        a, tau = paths.cir(out_type="numpy", normalize_delays=False)
        num_rx_antennas=paths.targets.shape[1]//paths.num_rx
        num_tx_antennas=paths.sources.shape[1]//paths.num_tx
        print(f"num rx:{paths.num_rx}, num_rx_antennas:{num_rx_antennas}")
        print(f"num tx:{paths.num_tx}, num_rx_antennas:{num_tx_antennas}")
        
        # set rx and tx name list
        rx_name_list=[]
        tx_name_list=[]
        for rx_name in scene.receivers:
            rx_name_list.append(rx_name)
        for tx_name in scene.transmitters:
            tx_name_list.append(tx_name)


        
        # 轉換 sources 和 targets 為 numpy array
        sources_np = paths.sources.numpy()  # shape: (3, num_tx * num_tx_antennas)
        targets_np = paths.targets.numpy()  # shape: (3, num_rx * num_rx_antennas)
        
        # 轉換 drjit TensorXf 為 numpy array
        doppler_np = paths.doppler.numpy()
        phi_r_np = paths.phi_r.numpy()
        phi_t_np = paths.phi_t.numpy()
        theta_r_np = paths.theta_r.numpy()
        theta_t_np = paths.theta_t.numpy()
        
        # 創建該時刻的 metadata
        channel_meta = {
            'timestamp': timestamp,
            'frequency': scene.frequency,
            'num_tx': paths.num_tx,
            'num_rx': paths.num_rx,
            'num_tx_antennas': num_tx_antennas,
            'num_rx_antennas': num_rx_antennas,
            'tx_names': tx_name_list,
            'rx_names': rx_name_list,
            'tx_ant_positions': {},
            'rx_ant_positions': {}
        }
        
        # 儲存天線位置
        for tx_idx in range(paths.num_tx):
            tx_positions = []
            tx_ant_dict={}
            for tx_ant_idx in range(num_tx_antennas):   
                ant_global_idx = tx_idx * num_tx_antennas + tx_ant_idx
                pos = (
                    float(sources_np[0, ant_global_idx]),
                    float(sources_np[1, ant_global_idx]),
                    float(sources_np[2, ant_global_idx])
                )
                tx_ant_dict[tx_ant_idx]=pos
            channel_meta['tx_ant_positions'][tx_name_list[tx_idx]]=tx_ant_dict
        
        for rx_idx in range(paths.num_rx):
            rx_positions = []
            rx_ant_dict={}
            for rx_ant_idx in range(num_rx_antennas):
                ant_global_idx = rx_idx * num_rx_antennas + rx_ant_idx
                pos = (
                    float(targets_np[0, ant_global_idx]),
                    float(targets_np[1, ant_global_idx]),
                    float(targets_np[2, ant_global_idx])
                )
                rx_ant_dict[rx_ant_idx]=pos
            channel_meta['rx_ant_positions'][rx_name_list[rx_idx]]=rx_ant_dict
        

        
        # 儲存 metadata 為 JSON（每個時刻一個檔案）
        # 結果會寫到: dataset/scenario_id/channel/frame_id/metadata.json
        self.channel_writer.write_channel_metadata(timestamp, channel_meta)


        
        # 按照 TX-RX 對儲存 pickle 檔案
        for tx_idx in range(paths.num_tx):
            for rx_idx in range(paths.num_rx):
                tx_name = tx_name_list[tx_idx]
                rx_name = rx_name_list[rx_idx]
                
                for tx_ant_idx in range(num_tx_antennas):
                    for rx_ant_idx in range(num_rx_antennas):
                        # 提取該天線對的數據（全部轉為 numpy array）
                        channel_data = {
                            'timestamp': timestamp,
                            'tx_name': tx_name,
                            'rx_name': rx_name,
                            'tx_ant_idx': tx_ant_idx,
                            'rx_ant_idx': rx_ant_idx,
                            # CIR與其他通訊參數（確保都是 numpy array）
                            'a': np.array(a[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :, 0]),
                            'tau': np.array(tau[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :]),
                            'doppler': np.array(doppler_np[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :]),
                            'phi_r': np.array(phi_r_np[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :]),
                            'phi_t': np.array(phi_t_np[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :]),
                            'theta_r': np.array(theta_r_np[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :]),
                            'theta_t': np.array(theta_t_np[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, :])
                        }

                        self.channel_writer.write_channel_data(
                            frame_id=timestamp,
                            tx_name=tx_name,
                            rx_name=rx_name,
                            tx_ant_idx=tx_ant_idx,
                            rx_ant_idx=rx_ant_idx,
                            data=channel_data
                        )
    def get_scene_obj_info(self):
        object_list=[]
        for tx in self.scene.transmitters:
            antenna_id=tx
            position=self.scene.get(antenna_id).position
            orientation=self.scene.get(antenna_id).orientation
            velocity=self.scene.get(antenna_id).velocity
            position=np.array(position).flatten().astype(float).tolist()
            orientation=np.array(orientation).flatten().astype(float).tolist()
            velocity=np.array(velocity).flatten().astype(float).tolist()

            data={
                "id":antenna_id,
                "type":"antenna",
                "position":position,
                "orientation":orientation,
                "velocity":velocity,
                "addictional_info":self.addictional_tx_antenna_info
            }
            object_list.append(data)
        for rx in self.scene.receivers:
            antenna_id=rx
            position=self.scene.get(antenna_id).position
            orientation=self.scene.get(antenna_id).orientation
            velocity=self.scene.get(antenna_id).velocity
            position=np.array(position).flatten().astype(float).tolist()
            orientation=np.array(orientation).flatten().astype(float).tolist()
            velocity=np.array(velocity).flatten().astype(float).tolist()

            data={
                "id":antenna_id,
                "type":"antenna",
                "position":position,
                "orientation":orientation,
                "velocity":velocity
            }
            object_list.append(data)
        return object_list


    def _run_simulation(self, message):
        timestamp=message["data"]["timestamp"]
        print(f"==============timestamp {timestamp}==========")
        # set object position
        # print(message["data"])
        tx_list=message["data"]["tx_list"]
        for tx_name in tx_list:
            position = [ tx_list[tx_name]["x_pos"], tx_list[tx_name]["y_pos"], tx_list[tx_name]["z_pos"] ]
            self.scene.get(tx_name).position = position
        rx_list=message["data"]["rx_list"]
        for rx_name in rx_list:
            position = [ rx_list[rx_name]["x_pos"], rx_list[rx_name]["y_pos"], rx_list[rx_name]["z_pos"] ]
            self.scene.get(rx_name).position = position
        
        print("run simulation...")


        paths = self.p_solver(scene=self.scene,
            max_depth=3,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=False,
            seed=41)
        
        ## CIR to ofdm channel
        temp_a, temp_tau = paths.cir(normalize_delays=False, out_type="numpy", num_time_steps=14)
        temp_a = np.expand_dims(temp_a, axis=0)
        temp_tau = np.expand_dims(temp_tau, axis=0)
        temp_h_freq = cir_to_ofdm_channel(self.frequencies, temp_a, temp_tau, normalize=True)
        print(f"temp_h_freq.shape={temp_h_freq.shape}")
        cir_entry = {
            "timestamp": timestamp,
            "cir_amplitude": temp_a,
            "cir_delay": temp_tau
        }
        if not hasattr(self, "cir_list"):
            self.cir_list = []
        self.cir_list.append(cir_entry)
        #temp_h_freq ([batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex)




        tx_power=self.tx_power
        noise=self.noise
        path_loss=-10*np.log10(tf.reduce_sum(tf.abs(paths.a[0])**2))
        temp_snr=(tx_power-path_loss)-noise  # dB
        bits, bits_hat = self.commu_sys(batch_size=self.batch_size, ebno_db=temp_snr, h_freq=temp_h_freq)

    

        temp_ber=sionna.phy.utils.compute_ber(bits, bits_hat)
        temp_ber=temp_ber.numpy().item()
        self.ber_list.append(temp_ber)
        self.snr_list.append(temp_snr)
        print_info=True
        if print_info:
            print(f"tx power: {tx_power:.2f} dBm")
            print(f"path loss: {path_loss:.2f} dB")
            print(f"snr: {temp_snr:.2f} dB")
            print(f"ber: {temp_ber:.2e}")
        
        timestamp=message["data"]["timestamp"]
        visualize_flag=True
        if visualize_flag and timestamp%10==0:
            my_cam = Camera(position=[0,-1000, 1000], look_at=[0,0,0])
            # Render scene with new camera*
            img=self.scene.render(camera=my_cam, resolution=[640, 480], paths=paths, num_samples=512); # Increase num_samples to increase image quality
            if send_to_web_flag:
                send_figure_to_web(fig=img, timestamp=message['data']['timestamp'])
            
        
        data={
            "timestamp":timestamp,
            "snr":temp_snr,
            "ber":temp_ber
        }
        self.send_message(target_component="visualizer", message_type="test", data=data)
        

    def _save_data(self ,message):
        print(f"==============save data==============")
        ## message sample
        # message['data']["target"]: "wireless_env"
        # message['data']["message_type"]: "save_data"
        # message['data']["timestamp"]: 299
        # message['data']["save_type"]: "cir"
        # message['data']["file_path"]: "/dataset/cir_data.json"
        save_type = message['data']['save_type']
        file_path = message['data']['file_path']

        # check folder exists
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            print(f"{dir_path} not exist, creating...")
            os.makedirs(dir_path)
            print(f"finish creating {dir_path}")
        if save_type=="cir":
            with open(file_path, "wb") as f:
                pickle.dump(self.cir_list, f)
            print(f"{file_path} saved")
        else:
            print(f"unknown save type: {save_type}")
    
    def _create_object(self ,message):
        self.create_count+=1
        # time.sleep(0.003)
        ## message sample
        # message['data']["id"]: "wireless_env"
        # message['data']["position"]: (px, py, pz)
        # message['data']["orientation"]: (yaw, pitch, and roll angles)
        if not hasattr(self, "scene"):
            print("scene does not exist, creating object failure")
            return
        if message['data']["type"] == "car":
            id = message['data']["id"]
            self.sys_print(f"start to create car:{id}")
            px, py, pz = message['data']["position"][0], message['data']["position"][1], message['data']["position"][2]
            yaw, pitch, roll = message['data']["orientation"][0], message['data']["orientation"][1], message['data']["orientation"][2]
            # look_at_x, look_at_y, look_at_z = message['data']["forward_vector"][0], message['data']["forward_vector"][1], message['data']["forward_vector"][2]
            car = SceneObject(fname=sionna.rt.scene.low_poly_car, # Simple mesh of a car
                                name=f"car_{id}",
                                radio_material=self.car_material)
            
            self.car_dict[id]=car
            self.scene.edit(add=car)
            self.scene.add(sionna.rt.Receiver(name=f"rx_{id}", position=mi.Point3f(px, py, pz+1.5), display_radius=1))
            car.position = mi.Point3f(px, py, pz+0.5)
            self.car_dict[id].scaling=1.2
        
            # car.orientation = mi.Point3f(yaw, pitch, roll)
            # car.look_at(mi.Point3f(x+look_at_x, y+look_at_y, z+look_at_z))
        elif message['data']["type"] == "rsu_camera":
            id = message['data']["id"]
            self.sys_print(f"start to create rsu_camera")
            px, py, pz = message['data']["position"][0], message['data']["position"][1], message['data']["position"][2]
            yaw, pitch, roll = message['data']["orientation"][0], message['data']["orientation"][1], message['data']["orientation"][2]
            sn_yaw, sn_pitch, sn_roll = carla_to_sionna_orientation(float(yaw), float(pitch), float(roll))
            self.sys_print(f"sn_yaw, sn_pitch, sn_roll:{sn_yaw, sn_pitch, sn_roll}")
            self.my_rsu_cam = Camera(position=(px, py, pz), orientation=(sn_yaw, sn_pitch, sn_roll))
        elif message['data']["type"] == "tx":
            id = message['data']["id"]
            self.sys_print(f"start to create new tx")
            px, py, pz = message['data']["position"][0], message['data']["position"][1], message['data']["position"][2]
            # yaw, pitch, roll = message['data']["orientation"][0], message['data']["orientation"][1], message['data']["orientation"][2]
            self.tx_list[id]["x_pos"] = px
            self.tx_list[id]["y_pos"] = py
            self.tx_list[id]["z_pos"] = pz
            self.scene.add(sionna.rt.Transmitter(name=id, position=(px, py, pz), display_radius=1))
            self.sys_print(f"finished creating new tx:{id} at ({px}, {py}, {pz})")




    def _update_object(self ,message):
        # time.sleep(0.001)
        ## message sample
        # message['data']["id"]: "wireless_env"
        # message['data']["position"]: (px, py, pz)
        # message['data']["orientation"]: (yaw, pitch, and roll angles)
        if not hasattr(self, "scene"):
            print(f"[{self.component_name}] scene does not exist, update object failure")
            return
        
        id = message['data']["id"]

        if id == "spectator":
            px, py, pz = message['data']["position"][0], message['data']["position"][1], message['data']["position"][2]
            yaw, pitch, roll = message['data']["orientation"][0], message['data']["orientation"][1], message['data']["orientation"][2]
            # self.my_cam = Camera(position=mi.Point3f(px, py, pz+200), orientation=mi.Point3f(yaw, pitch, roll))
            self.my_cam = Camera(position=mi.Point3f(px, py, pz+50), look_at=mi.Point3f(px, py, 0))
            return


        
        if message['data']["type"] =="car":
            self.sys_print(f"start to update car_{id}")
            px, py, pz = message['data']["position"][0], message['data']["position"][1], message['data']["position"][2]
            self.sys_print(f"finished get px, py, pz")
            yaw, pitch, roll = message['data']["orientation"][0], message['data']["orientation"][1], message['data']["orientation"][2]
            sn_yaw, sn_pitch, sn_roll = carla_to_sionna_orientation(yaw, pitch, roll)

            # print(f"[{self.component_name}] finished get yaw, pitch, roll")
            look_at_x, look_at_y, look_at_z = message['data']["forward_vector"][0], message['data']["forward_vector"][1], message['data']["forward_vector"][2]
            # 方向向量正規化
            direction = np.array([look_at_x, look_at_y, look_at_z])
            direction = direction / np.linalg.norm(direction)
            distance = 3
            look_at_position=np.array([px, py, pz]) + direction * distance
            self.car_dict[id].position=mi.Point3f(px, py, pz+0.5)
            self.car_dict[id].orientation=mi.Point3f(sn_yaw, sn_pitch, sn_roll)
            # print(f"[{self.component_name}] ID:{id}, position:{x}, {y}, {z}")
            self.car_dict[id].look_at(mi.Point3f(look_at_position))

            #set antenna position +=1
            look_at_position[2]+=1
            self.scene.get(f"rx_{id}").position = mi.Point3f(look_at_position)
        
        elif message['data']["type"] =="rsu_camera":
            self.sys_print(f"start to update rsu_camera")
            px, py, pz = message['data']["position"][0], message['data']["position"][1], message['data']["position"][2]
            yaw, pitch, roll = message['data']["orientation"][0], message['data']["orientation"][1], message['data']["orientation"][2]
            sn_yaw, sn_pitch, sn_roll = carla_to_sionna_orientation(yaw, pitch, roll)
            self.sys_print(f"finished calculate orientation")
            self.my_rsu_cam = Camera(position=(px, py, pz), orientation=(sn_yaw, sn_pitch, sn_roll))
            self.sys_print(f"finished updating rsu_camera")
        elif message['data']["type"] =="tx":
            pass


    def _spectator_render(self ,message): 
        self.timestamp=message['data']['timestamp']
        # self.scene._scene_params = mi.traverse(self.scene._scene)

        # for name in self.scene.receivers.keys():
        #     print(name)
        #     print(self.scene.get(name).position)
        # print(list(self.car_dict.keys()))
        # print(f"create count{self.create_count}")
        enable_paths=True
        if enable_paths==True:
            paths = self.p_solver(scene=self.scene,
                max_depth=5,
                los=True,
                specular_reflection=True,
                diffraction=True,
                diffuse_reflection=True,
                refraction=True,
                synthetic_array=False,
                seed=41)
            # img=self.scene.render(camera=self.my_cam, resolution=[640, 480], fov=120, paths=paths, num_samples=512); # Increase num_samples to increase image quality
            if enable_render:
                img=self.scene.render(camera=self.my_rsu_cam, resolution=[640, 480], fov=120, paths=paths, num_samples=512); # Increase num_samples to increase image quality
        else:
            if enable_render:
                # img=self.scene.render(camera=self.my_cam, resolution=[640, 480], fov=120, num_samples=512); # Increase num_samples to increase image quality
                img=self.scene.render(camera=self.my_rsu_cam, resolution=[640, 480], fov=120, num_samples=512); # Increase num_samples to increase image quality

        # output image=matplotlib figure
        if enable_render:
            flip_flag=True
            if flip_flag:
                img.canvas.draw()
                image_array = np.array(img.canvas.buffer_rgba())
                # 移除 alpha 通道，只保留 RGB
                image_array = image_array[:, :, :3]
                flipped_array = np.fliplr(image_array)
                img = Image.fromarray(flipped_array) # PIL image
            if send_figure_to_web and enable_render:
                send_figure_to_web(fig=img, timestamp=self.timestamp, target="sionna")


        data={
            "scene_obj_info": self.get_scene_obj_info()
        }

        if self.timestamp%10==0:
            self.send_message(
                    target_component="external_module", 
                    message_type="beam_control", 
                    data=data,
                    to_scheduler=True
            )

        data={
            "timestamp" : self.timestamp
        }
        
        self.send_message(
                target_component="visualizer", 
                message_type="update_world", 
                data=data,
                to_scheduler=True
            )
        if self.enable_dataset_writer==True:
            trajectory_list=self.get_scene_obj_info()
            self.trajectory_datawriter.write_trajectory(frame_id=self.timestamp, trajectory_data=trajectory_list, trajectory_type="antenna")
        if self.enable_dataset_writer==True and enable_paths==True:
            self.save_paths_info(timestamp=self.timestamp, scene=self.scene, paths=paths)
            if enable_render:
                self.bs_datawriter.write_rt_img(frame_id=self.timestamp, image=flipped_array)




        
        
        

        


    # def initialize_wireless_env(self, message):
    #     print(f"[{self.component_name}] 收到初始化訊息")
    #     print(f"當前執行緒: {threading.current_thread().name}")
    #     print(f"是否為主執行緒: {threading.current_thread() == threading.main_thread()}")
        
    #     # 直接使用成功的程式碼
    #     print("load scene...")
    #     scene_path = "my_scene/nycu_campus/nycu_campus.xml"  # 直接硬編碼
    #     print(f"scene_path: {scene_path}")
    #     self.scene = sionna.rt.load_scene(scene_path)
    #     print("load scene success")
        
    #     print("set material...")
    #     self.scene.frequency = 2.4e9  # 直接硬編碼
    #     new_material = sionna.rt.RadioMaterial(f"my_material_{self.scene.frequency}", 
    #                                         relative_permittivity=5.24, 
    #                                         conductivity=0.0462*(self.scene.frequency/1e9)**0.7822, 
    #                                         scattering_pattern='lambertian')
    #     for i, obj in enumerate(self.scene.objects.values()):
    #         self.scene.get(obj.name).radio_material = new_material
    #     print("set material success")

        
        

def main():
    """
    基於UDP+JSON的Wireless Digital Twin系統
    """
    
    # 啟動其他組件
    wireless_env = WirelessEnvironment()
    wireless_env.register_component('scheduler', 'localhost', 5000)
    wireless_env.register_component('end_user', 'localhost', 5002)
    wireless_env.register_component('visualizer', 'localhost', 5003)
    wireless_env.register_component('external_module', 'localhost', 5004)
    
    
    # 等待網路連接建立
    print(f"[{wireless_env.component_name}] 等待網路連接建立...")
    time.sleep(1)
    wireless_env.process_messages()
    
  
   


if __name__ == "__main__":
    main()