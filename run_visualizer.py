import os 
import numpy as np


import matplotlib.pyplot as plt

import socket
import json
import threading
import time
from typing import Dict, Any, Callable
from utils import NetworkComponent, read_json_array, carla_image_to_array
from dataset_writer import DatasetWriter
from visualize_web.app import send_figure_to_web , send_to_web
# from visualize_web.app_v1.app import send_figure_to_web, send_to_web

import base64
from io import BytesIO
from PIL import Image


import glob
import sys
import random
import cv2
import queue


use_carla=True
if use_carla:
    import carla

send_to_web_flag=False
class TickProcessor:
    def __init__(self, interval=10):
        self.interval = interval
        self.tick_count = 0
        self.latest_image = None

    def camera_callback(self, image):
        self.latest_image = image
        self.tick_count += 1
        
        if self.tick_count % self.interval==1:
            self.process_camera(image)


    def process_camera(self, image):
        if send_to_web_flag:
            # 將CARLA的raw_data轉成numpy陣列
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))  

            # convert to RGB
            rgb_array = array[:, :, [2, 1, 0]]  # 正確通道順序：R,G,B

            # 轉成PIL Image
            pil_image = Image.fromarray(rgb_array, mode="RGB")
        
            send_figure_to_web(fig=pil_image, timestamp=self.tick_count, target="carla")


class DataBuffer:
    def __init__(self):
        self.latest_rgb_image = None
        self.latest_semantic_image = None
        self.latest_depth_image = None
        self.latest_lidar = None
        self.latest_radar = None
        

    def rgb_camera_callback(self, image):
        self.latest_rgb_image = image
    def semantic_camera_callback(self, image):
        self.latest_semantic_image = image
        self.latest_semantic_image.convert(carla.ColorConverter.CityScapesPalette)
    def lidar_callback(self, point_cloud):
        num_points = len(point_cloud)
        data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        data = np.reshape(data, (num_points, 4)) # Each point has (x, y, z, intensity) data
        self.latest_lidar = data
    def depth_camera_callback(self, image):
        self.latest_depth_image = image
        self.latest_depth_image.convert(carla.ColorConverter.LogarithmicDepth)
        
        



# visualizer (Port: 5003)
class Visualizer(NetworkComponent):
    def __init__(self):
        super().__init__('visualizer', 5003)
        self.simulation_params = {}
        self.enable_dataset_writer=False
        # 註冊訊息處理器
        self.register_handler('test', self._test)
        self.register_handler('test_set_carla', self._test_set_carla)
        self.register_handler('update_world', self._test_run_one_step)
        self.register_handler('set_dataset_writer', self._set_dataset_writer)
        self.register_handler('reset_start_time', self._reset_start_time)
        self.data_buffer=DataBuffer()


        
        # 啟動網路監聽
        self.start_listening()
        
        # 註冊調度器位址
        self.register_component('scheduler', 'localhost', 5000)
        self.timestamp=0
    def _set_dataset_writer(self, message):
        """ Message example
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
        
        self.recorder_root=f"{self.dataset_root}/{self.scenario_id}"

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
    def _test(self, message):
        self.sys_print(f"=====timestamp: {message['data']['timestamp']}=====")
        snr=message["data"]["snr"]
        ber=message["data"]["ber"]
        send_to_web(data=snr, timestamp=message['data']['timestamp'])
        self.sys_print(f"SNR: {snr:.2f} dB, BER: {ber:.2e}")
        self.sys_print("=============")
    def _test_set_carla(self, message):
        try:
            carla_root=os.path.join(os.environ.get("CARLA_ROOT", ""), "PythonAPI")  # set CARLA_ROOT if carla is not pip-installed
            sys.path.append(glob.glob(f'{carla_root}/carla/dist/carla-*%d.%d-%s.egg' % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
        except IndexError:
            pass

        # 连接服务器
        self.client = carla.Client("127.0.0.1", 2000)
        
        self.client.set_timeout(10.0)


        # set synchronous
        self.world = self.client.reload_world()
        if self.enable_dataset_writer == True:
            pass
            # self.client.start_recorder(f"record_{self.scenario_id}.log", True)
        self.time_unit=message["data"]["time_unit"]
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.time_unit

        settings.no_rendering_mode = False  # 確保渲染
        
        # # 確保 substepping 參數一致
        # settings.substepping = True
        # settings.max_substep_delta_time = 0.1 
        # settings.max_substeps = 10  

        self.world.apply_settings(settings)
        # 先設定 Traffic Manager
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.global_percentage_speed_difference(-50)

        
        self.sys_print(f"===================\n\n\n{settings}\n\n\n===============")




        blueprint_library = self.world.get_blueprint_library()
        # print(blueprint_library)


        vehicle_bp = blueprint_library.find('vehicle.mercedes.coupe')
        vehicle_bp.set_attribute('color', '0, 0, 0')

        spawn_points = self.world.get_map().get_spawn_points()
        # for sp in spawn_points:
        #     print(f"location: {sp.location}")
        # print(f"choice: {spawn_points[2].location}")
        # transform = random.choice(spawn_points)
        transform = spawn_points[2]
        self.sys_print("start to create ego vehicle")
        while True:
            self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if self.ego_vehicle==None:
                continue
            else:
                self.sys_print("finish creating ego vehicle")
                break
        self.ego_vehicle.set_autopilot(True)

        

        
        transform=self.ego_vehicle.get_transform()
        location=transform.location
        rotation=transform.rotation
        forward_vector = transform.get_forward_vector()
        
        
        object_data={
            "id" : self.ego_vehicle.id,
            "type" : "car",
            "position" :  (location.x, location.y, location.z),
            "orientation" : (rotation.yaw, rotation.pitch, rotation.roll),
            "forward_vector" : (forward_vector.x, forward_vector.y, forward_vector.z)
        }
        self.send_message(
                target_component="wireless_env", 
                message_type="create_object", 
                data=object_data,
                to_scheduler=True
            )

        #===============test RSU transform===============
        self.sys_print("start to set rsu_transform")
        rsu_transform = carla.Transform(carla.Location(x=340.0, y=80.0, z=12.0),
                            carla.Rotation(pitch=-20.0, yaw=-110.0, roll=0.0))

        #test for camera
        # rsu_transform = carla.Transform(carla.Location(x=340.0, y=80.0, z=35.0),
        # carla.Rotation(pitch=-45.0, yaw=-110.0, roll=0.0))

        #===============initial RGB camera===============
        # generate RGB camera
        self.sys_print("start to set RGB camera")
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '120')
        self.camera = self.world.spawn_actor(camera_bp, rsu_transform)
        # 使用方式
        self.camera_processor = TickProcessor(10)
        self.camera.listen(self.camera_processor.camera_callback)

 
        
        object_data = {
            "id": "rsu",
            "type" : "rsu_camera",
            "position": (rsu_transform.location.x, rsu_transform.location.y, rsu_transform.location.z),
            "orientation": (rsu_transform.rotation.yaw, rsu_transform.rotation.pitch, rsu_transform.rotation.roll),
        }

        self.send_message(
            target_component="wireless_env", 
            message_type="create_object", 
            data=object_data,
            to_scheduler=True
        )

        self.sys_print("finish setting RGB camera")

        #===============initial semantic camera===============
        self.sys_print("start to set semantic camera")
        sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute("image_size_x",'640')
        sem_bp.set_attribute("image_size_y",'480')
        sem_bp.set_attribute("fov", '120')
        self.sem_cam = self.world.spawn_actor(sem_bp, rsu_transform)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        self.sem_cam.listen(self.data_buffer.semantic_camera_callback)

        self.sys_print("finish setting semantic camera")

        #===============initial lidar===============
        self.sys_print("start to set lidar")
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("horizontal_fov", "120")
        self.lidar = self.world.spawn_actor(lidar_bp, rsu_transform)
        self.lidar.listen(self.data_buffer.lidar_callback)
        self.sys_print("finish setting lidar")

        ##===============initial depth camera===============

        depth_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_bp.set_attribute("image_size_x",'640')
        depth_bp.set_attribute("image_size_y",'480')
        depth_bp.set_attribute("fov", '120')
        self.depth_cam = self.world.spawn_actor(depth_bp, rsu_transform)
        self.depth_cam.listen(self.data_buffer.depth_camera_callback)
        # This time, a color converter is applied to the image, to get the semantic segmentation view



        



        ## generate other cars
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        # number_of_vehicles default = 0
        number_of_vehicles = message["data"].get("number_of_other_vehicles", 0)
        self.vehicle_list = []
        if number_of_vehicles > 0:
            # 1. 取得目前世界上所有已經存在的車輛位置（包含你的主車）
            existing_vehicles = self.world.get_actors().filter('vehicle.*')
            existing_locations = [v.get_location() for v in existing_vehicles]
            
            # 為了公平分配，可以先將 spawn_points 打亂
            random.shuffle(spawn_points)
            
            spawned_count = 0
            min_distance = 10.0  # 設定最小距離為 10 公尺

            for transform in spawn_points:
                # 如果已經生成夠多台了，就停止
                if spawned_count >= number_of_vehicles:
                    break

                # 2. 檢查此生成點是否與「現有車輛」太近
                too_close = False
                for loc in existing_locations:
                    if transform.location.distance(loc) < min_distance:
                        too_close = True
                        break
                
                if too_close:
                    continue  # 跳過這個點，找下一個

                # 3. 嘗試生成
                try:
                    bp = random.choice(vehicle_blueprints)
                    transform.location.z = 1.5
                    vehicle = self.world.try_spawn_actor(bp, transform)
                    
                    if vehicle is not None:
                        self.sys_print(f"add new vehicle at {transform.location}")
                        vehicle.set_autopilot(True)
                        self.vehicle_list.append(vehicle)
                        
                        # 重要：將新生成的車輛位置加入 list，確保下一台車不會跟這台太近
                        existing_locations.append(vehicle.get_location())
                        spawned_count += 1
                        self.sys_print(f"已生成{spawned_count}輛車")

                        # --- 以下維持你原本的資料傳輸邏輯 ---
                        location = vehicle.get_transform().location
                        rotation = vehicle.get_transform().rotation
                        forward_vector = vehicle.get_transform().get_forward_vector()
                        object_data = {
                            "id": vehicle.id,
                            "type" : "car",
                            "position": (location.x, location.y, location.z),
                            "orientation": (rotation.yaw, rotation.pitch, rotation.roll),
                            "forward_vector": (forward_vector.x, forward_vector.y, forward_vector.z)
                        }
                        self.send_message(
                            target_component="wireless_env", 
                            message_type="create_object", 
                            data=object_data,
                            to_scheduler=True
                        )
                except Exception as e:
                    self.sys_print(f"Caught error: {e}")
                    # 如果噴錯了，就毀滅這台有問題的車，避免主程式崩潰
                    vehicle.destroy()


        self.sys_print("carla setting finished")
        
        

    def _test_run_one_step(self, message):
        self.sys_print("updating world...")
        for i in range (1):
            self.world.tick()
            self.timestamp+=1
        # world: carla.World
        # update world and send msg
        trajectory_list=[]
        transform=self.ego_vehicle.get_transform()
        location=transform.location
        rotation=transform.rotation
        forward_vector = transform.get_forward_vector()
        object_data={
            "id" : self.ego_vehicle.id,
            "type" : "car",
            "position" :  (location.x, location.y, location.z),
            "orientation" : (rotation.yaw, rotation.pitch, rotation.roll),
            "forward_vector" : (forward_vector.x, forward_vector.y, forward_vector.z)
        }
        trajectory_list.append(object_data)
        self.send_message(
                target_component="wireless_env", 
                message_type="update_object", 
                data=object_data,
                to_scheduler=True
            )
        
        if len(self.vehicle_list)>0:
            for vehicle in self.vehicle_list:
                transform=vehicle.get_transform()
                location=transform.location
                rotation=transform.rotation
                forward_vector = transform.get_forward_vector()
                object_data={
                    "id" : vehicle.id,
                    "type" : "car",
                    "position" :  (location.x, location.y, location.z),
                    "orientation" : (rotation.yaw, rotation.pitch, rotation.roll),
                    "forward_vector" : (forward_vector.x, forward_vector.y, forward_vector.z)
                }
                self.send_message(
                        target_component="wireless_env", 
                        message_type="update_object", 
                        data=object_data,
                        to_scheduler=True
                    )
                trajectory_list.append(object_data)

        # for vehicle in self.vehicle_list :
        #     loc = vehicle.get_location()
        #     print(f"ID={vehicle.id}, x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")


        # update spectator
        spectator = self.world.get_spectator()
        vehicle_transform = self.ego_vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(vehicle_transform.location + carla.Location(z=50),
                            carla.Rotation(pitch=-90))
        )
        transform=spectator.get_transform()
        location=transform.location
        rotation=transform.rotation
        # forward_vector = transform.get_forward_vector()
        # self.camera.set_transform(transform)
        # self.lidar.set_transform(transform)
        # self.sem_cam.set_transform(transform)
        # self.depth_cam.set_transform(transform)
        object_data={
            "id" : "spectator",
            "type" : "spectator",
            "position" :  (location.x, location.y, location.z),
            "orientation" : (rotation.yaw, rotation.pitch, rotation.roll)
        }
        self.send_message(
                target_component="wireless_env", 
                message_type="update_object", 
                data=object_data,
                to_scheduler=True
            )
        
        
        # wireless_env render
        data={
            "timestamp" : self.timestamp
        }
        self.send_message(
                target_component="wireless_env", 
                message_type="spectator_render", 
                data=data,
                to_scheduler=True
            )

        
        if self.enable_dataset_writer==True:
            current_camera_rgb_array = carla_image_to_array(self.camera_processor.latest_image)
            current_depth_image_array = carla_image_to_array(self.data_buffer.latest_depth_image)
            current_semantic_image_array = carla_image_to_array(self.data_buffer.latest_semantic_image)
            # self.sys_print(f"type of self.data_buffer.latest_semantic_image: {self.data_buffer.latest_semantic_image}")
            self.bs_datawriter.write_rgb(frame_id=self.timestamp, image=current_camera_rgb_array)
            self.bs_datawriter.write_semantic(frame_id=self.timestamp, image=current_semantic_image_array)
            self.bs_datawriter.write_depth(frame_id=self.timestamp, image=current_depth_image_array)
            self.bs_datawriter.write_lidar(frame_id=self.timestamp, points=self.data_buffer.latest_lidar)
            self.trajectory_datawriter.write_trajectory(frame_id=self.timestamp, trajectory_data=trajectory_list, trajectory_type="vehicle")
            



        self.sys_print("finish updating...")



    
    



def main():
    """
    基於UDP+JSON的Wireless Digital Twin系統
    """
    
    # 啟動其他組件
    visualizer = Visualizer()
    visualizer.register_component('scheduler', 'localhost', 5000)
    visualizer.register_component('wireless_env', 'localhost', 5001)
    visualizer.register_component('end_user', 'localhost', 5002)
    visualizer.register_component('external_module', 'localhost', 5004)
    
    
    # 等待網路連接建立
    print("等待網路連接建立...")
    time.sleep(1)
    visualizer.process_messages()
    
  
   


if __name__ == "__main__":
    main()