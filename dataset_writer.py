"""
DatasetWriter - 用於 CARLA + Sionna 多模態資料集的寫入工具

支援多程式獨立寫入，使用 thread + queue 實現非阻塞寫入。
"""

import os
import json
import csv
import pickle
import threading
import queue
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import filelock
from PIL import Image


# ============================================================================
# 資料結構定義（可依需求修改）
# ============================================================================

@dataclass
class TrajectoryData:
    """軌跡資料結構 - 可依需求增減欄位"""
    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    acceleration_z: float = 0.0
    
    @classmethod
    def get_fieldnames(cls) -> list:
        """取得 CSV 欄位名稱"""
        return list(cls.__dataclass_fields__.keys())


@dataclass
class ChannelMetadata:
    """Channel 元資料結構"""
    num_subcarriers: int = 64
    num_tx_antennas: int = 1
    num_rx_antennas: int = 1
    carrier_frequency_ghz: float = 3.5
    bandwidth_mhz: float = 100.0
    subcarrier_spacing_khz: float = 30.0
    description: str = ""


@dataclass 
class BasestationConfig:
    """基地台設定結構"""
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    rotation_roll: float = 0.0
    rotation_pitch: float = 0.0
    rotation_yaw: float = 0.0
    num_antennas: int = 1
    antenna_pattern: str = "omnidirectional"
    tx_power_dbm: float = 30.0
    carrier_frequency_ghz: float = 3.5
    description: str = ""


@dataclass
class ScenarioMetadata:
    """場景元資料結構"""
    scenario_id: str = ""
    map_name: str = ""
    weather: str = ""
    num_vehicles: int = 0
    num_basestations: int = 0
    duration_seconds: float = 0.0
    fps: float = 10.0
    created_at: str = ""
    description: str = ""


@dataclass
class DatasetConfig:
    """全域資料集設定"""
    dataset_name: str = "carla_sionna_dataset"
    version: str = "1.0.0"
    created_at: str = ""
    coordinate_system: str = "CARLA_UE4"  # 左手座標系
    image_format: str = "png"
    image_resolution: Dict[str, int] = None
    lidar_format: str = "npy"
    radar_format: str = "npy"
    channel_format: str = "pkl"
    description: str = ""
    
    def __post_init__(self):
        if self.image_resolution is None:
            self.image_resolution = {"width": 1920, "height": 1080}


# ============================================================================
# 寫入任務定義
# ============================================================================

@dataclass
class WriteTask:
    """寫入任務"""
    task_type: str  # "image", "npy", "csv_append", "json"
    file_path: Path
    data: Any
    extra_params: Dict = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


# ============================================================================
# DatasetWriter 主類別
# ============================================================================

class DatasetWriter:
    """
    資料集寫入器
    
    使用方式：
        writer = DatasetWriter(
            dataset_root="./dataset",
            scenario_id="scenario_001",
            entity_type="vehicle",
            entity_id="veh_0"
        )
        
        writer.write_rgb(frame_id=1, image=np_array)
        writer.write_lidar(frame_id=1, points=np_array)
        writer.close()
    """
    
    EntityType = Literal["vehicle", "basestation", "channel"]
    
    def __init__(
        self,
        dataset_root: Union[str, Path],
        scenario_id: str,
        entity_type: EntityType,
        entity_id: Optional[str] = None,
        queue_size: int = 100,
        num_workers: int = 2
    ):
        """
        初始化 DatasetWriter
        
        Args:
            dataset_root: 資料集根目錄
            scenario_id: 場景 ID（如 "scenario_001"）
            entity_type: 實體類型 ("vehicle", "basestation", "channel")
            entity_id: 實體 ID（如 "veh_0", "bs_0"），Channel 類型可為 None
            queue_size: 寫入佇列大小，滿了會阻塞
            num_workers: 寫入執行緒數量
        """
        self.dataset_root = Path(dataset_root)
        self.scenario_id = scenario_id
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.queue_size = queue_size
        self.num_workers = num_workers
        
        # 建立路徑
        self._setup_paths()
        
        # 建立目錄結構
        self._create_directories()
        
        # 初始化全域設定檔（如果不存在）
        self._init_global_configs()
        
        # 初始化場景設定檔（如果不存在）
        self._init_scenario_configs()
        
        # 初始化實體設定檔（如果不存在）
        self._init_entity_configs()
        
        # 設定寫入佇列和執行緒
        self._write_queue: queue.Queue[Optional[WriteTask]] = queue.Queue(maxsize=queue_size)
        self._workers: list[threading.Thread] = []
        self._shutdown_event = threading.Event()
        
        # 啟動寫入執行緒
        self._start_workers()
        
        # CSV 檔案鎖（用於 append 操作）
        self._csv_locks: Dict[str, threading.Lock] = {}
    
    # ------------------------------------------------------------------------
    # 路徑設定
    # ------------------------------------------------------------------------
    
    def _setup_paths(self):
        """設定各種路徑"""
        self.scenario_path = self.dataset_root / self.scenario_id
        
        if self.entity_type == "vehicle":
            # 如果 entity_id 為 None，建立共用的 trajectory 目錄
            if self.entity_id is None:
                self.entity_path = self.scenario_path / "trajectories"
            else:
                self.entity_path = self.scenario_path / "vehicles" / self.entity_id
        elif self.entity_type == "basestation":
            self.entity_path = self.scenario_path / "basestations" / self.entity_id
        elif self.entity_type == "channel":
            self.entity_path = self.scenario_path / "channel"
        else:
            raise ValueError(f"Unknown entity_type: {self.entity_type}")
        
        # 感測器資料路徑（vehicle 和 basestation 共用）
        if self.entity_type in ["vehicle", "basestation"]:
            self.rgb_path = self.entity_path / "rgb"
            self.semantic_path = self.entity_path / "semantic"
            self.lidar_path = self.entity_path / "lidar"
            self.radar_path = self.entity_path / "radar"
            self.depth_path = self.entity_path / "depth"
            self.rt_img_path = self.entity_path / "rt_img"
        if self.entity_type == "vehicle":
            self.trajectory_path = self.entity_path / "trajectory"
            self.vehicle_trajectory_path = self.trajectory_path / "vehicle"
            self.antenna_trajectory_path = self.trajectory_path / "antenna"
    
    def _create_directories(self):
        """建立目錄結構"""
        # 根目錄
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # 場景目錄
        self.scenario_path.mkdir(parents=True, exist_ok=True)
        
        # 實體目錄
        self.entity_path.mkdir(parents=True, exist_ok=True)
        
        # 感測器目錄
        if self.entity_type in ["vehicle", "basestation"]:
            self.rgb_path.mkdir(parents=True, exist_ok=True)
            self.semantic_path.mkdir(parents=True, exist_ok=True)
            self.lidar_path.mkdir(parents=True, exist_ok=True)
            self.radar_path.mkdir(parents=True, exist_ok=True)
            self.depth_path.mkdir(parents=True, exist_ok=True)
            self.rt_img_path.mkdir(parents=True, exist_ok=True)
        # Trajectory 目錄(僅 vehicle)
        if self.entity_type == "vehicle":
            self.trajectory_path.mkdir(parents=True, exist_ok=True)
            self.vehicle_trajectory_path.mkdir(parents=True, exist_ok=True)
            self.antenna_trajectory_path.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------------
    # 設定檔初始化
    # ------------------------------------------------------------------------
    
    def _init_global_configs(self):
        """初始化全域設定檔（如果不存在）"""
        # dataset_config.json
        config_path = self.dataset_root / "dataset_config.json"
        if not config_path.exists():
            config = DatasetConfig(
                created_at=datetime.now().isoformat()
            )
            self._write_json_safe(config_path, asdict(config))
        
        # class_mapping.json - 語意分割類別對應
        mapping_path = self.dataset_root / "class_mapping.json"
        if not mapping_path.exists():
            # CARLA 預設類別對應（可依需求修改）
            default_mapping = {
                "0": {"name": "unlabeled", "color": [0, 0, 0]},
                "1": {"name": "building", "color": [70, 70, 70]},
                "2": {"name": "fence", "color": [100, 40, 40]},
                "3": {"name": "other", "color": [55, 90, 80]},
                "4": {"name": "pedestrian", "color": [220, 20, 60]},
                "5": {"name": "pole", "color": [153, 153, 153]},
                "6": {"name": "road_line", "color": [157, 234, 50]},
                "7": {"name": "road", "color": [128, 64, 128]},
                "8": {"name": "sidewalk", "color": [244, 35, 232]},
                "9": {"name": "vegetation", "color": [107, 142, 35]},
                "10": {"name": "vehicle", "color": [0, 0, 142]},
                "11": {"name": "wall", "color": [102, 102, 156]},
                "12": {"name": "traffic_sign", "color": [220, 220, 0]},
                "13": {"name": "sky", "color": [70, 130, 180]},
                "14": {"name": "ground", "color": [81, 0, 81]},
                "15": {"name": "bridge", "color": [150, 100, 100]},
                "16": {"name": "rail_track", "color": [230, 150, 140]},
                "17": {"name": "guard_rail", "color": [180, 165, 180]},
                "18": {"name": "traffic_light", "color": [250, 170, 30]},
                "19": {"name": "static", "color": [110, 190, 160]},
                "20": {"name": "dynamic", "color": [170, 120, 50]},
                "21": {"name": "water", "color": [45, 60, 150]},
                "22": {"name": "terrain", "color": [145, 170, 100]}
            }
            self._write_json_safe(mapping_path, default_mapping)
    
    def _init_scenario_configs(self):
        """初始化場景設定檔（如果不存在）"""
        # metadata.json
        metadata_path = self.scenario_path / "metadata.json"
        if not metadata_path.exists():
            metadata = ScenarioMetadata(
                scenario_id=self.scenario_id,
                created_at=datetime.now().isoformat()
            )
            self._write_json_safe(metadata_path, asdict(metadata))
    
    def _init_entity_configs(self):
        """初始化實體設定檔（如果不存在）"""
        if self.entity_type == "vehicle":
            pass
        
        elif self.entity_type == "basestation":
            # config.json
            config_path = self.entity_path / "config.json"
            if not config_path.exists():
                config = BasestationConfig()
                self._write_json_safe(config_path, asdict(config))
        
        elif self.entity_type == "channel":
            # channel_metadata.json
            metadata_path = self.entity_path / "channel_metadata.json"
            if not metadata_path.exists():
                metadata = ChannelMetadata()
                self._write_json_safe(metadata_path, asdict(metadata))
    
    def _json_serializer(self, obj):
        """
        JSON 序列化轉換器，處理 numpy、tensorflow 等特殊類型
        """
        # Numpy types
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        if hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        
        # Tensorflow types
        if type(obj).__name__ == 'EagerTensor':
            return obj.numpy().tolist()
        
        # Python complex number
        if isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag, '_type': 'complex'}
        
        # Fallback for other types
        try:
            return str(obj)
        except:
            return f"<non-serializable: {type(obj).__name__}>"
    
    def _write_json_safe(self, path: Path, data: Union[dict, list]):
        """安全寫入 JSON"""
        # 移除 filelock
        path.parent.mkdir(parents=True, exist_ok=True)  # 確保目錄存在
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
    
    def _init_csv_with_header(self, path: Path, fieldnames: list):
        """初始化 CSV 檔案（寫入表頭）"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
    
    # ------------------------------------------------------------------------
    # 寫入執行緒管理
    # ------------------------------------------------------------------------
    
    def _start_workers(self):
        """啟動寫入執行緒"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"DatasetWriter-Worker-{i}",
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self):
        """寫入執行緒主迴圈"""
        while not self._shutdown_event.is_set():
            try:
                task = self._write_queue.get(timeout=0.1)
                if task is None:  # 結束信號
                    break
                self._execute_task(task)
                self._write_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DatasetWriter] Worker error: {e}")
                self._write_queue.task_done()
    
    def _execute_task(self, task: WriteTask):
        """執行寫入任務"""
        try:
            if task.task_type == "image":
                self._write_image(task.file_path, task.data, **task.extra_params)
            elif task.task_type == "npy":
                self._write_npy(task.file_path, task.data)
            elif task.task_type == "pickle":
                self._write_pickle(task.file_path, task.data)
            elif task.task_type == "csv_append":
                self._append_csv(task.file_path, task.data, **task.extra_params)
            elif task.task_type == "json":
                self._write_json_safe(task.file_path, task.data)
        except Exception as e:
            print(f"[DatasetWriter] Failed to write {task.file_path}: {e}")
    
    def _write_image(self, path: Path, image: np.ndarray, format: str = "png"):
        """寫入圖片"""
        import cv2
        # 確保目錄存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "png":
            cv2.imwrite(str(path), image)
        elif format.lower() == "jpg" or format.lower() == "jpeg":
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(path), image)
    
    def _write_npy(self, path: Path, data: np.ndarray):
        """寫入 numpy 陣列"""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), data)
    
    def _write_pickle(self, path: Path, data: Any):
        """寫入 pickle 檔案"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def _append_csv(self, path: Path, row_data: list, use_lock: bool = True):
        """附加寫入 CSV"""
        if use_lock:
            lock_key = str(path)
            if lock_key not in self._csv_locks:
                self._csv_locks[lock_key] = threading.Lock()
            
            with self._csv_locks[lock_key]:
                with open(path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
        else:
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
    
    def _enqueue_task(self, task: WriteTask):
        """將任務加入佇列（滿了會阻塞）"""
        self._write_queue.put(task, block=True)
    
    # ------------------------------------------------------------------------
    # 公開寫入介面 - 圖片類
    # ------------------------------------------------------------------------
    
    def write_rgb(self, frame_id: int, image: np.ndarray):
        """
        寫入 RGB 圖片
        
        Args:
            frame_id: 幀編號
            image: RGB 圖片 (H, W, 3) numpy array，BGR 格式（OpenCV）
        """
        if self.entity_type not in ["vehicle", "basestation"]:
            raise RuntimeError(f"RGB not supported for entity_type: {self.entity_type}")
        
        filename = f"frame_{frame_id:05d}.png"
        file_path = self.rgb_path / filename
        
        task = WriteTask(
            task_type="image",
            file_path=file_path,
            data=image,
            extra_params={"format": "png"}
        )
        self._enqueue_task(task)
    
    def write_semantic(self, frame_id: int, image: np.ndarray):
        """
        寫入語意分割圖
        
        Args:
            frame_id: 幀編號
            image: 語意分割圖 (H, W) 或 (H, W, 3) numpy array
        """
        if self.entity_type not in ["vehicle", "basestation"]:
            raise RuntimeError(f"Semantic not supported for entity_type: {self.entity_type}")
        
        filename = f"frame_{frame_id:05d}.png"
        file_path = self.semantic_path / filename
        
        task = WriteTask(
            task_type="image",
            file_path=file_path,
            data=image,
            extra_params={"format": "png"}
        )
        self._enqueue_task(task)

    def write_depth(self, frame_id: int, image):
        """
        寫入深度圖
        
        Args:
            frame_id: 幀編號
            image: 深度圖
        """
        if self.entity_type not in ["vehicle", "basestation"]:
            raise RuntimeError(f"Semantic not supported for entity_type: {self.entity_type}")
        
        filename = f"frame_{frame_id:05d}.png"
        file_path = self.depth_path / filename
        
        task = WriteTask(
            task_type="image",
            file_path=file_path,
            data=image,
            extra_params={"format": "png"}
        )
        self._enqueue_task(task)
    
    def write_rt_img(self, frame_id: int, image):
        """
        
        Args:
            frame_id: 幀編號
            image: ray tracing image (支援 np.ndarray 或 PIL.Image)
        """
        if self.entity_type not in ["vehicle", "basestation"]:
            raise RuntimeError(f"Semantic not supported for entity_type: {self.entity_type}")
        
        # 如果是 PIL Image，轉換成 numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        filename = f"frame_{frame_id:05d}.png"
        file_path = self.rt_img_path / filename
        
        task = WriteTask(
            task_type="image",
            file_path=file_path,
            data=image,
            extra_params={"format": "png"}
        )
        self._enqueue_task(task)
    
    # ------------------------------------------------------------------------
    # 公開寫入介面 - 點雲/雷達類
    # ------------------------------------------------------------------------
    
    def write_lidar(self, frame_id: int, points: np.ndarray):
        """
        寫入 LiDAR 點雲
        
        Args:
            frame_id: 幀編號
            points: 點雲資料 (N, 4) numpy array [x, y, z, intensity]
        """
        if self.entity_type not in ["vehicle", "basestation"]:
            raise RuntimeError(f"LiDAR not supported for entity_type: {self.entity_type}")
        
        filename = f"frame_{frame_id:05d}.npy"
        file_path = self.lidar_path / filename
        
        task = WriteTask(
            task_type="npy",
            file_path=file_path,
            data=points
        )
        self._enqueue_task(task)
    
    def write_radar(self, frame_id: int, data: np.ndarray):
        """
        寫入 Radar 資料
        
        Args:
            frame_id: 幀編號
            data: Radar 資料 numpy array（格式依需求定義）
        """
        if self.entity_type not in ["vehicle", "basestation"]:
            raise RuntimeError(f"Radar not supported for entity_type: {self.entity_type}")
        
        filename = f"frame_{frame_id:05d}.npy"
        file_path = self.radar_path / filename
        
        task = WriteTask(
            task_type="npy",
            file_path=file_path,
            data=data
        )
        self._enqueue_task(task)
    
    # ------------------------------------------------------------------------
    # 公開寫入介面 - 軌跡類
    # ------------------------------------------------------------------------
    
    def write_trajectory(
        self, 
        frame_id: int, 
        trajectory_data: Union[list, dict],
        trajectory_type: Literal["vehicle", "antenna"] = "vehicle"
    ):
        """
        寫入軌跡資料(每個 frame 一個 JSON 檔案)
        
        Args:
            frame_id: 幀編號
            trajectory_data: 軌跡資料,可以是:
                - list: [{"obj1": ...}, {"obj2": ...}, ...] (多個物件)
                - dict: 單一物件資料
            trajectory_type: 軌跡類型,"vehicle" 或 "antenna"
        """
        if self.entity_type != "vehicle":
            raise RuntimeError(f"Trajectory not supported for entity_type: {self.entity_type}")
        
        # 根據類型選擇目錄
        if trajectory_type == "vehicle":
            target_path = self.vehicle_trajectory_path
        elif trajectory_type == "antenna":
            target_path = self.antenna_trajectory_path
        else:
            raise ValueError(f"Invalid trajectory_type: {trajectory_type}. Must be 'vehicle' or 'antenna'")
        
        # 如果傳入的是 dict,轉換為 list(保持格式一致性)
        if isinstance(trajectory_data, dict):
            trajectory_data = [trajectory_data]
        
        filename = f"frame_{frame_id:05d}.json"
        trajectory_file = target_path / filename
        
        task = WriteTask(
            task_type="json",
            file_path=trajectory_file,
            data=trajectory_data
        )
        self._enqueue_task(task)
    
    # ------------------------------------------------------------------------
    # 公開寫入介面 - Channel 類
    # ------------------------------------------------------------------------
    
    def write_channel_metadata(self, frame_id: int, metadata: dict):
        """
        寫入 Channel metadata.json
        
        Args:
            frame_id: 幀編號
            metadata: metadata 字典
        """
        if self.entity_type != "channel":
            raise RuntimeError(f"Channel not supported for entity_type: {self.entity_type}")
        
        # 建立 frame 目錄
        frame_dir = self.entity_path / f"frame_{frame_id:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # 寫入 metadata.json
        metadata_file = frame_dir / "metadata.json"
        task = WriteTask(
            task_type="json",
            file_path=metadata_file,
            data=metadata
        )
        self._enqueue_task(task)
    
    def write_channel_data(
        self, 
        frame_id: int, 
        tx_name: str, 
        rx_name: str, 
        tx_ant_idx: int, 
        rx_ant_idx: int,
        data: dict
    ):
        """
        寫入 Channel data pickle 檔案
        
        Args:
            frame_id: 幀編號
            tx_name: 發射端名稱（如 "bs_0"）
            rx_name: 接收端名稱（如 "veh_0"）
            tx_ant_idx: 發射端天線索引
            rx_ant_idx: 接收端天線索引
            data: channel 資料字典
        """
        if self.entity_type != "channel":
            raise RuntimeError(f"Channel not supported for entity_type: {self.entity_type}")
        
        # 建立 frame 目錄和 tx-rx 對目錄
        frame_dir = self.entity_path / f"frame_{frame_id:05d}"
        link_dir = frame_dir / f"{tx_name}-{rx_name}"
        link_dir.mkdir(parents=True, exist_ok=True)
        
        # 檔案路徑
        filename = f"tx_ant{tx_ant_idx}-rx_ant{rx_ant_idx}_frame{frame_id:05d}.pkl"
        filepath = link_dir / filename
        
        # 創建寫入任務
        task = WriteTask(
            task_type="pickle",
            file_path=filepath,
            data=data
        )
        self._enqueue_task(task)
    
    # ------------------------------------------------------------------------
    # 設定檔更新介面
    # ------------------------------------------------------------------------
    
    def update_dataset_config(self, config: Union[DatasetConfig, dict]):
        """更新全域資料集設定"""
        if isinstance(config, DatasetConfig):
            data = asdict(config)
        else:
            data = config
        
        config_path = self.dataset_root / "dataset_config.json"
        self._write_json_safe(config_path, data)
    
    def update_scenario_metadata(self, metadata: Union[ScenarioMetadata, dict]):
        """更新場景元資料"""
        if isinstance(metadata, ScenarioMetadata):
            data = asdict(metadata)
        else:
            data = metadata
        
        metadata_path = self.scenario_path / "metadata.json"
        self._write_json_safe(metadata_path, data)
    
    def update_basestation_config(self, config: Union[BasestationConfig, dict]):
        """更新基地台設定"""
        if self.entity_type != "basestation":
            raise RuntimeError("Only basestation entity can update basestation config")
        
        if isinstance(config, BasestationConfig):
            data = asdict(config)
        else:
            data = config
        
        config_path = self.entity_path / "config.json"
        self._write_json_safe(config_path, data)
    
    def update_channel_metadata(self, metadata: Union[ChannelMetadata, dict]):
        """更新 Channel 元資料"""
        if self.entity_type != "channel":
            raise RuntimeError("Only channel entity can update channel metadata")
        
        if isinstance(metadata, ChannelMetadata):
            data = asdict(metadata)
        else:
            data = metadata
        
        metadata_path = self.entity_path / "channel_metadata.json"
        self._write_json_safe(metadata_path, data)
    
    def update_class_mapping(self, mapping: dict):
        """更新語意分割類別對應表"""
        mapping_path = self.dataset_root / "class_mapping.json"
        self._write_json_safe(mapping_path, mapping)
    
    # ------------------------------------------------------------------------
    # 生命週期管理
    # ------------------------------------------------------------------------
    
    def flush(self):
        """等待所有待處理的寫入任務完成"""
        self._write_queue.join()
    
    def close(self):
        """關閉 writer，等待所有任務完成後結束執行緒"""
        # 等待佇列清空
        self.flush()
        
        # 發送結束信號
        self._shutdown_event.set()
        for _ in self._workers:
            self._write_queue.put(None)
        
        # 等待所有執行緒結束
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        print(f"[DatasetWriter] Closed: {self.entity_type}/{self.entity_id}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    # ------------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------------
    
    def get_queue_size(self) -> int:
        """取得目前佇列中的任務數量"""
        return self._write_queue.qsize()
    
    def is_queue_full(self) -> bool:
        """檢查佇列是否已滿"""
        return self._write_queue.full()


# ============================================================================
# README 生成器（事後使用）
# ============================================================================

def generate_readme(dataset_root: Union[str, Path], output_path: Optional[Path] = None):
    """
    生成 README.md（事後執行）
    
    Args:
        dataset_root: 資料集根目錄
        output_path: 輸出路徑，預設為 dataset_root/README.md
    """
    dataset_root = Path(dataset_root)
    
    if output_path is None:
        output_path = dataset_root / "README.md"
    
    # 讀取設定
    config_path = dataset_root / "dataset_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    # 統計場景
    scenarios = [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("scenario_")]
    
    # 生成 README
    readme_content = f"""# {config.get('dataset_name', 'CARLA-Sionna Dataset')}

## Overview

Version: {config.get('version', 'N/A')}
Created: {config.get('created_at', 'N/A')}
Coordinate System: {config.get('coordinate_system', 'N/A')}

## Description

{config.get('description', 'Multi-modal dataset for 6G wireless communication research.')}

## Dataset Structure

```
dataset/
├── README.md
├── dataset_config.json
├── class_mapping.json
└── scenario_XXX/
    ├── metadata.json
    ├── timestamps.csv
    ├── vehicles/
    │   └── veh_X/
    │       ├── trajectory.csv
    │       ├── rgb/
    │       ├── semantic/
    │       ├── lidar/
    │       └── radar/
    ├── basestations/
    │   └── bs_X/
    │       ├── config.json
    │       ├── rgb/
    │       ├── semantic/
    │       ├── lidar/
    │       └── radar/
    └── channel/
        ├── channel_metadata.json
        └── frame_XXXXX/
            ├── metadata.json
            └── {tx_name}-{rx_name}/
                └── tx_antX-rx_antY_frameXXXXX.pkl
```

## Scenarios

Total: {len(scenarios)} scenarios

| Scenario ID | Map | Duration | Vehicles | Basestations |
|-------------|-----|----------|----------|--------------|
"""
    
    for scenario_dir in sorted(scenarios):
        metadata_path = scenario_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            readme_content += f"| {meta.get('scenario_id', scenario_dir.name)} | {meta.get('map_name', 'N/A')} | {meta.get('duration_seconds', 'N/A')}s | {meta.get('num_vehicles', 'N/A')} | {meta.get('num_basestations', 'N/A')} |\n"
        else:
            readme_content += f"| {scenario_dir.name} | N/A | N/A | N/A | N/A |\n"
    
    readme_content += """
## Data Formats

- **Images (RGB/Semantic)**: PNG format
- **LiDAR**: NumPy array (.npy), shape (N, 4) for [x, y, z, intensity]
- **Radar**: NumPy array (.npy)
- **Channel**: Pickle files (.pkl) containing CIR parameters (a, tau, doppler, phi_r, phi_t, theta_r, theta_t) for each TX-RX antenna pair
- **Trajectory**: CSV format with pose and velocity information

## Usage

```python
from dataset_writer import DatasetWriter

# Writing data
with DatasetWriter("./dataset", "scenario_001", "vehicle", "veh_0") as writer:
    writer.write_rgb(frame_id=1, image=rgb_array)
    writer.write_lidar(frame_id=1, points=lidar_array)
```

## License

[Specify your license here]
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README generated: {output_path}")


if __name__ == "__main__":
    # 簡單測試
    print("DatasetWriter module loaded successfully.")
    print("Run 'python example_usage.py' for usage examples.")