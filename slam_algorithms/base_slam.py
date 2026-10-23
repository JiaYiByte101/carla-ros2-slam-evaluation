"""
SLAM算法基类定义
支持多种SLAM算法的统一接口，包括LiDAR、IMU、视觉传感器
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import open3d as o3d
from sensor_msgs.msg import PointCloud2, Imu, Image
from nav_msgs.msg import Odometry
from enum import Enum


class SensorType(Enum):
    """传感器类型枚举"""
    LIDAR = "lidar"
    IMU = "imu"
    CAMERA_RGB = "camera_rgb"
    CAMERA_DEPTH = "camera_depth"


class SLAMType(Enum):
    """SLAM算法类型枚举"""
    LIDAR_ONLY = "lidar_only"           # 纯LiDAR SLAM
    LIDAR_IMU = "lidar_imu"             # LiDAR-IMU融合SLAM
    VISUAL = "visual"                   # 视觉SLAM
    VISUAL_INERTIAL = "visual_inertial" # 视觉-惯性SLAM
    MULTI_SENSOR = "multi_sensor"       # 多传感器融合SLAM


class SensorData:
    """传感器数据封装类"""

    def __init__(self):
        self.timestamp: float = 0.0
        self.lidar: Optional[o3d.geometry.PointCloud] = None
        self.imu: Optional[Imu] = None
        self.rgb_image: Optional[Image] = None
        self.depth_image: Optional[Image] = None
        self.imu_sequence: List[Imu] = []  # ordered batch spanning last lidar frame

    def has_sensor(self, sensor_type: SensorType) -> bool:
        """检查是否包含指定传感器数据"""
        if sensor_type == SensorType.LIDAR:
            return self.lidar is not None
        elif sensor_type == SensorType.IMU:
            return self.imu is not None
        elif sensor_type == SensorType.CAMERA_RGB:
            return self.rgb_image is not None
        elif sensor_type == SensorType.CAMERA_DEPTH:
            return self.depth_image is not None
        return False


class BaseSLAMAlgorithm(ABC):
    """SLAM算法基类，定义统一接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "base_slam"
        self.slam_type = SLAMType.LIDAR_ONLY
        self.required_sensors = [SensorType.LIDAR]
        self.is_initialized = False
        self.current_pose = np.eye(4)

    @abstractmethod
    def initialize(self) -> bool:
        """初始化算法"""
        pass

    @abstractmethod
    def process_frame(self, sensor_data: SensorData) -> Odometry:
        """
        处理传感器帧（新的统一接口）
        Args:
            sensor_data: 包含所有传感器数据的封装对象
        Returns:
            估计的里程计
        """
        pass

    # 保持向后兼容的旧接口
    def process_lidar_frame(self,
                           points: o3d.geometry.PointCloud,
                           timestamp: float,
                           imu_data: Optional[Imu] = None) -> Odometry:
        """
        处理LiDAR帧（兼容性接口）
        Args:
            points: 点云数据
            timestamp: 时间戳
            imu_data: IMU数据（可选）
        Returns:
            估计的里程计
        """
        sensor_data = SensorData()
        sensor_data.timestamp = timestamp
        sensor_data.lidar = points
        sensor_data.imu = imu_data
        return self.process_frame(sensor_data)

    @abstractmethod
    def get_map(self) -> Optional[o3d.geometry.PointCloud]:
        """获取当前地图"""
        pass

    @abstractmethod
    def reset(self):
        """重置算法状态"""
        pass

    @abstractmethod
    def get_trajectory(self) -> np.ndarray:
        """
        获取轨迹
        Returns:
            轨迹数组，每行为[timestamp, x, y, z, qx, qy, qz, qw]
        """
        pass

    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        return {
            "name": self.name,
            "slam_type": self.slam_type.value,
            "required_sensors": [sensor.value for sensor in self.required_sensors],
            "config": self.config,
            "is_initialized": self.is_initialized
        }

    def supports_sensor(self, sensor_type: SensorType) -> bool:
        """检查算法是否支持指定传感器"""
        return sensor_type in self.required_sensors

    def validate_sensor_data(self, sensor_data: SensorData) -> bool:
        """验证传感器数据是否满足算法要求"""
        for sensor_type in self.required_sensors:
            if not sensor_data.has_sensor(sensor_type):
                return False
        return True


class SLAMAlgorithmManager:
    """SLAM算法管理器"""

    def __init__(self):
        self.algorithms = {}
        self.current_algorithm = None
        self.sensor_buffer = {}  # 传感器数据缓冲区

    def register_algorithm(self, name: str, algorithm_class, config: Dict[str, Any]):
        """注册新算法"""
        self.algorithms[name] = {
            "class": algorithm_class,
            "config": config
        }

    def switch_algorithm(self, name: str) -> bool:
        """切换到指定算法"""
        if name not in self.algorithms:
            return False

        if self.current_algorithm:
            self.current_algorithm.reset()

        algo_info = self.algorithms[name]
        self.current_algorithm = algo_info["class"](algo_info["config"])
        return self.current_algorithm.initialize()

    def get_current_algorithm(self) -> Optional[BaseSLAMAlgorithm]:
        """获取当前算法实例"""
        return self.current_algorithm

    def list_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """列出所有可用算法及其信息"""
        algorithm_info = {}
        for name, algo_data in self.algorithms.items():
            # 创建临时实例获取算法信息
            temp_instance = algo_data["class"](algo_data["config"])
            algorithm_info[name] = temp_instance.get_algorithm_info()
        return algorithm_info

    def get_algorithms_by_type(self, slam_type: SLAMType) -> List[str]:
        """根据SLAM类型获取算法列表"""
        matching_algorithms = []
        for name, algo_data in self.algorithms.items():
            temp_instance = algo_data["class"](algo_data["config"])
            if temp_instance.slam_type == slam_type:
                matching_algorithms.append(name)
        return matching_algorithms

    def get_algorithms_by_sensor(self, sensor_type: SensorType) -> List[str]:
        """根据传感器类型获取支持的算法列表"""
        matching_algorithms = []
        for name, algo_data in self.algorithms.items():
            temp_instance = algo_data["class"](algo_data["config"])
            if temp_instance.supports_sensor(sensor_type):
                matching_algorithms.append(name)
        return matching_algorithms

    def process_sensor_data(self, sensor_data: SensorData) -> Optional[Odometry]:
        """处理传感器数据"""
        if not self.current_algorithm:
            return None

        # 验证传感器数据
        if not self.current_algorithm.validate_sensor_data(sensor_data):
            return None

        return self.current_algorithm.process_frame(sensor_data)

    def get_current_map(self) -> Optional[o3d.geometry.PointCloud]:
        """获取当前地图"""
        if not self.current_algorithm:
            return None
        return self.current_algorithm.get_map()

    def get_current_trajectory(self) -> np.ndarray:
        """获取当前轨迹"""
        if not self.current_algorithm:
            return np.empty((0, 8))
        return self.current_algorithm.get_trajectory()

    def reset_current_algorithm(self):
        """重置当前算法"""
        if self.current_algorithm:
            self.current_algorithm.reset()
