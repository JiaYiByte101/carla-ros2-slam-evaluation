"""
KISS-ICP 算法 —— 基于官方 kiss-icp 库 (v1.2.3) 的 C++ 引擎

不再使用 Open3D ICP 手写配准，而是直接调用官方：
- kiss_icp.kiss_icp.KissICP        → 完整 pipeline（配准+地图+自适应阈值）
- kiss_icp.mapping.VoxelHashMap     → C++ 体素哈希地图
- kiss_icp.registration.Registration → C++ Point-to-Point ICP（含 robust kernel）
- kiss_icp.threshold.AdaptiveThreshold → 自适应对应距离


"""
import math
from typing import Optional, Dict, Any, List

import numpy as np
import open3d as o3d
import transforms3d.quaternions as t3dq
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from .base_slam import BaseSLAMAlgorithm, SensorData, SensorType, SLAMType

# ---- 官方 kiss-icp 库 ----
from kiss_icp.kiss_icp import KissICP as _KissICPEngine
from kiss_icp.config.parser import KISSConfig


class KISSICP(BaseSLAMAlgorithm):
    """KISS-ICP wrapper：将官方 C++ 引擎包装为 BaseSLAMAlgorithm 接口。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "KISS_ICP"
        self.slam_type = SLAMType.LIDAR_ONLY
        self.required_sensors = [SensorType.LIDAR]

        # ---- 从用户配置中读取参数 ----
        self.voxel_size = config.get("voxel_size", 0.5)
        self.max_range = config.get("max_range", 55.0)
        self.min_range = config.get("min_range", 2.0)
        self.z_min = config.get("z_min", -3.0)
        self.z_max = config.get("z_max", 5.0)
        self.max_points_per_voxel = config.get("max_points_per_voxel", 20)
        self.initial_threshold = config.get("max_corr_dist", 2.0)

        # 地面去除参数
        # LiDAR 安装高度约 2m，地面在 z ≈ -2.0（lidar 坐标系）
        # ground_z: 低于此高度的点视为地面并去除
        # 默认 -1.5 去除大部分地面，保留路沿、低矮障碍物
        self.ground_z = config.get("ground_z", -1.5)

        # ---- 构建官方 KISSConfig ----
        self._kiss_config = self._build_config()

        # ---- 运行时状态 ----
        self._engine: Optional[_KissICPEngine] = None

        # 公开状态（与 kiss_icp_node.py 的接口约定一致）
        self.position = np.zeros(3, dtype=np.float64)
        self.orientation = np.eye(3, dtype=np.float64)
        self.current_pose = np.eye(4, dtype=np.float64)

        self.keyframes: List[Dict[str, Any]] = []
        self.trajectory: List[List[float]] = []
        self.is_initialized = False
        self.frame_count = 0

    #  配置构建

    def _build_config(self) -> KISSConfig:
        """将用户参数映射到官方 KISSConfig。"""
        cfg = KISSConfig()

        # 数据预处理
        cfg.data.max_range = self.max_range
        cfg.data.min_range = self.min_range
        cfg.data.deskew = False  # CARLA 点云已同步，无需运动补偿

        # 地图
        cfg.mapping.voxel_size = self.voxel_size
        cfg.mapping.max_points_per_voxel = self.max_points_per_voxel

        # 配准
        cfg.registration.max_num_iterations = 500
        cfg.registration.convergence_criterion = 0.0001

        # 自适应阈值
        cfg.adaptive_threshold.initial_threshold = self.initial_threshold
        cfg.adaptive_threshold.min_motion_th = 0.1

        return cfg

    #  生命周期

    def initialize(self) -> bool:
        self._engine = _KissICPEngine(config=self._kiss_config)

        self.position = np.zeros(3, dtype=np.float64)
        self.orientation = np.eye(3, dtype=np.float64)
        self.current_pose = np.eye(4, dtype=np.float64)

        self.keyframes.clear()
        self.trajectory.clear()
        self.is_initialized = True
        self.frame_count = 0
        return True


    #  主处理流程

    def process_frame(self, sensor_data: SensorData) -> Odometry:
        if not sensor_data.has_sensor(SensorType.LIDAR):
            raise ValueError("KISS-ICP requires LiDAR data.")

        timestamp = sensor_data.timestamp
        if not self.is_initialized:
            self.initialize()

        # 1. Open3D PointCloud → numpy (N, 3)
        pts = np.asarray(sensor_data.lidar.points, dtype=np.float64)
        if pts.shape[0] < 50:
            return self._make_odom(timestamp)

        # 2. 预处理：高度滤波 + 地面去除
        #    地面是无特征平面，对 yaw 旋转几乎没有约束力。
        #    去除地面点后，建筑墙面/路灯/车辆等垂直结构主导配准，
        #    旋转估计精度大幅提升（尤其在拐弯处）。
        mask = (pts[:, 2] >= self.ground_z) & (pts[:, 2] <= self.z_max)
        pts = pts[mask]
        if pts.shape[0] < 50:
            return self._make_odom(timestamp)

        # 3. 调用官方 C++ 引擎
        #    register_frame(frame, timestamps)
        #    - frame: (N, 3) np.float64
        #    - timestamps: 每点时间戳（用于去畸变），空数组 = 不去畸变
        self._engine.register_frame(pts, np.array([]))

        # 4. 提取最新位姿
        pose = self._engine.last_pose  # (4, 4) np.ndarray
        self.current_pose = pose.copy()
        self.position = pose[:3, 3].copy()
        self.orientation = pose[:3, :3].copy()

        self.frame_count += 1
        self._record_trajectory(timestamp)
        return self._make_odom(timestamp)


    #  轨迹记录

    def _record_trajectory(self, timestamp: float):
        quat = t3dq.mat2quat(self.orientation)  # [w, x, y, z]
        self.trajectory.append([
            timestamp,
            self.position[0], self.position[1], self.position[2],
            quat[1], quat[2], quat[3], quat[0],  # TUM 格式: x y z qx qy qz qw
        ])


    #  Odometry 消息

    def _make_odom(self, timestamp: float) -> Odometry:
        odom = Odometry()
        odom.header.stamp.sec = int(timestamp)
        odom.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(self.position[0])
        odom.pose.pose.position.y = float(self.position[1])
        odom.pose.pose.position.z = float(self.position[2])

        quat = t3dq.mat2quat(self.orientation)
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])

        odom.pose.covariance[0] = 0.1
        odom.pose.covariance[7] = 0.1
        odom.pose.covariance[14] = 0.1
        odom.pose.covariance[21] = 0.05
        odom.pose.covariance[28] = 0.05
        odom.pose.covariance[35] = 0.05
        return odom


    #  BaseSLAMAlgorithm 接口

    def get_map(self) -> Optional[o3d.geometry.PointCloud]:
        """从官方 VoxelHashMap 中提取地图点云。"""
        if self._engine is None:
            return None
        try:
            map_pts = self._engine.local_map.point_cloud()
            if map_pts.shape[0] == 0:
                return None
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(map_pts)
            return pcd
        except Exception:
            return None

    def reset(self):
        self.initialize()

    def get_trajectory(self) -> np.ndarray:
        if not self.trajectory:
            return np.empty((0, 8))
        return np.asarray(self.trajectory)

    def get_algorithm_info(self) -> Dict[str, Any]:
        info = super().get_algorithm_info()
        map_points = 0
        if self._engine is not None:
            try:
                map_points = self._engine.local_map.point_cloud().shape[0]
            except Exception:
                pass
        info.update({
            "num_keyframes": len(self.keyframes),
            "map_points": map_points,
            "frame_count": self.frame_count,
        })
        return info
