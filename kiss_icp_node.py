#!/usr/bin/env python3
"""
KISS-ICP 独立 ROS2 节点
包装 slam_algorithms/kiss_icp.py 中的 KISSICP 类。
纯 LiDAR SLAM，不需要 IMU。BEST_EFFORT QoS。

与 lio_sam_node.py 共享相同的架构模式，
区别在于：无 IMU 订阅，使用 Point-to-Point ICP。
"""

# ---------- Open3D ML 硬拦截 ----------
import os, sys, types
os.environ["OPEN3D_DISABLE_ML_MODULE"] = "1"
sys.modules['open3d.ml'] = types.ModuleType('open3d.ml')

# ---------- 标准依赖 ----------
import traceback
import numpy as np
from array import array

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as TimeMsg

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import open3d as o3d
import transforms3d.quaternions as t3dq

from slam_algorithms import DEFAULT_CONFIGS, SensorData, SensorType
from slam_algorithms.kiss_icp import KISSICP


# ---------- 工具函数 ----------
def ros_time_to_float(t: TimeMsg) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9


def pc2_to_o3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    """将 PointCloud2 直接转换为 Open3D 点云（不做任何滤波）。"""
    if len(msg.data) == 0:
        return o3d.geometry.PointCloud()
    arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)  # x y z intensity
    xyz = arr[:, :3].astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def xyz_to_pc2(xyz: np.ndarray, frame_id: str, stamp: TimeMsg) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(xyz.shape[0])
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    if xyz.size == 0:
        msg.data = b""
    else:
        inten = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        msg.data = np.hstack((xyz.astype(np.float32), inten)).astype(np.float32).tobytes()
    return msg


def quat_xyzw_from_R(R: np.ndarray):
    wxyz = t3dq.mat2quat(R)  # (w,x,y,z)
    return (float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0]))


# ---------- 主节点 ----------
class KISSICPNode(Node):
    def __init__(self):
        super().__init__("kiss_icp_node")

        # use_sim_time
        try:
            if not self.has_parameter("use_sim_time"):
                self.declare_parameter("use_sim_time", True)
        except Exception:
            pass
        try:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        except Exception:
            pass

        # 坐标/外参
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_z", 2.0)

        # KISS-ICP 参数（与LIO-SAM参数名一致，便于对比）
        self.declare_parameter("voxel_size", 0.5)
        self.declare_parameter("min_range", 2.0)
        self.declare_parameter("max_range", 55.0)
        self.declare_parameter("z_min", -3.0)
        self.declare_parameter("z_max", 5.0)
        self.declare_parameter("max_corr_dist", 1.5)
        self.declare_parameter("icp_max_iter", 50)
        self.declare_parameter("icp_fitness_min", 0.25)
        self.declare_parameter("icp_rmse_max", 0.60)
        self.declare_parameter("keyframe_trans_th", 1.5)
        self.declare_parameter("keyframe_rot_deg_th", 12.0)
        self.declare_parameter("map_window_size", 20)
        self.declare_parameter("publish_map", True)
        self.declare_parameter("map_publish_interval", 5)

        # 读取参数
        gp = self.get_parameter
        self.odom_frame = gp("odom_frame").value
        self.base_frame = gp("base_frame").value
        self.lidar_z = float(gp("lidar_z").value)
        self.publish_map_flag = bool(gp("publish_map").value)
        self.map_publish_interval = int(gp("map_publish_interval").value)

        # 构建 KISSICP 配置
        kiss_config = dict(DEFAULT_CONFIGS["kiss_icp"])
        kiss_config.update({
            "voxel_size": float(gp("voxel_size").value),
            "min_range": float(gp("min_range").value),
            "max_range": float(gp("max_range").value),
            "z_min": float(gp("z_min").value),
            "z_max": float(gp("z_max").value),
            "max_corr_dist": float(gp("max_corr_dist").value),
            "icp_max_iter": int(gp("icp_max_iter").value),
            "icp_fitness_min": float(gp("icp_fitness_min").value),
            "icp_rmse_max": float(gp("icp_rmse_max").value),
            "keyframe_trans_th": float(gp("keyframe_trans_th").value),
            "keyframe_rot_deg_th": float(gp("keyframe_rot_deg_th").value),
            "map_window_size": int(gp("map_window_size").value),
        })

        # 创建 KISSICP 实例
        self.slam = KISSICP(kiss_config)
        self.slam.initialize()

        self.frame_count = 0

        # 路径消息
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.odom_frame

        # 通信
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, "/lidar/points", self.cb_lidar, sensor_qos)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.path_pub = self.create_publisher(Path, "/odom_path", 10)
        self.map_pub = self.create_publisher(PointCloud2, "/map", 1)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)

        self._publish_static_tf()

        self.get_logger().info(
            f"[状态] KISS-ICP 节点启动 (纯LiDAR 帧到帧 Point-to-Point ICP) | "
            f"voxel={kiss_config['voxel_size']} "
            f"range=[{kiss_config['min_range']},{kiss_config['max_range']}] "
            f"max_corr={kiss_config['max_corr_dist']}m "
            f"kf_trans={kiss_config['keyframe_trans_th']}m "
            f"kf_rot={kiss_config['keyframe_rot_deg_th']}deg"
        )

    # ---------- 静态TF ----------
    def _publish_static_tf(self):
        now = self.get_clock().now().to_msg()
        ts = TransformStamped()
        ts.header.stamp = now
        ts.header.frame_id = self.base_frame
        ts.child_frame_id = "lidar"
        ts.transform.translation.x = 0.0
        ts.transform.translation.y = 0.0
        ts.transform.translation.z = self.lidar_z
        ts.transform.rotation.x = 0.0
        ts.transform.rotation.y = 0.0
        ts.transform.rotation.z = 0.0
        ts.transform.rotation.w = 1.0
        self.tf_static.sendTransform([ts])
        self.get_logger().info("[状态] 发布静态TF: base_link->lidar")

    # ---------- LiDAR回调 ----------
    def cb_lidar(self, msg: PointCloud2):
        t = ros_time_to_float(msg.header.stamp)

        pcd = pc2_to_o3d(msg)
        if len(pcd.points) < 50:
            return

        # 构建 SensorData（纯LiDAR，无IMU）
        sensor_data = SensorData()
        sensor_data.timestamp = t
        sensor_data.lidar = pcd

        # 调用 KISSICP.process_frame
        try:
            self.slam.process_frame(sensor_data)
        except Exception as e:
            self.get_logger().error(
                f"[状态] KISS-ICP处理异常: {e}\n{traceback.format_exc()}")
            return

        self.frame_count += 1

        # 发布所有输出
        self._publish_all(t)

    # ---------- 发布 ----------
    def _publish_all(self, t: float):
        R = self.slam.orientation
        pos = self.slam.position

        # lidar位姿 -> base_link位姿（减去lidar高度偏移）
        base_pos = pos.copy()
        base_pos[2] -= self.lidar_z
        base_R = R.copy()

        stamp = self._float_to_time(t)
        qx, qy, qz, qw = quat_xyzw_from_R(base_R)

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = float(base_pos[0])
        odom.pose.pose.position.y = float(base_pos[1])
        odom.pose.pose.position.z = float(base_pos[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        cov_xy, cov_z, cov_rot = 0.05, 0.1, 0.02
        odom.pose.covariance = array('d', [
            cov_xy, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, cov_xy, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, cov_z, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, cov_rot, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, cov_rot, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, cov_rot
        ])
        self.odom_pub.publish(odom)

        # TF: odom -> base_link
        ts = TransformStamped()
        ts.header = odom.header
        ts.child_frame_id = self.base_frame
        ts.transform.translation.x = odom.pose.pose.position.x
        ts.transform.translation.y = odom.pose.pose.position.y
        ts.transform.translation.z = odom.pose.pose.position.z
        ts.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(ts)

        # Path
        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(ps)
        if len(self.path_msg.poses) > 8000:
            self.path_msg.poses = self.path_msg.poses[-8000:]
        self.path_pub.publish(self.path_msg)

        # Map（间隔发布）
        if self.publish_map_flag and (self.frame_count % self.map_publish_interval == 0):
            local_map = self.slam.get_map()
            if local_map is not None and len(local_map.points) > 0:
                xyz = np.asarray(local_map.points, dtype=np.float32)
                self.map_pub.publish(xyz_to_pc2(xyz, self.odom_frame, stamp))

    def _float_to_time(self, t: float) -> TimeMsg:
        sec = int(t)
        nsec = int((t - sec) * 1e9)
        return TimeMsg(sec=sec, nanosec=nsec)


# ---------- 入口 ----------
def main():
    rclpy.init()
    node = KISSICPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
