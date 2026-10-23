import math
import random
import time
import numpy as np
import carla
import cv2  # 用于图像处理

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter as RclpyParameter
from builtin_interfaces.msg import Time as TimeMsg
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2, PointField, Imu, Image, CameraInfo, CompressedImage
from std_msgs.msg import Header, String
from geometry_msgs.msg import TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import StaticTransformBroadcaster
from cv_bridge import CvBridge  # 用于图像转换

# 导入数据录制模块
from data_manager import DataRecorder

# 导入车辆控制模块
from vehicle_control import AutoController, ManualController

# ========= 配置参数 =========
HOST = "192.168.1.243"
PORT = 2000
ROLE_NAME = "ego_vehicle"
FIXED_DT = 0.05  # 20 Hz

# 车辆控制参数
TARGET_SPEED_KMH = 12.0       # 降低速度避免数据质量问题
LOOKAHEAD_M = 5.0             # 减少前瞻距离
WHEELBASE_M = 2.80
KP_THROTTLE = 0.08
MAX_STEER = 0.5

# 观察者相机参数
CAMERA_DISTANCE = 15.0
CAMERA_HEIGHT = 8.0
CAMERA_PITCH = -25.0

# LiDAR配置 - 为SLAM优化
LIDAR_CFG = dict(
    channels="32",                # 32线足够，避免过多计算
    points_per_second="400000",   # 适中的点密度
    rotation_frequency="20",      # 匹配主循环
    range="80",
    upper_fov="15",               # 增加上视野
    lower_fov="-25",              # 减少下视野
    sensor_tick="0.05"            # 20Hz同步
)

# 相机配置
CAMERA_RGB_CFG = dict(
    image_size_x="1280",          # 分辨率宽度
    image_size_y="720",           # 分辨率高度
    fov="90",                     # 视场角
    sensor_tick="0.1",           # 20Hz同步
    gamma="2.2",
    shutter_speed="200.0",
    iso="100",
    motion_blur_intensity="0.45",
    motion_blur_max_distortion="0.35"
)

CAMERA_DEPTH_CFG = dict(
    image_size_x="1280",
    image_size_y="720",
    fov="90",
    sensor_tick="0.1"
)

# 相机位置配置（相对于车辆）
CAMERA_POSITIONS = {
    "front": carla.Transform(carla.Location(x=0.8, z=1.7)),  # 前视相机
    "rear": carla.Transform(carla.Location(x=-2.0, z=1.5), carla.Rotation(yaw=180)),  # 后视相机
    "left": carla.Transform(carla.Location(y=-0.9, z=1.5), carla.Rotation(yaw=-90)),  # 左侧相机
    "right": carla.Transform(carla.Location(y=0.9, z=1.5), carla.Rotation(yaw=90))  # 右侧相机
}

# ========= 工具函数 =========
def kmh(v):
    return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def clamp(x, a, b):
    return max(a, min(b, x))

def wrap(a):
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

def euler_to_quat(roll, pitch, yaw):
    """欧拉角转四元数"""
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = sy*cp*sr + cy*sp*cr
    z = sy*cp*cr - cy*sp*sr
    return x, y, z, w

# ========= 纯跟踪控制 =========
# 注意：以下函数已迁移到vehicle_control.AutoController中
# 保留此处仅作为参考

def choose_next_waypoint(cur_wp, lookahead):
    """[已弃用] 选择下一个路点 - 已迁移到AutoController"""
    pass

def pure_pursuit_control(world, veh):
    """[已弃用] 纯跟踪控制 - 已迁移到AutoController"""
    # 为了向后兼容，返回默认值
    return 0.0, 0.0, 0.0

def update_spectator_camera(world, vehicle):
    """更新观察者相机位置"""
    if not vehicle or not vehicle.is_alive:
        return
    
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    
    yaw_rad = math.radians(vehicle_rotation.yaw)
    camera_x = vehicle_location.x - CAMERA_DISTANCE * math.cos(yaw_rad)
    camera_y = vehicle_location.y - CAMERA_DISTANCE * math.sin(yaw_rad)
    camera_z = vehicle_location.z + CAMERA_HEIGHT
    
    camera_location = carla.Location(x=camera_x, y=camera_y, z=camera_z)
    camera_rotation = carla.Rotation(
        pitch=CAMERA_PITCH,
        yaw=vehicle_rotation.yaw,
        roll=0.0
    )
    
    camera_transform = carla.Transform(camera_location, camera_rotation)
    spectator = world.get_spectator()
    spectator.set_transform(camera_transform)

# ========= ROS2发布节点 =========
class CarlaRos(Node):
    def __init__(self):
        super().__init__("carla_slam_publisher")

        # 安全设置use_sim_time
        try:
            self.declare_parameter("use_sim_time", True)
        except Exception:
            pass

        try:
            self.set_parameters([RclpyParameter(
                "use_sim_time", RclpyParameter.Type.BOOL, True
            )])
        except Exception:
            try:
                current_value = self.get_parameter("use_sim_time").value
                if not current_value:
                    self.get_logger().warn("use_sim_time is False, SLAM may have timing issues")
            except Exception:
                self.get_logger().warn("Could not check use_sim_time parameter")

        # 创建发布者 - 使用标准话题名
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.lidar_pub = self.create_publisher(PointCloud2, "/lidar/points", 10)
        self.lidar_lio_pub = self.create_publisher(PointCloud2, "/lidar/points_lio", 10)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", 10)

        # 相机发布者
        self.camera_pubs = {}
        self.camera_info_pubs = {}
        self.depth_pubs = {}

        # 前视RGB相机
        self.camera_pubs["front"] = self.create_publisher(
            Image, "/camera/front/image_raw", 10
        )
        self.camera_info_pubs["front"] = self.create_publisher(
            CameraInfo, "/camera/front/camera_info", 10
        )

        # 前视深度相机
        self.depth_pubs["front"] = self.create_publisher(
            Image, "/camera/front/depth/image_raw", 10
        )

        # Ground Truth发布器 - 新增功能
        self.gt_odom_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth/path", 10)

        # 录制命令订阅器 - 新增功能
        self.record_cmd_sub = self.create_subscription(
            String, "/recording/command", self.recording_command_callback, 10
        )

        # 车辆控制订阅器 - 新增功能
        self.manual_control_sub = self.create_subscription(
            String, "/vehicle/manual_control", self.manual_control_callback, 10
        )
        self.control_mode_sub = self.create_subscription(
            String, "/vehicle/control_mode", self.control_mode_callback, 10
        )

        # 可选：压缩图像发布者
        self.compressed_pubs = {}
        self.compressed_pubs["front"] = self.create_publisher(
            CompressedImage, "/camera/front/image_compressed", 10
        )

        # CV Bridge用于图像转换
        self.bridge = CvBridge()

        # 仅发布必要的静态TF
        self.tf_static = StaticTransformBroadcaster(self)

        # 内部状态
        self._last_stamp = TimeMsg()
        self._pending_lidar = None  # 延迟发布LiDAR，确保时间戳正确
        self._pending_camera_rgb = None   # 延迟发布相机，同理
        self._pending_camera_depth = None
        self._lidar_count = 0
        self._imu_count = 0
        self._camera_counts = {"front": 0, "rear": 0, "left": 0, "right": 0}
        self._depth_count = 0

        # Ground Truth状态 - 新增功能
        self._gt_count = 0
        self._gt_T0_inv = None  # 第一帧GT位姿的逆，用于将GT转换到以起始点为原点的坐标系
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = "odom"  # Ground Truth在odom坐标系下
        self.gt_trajectory_data = []  # 保存TUM格式数据

        # 数据录制器 - 新增功能
        self.data_recorder = DataRecorder(base_dir="datasets", auto_create_dirs=True)
        self._recording_status_count = 0

        # 车辆控制器 - 新增功能
        self.auto_controller = AutoController()
        self.manual_controller = ManualController()
        self.current_controller = self.auto_controller  # 默认自动模式
        self.current_controller.activate()
        self.control_mode = "auto"

        # 发布静态TF
        self.publish_static_tf()

        self.get_logger().info("CARLA-ROS2 SLAM Publisher initialized")
        self.get_logger().info("Using standard ROS topic names:")
        self.get_logger().info("  - LiDAR: /lidar/points")
        self.get_logger().info("  - IMU: /imu/data")
        self.get_logger().info("  - Camera RGB: /camera/front/image_raw")
        self.get_logger().info("  - Camera Depth: /camera/front/depth/image_raw")
        self.get_logger().info("  - Ground Truth Odom: /ground_truth/odom")
        self.get_logger().info("  - Ground Truth Path: /ground_truth/path")

    def publish_static_tf(self):
        """发布静态TF变换 - 最小化设置"""
        now = self.get_clock().now().to_msg()
        tfs = []

        def add_tf(parent, child, xyz=(0, 0, 0), rpy=(0, 0, 0)):
            x, y, z = xyz
            roll, pitch, yaw = rpy
            qx, qy, qz, qw = euler_to_quat(roll, pitch, yaw)

            ts = TransformStamped()
            ts.header.stamp = now
            ts.header.frame_id = parent
            ts.child_frame_id = child
            ts.transform.translation.x = float(x)
            ts.transform.translation.y = float(y)
            ts.transform.translation.z = float(z)
            ts.transform.rotation.x = qx
            ts.transform.rotation.y = qy
            ts.transform.rotation.z = qz
            ts.transform.rotation.w = qw
            tfs.append(ts)

        # 发布传感器相对于base_link的固定变换
        add_tf("base_link", "lidar", (0, 0, 2.0), (0, 0, 0))
        add_tf("base_link", "imu", (0, 0, 1.0), (0, 0, 0))
        add_tf("base_link", "camera_front", (0.8, 0, 1.7), (0, 0, 0))
        add_tf("base_link", "camera_front_depth", (0.8, 0, 1.7), (0, 0, 0))

        if tfs:
            self.tf_static.sendTransform(tfs)
            self.get_logger().info(f"Published {len(tfs)} static transforms")
            self.get_logger().info("Expected TF tree:")
            self.get_logger().info("  odom -> base_link (by SLAM)")
            self.get_logger().info("  base_link -> lidar (static)")
            self.get_logger().info("  base_link -> imu (static)")
            self.get_logger().info("  base_link -> camera_front (static)")
            self.get_logger().info("  base_link -> camera_front_depth (static)")

    def set_sim_time(self, stamp: TimeMsg):
        """设置仿真时间戳"""
        self._last_stamp = stamp

    def publish_clock(self, sim_seconds: float):
        """发布仿真时钟"""
        msg = Clock()
        msg.clock = TimeMsg(
            sec=int(sim_seconds),
            nanosec=int((sim_seconds % 1.0) * 1e9)
        )
        self.clock_pub.publish(msg)
        self.set_sim_time(msg.clock)

    def publish_lidar(self, carla_points):
        """发布LiDAR点云数据
        - /lidar/points: 原始4字段格式 (x,y,z,intensity) 供KISS-ICP等使用
        - /lidar/points_lio: Velodyne兼容格式 (ring=UINT16) 供LIO-SAM使用
        """
        self._lidar_count += 1

        # 解析CARLA点云数据
        raw_data = carla_points.raw_data
        points = np.frombuffer(raw_data, dtype=np.float32).reshape(-1, 4)

        # CARLA到ROS的坐标系转换: Y' = -Y
        points_ros = points.copy()
        points_ros[:, 1] = -points_ros[:, 1]

        # 过滤点云
        distances = np.linalg.norm(points_ros[:, :3], axis=1)
        distance_mask = (distances > 1.0) & (distances < 80.0)
        intensity_mask = points_ros[:, 3] > 0.0
        valid_mask = distance_mask & intensity_mask
        points_filtered = points_ros[valid_mask]

        stamp = self._last_stamp

        # === 发布原始格式到 /lidar/points (4×FLOAT32, 16字节/点) ===
        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id="lidar")
        msg.height = 1
        msg.width = len(points_filtered)
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
        msg.data = points_filtered.astype(np.float32).tobytes()
        self.lidar_pub.publish(msg)

        # === 发布Velodyne兼容格式到 /lidar/points_lio (供LIO-SAM使用) ===
        n_pts = len(points_filtered)
        if n_pts > 0:
            xyz = points_filtered[:, :3]

            # 计算ring: 根据垂直角度映射到0~31
            xy_dist = np.linalg.norm(xyz[:, :2], axis=1)
            vertical_angle = np.degrees(np.arctan2(xyz[:, 2], xy_dist))
            lower_fov, upper_fov, n_scan = -25.0, 15.0, 32
            ring = np.clip(
                np.round((vertical_angle - lower_fov) / (upper_fov - lower_fov) * (n_scan - 1)),
                0, n_scan - 1
            ).astype(np.uint16)

            # 计算time: 水平角度归一化到0~1
            horiz_angle = np.arctan2(xyz[:, 1], xyz[:, 0])
            point_time = ((horiz_angle + np.pi) / (2.0 * np.pi)).astype(np.float32)

            # 打包为Velodyne兼容的二进制格式:
            # x(f32) y(f32) z(f32) intensity(f32) ring(u16) pad(u16) time(f32) = 24字节
            dt_lio = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
                ('ring', '<u2'), ('pad', '<u2'), ('time', '<f4')
            ])
            cloud_lio = np.zeros(n_pts, dtype=dt_lio)
            cloud_lio['x'] = xyz[:, 0]
            cloud_lio['y'] = xyz[:, 1]
            cloud_lio['z'] = xyz[:, 2]
            cloud_lio['intensity'] = points_filtered[:, 3]
            cloud_lio['ring'] = ring
            cloud_lio['pad'] = 0
            cloud_lio['time'] = point_time

            msg_lio = PointCloud2()
            msg_lio.header = Header(stamp=stamp, frame_id="lidar")
            msg_lio.height = 1
            msg_lio.width = n_pts
            msg_lio.fields = [
                PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
                PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
                PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='ring',      offset=16, datatype=PointField.UINT16,  count=1),
                PointField(name='time',      offset=20, datatype=PointField.FLOAT32, count=1),
            ]
            msg_lio.is_bigendian = False
            msg_lio.point_step = 24
            msg_lio.row_step = msg_lio.point_step * n_pts
            msg_lio.is_dense = True
            msg_lio.data = cloud_lio.tobytes()
            self.lidar_lio_pub.publish(msg_lio)

        # 定期打印调试信息
        if self._lidar_count % 100 == 0:
            self.get_logger().info(
                f"LiDAR: {self._lidar_count} frames, "
                f"current: {len(points_filtered)}/{len(points)} points (filtered/total)"
            )

    def publish_imu(self, imu_meas: carla.IMUMeasurement):
        """发布IMU数据（确保时间戳严格递增，避免GTSAM dt<=0崩溃）"""
        self._imu_count += 1

        # 确保IMU时间戳严格递增（GTSAM预积分要求 dt > 0）
        stamp = self._last_stamp
        if not hasattr(self, '_last_imu_stamp_ns'):
            self._last_imu_stamp_ns = 0
        cur_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if cur_ns <= self._last_imu_stamp_ns:
            # 时间戳未递增，加1ms保证严格递增
            cur_ns = self._last_imu_stamp_ns + 1_000_000  # +1ms
            stamp = TimeMsg(
                sec=int(cur_ns // 1_000_000_000),
                nanosec=int(cur_ns % 1_000_000_000)
            )
        self._last_imu_stamp_ns = cur_ns

        msg = Imu()
        msg.header = Header(
            stamp=stamp,
            frame_id="imu"
        )

        # 姿态未知，设置协方差为-1
        msg.orientation_covariance[0] = -1.0

        # 角速度 (rad/s) - CARLA到ROS坐标系转换
        msg.angular_velocity.x = float(imu_meas.gyroscope.x)
        msg.angular_velocity.y = -float(imu_meas.gyroscope.y)  # Y轴取反
        msg.angular_velocity.z = float(imu_meas.gyroscope.z)

        # 线加速度 (m/s^2) - CARLA到ROS坐标系转换
        msg.linear_acceleration.x = float(imu_meas.accelerometer.x)
        msg.linear_acceleration.y = -float(imu_meas.accelerometer.y)  # Y轴取反
        msg.linear_acceleration.z = float(imu_meas.accelerometer.z)

        self.imu_pub.publish(msg)

    def publish_camera_rgb(self, carla_image, camera_name="front"):
        """发布RGB相机图像"""
        self._camera_counts[camera_name] += 1

        # 将CARLA图像转换为numpy数组
        img_bgra = np.array(carla_image.raw_data).reshape(
            carla_image.height, carla_image.width, 4
        )
        img_bgr = img_bgra[:, :, :3]  # 去掉alpha通道
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 创建ROS Image消息
        msg = self.bridge.cv2_to_imgmsg(img_rgb, encoding="rgb8")
        msg.header = Header(
            stamp=self._last_stamp,
            frame_id=f"camera_{camera_name}"
        )

        # 发布原始图像
        self.camera_pubs[camera_name].publish(msg)

        # 发布压缩图像（可选）
        if camera_name in self.compressed_pubs:
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "jpeg"
            _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            compressed_msg.data = buffer.tobytes()
            self.compressed_pubs[camera_name].publish(compressed_msg)

        # 发布相机信息
        camera_info = self.generate_camera_info(carla_image, camera_name)
        self.camera_info_pubs[camera_name].publish(camera_info)

        # 定期打印调试信息
        if self._camera_counts[camera_name] % 100 == 0:
            self.get_logger().info(
                f"Camera {camera_name}: {self._camera_counts[camera_name]} frames"
            )

    def publish_camera_depth(self, carla_image, camera_name="front"):
        """发布深度相机图像"""
        self._depth_count += 1

        # 将CARLA深度图像转换为numpy数组
        img_bgra = np.array(carla_image.raw_data).reshape(
            carla_image.height, carla_image.width, 4
        )

        # CARLA深度图像编码：R + G*256 + B*256*256 (单位：米)
        # 需要转换为米为单位的深度值
        R = img_bgra[:, :, 2].astype(np.float32)
        G = img_bgra[:, :, 1].astype(np.float32)
        B = img_bgra[:, :, 0].astype(np.float32)

        # 深度值计算 (单位：米)
        depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0) * 1000.0

        # 创建ROS Image消息
        msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        msg.header = Header(
            stamp=self._last_stamp,
            frame_id=f"camera_{camera_name}_depth"
        )

        self.depth_pubs[camera_name].publish(msg)

        # 定期打印调试信息
        if self._depth_count % 100 == 0:
            self.get_logger().info(
                f"Depth camera {camera_name}: {self._depth_count} frames"
            )

    def generate_camera_info(self, carla_image, camera_name):
        """生成相机内参信息"""
        camera_info = CameraInfo()
        camera_info.header = Header(
            stamp=self._last_stamp,
            frame_id=f"camera_{camera_name}"
        )

        camera_info.height = carla_image.height
        camera_info.width = carla_image.width

        # 相机内参矩阵 (假设FOV=90度)
        fov_rad = math.radians(carla_image.fov)
        fx = float(carla_image.width / (2.0 * math.tan(fov_rad / 2.0)))
        fy = fx  # 假设像素是正方形
        cx = float(carla_image.width / 2.0)
        cy = float(carla_image.height / 2.0)

        # K矩阵 (3x3) - 必须是float类型
        camera_info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        # D畸变参数 (假设无畸变)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # R旋转矩阵 (单位矩阵) - 必须是float类型
        camera_info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        # P投影矩阵 (3x4) - 必须是float类型
        camera_info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        camera_info.distortion_model = "plumb_bob"
        camera_info.binning_x = 0
        camera_info.binning_y = 0

        return camera_info

    def publish_ground_truth(self, vehicle_transform: carla.Transform, sim_time: float):
        """发布Ground Truth位姿数据（相对于起始点的位姿）"""
        self._gt_count += 1

        # CARLA坐标系到ROS坐标系转换
        # CARLA: X前, Y右, Z上
        # ROS:   X前, Y左, Z上
        carla_pos = vehicle_transform.location
        carla_rot = vehicle_transform.rotation

        # 位置转换 (Y取反)
        pos_x = float(carla_pos.x)
        pos_y = -float(carla_pos.y)  # Y轴取反
        pos_z = float(carla_pos.z)

        # 旋转转换 (需要适当调整角度)
        # CARLA角度是度，需要转换为弧度，然后转换为四元数
        roll_rad = math.radians(carla_rot.roll)
        pitch_rad = math.radians(carla_rot.pitch)
        yaw_rad = math.radians(carla_rot.yaw)

        # 考虑坐标系转换，Y轴取反会影响yaw角
        yaw_rad = -yaw_rad  # yaw角取反

        # 构建当前帧的4x4位姿矩阵
        qx, qy, qz, qw = euler_to_quat(roll_rad, pitch_rad, yaw_rad)
        # 四元数转旋转矩阵
        R = self._quat_to_rotation_matrix(qw, qx, qy, qz)
        T_cur = [[R[0][0], R[0][1], R[0][2], pos_x],
                 [R[1][0], R[1][1], R[1][2], pos_y],
                 [R[2][0], R[2][1], R[2][2], pos_z],
                 [0.0,     0.0,     0.0,     1.0  ]]

        # 记录第一帧位姿的逆
        if self._gt_T0_inv is None:
            # 计算逆矩阵: T_inv = [R^T, -R^T @ t; 0, 1]
            Rt = [[R[0][0], R[1][0], R[2][0]],
                  [R[0][1], R[1][1], R[2][1]],
                  [R[0][2], R[1][2], R[2][2]]]
            t_inv = [-(Rt[0][0]*pos_x + Rt[0][1]*pos_y + Rt[0][2]*pos_z),
                     -(Rt[1][0]*pos_x + Rt[1][1]*pos_y + Rt[1][2]*pos_z),
                     -(Rt[2][0]*pos_x + Rt[2][1]*pos_y + Rt[2][2]*pos_z)]
            self._gt_T0_inv = [[Rt[0][0], Rt[0][1], Rt[0][2], t_inv[0]],
                               [Rt[1][0], Rt[1][1], Rt[1][2], t_inv[1]],
                               [Rt[2][0], Rt[2][1], Rt[2][2], t_inv[2]],
                               [0.0,      0.0,      0.0,      1.0     ]]
            self.get_logger().info(
                f"[状态] GT起始位姿已记录: ({pos_x:.1f}, {pos_y:.1f}, {pos_z:.1f}), "
                f"yaw={math.degrees(yaw_rad):.1f}°")

        # 计算相对位姿: T_rel = T0_inv @ T_cur
        T0i = self._gt_T0_inv
        T_rel = [[0.0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    T_rel[i][j] += T0i[i][k] * T_cur[k][j]

        # 从相对位姿提取位置和四元数
        rel_x = T_rel[0][3]
        rel_y = T_rel[1][3]
        rel_z = T_rel[2][3]
        rel_qx, rel_qy, rel_qz, rel_qw = self._rotation_matrix_to_quat(
            [[T_rel[0][0], T_rel[0][1], T_rel[0][2]],
             [T_rel[1][0], T_rel[1][1], T_rel[1][2]],
             [T_rel[2][0], T_rel[2][1], T_rel[2][2]]])

        # 创建Odometry消息
        odom_msg = Odometry()
        odom_msg.header = Header(
            stamp=self._last_stamp,
            frame_id="odom"
        )
        odom_msg.child_frame_id = "base_link"

        # 设置相对位置
        odom_msg.pose.pose.position.x = rel_x
        odom_msg.pose.pose.position.y = rel_y
        odom_msg.pose.pose.position.z = rel_z

        # 设置相对方向
        odom_msg.pose.pose.orientation.x = rel_qx
        odom_msg.pose.pose.orientation.y = rel_qy
        odom_msg.pose.pose.orientation.z = rel_qz
        odom_msg.pose.pose.orientation.w = rel_qw

        # 设置速度 (可选，此处设为0)
        odom_msg.twist.twist.linear.x = 0.0
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = 0.0

        # 发布Odometry消息
        self.gt_odom_pub.publish(odom_msg)

        # 更新轨迹路径
        pose_stamped = PoseStamped()
        pose_stamped.header = odom_msg.header
        pose_stamped.pose = odom_msg.pose.pose

        self.gt_path_msg.header.stamp = odom_msg.header.stamp
        self.gt_path_msg.poses.append(pose_stamped)

        # 限制路径长度，避免内存过度使用
        if len(self.gt_path_msg.poses) > 10000:
            self.gt_path_msg.poses = self.gt_path_msg.poses[-10000:]

        # 发布路径消息
        self.gt_path_pub.publish(self.gt_path_msg)

        # 保存TUM格式数据 (timestamp tx ty tz qx qy qz qw)
        tum_entry = f"{sim_time:.6f} {rel_x:.6f} {rel_y:.6f} {rel_z:.6f} {rel_qx:.6f} {rel_qy:.6f} {rel_qz:.6f} {rel_qw:.6f}"
        self.gt_trajectory_data.append(tum_entry)

        # 定期打印调试信息
        if self._gt_count % 100 == 0:
            self.get_logger().info(
                f"Ground Truth: {self._gt_count} frames, "
                f"位置: ({rel_x:.2f}, {rel_y:.2f}, {rel_z:.2f}), "
                f"轨迹点数: {len(self.gt_path_msg.poses)}"
            )

    @staticmethod
    def _quat_to_rotation_matrix(w, x, y, z):
        """四元数 (w,x,y,z) 转 3x3 旋转矩阵（嵌套列表）"""
        return [
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)  ],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)  ],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
        ]

    @staticmethod
    def _rotation_matrix_to_quat(R):
        """3x3 旋转矩阵 → 四元数 (x,y,z,w)"""
        tr = R[0][0] + R[1][1] + R[2][2]
        if tr > 0:
            s = 0.5 / math.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R[2][1] - R[1][2]) * s
            y = (R[0][2] - R[2][0]) * s
            z = (R[1][0] - R[0][1]) * s
        elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
            s = 2.0 * math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2])
            w = (R[2][1] - R[1][2]) / s
            x = 0.25 * s
            y = (R[0][1] + R[1][0]) / s
            z = (R[0][2] + R[2][0]) / s
        elif R[1][1] > R[2][2]:
            s = 2.0 * math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2])
            w = (R[0][2] - R[2][0]) / s
            x = (R[0][1] + R[1][0]) / s
            y = 0.25 * s
            z = (R[1][2] + R[2][1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1])
            w = (R[1][0] - R[0][1]) / s
            x = (R[0][2] + R[2][0]) / s
            y = (R[1][2] + R[2][1]) / s
            z = 0.25 * s
        return (x, y, z, w)

    def save_ground_truth_trajectory(self, filename: str = "ground_truth_trajectory.txt"):
        """保存Ground Truth轨迹为TUM格式文件"""
        try:
            with open(filename, 'w') as f:
                f.write("# Ground Truth trajectory in TUM format\n")
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                for entry in self.gt_trajectory_data:
                    f.write(entry + "\n")

            self.get_logger().info(f"Ground Truth轨迹已保存到: {filename} ({len(self.gt_trajectory_data)} 个位姿)")
        except Exception as e:
            self.get_logger().error(f"保存Ground Truth轨迹失败: {e}")

    def start_data_recording(self, scenario_name: str = None, include_optional: bool = False) -> bool:
        """开始数据录制"""
        if self.data_recorder.is_recording:
            self.get_logger().warn("已在录制中，无法重复开始")
            return False

        try:
            # 准备配置数据
            config_data = {
                "dataset_info": {
                    "carla_map": "Town01",  # 可以从CARLA获取实际地图名
                    "vehicle_model": "vehicle.tesla.model3",
                    "recorder_version": "1.0.0"
                },
                "sensors": {
                    "lidar": {
                        "channels": 32,
                        "range": 80,
                        "rotation_frequency": 20,
                        "topic": "/lidar/points"
                    },
                    "imu": {
                        "topic": "/imu/data"
                    },
                    "camera_rgb": {
                        "resolution": "1280x720",
                        "fov": 90,
                        "topic": "/camera/front/image_raw"
                    },
                    "camera_depth": {
                        "resolution": "1280x720",
                        "fov": 90,
                        "topic": "/camera/front/depth/image_raw"
                    }
                }
            }

            success = self.data_recorder.start_recording(
                scenario_name=scenario_name,
                include_optional=include_optional,
                config_data=config_data
            )

            if success:
                self.get_logger().info("✅ 数据录制已开始")
            else:
                self.get_logger().error("❌ 数据录制启动失败")

            return success

        except Exception as e:
            self.get_logger().error(f"录制启动异常: {e}")
            return False

    def stop_data_recording(self) -> bool:
        """停止数据录制"""
        if not self.data_recorder.is_recording:
            self.get_logger().warn("当前未在录制")
            return False

        try:
            success = self.data_recorder.stop_recording()
            if success:
                self.get_logger().info("✅ 数据录制已停止")
            else:
                self.get_logger().error("❌ 数据录制停止失败")

            return success

        except Exception as e:
            self.get_logger().error(f"停止录制异常: {e}")
            return False

    def get_recording_status_info(self) -> str:
        """获取录制状态信息字符串"""
        status = self.data_recorder.get_recording_status()
        if status["is_recording"]:
            duration = int(status["recording_duration"])
            dataset_name = status["current_dataset"]
            if dataset_name:
                dataset_name = dataset_name.split("/")[-1]  # 只显示目录名
            return f"录制中: {dataset_name} ({duration}s)"
        else:
            return "未录制"

    def recording_command_callback(self, msg: String):
        """处理录制控制命令"""
        command = msg.data.strip().lower()
        parts = command.split()

        if not parts:
            return

        cmd = parts[0]

        if cmd == "start":
            if self.data_recorder.is_recording:
                self.get_logger().warn("已在录制中，无法重复开始")
                return

            # 获取场景名称（如果提供）
            scenario_name = parts[1] if len(parts) > 1 else None
            success = self.start_data_recording(scenario_name)

            if success:
                self.get_logger().info("✅ 录制已开始")
            else:
                self.get_logger().error("❌ 录制启动失败")

        elif cmd == "stop":
            if not self.data_recorder.is_recording:
                self.get_logger().warn("当前未在录制")
                return

            success = self.stop_data_recording()
            if success:
                self.get_logger().info("✅ 录制已停止")
            else:
                self.get_logger().error("❌ 录制停止失败")

        elif cmd == "status":
            status = self.get_recording_status_info()
            self.get_logger().info(f"录制状态: {status}")

            # 显示数据集列表
            datasets = self.data_recorder.list_datasets()
            if datasets:
                self.get_logger().info(f"已有数据集: {', '.join(datasets[-5:])}")  # 显示最近5个
            else:
                self.get_logger().info("暂无数据集")

        else:
            self.get_logger().warn(f"未知录制命令: {command}")

    def manual_control_callback(self, msg: String):
        """处理手动控制命令"""
        if self.control_mode != "manual":
            return  # 只在手动模式下响应

        command_data = msg.data.strip()
        parts = command_data.split(':')

        if len(parts) != 2:
            return

        command, action = parts

        # 处理按键命令
        if action == "pressed":
            self.manual_controller.apply_key_command(command, True)
        elif action == "released":
            self.manual_controller.apply_key_command(command, False)

    def control_mode_callback(self, msg: String):
        """处理控制模式切换命令"""
        new_mode = msg.data.strip().lower()

        if new_mode == self.control_mode:
            return  # 已经是当前模式

        # 停用当前控制器
        self.current_controller.deactivate()

        # 切换控制器
        if new_mode == "manual":
            self.current_controller = self.manual_controller
            self.control_mode = "manual"
        elif new_mode == "auto":
            self.current_controller = self.auto_controller
            self.control_mode = "auto"
        else:
            self.get_logger().warn(f"未知控制模式: {new_mode}")
            return

        # 激活新控制器
        self.current_controller.activate()
        self.get_logger().info(f"✅ 切换到{self.control_mode}控制模式")

    def get_vehicle_control(self, world, vehicle, dt: float):
        """获取车辆控制命令"""
        if self.current_controller and self.current_controller.is_active:
            return self.current_controller.update(world, vehicle, dt)
        else:
            # 没有激活的控制器，返回空控制
            return carla.VehicleControl()

    def get_control_status_info(self) -> str:
        """获取控制状态信息"""
        if self.current_controller:
            status = self.current_controller.get_status_info()
            mode = status.get("control_mode", "unknown")
            if mode == "manual":
                throttle = status.get("throttle", 0.0)
                brake = status.get("brake", 0.0)
                steer = status.get("steer", 0.0)
                return f"手动控制 T:{throttle:.2f} B:{brake:.2f} S:{steer:.2f}"
            else:
                return f"自动控制 (目标速度: {self.auto_controller.target_speed_kmh:.1f} km/h)"
        else:
            return "无控制器"

# ========= 主函数 =========
def main():
    print("="*70)
    print("CARLA-KISS-ICP SLAM集成系统")
    print("="*70)
    
    # 连接CARLA
    print("[1/6] 连接CARLA服务器...")
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        print(f"      已连接到CARLA世界: {world.get_map().name}")
    except Exception as e:
        print(f"      连接失败: {e}")
        return

    # 设置同步模式
    print("[2/6] 配置同步模式...")
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)
    print(f"      同步模式: {FIXED_DT}s ({1/FIXED_DT:.0f} Hz)")

    # 生成车辆
    print("[3/6] 生成自车...")
    bp = world.get_blueprint_library()
    vehicle_bp = bp.find("vehicle.tesla.model3")
    vehicle_bp.set_attribute("role_name", ROLE_NAME)

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    
    if not vehicle:
        print("      车辆生成失败!")
        return
    
    # 调整到车道中心
    world_map = world.get_map()
    waypoint = world_map.get_waypoint(
        vehicle.get_location(), 
        project_to_road=True, 
        lane_type=carla.LaneType.Driving
    )
    if waypoint:
        transform = waypoint.transform
        transform.rotation.pitch = 0.0
        transform.rotation.roll = 0.0
        vehicle.set_transform(transform)
    
    vehicle.set_simulate_physics(True)
    print(f"      车辆已生成，位置: ({vehicle.get_location().x:.1f}, {vehicle.get_location().y:.1f})")

    # 创建传感器
    print("[4/6] 创建传感器...")
    
    # LiDAR传感器
    lidar_bp = bp.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("role_name", "lidar")
    for key, value in LIDAR_CFG.items():
        lidar_bp.set_attribute(key, value)
    
    lidar = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(0, 0, 2.0)),
        attach_to=vehicle
    )
    
    # IMU传感器
    imu_bp = bp.find("sensor.other.imu")
    imu_bp.set_attribute("role_name", "imu")
    imu = world.spawn_actor(
        imu_bp,
        carla.Transform(carla.Location(0, 0, 1.0)),
        attach_to=vehicle
    )

    # RGB相机传感器（前视）
    camera_rgb_bp = bp.find("sensor.camera.rgb")
    camera_rgb_bp.set_attribute("role_name", "camera_front")
    for key, value in CAMERA_RGB_CFG.items():
        camera_rgb_bp.set_attribute(key, value)

    camera_rgb = world.spawn_actor(
        camera_rgb_bp,
        CAMERA_POSITIONS["front"],
        attach_to=vehicle
    )

    # 深度相机传感器（前视）
    camera_depth_bp = bp.find("sensor.camera.depth")
    camera_depth_bp.set_attribute("role_name", "camera_front_depth")
    for key, value in CAMERA_DEPTH_CFG.items():
        camera_depth_bp.set_attribute(key, value)

    camera_depth = world.spawn_actor(
        camera_depth_bp,
        CAMERA_POSITIONS["front"],
        attach_to=vehicle
    )

    print(f"      LiDAR: {LIDAR_CFG['channels']}线, {LIDAR_CFG['range']}m测距")
    print(f"      IMU: 标准6自由度传感器")
    print(f"      RGB相机: {CAMERA_RGB_CFG['image_size_x']}x{CAMERA_RGB_CFG['image_size_y']}, FOV={CAMERA_RGB_CFG['fov']}°")
    print(f"      深度相机: {CAMERA_DEPTH_CFG['image_size_x']}x{CAMERA_DEPTH_CFG['image_size_y']}, FOV={CAMERA_DEPTH_CFG['fov']}°")

    # 初始化ROS2
    print("[5/6] 初始化ROS2...")
    rclpy.init()
    ros_node = CarlaRos()

    # 设置传感器回调
    def on_lidar_data(data):
        # 不在回调中直接发布（此时_last_stamp是上一帧的时间戳）
        # 而是暂存数据，在主循环中publish_clock之后发布
        ros_node._pending_lidar = data

    def on_imu_data(data):
        ros_node.publish_imu(data)

    def on_camera_rgb_data(data):
        # 与LiDAR相同的修复：不在回调中直接发布（此时_last_stamp是上一帧的）
        # 暂存数据，在主循环中publish_clock之后发布
        ros_node._pending_camera_rgb = data

    def on_camera_depth_data(data):
        ros_node._pending_camera_depth = data

    lidar.listen(on_lidar_data)
    imu.listen(on_imu_data)
    camera_rgb.listen(on_camera_rgb_data)
    camera_depth.listen(on_camera_depth_data)
    print("      传感器已连接到ROS2话题")

    # 系统就绪
    print("[6/6] 系统就绪!")
    print()
    print("发布的ROS2话题:")
    print("  /lidar/points (sensor_msgs/PointCloud2)")
    print("  /imu/data (sensor_msgs/Imu)")
    print("  /camera/front/image_raw (sensor_msgs/Image)")
    print("  /camera/front/camera_info (sensor_msgs/CameraInfo)")
    print("  /camera/front/image_compressed (sensor_msgs/CompressedImage)")
    print("  /camera/front/depth/image_raw (sensor_msgs/Image)")
    print("  /clock (rosgraph_msgs/Clock)")
    print("  /ground_truth/odom (nav_msgs/Odometry)")
    print("  /ground_truth/path (nav_msgs/Path)")
    print()
    print("数据录制控制:")
    print("  按 'R' 键 - 开始/停止数据录制")
    print("  按 'L' 键 - 列出已有数据集")
    print()
    print("车辆控制模式:")
    print("  默认: 自动驾驶 (Pure Pursuit)")
    print("  手动控制: 运行 python manual_control_input.py")
    print("  切换模式: Tab键 (在手动控制程序中)")
    print("开始主循环... (Ctrl+C停止)")
    print("-" * 70)
    
    last_move_time = time.time()
    frame_count = 0
    start_time = time.time()

    # 键盘输入监听 (简单版本，用于录制控制)
    last_key_check = time.time()
    recording_toggle_time = 0
    
    try:
        while rclpy.ok():
            # 推进仿真
            world.tick()
            
            # 获取仿真时间戳
            snapshot = world.get_snapshot()
            sim_time = snapshot.timestamp.elapsed_seconds
            frame_count += 1
            
            # 发布时钟（更新_last_stamp为当前帧时间戳）
            ros_node.publish_clock(sim_time)

            # 发布挂起的LiDAR数据（在publish_clock之后，确保时间戳与GT一致）
            if ros_node._pending_lidar is not None:
                ros_node.publish_lidar(ros_node._pending_lidar)
                ros_node._pending_lidar = None

            # 发布挂起的相机数据（与LiDAR同理，修复ORB-SLAM3时间戳偏移）
            if ros_node._pending_camera_rgb is not None:
                ros_node.publish_camera_rgb(ros_node._pending_camera_rgb, "front")
                ros_node._pending_camera_rgb = None
            if ros_node._pending_camera_depth is not None:
                ros_node.publish_camera_depth(ros_node._pending_camera_depth, "front")
                ros_node._pending_camera_depth = None

            # 发布Ground Truth数据
            vehicle_transform = vehicle.get_transform()
            ros_node.publish_ground_truth(vehicle_transform, sim_time)

            # 车辆控制 - 使用新的控制器系统
            control = ros_node.get_vehicle_control(world, vehicle, FIXED_DT)
            vehicle.apply_control(control)
            
            # 更新观察者相机
            update_spectator_camera(world, vehicle)
            
            # 检查车辆卡住
            current_speed = kmh(vehicle.get_velocity())
            if current_speed > 1.0:
                last_move_time = time.time()
            elif time.time() - last_move_time > 6.0:
                print("      车辆卡住，重新生成...")
                new_spawn = random.choice(spawn_points)
                vehicle.set_transform(new_spawn)
                last_move_time = time.time()
            
            # 处理ROS回调
            rclpy.spin_once(ros_node, timeout_sec=0.0)
            
            # 定期状态报告
            if frame_count % 200 == 0:  # 每10秒报告一次
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                pos = vehicle.get_location()

                print(f"[状态] 帧数: {frame_count}, "
                      f"FPS: {fps:.1f}, "
                      f"速度: {current_speed:.1f} km/h, "
                      f"位置: ({pos.x:.0f}, {pos.y:.0f})")

                # 检查数据流
                print(f"[数据] LiDAR: {ros_node._lidar_count} 帧, "
                      f"IMU: {ros_node._imu_count} 帧, "
                      f"RGB相机: {ros_node._camera_counts.get('front', 0)} 帧, "
                      f"深度相机: {ros_node._depth_count} 帧, "
                      f"Ground Truth: {ros_node._gt_count} 帧")

                # 显示录制状态 - 新增功能
                recording_status = ros_node.get_recording_status_info()
                print(f"[录制] {recording_status}")

                # 显示控制状态 - 新增功能
                control_status = ros_node.get_control_status_info()
                print(f"[控制] {control_status}")

    except KeyboardInterrupt:
        print("\n[停止] 请求关闭")
    except Exception as e:
        print(f"\n[错误] 主循环异常: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[清理] 销毁资源...")

        # 停止数据录制 - 新增功能
        if 'ros_node' in locals():
            if ros_node.data_recorder.is_recording:
                print("        停止数据录制...")
                ros_node.stop_data_recording()

            # 停用车辆控制器 - 新增功能
            if hasattr(ros_node, 'current_controller') and ros_node.current_controller:
                print("        停用车辆控制器...")
                ros_node.current_controller.deactivate()

        # 停止传感器
        for sensor in [lidar, imu, camera_rgb, camera_depth]:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()

        # 销毁车辆
        if vehicle and vehicle.is_alive:
            vehicle.destroy()

        # 恢复世界设置
        try:
            world.apply_settings(original_settings)
            print("        世界设置已恢复")
        except:
            pass

        # 关闭ROS2
        if 'ros_node' in locals():
            # 保存Ground Truth轨迹 - 新增功能
            ros_node.save_ground_truth_trajectory("carla_ground_truth_trajectory.txt")
            ros_node.destroy_node()
        rclpy.shutdown()

        print("        清理完成!")
        print("="*70)

if __name__ == "__main__":
    main()
