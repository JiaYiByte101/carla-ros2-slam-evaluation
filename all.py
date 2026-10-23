# ---------- Open3D ML 硬拦截 ----------
import os, sys, types
os.environ["OPEN3D_DISABLE_ML_MODULE"] = "1"
sys.modules['open3d.ml'] = types.ModuleType('open3d.ml')

# ---------- 标准依赖 ----------
import math
import numpy as np
from collections import deque
from array import array

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as TimeMsg

from sensor_msgs.msg import PointCloud2, Imu, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import open3d as o3d
import transforms3d.quaternions as t3dq
import transforms3d.axangles as t3da

# ---------- 工具 ----------
def ros_time_to_float(t: TimeMsg) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9

def so3_exp(omega: np.ndarray):
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta)*K + (1 - math.cos(theta))*(K @ K)

def se3_mul(A, B):
    return A @ B

def se3_inv(T):
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def se3_from_mat4(T):
    """保证旋转正交化"""
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    T[:3, :3] = U @ Vt
    return T

def yaw_from_R(R):
    return math.atan2(R[1,0], R[0,0])

def yaw_from_T(T):
    return yaw_from_R(T[:3, :3])

def quat_xyzw_from_R(R: np.ndarray):
    wxyz = t3dq.mat2quat(R)  # (w,x,y,z)
    return (float(wxyz[1]), float(wxyz[2]), float(wxyz[3]), float(wxyz[0]))

def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    if len(msg.data) == 0: return np.empty((0,3), dtype=np.float32)
    arr = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)  # x y z intensity
    return arr[:, :3]

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

def make_o3d_cloud(pts_xyz: np.ndarray, voxel_size=0.5,
                   min_range=1.0, max_range=80.0,
                   z_min=-3.0, z_max=5.0):
    if pts_xyz.size == 0:
        return None
    d = np.linalg.norm(pts_xyz, axis=1)
    m = (d > min_range) & (d < max_range) & (pts_xyz[:, 2] > z_min) & (pts_xyz[:, 2] < z_max)
    pts = pts_xyz[m]
    if pts.shape[0] < 50:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd

def estimate_normals_inplace(pcd: o3d.geometry.PointCloud, voxel_size: float):
    if pcd is None or len(pcd.points) == 0: return
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0*voxel_size + 0.2, max_nn=30)
    )

def ground_remove_ransac(pcd: o3d.geometry.PointCloud, dist_th: float = 0.15, n_iter: int = 100):
    if pcd is None or len(pcd.points) < 200: return pcd
    try:
        _, inliers = pcd.segment_plane(distance_threshold=dist_th, ransac_n=3, num_iterations=n_iter)
        return pcd.select_by_index(inliers, invert=True)
    except Exception:
        return pcd

# ---------- 轻量闭环 ----------
def o3d_downsample(pcd, voxel):
    return pcd if voxel <= 0 else pcd.voxel_down_sample(voxel)

def icp_point_to_plane(src, tgt, init_T, max_corr, max_iter):
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr*2.0, max_nn=30))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    return reg

class LightLoopCloser:
    """
    近邻再访的小窗口闭环（超轻量）
    依赖键值：
      keyframe = {'T': 4x4(odom<-lidar), 'pcd': open3d点云(已在odom)}
    """
    def __init__(self, p):
        self.enable  = p['lc_enable']
        self.step    = p['lc_check_every_n_keyframes']
        self.window  = p['lc_recent_window']
        self.dist_th = p['lc_dist_thresh']
        self.yaw_th  = math.radians(p['lc_yaw_thresh_deg'])
        self.voxel   = p['lc_icp_voxel']
        self.max_it  = p['lc_icp_max_iter']
        self.fit_min = p['lc_icp_fitness_min']
        self.rmse_max= p['lc_icp_rmse_max']
        self.alpha   = p['lc_correction_alpha']
        self.T_corr  = np.eye(4)

    def maybe_close(self, kf_idx, kfs, logger=None):
        if (not self.enable) or (kf_idx % self.step != 0) or (kf_idx < 6):
            return False

        cur = kfs[kf_idx]
        T_cur = cur['T']; pos_cur = T_cur[:3,3]; yaw_cur = yaw_from_T(T_cur)

        start = max(0, kf_idx - self.window)
        candidates = []
        for j in range(start, kf_idx-5):
            Tj = kfs[j]['T']
            d  = Tj[:3,3] - pos_cur
            dist = np.linalg.norm(d[:2])
            if dist > self.dist_th:
                continue
            dyaw = abs(yaw_from_T(Tj) - yaw_cur)
            dyaw = min(dyaw, abs(2*np.pi - dyaw))
            if dyaw > self.yaw_th:
                continue
            candidates.append((j, dist, dyaw))

        if not candidates:
            return False

        j, _, _ = min(candidates, key=lambda x: x[1])

        pc_cur = o3d_downsample(cur['pcd'], self.voxel)
        pc_ref = o3d_downsample(kfs[j]['pcd'], self.voxel)
        T_init = se3_from_mat4(se3_inv(kfs[j]['T']) @ T_cur)

        max_corr = max(0.8, 1.2*self.voxel)
        try:
            reg = icp_point_to_plane(pc_cur, pc_ref, T_init, max_corr, self.max_it)
            fitness, rmse = reg.fitness, reg.inlier_rmse
        except Exception as e:
            if logger: logger.warn(f"[LC] ICP failed: {e}")
            return False

        if fitness < self.fit_min or rmse > self.rmse_max:
            if logger: logger.info(f"[LC] reject (fit={fitness:.2f}, rmse={rmse:.2f})")
            return False

        # 误差（在ref系）：E = T_icp^{-1} * T_expect
        T_expect = T_init
        T_icp    = reg.transformation
        E_ref = se3_inv(T_icp) @ T_expect

        # 提回世界：Delta_w = T_wj * E_ref * T_wj^{-1}
        T_wj = kfs[j]['T']
        Delta_w = T_wj @ E_ref @ se3_inv(T_wj)

        # 幂插值（柔和融合）
        R = Delta_w[:3,:3]; t = Delta_w[:3,3]
        axis, angle = t3da.mat2axangle(R)
        R_a = t3da.axangle2mat(axis, angle * self.alpha)
        t_a = t * self.alpha
        Delta_a = np.eye(4); Delta_a[:3,:3] = R_a; Delta_a[:3,3] = t_a

        self.T_corr = se3_from_mat4(self.T_corr @ Delta_a)
        if logger: logger.info(f"[LC] accepted -> update T_corr (fit={fitness:.2f}, rmse={rmse:.2f})")
        return True

    def apply(self, T_est):
        return se3_from_mat4(self.T_corr @ T_est)

# ---------- 主节点 ----------
class LidarImuSubmapGICP(Node):
    def __init__(self):
        super().__init__("lidar_imu_submap_gicp")

        # use_sim_time
        try:
            if not self.has_parameter("use_sim_time"):
                self.declare_parameter("use_sim_time", True)
        except Exception: pass
        try:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        except Exception: pass

        # 坐标/外参
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_z", 2.0)  # base_link->lidar 高度

        # 过滤/ICP 参数（偏稳）
        self.declare_parameter("voxel_size", 0.5)
        self.declare_parameter("max_corr_dist", 1.2)
        self.declare_parameter("min_range", 2.0)
        self.declare_parameter("max_range", 60.0)
        self.declare_parameter("z_min", -2.0)
        self.declare_parameter("z_max", 3.0)
        self.declare_parameter("icp_max_iter", 60)
        self.declare_parameter("icp_fitness_min", 0.25)
        self.declare_parameter("icp_rmse_max", 0.60)

        # 地面剔除
        self.declare_parameter("use_ground_removal", True)
        self.declare_parameter("ground_ransac_dist", 0.15)

        # 关键帧/子图
        self.declare_parameter("keyframe_trans_th", 1.0)      # m
        self.declare_parameter("keyframe_rot_deg_th", 8.0)    # deg
        self.declare_parameter("submap_size", 12)
        self.declare_parameter("publish_submap", False)

        # 初值策略
        self.declare_parameter("trans_init_scale", 0.6)       # 0=禁用平移初值

        # 轻量闭环参数
        self.declare_parameter('lc_enable', True)
        self.declare_parameter('lc_check_every_n_keyframes', 10)
        self.declare_parameter('lc_recent_window', 100)
        self.declare_parameter('lc_dist_thresh', 25.0)
        self.declare_parameter('lc_yaw_thresh_deg', 20.0)
        self.declare_parameter('lc_icp_voxel', 0.6)
        self.declare_parameter('lc_icp_max_iter', 30)
        self.declare_parameter('lc_icp_fitness_min', 0.35)
        self.declare_parameter('lc_icp_rmse_max', 0.50)
        self.declare_parameter('lc_correction_alpha', 0.35)

        # 读取参数
        gp = self.get_parameter
        self.odom_frame = gp("odom_frame").value
        self.base_frame = gp("base_frame").value
        self.lidar_z = float(gp("lidar_z").value)

        self.voxel = float(gp("voxel_size").value)
        self.max_corr = float(gp("max_corr_dist").value)
        self.min_range = float(gp("min_range").value)
        self.max_range = float(gp("max_range").value)
        self.z_min = float(gp("z_min").value)
        self.z_max = float(gp("z_max").value)
        self.icp_max_iter = int(gp("icp_max_iter").value)
        self.icp_fitness_min = float(gp("icp_fitness_min").value)
        self.icp_rmse_max = float(gp("icp_rmse_max").value)

        self.use_ground = bool(gp("use_ground_removal").value)
        self.ground_dist = float(gp("ground_ransac_dist").value)

        self.kf_trans_th = float(gp("keyframe_trans_th").value)
        self.kf_rot_deg_th = float(gp("keyframe_rot_deg_th").value)
        self.submap_size = int(gp("submap_size").value)
        self.publish_submap = bool(gp("publish_submap").value)

        self.trans_init_scale = float(gp("trans_init_scale").value)

        # 闭环器
        p = {
            'lc_enable': gp('lc_enable').value,
            'lc_check_every_n_keyframes': gp('lc_check_every_n_keyframes').value,
            'lc_recent_window': gp('lc_recent_window').value,
            'lc_dist_thresh': gp('lc_dist_thresh').value,
            'lc_yaw_thresh_deg': gp('lc_yaw_thresh_deg').value,
            'lc_icp_voxel': gp('lc_icp_voxel').value,
            'lc_icp_max_iter': gp('lc_icp_max_iter').value,
            'lc_icp_fitness_min': gp('lc_icp_fitness_min').value,
            'lc_icp_rmse_max': gp('lc_icp_rmse_max').value,
            'lc_correction_alpha': gp('lc_correction_alpha').value,
        }
        self.loop = LightLoopCloser(p) if p['lc_enable'] else None

        # 通信（Sensor QoS）
        sensor_qos = QoSProfile(depth=5,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.pc_sub = self.create_subscription(PointCloud2, "/lidar/points", self.cb_lidar, sensor_qos)
        self.imu_sub = self.create_subscription(Imu, "/imu/data", self.cb_imu, sensor_qos)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.path_pub = self.create_publisher(Path, "/odom_path", 10)
        self.map_pub  = self.create_publisher(PointCloud2, "/map", 1)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_static      = StaticTransformBroadcaster(self)

        # 状态
        self.last_pc_time = None
        self.last_delta_T = np.eye(4)
        self.last_delta_dt = 0.05  # 上一帧间隔（秒），默认一个CARLA tick
        self.T_odom_lidar = np.eye(4)

        self.path_msg = Path(); self.path_msg.header.frame_id = self.odom_frame
        self.imu_buf = deque(maxlen=4000)
        self._gicp_reject_count = 0
        self._gicp_accept_count = 0

        # 关键帧/子图
        self.keyframes = []    # list of {'T': 4x4, 'pcd': o3d.geometry.PointCloud(odom)}
        self.submap = None
        self.last_key_T = np.eye(4)

        # 发布静态TF
        self.publish_static_tf()

        self.get_logger().info("LiDAR+IMU Submap GICP + LightLoopCloser started.")

    # ---------- 静态TF ----------
    def publish_static_tf(self):
        now = self.get_clock().now().to_msg()
        def make_tf(parent, child, xyz=(0.0,0.0,0.0)):
            ts = TransformStamped()
            ts.header.stamp = now
            ts.header.frame_id = parent
            ts.child_frame_id = child
            ts.transform.translation.x = float(xyz[0])
            ts.transform.translation.y = float(xyz[1])
            ts.transform.translation.z = float(xyz[2])
            ts.transform.rotation.x = 0.0
            ts.transform.rotation.y = 0.0
            ts.transform.rotation.z = 0.0
            ts.transform.rotation.w = 1.0
            return ts
        tfs = [make_tf(self.base_frame, "lidar", (0.0, 0.0, self.lidar_z))]
        self.tf_static.sendTransform(tfs)
        self.get_logger().info("Published static TF: base_link->lidar")

    # ---------- IMU ----------
    def cb_imu(self, msg: Imu):
        t = ros_time_to_float(msg.header.stamp)
        w = np.array([msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z], dtype=np.float64)
        self.imu_buf.append((t, w))

    def integrate_gyro(self, t0: float, t1: float) -> np.ndarray:
        if t1 <= t0 or not self.imu_buf:
            return np.eye(3)
        R = np.eye(3); prev_t = t0
        for (ti, w) in self.imu_buf:
            if ti <= t0: continue
            if ti >= t1:
                dt = t1 - prev_t
                if dt > 0: R = R @ so3_exp(w * dt)
                break
            dt = ti - prev_t
            if dt > 0:
                R = R @ so3_exp(w * dt); prev_t = ti
        return R

    # ---------- LiDAR ----------
    def cb_lidar(self, msg: PointCloud2):
        t = ros_time_to_float(msg.header.stamp)
        pts = pc2_to_xyz(msg)
        pcd = make_o3d_cloud(pts, voxel_size=self.voxel,
                             min_range=self.min_range, max_range=self.max_range,
                             z_min=self.z_min, z_max=self.z_max)
        if pcd is None or len(pcd.points) < 50:
            return
        if self.use_ground:
            pcd = ground_remove_ransac(pcd, self.ground_dist, 80)

        # 初帧：建立第一关键帧（转到 odom 再加入）
        if self.last_pc_time is None:
            pcd0 = o3d.geometry.PointCloud(pcd)    # copy
            pcd0.transform(self.T_odom_lidar)      # 当前为 I
            estimate_normals_inplace(pcd0, self.voxel)
            self.add_keyframe(self.T_odom_lidar.copy(), pcd0)
            self.last_pc_time = t
            self.publish_all(t)
            return

        # IMU + 平移初值 -> 帧间增量
        dt = t - self.last_pc_time
        R_imu = self.integrate_gyro(self.last_pc_time, t)
        T_delta_guess = np.eye(4)
        T_delta_guess[:3, :3] = R_imu
        # 使用时间比例进行运动预测，适应帧跳跃（SLAM处理速度<LiDAR发布速度时）
        # 例：上一帧间隔0.05s移动0.14m，当前间隔0.175s，则预测移动0.14*0.175/0.05=0.49m
        time_ratio = dt / self.last_delta_dt if self.last_delta_dt > 0 else 1.0
        T_delta_guess[:3, 3] = self.trans_init_scale * time_ratio * self.last_delta_T[:3, 3]

        # 映射到全局初值：上一时刻全局位姿 @ 帧间增量
        T_init_global = se3_mul(self.T_odom_lidar, T_delta_guess)

        # 准备源/目标
        source = pcd.voxel_down_sample(voxel_size=self.voxel)  # 当前帧（lidar坐标）
        estimate_normals_inplace(source, self.voxel)
        if self.submap is None or len(self.submap.points) < 100:
            self.rebuild_submap()
        target = self.submap  # odom坐标

        # GICP（source->target），初值=全局位姿猜测
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iter)
        try:
            est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
            reg = o3d.pipelines.registration.registration_generalized_icp(
                source, target, self.max_corr, T_init_global, est, criteria
            )
            T_cur_global = reg.transformation   # 直接是"当前绝对位姿(odom<-lidar)"
            fitness, rmse = reg.fitness, reg.inlier_rmse
        except Exception as e:
            self.get_logger().warn(f"GICP failed: {e}")
            T_cur_global, fitness, rmse = T_init_global, 0.0, 1e9

        # 质量门控
        if (fitness < self.icp_fitness_min) or (rmse > self.icp_rmse_max):
            self._gicp_reject_count += 1
            if self._gicp_reject_count % 20 == 1:
                total = self._gicp_accept_count + self._gicp_reject_count
                rate = self._gicp_reject_count / total * 100 if total > 0 else 0
                self.get_logger().warn(
                    f"GICP rejected (fitness={fitness:.2f}, rmse={rmse:.2f}), "
                    f"using IMU+prediction fallback. "
                    f"Reject rate: {rate:.0f}% ({self._gicp_reject_count}/{total})"
                )
            # 使用运动预测作为回退（而非零运动），避免丢失帧间位移
            T_cur_global = T_init_global
        else:
            self._gicp_accept_count += 1

        # 更新绝对姿态 & 帧间增量
        T_prev = self.T_odom_lidar.copy()
        self.T_odom_lidar = T_cur_global
        self.last_delta_T = se3_mul(se3_inv(T_prev), self.T_odom_lidar)
        self.last_delta_dt = t - self.last_pc_time  # 保存帧间时间间隔
        self.last_pc_time = t

        # 关键帧判断
        if self.is_keyframe(self.last_key_T, self.T_odom_lidar):
            cur_in_odom = o3d.geometry.PointCloud(source)
            cur_in_odom.transform(self.T_odom_lidar)  # 放到 odom
            self.add_keyframe(self.T_odom_lidar.copy(), cur_in_odom)
            self.truncate_submap()

        self.publish_all(t)

    # ---------- Keyframe/Submap ----------
    def is_keyframe(self, T_ref: np.ndarray, T_now: np.ndarray) -> bool:
        dT = se3_mul(se3_inv(T_ref), T_now)
        trans = np.linalg.norm(dT[:3, 3])
        R = dT[:3, :3]
        yaw = math.degrees(yaw_from_R(R))
        return (trans > self.kf_trans_th) or (abs(yaw) > self.kf_rot_deg_th)

    def add_keyframe(self, T_odom_lidar: np.ndarray, pcd_in_odom: o3d.geometry.PointCloud):
        pcd_ds = pcd_in_odom.voxel_down_sample(self.voxel)
        estimate_normals_inplace(pcd_ds, self.voxel)
        self.keyframes.append({'T': T_odom_lidar.copy(), 'pcd': pcd_ds})
        self.last_key_T = T_odom_lidar.copy()
        self.rebuild_submap()

        # ---- 小窗口闭环尝试 ----
        if self.loop is not None:
            self.loop.maybe_close(len(self.keyframes)-1, self.keyframes, logger=self.get_logger())

    def truncate_submap(self):
        if len(self.keyframes) > self.submap_size:
            self.keyframes = self.keyframes[-self.submap_size:]
            self.rebuild_submap()

    def rebuild_submap(self):
        if len(self.keyframes) == 0:
            self.submap = None; return
        pcs = [k['pcd'] for k in self.keyframes]
        acc = pcs[0]
        for i in range(1, len(pcs)):
            acc = acc + pcs[i]
        self.submap = acc.voxel_down_sample(self.voxel)
        estimate_normals_inplace(self.submap, self.voxel)

    # ---------- 发布 ----------
    def publish_all(self, t: float):
        # 应用闭环纠偏
        T_pub_lidar = self.loop.apply(self.T_odom_lidar) if self.loop is not None else self.T_odom_lidar

        # odom->base_link = (odom->lidar) * (lidar->base_link)
        T_lidar_base = np.eye(4); T_lidar_base[:3, 3] = np.array([0.0, 0.0, -self.lidar_z])
        T_odom_base = se3_mul(T_pub_lidar, T_lidar_base)

        odom = Odometry()
        odom.header.stamp = self._float_to_time(t)
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        R = T_odom_base[:3, :3]; tvec = T_odom_base[:3, 3]
        qx, qy, qz, qw = quat_xyzw_from_R(R)
        odom.pose.pose.position.x = float(tvec[0])
        odom.pose.pose.position.y = float(tvec[1])
        odom.pose.pose.position.z = float(tvec[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        cov_xy, cov_z, cov_rot = 0.02, 0.05, 0.01
        odom.pose.covariance = array('d', [
            cov_xy, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, cov_xy, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, cov_z, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, cov_rot, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, cov_rot, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, cov_rot
        ])
        self.odom_pub.publish(odom)

        ts = TransformStamped()
        ts.header = odom.header
        ts.child_frame_id = self.base_frame
        ts.transform.translation.x = odom.pose.pose.position.x
        ts.transform.translation.y = odom.pose.pose.position.y
        ts.transform.translation.z = odom.pose.pose.position.z
        ts.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(ts)

        ps = PoseStamped(); ps.header = odom.header; ps.pose = odom.pose.pose
        self.path_msg.header.stamp = odom.header.stamp
        self.path_msg.poses.append(ps)
        if len(self.path_msg.poses) > 8000:
            self.path_msg.poses = self.path_msg.poses[-8000:]
        self.path_pub.publish(self.path_msg)

        if self.publish_submap and (self.submap is not None) and len(self.submap.points) > 0:
            xyz = np.asarray(self.submap.points, dtype=np.float32)
            self.map_pub.publish(xyz_to_pc2(xyz, self.odom_frame, odom.header.stamp))

    def _float_to_time(self, t: float) -> TimeMsg:
        sec = int(t); nsec = int((t - sec) * 1e9)
        return TimeMsg(sec=sec, nanosec=nsec)

# ---------- 入口 ----------
def main():
    rclpy.init()
    node = LidarImuSubmapGICP()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
