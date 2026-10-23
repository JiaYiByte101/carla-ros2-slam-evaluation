"""
Lightweight LiDAR-IMU Submap GICP SLAM

从 all.py 提取的纯算法逻辑，继承 BaseSLAMAlgorithm 接口。
ROS2 节点层（all.py）负责订阅/发布，本模块只做 SLAM 计算。
"""
import math
from typing import Optional, Dict, Any, List

import numpy as np
import open3d as o3d
import transforms3d.quaternions as t3dq
import transforms3d.axangles as t3da
from nav_msgs.msg import Odometry

from .base_slam import BaseSLAMAlgorithm, SensorData, SensorType, SLAMType


# ============================================================ #
#  工具函数
# ============================================================ #

def so3_exp(omega: np.ndarray):
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


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
    return math.atan2(R[1, 0], R[0, 0])


def yaw_from_T(T):
    return yaw_from_R(T[:3, :3])


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
    if pcd is None or len(pcd.points) == 0:
        return
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=2.0 * voxel_size + 0.2, max_nn=30)
    )


def ground_remove_ransac(pcd: o3d.geometry.PointCloud,
                         dist_th: float = 0.15, n_iter: int = 100):
    if pcd is None or len(pcd.points) < 200:
        return pcd
    try:
        _, inliers = pcd.segment_plane(
            distance_threshold=dist_th, ransac_n=3, num_iterations=n_iter)
        return pcd.select_by_index(inliers, invert=True)
    except Exception:
        return pcd


# ============================================================ #
#  ICP 辅助
# ============================================================ #

def o3d_downsample(pcd, voxel):
    return pcd if voxel <= 0 else pcd.voxel_down_sample(voxel)


def icp_point_to_plane(src, tgt, init_T, max_corr, max_iter):
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr * 2.0, max_nn=30))
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iter)
    return o3d.pipelines.registration.registration_icp(
        src, tgt, max_corr, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )


# ============================================================ #
#  轻量闭环
# ============================================================ #

class LightLoopCloser:
    """近邻再访的小窗口闭环（超轻量）"""

    def __init__(self, p: Dict[str, Any]):
        self.enable = p['lc_enable']
        self.step = p['lc_check_every_n_keyframes']
        self.window = p['lc_recent_window']
        self.dist_th = p['lc_dist_thresh']
        self.yaw_th = math.radians(p['lc_yaw_thresh_deg'])
        self.voxel = p['lc_icp_voxel']
        self.max_it = p['lc_icp_max_iter']
        self.fit_min = p['lc_icp_fitness_min']
        self.rmse_max = p['lc_icp_rmse_max']
        self.alpha = p['lc_correction_alpha']
        self.T_corr = np.eye(4)

    def maybe_close(self, kf_idx, kfs, logger=None):
        if (not self.enable) or (kf_idx % self.step != 0) or (kf_idx < 6):
            return False

        cur = kfs[kf_idx]
        T_cur = cur['T']; pos_cur = T_cur[:3, 3]; yaw_cur = yaw_from_T(T_cur)

        start = max(0, kf_idx - self.window)
        candidates = []
        for j in range(start, kf_idx - 5):
            Tj = kfs[j]['T']
            d = Tj[:3, 3] - pos_cur
            dist = np.linalg.norm(d[:2])
            if dist > self.dist_th:
                continue
            dyaw = abs(yaw_from_T(Tj) - yaw_cur)
            dyaw = min(dyaw, abs(2 * np.pi - dyaw))
            if dyaw > self.yaw_th:
                continue
            candidates.append((j, dist, dyaw))

        if not candidates:
            return False

        j, _, _ = min(candidates, key=lambda x: x[1])

        pc_cur = o3d_downsample(cur['pcd'], self.voxel)
        pc_ref = o3d_downsample(kfs[j]['pcd'], self.voxel)
        T_init = se3_from_mat4(se3_inv(kfs[j]['T']) @ T_cur)

        max_corr = max(0.8, 1.2 * self.voxel)
        try:
            reg = icp_point_to_plane(pc_cur, pc_ref, T_init, max_corr, self.max_it)
            fitness, rmse = reg.fitness, reg.inlier_rmse
        except Exception as e:
            if logger:
                logger.warn(f"[LC] ICP failed: {e}")
            return False

        if fitness < self.fit_min or rmse > self.rmse_max:
            if logger:
                logger.info(f"[LC] reject (fit={fitness:.2f}, rmse={rmse:.2f})")
            return False

        T_expect = T_init
        T_icp = reg.transformation
        E_ref = se3_inv(T_icp) @ T_expect

        T_wj = kfs[j]['T']
        Delta_w = T_wj @ E_ref @ se3_inv(T_wj)

        R = Delta_w[:3, :3]; t = Delta_w[:3, 3]
        axis, angle = t3da.mat2axangle(R)
        R_a = t3da.axangle2mat(axis, angle * self.alpha)
        t_a = t * self.alpha
        Delta_a = np.eye(4); Delta_a[:3, :3] = R_a; Delta_a[:3, 3] = t_a

        self.T_corr = se3_from_mat4(self.T_corr @ Delta_a)
        if logger:
            logger.info(
                f"[LC] accepted -> update T_corr (fit={fitness:.2f}, rmse={rmse:.2f})")
        return True

    def apply(self, T_est):
        return se3_from_mat4(self.T_corr @ T_est)


# ============================================================ #
#  LightweightSLAM — BaseSLAMAlgorithm 实现
# ============================================================ #

class LightweightSLAM(BaseSLAMAlgorithm):
    """Lightweight LiDAR-IMU Submap GICP SLAM 算法。

    通过 ``config`` 字典接收全部参数，不依赖 ROS2 参数系统。
    IMU 数据通过 ``SensorData.imu_sequence`` 传入（由 ROS2 节点负责缓冲）。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "Lightweight_SLAM"
        self.slam_type = SLAMType.LIDAR_IMU
        self.required_sensors = [SensorType.LIDAR]

        # 读参数（与 KISSICP 风格一致）
        self.voxel = config.get("voxel_size", 0.5)
        self.max_corr = config.get("max_corr_dist", 1.2)
        self.min_range = config.get("min_range", 2.0)
        self.max_range = config.get("max_range", 60.0)
        self.z_min = config.get("z_min", -2.0)
        self.z_max = config.get("z_max", 3.0)
        self.icp_max_iter = config.get("icp_max_iter", 60)
        self.icp_fitness_min = config.get("icp_fitness_min", 0.25)
        self.icp_rmse_max = config.get("icp_rmse_max", 0.60)

        self.use_ground = config.get("use_ground_removal", True)
        self.ground_dist = config.get("ground_ransac_dist", 0.15)

        self.kf_trans_th = config.get("keyframe_trans_th", 1.0)
        self.kf_rot_deg_th = config.get("keyframe_rot_deg_th", 8.0)
        self.submap_size = config.get("submap_size", 12)

        self.trans_init_scale = config.get("trans_init_scale", 0.6)

        # 闭环参数
        lc_params = {
            'lc_enable': config.get('lc_enable', True),
            'lc_check_every_n_keyframes': config.get('lc_check_every_n_keyframes', 10),
            'lc_recent_window': config.get('lc_recent_window', 100),
            'lc_dist_thresh': config.get('lc_dist_thresh', 25.0),
            'lc_yaw_thresh_deg': config.get('lc_yaw_thresh_deg', 20.0),
            'lc_icp_voxel': config.get('lc_icp_voxel', 0.6),
            'lc_icp_max_iter': config.get('lc_icp_max_iter', 30),
            'lc_icp_fitness_min': config.get('lc_icp_fitness_min', 0.35),
            'lc_icp_rmse_max': config.get('lc_icp_rmse_max', 0.50),
            'lc_correction_alpha': config.get('lc_correction_alpha', 0.35),
        }
        self._lc_params = lc_params

        # 运行时状态（由 initialize() 设置）
        self.loop: Optional[LightLoopCloser] = None
        self.keyframes: List[Dict[str, Any]] = []
        self.submap: Optional[o3d.geometry.PointCloud] = None
        self.last_key_T = np.eye(4)
        self.T_odom_lidar = np.eye(4)
        self.last_delta_T = np.eye(4)
        self.last_delta_dt = 0.05
        self.last_pc_time: Optional[float] = None
        self._gicp_reject_count = 0
        self._gicp_accept_count = 0
        self.trajectory: List[List[float]] = []
        self._logger = None
        self._frame_count = 0

    # -------------------------------------------------------- #
    #  BaseSLAMAlgorithm 抽象方法实现
    # -------------------------------------------------------- #

    def initialize(self) -> bool:
        self.keyframes.clear()
        self.trajectory.clear()
        self.submap = None
        self.last_key_T = np.eye(4)
        self.T_odom_lidar = np.eye(4)
        self.current_pose = np.eye(4)
        self.last_delta_T = np.eye(4)
        self.last_delta_dt = 0.05
        self.last_pc_time = None
        self._gicp_reject_count = 0
        self._gicp_accept_count = 0
        self._frame_count = 0

        if self._lc_params['lc_enable']:
            self.loop = LightLoopCloser(self._lc_params)
        else:
            self.loop = None

        self.is_initialized = True
        return True

    def process_frame(self, sensor_data: SensorData) -> Odometry:
        if not self.is_initialized:
            self.initialize()

        raw_pcd = sensor_data.lidar
        if raw_pcd is None or len(raw_pcd.points) < 50:
            return self._make_odom(sensor_data.timestamp)

        timestamp = sensor_data.timestamp

        # 点云预处理（距离滤波 + 体素降采样）—— 与原 all.py cb_lidar 一致
        pts_xyz = np.asarray(raw_pcd.points)
        pcd = make_o3d_cloud(pts_xyz, voxel_size=self.voxel,
                             min_range=self.min_range, max_range=self.max_range,
                             z_min=self.z_min, z_max=self.z_max)
        if pcd is None or len(pcd.points) < 50:
            return self._make_odom(timestamp)

        # 地面剔除
        if self.use_ground:
            pcd = ground_remove_ransac(pcd, self.ground_dist, 80)

        # ---------- 初帧 ----------
        if self.last_pc_time is None:
            pcd0 = o3d.geometry.PointCloud(pcd)
            pcd0.transform(self.T_odom_lidar)
            estimate_normals_inplace(pcd0, self.voxel)
            self._add_keyframe(self.T_odom_lidar.copy(), pcd0)
            self.last_pc_time = timestamp
            self._record_trajectory(timestamp)
            return self._make_odom(timestamp)

        # ---------- IMU 陀螺仪积分 ----------
        dt = timestamp - self.last_pc_time
        R_imu = self._integrate_gyro(sensor_data.imu_sequence,
                                     self.last_pc_time, timestamp)

        # ---------- 帧间增量初值 ----------
        T_delta_guess = np.eye(4)
        T_delta_guess[:3, :3] = R_imu
        time_ratio = dt / self.last_delta_dt if self.last_delta_dt > 0 else 1.0
        T_delta_guess[:3, 3] = (self.trans_init_scale * time_ratio
                                * self.last_delta_T[:3, 3])

        T_init_global = se3_mul(self.T_odom_lidar, T_delta_guess)

        # ---------- 准备源/目标 ----------
        source = pcd.voxel_down_sample(voxel_size=self.voxel)
        estimate_normals_inplace(source, self.voxel)
        if self.submap is None or len(self.submap.points) < 100:
            self._rebuild_submap()
        target = self.submap

        if target is None or len(target.points) < 100:
            # 子图不可用，使用预测值
            T_cur_global = T_init_global
            fitness, rmse = 0.0, 1e9
        else:
            # ---------- GICP ----------
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.icp_max_iter)
            try:
                est = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
                reg = o3d.pipelines.registration.registration_generalized_icp(
                    source, target, self.max_corr, T_init_global, est, criteria)
                T_cur_global = reg.transformation.copy()
                fitness, rmse = reg.fitness, reg.inlier_rmse
            except Exception:
                T_cur_global, fitness, rmse = T_init_global, 0.0, 1e9

        # ---------- 质量门控 ----------
        if fitness < self.icp_fitness_min or rmse > self.icp_rmse_max:
            self._gicp_reject_count += 1
            if self._gicp_reject_count % 20 == 1 and self._logger:
                total = self._gicp_accept_count + self._gicp_reject_count
                rate = self._gicp_reject_count / total * 100 if total > 0 else 0
                self._logger.warn(
                    f"GICP rejected (fitness={fitness:.2f}, rmse={rmse:.2f}), "
                    f"using IMU+prediction fallback. "
                    f"Reject rate: {rate:.0f}% "
                    f"({self._gicp_reject_count}/{total})")
            T_cur_global = T_init_global
        else:
            self._gicp_accept_count += 1

        # ---------- 更新状态 ----------
        self._frame_count += 1
        if self._logger and self._frame_count <= 10:
            pos = T_cur_global[:3, 3]
            self._logger.info(
                f"[DIAG] frame={self._frame_count} "
                f"fitness={fitness:.3f} rmse={rmse:.3f} "
                f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"src={len(source.points)} tgt={len(target.points) if target else 0} "
                f"dt={dt:.4f}")
        T_prev = self.T_odom_lidar.copy()
        self.T_odom_lidar = T_cur_global
        self.current_pose = T_cur_global.copy()
        self.last_delta_T = se3_mul(se3_inv(T_prev), self.T_odom_lidar)
        self.last_delta_dt = dt
        self.last_pc_time = timestamp

        # ---------- 关键帧判断 ----------
        if self._is_keyframe(self.last_key_T, self.T_odom_lidar):
            cur_in_odom = o3d.geometry.PointCloud(source)
            cur_in_odom.transform(self.T_odom_lidar)
            self._add_keyframe(self.T_odom_lidar.copy(), cur_in_odom)
            self._truncate_submap()

        self._record_trajectory(timestamp)
        return self._make_odom(timestamp)

    def get_map(self) -> Optional[o3d.geometry.PointCloud]:
        return self.submap

    def reset(self):
        self.initialize()

    def get_trajectory(self) -> np.ndarray:
        if not self.trajectory:
            return np.empty((0, 8))
        return np.asarray(self.trajectory)

    # -------------------------------------------------------- #
    #  公开辅助方法（供 ROS2 节点使用）
    # -------------------------------------------------------- #

    def set_logger(self, logger):
        """允许 ROS2 节点注入 logger，用于 GICP/闭环日志输出。"""
        self._logger = logger

    def get_corrected_pose(self) -> np.ndarray:
        """返回闭环校正后的 odom<-lidar 位姿。"""
        if self.loop is not None:
            return self.loop.apply(self.T_odom_lidar)
        return self.T_odom_lidar

    # -------------------------------------------------------- #
    #  内部方法
    # -------------------------------------------------------- #

    @staticmethod
    def _integrate_gyro(imu_sequence: list, t0: float, t1: float) -> np.ndarray:
        """从 IMU 序列中积分陀螺仪，得到 t0->t1 的旋转。"""
        if t1 <= t0 or not imu_sequence:
            return np.eye(3)
        R = np.eye(3)
        prev_t = t0
        for (ti, w) in imu_sequence:
            if ti <= t0:
                continue
            if ti >= t1:
                d = t1 - prev_t
                if d > 0:
                    R = R @ so3_exp(w * d)
                break
            d = ti - prev_t
            if d > 0:
                R = R @ so3_exp(w * d)
                prev_t = ti
        return R

    def _is_keyframe(self, T_ref: np.ndarray, T_now: np.ndarray) -> bool:
        dT = se3_mul(se3_inv(T_ref), T_now)
        trans = np.linalg.norm(dT[:3, 3])
        yaw = math.degrees(yaw_from_R(dT[:3, :3]))
        return trans > self.kf_trans_th or abs(yaw) > self.kf_rot_deg_th

    def _add_keyframe(self, T_odom_lidar: np.ndarray,
                      pcd_in_odom: o3d.geometry.PointCloud):
        pcd_ds = pcd_in_odom.voxel_down_sample(self.voxel)
        estimate_normals_inplace(pcd_ds, self.voxel)
        self.keyframes.append({'T': T_odom_lidar.copy(), 'pcd': pcd_ds})
        self.last_key_T = T_odom_lidar.copy()
        self._rebuild_submap()

        if self.loop is not None:
            self.loop.maybe_close(
                len(self.keyframes) - 1, self.keyframes, logger=self._logger)

    def _truncate_submap(self):
        if len(self.keyframes) > self.submap_size:
            self.keyframes = self.keyframes[-self.submap_size:]
            self._rebuild_submap()

    def _rebuild_submap(self):
        if not self.keyframes:
            self.submap = None
            return
        pcs = [k['pcd'] for k in self.keyframes]
        acc = pcs[0]
        for i in range(1, len(pcs)):
            acc = acc + pcs[i]
        self.submap = acc.voxel_down_sample(self.voxel)
        estimate_normals_inplace(self.submap, self.voxel)

    def _record_trajectory(self, timestamp: float):
        R = self.T_odom_lidar[:3, :3]
        t = self.T_odom_lidar[:3, 3]
        quat = t3dq.mat2quat(R)  # [w, x, y, z]
        self.trajectory.append([
            timestamp,
            t[0], t[1], t[2],
            quat[1], quat[2], quat[3], quat[0],  # TUM: x y z qx qy qz qw
        ])

    def _make_odom(self, timestamp: float) -> Odometry:
        T_pub = self.get_corrected_pose()
        R = T_pub[:3, :3]
        tvec = T_pub[:3, 3]
        quat = t3dq.mat2quat(R)  # [w, x, y, z]

        odom = Odometry()
        odom.header.stamp.sec = int(timestamp)
        odom.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(tvec[0])
        odom.pose.pose.position.y = float(tvec[1])
        odom.pose.pose.position.z = float(tvec[2])
        odom.pose.pose.orientation.w = float(quat[0])
        odom.pose.pose.orientation.x = float(quat[1])
        odom.pose.pose.orientation.y = float(quat[2])
        odom.pose.pose.orientation.z = float(quat[3])

        odom.pose.covariance[0] = 0.02
        odom.pose.covariance[7] = 0.02
        odom.pose.covariance[14] = 0.05
        odom.pose.covariance[21] = 0.01
        odom.pose.covariance[28] = 0.01
        odom.pose.covariance[35] = 0.01
        return odom
