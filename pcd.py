import os, signal, numpy as np
from datetime import datetime
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry

# 关掉 Open3D-ML
import sys, types as _types
os.environ["OPEN3D_DISABLE_ML_MODULE"] = "1"
sys.modules['open3d.ml'] = _types.ModuleType('open3d.ml')
import open3d as o3d


def ros_time_to_float(t: TimeMsg) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9


def pc2_to_xyz(pc2: PointCloud2) -> np.ndarray:
    if len(pc2.data) == 0 or pc2.point_step < 12:
        return np.empty((0, 3), dtype=np.float32)
    cols = pc2.point_step // 4  # 每点 float32 个数
    arr = np.frombuffer(pc2.data, dtype=np.float32)
    if arr.size % cols != 0:
        arr = arr[: (arr.size // cols) * cols]
    arr = arr.reshape((-1, cols))
    return arr[:, 0:3].astype(np.float32, copy=False)


def voxel_downsample_np(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """纯 numpy 体素下采样（取每个格子的第一个点），速度很快。"""
    if xyz.size == 0 or voxel <= 1e-9:
        return xyz
    ijk = np.floor(xyz / voxel).astype(np.int32, copy=False)
    # 用结构化数组做去重
    keys = ijk.view([('i', np.int32), ('j', np.int32), ('k', np.int32)]).reshape(-1)
    _, idx = np.unique(keys, return_index=True)
    return xyz[idx]


class AccumulateAndSave(Node):
    def __init__(self):
        super().__init__("submap_accumulate_saver")

        # 参数
        self.declare_parameter("topic", "/map")
        self.declare_parameter("map_outfile", "output/carla_global.ply")
        self.declare_parameter("voxel_size", 0.5)          # 保存前的全局体素
        self.declare_parameter("chunk_voxel", 0.5)         # 每个子图块先体素（减内存）
        self.declare_parameter("max_points", 3_000_000)    # 最终最多点数（超了会随机抽样）
        self.declare_parameter("save_every_sec", 0.0)

        self.declare_parameter("traj_topic", "/odom")
        self.declare_parameter("traj_outfile", "output/carla_odom_tum.txt")
        self.declare_parameter("traj_skip", 0)

        self.declare_parameter("gt_topic", "/ground_truth/odom")
        self.declare_parameter("gt_outfile", "output/carla_ground_truth_trajectory.txt")
        self.declare_parameter("gt_skip", 0)

        gp = self.get_parameter
        self.topic = gp("topic").value
        self.map_outfile = os.path.expanduser(gp("map_outfile").value)
        self.voxel_size = float(gp("voxel_size").value)
        self.chunk_voxel = float(gp("chunk_voxel").value)
        self.max_points = int(gp("max_points").value)
        self.save_every_sec = float(gp("save_every_sec").value)

        self.traj_topic = gp("traj_topic").value
        self.traj_outfile = os.path.expanduser(gp("traj_outfile").value)
        self.traj_skip = int(gp("traj_skip").value)

        self.gt_topic = gp("gt_topic").value
        self.gt_outfile = os.path.expanduser(gp("gt_outfile").value)
        self.gt_skip = int(gp("gt_skip").value)

        # 累积容器（增量体素后再存块）
        self._chunks = []
        self._total_pts = 0
        self._traj = []
        self._traj_cnt = 0
        self._gt_traj = []
        self._gt_traj_cnt = 0
        self._odom_received = False  # GT同步：收到第一条/odom后才开始记录GT

        qos = QoSProfile(depth=5,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.sub_map = self.create_subscription(PointCloud2, self.topic, self.cb_map, qos)
        self.sub_odom = self.create_subscription(Odometry, self.traj_topic, self.cb_odom, qos)
        self.sub_gt = self.create_subscription(Odometry, self.gt_topic, self.cb_gt, qos)

        self.timer = None
        if self.save_every_sec > 0:
            self.timer = self.create_timer(self.save_every_sec, self.save_now)

        self.get_logger().info(
            f"Accumulating {self.topic} → {self.map_outfile} (PLY), "
            f"chunk_voxel={self.chunk_voxel}, voxel_size={self.voxel_size}, "
            f"max_points={self.max_points}, "
            + (f"autosave={self.save_every_sec}s, " if self.timer else "autosave=off, ")
            + f"trajectory {self.traj_topic} → {self.traj_outfile}, "
            + f"GT {self.gt_topic} → {self.gt_outfile}"
        )

    # ----------------- 回调 -----------------
    def cb_map(self, msg: PointCloud2):
        xyz = pc2_to_xyz(msg)
        if xyz.size == 0:
            self.get_logger().warn(f"[map] received empty PointCloud2 (data_len={len(msg.data)}, step={msg.point_step})")
            return
        # 每块先体素，立刻减小规模
        if self.chunk_voxel > 1e-6:
            xyz = voxel_downsample_np(xyz, self.chunk_voxel)
        if xyz.size == 0:
            return
        self._chunks.append(xyz)
        self._total_pts += xyz.shape[0]
        if len(self._chunks) <= 3 or len(self._chunks) % 50 == 0:
            self.get_logger().info(f"[map] accumulated {self._total_pts} pts, chunks={len(self._chunks)}")

    def cb_odom(self, odom: Odometry):
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info("[sync] First /odom received, GT recording enabled")
        if self.traj_skip > 0:
            self._traj_cnt += 1
            if self._traj_cnt % (self.traj_skip + 1) != 0:
                return
        t = ros_time_to_float(odom.header.stamp)
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        self._traj.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def cb_gt(self, odom: Odometry):
        if not self._odom_received:
            return
        if self.gt_skip > 0:
            self._gt_traj_cnt += 1
            if self._gt_traj_cnt % (self.gt_skip + 1) != 0:
                return
        t = ros_time_to_float(odom.header.stamp)
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        self._gt_traj.append((t, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    # ----------------- 保存 -----------------
    def save_now(self):
        # 在各输出路径的父目录下创建时间戳子目录
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        def stamped_path(orig):
            d = os.path.dirname(orig)
            base = os.path.basename(orig)
            out_dir = os.path.join(d, f"run_{stamp}") if d else f"run_{stamp}"
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, base)

        map_out = stamped_path(self.map_outfile)
        traj_out = stamped_path(self.traj_outfile)
        gt_out = stamped_path(self.gt_outfile)
        # 保存点云
        if len(self._chunks) == 0:
            self.get_logger().warn("No accumulated map yet, skip saving.")
        else:
            # 1) 组装 + 全局体素（可能很大；为避免被误中断，临时屏蔽 Ctrl+C）
            old_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda s, f: self.get_logger().warn("Saving... please wait (Ctrl+C ignored during save)"))
            try:
                all_xyz = np.vstack(self._chunks).astype(np.float32, copy=False)
                if self.voxel_size > 1e-6:
                    all_xyz = voxel_downsample_np(all_xyz, self.voxel_size)
                # 2) 限制最大点数
                if self.max_points > 0 and all_xyz.shape[0] > self.max_points:
                    idx = np.random.choice(all_xyz.shape[0], self.max_points, replace=False)
                    all_xyz = all_xyz[idx]
                # 3) 构建并写盘（非压缩二进制，速度最快）
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_xyz))
                ok = o3d.io.write_point_cloud(map_out, pcd,
                                              write_ascii=False, compressed=False)
                if ok:
                    self.get_logger().info(f"Saved global PLY: {map_out} ({len(pcd.points)} pts).")
                else:
                    self.get_logger().error("Open3D write_point_cloud returned False.")
            finally:
                signal.signal(signal.SIGINT, old_handler)

        # 保存SLAM轨迹（TUM）
        if len(self._traj) == 0:
            self.get_logger().warn("No trajectory yet, skip saving traj.")
        else:
            try:
                with open(traj_out, "w") as f:
                    for t, x, y, z, qx, qy, qz, qw in self._traj:
                        f.write(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
                self.get_logger().info(f"Saved trajectory (TUM): {traj_out} ({len(self._traj)} poses).")
            except Exception as e:
                self.get_logger().error(f"Save trajectory failed: {e}")

        # 保存GT轨迹（TUM）
        if len(self._gt_traj) == 0:
            self.get_logger().warn("No GT trajectory yet, skip saving GT traj.")
        else:
            try:
                with open(gt_out, "w") as f:
                    for t, x, y, z, qx, qy, qz, qw in self._gt_traj:
                        f.write(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
                self.get_logger().info(f"Saved GT trajectory (TUM): {gt_out} ({len(self._gt_traj)} poses).")
            except Exception as e:
                self.get_logger().error(f"Save GT trajectory failed: {e}")

def main():
    rclpy.init()
    node = AccumulateAndSave()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.save_now()
        except Exception as e:
            node.get_logger().error(f"Final save failed: {e}")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
