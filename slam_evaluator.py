"""
SLAM评估节点
实时订阅Ground Truth和SLAM估计，计算ATE/RPE并生成评估报告
"""

import os
import json
import numpy as np
from datetime import datetime
from collections import deque
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time as TimeMsg
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from evaluation.metrics import compute_ate, compute_rpe, associate_timestamps
from evaluation.alignment import align_trajectories
from evaluation.visualizer import (
    plot_trajectory_comparison,
    plot_error_distribution,
    plot_rpe_analysis
)


def ros_time_to_float(t: TimeMsg) -> float:
    """ROS时间戳转浮点数"""
    return float(t.sec) + float(t.nanosec) * 1e-9


class SLAMEvaluator(Node):
    def __init__(self):
        super().__init__("slam_evaluator")

        # 参数声明
        self.declare_parameter("gt_topic", "/ground_truth/odom")
        self.declare_parameter("est_topic", "/odom")
        self.declare_parameter("align_method", "sim3")  # translation, se3, sim3
        self.declare_parameter("max_time_diff", 0.05)  # 时间戳匹配容差（秒）
        self.declare_parameter("report_hz", 1.0)  # 实时报告频率
        self.declare_parameter("output_dir", "output/slam_evaluation")
        self.declare_parameter("rpe_delta", 10)  # RPE帧间隔
        self.declare_parameter("min_poses", 10)  # 最小位姿数量才开始评估
        self.declare_parameter("use_first_frame_align", True)  # 使用首帧对齐
        self.declare_parameter("skip_initial_frames", 5)  # 跳过前N帧（等待SLAM稳定）

        # 获取参数
        self.gt_topic = self.get_parameter("gt_topic").value
        self.est_topic = self.get_parameter("est_topic").value
        self.align_method = self.get_parameter("align_method").value
        self.max_time_diff = self.get_parameter("max_time_diff").value
        self.report_hz = self.get_parameter("report_hz").value
        self.output_dir = os.path.expanduser(self.get_parameter("output_dir").value)
        self.rpe_delta = self.get_parameter("rpe_delta").value
        self.min_poses = self.get_parameter("min_poses").value
        self.use_first_frame_align = self.get_parameter("use_first_frame_align").value
        self.skip_initial_frames = self.get_parameter("skip_initial_frames").value

        # 创建带时间戳的输出子目录
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.output_dir, f"run_{stamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # 数据缓存
        self.gt_buffer = deque(maxlen=50000)  # (timestamp, x, y, z, qx, qy, qz, qw)
        self.est_buffer = deque(maxlen=50000)

        # 订阅器
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.gt_sub = self.create_subscription(
            Odometry, self.gt_topic, self.gt_callback, qos
        )
        self.est_sub = self.create_subscription(
            Odometry, self.est_topic, self.est_callback, qos
        )

        # 发布器（实时评估结果）
        self.eval_pub = self.create_publisher(String, "/slam/evaluation", 10)

        # 定时器（定期报告）
        if self.report_hz > 0:
            self.timer = self.create_timer(1.0 / self.report_hz, self.report_callback)

        # 统计
        self.last_ate_stats = None
        self.last_rpe_stats = None
        self.first_frame_offset = None  # 首帧偏移量

        self.get_logger().info("="*70)
        self.get_logger().info("[评估器] SLAM评估节点已启动")
        self.get_logger().info(f"  Ground Truth: {self.gt_topic}")
        self.get_logger().info(f"  SLAM估计:     {self.est_topic}")
        self.get_logger().info(f"  对齐方法:     {self.align_method}")
        self.get_logger().info(f"  首帧对齐:     {'启用' if self.use_first_frame_align else '禁用'}")
        self.get_logger().info(f"  跳过初始帧:   {self.skip_initial_frames}")
        self.get_logger().info(f"  输出目录:     {self.output_dir}")
        self.get_logger().info("="*70)

    def gt_callback(self, msg: Odometry):
        """Ground Truth回调"""
        t = ros_time_to_float(msg.header.stamp)
        pose = msg.pose.pose
        self.gt_buffer.append((
            t,
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w
        ))

    def est_callback(self, msg: Odometry):
        """SLAM估计回调"""
        t = ros_time_to_float(msg.header.stamp)
        pose = msg.pose.pose
        self.est_buffer.append((
            t,
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w
        ))

    def report_callback(self):
        """定期评估报告"""
        if len(self.gt_buffer) < self.min_poses or len(self.est_buffer) < self.min_poses:
            return

        try:
            # 时间戳关联
            gt_stamps = [x[0] for x in self.gt_buffer]
            est_stamps = [x[0] for x in self.est_buffer]
            matches = associate_timestamps(gt_stamps, est_stamps, self.max_time_diff)

            if len(matches) < self.min_poses:
                self.get_logger().warn(
                    f"[评估器] 匹配位姿不足: {len(matches)} < {self.min_poses}"
                )
                return

            # 跳过初始帧（等待SLAM稳定）
            if self.skip_initial_frames > 0 and len(matches) > self.skip_initial_frames:
                matches = matches[self.skip_initial_frames:]

            # 提取匹配的位姿
            gt_list = list(self.gt_buffer)
            est_list = list(self.est_buffer)
            gt_poses_7d = np.array([gt_list[i][1:] for i, _ in matches])
            est_poses_7d = np.array([est_list[j][1:] for _, j in matches])

            gt_poses_3d = gt_poses_7d[:, :3]
            est_poses_3d = est_poses_7d[:, :3]

            # 首次运行时记录初始偏移（仅供日志参考，不应用）
            if self.first_frame_offset is None:
                offset = gt_poses_3d[0] - est_poses_3d[0]
                self.first_frame_offset = offset
                self.get_logger().info(
                    f"[信息] 初始偏移: "
                    f"[{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}] = "
                    f"{np.linalg.norm(offset):.2f}m（由Sim3对齐自动处理）"
                )

            # 轨迹对齐（Sim3 自动处理平移+旋转+缩放差异）
            aligned_est_3d, align_params = align_trajectories(
                gt_poses_3d, est_poses_3d, method=self.align_method
            )

            # 计算ATE
            ate_stats = compute_ate(gt_poses_3d, aligned_est_3d)
            self.last_ate_stats = ate_stats

            # 计算RPE（需要完整7D位姿）
            if len(matches) >= self.rpe_delta + 1:
                aligned_est_7d = est_poses_7d.copy()
                aligned_est_7d[:, :3] = aligned_est_3d

                rpe_stats = compute_rpe(gt_poses_7d, aligned_est_7d, delta=self.rpe_delta)
                self.last_rpe_stats = rpe_stats
            else:
                rpe_stats = None

            # 发布实时结果
            eval_msg = self._format_eval_message(ate_stats, rpe_stats, len(matches))
            self.eval_pub.publish(String(data=eval_msg))

            # 控制台输出（包含对齐参数供参考）
            scale_str = f" scale={align_params.get('scale', 1.0):.3f}" if 'scale' in align_params else ""
            self.get_logger().info(
                f"[评估] 匹配={len(matches)} | "
                f"ATE RMSE={ate_stats['rmse']:.3f}m | "
                f"Mean={ate_stats['mean']:.3f}m | "
                f"Median={ate_stats['median']:.3f}m{scale_str}"
            )

        except Exception as e:
            self.get_logger().error(f"[评估器] 评估失败: {e}")

    def _format_eval_message(self, ate_stats: dict, rpe_stats: Optional[dict], n_poses: int) -> str:
        """格式化评估消息"""
        msg = f"[SLAM评估] 位姿数={n_poses}\n"
        msg += f"ATE: RMSE={ate_stats['rmse']:.4f}m, Mean={ate_stats['mean']:.4f}m, "
        msg += f"Median={ate_stats['median']:.4f}m, Std={ate_stats['std']:.4f}m\n"

        if rpe_stats:
            msg += f"RPE: Trans={rpe_stats['trans_rmse']:.4f}m, Rot={rpe_stats['rot_rmse']:.4f}°"

        return msg

    def save_final_report(self):
        """保存最终评估报告"""
        if len(self.gt_buffer) < self.min_poses or len(self.est_buffer) < self.min_poses:
            self.get_logger().warn("[评估器] 数据不足，跳过报告生成")
            return

        try:
            self.get_logger().info("="*70)
            self.get_logger().info("[评估器] 生成最终评估报告...")

            # 时间戳关联
            gt_stamps = [x[0] for x in self.gt_buffer]
            est_stamps = [x[0] for x in self.est_buffer]

            # 诊断：时间范围
            self.get_logger().info(f"  GT缓冲:  {len(gt_stamps)} 帧, "
                f"时间 {gt_stamps[0]:.2f}~{gt_stamps[-1]:.2f}s "
                f"(时长 {gt_stamps[-1] - gt_stamps[0]:.1f}s)")
            self.get_logger().info(f"  EST缓冲: {len(est_stamps)} 帧, "
                f"时间 {est_stamps[0]:.2f}~{est_stamps[-1]:.2f}s "
                f"(时长 {est_stamps[-1] - est_stamps[0]:.1f}s)")

            matches = associate_timestamps(gt_stamps, est_stamps, self.max_time_diff)

            if len(matches) < self.min_poses:
                self.get_logger().error(f"[评估器] 匹配位姿不足: {len(matches)}")
                return

            # 提取位姿
            gt_list = list(self.gt_buffer)
            est_list = list(self.est_buffer)
            gt_poses_7d = np.array([gt_list[i][1:] for i, _ in matches])
            est_poses_7d = np.array([est_list[j][1:] for _, j in matches])

            gt_poses_3d = gt_poses_7d[:, :3]
            est_poses_3d = est_poses_7d[:, :3]

            # 诊断：轨迹路径长度
            gt_path_len = float(np.sum(np.linalg.norm(np.diff(gt_poses_3d, axis=0), axis=1)))
            est_path_len = float(np.sum(np.linalg.norm(np.diff(est_poses_3d, axis=0), axis=1)))
            self.get_logger().info(f"  匹配位姿: {len(matches)}")
            self.get_logger().info(f"  GT路径长度:   {gt_path_len:.1f}m")
            self.get_logger().info(f"  SLAM路径长度: {est_path_len:.1f}m")
            if est_path_len > 0:
                self.get_logger().info(f"  路径长度比:   {gt_path_len / est_path_len:.3f}")
            self.get_logger().info(f"  GT起点:  [{gt_poses_3d[0, 0]:.2f}, {gt_poses_3d[0, 1]:.2f}, {gt_poses_3d[0, 2]:.2f}]")
            self.get_logger().info(f"  EST起点: [{est_poses_3d[0, 0]:.2f}, {est_poses_3d[0, 1]:.2f}, {est_poses_3d[0, 2]:.2f}]")

            # 轨迹对齐
            aligned_est_3d, align_params = align_trajectories(
                gt_poses_3d, est_poses_3d, method=self.align_method
            )

            # 计算ATE
            ate_stats = compute_ate(gt_poses_3d, aligned_est_3d)
            errors = np.linalg.norm(gt_poses_3d - aligned_est_3d, axis=1)

            # 计算RPE
            aligned_est_7d = est_poses_7d.copy()
            aligned_est_7d[:, :3] = aligned_est_3d
            rpe_stats = compute_rpe(gt_poses_7d, aligned_est_7d, delta=self.rpe_delta)

            # 保存JSON报告
            report = {
                'metadata': {
                    'gt_topic': self.gt_topic,
                    'est_topic': self.est_topic,
                    'align_method': self.align_method,
                    'n_poses': len(matches),
                    'rpe_delta': self.rpe_delta
                },
                'alignment': align_params,
                'ate': ate_stats,
                'rpe': rpe_stats
            }

            json_path = os.path.join(self.output_dir, "evaluation_report.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.get_logger().info(f"  JSON报告: {json_path}")

            # 生成可视化图表
            plot_trajectory_comparison(
                gt_poses_3d, est_poses_3d, aligned_est_3d,
                output_path=os.path.join(self.output_dir, "trajectory_comparison.png"),
                title="SLAM Trajectory Evaluation"
            )

            plot_error_distribution(
                errors, ate_stats,
                output_path=os.path.join(self.output_dir, "error_distribution.png"),
                title="ATE Error Analysis"
            )

            # RPE可视化（需要提取误差数组）
            trans_errors, rot_errors = self._compute_rpe_arrays(gt_poses_7d, aligned_est_7d)
            if len(trans_errors) > 0:
                plot_rpe_analysis(
                    trans_errors, rot_errors,
                    output_path=os.path.join(self.output_dir, "rpe_analysis.png"),
                    title=f"RPE Analysis (delta={self.rpe_delta} frames)"
                )

            # 控制台摘要
            self.get_logger().info("="*70)
            self.get_logger().info("[评估摘要]")
            self.get_logger().info(f"  匹配位姿数: {len(matches)}")
            self.get_logger().info(f"  ATE RMSE:   {ate_stats['rmse']:.4f} m")
            self.get_logger().info(f"  ATE Mean:   {ate_stats['mean']:.4f} m")
            self.get_logger().info(f"  ATE Median: {ate_stats['median']:.4f} m")
            self.get_logger().info(f"  RPE Trans:  {rpe_stats['trans_rmse']:.4f} m")
            self.get_logger().info(f"  RPE Rot:    {rpe_stats['rot_rmse']:.4f} °")
            self.get_logger().info("="*70)

        except Exception as e:
            self.get_logger().error(f"[评估器] 报告生成失败: {e}")
            import traceback
            traceback.print_exc()

    def _compute_rpe_arrays(self, gt_poses: np.ndarray, est_poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算RPE误差数组（用于可视化）"""
        from evaluation.metrics import pose_to_matrix, rotation_error

        n = gt_poses.shape[0]
        delta = self.rpe_delta

        trans_errors = []
        rot_errors = []

        for i in range(n - delta):
            gt_T_i = pose_to_matrix(gt_poses[i])
            gt_T_j = pose_to_matrix(gt_poses[i + delta])
            gt_rel = np.linalg.inv(gt_T_i) @ gt_T_j

            est_T_i = pose_to_matrix(est_poses[i])
            est_T_j = pose_to_matrix(est_poses[i + delta])
            est_rel = np.linalg.inv(est_T_i) @ est_T_j

            error_T = np.linalg.inv(gt_rel) @ est_rel

            trans_errors.append(np.linalg.norm(error_T[:3, 3]))
            rot_errors.append(rotation_error(error_T[:3, :3]))

        return np.array(trans_errors), np.array(rot_errors)


def main():
    rclpy.init()
    node = SLAMEvaluator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 保存最终报告
        node.save_final_report()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
