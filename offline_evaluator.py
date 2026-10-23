"""
离线SLAM评估工具
读取TUM格式轨迹文件，计算ATE/RPE并生成完整评估报告
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
from typing import Tuple, List


# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import compute_ate, compute_rpe, associate_timestamps
from evaluation.alignment import align_trajectories
from evaluation.visualizer import (
    plot_trajectory_comparison,
    plot_error_distribution,
    plot_rpe_analysis
)


def load_tum_trajectory(filepath: str) -> Tuple[List[float], np.ndarray]:
    """
    加载TUM格式轨迹文件

    格式: timestamp x y z qx qy qz qw

    返回:
        (时间戳列表, 位姿数组(N, 7))
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"轨迹文件不存在: {filepath}")

    timestamps = []
    poses = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            try:
                t = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                timestamps.append(t)
                poses.append([x, y, z, qx, qy, qz, qw])
            except ValueError:
                continue

    if len(poses) == 0:
        raise ValueError(f"未能从文件中解析出有效位姿: {filepath}")

    return timestamps, np.array(poses)


def _match_sequential(gt_poses, est_poses, skip_initial_frames):
    """按序号均匀匹配（忽略时间戳）"""
    n_gt = len(gt_poses)
    n_est = len(est_poses)
    n_min = min(n_gt, n_est)

    if n_gt <= n_est:
        # GT较短或等长：GT全取，EST等间隔采样
        gt_idx = np.arange(n_gt)
        est_idx = np.round(np.linspace(0, n_est - 1, n_gt)).astype(int)
    else:
        # EST较短：EST全取，GT等间隔采样
        est_idx = np.arange(n_est)
        gt_idx = np.round(np.linspace(0, n_gt - 1, n_est)).astype(int)

    matches = list(zip(gt_idx.tolist(), est_idx.tolist()))

    # 跳过初始帧
    if skip_initial_frames > 0 and len(matches) > skip_initial_frames + 10:
        matches = matches[skip_initial_frames:]

    return matches


def evaluate_trajectories(gt_file: str, est_file: str,
                         align_method: str = 'sim3',
                         max_time_diff: float = 0.05,
                         rpe_delta: int = 10,
                         output_dir: str = './evaluation_results',
                         skip_initial_frames: int = 5,
                         sequential: bool = False) -> dict:
    """
    评估两条轨迹

    参数:
        gt_file: Ground truth轨迹文件路径
        est_file: 估计轨迹文件路径
        align_method: 对齐方法 ('translation', 'se3', 'sim3')
        max_time_diff: 时间戳匹配容差（秒）
        rpe_delta: RPE帧间隔
        output_dir: 输出目录
        skip_initial_frames: 跳过前N帧
        sequential: 按序号匹配（忽略时间戳）

    返回:
        评估结果字典
    """
    print("="*70)
    print("[离线评估] 开始评估SLAM轨迹")
    print("="*70)

    # 加载轨迹
    print(f"[加载] Ground Truth: {gt_file}")
    gt_stamps, gt_poses = load_tum_trajectory(gt_file)
    print(f"       加载 {len(gt_poses)} 个位姿")
    print(f"       时间范围: {gt_stamps[0]:.2f} ~ {gt_stamps[-1]:.2f}s "
          f"(时长 {gt_stamps[-1] - gt_stamps[0]:.1f}s)")

    print(f"[加载] SLAM估计:     {est_file}")
    est_stamps, est_poses = load_tum_trajectory(est_file)
    print(f"       加载 {len(est_poses)} 个位姿")
    print(f"       时间范围: {est_stamps[0]:.2f} ~ {est_stamps[-1]:.2f}s "
          f"(时长 {est_stamps[-1] - est_stamps[0]:.1f}s)")

    # 时间戳关联或按序号匹配
    match_method_label = ""
    if sequential:
        print(f"[关联] 按序号匹配模式（忽略时间戳）...")
        matches = _match_sequential(gt_poses, est_poses, skip_initial_frames)
        print(f"       匹配 {len(matches)} 对位姿")
        match_method_label = "sequential"
    else:
        print(f"[关联] 时间戳匹配（容差={max_time_diff}s）...")
        matches = associate_timestamps(gt_stamps, est_stamps, max_time_diff)
        print(f"       匹配成功 {len(matches)} 对位姿")
        match_method_label = "timestamp"

        if len(matches) < 10:
            # 诊断：打印时间戳范围对比
            print(f"\n[诊断] 匹配失败！时间戳范围对比:")
            print(f"       GT  时间戳: [{gt_stamps[0]:.6f} ~ {gt_stamps[-1]:.6f}]")
            print(f"       EST 时间戳: [{est_stamps[0]:.6f} ~ {est_stamps[-1]:.6f}]")
            overlap_start = max(gt_stamps[0], est_stamps[0])
            overlap_end = min(gt_stamps[-1], est_stamps[-1])
            if overlap_start > overlap_end:
                print(f"       原因: 时间戳范围完全不重叠！")
                print(f"       偏移量: {abs(gt_stamps[0] - est_stamps[0]):.2f}s")
                print(f"       提示: GT和EST文件可能来自不同的仿真会话，"
                      f"尝试 --sequential 按序号匹配")
            else:
                min_diff = float('inf')
                for g in gt_stamps[:20]:
                    for e in est_stamps[:20]:
                        d = abs(g - e)
                        if d < min_diff:
                            min_diff = d
                print(f"       重叠区间: [{overlap_start:.2f} ~ {overlap_end:.2f}]")
                print(f"       前20帧最小时间差: {min_diff:.6f}s (容差: {max_time_diff}s)")
                if min_diff > max_time_diff:
                    print(f"       提示: 最小时间差超过容差，尝试增大 --max-time-diff")
            print()
            raise ValueError(f"匹配位姿数量不足: {len(matches)} < 10")

        # 跳过初始帧（等待SLAM稳定）
        if skip_initial_frames > 0 and len(matches) > skip_initial_frames + 10:
            print(f"[跳过] 跳过前 {skip_initial_frames} 帧")
            matches = matches[skip_initial_frames:]

    if len(matches) < 10:
        raise ValueError(f"匹配位姿数量不足: {len(matches)} < 10")

    # 提取匹配的位姿
    gt_matched = np.array([gt_poses[i] for i, _ in matches])
    est_matched = np.array([est_poses[j] for _, j in matches])

    gt_xyz = gt_matched[:, :3]
    est_xyz = est_matched[:, :3]

    # 诊断：计算轨迹路径长度
    gt_path_len = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))
    est_path_len = float(np.sum(np.linalg.norm(np.diff(est_xyz, axis=0), axis=1)))
    print(f"[诊断] GT轨迹路径长度:   {gt_path_len:.1f}m")
    print(f"[诊断] SLAM轨迹路径长度: {est_path_len:.1f}m")
    if est_path_len > 0:
        print(f"[诊断] 路径长度比:       {gt_path_len / est_path_len:.3f}")

    # 记录初始偏移（仅供参考，不应用——由Sim3对齐自动处理）
    first_frame_offset = gt_xyz[0] - est_xyz[0]
    offset_dist = np.linalg.norm(first_frame_offset)
    print(f"[信息] 初始偏移: "
          f"[{first_frame_offset[0]:.2f}, {first_frame_offset[1]:.2f}, {first_frame_offset[2]:.2f}] "
          f"= {offset_dist:.2f}m（由{align_method}对齐自动处理）")

    # 保存原始EST用于可视化（对齐前）
    est_xyz_original = est_xyz.copy()

    # 轨迹对齐（Sim3 自动处理平移+旋转+缩放差异）
    print(f"[对齐] 使用 {align_method} 方法对齐轨迹...")
    aligned_est_xyz, align_params = align_trajectories(gt_xyz, est_xyz, method=align_method)

    if align_method == 'sim3':
        print(f"       缩放因子: {align_params['scale']:.6f}")
    print(f"       平移向量: {np.array(align_params['translation'])}")

    # 计算ATE
    print("[计算] 绝对轨迹误差 (ATE)...")
    ate_stats = compute_ate(gt_xyz, aligned_est_xyz)
    errors = np.linalg.norm(gt_xyz - aligned_est_xyz, axis=1)

    print(f"       RMSE:   {ate_stats['rmse']:.4f} m")
    print(f"       均值:   {ate_stats['mean']:.4f} m")
    print(f"       中位数: {ate_stats['median']:.4f} m")
    print(f"       标准差: {ate_stats['std']:.4f} m")
    print(f"       最小值: {ate_stats['min']:.4f} m")
    print(f"       最大值: {ate_stats['max']:.4f} m")

    # 计算RPE
    print(f"[计算] 相对位姿误差 (RPE, delta={rpe_delta})...")
    aligned_est_full = est_matched.copy()
    aligned_est_full[:, :3] = aligned_est_xyz

    rpe_stats = compute_rpe(gt_matched, aligned_est_full, delta=rpe_delta)

    print(f"       平移 RMSE: {rpe_stats['trans_rmse']:.4f} m")
    print(f"       平移 均值: {rpe_stats['trans_mean']:.4f} m")
    print(f"       旋转 RMSE: {rpe_stats['rot_rmse']:.4f} deg")
    print(f"       旋转 均值: {rpe_stats['rot_mean']:.4f} deg")

    # 创建带时间戳的输出子目录
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"run_{stamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 生成可视化（原始对比使用未修改的EST轨迹）
    print(f"[可视化] 生成评估图表...")
    plot_trajectory_comparison(
        gt_xyz, est_xyz_original, aligned_est_xyz,
        output_path=os.path.join(output_dir, "trajectory_comparison.png"),
        title="SLAM Trajectory Evaluation"
    )

    plot_error_distribution(
        errors, ate_stats,
        output_path=os.path.join(output_dir, "error_distribution.png"),
        title="SLAM Trajectory - Error Distribution"
    )

    # RPE可视化（需要提取误差数组）
    trans_errors, rot_errors = extract_rpe_errors(gt_matched, aligned_est_full, rpe_delta)
    if len(trans_errors) > 0:
        plot_rpe_analysis(
            trans_errors, rot_errors,
            output_path=os.path.join(output_dir, "rpe_analysis.png"),
            title="SLAM Trajectory - Relative Pose Error"
        )

    # 保存评估报告
    report = {
        'files': {
            'ground_truth': gt_file,
            'estimated': est_file
        },
        'statistics': {
            'total_gt_poses': len(gt_poses),
            'total_est_poses': len(est_poses),
            'matched_poses': len(matches),
            'match_method': match_method_label,
            'match_rate': len(matches) / min(len(gt_poses), len(est_poses))
        },
        'alignment': {
            'method': align_method,
            'parameters': align_params
        },
        'ate': ate_stats,
        'rpe': rpe_stats
    }

    report_file = os.path.join(output_dir, "evaluation_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[保存] 评估报告: {report_file}")
    print("="*70)
    print("[完成] 评估完成！")
    print("="*70)

    return report


def extract_rpe_errors(gt_poses: np.ndarray, est_poses: np.ndarray,
                      delta: int) -> Tuple[np.ndarray, np.ndarray]:
    """提取RPE误差数组用于可视化"""
    from evaluation.metrics import pose_to_matrix, rotation_error

    n = gt_poses.shape[0]
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
    parser = argparse.ArgumentParser(
        description="离线SLAM轨迹评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本评估（时间戳匹配）
  python3 offline_evaluator.py gt.txt slam_odom.txt

  # 按序号匹配（GT和EST来自不同仿真会话时使用）
  python3 offline_evaluator.py gt.txt slam_odom.txt --sequential

  # 指定对齐方法和输出目录
  python3 offline_evaluator.py gt.txt slam_odom.txt --align sim3 --output ./results

  # 调整RPE参数
  python3 offline_evaluator.py gt.txt slam_odom.txt --rpe-delta 20
        """
    )

    parser.add_argument('gt_file', help='Ground truth轨迹文件 (TUM格式)')
    parser.add_argument('est_file', help='SLAM估计轨迹文件 (TUM格式)')
    parser.add_argument('--align', default='sim3',
                       choices=['translation', 'se3', 'sim3'],
                       help='轨迹对齐方法 (默认: sim3)')
    parser.add_argument('--max-time-diff', type=float, default=0.05,
                       help='时间戳匹配容差/秒 (默认: 0.05)')
    parser.add_argument('--rpe-delta', type=int, default=10,
                       help='RPE帧间隔 (默认: 10)')
    parser.add_argument('--output', default='./evaluation_results',
                       help='输出目录 (默认: ./evaluation_results)')
    parser.add_argument('--skip-frames', type=int, default=5,
                       help='跳过前N帧 (默认: 5)')
    parser.add_argument('--sequential', action='store_true',
                       help='按序号匹配（不依赖时间戳，适用于GT和EST来自不同会话）')

    args = parser.parse_args()

    try:
        evaluate_trajectories(
            gt_file=args.gt_file,
            est_file=args.est_file,
            align_method=args.align,
            max_time_diff=args.max_time_diff,
            rpe_delta=args.rpe_delta,
            output_dir=args.output,
            skip_initial_frames=args.skip_frames,
            sequential=args.sequential
        )
    except Exception as e:
        print(f"\n[错误] 评估失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
