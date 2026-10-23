"""
评估结果可视化
生成轨迹对比图和误差分布图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os


# 使用英文标签避免中文字体渲染问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _compute_path_length(poses: np.ndarray) -> float:
    """计算轨迹路径长度"""
    if len(poses) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)))


def plot_trajectory_comparison(gt_poses: np.ndarray, est_poses: np.ndarray,
                               aligned_est_poses: Optional[np.ndarray] = None,
                               output_path: str = "trajectory_comparison.png",
                               title: str = "Trajectory Comparison"):
    """
    绘制轨迹对比图（俯视图）
    当提供aligned_est_poses时，显示对齐后的SLAM轨迹与GT对比

    参数:
        gt_poses: Ground truth位姿 (N, 3)
        est_poses: 原始估计位姿 (N, 3)
        aligned_est_poses: Sim3/SE3对齐后的估计位姿 (N, 3)，若提供则用于绘图
        output_path: 输出文件路径
        title: 图表标题
    """
    # 优先使用对齐后的轨迹进行绘图
    plot_est = aligned_est_poses if aligned_est_poses is not None else est_poses

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 计算路径长度（使用原始轨迹计算，反映真实SLAM输出）
    gt_path_len = _compute_path_length(gt_poses)
    est_path_len = _compute_path_length(est_poses)
    ratio = gt_path_len / est_path_len if est_path_len > 0 else float('inf')

    aligned_tag = " (aligned)" if aligned_est_poses is not None else ""

    # 绘制GT轨迹
    ax.plot(gt_poses[:, 0], gt_poses[:, 1], 'g-', linewidth=2,
            label=f'Ground Truth ({gt_path_len:.0f}m)', alpha=0.8)
    # 绘制SLAM轨迹（对齐后）
    ax.plot(plot_est[:, 0], plot_est[:, 1], 'b--', linewidth=1.5,
            label=f'SLAM Estimate ({est_path_len:.0f}m){aligned_tag}', alpha=0.7)
    # GT起点和终点
    ax.scatter(gt_poses[0, 0], gt_poses[0, 1], c='green', s=100, marker='o',
               label='GT Start', zorder=5)
    ax.scatter(gt_poses[-1, 0], gt_poses[-1, 1], c='red', s=100, marker='x',
               label='GT End', zorder=5)
    # SLAM起点和终点
    ax.scatter(plot_est[0, 0], plot_est[0, 1], c='blue', s=100, marker='s',
               label='SLAM Start', zorder=5)
    ax.scatter(plot_est[-1, 0], plot_est[-1, 1], c='purple', s=100, marker='D',
               label='SLAM End', zorder=5)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{len(gt_poses)} matched poses | '
                 f'Path ratio: {ratio:.2f}x',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[可视化] 轨迹对比图已保存: {output_path}")


def plot_error_distribution(errors: np.ndarray, ate_stats: Dict[str, float],
                           output_path: str = "error_distribution.png",
                           title: str = "Error Distribution"):
    """
    绘制误差分布图

    参数:
        errors: 误差数组 (N,)
        ate_stats: ATE统计字典
        output_path: 输出文件路径
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 误差时序图
    ax = axes[0, 0]
    ax.plot(errors, 'b-', linewidth=1, alpha=0.7)
    ax.axhline(ate_stats['mean'], color='r', linestyle='--', linewidth=2,
               label=f"Mean: {ate_stats['mean']:.3f}m")
    ax.axhline(ate_stats['median'], color='g', linestyle='--', linewidth=2,
               label=f"Median: {ate_stats['median']:.3f}m")
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Error (m)', fontsize=11)
    ax.set_title('Error over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 误差直方图
    ax = axes[0, 1]
    ax.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(ate_stats['mean'], color='r', linestyle='--', linewidth=2,
               label=f"Mean: {ate_stats['mean']:.3f}m")
    ax.axvline(ate_stats['median'], color='g', linestyle='--', linewidth=2,
               label=f"Median: {ate_stats['median']:.3f}m")
    ax.set_xlabel('Error (m)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Error Histogram', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 累积分布函数（CDF）
    ax = axes[1, 0]
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cdf, 'b-', linewidth=2)
    ax.axvline(ate_stats['median'], color='g', linestyle='--', linewidth=2,
               label=f"50%: {ate_stats['median']:.3f}m")
    p75 = np.percentile(errors, 75)
    p95 = np.percentile(errors, 95)
    ax.axvline(p75, color='orange', linestyle='--', linewidth=2, label=f"75%: {p75:.3f}m")
    ax.axvline(p95, color='red', linestyle='--', linewidth=2, label=f"95%: {p95:.3f}m")
    ax.set_xlabel('Error (m)', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('CDF', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = (
        f"ATE Statistics\n"
        f"{'='*28}\n"
        f"RMSE:      {ate_stats['rmse']:.4f} m\n"
        f"Mean:      {ate_stats['mean']:.4f} m\n"
        f"Median:    {ate_stats['median']:.4f} m\n"
        f"Std:       {ate_stats['std']:.4f} m\n"
        f"Min:       {ate_stats['min']:.4f} m\n"
        f"Max:       {ate_stats['max']:.4f} m\n"
        f"{'='*28}\n"
        f"Samples:   {len(errors)}"
    )
    ax.text(0.1, 0.5, stats_text, fontsize=13, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[可视化] 误差分布图已保存: {output_path}")


def plot_rpe_analysis(trans_errors: np.ndarray, rot_errors: np.ndarray,
                     output_path: str = "rpe_analysis.png",
                     title: str = "Relative Pose Error Analysis"):
    """
    绘制RPE分析图

    参数:
        trans_errors: 平移误差数组 (N,)
        rot_errors: 旋转误差数组（度） (N,)
        output_path: 输出文件路径
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 平移误差时序
    ax = axes[0, 0]
    ax.plot(trans_errors, 'b-', linewidth=1, alpha=0.7)
    ax.axhline(np.mean(trans_errors), color='r', linestyle='--', linewidth=2,
               label=f"Mean: {np.mean(trans_errors):.4f}m")
    ax.set_xlabel('Frame Pair Index', fontsize=11)
    ax.set_ylabel('Translation Error (m)', fontsize=11)
    ax.set_title('RPE Translation Error', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 旋转误差时序
    ax = axes[0, 1]
    ax.plot(rot_errors, 'g-', linewidth=1, alpha=0.7)
    ax.axhline(np.mean(rot_errors), color='r', linestyle='--', linewidth=2,
               label=f"Mean: {np.mean(rot_errors):.4f} deg")
    ax.set_xlabel('Frame Pair Index', fontsize=11)
    ax.set_ylabel('Rotation Error (deg)', fontsize=11)
    ax.set_title('RPE Rotation Error', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 平移误差直方图
    ax = axes[1, 0]
    ax.hist(trans_errors, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(trans_errors), color='r', linestyle='--', linewidth=2,
               label=f"Mean: {np.mean(trans_errors):.4f}m")
    ax.set_xlabel('Translation Error (m)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Translation Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 旋转误差直方图
    ax = axes[1, 1]
    ax.hist(rot_errors, bins=40, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rot_errors), color='r', linestyle='--', linewidth=2,
               label=f"Mean: {np.mean(rot_errors):.4f} deg")
    ax.set_xlabel('Rotation Error (deg)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Rotation Error Distribution', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[可视化] RPE分析图已保存: {output_path}")
