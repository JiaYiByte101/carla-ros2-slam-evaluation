"""
SLAM评估模块
提供轨迹对齐、误差计算和可视化功能
"""

from .metrics import compute_ate, compute_rpe
from .alignment import align_trajectories
from .visualizer import plot_trajectory_comparison, plot_error_distribution

__all__ = [
    'compute_ate',
    'compute_rpe',
    'align_trajectories',
    'plot_trajectory_comparison',
    'plot_error_distribution'
]
