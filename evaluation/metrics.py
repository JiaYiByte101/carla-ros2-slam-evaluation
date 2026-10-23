"""
SLAM评估指标计算
实现ATE（绝对轨迹误差）和RPE（相对位姿误差）
"""

import numpy as np
from typing import Tuple, Dict, List


def compute_ate(gt_poses: np.ndarray, est_poses: np.ndarray) -> Dict[str, float]:
    """
    计算绝对轨迹误差（Absolute Trajectory Error）

    参数:
        gt_poses: Ground truth位姿 (N, 3) - xyz坐标
        est_poses: 估计位姿 (N, 3) - xyz坐标

    返回:
        包含RMSE、Mean、Median、Std、Min、Max的字典
    """
    if gt_poses.shape[0] != est_poses.shape[0]:
        raise ValueError(f"位姿数量不匹配: GT={gt_poses.shape[0]}, Est={est_poses.shape[0]}")

    if gt_poses.shape[0] == 0:
        return {
            'rmse': 0.0, 'mean': 0.0, 'median': 0.0,
            'std': 0.0, 'min': 0.0, 'max': 0.0
        }

    # 计算欧氏距离误差
    errors = np.linalg.norm(gt_poses - est_poses, axis=1)

    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors))
    }


def compute_rpe(gt_poses: np.ndarray, est_poses: np.ndarray,
                delta: int = 10) -> Dict[str, float]:
    """
    计算相对位姿误差（Relative Pose Error）

    参数:
        gt_poses: Ground truth位姿 (N, 7) - [x, y, z, qx, qy, qz, qw]
        est_poses: 估计位姿 (N, 7) - [x, y, z, qx, qy, qz, qw]
        delta: 帧间隔（默认10帧）

    返回:
        包含平移和旋转误差统计的字典
    """
    if gt_poses.shape[0] != est_poses.shape[0]:
        raise ValueError(f"位姿数量不匹配: GT={gt_poses.shape[0]}, Est={est_poses.shape[0]}")

    n = gt_poses.shape[0]
    if n < delta + 1:
        return {
            'trans_rmse': 0.0, 'trans_mean': 0.0,
            'rot_rmse': 0.0, 'rot_mean': 0.0
        }

    trans_errors = []
    rot_errors = []

    for i in range(n - delta):
        # Ground truth相对变换
        gt_T_i = pose_to_matrix(gt_poses[i])
        gt_T_j = pose_to_matrix(gt_poses[i + delta])
        gt_rel = np.linalg.inv(gt_T_i) @ gt_T_j

        # 估计相对变换
        est_T_i = pose_to_matrix(est_poses[i])
        est_T_j = pose_to_matrix(est_poses[i + delta])
        est_rel = np.linalg.inv(est_T_i) @ est_T_j

        # 相对误差
        error_T = np.linalg.inv(gt_rel) @ est_rel

        # 平移误差
        trans_error = np.linalg.norm(error_T[:3, 3])
        trans_errors.append(trans_error)

        # 旋转误差（角度）
        rot_error = rotation_error(error_T[:3, :3])
        rot_errors.append(rot_error)

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    return {
        'trans_rmse': float(np.sqrt(np.mean(trans_errors ** 2))),
        'trans_mean': float(np.mean(trans_errors)),
        'trans_median': float(np.median(trans_errors)),
        'rot_rmse': float(np.sqrt(np.mean(rot_errors ** 2))),
        'rot_mean': float(np.mean(rot_errors)),
        'rot_median': float(np.median(rot_errors))
    }


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    将位姿向量转换为4x4变换矩阵

    参数:
        pose: [x, y, z, qx, qy, qz, qw]

    返回:
        4x4变换矩阵
    """
    T = np.eye(4)
    T[:3, 3] = pose[:3]

    # 四元数转旋转矩阵
    qx, qy, qz, qw = pose[3:7]
    T[:3, :3] = quat_to_rotation_matrix(qx, qy, qz, qw)

    return T


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """四元数转旋转矩阵"""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def rotation_error(R: np.ndarray) -> float:
    """
    计算旋转矩阵的角度误差（度）

    参数:
        R: 3x3旋转矩阵

    返回:
        角度误差（度）
    """
    trace = np.trace(R)
    # 数值稳定性处理
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle)


def associate_timestamps(gt_stamps: List[float], est_stamps: List[float],
                        max_diff: float = 0.05) -> List[Tuple[int, int]]:
    """
    时间戳关联（一对一最近邻匹配）

    参数:
        gt_stamps: Ground truth时间戳列表
        est_stamps: 估计时间戳列表
        max_diff: 最大时间差容差（秒）

    返回:
        匹配的索引对列表 [(gt_idx, est_idx), ...]
    """
    matches = []
    est_used = set()

    est_array = np.array(est_stamps)
    for i, gt_t in enumerate(gt_stamps):
        diffs = np.abs(est_array - gt_t)
        # 按距离排序，找到最近且未被使用的EST
        sorted_indices = np.argsort(diffs)
        for min_idx in sorted_indices:
            if diffs[min_idx] > max_diff:
                break  # 剩余的都超过阈值
            if int(min_idx) not in est_used:
                matches.append((i, int(min_idx)))
                est_used.add(int(min_idx))
                break

    return matches
