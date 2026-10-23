"""
轨迹对齐算法
实现SE(3)和Sim(3)对齐
"""

import numpy as np
from typing import Tuple


def align_trajectories(gt_poses: np.ndarray, est_poses: np.ndarray,
                      method: str = 'sim3') -> Tuple[np.ndarray, dict]:
    """
    对齐两条轨迹

    参数:
        gt_poses: Ground truth位姿 (N, 3) - xyz坐标
        est_poses: 估计位姿 (N, 3) - xyz坐标
        method: 对齐方法 ('translation', 'se3', 'sim3')

    返回:
        (对齐后的估计位姿, 变换参数字典)
    """
    if gt_poses.shape[0] != est_poses.shape[0]:
        raise ValueError(f"位姿数量不匹配: GT={gt_poses.shape[0]}, Est={est_poses.shape[0]}")

    if method == 'translation':
        return align_translation(gt_poses, est_poses)
    elif method == 'se3':
        return align_se3(gt_poses, est_poses)
    elif method == 'sim3':
        return align_sim3(gt_poses, est_poses)
    else:
        raise ValueError(f"未知对齐方法: {method}")


def align_translation(gt_poses: np.ndarray, est_poses: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    仅平移对齐（将估计轨迹的质心移动到GT质心）
    """
    gt_centroid = np.mean(gt_poses, axis=0)
    est_centroid = np.mean(est_poses, axis=0)

    translation = gt_centroid - est_centroid
    aligned_est = est_poses + translation

    params = {
        'method': 'translation',
        'translation': translation.tolist()
    }

    return aligned_est, params


def align_se3(gt_poses: np.ndarray, est_poses: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    SE(3)对齐（旋转+平移，无缩放）
    使用Kabsch算法
    """
    # 去中心化
    gt_centroid = np.mean(gt_poses, axis=0)
    est_centroid = np.mean(est_poses, axis=0)

    gt_centered = gt_poses - gt_centroid
    est_centered = est_poses - est_centroid

    # 计算协方差矩阵
    H = est_centered.T @ gt_centered

    # SVD分解
    U, S, Vt = np.linalg.svd(H)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 确保是右手坐标系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 计算平移
    t = gt_centroid - R @ est_centroid

    # 应用变换
    aligned_est = (R @ est_poses.T).T + t

    params = {
        'method': 'se3',
        'rotation': R.tolist(),
        'translation': t.tolist()
    }

    return aligned_est, params


def align_sim3(gt_poses: np.ndarray, est_poses: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Sim(3)对齐（旋转+平移+缩放）
    使用标准Umeyama算法（S. Umeyama, 1991）
    """
    n = gt_poses.shape[0]

    # 去中心化
    gt_centroid = np.mean(gt_poses, axis=0)
    est_centroid = np.mean(est_poses, axis=0)

    gt_centered = gt_poses - gt_centroid
    est_centered = est_poses - est_centroid

    # 计算方差
    sigma_est_sq = np.sum(est_centered ** 2) / n

    if sigma_est_sq < 1e-12:
        # 估计轨迹退化为一个点
        scale = 1.0
        R = np.eye(3)
        t = gt_centroid - est_centroid
    else:
        # 计算协方差矩阵
        H = est_centered.T @ gt_centered / n

        # SVD分解
        U, D, Vt = np.linalg.svd(H)

        # 反射修正矩阵
        S = np.ones(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1] = -1

        # 计算旋转矩阵
        R = Vt.T @ np.diag(S) @ U.T

        # 计算尺度（标准Umeyama公式）
        scale = float(np.sum(D * S) / sigma_est_sq)

        # 计算平移
        t = gt_centroid - scale * R @ est_centroid

    # 应用变换
    aligned_est = scale * (R @ est_poses.T).T + t

    params = {
        'method': 'sim3',
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'scale': float(scale)
    }

    return aligned_est, params
