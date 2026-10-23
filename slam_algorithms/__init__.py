"""
SLAM算法包初始化
提供统一的算法接口和管理功能
"""

from .base_slam import (
    BaseSLAMAlgorithm,
    SLAMAlgorithmManager,
    SensorData,
    SensorType,
    SLAMType
)
from .lightweight_slam import LightweightSLAM

try:
    from .kiss_icp import KISSICP
except ImportError:
    KISSICP = None

try:
    from .lio_sam import LIOSAM
except ImportError:
    LIOSAM = None

# 导出主要类
__all__ = [
    'BaseSLAMAlgorithm',
    'SLAMAlgorithmManager',
    'SensorData',
    'SensorType',
    'SLAMType',
    'LightweightSLAM',
    'KISSICP',
    'LIOSAM'
]

# 预定义算法配置
DEFAULT_CONFIGS = {
    "lightweight": {
        "voxel_size": 0.5,
        "max_corr_dist": 1.2,
        "min_range": 2.0,
        "max_range": 60.0,
        "icp_max_iter": 60,
        "icp_fitness_min": 0.25,
        "icp_rmse_max": 0.60,
        "keyframe_trans_th": 1.0,
        "keyframe_rot_deg_th": 8.0,
        "submap_size": 12
    },
    "kiss_icp": {
        "voxel_size": 0.5,
        "min_range": 2.0,
        "max_range": 55.0,
        "z_min": -3.0,
        "z_max": 5.0,
        "ground_z": -1.5,
        "max_corr_dist": 1.5,
        "icp_max_iter": 50,
        "icp_fitness_min": 0.25,
        "icp_rmse_max": 0.60,
        "keyframe_trans_th": 1.5,
        "keyframe_rot_deg_th": 12.0,
        "map_window_size": 20,
    },
    "lio_sam": {
        "voxel_size": 0.5,
        "min_range": 2.0,
        "max_range": 55.0,
        "z_min": -2.0,
        "z_max": 3.0,
        "ground_filter": True,
        "ground_ransac_thresh": 0.15,
        "max_corr_dist": 1.5,
        "icp_max_iter": 50,
        "icp_fitness_min": 0.3,
        "icp_rmse_max": 0.5,
        "keyframe_trans_th": 1.0,
        "keyframe_rot_deg_th": 10.0,
        "map_window_size": 30,
        "trans_init_scale": 1.0,
    }
}

def create_algorithm_manager() -> SLAMAlgorithmManager:
    """创建预配置的算法管理器"""
    manager = SLAMAlgorithmManager()

    # 注册可用算法
    manager.register_algorithm("lightweight", LightweightSLAM, DEFAULT_CONFIGS["lightweight"])
    if KISSICP is not None:
        manager.register_algorithm("kiss_icp", KISSICP, DEFAULT_CONFIGS["kiss_icp"])
    if LIOSAM is not None:
        manager.register_algorithm("lio_sam", LIOSAM, DEFAULT_CONFIGS["lio_sam"])

    return manager

def get_algorithm_info():
    """获取所有算法的信息"""
    manager = create_algorithm_manager()
    return manager.list_algorithms()
