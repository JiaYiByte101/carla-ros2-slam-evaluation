"""
车辆控制器基类
定义统一的控制接口
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import carla


class BaseController(ABC):
    """车辆控制器基类"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化控制器

        Args:
            config: 控制器配置参数
        """
        self.config = config or {}
        self.name = "base_controller"
        self.is_active = False

        # 基础控制参数
        self.max_throttle = self.config.get("max_throttle", 1.0)
        self.max_brake = self.config.get("max_brake", 1.0)
        self.max_steer = self.config.get("max_steer", 1.0)

        # 安全参数
        self.enable_safety = self.config.get("enable_safety", True)
        self.max_speed_kmh = self.config.get("max_speed_kmh", 60.0)

    @abstractmethod
    def update(self, world: carla.World, vehicle: carla.Vehicle, dt: float) -> carla.VehicleControl:
        """
        更新控制器并返回控制命令

        Args:
            world: CARLA世界对象
            vehicle: 车辆对象
            dt: 时间步长

        Returns:
            车辆控制命令
        """
        pass

    @abstractmethod
    def reset(self):
        """重置控制器状态"""
        pass

    def activate(self):
        """激活控制器"""
        self.is_active = True

    def deactivate(self):
        """停用控制器"""
        self.is_active = False

    def get_info(self) -> Dict[str, Any]:
        """获取控制器信息"""
        return {
            "name": self.name,
            "is_active": self.is_active,
            "config": self.config
        }

    def clamp(self, value: float, min_val: float, max_val: float) -> float:
        """限制数值范围"""
        return max(min_val, min(max_val, value))

    def apply_safety_limits(self, control: carla.VehicleControl, vehicle: carla.Vehicle) -> carla.VehicleControl:
        """应用安全限制"""
        if not self.enable_safety:
            return control

        # 速度限制
        velocity = vehicle.get_velocity()
        speed_kmh = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5

        if speed_kmh > self.max_speed_kmh:
            # 超速时强制刹车
            control.throttle = 0.0
            control.brake = min(1.0, (speed_kmh - self.max_speed_kmh) / 10.0)

        # 限制控制量范围
        control.throttle = self.clamp(control.throttle, 0.0, self.max_throttle)
        control.brake = self.clamp(control.brake, 0.0, self.max_brake)
        control.steer = self.clamp(control.steer, -self.max_steer, self.max_steer)

        return control