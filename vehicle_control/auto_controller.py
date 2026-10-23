"""
自动驾驶控制器
基于Pure Pursuit算法的路径跟踪控制
"""

import math
import carla
from typing import Dict, Any
from .base_controller import BaseController


class AutoController(BaseController):
    """自动驾驶控制器"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "auto_controller"

        # Pure Pursuit参数
        self.target_speed_kmh = self.config.get("target_speed_kmh", 12.0)
        self.lookahead_m = self.config.get("lookahead_m", 5.0)
        self.wheelbase_m = self.config.get("wheelbase_m", 2.80)
        self.kp_throttle = self.config.get("kp_throttle", 0.08)

        # 状态变量
        self.last_move_time = 0
        self.stuck_threshold = 6.0  # 卡住检测阈值（秒）

    def update(self, world: carla.World, vehicle: carla.Vehicle, dt: float) -> carla.VehicleControl:
        """更新自动驾驶控制"""
        if not self.is_active:
            return carla.VehicleControl()

        try:
            # 执行Pure Pursuit控制
            throttle, steer, brake = self._pure_pursuit_control(world, vehicle)

            # 创建控制命令
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False
            )

            # 应用安全限制
            control = self.apply_safety_limits(control, vehicle)

            return control

        except Exception as e:
            print(f"自动控制器错误: {e}")
            # 紧急刹车
            return carla.VehicleControl(brake=1.0)

    def _pure_pursuit_control(self, world: carla.World, vehicle: carla.Vehicle):
        """Pure Pursuit路径跟踪控制"""
        map_obj = world.get_map()
        transform = vehicle.get_transform()
        current_location = transform.location
        yaw = math.radians(transform.rotation.yaw)

        # 获取当前路点
        current_wp = map_obj.get_waypoint(
            current_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if current_wp is None:
            return 0.2, 0.0, 0.0

        # 选择目标路点
        target_wp = self._choose_next_waypoint(current_wp, self.lookahead_m)
        if target_wp is None:
            return 0.1, 0.0, 0.0

        # 计算转向角
        target_location = target_wp.transform.location
        dx = target_location.x - current_location.x
        dy = target_location.y - current_location.y
        alpha = self._wrap_angle(math.atan2(dy, dx) - yaw)

        # Pure Pursuit算法
        distance = max(1.5, current_location.distance(target_location))
        curvature = 2.0 * math.sin(alpha) / distance
        steer = self.clamp(curvature * self.wheelbase_m, -self.max_steer, self.max_steer)

        # 速度控制
        velocity = vehicle.get_velocity()
        current_speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_error = self.target_speed_kmh - current_speed_kmh

        if speed_error >= 0:
            throttle = self.clamp(self.kp_throttle * speed_error, 0.0, 0.3)
            brake = 0.0
        else:
            throttle = 0.0
            brake = self.clamp(-0.04 * speed_error, 0.0, 0.7)

        return throttle, steer, brake

    def _choose_next_waypoint(self, current_wp, lookahead):
        """选择下一个目标路点"""
        candidates = current_wp.next(lookahead)
        if not candidates:
            # 尝试更短的前瞻距离
            for distance in (lookahead * 0.7, lookahead * 0.5, 3.0, 1.5):
                candidates = current_wp.next(distance)
                if candidates:
                    break

        if not candidates:
            return None

        # 选择角度变化最小的路点
        current_yaw = math.radians(current_wp.transform.rotation.yaw)

        def yaw_diff(wp):
            wp_yaw = math.radians(wp.transform.rotation.yaw)
            return abs(self._wrap_angle(wp_yaw - current_yaw))

        return min(candidates, key=yaw_diff)

    def _wrap_angle(self, angle):
        """角度归一化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def reset(self):
        """重置控制器状态"""
        self.last_move_time = 0

    def check_vehicle_stuck(self, vehicle: carla.Vehicle, current_time: float) -> bool:
        """检查车辆是否卡住"""
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        if speed > 1.0:  # 如果在移动，更新时间
            self.last_move_time = current_time
            return False
        else:
            # 检查是否超过卡住阈值
            return (current_time - self.last_move_time) > self.stuck_threshold

    def get_status_info(self) -> Dict[str, Any]:
        """获取控制器状态信息"""
        info = self.get_info()
        info.update({
            "target_speed_kmh": self.target_speed_kmh,
            "lookahead_m": self.lookahead_m,
            "control_mode": "automatic"
        })
        return info