"""
手动控制器
支持键盘输入的手动车辆控制
"""

import math
import time
import threading
import carla
from typing import Dict, Any
from .base_controller import BaseController


class ManualController(BaseController):
    """手动控制器"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "manual_controller"

        # 控制参数
        self.throttle_increment = self.config.get("throttle_increment", 0.1)
        self.brake_increment = self.config.get("brake_increment", 0.2)
        self.steer_increment = self.config.get("steer_increment", 0.1)
        self.steer_decay = self.config.get("steer_decay", 0.95)  # 转向自动回正

        # 控制状态
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_steer = 0.0

        # 键盘状态
        self.keys_pressed = set()
        self.input_enabled = False
        self.input_thread = None
        self.running = False

        # 控制映射
        self.key_mapping = {
            'w': 'throttle_up',
            's': 'brake_up',
            'a': 'steer_left',
            'd': 'steer_right',
            ' ': 'emergency_brake',  # 空格键
            'r': 'reset_controls'
        }

        print("手动控制器已初始化")
        print("控制说明:")
        print("  W - 油门")
        print("  S - 刹车")
        print("  A - 左转")
        print("  D - 右转")
        print("  空格 - 紧急制动")
        print("  R - 重置控制")

    def activate(self):
        """激活手动控制"""
        super().activate()
        self.start_input_monitoring()
        print("✅ 手动控制已激活")

    def deactivate(self):
        """停用手动控制"""
        super().deactivate()
        self.stop_input_monitoring()
        self.reset()
        print("⏹️ 手动控制已停用")

    def start_input_monitoring(self):
        """开始键盘输入监控"""
        if self.input_thread is None or not self.input_thread.is_alive():
            self.running = True
            self.input_enabled = True
            self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self.input_thread.start()

    def stop_input_monitoring(self):
        """停止键盘输入监控"""
        self.running = False
        self.input_enabled = False
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)

    def _input_loop(self):
        """输入监控循环（简化版本）"""
        # 注意：这是一个简化的输入系统
        # 在实际应用中可能需要更高级的键盘监听库如pynput
        print("键盘输入监控已启动 (简化版本)")
        print("注意：此版本使用简化的输入系统")

        while self.running and self.input_enabled:
            try:
                # 这里使用简化的状态管理
                # 实际的键盘监听需要更复杂的实现
                time.sleep(0.05)  # 20Hz更新频率

                # 在实际实现中，这里会检测实际的键盘状态


            except Exception as e:
                print(f"输入监控错误: {e}")
                break

    def update(self, world: carla.World, vehicle: carla.Vehicle, dt: float) -> carla.VehicleControl:
        """更新手动控制"""
        if not self.is_active:
            return carla.VehicleControl()

        try:
            # 处理键盘输入
            self._process_input(dt)

            # 创建控制命令
            control = carla.VehicleControl(
                throttle=self.current_throttle,
                steer=self.current_steer,
                brake=self.current_brake,
                hand_brake=False,
                reverse=False
            )

            # 应用安全限制
            control = self.apply_safety_limits(control, vehicle)

            return control

        except Exception as e:
            print(f"手动控制器错误: {e}")
            return carla.VehicleControl(brake=1.0)

    def _process_input(self, dt: float):
        """处理输入命令"""
        # 转向自动回正
        self.current_steer *= self.steer_decay

        # 简化的输入处理
        # 在实际实现中，这里会根据self.keys_pressed来更新控制值

        # 限制控制值范围
        self.current_throttle = self.clamp(self.current_throttle, 0.0, 1.0)
        self.current_brake = self.clamp(self.current_brake, 0.0, 1.0)
        self.current_steer = self.clamp(self.current_steer, -1.0, 1.0)

    def set_manual_input(self, throttle: float = None, brake: float = None, steer: float = None):
        """
        直接设置控制输入（用于外部控制）

        Args:
            throttle: 油门值 [0.0, 1.0]
            brake: 刹车值 [0.0, 1.0]
            steer: 转向值 [-1.0, 1.0]
        """
        if throttle is not None:
            self.current_throttle = self.clamp(throttle, 0.0, 1.0)
        if brake is not None:
            self.current_brake = self.clamp(brake, 0.0, 1.0)
        if steer is not None:
            self.current_steer = self.clamp(steer, -1.0, 1.0)

    def apply_key_command(self, command: str, pressed: bool = True):
        """
        应用键盘命令（用于外部键盘监听）

        Args:
            command: 命令名称
            pressed: 是否按下（True）或释放（False）
        """
        if not self.is_active:
            return

        if pressed:
            if command == 'throttle_up':
                self.current_throttle = min(1.0, self.current_throttle + self.throttle_increment)
                self.current_brake = 0.0  # 油门和刹车互斥
            elif command == 'brake_up':
                self.current_brake = min(1.0, self.current_brake + self.brake_increment)
                self.current_throttle = 0.0  # 油门和刹车互斥
            elif command == 'steer_left':
                self.current_steer = max(-1.0, self.current_steer - self.steer_increment)
            elif command == 'steer_right':
                self.current_steer = min(1.0, self.current_steer + self.steer_increment)
            elif command == 'emergency_brake':
                self.current_throttle = 0.0
                self.current_brake = 1.0
                self.current_steer = 0.0
            elif command == 'reset_controls':
                self.reset()
        else:
            # 按键释放时的处理
            if command in ['throttle_up', 'brake_up']:
                # 逐渐减少油门/刹车
                self.current_throttle *= 0.8
                self.current_brake *= 0.8

    def reset(self):
        """重置控制器状态"""
        self.current_throttle = 0.0
        self.current_brake = 0.0
        self.current_steer = 0.0
        self.keys_pressed.clear()

    def get_status_info(self) -> Dict[str, Any]:
        """获取控制器状态信息"""
        info = self.get_info()
        info.update({
            "throttle": self.current_throttle,
            "brake": self.current_brake,
            "steer": self.current_steer,
            "input_enabled": self.input_enabled,
            "control_mode": "manual"
        })
        return info

    def __del__(self):
        """析构函数，确保清理资源"""
        self.stop_input_monitoring()
