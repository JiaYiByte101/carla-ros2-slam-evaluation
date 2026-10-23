#!/usr/bin/env python3
"""
数据录制控制脚本
提供手动控制数据录制功能
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import time


class RecordingController(Node):
    """录制控制节点"""

    def __init__(self):
        super().__init__('recording_controller')

        # 创建发布器用于控制录制
        self.record_cmd_pub = self.create_publisher(String, '/recording/command', 10)

        print("录制控制器已启动")
        print("可用命令:")
        print("  start [场景名] - 开始录制")
        print("  stop - 停止录制")
        print("  status - 查看录制状态")
        print("  quit - 退出控制器")
        print()

    def send_command(self, command: str):
        """发送录制控制命令"""
        msg = String()
        msg.data = command
        self.record_cmd_pub.publish(msg)
        print(f"命令已发送: {command}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 命令行模式
        rclpy.init()
        controller = RecordingController()

        command = " ".join(sys.argv[1:])
        controller.send_command(command)

        # 等待一下确保消息发送
        time.sleep(0.5)

        controller.destroy_node()
        rclpy.shutdown()

    else:
        # 交互模式
        rclpy.init()
        controller = RecordingController()

        try:
            while True:
                user_input = input("录制控制> ").strip()

                if user_input.lower() in ['quit', 'q', 'exit']:
                    break
                elif user_input:
                    controller.send_command(user_input)

                # 处理ROS回调
                rclpy.spin_once(controller, timeout_sec=0.1)

        except KeyboardInterrupt:
            print("\n退出录制控制器")
        except EOFError:
            print("\n退出录制控制器")
        finally:
            controller.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()