#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动控制输入器：将键盘输入转换为 ROS2 话题
话题：
  - /vehicle/manual_control : std_msgs/String  控制命令
  - /vehicle/control_mode   : std_msgs/String  模式切换(auto/manual)
"""

import os
import time
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

DEBUG = False  # 便于确认是否捕获到按键/是否成功发布

# --- 键盘库选择：Xorg 优先使用 pynput；若不可用则退回 keyboard 或简化模式 ---
KEYBOARD_LIB = "simple"
keyboard = None
pynput_keyboard = None
Key = KeyCode = None

try:
    from pynput import keyboard as pynput_keyboard
    from pynput.keyboard import Key, KeyCode
    KEYBOARD_LIB = "pynput"
except Exception:
    try:
        import keyboard as kb
        keyboard = kb
        KEYBOARD_LIB = "keyboard"
    except Exception:
        KEYBOARD_LIB = "simple"


class ManualControlInput(Node):
    """手动控制输入节点"""

    def __init__(self):
        super().__init__('manual_control_input')

        # 发布器
        self.control_pub = self.create_publisher(String, '/vehicle/manual_control', 10)
        self.mode_pub = self.create_publisher(String, '/vehicle/control_mode', 10)

        # 状态
        self.running = True
        self.current_mode = "auto"
        self.keyboard_listener = None

        # 字符键映射
        self.key_mapping = {
            'w': 'throttle_up',
            's': 'brake_up',
            'a': 'steer_left',
            'd': 'steer_right',
            'r': 'reset_controls'
        }

        session = os.environ.get("XDG_SESSION_TYPE", "unknown")
        print("手动控制输入器已启动")
        print(f"桌面会话: {session} | 使用键盘库: {KEYBOARD_LIB}")
        print("控制说明:")
        print("  W - 油门")
        print("  S - 刹车")
        print("  A - 左转")
        print("  D - 右转")
        print("  空格 - 紧急制动")
        print("  R - 重置控制")
        print("  Tab - 切换控制模式")
        print("  ESC/Q - 退出\n")

    # ------------------------ 发送函数 ------------------------
    def send_control_command(self, command: str):
        if DEBUG:
            print(f"[PUB] /vehicle/manual_control -> {command}", flush=True)
        try:
            self.control_pub.publish(String(data=command))
        except Exception as e:
            print(f"[WARN] 发布 manual_control 失败: {e}", flush=True)

    def send_mode_command(self, mode: str):
        # 先打印，再尝试发布；即使发布失败也能看到切换提示
        print(f"[MODE] 切换到 {mode} 模式", flush=True)
        try:
            self.mode_pub.publish(String(data=mode))
        except Exception as e:
            print(f"[WARN] 发布 control_mode 失败: {e}", flush=True)

    def toggle_control_mode(self):
        new_mode = "manual" if self.current_mode == "auto" else "auto"
        self.current_mode = new_mode
        self.send_mode_command(new_mode)

    # ------------------------ 入口 ------------------------
    def start_keyboard_monitoring(self):
        if KEYBOARD_LIB == "pynput":
            self._start_pynput()
        elif KEYBOARD_LIB == "keyboard":
            self._start_keyboard_lib()
        else:
            self._start_simple_input()

    # ------------------------ pynput 分支（Xorg） ------------------------
    def _start_pynput(self):
        print("使用 pynput 进行实时键盘监控（ESC/Q 退出）")

        def _is_tab(k):
            # 兼容 Key.tab 或 KeyCode('\t')
            try:
                return (k == Key.tab) or (isinstance(k, KeyCode) and getattr(k, 'char', '') == '\t')
            except Exception:
                return False

        def _is_space(k):
            try:
                return (k == Key.space) or (isinstance(k, KeyCode) and getattr(k, 'char', '') == ' ')
            except Exception:
                return False

        def _is_esc(k):
            try:
                return (k == Key.esc) or (isinstance(k, KeyCode) and getattr(k, 'char', '') == '\x1b')
            except Exception:
                return False

        def on_key_press(key):
            try:
                if DEBUG:
                    print(f"[PRESS] {key!r}", flush=True)

                # --- 特殊键（优先判断） ---
                if _is_tab(key):
                    print("[DEBUG] TAB detected", flush=True)
                    self.toggle_control_mode()
                    return
                if _is_space(key):
                    self.send_control_command("emergency_brake:pressed")
                    return
                if _is_esc(key):
                    print("退出程序", flush=True)
                    self.running = False
                    return False

                # --- 字符键 ---
                if isinstance(key, KeyCode) and key.char:
                    ch = key.char.lower()
                    if ch == 'q':
                        print("退出程序", flush=True)
                        self.running = False
                        return False
                    if ch in self.key_mapping:
                        cmd = self.key_mapping[ch]
                        self.send_control_command(f"{cmd}:pressed")

            except Exception as e:
                print(f"[ERR] on_key_press 异常: {e}", flush=True)

        def on_key_release(key):
            try:
                if DEBUG:
                    print(f"[RELEASE] {key!r}", flush=True)

                if isinstance(key, KeyCode) and getattr(key, 'char', None):
                    ch = key.char.lower()
                    if ch in self.key_mapping:
                        cmd = self.key_mapping[ch]
                        if cmd in ['throttle_up', 'brake_up', 'steer_left', 'steer_right']:
                            self.send_control_command(f"{cmd}:released")
                if _is_esc(key):
                    return False
            except Exception as e:
                print(f"[ERR] on_key_release 异常: {e}", flush=True)

        try:
            self.keyboard_listener = pynput_keyboard.Listener(
                on_press=on_key_press,
                on_release=on_key_release
            )
            self.keyboard_listener.start()
            while self.running:
                time.sleep(0.1)
        except Exception as e:
            print(f"pynput 监听错误: {e}", flush=True)
            self.running = False
        finally:
            if self.keyboard_listener:
                try:
                    self.keyboard_listener.stop()
                except Exception:
                    pass

    # ------------------------ keyboard 分支（Wayland/需权限） ------------------------
    def _start_keyboard_lib(self):
        print("使用 keyboard 库进行实时键盘监控（Linux 下通常需 sudo 或 udev 授权）")

        def on_key_event(event):
            name = (event.name or '').lower()
            if DEBUG:
                print(f"[{event.event_type}] {name}", flush=True)

            if event.event_type == keyboard.KEY_DOWN:
                if name in ('q', 'esc'):
                    self.running = False
                    return
                if name == 'tab':
                    print("[DEBUG] TAB detected", flush=True)
                    self.toggle_control_mode()
                    return
                if name == 'space':
                    self.send_control_command("emergency_brake:pressed")
                    return
                if name in self.key_mapping:
                    self.send_control_command(f"{self.key_mapping[name]}:pressed")

            elif event.event_type == keyboard.KEY_UP:
                if name in self.key_mapping and name != 'r':
                    self.send_control_command(f"{self.key_mapping[name]}:released")

        try:
            keyboard.hook(on_key_event)
            while self.running:
                time.sleep(0.1)
        except Exception as e:
            print(f"keyboard 库错误: {e}\n提示：Wayland 下可通过加入 input 组/udev 规则避免 sudo。", flush=True)
        finally:
            try:
                keyboard.unhook_all()
            except Exception:
                pass

    # ------------------------ 简化输入分支 ------------------------
    def _start_simple_input(self):
        print("使用简化输入模式（w/s/a/d/space/r/tab/q）")
        try:
            while self.running:
                s = input("控制> ").strip().lower()
                if s in ('q', 'quit', 'exit'):
                    break
                if s == 'tab':
                    print("[DEBUG] TAB detected", flush=True)
                    self.toggle_control_mode()
                elif s == 'space':
                    self.send_control_command("emergency_brake:pressed")
                elif s in self.key_mapping:
                    cmd = self.key_mapping[s]
                    self.send_control_command(f"{cmd}:pressed")
                    time.sleep(0.1)
                    if s != 'r':
                        self.send_control_command(f"{cmd}:released")
                else:
                    print("可用命令: w, s, a, d, space, r, tab, q")
        except (KeyboardInterrupt, EOFError):
            pass

    # ------------------------ 关闭 ------------------------
    def shutdown(self):
        self.running = False
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
        print("手动控制输入器已关闭")


# ======================== main ========================
def main():
    rclpy.init()
    node = None
    try:
        node = ManualControlInput()
        th = threading.Thread(target=node.start_keyboard_monitoring, daemon=True)
        th.start()
        while node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\n退出控制器")
    finally:
        if node:
            node.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
