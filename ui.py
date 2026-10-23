#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA-ROS2 SLAM 启动面板 (ttkbootstrap 版)
集成：桥接启动、车辆控制、数据录制、点云累积、SLAM评估、可视化、地图切换
"""

import os
import sys
import json
import time
import signal
import shlex
import threading
import subprocess
from enum import Enum
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import messagebox, filedialog

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *
    HAS_TTKBOOTSTRAP = True
except ImportError:
    # 降级到标准 ttk
    from tkinter import ttk
    HAS_TTKBOOTSTRAP = False
    print("[warn] ttkbootstrap 未安装，使用标准主题。运行: pip install ttkbootstrap")


# ==================== 工具函数 ====================

def expand(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(p)) if p else p


def bash_wrap(cmd: str, ros_setup: str | None) -> str:
    if ros_setup:
        return f"bash -lc 'source {shlex.quote(ros_setup)} >/dev/null 2>&1; {cmd}'"
    else:
        return f"bash -lc '{cmd}'"


# ==================== 进程状态枚举 ====================

class ProcState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    CRASHED = "crashed"


# ==================== 增强版进程管理器 ====================

class Proc:
    def __init__(self, name: str, cmd: str, cwd: str | None, on_log,
                 on_state_change=None):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.on_log = on_log
        self.on_state_change = on_state_change
        self.p: subprocess.Popen | None = None
        self._th: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self.state = ProcState.STOPPED
        self.start_time: float | None = None

    def _set_state(self, s: ProcState):
        self.state = s
        if self.on_state_change:
            try:
                self.on_state_change(self.name, s)
            except Exception:
                pass

    def start(self):
        if self.p and self.p.poll() is None:
            self.on_log(self.name, "[warn] 进程已在运行\n")
            return
        self.on_log(self.name, f"[cmd] {self.cmd}\n")
        try:
            self.p = subprocess.Popen(
                self.cmd,
                cwd=self.cwd or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                preexec_fn=os.setsid
            )
        except Exception as e:
            self.on_log(self.name, f"[error] 启动失败：{e}\n")
            self._set_state(ProcState.CRASHED)
            return
        self.start_time = time.time()
        self._stop_evt.clear()
        self._set_state(ProcState.RUNNING)
        self._th = threading.Thread(target=self._pump, daemon=True)
        self._th.start()

    def _pump(self):
        assert self.p is not None
        f = self.p.stdout
        if f is None:
            return
        for line in iter(f.readline, b""):
            if self._stop_evt.is_set():
                break
            try:
                s = line.decode(errors="ignore")
            except Exception:
                s = str(line)
            self.on_log(self.name, s)
        rc = self.p.poll()
        self.on_log(self.name, f"\n[exit] 退出码：{rc}\n")
        if rc is not None and rc != 0 and not self._stop_evt.is_set():
            self._set_state(ProcState.CRASHED)
        else:
            self._set_state(ProcState.STOPPED)

    def stop(self, force_after=3.0):
        if not self.p or self.p.poll() is not None:
            self._set_state(ProcState.STOPPED)
            return
        self._stop_evt.set()
        self.on_log(self.name, "[info] 正在终止...\n")
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
        except Exception:
            try:
                self.p.terminate()
            except Exception:
                pass
        t0 = time.time()
        while time.time() - t0 < force_after:
            if self.p.poll() is not None:
                self._set_state(ProcState.STOPPED)
                return
            time.sleep(0.1)
        self.on_log(self.name, "[warn] 升级为 SIGKILL\n")
        try:
            os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass
        self._set_state(ProcState.STOPPED)

    def is_running(self) -> bool:
        return self.p is not None and self.p.poll() is None


# ==================== 底部状态栏 ====================

class StatusBar(ttk.Frame):
    """底部状态栏：进程指示器 + 录制状态 + 控制模式 + 运行时间"""

    PROC_LABELS = {
        "official": "桥接",
        "all": "SLAM",
        "pcd": "PCD",
        "manual": "手控",
        "evaluator": "评估",
        "viz": "可视",
    }

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._indicators: dict[str, ttk.Label] = {}
        self._dots: dict[str, ttk.Label] = {}

        # 进程指示器区域
        proc_frame = ttk.Frame(self)
        proc_frame.pack(side=tk.LEFT, padx=8)
        for key, label in self.PROC_LABELS.items():
            dot = ttk.Label(proc_frame, text="●", font=("", 11))
            dot.pack(side=tk.LEFT, padx=(0, 1))
            lbl = ttk.Label(proc_frame, text=label, font=("", 9))
            lbl.pack(side=tk.LEFT, padx=(0, 10))
            self._dots[key] = dot
            self._indicators[key] = lbl
            self._color_dot(key, ProcState.STOPPED)

        # 分隔
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                       padx=8, pady=2)

        # 录制状态
        self.rec_label = ttk.Label(self, text="录制: 停止", font=("", 9))
        self.rec_label.pack(side=tk.LEFT, padx=8)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                       padx=8, pady=2)

        # 控制模式
        self.mode_label = ttk.Label(self, text="控制: 自动", font=("", 9))
        self.mode_label.pack(side=tk.LEFT, padx=8)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                       padx=8, pady=2)

        # 运行时间
        self.time_label = ttk.Label(self, text="运行时间 00:00:00", font=("", 9))
        self.time_label.pack(side=tk.LEFT, padx=8)

    def _color_dot(self, key: str, state: ProcState):
        dot = self._dots.get(key)
        if not dot:
            return
        if state == ProcState.RUNNING:
            dot.configure(foreground="#00cc66")
        elif state == ProcState.CRASHED:
            dot.configure(foreground="#ff4444")
        else:
            dot.configure(foreground="#666666")

    def update_process(self, key: str, state: ProcState):
        self._color_dot(key, state)

    def update_recording(self, is_recording: bool, elapsed: str = ""):
        if is_recording:
            self.rec_label.configure(text=f"录制: 录制中 {elapsed}",
                                     foreground="#ff4444")
        else:
            self.rec_label.configure(text="录制: 停止", foreground="")

    def update_control_mode(self, mode: str):
        text = "控制: 手动" if mode == "manual" else "控制: 自动"
        self.mode_label.configure(text=text)

    def update_runtime(self, elapsed_str: str):
        self.time_label.configure(text=f"运行时间 {elapsed_str}")


# ==================== 主应用 ====================

class App:
    # 日志样式常量
    LOG_BG = "#1e1e2e"
    LOG_FG = "#cdd6f4"
    LOG_FONT = ("Consolas", 10)
    LOG_MAX_LINES = 10000

    def __init__(self):
        if HAS_TTKBOOTSTRAP:
            self.root = ttk.Window(themename="darkly",
                                   title="CARLA-ROS2 SLAM 启动面板",
                                   size=(1280, 800))
        else:
            self.root = tk.Tk()
            self.root.title("CARLA-ROS2 SLAM 启动面板")
            self.root.geometry("1280x800")

        # ---------- 变量 ----------
        self.ros_setup_var = tk.StringVar(value=os.environ.get("ROS_SETUP", ""))
        self.workdir_var = tk.StringVar(value=str(Path.cwd()))

        # 桥接
        self.official_script_var = tk.StringVar(value="official.py")

        # 数据源（实时 / 回放）
        self.data_source_var = tk.StringVar(value="live")
        self.bag_path_var = tk.StringVar(value="")
        self.playback_rate_var = tk.StringVar(value="1.0")
        self.playback_loop_var = tk.BooleanVar(value=False)

        # SLAM 算法
        self.slam_algorithm_var = tk.StringVar(value="all.py (Lightweight)")
        self.lio_sam_ws_var = tk.StringVar(value="~/lio_sam_ws")
        self.orb_slam3_ws_var = tk.StringVar(value="~/orb_slam3_ws")
        self.orb_slam3_root_var = tk.StringVar(value="~/ORB_SLAM3-master")
        self.all_script_var = tk.StringVar(value="all.py")

        # 各算法默认参数
        self._lightweight_default_params = (
            "--ros-args "
            "-p voxel_size:=0.6 "
            "-p max_corr_dist:=1.4 "
            "-p min_range:=2.0 -p max_range:=55.0 "
            "-p z_min:=-2.0 -p z_max:=3.0 "
            "-p icp_max_iter:=50 "
            "-p icp_rmse_max:=0.60 "
            "-p icp_fitness_min:=0.25 "
            "-p use_ground_removal:=true "
            "-p ground_ransac_dist:=0.15 "
            "-p keyframe_trans_th:=1.5 "
            "-p keyframe_rot_deg_th:=12.0 "
            "-p submap_size:=25 "
            "-p publish_submap:=true "
            "-p trans_init_scale:=0.4 "
            "-p lc_enable:=true "
            "-p lc_check_every_n_keyframes:=10 "
            "-p lc_recent_window:=100 "
            "-p lc_dist_thresh:=25.0 "
            "-p lc_yaw_thresh_deg:=20.0 "
            "-p lc_icp_voxel:=0.6 "
            "-p lc_icp_max_iter:=30 "
            "-p lc_icp_fitness_min:=0.35 "
            "-p lc_icp_rmse_max:=0.50 "
            "-p lc_correction_alpha:=0.35"
        )
        self._lio_sam_default_params = "(参数在 lio_sam_ws 的 params.yaml 中配置)"
        self._orb_slam3_default_params = "(参数在 orb_slam3_carla.yaml 中配置)"
        self._kiss_icp_default_params = (
            "--ros-args "
            "-p voxel_size:=0.5 "
            "-p min_range:=2.0 -p max_range:=55.0 "
            "-p z_min:=-3.0 -p z_max:=5.0 "
            "-p max_corr_dist:=1.5 "
            "-p icp_max_iter:=50 "
            "-p icp_fitness_min:=0.25 "
            "-p icp_rmse_max:=0.60 "
            "-p keyframe_trans_th:=1.5 "
            "-p keyframe_rot_deg_th:=12.0 "
            "-p map_window_size:=20 "
            "-p publish_map:=true "
            "-p map_publish_interval:=5"
        )

        self.lightweight_params_var = tk.StringVar(value=self._lightweight_default_params)
        self.lio_sam_params_var = tk.StringVar(value=self._lio_sam_default_params)
        self.kiss_icp_params_var = tk.StringVar(value=self._kiss_icp_default_params)
        self.orb_slam3_params_var = tk.StringVar(value=self._orb_slam3_default_params)
        self.all_params_var = tk.StringVar(value=self._lightweight_default_params)

        # pcd.py
        self.pcd_script_var = tk.StringVar(value="pcd.py")
        self.pcd_params_var = tk.StringVar(value=(
            "--ros-args "
            "-p topic:=/map "
            "-p map_outfile:=output/carla_global.ply "
            "-p chunk_voxel:=0.8 "
            "-p voxel_size:=0.6 "
            "-p max_points:=2000000 "
            "-p save_every_sec:=0.0 "
            "-p traj_topic:=/odom "
            "-p traj_outfile:=output/carla_odom_tum.txt "
            "-p traj_skip:=0"
        ))

        # 可视化
        self.use_sim_time_var = tk.BooleanVar(value=True)
        self.cloudcompare_cmd_var = tk.StringVar(value="CloudCompare")
        self.ply_file_var = tk.StringVar(value="output/carla_global.ply")

        # 车辆控制
        self.control_mode_var = tk.StringVar(value="auto")
        self.manual_script_var = tk.StringVar(value="manual_control_input.py")

        # 数据录制
        self.scenario_name_var = tk.StringVar(value="")
        self.recording_active = False
        self.recording_start_time: float | None = None
        self._rec_blink_state = False
        self.rec_compressed_var = tk.BooleanVar(value=False)
        self.rec_tf_var = tk.BooleanVar(value=True)
        self.rec_tf_static_var = tk.BooleanVar(value=True)

        # SLAM 评估
        self.eval_align_var = tk.StringVar(value="sim3")
        self.eval_hz_var = tk.StringVar(value="1.0")
        self.offline_gt_var = tk.StringVar(value="output/carla_ground_truth_trajectory.txt")
        self.offline_est_var = tk.StringVar(value="output/carla_odom_tum.txt")
        self.offline_sequential_var = tk.BooleanVar(value=False)
        self.offline_align_var = tk.StringVar(value="sim3")
        self.offline_output_var = tk.StringVar(value="output/slam_evaluation_offline")

        # 地图切换
        self.switch_method_var = tk.StringVar(value="python_api")
        self.map_host_var = tk.StringVar(value="127.0.0.1")
        self.map_port_var = tk.StringVar(value="2000")
        self.map_name_var = tk.StringVar(value="Town03")

        # 进程
        self.procs: dict[str, Proc] = {}
        self.log_widgets: dict[str, tk.Text] = {}

        # 应用启动时间（用于运行时间显示）
        self._app_start_time: float | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_status()

    # ==================== UI 构建 ====================

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # --- 顶部栏 ---
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        # 第一行：ROS 环境脚本
        row1 = ttk.Frame(top_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="ROS环境脚本：", width=12).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.ros_setup_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(row1, "浏览", self._pick_ros_setup).pack(side=tk.LEFT)

        # 第二行：工作目录 + 预设按钮
        row2 = ttk.Frame(top_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="工作目录：", width=12).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.workdir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(row2, "浏览", self._pick_workdir).pack(side=tk.LEFT)
        self._btn(row2, "加载预设", self._load_preset,
                  style="info-outline").pack(side=tk.RIGHT, padx=(4, 0))
        self._btn(row2, "保存预设", self._save_preset,
                  style="info-outline").pack(side=tk.RIGHT)

        # --- Notebook (7 标签页) ---
        if HAS_TTKBOOTSTRAP:
            try:
                nb = ttk.Notebook(self.root, bootstyle="dark")
            except Exception:
                nb = ttk.Notebook(self.root)
        else:
            nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.nb = nb

        tabs = [
            ("启动", self._build_launch_tab),
            ("车辆控制", self._build_vehicle_tab),
            ("数据录制", self._build_recording_tab),
            ("点云累积", self._build_pcd_tab),
            ("SLAM评估", self._build_eval_tab),
            ("可视化", self._build_viz_tab),
            ("地图切换", self._build_map_tab),
        ]
        for label, builder in tabs:
            frame = ttk.Frame(nb)
            nb.add(frame, text=f"  {label}  ")
            builder(frame)

        # --- 底部状态栏 ---
        self.status_bar = StatusBar(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 6))

    # ---------- 辅助：创建统一风格按钮 ----------
    def _btn(self, parent, text, cmd, style=None):
        if HAS_TTKBOOTSTRAP and style:
            try:
                return ttk.Button(parent, text=text, command=cmd,
                                  bootstyle=style)
            except Exception:
                pass
        return ttk.Button(parent, text=text, command=cmd)

    def _start_btn(self, parent, text, cmd):
        if HAS_TTKBOOTSTRAP:
            try:
                return ttk.Button(parent, text=text, command=cmd,
                                  bootstyle="success-outline")
            except Exception:
                pass
        return ttk.Button(parent, text=text, command=cmd)

    def _stop_btn(self, parent, text, cmd):
        if HAS_TTKBOOTSTRAP:
            try:
                return ttk.Button(parent, text=text, command=cmd,
                                  bootstyle="danger-outline")
            except Exception:
                pass
        return ttk.Button(parent, text=text, command=cmd)

    # ---------- 辅助：创建日志区 ----------
    def _make_log(self, parent, key: str, height=12) -> tk.Text:
        frame = ttk.Frame(parent)
        txt = tk.Text(frame, height=height, wrap=tk.NONE,
                      bg=self.LOG_BG, fg=self.LOG_FG,
                      font=self.LOG_FONT, insertbackground=self.LOG_FG,
                      selectbackground="#45475a", selectforeground="#cdd6f4",
                      relief=tk.FLAT, borderwidth=2)
        ybar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=txt.yview)
        xbar = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=txt.xview)
        txt.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        ybar.pack(side=tk.RIGHT, fill=tk.Y)
        xbar.pack(side=tk.BOTTOM, fill=tk.X)
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_widgets[key] = txt
        return frame

    def _make_labelframe(self, parent, text):
        """创建 LabelFrame，逐步降级：bootstyle+padding → padding → 纯文本"""
        combos = [
            dict(text=text, bootstyle="info", padding=8),
            dict(text=text, padding=8),
            dict(text=text),
        ] if HAS_TTKBOOTSTRAP else [
            dict(text=text, padding=8),
            dict(text=text),
        ]
        for kw in combos:
            try:
                return ttk.LabelFrame(parent, **kw)
            except Exception:
                continue
        return ttk.LabelFrame(parent, text=text)

    # ==================== Tab 1: 启动 ====================

    def _build_launch_tab(self, root: ttk.Frame):
        pad = {"padx": 8, "pady": 3}

        # ---- 数据源选择 ----
        src_frame = self._make_labelframe(root, "数据源")
        src_frame.pack(fill=tk.X, **pad)

        radio_row = ttk.Frame(src_frame)
        radio_row.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(radio_row, text="实时仿真 (official.py)",
                        variable=self.data_source_var, value="live",
                        command=self._on_data_source_changed).pack(
            side=tk.LEFT, padx=(4, 20))
        ttk.Radiobutton(radio_row, text="回放录制数据 (rosbag)",
                        variable=self.data_source_var, value="playback",
                        command=self._on_data_source_changed).pack(
            side=tk.LEFT)

        # 实时模式：桥接脚本
        self.live_frame = ttk.Frame(src_frame)
        self.live_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.live_frame, text="桥接脚本：", width=14).pack(side=tk.LEFT)
        ttk.Entry(self.live_frame, textvariable=self.official_script_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(self.live_frame, "浏览",
                  lambda: self._pick_script(self.official_script_var)).pack(
            side=tk.LEFT)

        # 回放模式：数据路径 + 速率 + 循环
        self.playback_frame = ttk.Frame(src_frame)
        # 不 pack，默认隐藏

        pb_row1 = ttk.Frame(self.playback_frame)
        pb_row1.pack(fill=tk.X, pady=2)
        ttk.Label(pb_row1, text="数据路径：", width=14).pack(side=tk.LEFT)
        ttk.Entry(pb_row1, textvariable=self.bag_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(pb_row1, "浏览", self._pick_bag_path).pack(side=tk.LEFT)

        pb_row2 = ttk.Frame(self.playback_frame)
        pb_row2.pack(fill=tk.X, pady=2)
        ttk.Label(pb_row2, text="回放速率：", width=14).pack(side=tk.LEFT)
        ttk.Entry(pb_row2, textvariable=self.playback_rate_var, width=6).pack(
            side=tk.LEFT)
        ttk.Label(pb_row2, text="x").pack(side=tk.LEFT, padx=(2, 16))
        ttk.Checkbutton(pb_row2, text="循环播放",
                        variable=self.playback_loop_var).pack(side=tk.LEFT)

        # SLAM 算法选择
        row = ttk.Frame(root); row.pack(fill=tk.X, **pad)
        ttk.Label(row, text="SLAM算法：", width=14).pack(side=tk.LEFT)
        algo_combo = ttk.Combobox(
            row, textvariable=self.slam_algorithm_var,
            values=["无 (仅桥接)", "all.py (Lightweight)",
                    "LIO-SAM (Official C++)",
                    "kiss_icp_node.py (KISS-ICP)",
                    "ORB-SLAM3 (Official C++ RGB-D)"],
            state="readonly", width=30
        )
        algo_combo.pack(side=tk.LEFT, padx=(0, 4))
        algo_combo.bind("<<ComboboxSelected>>", self._on_slam_algorithm_changed)

        # LIO-SAM 工作空间路径（条件显示）
        self.lio_ws_frame = ttk.Frame(root)
        ttk.Label(self.lio_ws_frame, text="LIO-SAM 工作空间：", width=14).pack(
            side=tk.LEFT)
        ttk.Entry(self.lio_ws_frame, textvariable=self.lio_sam_ws_var,
                  width=40).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=(0, 4))
        self._btn(self.lio_ws_frame, "浏览",
                  lambda: self._pick_workdir_to(self.lio_sam_ws_var)).pack(
            side=tk.LEFT)

        # ORB-SLAM3 路径（条件显示）
        self.orb_ws_frame = ttk.Frame(root)
        ttk.Label(self.orb_ws_frame, text="ORB-SLAM3 工作空间：",
                  width=14).pack(side=tk.LEFT)
        ttk.Entry(self.orb_ws_frame, textvariable=self.orb_slam3_ws_var,
                  width=25).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                 padx=(0, 4))
        self._btn(self.orb_ws_frame, "浏览",
                  lambda: self._pick_workdir_to(self.orb_slam3_ws_var)).pack(
            side=tk.LEFT, padx=(0, 8))
        ttk.Label(self.orb_ws_frame, text="库路径：").pack(side=tk.LEFT)
        ttk.Entry(self.orb_ws_frame, textvariable=self.orb_slam3_root_var,
                  width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # SLAM 脚本 + 参数
        self.slam_opts_frame = ttk.Frame(root)
        self.slam_opts_frame.pack(fill=tk.X, **pad)
        ttk.Label(self.slam_opts_frame, text="SLAM脚本：", width=14).pack(
            side=tk.LEFT)
        self.slam_script_entry = ttk.Entry(self.slam_opts_frame,
                                           textvariable=self.all_script_var)
        self.slam_script_entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                                    padx=(0, 4))
        self._btn(self.slam_opts_frame, "浏览",
                  lambda: self._pick_script(self.all_script_var)).pack(
            side=tk.LEFT)

        self.slam_params_frame = ttk.Frame(root)
        self.slam_params_frame.pack(fill=tk.X, **pad)
        ttk.Label(self.slam_params_frame, text="SLAM参数：", width=14).pack(
            side=tk.LEFT)
        self.slam_params_entry = ttk.Entry(self.slam_params_frame,
                                           textvariable=self.all_params_var)
        self.slam_params_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 启动 / 停止 按钮
        row = ttk.Frame(root); row.pack(fill=tk.X, **pad)
        self._start_btn(row, "  启动  ", self.start_bridge_and_slam).pack(
            side=tk.LEFT)
        self._stop_btn(row, "  停止  ", self.stop_bridge_and_slam).pack(
            side=tk.LEFT, padx=8)
        self.launch_status_label = ttk.Label(row, text="")
        self.launch_status_label.pack(side=tk.LEFT, padx=10)

        # 日志区：PanedWindow 上下分割
        if HAS_TTKBOOTSTRAP:
            log_pane = ttk.Panedwindow(root, orient=tk.VERTICAL)
        else:
            log_pane = ttk.Panedwindow(root, orient=tk.VERTICAL)
        log_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        off_frame = self._make_labelframe(log_pane, "数据源日志 (official.py / rosbag)")
        log_off = self._make_log(off_frame, "official", height=10)
        log_off.pack(fill=tk.BOTH, expand=True)
        log_pane.add(off_frame, weight=1)

        slam_frame = self._make_labelframe(log_pane, "SLAM 日志")
        log_slam = self._make_log(slam_frame, "all", height=10)
        log_slam.pack(fill=tk.BOTH, expand=True)
        log_pane.add(slam_frame, weight=1)

    # ==================== Tab 2: 车辆控制 ====================

    def _build_vehicle_tab(self, root: ttk.Frame):
        pad = {"padx": 8, "pady": 4}

        # 控制模式切换
        mode_frame = self._make_labelframe(root, "控制模式")
        mode_frame.pack(fill=tk.X, **pad)

        btn_row = ttk.Frame(mode_frame)
        btn_row.pack(fill=tk.X, pady=4)

        self.auto_mode_btn = self._start_btn(btn_row, "  自动驾驶  ",
                                             lambda: self._switch_control_mode("auto"))
        self.auto_mode_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.manual_mode_btn = self._btn(btn_row, "  手动驾驶  ",
                                         lambda: self._switch_control_mode("manual"),
                                         style="warning-outline")
        self.manual_mode_btn.pack(side=tk.LEFT)

        self.mode_display_label = ttk.Label(mode_frame, text="当前模式: 自动驾驶",
                                            font=("", 11, "bold"))
        self.mode_display_label.pack(pady=4)

        # 自动驾驶参数（只读展示）
        auto_frame = self._make_labelframe(root, "自动驾驶参数（只读）")
        auto_frame.pack(fill=tk.X, **pad)
        info_text = "目标速度: 30 km/h  |  前视距离: 15.0 m  |  算法: Pure Pursuit"
        ttk.Label(auto_frame, text=info_text, font=("", 9)).pack(anchor=tk.W)

        # 手动控制子进程
        manual_frame = self._make_labelframe(root, "手动控制输入器")
        manual_frame.pack(fill=tk.X, **pad)

        mr = ttk.Frame(manual_frame)
        mr.pack(fill=tk.X, pady=4)
        ttk.Label(mr, text="脚本：", width=8).pack(side=tk.LEFT)
        ttk.Entry(mr, textvariable=self.manual_script_var, width=30).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._start_btn(mr, "启动", self._start_manual_input).pack(
            side=tk.LEFT, padx=(0, 4))
        self._stop_btn(mr, "停止",
                       lambda: self.stop_proc("manual")).pack(side=tk.LEFT)

        # 按键说明
        keys_frame = self._make_labelframe(root, "按键说明")
        keys_frame.pack(fill=tk.X, **pad)

        keys_data = [
            ("W", "油门 (加速)"), ("S", "刹车 (减速)"),
            ("A", "左转"), ("D", "右转"),
            ("Space", "紧急制动"), ("R", "重置控制"),
            ("Tab", "切换控制模式"), ("ESC/Q", "退出输入器"),
        ]
        for i, (key, desc) in enumerate(keys_data):
            r, c = divmod(i, 2)
            ttk.Label(keys_frame, text=f"  {key}  ", font=("Consolas", 10, "bold"),
                      anchor=tk.CENTER, width=8).grid(
                row=r, column=c * 2, padx=(4, 2), pady=2, sticky=tk.W)
            ttk.Label(keys_frame, text=desc, font=("", 9)).grid(
                row=r, column=c * 2 + 1, padx=(0, 16), pady=2, sticky=tk.W)

        # 日志
        log = self._make_log(root, "manual", height=8)
        log.pack(fill=tk.BOTH, expand=True, **pad)

    # ==================== Tab 3: 数据录制 ====================

    def _build_recording_tab(self, root: ttk.Frame):
        pad = {"padx": 8, "pady": 4}

        # 录制控制
        ctrl_frame = self._make_labelframe(root, "录制控制")
        ctrl_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(ctrl_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="场景名称：", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.scenario_name_var, width=30).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._start_btn(row, "  开始录制  ", self._start_recording).pack(
            side=tk.LEFT, padx=(0, 4))
        self._stop_btn(row, "  停止录制  ", self._stop_recording).pack(
            side=tk.LEFT)

        # 录制状态指示
        status_row = ttk.Frame(ctrl_frame)
        status_row.pack(fill=tk.X, pady=4)
        self.rec_dot_label = ttk.Label(status_row, text="●", font=("", 14),
                                       foreground="#666666")
        self.rec_dot_label.pack(side=tk.LEFT, padx=(0, 4))
        self.rec_status_label = ttk.Label(status_row, text="未录制",
                                          font=("", 10))
        self.rec_status_label.pack(side=tk.LEFT)
        self.rec_timer_label = ttk.Label(status_row, text="", font=("", 10))
        self.rec_timer_label.pack(side=tk.LEFT, padx=10)

        # 可选话题
        opt_frame = self._make_labelframe(root, "可选录制话题")
        opt_frame.pack(fill=tk.X, **pad)
        ttk.Checkbutton(opt_frame, text="compressed image",
                        variable=self.rec_compressed_var).pack(
            side=tk.LEFT, padx=8)
        ttk.Checkbutton(opt_frame, text="tf",
                        variable=self.rec_tf_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(opt_frame, text="tf_static",
                        variable=self.rec_tf_static_var).pack(
            side=tk.LEFT, padx=8)

        # 数据集浏览器
        ds_frame = self._make_labelframe(root, "数据集浏览器")
        ds_frame.pack(fill=tk.BOTH, expand=True, **pad)

        ds_btn_row = ttk.Frame(ds_frame)
        ds_btn_row.pack(fill=tk.X, pady=(0, 4))
        self._btn(ds_btn_row, "刷新", self._refresh_datasets,
                  style="info-outline").pack(side=tk.LEFT)
        self._btn(ds_btn_row, "回放选中", self._playback_selected_dataset,
                  style="success-outline").pack(side=tk.LEFT, padx=4)

        # Treeview
        cols = ("name", "time", "size")
        self.ds_tree = ttk.Treeview(ds_frame, columns=cols, show="headings",
                                    height=6)
        self.ds_tree.heading("name", text="名称")
        self.ds_tree.heading("time", text="创建时间")
        self.ds_tree.heading("size", text="大小")
        self.ds_tree.column("name", width=200)
        self.ds_tree.column("time", width=160)
        self.ds_tree.column("size", width=100)

        ds_scroll = ttk.Scrollbar(ds_frame, orient=tk.VERTICAL,
                                  command=self.ds_tree.yview)
        self.ds_tree.configure(yscrollcommand=ds_scroll.set)
        self.ds_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ds_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 日志
        log = self._make_log(root, "recording", height=6)
        log.pack(fill=tk.BOTH, expand=True, **pad)

        # 初始刷新
        self.root.after(500, self._refresh_datasets)

    # ==================== Tab 4: 点云累积 ====================

    def _build_pcd_tab(self, root: ttk.Frame):
        pad = {"padx": 8, "pady": 4}

        ctrl_frame = self._make_labelframe(root, "pcd.py 配置")
        ctrl_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(ctrl_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="脚本路径：", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.pcd_script_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(row, "浏览",
                  lambda: self._pick_script(self.pcd_script_var)).pack(
            side=tk.LEFT)

        row = ttk.Frame(ctrl_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="参数：", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.pcd_params_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        row = ttk.Frame(ctrl_frame)
        row.pack(fill=tk.X, pady=4)
        self._start_btn(row, "  启动  ", self.start_pcd).pack(side=tk.LEFT)
        self._stop_btn(row, "  停止  ",
                       lambda: self.stop_proc("pcd")).pack(
            side=tk.LEFT, padx=8)

        log = self._make_log(root, "pcd")
        log.pack(fill=tk.BOTH, expand=True, **pad)

    # ==================== Tab 5: SLAM 评估 ====================

    def _build_eval_tab(self, root: ttk.Frame):
        pad = {"padx": 8, "pady": 4}

        # 实时评估
        rt_frame = self._make_labelframe(root, "实时评估")
        rt_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(rt_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="对齐方法：", width=12).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.eval_align_var,
                     values=["translation", "se3", "sim3"],
                     state="readonly", width=15).pack(side=tk.LEFT)

        row = ttk.Frame(rt_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="报告频率(Hz)：", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.eval_hz_var, width=10).pack(
            side=tk.LEFT)

        row = ttk.Frame(rt_frame)
        row.pack(fill=tk.X, pady=4)
        self._start_btn(row, "启动实时评估", self.start_evaluator).pack(
            side=tk.LEFT, padx=(0, 8))
        self._stop_btn(row, "停止评估", self.stop_evaluator).pack(
            side=tk.LEFT)

        # 离线评估（内嵌折叠面板）
        off_frame = self._make_labelframe(root, "离线评估")
        off_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(off_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="GT轨迹文件：", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.offline_gt_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(row, "浏览",
                  lambda: self._browse_file(self.offline_gt_var)).pack(
            side=tk.LEFT)

        row = ttk.Frame(off_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="SLAM估计文件：", width=12).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.offline_est_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(row, "浏览",
                  lambda: self._browse_file(self.offline_est_var)).pack(
            side=tk.LEFT)

        row = ttk.Frame(off_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="对齐方法：", width=12).pack(side=tk.LEFT)
        ttk.Combobox(row, textvariable=self.offline_align_var,
                     values=["translation", "se3", "sim3"],
                     state="readonly", width=15).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(row, text="输出目录：").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.offline_output_var, width=25).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(row, "浏览",
                  lambda: self._browse_dir(self.offline_output_var)).pack(
            side=tk.LEFT)

        row = ttk.Frame(off_frame)
        row.pack(fill=tk.X, pady=4)
        self._start_btn(row, "开始离线评估",
                        self._run_offline_eval).pack(side=tk.LEFT)
        ttk.Checkbutton(
            row, text="按序号匹配（忽略时间戳）",
            variable=self.offline_sequential_var
        ).pack(side=tk.LEFT, padx=(16, 0))

        # 日志
        log = self._make_log(root, "evaluator")
        log.pack(fill=tk.BOTH, expand=True, **pad)

    # ==================== Tab 6: 可视化 ====================

    def _build_viz_tab(self, root: ttk.Frame):
        pad = {"padx": 8, "pady": 4}

        # RViz2
        rviz_frame = self._make_labelframe(root, "RViz2")
        rviz_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(rviz_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Checkbutton(row, text="use_sim_time",
                        variable=self.use_sim_time_var).pack(side=tk.LEFT)
        self._start_btn(row, "启动 RViz2", self.start_rviz).pack(
            side=tk.LEFT, padx=16)

        # CloudCompare
        cc_frame = self._make_labelframe(root, "CloudCompare")
        cc_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(cc_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="命令：", width=8).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.cloudcompare_cmd_var, width=20).pack(
            side=tk.LEFT, padx=(0, 8))
        ttk.Label(row, text="PLY文件：").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.ply_file_var, width=30).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._btn(row, "浏览",
                  lambda: self._browse_file(self.ply_file_var)).pack(
            side=tk.LEFT, padx=(0, 4))
        self._start_btn(row, "打开 PLY", self.open_cloudcompare).pack(
            side=tk.LEFT)

        # 日志
        log = self._make_log(root, "viz")
        log.pack(fill=tk.BOTH, expand=True, **pad)

    # ==================== Tab 7: 地图切换 ====================

    def _build_map_tab(self, root: ttk.Frame):
        pad = {"padx": 8, "pady": 4}

        map_frame = self._make_labelframe(root, "地图切换")
        map_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(map_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="切换方式：", width=10).pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Python API (client.load_world)",
                        variable=self.switch_method_var,
                        value="python_api").pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="ROS Service (/carla/load_world)",
                        variable=self.switch_method_var,
                        value="ros_service").pack(side=tk.LEFT, padx=12)

        row = ttk.Frame(map_frame)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text="Host：", width=10).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.map_host_var, width=18).pack(
            side=tk.LEFT)
        ttk.Label(row, text="  RPC Port：").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.map_port_var, width=8).pack(
            side=tk.LEFT)

        row = ttk.Frame(map_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="地图：", width=10).pack(side=tk.LEFT)
        towns = ["Town01", "Town02", "Town03", "Town04", "Town05",
                 "Town06", "Town07", "Town10HD", "Town10HD_Opt"]
        ttk.Combobox(row, textvariable=self.map_name_var, values=towns,
                     width=20, state="readonly").pack(side=tk.LEFT, padx=(0, 8))
        self._btn(row, "  切换地图  ", self.switch_map,
                  style="warning").pack(side=tk.LEFT)

        log = self._make_log(root, "map")
        log.pack(fill=tk.BOTH, expand=True, **pad)

    # ==================== 日志写入 ====================

    def _log(self, key: str, s: str):
        w = self.log_widgets.get(key)
        if not w:
            return

        def _do():
            w.insert(tk.END, s)
            # 行数限制
            line_count = int(w.index("end-1c").split(".")[0])
            if line_count > self.LOG_MAX_LINES:
                w.delete("1.0", f"{line_count - self.LOG_MAX_LINES}.0")
            w.see(tk.END)

        try:
            w.after_idle(_do)
        except Exception:
            pass

    # ==================== 进程管理 ====================

    def _on_proc_state_change(self, key: str, state: ProcState):
        """进程状态回调（可能从工作线程调用）"""
        try:
            self.root.after_idle(lambda: self.status_bar.update_process(
                key, state))
        except Exception:
            pass

    def _start_proc(self, key: str, cmd: str):
        cwd = expand(self.workdir_var.get().strip()) or None
        ros_setup = expand(self.ros_setup_var.get().strip()) or None
        wrapped = bash_wrap(cmd, ros_setup)
        p = Proc(key, wrapped, cwd, self._log,
                 on_state_change=self._on_proc_state_change)
        self.procs[key] = p
        p.start()
        # 记录启动时间
        if self._app_start_time is None:
            self._app_start_time = time.time()

    def stop_proc(self, key: str):
        p = self.procs.get(key)
        if p:
            threading.Thread(target=p.stop, daemon=True).start()
        if key == "all":
            relay = self.procs.get("all_relay")
            if relay:
                threading.Thread(target=relay.stop, daemon=True).start()
            map_relay = self.procs.get("all_map_relay")
            if map_relay:
                threading.Thread(target=map_relay.stop, daemon=True).start()

    def stop_all(self):
        for k in list(self.procs.keys()):
            self.stop_proc(k)

    # ==================== SLAM 算法切换 ====================

    def _on_slam_algorithm_changed(self, event=None):
        sel = self.slam_algorithm_var.get()
        cur_script = self.all_script_var.get()
        cur_params = self.all_params_var.get()

        # 保存当前参数
        if "kiss_icp" in cur_script:
            self.kiss_icp_params_var.set(cur_params)
        elif cur_script == "lio_sam_official":
            pass
        elif cur_script == "orb_slam3_official":
            pass
        elif "lio_sam" in cur_script:
            self.lio_sam_params_var.set(cur_params)
        else:
            self.lightweight_params_var.set(cur_params)

        pad_kw = dict(fill=tk.X, padx=8, pady=3)

        if sel.startswith("无"):
            self.all_script_var.set("")
            self.all_params_var.set("")
            self.lio_ws_frame.pack_forget()
            self.orb_ws_frame.pack_forget()
            self.slam_opts_frame.pack_forget()
            self.slam_params_frame.pack_forget()
        elif "kiss-icp" in sel.lower() or "kiss_icp" in sel.lower():
            self.all_script_var.set("kiss_icp_node.py")
            self.all_params_var.set(self.kiss_icp_params_var.get())
            self.lio_ws_frame.pack_forget()
            self.orb_ws_frame.pack_forget()
            self.slam_opts_frame.pack(**pad_kw)
            self.slam_params_frame.pack(**pad_kw)
            self.slam_params_entry.configure(state="normal")
        elif "orb-slam3" in sel.lower() or "orb_slam3" in sel.lower():
            self.all_script_var.set("orb_slam3_official")
            self.all_params_var.set(self.orb_slam3_params_var.get())
            self.lio_ws_frame.pack_forget()
            self.orb_ws_frame.pack(**pad_kw)
            self.slam_opts_frame.pack(**pad_kw)
            self.slam_params_frame.pack(**pad_kw)
            self.slam_params_entry.configure(state="readonly")
        elif "lio-sam" in sel.lower() and "official" in sel.lower():
            self.all_script_var.set("lio_sam_official")
            self.all_params_var.set(self.lio_sam_params_var.get())
            self.lio_ws_frame.pack(**pad_kw)
            self.orb_ws_frame.pack_forget()
            self.slam_opts_frame.pack(**pad_kw)
            self.slam_params_frame.pack(**pad_kw)
            self.slam_params_entry.configure(state="readonly")
        else:
            self.all_script_var.set("all.py")
            self.all_params_var.set(self.lightweight_params_var.get())
            self.lio_ws_frame.pack_forget()
            self.orb_ws_frame.pack_forget()
            self.slam_opts_frame.pack(**pad_kw)
            self.slam_params_frame.pack(**pad_kw)
            self.slam_params_entry.configure(state="normal")

    # ==================== 启动命令 ====================

    def _on_data_source_changed(self):
        if self.data_source_var.get() == "playback":
            self.live_frame.pack_forget()
            self.playback_frame.pack(fill=tk.X, pady=2)
        else:
            self.playback_frame.pack_forget()
            self.live_frame.pack(fill=tk.X, pady=2)

    def _pick_bag_path(self):
        d = filedialog.askdirectory(title="选择 rosbag 数据目录")
        if d:
            self.bag_path_var.set(d)

    def start_bridge_and_slam(self):
        if self.data_source_var.get() == "playback":
            self._start_playback()
            # 回放模式：延迟启动SLAM，等待/clock稳定后再启动
            script = self.all_script_var.get().strip()
            if script:
                delay_ms = 3000  # 3秒延迟，确保/clock已开始发布
                self._log("official",
                          f"[状态] 回放模式: {delay_ms/1000:.0f}秒后启动SLAM算法，等待/clock稳定...\n")
                self.root.after(delay_ms, self._delayed_start_slam)
            else:
                self.launch_status_label.configure(text="已启动: 回放 (无SLAM)")
        else:
            self.start_official()
            script = self.all_script_var.get().strip()
            if script:
                self.start_all_py()
                algo = self.slam_algorithm_var.get()
                self.launch_status_label.configure(
                    text=f"已启动: official.py + {algo}")
            else:
                self.launch_status_label.configure(text="已启动: official.py (无SLAM)")
        self._log("official",
                  "[状态] 桥接与SLAM同时启动，GT起点将与SLAM对齐\n")

    def _delayed_start_slam(self):
        """回放模式下延迟启动SLAM，确保/clock已稳定"""
        self.start_all_py()
        algo = self.slam_algorithm_var.get()
        self.launch_status_label.configure(
            text=f"已启动: 回放 + {algo}")
        self._log("official", "[状态] SLAM算法已启动\n")

    def stop_bridge_and_slam(self):
        self.stop_proc("official")
        self.stop_proc("all")
        self.launch_status_label.configure(text="已停止")

    def start_official(self):
        script = expand(self.official_script_var.get().strip())
        cmd = f"python3 {shlex.quote(script)}"
        self._start_proc("official", cmd)

    def _start_playback(self):
        bag_path = expand(self.bag_path_var.get().strip())
        if not bag_path:
            self._log("official", "[error] 未选择回放数据路径\n")
            return
        rate = self.playback_rate_var.get().strip() or "1.0"
        cmd = f"ros2 bag play {shlex.quote(bag_path)} --clock --rate {rate}"
        if self.playback_loop_var.get():
            cmd += " --loop"
        self._start_proc("official", cmd)
        self._log("official",
                  f"[状态] 开始回放: {bag_path} (速率={rate}x)\n")

    def start_all_py(self):
        script = expand(self.all_script_var.get().strip())
        if not script:
            self._log("all", "[状态] 未选择SLAM算法（当前为'无'），跳过启动\n")
            return
        if script == "lio_sam_official":
            self._start_lio_sam_official()
            return
        if script == "orb_slam3_official":
            self._start_orb_slam3_official()
            return
        params = self.all_params_var.get().strip()
        cmd = f"python3 {shlex.quote(script)} {params}"
        self._start_proc("all", cmd)

    def _start_lio_sam_official(self):
        ws = expand(self.lio_sam_ws_var.get().strip()) or expand("~/lio_sam_ws")
        setup_bash = f"{ws}/install/setup.bash"
        cwd = expand(self.workdir_var.get().strip()) or None
        ros_setup = expand(self.ros_setup_var.get().strip()) or None
        source_cmd = f"source {shlex.quote(setup_bash)} >/dev/null 2>&1"
        if ros_setup:
            source_cmd = (f"source {shlex.quote(ros_setup)} >/dev/null 2>&1"
                          f" && {source_cmd}")
        launch_cmd = (f"bash -lc '{source_cmd}"
                      f" && ros2 launch lio_sam run.launch.py'")
        self._log("all", "[info] 启动官方 LIO-SAM (C++ 引擎)...\n")
        p = Proc("all", launch_cmd, cwd, self._log,
                 on_state_change=self._on_proc_state_change)
        self.procs["all"] = p
        p.start()

        def _start_relay():
            relay_cmd = (f"bash -lc '{source_cmd}"
                         f" && ros2 run topic_tools relay"
                         f" /lio_sam/mapping/odometry /odom'")
            self._log("all",
                      "[info] 启动话题转发: /lio_sam/mapping/odometry -> /odom\n")
            p_relay = Proc("all_relay", relay_cmd, cwd, self._log,
                           on_state_change=self._on_proc_state_change)
            self.procs["all_relay"] = p_relay
            p_relay.start()

            # 转发地图话题：/lio_sam/mapping/cloud_registered -> /map
            map_relay_cmd = (f"bash -lc '{source_cmd}"
                             f" && ros2 run topic_tools relay"
                             f" /lio_sam/mapping/cloud_registered /map'")
            self._log("all",
                      "[info] 启动话题转发: /lio_sam/mapping/cloud_registered -> /map\n")
            p_map_relay = Proc("all_map_relay", map_relay_cmd, cwd, self._log,
                               on_state_change=self._on_proc_state_change)
            self.procs["all_map_relay"] = p_map_relay
            p_map_relay.start()
        self.root.after(3000, _start_relay)

    def _start_orb_slam3_official(self):
        ws = expand(self.orb_slam3_ws_var.get().strip()) or expand(
            "~/orb_slam3_ws")
        orb_root = expand(self.orb_slam3_root_var.get().strip()) or expand(
            "~/ORB_SLAM3-master")
        setup_bash = f"{ws}/install/setup.bash"
        vocab = f"{orb_root}/Vocabulary/ORBvoc.txt"
        config = expand(
            self.workdir_var.get().strip()) + "/orb_slam3_carla.yaml"

        cwd = expand(self.workdir_var.get().strip()) or None
        ros_setup = expand(self.ros_setup_var.get().strip()) or None
        source_cmd = f"source {shlex.quote(setup_bash)} >/dev/null 2>&1"
        if ros_setup:
            source_cmd = (f"source {shlex.quote(ros_setup)} >/dev/null 2>&1"
                          f" && {source_cmd}")

        run_cmd = (
            f"ros2 run orbslam3 rgbd "
            f"{shlex.quote(vocab)} {shlex.quote(config)} "
            f"--ros-args -p use_sim_time:=true "
            f"-r camera/rgb:=/camera/front/image_raw "
            f"-r camera/depth:=/camera/front/depth/image_raw"
        )
        launch_cmd = f"bash -lc '{source_cmd} && {run_cmd}'"
        self._log("all", "[info] 启动官方 ORB-SLAM3 (C++ RGB-D 引擎)...\n")
        p = Proc("all", launch_cmd, cwd, self._log,
                 on_state_change=self._on_proc_state_change)
        self.procs["all"] = p
        p.start()

        def _start_relay():
            relay_cmd = (f"bash -lc '{source_cmd}"
                         f" && ros2 run topic_tools relay"
                         f" /orb_slam3/odom /odom'")
            self._log("all",
                      "[info] 启动话题转发: /orb_slam3/odom -> /odom\n")
            p_relay = Proc("all_relay", relay_cmd, cwd, self._log,
                           on_state_change=self._on_proc_state_change)
            self.procs["all_relay"] = p_relay
            p_relay.start()
            self._log("all",
                      "[info] ORB-SLAM3 为稀疏视觉SLAM，不发布点云地图话题，PLY文件将为空\n")
        self.root.after(5000, _start_relay)

    def start_pcd(self):
        script = expand(self.pcd_script_var.get().strip())
        params = self.pcd_params_var.get().strip()
        cmd = f"python3 {shlex.quote(script)} {params}"
        self._start_proc("pcd", cmd)

    def start_rviz(self):
        use_sim = "true" if self.use_sim_time_var.get() else "false"
        cmd = f"rviz2 --ros-args -p use_sim_time:={use_sim}"
        self._start_proc("viz", cmd)

    def open_cloudcompare(self):
        exe = self.cloudcompare_cmd_var.get().strip() or "CloudCompare"
        ply = expand(self.ply_file_var.get().strip()) or expand(
            "~/carla_global.ply")
        cmd = f"{shlex.quote(exe)} {shlex.quote(ply)}"
        self._start_proc("viz", cmd)

    def start_all(self):
        self.start_bridge_and_slam()
        self.root.after(2400, self.start_pcd)

    # ==================== 车辆控制 ====================

    def _switch_control_mode(self, mode: str):
        self.control_mode_var.set(mode)
        self._send_ros2_topic("/vehicle/control_mode", mode)
        if mode == "manual":
            self.mode_display_label.configure(text="当前模式: 手动驾驶")
        else:
            self.mode_display_label.configure(text="当前模式: 自动驾驶")
        self.status_bar.update_control_mode(mode)
        self._log("manual", f"[状态] 切换控制模式: {mode}\n")

    def _start_manual_input(self):
        script = expand(self.manual_script_var.get().strip())
        cmd = f"python3 {shlex.quote(script)}"
        self._start_proc("manual", cmd)

    # ==================== 数据录制 ====================

    def _start_recording(self):
        name = self.scenario_name_var.get().strip()
        if not name:
            name = datetime.now().strftime("scene_%Y%m%d_%H%M%S")
            self.scenario_name_var.set(name)
        self._send_ros2_topic("/recording/command", f"start {name}")
        self.recording_active = True
        self.recording_start_time = time.time()
        self.rec_dot_label.configure(foreground="#ff4444")
        self.rec_status_label.configure(text="录制中")
        self._log("recording", f"[录制] 开始录制: {name}\n")
        self._blink_recording_dot()

    def _stop_recording(self):
        self._send_ros2_topic("/recording/command", "stop")
        self.recording_active = False
        self.recording_start_time = None
        self.rec_dot_label.configure(foreground="#666666")
        self.rec_status_label.configure(text="未录制")
        self.rec_timer_label.configure(text="")
        self.status_bar.update_recording(False)
        self._log("recording", "[录制] 停止录制\n")
        self.root.after(1000, self._refresh_datasets)

    def _blink_recording_dot(self):
        if not self.recording_active:
            return
        self._rec_blink_state = not self._rec_blink_state
        color = "#ff4444" if self._rec_blink_state else "#661111"
        self.rec_dot_label.configure(foreground=color)

        # 更新计时
        if self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            self.rec_timer_label.configure(text=elapsed_str)
            self.status_bar.update_recording(True, elapsed_str)

        self.root.after(500, self._blink_recording_dot)

    def _refresh_datasets(self):
        # 清空
        for item in self.ds_tree.get_children():
            self.ds_tree.delete(item)

        ds_dir = Path(expand(self.workdir_var.get().strip())) / "datasets"
        if not ds_dir.exists():
            return
        try:
            for d in sorted(ds_dir.iterdir(), reverse=True):
                if not d.is_dir():
                    continue
                # 计算大小
                total = sum(f.stat().st_size for f in d.rglob("*")
                            if f.is_file())
                if total > 1024 * 1024:
                    size_str = f"{total / 1024 / 1024:.1f} MB"
                elif total > 1024:
                    size_str = f"{total / 1024:.1f} KB"
                else:
                    size_str = f"{total} B"
                mtime = datetime.fromtimestamp(d.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S")
                self.ds_tree.insert("", tk.END,
                                    values=(d.name, mtime, size_str))
        except Exception as e:
            self._log("recording", f"[error] 扫描数据集失败: {e}\n")

    def _playback_selected_dataset(self):
        sel = self.ds_tree.selection()
        if not sel:
            self._log("recording", "[warn] 请先在列表中选中一个数据集\n")
            return
        name = self.ds_tree.item(sel[0], "values")[0]
        ds_raw = Path(expand(self.workdir_var.get().strip())) / "datasets" / name / "raw_data"
        # ros2 bag record -o 创建的是子目录（含 metadata.yaml），自动定位
        bag_dir = str(ds_raw)
        if ds_raw.is_dir():
            for child in ds_raw.iterdir():
                if child.is_dir() and (child / "metadata.yaml").exists():
                    bag_dir = str(child)
                    break
        self.data_source_var.set("playback")
        self.bag_path_var.set(bag_dir)
        self._on_data_source_changed()
        self.nb.select(0)
        self._log("recording",
                  f"[状态] 已选择数据集 '{name}' 用于回放，请切换到启动页点击启动\n")

    # ==================== SLAM 评估 ====================

    def start_evaluator(self):
        if "evaluator" in self.procs and self.procs["evaluator"].is_running():
            self._log("evaluator", "[warn] 评估器已在运行\n")
            return
        align_method = self.eval_align_var.get()
        report_hz = self.eval_hz_var.get().strip() or "1.0"
        cmd = (
            f"python3 slam_evaluator.py "
            f"--ros-args "
            f"-p align_method:={align_method} "
            f"-p report_hz:={report_hz} "
            f"-p output_dir:=output/slam_evaluation"
        )
        self._start_proc("evaluator", cmd)

    def stop_evaluator(self):
        if "evaluator" in self.procs:
            self.stop_proc("evaluator")
        else:
            self._log("evaluator", "[warn] 评估器未运行\n")

    def _run_offline_eval(self):
        gt_file = expand(self.offline_gt_var.get().strip())
        est_file = expand(self.offline_est_var.get().strip())
        align_method = self.offline_align_var.get()
        output_dir = expand(self.offline_output_var.get().strip())

        if not gt_file or not est_file:
            messagebox.showerror("错误", "请指定轨迹文件")
            return

        cmd = (
            f"python3 offline_evaluator.py "
            f"{shlex.quote(gt_file)} "
            f"{shlex.quote(est_file)} "
            f"--align {align_method} "
            f"--output {shlex.quote(output_dir)}"
        )
        if self.offline_sequential_var.get():
            cmd += " --sequential"
        self._log("evaluator",
                  f"[评估] 开始离线评估: GT={gt_file}, EST={est_file}\n")
        self._start_proc("evaluator", cmd)

    # ==================== 地图切换 ====================

    def switch_map(self):
        method = self.switch_method_var.get()
        host = self.map_host_var.get().strip() or "127.0.0.1"
        port_str = self.map_port_var.get().strip() or "2000"
        try:
            port = int(port_str)
        except Exception:
            self._log("map", f"[error] 端口无效：{port_str}\n")
            return
        town = self.map_name_var.get().strip() or "Town03"

        if method == "python_api":
            def _worker():
                self._log("map",
                          f"[info] 使用 Python API 切换到 {town}"
                          f" @ {host}:{port}...\n")
                try:
                    import importlib
                    carla = importlib.import_module("carla")
                    client = carla.Client(host, port)
                    client.set_timeout(10.0)
                    world = client.load_world(town)
                    self._log("map",
                              f"[ok] 已加载 {town}，同步设置:"
                              f" {world.get_settings().synchronous_mode}\n")
                except ModuleNotFoundError:
                    self._log("map",
                              "[error] 未安装 carla Python 包；"
                              "请在该环境中安装或改用 ROS Service 方式。\n")
                except Exception as e:
                    self._log("map", f"[error] 切换失败：{e}\n")
            threading.Thread(target=_worker, daemon=True).start()
        else:
            cmd = (f"ros2 service call /carla/load_world"
                   f" carla_ros_bridge/srv/LoadWorld"
                   f" \"{{map_name: {town}}}\"")
            self._start_proc("map", cmd)

    # ==================== ROS2 话题发送 ====================

    def _send_ros2_topic(self, topic: str, data: str):
        """通过子进程发送单次 ROS2 话题"""
        ros_setup = expand(self.ros_setup_var.get().strip()) or None
        # 构建完整 bash 命令（不经过 bash_wrap 的单引号包裹，避免与 YAML 单引号冲突）
        pub_cmd = (f"ros2 topic pub --once {topic}"
                   f" std_msgs/msg/String \"{{data: '{data}'}}\"")
        if ros_setup:
            full_cmd = (f"source {shlex.quote(ros_setup)} >/dev/null 2>&1"
                        f" && {pub_cmd}")
        else:
            full_cmd = pub_cmd

        def _worker():
            try:
                # 使用列表形式传参，full_cmd 作为 bash -lc 的单个参数
                # 避免额外的引号层导致 YAML 中的单引号被截断
                result = subprocess.run(
                    ["bash", "-lc", full_cmd],
                    timeout=10,
                    capture_output=True, text=True)
                if result.returncode != 0 and result.stderr:
                    self._log("recording",
                              f"[error] ros2 topic pub 失败: {result.stderr.strip()}\n")
            except subprocess.TimeoutExpired:
                self._log("recording",
                          f"[error] 发送话题超时 {topic}\n")
            except Exception as e:
                self._log("recording",
                          f"[error] 发送话题失败 {topic}: {e}\n")

        threading.Thread(target=_worker, daemon=True).start()

    # ==================== 状态轮询 ====================

    def _poll_status(self):
        """每秒轮询进程状态，更新状态栏"""
        for key in StatusBar.PROC_LABELS:
            p = self.procs.get(key)
            if p:
                if p.is_running():
                    self.status_bar.update_process(key, ProcState.RUNNING)
                elif p.state == ProcState.CRASHED:
                    self.status_bar.update_process(key, ProcState.CRASHED)
                else:
                    self.status_bar.update_process(key, ProcState.STOPPED)
            else:
                self.status_bar.update_process(key, ProcState.STOPPED)

        # 更新运行时间
        if self._app_start_time is not None:
            # 如果任何进程还在运行，显示时间
            any_running = any(
                p.is_running() for p in self.procs.values())
            if any_running:
                elapsed = time.time() - self._app_start_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                self.status_bar.update_runtime(elapsed_str)

        self.root.after(1000, self._poll_status)

    # ==================== 文件/目录选择 ====================

    def _pick_ros_setup(self):
        p = filedialog.askopenfilename(title="选择 ROS 环境脚本 (setup.bash)")
        if p:
            self.ros_setup_var.set(p)

    def _pick_workdir(self):
        p = filedialog.askdirectory(title="选择工作目录")
        if p:
            self.workdir_var.set(p)

    def _pick_script(self, var: tk.StringVar):
        p = filedialog.askopenfilename(
            title="选择脚本",
            filetypes=[("Python", "*.py"), ("全部", "*.*")])
        if p:
            var.set(p)

    def _pick_workdir_to(self, var: tk.StringVar):
        p = filedialog.askdirectory(title="选择目录")
        if p:
            var.set(p)

    def _browse_file(self, var: tk.StringVar):
        filepath = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[("TUM轨迹文件", "*.txt"), ("PLY文件", "*.ply"),
                       ("所有文件", "*.*")])
        if filepath:
            var.set(filepath)

    def _browse_dir(self, var: tk.StringVar):
        dirpath = filedialog.askdirectory(title="选择目录")
        if dirpath:
            var.set(dirpath)

    # ==================== 预设系统 ====================

    def _save_preset(self):
        # 保存前同步当前参数
        cur_script = self.all_script_var.get()
        cur_params = self.all_params_var.get()
        if "kiss_icp" in cur_script:
            self.kiss_icp_params_var.set(cur_params)
        elif "lio_sam" in cur_script:
            self.lio_sam_params_var.set(cur_params)
        else:
            self.lightweight_params_var.set(cur_params)

        d = {
            "ros_setup": self.ros_setup_var.get(),
            "workdir": self.workdir_var.get(),
            "official_script": self.official_script_var.get(),
            "slam_algorithm": self.slam_algorithm_var.get(),
            "all_script": self.all_script_var.get(),
            "lightweight_params": self.lightweight_params_var.get(),
            "lio_sam_params": self.lio_sam_params_var.get(),
            "lio_sam_ws": self.lio_sam_ws_var.get(),
            "kiss_icp_params": self.kiss_icp_params_var.get(),
            "orb_slam3_ws": self.orb_slam3_ws_var.get(),
            "orb_slam3_root": self.orb_slam3_root_var.get(),
            "orb_slam3_params": self.orb_slam3_params_var.get(),
            "pcd_script": self.pcd_script_var.get(),
            "pcd_params": self.pcd_params_var.get(),
            "use_sim_time": self.use_sim_time_var.get(),
            "cloudcompare": self.cloudcompare_cmd_var.get(),
            "ply_file": self.ply_file_var.get(),
            # 新增字段
            "control_mode": self.control_mode_var.get(),
            "manual_input_script": self.manual_script_var.get(),
            "recording_base_dir": "datasets/",
            # 评估参数
            "eval_align": self.eval_align_var.get(),
            "eval_hz": self.eval_hz_var.get(),
            "offline_gt": self.offline_gt_var.get(),
            "offline_est": self.offline_est_var.get(),
            "offline_align": self.offline_align_var.get(),
            "offline_output": self.offline_output_var.get(),
            # 地图
            "map_host": self.map_host_var.get(),
            "map_port": self.map_port_var.get(),
            "map_name": self.map_name_var.get(),
            "switch_method": self.switch_method_var.get(),
            # 回放
            "data_source": self.data_source_var.get(),
            "bag_path": self.bag_path_var.get(),
            "playback_rate": self.playback_rate_var.get(),
            "playback_loop": self.playback_loop_var.get(),
        }
        p = filedialog.asksaveasfilename(
            title="保存预设", defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("保存成功", f"已保存到：{p}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _load_preset(self):
        p = filedialog.askopenfilename(
            title="加载预设",
            filetypes=[("JSON", "*.json"), ("全部", "*.*")])
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)

            self.ros_setup_var.set(d.get("ros_setup", ""))
            self.workdir_var.set(d.get("workdir", str(Path.cwd())))
            self.official_script_var.set(
                d.get("official_script", "official.py"))
            self.slam_algorithm_var.set(
                d.get("slam_algorithm", "all.py (Lightweight)"))
            self.all_script_var.set(d.get("all_script", "all.py"))

            # 恢复各算法参数
            if "lightweight_params" in d:
                self.lightweight_params_var.set(d["lightweight_params"])
            elif "all_params" in d:
                self.lightweight_params_var.set(d["all_params"])
            if "lio_sam_params" in d:
                self.lio_sam_params_var.set(d["lio_sam_params"])
            if "lio_sam_ws" in d:
                self.lio_sam_ws_var.set(d["lio_sam_ws"])
            if "kiss_icp_params" in d:
                self.kiss_icp_params_var.set(d["kiss_icp_params"])
            if "orb_slam3_ws" in d:
                self.orb_slam3_ws_var.set(d["orb_slam3_ws"])
            if "orb_slam3_root" in d:
                self.orb_slam3_root_var.set(d["orb_slam3_root"])
            if "orb_slam3_params" in d:
                self.orb_slam3_params_var.set(d["orb_slam3_params"])

            # 根据选中算法设置当前显示参数
            sel = self.slam_algorithm_var.get()
            if "kiss-icp" in sel.lower() or "kiss_icp" in sel.lower():
                self.all_params_var.set(self.kiss_icp_params_var.get())
            elif "orb-slam3" in sel.lower() or "orb_slam3" in sel.lower():
                self.all_script_var.set("orb_slam3_official")
                self.all_params_var.set(self.orb_slam3_params_var.get())
            elif "lio-sam" in sel.lower() and "official" in sel.lower():
                self.all_script_var.set("lio_sam_official")
                self.all_params_var.set(self.lio_sam_params_var.get())
            elif "lio_sam" in sel.lower() or "lio-sam" in sel.lower():
                self.all_params_var.set(self.lio_sam_params_var.get())
            else:
                self.all_params_var.set(self.lightweight_params_var.get())

            self.pcd_script_var.set(d.get("pcd_script", "pcd.py"))
            self.pcd_params_var.set(
                d.get("pcd_params", self.pcd_params_var.get()))
            self.use_sim_time_var.set(bool(d.get("use_sim_time", True)))
            self.cloudcompare_cmd_var.set(
                d.get("cloudcompare", "CloudCompare"))
            self.ply_file_var.set(
                d.get("ply_file", "output/carla_global.ply"))

            # 新增字段（向后兼容）
            self.control_mode_var.set(d.get("control_mode", "auto"))
            self.manual_script_var.set(
                d.get("manual_input_script", "manual_control_input.py"))

            # 评估参数
            self.eval_align_var.set(d.get("eval_align", "sim3"))
            self.eval_hz_var.set(d.get("eval_hz", "1.0"))
            self.offline_gt_var.set(
                d.get("offline_gt",
                      "output/carla_ground_truth_trajectory.txt"))
            self.offline_est_var.set(
                d.get("offline_est", "output/carla_odom_tum.txt"))
            self.offline_align_var.set(d.get("offline_align", "sim3"))
            self.offline_output_var.set(
                d.get("offline_output", "output/slam_evaluation_offline"))

            # 地图
            self.map_host_var.set(d.get("map_host", "127.0.0.1"))
            self.map_port_var.set(d.get("map_port", "2000"))
            self.map_name_var.set(d.get("map_name", "Town03"))
            self.switch_method_var.set(
                d.get("switch_method", "python_api"))

            # 回放
            self.data_source_var.set(d.get("data_source", "live"))
            self.bag_path_var.set(d.get("bag_path", ""))
            self.playback_rate_var.set(d.get("playback_rate", "1.0"))
            self.playback_loop_var.set(bool(d.get("playback_loop", False)))
            self._on_data_source_changed()

            messagebox.showinfo("加载成功", f"已加载：{p}")
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    # ==================== 退出清理 ====================

    def _on_close(self):
        """退出时清理所有进程"""
        for key, p in self.procs.items():
            if p.is_running():
                try:
                    os.killpg(os.getpgid(p.p.pid), signal.SIGTERM)
                except Exception:
                    try:
                        p.p.terminate()
                    except Exception:
                        pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ==================== 入口 ====================

if __name__ == "__main__":
    app = App()
    app.run()
