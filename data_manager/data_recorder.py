"""
数据录制模块
自动录制ROS2话题数据并生成结构化数据集
"""

import os
import time
import yaml
import subprocess
import signal
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import threading


class DataRecorder:
    """数据录制器类"""

    def __init__(self, base_dir: str = "datasets", auto_create_dirs: bool = True):
        """
        初始化数据录制器

        Args:
            base_dir: 数据集基础目录
            auto_create_dirs: 是否自动创建目录结构
        """
        self.base_dir = Path(base_dir)
        self.auto_create_dirs = auto_create_dirs

        # 录制状态
        self.is_recording = False
        self.rosbag_process = None
        self.current_dataset_dir = None
        self.recording_start_time = None

        # 默认录制话题
        self.default_topics = [
            "/clock",
            "/lidar/points",
            "/imu/data",
            "/camera/front/image_raw",
            "/camera/front/camera_info",
            "/camera/front/depth/image_raw",
            "/ground_truth/odom",
            "/ground_truth/path"
        ]

        # 可选话题（用户可选择是否录制）
        self.optional_topics = [
            "/camera/front/image_compressed",
            "/tf",
            "/tf_static"
        ]

        if auto_create_dirs:
            self.base_dir.mkdir(exist_ok=True)

    def create_dataset_directory(self, scenario_name: Optional[str] = None) -> Path:
        """
        创建新的数据集目录

        Args:
            scenario_name: 场景名称，如果为空则自动生成

        Returns:
            创建的数据集目录路径
        """
        if scenario_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario_name = f"scenario_{timestamp}"

        dataset_dir = self.base_dir / scenario_name

        # 创建目录结构
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "raw_data").mkdir(exist_ok=True)
        (dataset_dir / "slam_results").mkdir(exist_ok=True)
        (dataset_dir / "slam_results" / "maps").mkdir(exist_ok=True)
        (dataset_dir / "evaluation").mkdir(exist_ok=True)

        return dataset_dir

    def generate_config_file(self, dataset_dir: Path, config_data: Dict):
        """
        生成数据集配置文件

        Args:
            dataset_dir: 数据集目录
            config_data: 配置数据
        """
        config_file = dataset_dir / "config.yaml"

        # 默认配置
        default_config = {
            "dataset_info": {
                "name": dataset_dir.name,
                "created_time": datetime.now().isoformat(),
                "description": "CARLA SLAM数据集",
                "carla_map": "Town01",
                "vehicle_model": "vehicle.tesla.model3"
            },
            "sensors": {
                "lidar": {
                    "channels": 32,
                    "range": 80,
                    "rotation_frequency": 20,
                    "topic": "/lidar/points"
                },
                "imu": {
                    "topic": "/imu/data"
                },
                "camera_rgb": {
                    "resolution": "1280x720",
                    "fov": 90,
                    "topic": "/camera/front/image_raw"
                },
                "camera_depth": {
                    "resolution": "1280x720",
                    "fov": 90,
                    "topic": "/camera/front/depth/image_raw"
                }
            },
            "ground_truth": {
                "odom_topic": "/ground_truth/odom",
                "path_topic": "/ground_truth/path",
                "trajectory_file": "ground_truth_trajectory.txt"
            },
            "recording": {
                "duration_seconds": 0,  # 会在录制结束时更新
                "topics_recorded": self.default_topics,
                "bag_file": "raw_data.mcap"
            }
        }

        # 合并用户配置
        default_config.update(config_data)

        # 保存配置文件
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)

        print(f"      配置文件已创建: {config_file}")

    def start_recording(self,
                       scenario_name: Optional[str] = None,
                       topics: Optional[List[str]] = None,
                       include_optional: bool = False,
                       config_data: Dict = None) -> bool:
        """
        开始录制数据

        Args:
            scenario_name: 场景名称
            topics: 自定义录制话题列表
            include_optional: 是否包含可选话题
            config_data: 额外的配置数据

        Returns:
            是否成功开始录制
        """
        if self.is_recording:
            print("      错误: 已在录制中")
            return False

        try:
            # 创建数据集目录
            self.current_dataset_dir = self.create_dataset_directory(scenario_name)
            print(f"      数据集目录: {self.current_dataset_dir}")

            # 确定录制话题
            if topics is None:
                topics = self.default_topics.copy()
                if include_optional:
                    topics.extend(self.optional_topics)

            # 生成配置文件
            if config_data is None:
                config_data = {}
            config_data.setdefault("recording", {})["topics_recorded"] = topics
            self.generate_config_file(self.current_dataset_dir, config_data)

            # 构建rosbag命令
            bag_file = self.current_dataset_dir / "raw_data" / "raw_data.mcap"
            cmd = ["ros2", "bag", "record", "-o", str(bag_file)]
            cmd.extend(topics)

            print(f"      开始录制话题: {topics}")
            print(f"      输出文件: {bag_file}")

            # 启动rosbag录制
            # 注意：不能用PIPE捕获stdout/stderr，否则无人读取会导致管道缓冲区满
            # rosbag进程阻塞 → SIGINT无法响应 → 被SIGKILL强杀 → 数据丢失
            # 使用os.setsid使rosbag在独立进程组中，确保SIGINT能传递到所有子进程
            self.rosbag_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )

            self.is_recording = True
            self.recording_start_time = time.time()

            # 启动监控线程
            self._start_monitoring_thread()

            print("      ✅ 数据录制已开始")
            return True

        except Exception as e:
            print(f"      ❌ 录制启动失败: {e}")
            return False

    def stop_recording(self) -> bool:
        """
        停止录制数据

        Returns:
            是否成功停止录制
        """
        if not self.is_recording:
            print("      警告: 当前未在录制")
            return False

        try:
            # 发送SIGINT信号给rosbag进程组（模拟Ctrl+C，触发正常关闭和数据刷盘）
            if self.rosbag_process:
                try:
                    os.killpg(os.getpgid(self.rosbag_process.pid), signal.SIGINT)
                except ProcessLookupError:
                    pass  # 进程已结束

                # 等待进程结束
                try:
                    self.rosbag_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("      警告: rosbag进程未能正常结束，强制终止")
                    self.rosbag_process.kill()

            self.is_recording = False
            recording_duration = time.time() - self.recording_start_time

            # 更新配置文件中的录制时长
            if self.current_dataset_dir:
                self._update_recording_duration(recording_duration)

            print(f"      ✅ 数据录制已停止 (时长: {recording_duration:.1f}秒)")

            # 显示录制统计信息
            self._show_recording_statistics()

            return True

        except Exception as e:
            print(f"      ❌ 停止录制失败: {e}")
            return False

    def _start_monitoring_thread(self):
        """启动录制监控线程"""
        def monitor():
            while self.is_recording and self.rosbag_process:
                if self.rosbag_process.poll() is not None:
                    # 进程已结束
                    self.is_recording = False
                    print("      ⚠️  rosbag进程意外结束")
                    break
                time.sleep(1)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def _update_recording_duration(self, duration: float):
        """更新配置文件中的录制时长"""
        try:
            config_file = self.current_dataset_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                config["recording"]["duration_seconds"] = duration

                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        except Exception as e:
            print(f"      警告: 无法更新配置文件: {e}")

    def _show_recording_statistics(self):
        """显示录制统计信息"""
        if not self.current_dataset_dir:
            return

        try:
            # 检查生成的bag文件
            bag_dir = self.current_dataset_dir / "raw_data" / "raw_data.mcap"
            if bag_dir.exists():
                # 获取文件大小
                total_size = 0
                for file_path in bag_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

                print(f"      📊 录制统计:")
                print(f"         数据集目录: {self.current_dataset_dir.name}")
                print(f"         文件大小: {total_size / 1024 / 1024:.1f} MB")
                print(f"         存储位置: {self.current_dataset_dir}")

        except Exception as e:
            print(f"      警告: 无法获取统计信息: {e}")

    def get_recording_status(self) -> Dict:
        """
        获取录制状态信息

        Returns:
            录制状态字典
        """
        status = {
            "is_recording": self.is_recording,
            "current_dataset": str(self.current_dataset_dir) if self.current_dataset_dir else None,
            "recording_duration": 0
        }

        if self.is_recording and self.recording_start_time:
            status["recording_duration"] = time.time() - self.recording_start_time

        return status

    def list_datasets(self) -> List[str]:
        """
        列出所有数据集

        Returns:
            数据集名称列表
        """
        if not self.base_dir.exists():
            return []

        datasets = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "config.yaml").exists():
                datasets.append(item.name)

        return sorted(datasets)

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """
        获取数据集信息

        Args:
            dataset_name: 数据集名称

        Returns:
            数据集信息字典，如果不存在则返回None
        """
        dataset_dir = self.base_dir / dataset_name
        config_file = dataset_dir / "config.yaml"

        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"读取数据集配置失败: {e}")
            return None