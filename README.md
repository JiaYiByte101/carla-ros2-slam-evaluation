# CARLA-ROS2 多算法 SLAM 测试与评估平台

基于 CARLA 仿真器与 ROS 2 的 SLAM 测试平台，提供传感器数据发布、算法接入、数据录制、轨迹评估与可视化功能。

## 主要功能

- 接入 Lightweight ICP、KISS-ICP、LIO-SAM 和 ORB-SLAM3。
- 支持 CARLA 实时仿真与 rosbag 数据回放。
- 提供在线、离线轨迹评估，计算 ATE、RPE 等指标。
- 通过图形界面进行车辆控制、录制、算法启动及结果查看。
- 记录点云地图、SLAM 轨迹与 Ground Truth 轨迹。

## 运行环境

| 组件 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 |
| ROS 2 | Humble |
| CARLA | 0.9.15 |
| Python | 3.10 |
| Open3D | ≥ 0.17 |

CARLA 服务端可独立部署；本平台的 ROS 2 节点按上述 Ubuntu 环境运行。

## 安装与启动

完整环境搭建步骤见 [安装指南](Install.md)，平台架构与操作说明见 [使用指南](Guide.md)。

1. 按安装指南准备 ROS 2、CARLA Python API 及所需算法环境。
2. 启动 CARLA 服务端。
3. 根据部署情况修改 `official.py` 中的 `HOST` 和 `PORT`。
4. 在已加载 ROS 2 和相关算法工作空间环境的终端中，进入本项目目录并执行：

```bash
source /opt/ros/humble/setup.bash
python3 ui.py
```

在图形界面中选择所需算法并启动对应流程。LIO-SAM、ORB-SLAM3 等外部算法需要单独安装与配置。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| `ui.py` | 图形界面入口 |
| `official.py` | CARLA 连接与 ROS 2 传感器数据发布 |
| `vehicle_control/` | 自动与手动车辆控制 |
| `data_manager/` | 数据录制与管理 |
| `slam_algorithms/` | SLAM 算法接口及实现 |
| `evaluation/` | 轨迹对齐、指标计算与可视化 |
| `slam_evaluator.py` | 在线评估 |
| `offline_evaluator.py` | 离线评估 |
| `kiss_icp_node.py` | KISS-ICP 节点 |
| `lio_sam_params_carla.yaml` | LIO-SAM 配置 |
| `orb_slam3_carla.yaml` | ORB-SLAM3 配置 |

## 数据与输出

采集数据、运行结果、Python 缓存及本地环境文件不纳入版本控制。

`datasets/` 和 `output/` 默认不随仓库上传；若运行时需要且尚不存在，可在项目目录执行：

```bash
mkdir -p datasets output/slam_evaluation output/slam_evaluation_offline
```

## 第三方项目

本平台使用或对接 CARLA、ROS 2、Open3D、KISS-ICP、LIO-SAM 和 ORB-SLAM3 等项目。第三方组件的使用与分发须遵循各自许可证。
