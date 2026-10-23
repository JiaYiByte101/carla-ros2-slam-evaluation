# CARLA-ROS2 多算法 SLAM 测试与评估平台

## 1. 项目概述

本项目基于 CARLA 自动驾驶仿真器和 ROS2（Humble），构建了一套完整的 SLAM 算法测试与评估平台。平台提供统一的传感器数据发布、多种 SLAM 算法接入、自动化轨迹评估和可视化功能，用户可通过图形界面一站式完成"仿真驾驶 → SLAM 建图 → 轨迹评估"的全流程。

### 1.1 核心特性

- **多算法支持**：集成 4 种 SLAM 算法（Lightweight ICP、KISS-ICP、LIO-SAM、ORB-SLAM3），支持用户扩展自定义算法
- **双数据源**：支持实时 CARLA 仿真和 rosbag 回放两种数据输入模式
- **自动化评估**：提供在线/离线两种评估方式，自动计算 ATE、RPE 等标准 SLAM 指标
- **图形化控制**：7 个功能标签页覆盖启动、控制、录制、建图、评估、可视化、地图切换
- **数据管理**：所有输出按时间戳独立存储，支持历史记录查看

### 1.2 运行环境

| 组件      | 版本                   |
| ------- | -------------------- |
| 操作系统    | Ubuntu 22.04（VM 或原生） |
| CARLA   | 0.9.15               |
| ROS2    | Humble               |
| Python  | 3.10                 |
| Open3D  | ≥ 0.17               |
| DDS 中间件 | FastDDS（ROS2 默认）     |

***

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                        ui.py (GUI)                            │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌────┐ ┌────┐ │
│  │ 启动 │ │ 车辆 │ │ 录制 │ │ 点云 │ │ 评估 │ │可视│ │地图│ │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬─┘ └──┬─┘ │
└─────┼────────┼────────┼────────┼────────┼────────┼───────┼───┘
      │        │        │        │        │        │       │
      ▼        ▼        ▼        ▼        ▼        ▼       ▼
┌──────────┐ ┌─────────┐ ┌──────┐ ┌─────┐ ┌────────────┐ ┌──────┐
│official.py│ │manual_  │ │data_ │ │pcd  │ │slam_       │ │rviz2 │
│(CARLA桥接)│ │control  │ │recorder│ │.py  │ │evaluator.py│ │      │
└─────┬─────┘ └─────────┘ └──────┘ └──┬──┘ └──────┬─────┘ └──────┘
      │                                │           │
      │  ROS2 Topics                   │           │
      ▼                                ▼           ▼
┌─────────────────────────────────────────────────────────┐
│                    ROS2 话题总线                          │
│  /lidar/points  /imu/data  /camera/front/*  /clock      │
│  /ground_truth/odom  /odom  /map  /odom_path            │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌───────────┐   ┌───────────┐
    │ all.py   │   │ KISS-ICP  │   │ LIO-SAM   │
    │(轻量ICP) │   │(kiss_icp_ │   │(Official  │
    │          │   │ node.py)  │   │ C++ 引擎) │
    └──────────┘   └───────────┘   └───────────┘
                                   ┌───────────┐
                                   │ ORB-SLAM3 │
                                   │(Official  │
                                   │ C++ RGB-D)│
                                   └───────────┘
```

### 2.1 数据流

1. **数据源层**：`official.py` 连接 CARLA 仿真器，以同步模式（20Hz）发布 LiDAR 点云、IMU、RGB/深度相机图像、Ground Truth 里程计和 `/clock` 时钟。回放模式下由 `ros2 bag play` 替代 `official.py`，发布相同的话题。
2. **SLAM 算法层**：各 SLAM 节点订阅传感器话题，输出 `/odom`（里程计）和 `/map`（点云地图）。对于官方 C++ 引擎（LIO-SAM、ORB-SLAM3），通过 `topic_tools relay` 将其专有话题（如 `/lio_sam/mapping/odometry`）转发到标准话题。
3. **数据采集层**：`pcd.py` 订阅 `/map`、`/odom`、`/ground_truth/odom`，累积点云并保存 PLY 文件，同时记录 SLAM 和 GT 轨迹为 TUM 格式。
4. **评估层**：`slam_evaluator.py`（在线）或 `offline_evaluator.py`（离线）对轨迹进行 Sim3/SE3 对齐后计算 ATE/RPE 指标，生成可视化图表。

### 2.2 时间戳与同步机制（关键设计）

本平台的时间同步不是“传感器到了就立即发布”，而是采用**以 CARLA 同步帧为唯一时间源**的统一策略。其核心目的是保证 LiDAR、相机、Ground Truth 与 SLAM 所看到的是同一仿真时刻的数据，而不是“内容属于当前帧、时间戳却落在上一帧”的混乱状态。

**核心规则**：

1. `official.py` 在主循环中先执行 `world.tick()` 推进 CARLA 一帧；
2. 读取 `snapshot.timestamp.elapsed_seconds` 作为当前仿真时间；
3. 先发布 `/clock`，并将其保存为当前统一时间戳；
4. 再发布本帧 LiDAR、RGB、Depth、GT 等消息，所有这些消息共享同一个 ROS 时间戳；
5. SLAM 节点只消费这些已带统一时间戳的数据，不自行定义“当前时刻”。

**LiDAR / 相机需要先缓存再发布**：

- CARLA 传感器回调触发时，主循环中的“当前帧时间”可能尚未更新；
- 如果在回调里立刻发布，会出现消息内容来自当前帧、但 Header 时间戳仍是上一帧的风险；
- 因此 `official.py` 对 LiDAR / 相机采用“先暂存到 pending，再在 `publish_clock()` 之后统一发布”的策略。

**IMU 的特殊处理**：

- IMU 更强调时间戳严格递增；
- 为避免后端预积分或 GTSAM 在 `dt<=0` 时崩溃，桥接层会对 IMU 时间戳做单调递增保护；
- 这意味着 IMU 时间设计的首要目标是“合法且可积分”，而 LiDAR / 相机 / GT 的首要目标是“同帧对齐”。

若时间戳体系被破坏，即使轨迹在 RViz2 中看起来重合，ATE / RPE 也可能显著偏高。

***

## 3. 文件结构

```
CARLA/
├── official.py              # CARLA-ROS2 桥接节点（数据源）
├── ui.py                    # 图形化控制界面（主入口）
├── all.py                   # 轻量级 LiDAR-IMU SLAM（GICP + 闭环）
├── kiss_icp_node.py         # KISS-ICP ROS2 节点包装
├── pcd.py                   # 点云累积 + 轨迹记录
├── slam_evaluator.py        # 在线 SLAM 评估器
├── offline_evaluator.py     # 离线 SLAM 评估器
├── manual_control_input.py  # 手动键盘控制节点
├── recording_control.py     # 数据录制控制
│
├── evaluation/              # 评估算法库
│   ├── alignment.py         #   Sim3/SE3 轨迹对齐（Umeyama 算法）
│   ├── metrics.py           #   ATE/RPE 计算 + 时间戳关联
│   └── visualizer.py        #   轨迹对比图 + 误差分布图
│
├── vehicle_control/         # 车辆控制模块
│   ├── base_controller.py   #   控制器基类
│   ├── auto_controller.py   #   自动驾驶（Pure Pursuit + PID）
│   └── manual_controller.py #   手动控制
│
├── data_manager/            # 数据管理模块
│   └── data_recorder.py     #   rosbag 录制器
│
├── slam_algorithms/         # SLAM 算法框架
│   ├── base_slam.py         #   SLAM 基类（接口定义）
│   └── kiss_icp.py          #   KISS-ICP 算法封装
│
├── orb_slam3_carla.yaml     # ORB-SLAM3 参数配置
├── lio_sam_params_carla.yaml# LIO-SAM 参数配置
│
├── datasets/                # 录制数据集（自动生成）
│   └── scenario_YYYYMMDD_HHMMSS/
│       ├── config.yaml
│       └── raw_data/
│           └── raw_data.mcap
│
└── output/                  # 运行输出（自动生成）
    ├── run_YYYYMMDD_HHMMSS/ #   pcd.py 输出
    │   ├── carla_global.ply
    │   ├── carla_odom_tum.txt
    │   └── carla_ground_truth_trajectory.txt
    ├── slam_evaluation/     #   在线评估输出
    │   └── run_YYYYMMDD_HHMMSS/
    └── slam_evaluation_offline/ # 离线评估输出
        └── run_YYYYMMDD_HHMMSS/
            ├── evaluation_report.json
            ├── trajectory_comparison.png
            ├── error_distribution.png
            └── rpe_analysis.png
```

***

## 4. 传感器配置

### 4.1 LiDAR

| 参数   | 值             | 说明         |
| ---- | ------------- | ---------- |
| 线数   | 32            | 足够 SLAM 使用 |
| 点频   | 400,000 pts/s | 适中密度       |
| 旋转频率 | 20 Hz         | 与主循环同步     |
| 最大距离 | 80 m          | <br />     |
| 视场角  | +15° \~ -25°  | <br />     |
| 安装位置 | 车顶 z=2.0m     | <br />     |

### 4.2 相机（仅 ORB-SLAM3 使用）

| 参数   | 值                     |
| ---- | --------------------- |
| 分辨率  | 1280 × 720            |
| 视场角  | 90°                   |
| 帧率   | 10 Hz                 |
| 类型   | RGB + Depth（RGB-D 模式） |
| 安装位置 | 前方 x=0.8m, z=1.7m     |

> 注：相机帧率设为 10Hz 而非 20Hz，以匹配 ORB-SLAM3 在虚拟机上的处理能力，避免帧率不匹配导致的轨迹漂移。

### 4.3 IMU

| 参数 | 值                   |
| -- | ------------------- |
| 帧率 | 与仿真同步（200Hz）        |
| 数据 | 三轴加速度 + 三轴角速度       |
| 用途 | 为 LiDAR SLAM 提供旋转预测 |

***

## 5. SLAM 算法说明

### 5.1 算法对比

| 算法                   | 类型        | 传感器         | 实现方式          | 回放支持  |
| -------------------- | --------- | ----------- | ------------- | ----- |
| Lightweight (all.py) | LiDAR-IMU | 点云 + 陀螺仪    | 自研 Python     | 仅实时模式 |
| KISS-ICP             | LiDAR     | 点云          | Python 封装     | 实时/回放 |
| LIO-SAM              | LiDAR-IMU | 点云 + IMU    | 官方 C++ ROS2 包 | 实时/回放 |
| ORB-SLAM3            | 视觉        | RGB + Depth | 官方 C++ ROS2 包 | 实时/回放 |

### 5.2 Lightweight SLAM (all.py)

自研的轻量级 LiDAR-IMU SLAM，主要用于提供"用户接入自定义 SLAM 算法"的参考范式。

**核心流程**：

1. 点云预处理：距离/高度滤波 + RANSAC 地面移除
2. 运动预测：IMU 陀螺仪旋转 + 时间比例等速平移
3. 帧到子图 Generalized ICP 配准
4. 滑动窗口关键帧管理
5. 轻量级闭环检测与校正

> 注：依赖帧间连续性，不支持 rosbag 回放模式（回放时丢帧严重导致 ICP 发散）。

### 5.3 话题转发

官方 C++ SLAM 引擎使用专有话题名，平台通过 `topic_tools relay` 自动转发到标准话题：

| SLAM 算法   | 原始话题                                | 转发目标    |
| --------- | ----------------------------------- | ------- |
| LIO-SAM   | `/lio_sam/mapping/odometry`         | `/odom` |
| LIO-SAM   | `/lio_sam/mapping/cloud_registered` | `/map`  |
| ORB-SLAM3 | `/orb_slam3/odom`                   | `/odom` |

> 注：ORB-SLAM3 为稀疏视觉 SLAM，其 ROS2 wrapper 不发布点云地图话题，因此无法生成 PLY 文件。

### 5.4 Lightweight SLAM 的时间逻辑

`all.py` / `lightweight_slam.py` 中，LiDAR 帧被视为 SLAM 的主时间锚点：

- LiDAR 回调读取 `PointCloud2.header.stamp` 作为当前 SLAM 帧时间；
- IMU 回调只负责把高频陀螺仪数据缓存到缓冲区；
- 当一帧 LiDAR 到达时，节点把该帧时间之前积累的 IMU 序列一并送入算法；

这意味着平台的默认假设是：

- **LiDAR 决定关键帧时刻**；
- **IMU 提供关键帧之间的连续运动补偿**；
- **SLAM 不重新定义时间，只继承桥接层给出的统一时间戳**。

### 5.5 控制模块设计

平台支持自动驾驶与手动驾驶两套控制器，二者都继承自 `BaseController`，对外暴露统一接口，因此 `official.py` 的主循环只需在每帧调用当前控制器的 `update()`，再把结果传给 `vehicle.apply_control()`。

#### 自动驾驶：`auto_controller.py`

自动控制器的目标是**稳定地产生可用于 SLAM 采集的连续行驶轨迹**。其核心逻辑为：

1. 通过 `world.get_map().get_waypoint()` 将当前车辆位置投影到最近可行驶车道路点；
2. 基于 Pure Pursuit 思想，在前方 `lookahead_m` 处选择目标路点；
3. 根据目标路点相对车头的几何关系计算曲率，并映射为方向盘转角；
4. 使用简单比例速度控制维持目标巡航速度（默认 12 km/h）；

该控制器还包含卡住检测逻辑：若车辆长时间几乎不动，可在上层触发重新生成或重置，避免采集到大量静止帧。

#### 手动驾驶：`manual_controller.py` + `manual_control_input.py`

手动驾驶采用两层的设计：

- `manual_control_input.py`：监听键盘，把按键事件转换为 ROS2 字符串命令；
- `manual_controller.py`：维护当前油门、刹车、转向状态，并在每帧生成 `VehicleControl`。

默认键位：

- `W`：加油门
- `S`：刹车
- `A/D`：左/右转
- `Space`：紧急制动
- `R`：控制复位
- `Tab`：自动 / 手动模式切换

这种设计的优点是：主循环不直接依赖键盘输入，未来可很方便替换为手柄、远程控制或其它输入源。但不支持虚拟机-客户机双端分离架构的用户使用。

### 5.6 录制模块设计

数据录制由三层组成：

1. `recording_control.py`：命令行前端，向 `/recording/command` 发送 `start`、`stop`、`status` 等指令；
2. `official.py`：接收录制命令，并把当前场景、车辆、传感器参数补充到录制配置中；
3. `data_recorder.py`：真正管理 `ros2 bag record` 进程。

***

## 6. 评估体系

### 6.1 评估指标

| 指标                  | 全称                                | 含义                           |
| ------------------- | --------------------------------- | ---------------------------- |
| ATE RMSE            | Absolute Trajectory Error         | 轨迹绝对位置误差的均方根，反映全局精度          |
| ATE Mean/Median/Std | <br />                            | 误差的统计分布                      |
| RPE Trans           | Relative Pose Error (Translation) | 相邻帧间平移误差，反映局部一致性             |
| RPE Rot             | Relative Pose Error (Rotation)    | 相邻帧间旋转误差（度）                  |
| Path Ratio          | <br />                            | GT 路径长度 / SLAM 路径长度，理想值为 1.0 |

### 6.2 轨迹对齐

评估前需要对 SLAM 轨迹和 GT 轨迹进行对齐，消除坐标系差异：

- **Sim3 对齐**（默认）：Umeyama 算法，校正平移、旋转和尺度。适用于单目视觉 SLAM 或存在尺度漂移的场景。
- **SE3 对齐**：仅校正平移和旋转，保留原始尺度。适用于有绝对尺度的 LiDAR SLAM。

### 6.3 在线评估 vs 离线评估

| <br /> | 在线评估                                | 离线评估            |
| ------ | ----------------------------------- | --------------- |
| 触发方式   | 实时订阅 `/odom` 和 `/ground_truth/odom` | 读取 TUM 格式轨迹文件   |
| 输出时机   | 运行中周期性输出                            | 数据采集完毕后一次性评估    |
| 适用场景   | 实时观察 SLAM 性能                        | 精确的最终评估         |
| 时间戳匹配  | 实时对齐                                | 最近邻匹配（支持序号匹配模式） |

### 6.4 GT 同步机制

为确保 GT 和 SLAM 轨迹的起点对齐，`pcd.py` 采用了 GT 同步策略：

- GT 轨迹记录在收到第一条 `/odom` 消息后才开始
- 这样无论 SLAM 算法初始化多慢（如 ORB-SLAM3 词汇表加载），GT 和 SLAM 的时间范围都能对齐

### 6.5 评估器内部流程

评估器并不是简单地把两条轨迹直接做差，而是分为以下几个步骤：

1. **轨迹读取**：
   - 在线评估器实时订阅 `/ground_truth/odom` 与 `/odom`；
   - 离线评估器读取 TUM 轨迹文件。
2. **时间关联**：
   - 默认采用最近邻时间戳匹配，并要求时间差小于阈值；
3. **跳过初始帧**：
   - 默认略过前若干帧，避免 SLAM 初始化阶段对结果造成过大干扰。
4. **轨迹对齐**：
   - `translation`：仅消除平移偏移；
   - `SE3`：校正平移 + 旋转；
   - `Sim3`：校正平移 + 旋转 + 尺度（默认）。
5. **误差计算**：
   - ATE 评估全局位置偏差；
   - RPE 评估一段时间间隔内的相对平移/旋转误差。
6. **可视化与报告输出**：
   - 轨迹对比图、ATE 分布图、RPE 分析图；
   - 最终报告输出为 JSON，便于后续论文或表格整理。

### 6.6 为什么会出现“轨迹图很贴，但 ATE / RPE 偏高”

这是本平台中非常常见、也最容易误判的现象。常见原因包括：

1. **时间关联只是最近邻，而不是严格同帧**：
   两条轨迹看起来形状相似，但同一时刻的点没有真正配准到一起。
2. **轨迹图通常展示的是对齐后的结果**：
   默认 Sim3 会把平移、旋转、尺度一起校正，因此图像上容易看起来“非常贴”，但这并不代表原始算法输出已经天然准确。
3. **ATE 与 RPE关注的层面不同**：
   - ATE 更强调全局位置是否漂移；
   - RPE 更强调局部运动是否准确。
     因此会出现“整体形状对，但局部运动抖动很大”的情况。
4. **轨迹长度、采样密度或节奏不一致**：
   轨迹线条在俯视图中看似重叠，但按时间点一一比较时，误差仍可能较大。

因此，在分析结果时，不应只看轨迹图是否重合，而应同时检查：

- 时间匹配是否合理；
- Sim3 scale 是否接近 1；
- 路径长度比是否合理；
- ATE / RPE 时序图是否存在系统性偏移或起步异常。

***

## 7. UI 功能说明

### 7.1 启动标签页

**数据源选择**：

- **实时仿真**：启动 `official.py` 连接 CARLA 仿真器
- **回放录制数据**：指定 rosbag 数据目录，通过 `ros2 bag play --clock` 回放

**SLAM 算法选择**：下拉框选择算法，自动切换参数面板和启动命令。

**回放模式注意事项**：

- 回放延迟 3 秒后再启动 SLAM 算法，等待 `/clock` 话题稳定
- Lightweight SLAM (all.py) 不支持回放模式
- 回放速率可调（默认 1.0x），支持循环播放

### 7.2 车辆控制标签页

- **自动驾驶**：启用 `auto_controller.py`，使用基于车道路点的 Pure Pursuit 横向控制 + 简单比例纵向速度控制，使车辆稳定巡航。
- **手动驾驶**：启用 `manual_control_input.py` 与 `manual_controller.py`，通过 WASD/空格/R/Tab 等键位完成人工驾驶与模式切换。
- **控制器切换逻辑**：UI 实际上不是切换一套新的主循环，而是通过 `/vehicle/control_mode` 通知 `official.py` 切换当前控制器；主循环保持不变。

### 7.3 数据录制标签页

- 录制传感器数据为 MCAP 格式 rosbag；
- 数据集浏览器：列出已录制的数据集，支持一键选中回放；
- 录制的数据集存储在 `datasets/` 目录下；

### 7.4 点云累积标签页

- 配置 `pcd.py` 参数：体素大小、最大点数、轨迹抽样间隔等；
- 订阅 `/map` 后按块累积点云，并进行两级体素降采样（块级 + 全局级）；
- 输出 PLY 点云地图，以及 `/odom`、`/ground_truth/odom` 对应的 TUM 轨迹文件；
- 每次运行自动创建 `output/run_YYYYMMDD_HHMMSS/` 子目录；

### 7.5 SLAM 评估标签页

- **在线评估**：实时计算 ATE/RPE，周期性输出报告
- **离线评估**：选择 GT 和 SLAM 轨迹文件，一次性评估并生成图表
- 支持 Sim3/SE3 对齐方式选择

### 7.6 可视化标签页

- 启动 RViz2 查看实时点云和轨迹
- 启动 CloudCompare 查看 PLY 点云文件

### 7.7 地图切换标签页

- 通过 Python API 或 ROS Service 切换 CARLA 地图

***

## 8. 部署模式

本项目支持两种部署方式：

### 8.1 单机模式（推荐）

CARLA 和 ROS2 运行在同一台 Linux 机器上，这是大多数用户的典型场景。

**配置**：`official.py` 中的连接地址改为本机：

```python
HOST = "localhost"   # 或 "127.0.0.1"
PORT = 2000
```

无需额外网络配置，直接启动即可。

### 8.2 分离模式（跨主机）

CARLA 运行在一台机器（如 Windows 主机，GPU 性能更强），ROS2 运行在另一台机器（如 Linux 虚拟机）。本项目开发时采用的就是这种模式。

**配置**：`official.py` 中的 `HOST` 设为 CARLA 所在机器的局域网 IP：

```python
HOST = "192.168.1.243"  # CARLA 主机的 IP 地址
PORT = 2000
```

**注意事项**：

- 确保两台机器在同一局域网，且 CARLA 端口（默认 2000/2001）未被防火墙拦截
- CARLA 启动时需加 `-carla-rpc-port=2000` 参数（默认值）
- 网络延迟可能影响同步模式的稳定性，建议使用有线连接
- 无法正常使用手动驾驶功能

***

## 9. 使用流程

### 9.1 实时仿真流程

```
1. 启动 CARLA 服务器
2. 打开 UI：python3 ui.py
3. 启动标签页 → 选择"实时仿真" → 选择 SLAM 算法 → 点击"启动"
4. （可选）点云累积标签页 → 启动 pcd.py
5. 等待数据采集（Rviz2 可实时观察）
6. 停止所有进程（pcd.py 自动保存 PLY + TUM 文件）
7. SLAM 评估标签页 → 选择 GT/EST 轨迹文件 → 开始离线评估
8. 查看 output/ 目录下的评估报告和图表
```

### 9.2 回放评估流程

```
1. 确保已有录制数据（datasets/ 目录下）
2. 数据录制标签页 → 选中数据集 → 点击"回放选中"
3. 自动跳转到启动标签页，数据源切换为"回放"
4. 选择 SLAM 算法 → 点击"启动"
5. 后续流程与实时仿真相同
```

### 9.3 扩展自定义 SLAM 算法

用户可参考 `all.py` 的实现范式接入自定义 SLAM 算法，需满足：

1. 订阅 `/lidar/points`（PointCloud2）和/或 `/imu/data`（Imu）
2. 发布 `/odom`（Odometry）— 里程计输出
3. 发布 `/map`（PointCloud2）— 点云地图（可选）
4. 设置 `use_sim_time:=true` 参数

在 UI 的 SLAM 脚本路径中指定自定义脚本即可启动。

***

