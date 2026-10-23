# 环境安装指南

本文档说明 CARLA-ROS2 多算法 SLAM 测试与评估平台的完整环境搭建步骤。

目标系统：Ubuntu 22.04（原生或虚拟机）

---

## 1. ROS2 Humble 安装

```bash
# 设置 locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加 ROS2 apt 源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装 ROS2 Humble 桌面版
sudo apt update
sudo apt install ros-humble-desktop -y

# 安装开发工具和额外依赖
sudo apt install ros-humble-topic-tools python3-colcon-common-extensions python3-rosdep -y

# 初始化 rosdep
sudo rosdep init
rosdep update

# 添加到 bashrc（每次终端自动加载）
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

验证安装：
```bash
ros2 topic list
# 应输出 /parameter_events 和 /rosout
```

---

## 2. Python 依赖安装

```bash
# 核心依赖
pip install numpy open3d transforms3d matplotlib pyyaml

# GUI（可选，缺失时自动回退到标准 ttk）
pip install ttkbootstrap

# OpenCV
pip install opencv-python

# cv_bridge（ROS2 包）
sudo apt install ros-humble-cv-bridge -y
```

---

## 3. CARLA 仿真器安装

### 3.1 下载 CARLA

从 [CARLA GitHub Releases](https://github.com/carla-simulator/carla/releases) 下载：

```bash
# 下载并解压（以 0.9.15 为例，也可0.9.14，但与之对应的 CARLA API 也需要 0.9.14）
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
mkdir -p ~/carla && tar -xzf CARLA_0.9.15.tar.gz -C ~/carla
```

### 3.2 安装 CARLA Python API

```bash
# 方式一：pip 安装
pip install carla==0.9.15

# 方式二：从 CARLA 目录安装 egg/whl
pip install ~/carla/PythonAPI/carla/dist/carla-0.9.15-cp310-cp310-linux_x86_64.whl
```

### 3.3 启动 CARLA 服务器

```bash
cd ~/carla
./CarlaUE4.sh 
# 或
.\CarlaUE4.exe
# 也可增加相关启动参数以确定执行端口以及降低图形渲染负担
.\CarlaUE4.exe -carla-server -quality-level=Low -fps=20 -carla-rpc-port=2000 -carla-streaming-port=2001 -carla-map=Town10HD_Opt
```

> 如果 CARLA 运行在另一台机器上（分离模式，如虚拟机与客户机），需修改 `official.py` 中的 `HOST` 为该机器的 IP 地址。

---

## 4. KISS-ICP 安装

```bash
pip install kiss-icp
```

验证：
```bash
python3 -c "import kiss_icp; print(kiss_icp.__version__)"
```

---

## 5. LIO-SAM 安装（官方 C++ ROS2 版）

### 5.1 安装依赖

```bash
# GTSAM
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt update
sudo apt install libgtsam-dev libgtsam-unstable-dev -y

# PCL（通常 ROS2 已包含）
sudo apt install libpcl-dev -y
```

### 5.2 编译 LIO-SAM

```bash
mkdir -p ~/lio_sam_ws/src && cd ~/lio_sam_ws/src

# 克隆 LIO-SAM（ROS2 分支）
git clone -b ros2 https://github.com/TixiaoShan/LIO-SAM.git

cd ~/lio_sam_ws

# 编译
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# 添加到 bashrc
echo "source ~/lio_sam_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 5.3 配置参数文件

将项目文件中配套的的 `lio_sam_params_carla.yaml` 复制到 LIO-SAM 的配置文件 `config.yaml`，或在 UI 中指定参数文件路径。

验证：
```bash
ros2 launch lio_sam run.launch.py
```

---

## 6. ORB-SLAM3 安装（官方 C++ ROS2 版）

### 6.1 安装依赖

```bash
# Eigen3
sudo apt install libeigen3-dev -y

# Pangolin（可视化库）
cd ~
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# OpenCV（通常已安装）
sudo apt install libopencv-dev -y
```

### 6.2 编译 ORB-SLAM3 库

```bash
cd ~
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3-master
cd ORB_SLAM3-master

# 编译第三方库
chmod +x build.sh
./build.sh
```

### 6.3 编译 ORB-SLAM3 ROS2 包装

```bash
mkdir -p ~/orb_slam3_ws/src && cd ~/orb_slam3_ws/src

# 克隆 ROS2 wrapper（根据实际使用的 wrapper 版本调整）
git clone https://github.com/zang09/ORB-SLAM3-STEREO-FIXED.git orbslam3_ros2
# 或其他 ROS2 wrapper

cd ~/orb_slam3_ws

# 编译
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "source ~/orb_slam3_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 6.4 配置

项目中的 `orb_slam3_carla.yaml` 已针对 CARLA 相机参数进行了标定，依旧复制到对应工作空间下的配置文件 `config.yaml`：
- 分辨率：1280×720
- FOV：90°
- fx = fy = 640, cx = 640, cy = 360

验证：
```bash
ros2 run orbslam3 rgbd ~/ORB_SLAM3-master/Vocabulary/ORBvoc.txt orb_slam3_carla.yaml
# 应正常启动，等待相机数据
```

---

## 7. 快速验证

所有组件安装完成后，执行以下步骤验证环境：

```bash
# 1. 启动 CARLA（另一个终端）
cd ~/carla && ./CarlaUE4.sh

# 2. 进入项目目录
cd ~/桌面/CARLA   # 或项目路径

# 3. 启动 UI
python3 ui.py

# 4. 在 UI 中：
#    - 启动标签页 → 选择 SLAM 算法 → 点击"启动"
#    - 观察日志输出是否正常
#    - Rviz2 中是否能看到点云和轨迹
```

---

## 8. 常见问题


### Q: LIO-SAM 编译报错找不到 GTSAM
```bash
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt update && sudo apt install libgtsam-dev libgtsam-unstable-dev -y
# 注意：此步骤视主机环境版本，可能存在一定的 g++ 版本不兼容问题，需手动更改脚本改用 g++10 等版本进行编译
```

### Q: `ros2: command not found`
```bash
source /opt/ros/humble/setup.bash
```

### Q: CARLA 连接失败 `RuntimeError: time-out`
- 确认 `official.py` 中的 `HOST` 和 `PORT` 正确
- 单机模式用 `localhost`，分离模式用 CARLA 主机 IP
- 检查防火墙是否放行 2000/2001 端口
- 若是分离模式，需确认是否两端在同一网关下，若不在，需配置路由转发或设置静态ip

### Q: pcd.py 订阅不到 LIO-SAM 的话题
LIO-SAM 的话题名与标准不同，平台已通过 `topic_tools relay` 自动转发。确保 `ros-humble-topic-tools` 已安装：
```bash
sudo apt install ros-humble-topic-tools -y
```
