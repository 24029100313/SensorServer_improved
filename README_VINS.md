# 基于VINS-Mono架构的视觉惯性里程计系统

这个项目实现了一个基于VINS-Mono架构的视觉惯性里程计(Visual-Inertial Odometry, VIO)系统，用于整合IMU数据、陀螺仪数据和视频流来估计相机的运动轨迹。

## VINS-Mono简介

[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)是香港科技大学开发的一个优秀的单目视觉惯性里程计系统，具有以下特点：

- 使用滑动窗口优化方法进行状态估计
- 紧耦合的视觉和IMU数据融合
- 在线外部参数标定
- 闭环检测
- 重定位功能

## 我们的实现

本项目参考VINS-Mono的架构，实现了一个简化版的VIO系统，包含以下核心模块：

### 1. 特征跟踪器 (feature_tracker.py)

- 使用光流法跟踪特征点
- 管理特征点ID和生命周期
- 提供特征点轨迹可视化

### 2. IMU预积分 (imu_preintegration.py)

- 执行IMU数据预积分
- 处理偏置校正
- 提供雅可比矩阵计算

### 3. 初始化器 (initializer.py)

- 执行视觉SFM初始化
- 估计重力方向和尺度
- 对齐视觉和IMU坐标系

### 4. 滑动窗口优化器 (sliding_window_optimizer.py)

- 维护滑动窗口状态
- 融合视觉和IMU约束
- 执行边缘化操作

### 5. VIO系统 (vio_system.py)

- 整合所有模块
- 处理传感器数据
- 提供状态估计和可视化

## 使用方法

### 步骤1：收集数据

首先使用`data_collector.py`收集数据：

```bash
python data_collector.py
```

按照提示完成六种不同的运动模式数据收集。

### 步骤2：运行VIO系统

收集完数据后，运行VIO系统处理数据：

```bash
python run_vins_inspired_vio.py
```

程序会自动选择最新的数据会话，并对每种运动类型进行处理。

## 结果输出

对于每种运动类型，VIO系统会生成以下输出：

1. **轨迹图**：显示估计的3D相机运动轨迹
2. **可视化视频**：包含特征点跟踪和状态信息的视频
3. **轨迹数据**：以JSON格式保存的轨迹坐标数据
4. **HTML报告**：汇总所有运动类型的处理结果

所有结果都保存在对应运动类型的`vio_results`子目录中，并在会话目录下生成`vio_report.html`汇总报告。

## 与VINS-Mono的区别

我们的实现是VINS-Mono的简化版本，主要区别包括：

1. 使用Python而非C++实现，便于理解和修改
2. 简化了优化框架，使用scipy.optimize而非Ceres
3. 没有实现闭环检测和重定位功能
4. 简化了初始化和边缘化过程

## 系统要求

- Python 3.6+
- OpenCV
- NumPy
- SciPy
- Matplotlib

## 安装依赖

```bash
pip install numpy scipy matplotlib opencv-python
```

## 进一步改进

- 实现闭环检测功能
- 添加重定位功能
- 改进优化框架，使用更高效的求解器
- 实现更精确的相机-IMU标定
- 添加多传感器融合支持
