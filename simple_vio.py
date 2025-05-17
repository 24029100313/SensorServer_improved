import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from datetime import datetime

class SimpleVIO:
    """
    简单的视觉惯性里程计(VIO)实现
    整合IMU数据、陀螺仪数据和视频帧来估计相机的运动轨迹
    """
    
    def __init__(self):
        # 相机内参矩阵 (需要根据实际相机进行标定)
        self.K = np.array([
            [525.0, 0, 320.0],  # fx, 0, cx
            [0, 525.0, 240.0],  # 0, fy, cy
            [0, 0, 1.0]         # 0, 0, 1
        ])
        
        # 特征检测器
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # 特征匹配器
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 当前位置和姿态
        self.position = np.zeros(3)  # [x, y, z]
        self.orientation = np.eye(3)  # 旋转矩阵
        
        # 轨迹历史
        self.trajectory = [self.position.copy()]
        
        # 上一帧的关键点和描述符
        self.prev_kp = None
        self.prev_des = None
        self.prev_frame = None
        
        # 上一时刻的时间戳
        self.prev_timestamp = None
        
        # IMU偏置
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)
        
        # 重力向量 (假设z轴向上)
        self.gravity = np.array([0, 0, 9.81])
        
        # 速度
        self.velocity = np.zeros(3)
        
        # 协方差矩阵 (简化版)
        self.covariance = np.eye(15) * 0.01  # 位置(3), 速度(3), 姿态(3), 陀螺仪偏置(3), 加速度计偏置(3)
        
        # 过程噪声
        self.process_noise = np.eye(12) * 0.01  # 加速度(3), 角速度(3), 陀螺仪偏置变化(3), 加速度计偏置变化(3)
        
        # 是否初始化
        self.initialized = False
    
    def process_imu_data(self, imu_data, dt):
        """处理IMU数据进行预测"""
        # 提取加速度数据
        acc_x = imu_data['values'][0]
        acc_y = imu_data['values'][1]
        acc_z = imu_data['values'][2]
        
        # 减去偏置并转换到世界坐标系
        acc_body = np.array([acc_x, acc_y, acc_z]) - self.acc_bias
        acc_world = self.orientation @ acc_body
        
        # 减去重力影响
        acc_world = acc_world - self.gravity
        
        # 更新速度和位置 (简单积分)
        self.velocity += acc_world * dt
        self.position += self.velocity * dt + 0.5 * acc_world * dt * dt
        
        return acc_body, acc_world
    
    def process_gyro_data(self, gyro_data, dt):
        """处理陀螺仪数据更新姿态"""
        # 提取角速度数据 (弧度/秒)
        gyro_x = gyro_data['values'][0]
        gyro_y = gyro_data['values'][1]
        gyro_z = gyro_data['values'][2]
        
        # 减去偏置
        gyro = np.array([gyro_x, gyro_y, gyro_z]) - self.gyro_bias
        
        # 计算旋转增量
        angle = np.linalg.norm(gyro) * dt
        if angle > 0:
            axis = gyro / np.linalg.norm(gyro)
            
            # 创建旋转矩阵
            r = R.from_rotvec(axis * angle)
            delta_R = r.as_matrix()
            
            # 更新姿态
            self.orientation = self.orientation @ delta_R
        
        return gyro
    
    def detect_features(self, frame):
        """检测图像中的特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = self.orb.detect(gray, None)
        kp, des = self.orb.compute(gray, kp)
        return kp, des, gray
    
    def process_frame(self, frame, timestamp):
        """处理视频帧进行位置校正"""
        # 检测特征点
        kp, des, gray = self.detect_features(frame)
        
        # 如果是第一帧，只保存特征点
        if self.prev_kp is None or self.prev_des is None:
            self.prev_kp = kp
            self.prev_des = des
            self.prev_frame = gray
            return None
        
        # 匹配特征点
        matches = self.bf.match(self.prev_des, des)
        
        # 根据距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 只使用好的匹配
        good_matches = matches[:50] if len(matches) > 50 else matches
        
        if len(good_matches) < 8:
            # 匹配点太少，无法可靠估计运动
            self.prev_kp = kp
            self.prev_des = des
            self.prev_frame = gray
            return None
        
        # 提取匹配点坐标
        prev_pts = np.float32([self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(prev_pts, curr_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is None:
            self.prev_kp = kp
            self.prev_des = des
            self.prev_frame = gray
            return None
        
        # 从本质矩阵恢复R和t
        _, R_cv, t_cv, mask = cv2.recoverPose(E, prev_pts, curr_pts, self.K, mask=mask)
        
        # 尺度不确定性问题 - 使用IMU数据辅助确定尺度
        # 这里使用一个简单的启发式方法，实际应用中应该使用更复杂的滤波方法
        scale = 0.1  # 这个尺度因子需要根据实际情况调整
        
        # 更新位置和姿态 (视觉部分)
        t = scale * t_cv.reshape(3)
        R_delta = R_cv
        
        # 将视觉估计与IMU预测融合 (简单加权平均)
        vision_weight = 0.7
        imu_weight = 1.0 - vision_weight
        
        # 位置融合
        position_vision = self.position + self.orientation @ t
        self.position = imu_weight * self.position + vision_weight * position_vision
        
        # 姿态融合 (使用SLERP)
        r1 = R.from_matrix(self.orientation)
        r2 = R.from_matrix(self.orientation @ R_delta)
        r_fused = R.from_matrix(r1.as_matrix().T @ r2.as_matrix())
        angle_axis = r_fused.as_rotvec()
        angle_axis = angle_axis * vision_weight
        r_fused = R.from_rotvec(angle_axis)
        self.orientation = self.orientation @ r_fused.as_matrix()
        
        # 更新轨迹
        self.trajectory.append(self.position.copy())
        
        # 保存当前帧的特征点用于下一次匹配
        self.prev_kp = kp
        self.prev_des = des
        self.prev_frame = gray
        
        # 可视化匹配结果
        vis_img = cv2.drawMatches(self.prev_frame, self.prev_kp, gray, kp, good_matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        return vis_img
    
    def process_synchronized_data(self, imu_data, gyro_data, frame, timestamp):
        """处理同步的IMU、陀螺仪和视频帧数据"""
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            self.process_frame(frame, timestamp)
            return None
        
        # 计算时间差
        dt = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp
        
        if dt <= 0:
            return None
        
        # 处理IMU和陀螺仪数据进行预测
        self.process_gyro_data(gyro_data, dt)
        self.process_imu_data(imu_data, dt)
        
        # 处理视频帧进行校正
        vis_img = self.process_frame(frame, timestamp)
        
        return vis_img
    
    def plot_trajectory(self):
        """绘制估计的轨迹"""
        trajectory = np.array(self.trajectory)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', label='Estimated Trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c='g', marker='o', s=100, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c='r', marker='o', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Estimated Camera Trajectory')
        ax.legend()
        
        plt.tight_layout()
        return fig


def load_data_from_session(session_dir, movement_type):
    """从数据收集会话中加载特定运动类型的数据"""
    movement_dir = os.path.join(session_dir, movement_type)
    
    # 加载IMU数据
    imu_file = os.path.join(movement_dir, "imu_data", "imu_data.json")
    with open(imu_file, 'r') as f:
        imu_data = json.load(f)
    
    # 加载陀螺仪数据
    gyro_file = os.path.join(movement_dir, "gyro_data", "gyro_data.json")
    with open(gyro_file, 'r') as f:
        gyro_data = json.load(f)
    
    # 加载视频帧
    frames_dir = os.path.join(movement_dir, "video_frames")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg') and f.startswith('frame_')])
    
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    
    return imu_data, gyro_data, frames


def synchronize_data(imu_data, gyro_data, frames, fps=30):
    """同步IMU、陀螺仪和视频帧数据"""
    # 假设IMU和陀螺仪数据已经有时间戳
    # 为视频帧创建均匀的时间戳
    frame_timestamps = np.linspace(
        imu_data[0]['timestamp'] if imu_data else 0,
        imu_data[-1]['timestamp'] if imu_data else len(frames) / fps,
        len(frames)
    )
    
    # 创建同步的数据点
    synchronized_data = []
    
    for i, frame_ts in enumerate(frame_timestamps):
        # 找到最接近的IMU数据点
        closest_imu_idx = min(range(len(imu_data)), 
                             key=lambda j: abs(imu_data[j]['timestamp'] - frame_ts))
        
        # 找到最接近的陀螺仪数据点
        closest_gyro_idx = min(range(len(gyro_data)), 
                              key=lambda j: abs(gyro_data[j]['timestamp'] - frame_ts))
        
        synchronized_data.append({
            'timestamp': frame_ts,
            'imu': imu_data[closest_imu_idx],
            'gyro': gyro_data[closest_gyro_idx],
            'frame': frames[i]
        })
    
    return synchronized_data


def process_movement_data(session_dir, movement_type):
    """处理特定运动类型的数据并运行VIO"""
    print(f"处理 {movement_type} 运动数据...")
    
    # 加载数据
    imu_data, gyro_data, frames = load_data_from_session(session_dir, movement_type)
    
    if not imu_data or not gyro_data or not frames:
        print(f"缺少数据，无法处理 {movement_type}")
        return None
    
    print(f"加载了 {len(imu_data)} 个IMU数据点, {len(gyro_data)} 个陀螺仪数据点, {len(frames)} 帧视频")
    
    # 同步数据
    synchronized_data = synchronize_data(imu_data, gyro_data, frames)
    print(f"同步后有 {len(synchronized_data)} 个数据点")
    
    # 创建VIO实例
    vio = SimpleVIO()
    
    # 处理数据
    result_frames = []
    
    for i, data_point in enumerate(synchronized_data):
        if i % 10 == 0:
            print(f"处理数据点 {i+1}/{len(synchronized_data)}")
        
        vis_img = vio.process_synchronized_data(
            data_point['imu'],
            data_point['gyro'],
            data_point['frame'],
            data_point['timestamp']
        )
        
        if vis_img is not None:
            result_frames.append(vis_img)
    
    # 绘制轨迹
    fig = vio.plot_trajectory()
    
    # 保存结果
    results_dir = os.path.join(session_dir, movement_type, "vio_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存轨迹图
    fig.savefig(os.path.join(results_dir, "trajectory.png"))
    
    # 保存轨迹数据
    trajectory = np.array(vio.trajectory)
    np.save(os.path.join(results_dir, "trajectory.npy"), trajectory)
    
    # 保存可视化结果
    for i, frame in enumerate(result_frames):
        cv2.imwrite(os.path.join(results_dir, f"result_{i:04d}.jpg"), frame)
    
    # 创建视频
    if result_frames:
        height, width = result_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(os.path.join(results_dir, "vio_visualization.avi"), 
                                      fourcc, 10.0, (width, height))
        
        for frame in result_frames:
            video_writer.write(frame)
        
        video_writer.release()
    
    print(f"{movement_type} 处理完成，结果保存在 {results_dir}")
    return results_dir


def main():
    """主函数"""
    print("===== 简单VIO实现 =====")
    print("此程序将处理收集的数据并运行视觉惯性里程计算法")
    
    # 获取最新的数据会话目录
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collected_data")
    if not os.path.exists(base_dir):
        print(f"错误: 找不到数据目录 {base_dir}")
        return
    
    session_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("session_")]
    if not session_dirs:
        print(f"错误: 在 {base_dir} 中找不到会话目录")
        return
    
    # 选择最新的会话
    latest_session = max(session_dirs, key=os.path.getctime)
    print(f"使用最新的数据会话: {latest_session}")
    
    # 获取所有运动类型
    movement_types = [d for d in os.listdir(latest_session) 
                     if os.path.isdir(os.path.join(latest_session, d)) and not d.startswith(".")]
    
    if not movement_types:
        print(f"错误: 在 {latest_session} 中找不到运动数据")
        return
    
    print(f"找到以下运动类型: {', '.join(movement_types)}")
    
    # 处理每种运动类型
    results = {}
    for movement in movement_types:
        results[movement] = process_movement_data(latest_session, movement)
    
    print("\n所有运动类型处理完成!")
    print("结果摘要:")
    for movement, result_dir in results.items():
        status = "成功" if result_dir else "失败"
        print(f"- {movement}: {status}")


if __name__ == "__main__":
    main()
