import numpy as np
import cv2
import time
import os
import numpy as np
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from .feature_tracker import FeatureTracker
from .imu_preintegration import IMUPreintegration
from .initializer import VIOInitializer
from .sliding_window_optimizer import SlidingWindowOptimizer

class VIOSystem:
    """
    视觉惯性里程计系统
    整合特征跟踪、IMU预积分、初始化和优化模块
    """
    
    def __init__(self, camera_matrix, window_size=10):
        """
        初始化VIO系统
        
        参数:
            camera_matrix: 相机内参矩阵
            window_size: 滑动窗口大小
        """
        self.K = camera_matrix
        
        # 创建各个模块
        self.feature_tracker = FeatureTracker()
        self.imu_preintegration = IMUPreintegration()
        self.initializer = VIOInitializer(camera_matrix)
        self.optimizer = SlidingWindowOptimizer(window_size)
        
        # 设置相机参数
        self.optimizer.set_camera_params(
            K=camera_matrix,
            R_cam_imu=np.eye(3),  # 假设IMU和相机坐标系对齐
            t_cam_imu=np.zeros(3)
        )
        
        # 系统状态
        self.initialized = False
        self.current_state = None
        self.states_history = []
        
        # 当前帧和时间戳
        self.current_frame = None
        self.current_timestamp = None
        
        # IMU数据缓冲
        self.imu_buffer = []
        
        # 上一帧的时间戳
        self.last_frame_timestamp = None
    
    def process_imu(self, imu_data, timestamp):
        """
        处理IMU数据
        
        参数:
            imu_data: IMU数据 {'acc': [ax, ay, az], 'gyro': [gx, gy, gz]}
            timestamp: 时间戳
        """
        # 添加到IMU缓冲区
        self.imu_buffer.append({
            'timestamp': timestamp,
            'acc': np.array(imu_data['acc']),
            'gyro': np.array(imu_data['gyro'])
        })
        
        # 如果已初始化，执行IMU预测
        if self.initialized and self.current_state is not None:
            self._imu_prediction(imu_data, timestamp)
    
    def process_frame(self, frame, timestamp):
        """
        处理视频帧
        
        参数:
            frame: 视频帧
            timestamp: 时间戳
            
        返回:
            处理结果，包括特征点、轨迹等
        """
        self.current_frame = frame
        self.current_timestamp = timestamp
        
        # 转换为灰度图
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # 跟踪特征点
        curr_pts, curr_ids, feature_tracks = self.feature_tracker.track_features(gray)
        
        # 准备特征观测
        feature_observations = {}
        if curr_pts is not None:
            for i, (pt, id_) in enumerate(zip(curr_pts, curr_ids)):
                feature_observations[int(id_)] = pt[0]
        
        # 如果尚未初始化
        if not self.initialized:
            # 收集IMU数据
            imu_data_since_last = []
            if self.last_frame_timestamp is not None:
                for data in self.imu_buffer:
                    if self.last_frame_timestamp <= data['timestamp'] <= timestamp:
                        imu_data_since_last.append(data)
            
            # 准备帧数据
            frame_data = {
                'timestamp': timestamp,
                'features': {int(id_): pt[0].tolist() for id_, pt in zip(curr_ids, curr_pts)} if curr_pts is not None else {},
                'imu_data': imu_data_since_last
            }
            
            # 尝试初始化
            init_success = self.initializer.add_frame(frame_data)
            
            print(f"初始化尝试: {'成功' if init_success else '失败'}")
            print(f"当前特征点数量: {len(curr_pts) if curr_pts is not None else 0}")
            print(f"初始化帧数量: {len(self.initializer.init_frames)}")
            print(f"IMU数据点数量: {len(imu_data_since_last)}")
            
            # 即使初始化失败，也创建一个基本的轨迹
            # 如果我们有足够的帧和特征点
            # 降低初始化要求：减少所需的最小帧数和特征点数
            if not init_success and len(self.initializer.init_frames) >= 2 and curr_pts is not None and len(curr_pts) >= 8:
                # 根据IMU数据创建一个简单的初始状态
                # 计算平均加速度和角速度
                avg_acc = np.zeros(3)
                avg_gyro = np.zeros(3)
                count = 0
                
                for data in imu_data_since_last:
                    avg_acc += np.array(data['acc'])
                    avg_gyro += np.array(data['gyro'])
                    count += 1
                
                if count > 0:
                    avg_acc /= count
                    avg_gyro /= count
                
                print(f"使用简化初始化:")
                print(f"  平均加速度: {avg_acc.tolist()}")
                print(f"  平均角速度: {avg_gyro.tolist()}")
                
                # 创建一个基本的初始状态
                self.current_state = {
                    'timestamp': timestamp,
                    'position': np.zeros(3),  # 初始位置设为原点
                    'rotation': np.eye(3),    # 初始旋转设为单位矩阵
                    'velocity': np.zeros(3),   # 初始速度设为零
                    'acc_bias': np.zeros(3),   # 初始加速度偏置设为零
                    'gyro_bias': np.zeros(3)   # 初始陀螺仪偏置设为零
                }
                
                # 设置重力向量为默认值
                self.optimizer.set_gravity(np.array([0, 0, 9.81]))
                
                # 重置IMU预积分
                self.imu_preintegration.reset()
                
                # 添加初始帧到优化器
                self.optimizer.add_frame(
                    self.current_state,
                    None,  # 没有预积分
                    feature_observations
                )
                
                # 标记为已初始化
                self.initialized = True
                
                # 保存状态历史
                self.states_history.append(self.current_state.copy())
                
                print("VIO系统使用简化初始化!")
            
            elif init_success:
                # 获取初始化结果
                init_result = self.initializer.get_initialization_result()
                
                print(f"正常初始化成功:")
                print(f"  初始旋转: {[round(x, 3) for x in R.from_matrix(init_result['init_rotation']).as_euler('xyz', degrees=True).tolist()]}度")
                print(f"  初始速度: {init_result['init_velocity'].tolist()}")
                print(f"  重力向量: {init_result['gravity'].tolist()}")
                
                # 设置初始状态
                self.current_state = {
                    'timestamp': timestamp,
                    'position': np.zeros(3),  # 初始位置设为原点
                    'rotation': init_result['init_rotation'],
                    'velocity': init_result['init_velocity'],
                    'acc_bias': init_result['init_acc_bias'],
                    'gyro_bias': init_result['init_gyro_bias']
                }
                
                # 设置重力向量
                self.optimizer.set_gravity(init_result['gravity'])
                
                # 设置IMU预积分器的偏置
                self.imu_preintegration.acc_bias = init_result['init_acc_bias']
                self.imu_preintegration.gyro_bias = init_result['init_gyro_bias']
                
                # 重置IMU预积分
                self.imu_preintegration.reset()
                
                # 添加初始帧到优化器
                self.optimizer.add_frame(
                    self.current_state,
                    None,  # 没有预积分
                    feature_observations
                )
                
                # 标记为已初始化
                self.initialized = True
                
                # 保存状态历史
                self.states_history.append(self.current_state.copy())
        else:
            # 已初始化，执行正常处理
            
            # 收集上一帧到当前帧之间的IMU数据
            imu_data_since_last = []
            for data in self.imu_buffer:
                if self.last_frame_timestamp <= data['timestamp'] <= timestamp:
                    imu_data_since_last.append(data)
            
            # 执行IMU预积分
            self.imu_preintegration.reset()
            for i in range(1, len(imu_data_since_last)):
                prev_data = imu_data_since_last[i-1]
                curr_data = imu_data_since_last[i]
                
                dt = curr_data['timestamp'] - prev_data['timestamp']
                if dt > 0:
                    self.imu_preintegration.integrate(
                        curr_data['acc'],
                        curr_data['gyro'],
                        dt
                    )
            
            # 使用IMU预积分结果预测当前状态
            prev_state = self.current_state
            dt = timestamp - self.last_frame_timestamp
            
            if dt > 0:
                # 预测位置
                delta_p = self.imu_preintegration.delta_p
                delta_v = self.imu_preintegration.delta_v
                delta_q = self.imu_preintegration.delta_q
                
                prev_rot = prev_state['rotation']
                prev_pos = prev_state['position']
                prev_vel = prev_state['velocity']
                
                # 预测位置
                pred_pos = prev_pos + prev_vel * dt + 0.5 * (prev_rot @ delta_p)
                
                # 预测速度
                gravity = self.optimizer.gravity
                pred_vel = prev_vel + prev_rot @ delta_v - gravity * dt
                
                # 预测旋转
                r_delta = R.from_quat(delta_q)
                pred_rot = prev_rot @ r_delta.as_matrix()
                
                # 更新当前状态
                self.current_state = {
                    'timestamp': timestamp,
                    'position': pred_pos,
                    'rotation': pred_rot,
                    'velocity': pred_vel,
                    'acc_bias': prev_state['acc_bias'],
                    'gyro_bias': prev_state['gyro_bias']
                }
                
                # 添加帧到优化器
                self.optimizer.add_frame(
                    self.current_state,
                    {
                        'delta_p': self.imu_preintegration.delta_p,
                        'delta_v': self.imu_preintegration.delta_v,
                        'delta_q': self.imu_preintegration.delta_q,
                        'covariance': self.imu_preintegration.covariance
                    },
                    feature_observations
                )
                
                # 执行优化
                optimized_states = self.optimizer.optimize()
                
                # 添加调试信息：检查优化结果
                print(f"优化结果: {'成功' if len(optimized_states) > 0 else '失败'}")
                if len(optimized_states) > 0:
                    print(f"  优化前位置: {self.current_state['position'].tolist()}")
                    print(f"  优化后位置: {optimized_states[-1]['position'].tolist()}")
                    print(f"  位置变化: {np.linalg.norm(optimized_states[-1]['position'] - self.current_state['position']):.4f}米")
                
                # 更新当前状态
                if len(optimized_states) > 0:
                    self.current_state = optimized_states[-1]
                
                # 保存状态历史
                self.states_history.append(self.current_state.copy())
        
        # 更新上一帧时间戳
        self.last_frame_timestamp = timestamp
        
        # 清理旧的IMU数据
        self._clean_imu_buffer()
        
        # 准备返回结果
        result = {
            'timestamp': timestamp,
            'features': {
                'points': curr_pts,
                'ids': curr_ids,
                'tracks': feature_tracks
            },
            'state': self.current_state,
            'initialized': self.initialized
        }
        
        return result
    
    def _imu_prediction(self, imu_data, timestamp):
        """
        使用IMU数据进行状态预测
        
        参数:
            imu_data: IMU数据
            timestamp: 时间戳
        """
        if self.current_state is None or self.last_frame_timestamp is None:
            return
        
        # 计算时间差
        dt = timestamp - self.current_state['timestamp']
        
        if dt <= 0:
            return
        
        # 提取IMU数据
        acc = np.array(imu_data['acc'])
        gyro = np.array(imu_data['gyro'])
        
        # 去除偏置
        acc_corrected = acc - self.current_state['acc_bias']
        gyro_corrected = gyro - self.current_state['gyro_bias']
        
        # 提取当前状态
        position = self.current_state['position']
        velocity = self.current_state['velocity']
        rotation = self.current_state['rotation']
        
        # 重力向量
        gravity = self.optimizer.gravity
        
        # 旋转加速度到世界坐标系
        acc_world = rotation @ acc_corrected - gravity
        
        # 更新位置和速度
        velocity_new = velocity + acc_world * dt
        position_new = position + velocity * dt + 0.5 * acc_world * dt * dt
        
        # 更新旋转 - 增强错误处理
        try:
            angle = np.linalg.norm(gyro_corrected) * dt
            if angle > 1e-6:  # 增大阈值，避免数值问题
                axis = gyro_corrected / np.linalg.norm(gyro_corrected)
                r = R.from_rotvec(axis * angle)
                rotation_delta = r.as_matrix()
                rotation_new = rotation @ rotation_delta
            else:
                # 当角速度非常小时，直接使用原来的旋转
                rotation_new = rotation
        except Exception as e:
            print(f"IMU旋转更新错误: {e}")
            print(f"gyro_corrected: {gyro_corrected}, dt: {dt}, angle: {angle if 'angle' in locals() else 'N/A'}")
            # 出错时保持原来的旋转
            rotation_new = rotation
        
        # 更新状态
        self.current_state['timestamp'] = timestamp
        self.current_state['position'] = position_new
        self.current_state['velocity'] = velocity_new
        self.current_state['rotation'] = rotation_new
    
    def _clean_imu_buffer(self):
        """清理旧的IMU数据"""
        if self.last_frame_timestamp is None:
            return
        
        # 保留最近的IMU数据
        self.imu_buffer = [data for data in self.imu_buffer 
                          if data['timestamp'] >= self.last_frame_timestamp - 1.0]
    
    def get_trajectory(self):
        """
        获取轨迹
        
        返回:
            轨迹点列表 [(timestamp, position, rotation), ...]
        """
        trajectory = []
        
        for state in self.states_history:
            trajectory.append((
                state['timestamp'],
                state['position'].copy(),
                state['rotation'].copy()
            ))
        
        return trajectory
    
    def reset(self):
        """重置VIO系统"""
        self.feature_tracker.reset()
        self.imu_preintegration.reset()
        self.initializer = VIOInitializer(self.K)
        self.optimizer = SlidingWindowOptimizer()
        
        # 重置状态
        self.initialized = False
        self.current_state = None
        self.states_history = []
        
        self.current_frame = None
        self.current_timestamp = None
        
        self.imu_buffer = []
        self.last_frame_timestamp = None
    
    def visualize_current_frame(self):
        """
        可视化当前帧
        
        返回:
            可视化图像
        """
        if self.current_frame is None:
            return None
        
        vis_img = self.current_frame.copy()
        
        # 如果是灰度图，转换为彩色图
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # 绘制特征点和轨迹
        if self.feature_tracker.curr_pts is not None:
            vis_img = self.feature_tracker.draw_tracks(vis_img)
        
        # 绘制状态信息
        if self.current_state is not None:
            pos = self.current_state['position']
            vel = self.current_state['velocity']
            
            # 位置
            pos_text = f"Pos: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}"
            cv2.putText(vis_img, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 速度
            vel_text = f"Vel: {vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}"
            cv2.putText(vis_img, vel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 初始化状态
            init_text = "Initialized" if self.initialized else "Initializing..."
            cv2.putText(vis_img, init_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if self.initialized else (0, 0, 255), 2)
        
        return vis_img
