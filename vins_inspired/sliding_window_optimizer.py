import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import numba
import time

# 使用Numba加速的独立函数
@numba.jit(nopython=True)
def compute_reprojection_error_batch(points_3d, points_2d, K, R, t):
    """
    批量计算重投影误差
    
    参数:
        points_3d: 3D点数组 (N, 3)
        points_2d: 2D点数组 (N, 2)
        K: 相机内参矩阵 (3, 3)
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
        
    返回:
        重投影误差数组 (N,)
    """
    errors = np.zeros(len(points_3d))
    
    for i in range(len(points_3d)):
        # 转换到相机坐标系
        p_cam = R @ points_3d[i] + t
        
        # 投影到图像平面
        p_img = K @ p_cam
        p_img = p_img[:2] / p_img[2]
        
        # 计算误差
        errors[i] = np.linalg.norm(points_2d[i] - p_img)
    
    return errors

class SlidingWindowOptimizer:
    """
    滑动窗口优化器
    负责融合视觉和IMU数据进行状态估计，类似于VINS-Mono中的backend/optimization.cpp
    """
    
    def __init__(self, window_size=10):
        """
        初始化滑动窗口优化器
        
        参数:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        
        # 滑动窗口中的状态
        self.states = []
        
        # IMU预积分结果
        self.imu_preintegrations = []
        
        # 特征观测
        self.feature_observations = {}
        
        # 边缘化标记
        self.marginalization_flag = 0  # 0: 边缘化最老帧, 1: 边缘化次新帧
        
        # 先验信息
        self.prior_info = None
        
        # 相机内参
        self.K = None
        
        # IMU与相机之间的变换
        self.R_cam_imu = np.eye(3)
        self.t_cam_imu = np.zeros(3)
        
        # 重力向量
        self.gravity = np.array([0, 0, 9.81])
        
        # 缓存
        self.rotation_cache = {}
        self.last_optimization_time = 0
        self.optimization_count = 0
        
        # 优化参数
        self.max_iterations = 10
        self.convergence_threshold = 1e-6
        self.use_robust_cost = True
    
    def set_camera_params(self, K, R_cam_imu, t_cam_imu):
        """
        设置相机参数
        
        参数:
            K: 相机内参矩阵
            R_cam_imu: 从IMU到相机的旋转矩阵
            t_cam_imu: 从IMU到相机的平移向量
        """
        self.K = K
        self.R_cam_imu = R_cam_imu
        self.t_cam_imu = t_cam_imu
    
    def set_gravity(self, gravity):
        """
        设置重力向量
        
        参数:
            gravity: 重力向量
        """
        self.gravity = gravity
    
    def add_frame(self, frame_state, imu_preintegration, feature_observations):
        """
        向滑动窗口添加一帧
        
        参数:
            frame_state: 帧的初始状态估计
                {
                    'timestamp': 时间戳,
                    'position': 位置,
                    'rotation': 旋转矩阵,
                    'velocity': 速度,
                    'acc_bias': 加速度计偏置,
                    'gyro_bias': 陀螺仪偏置
                }
            imu_preintegration: IMU预积分结果
            feature_observations: 特征观测
                {feature_id: [x, y], ...}
        """
        # 添加状态
        self.states.append(frame_state)
        
        # 添加IMU预积分
        if imu_preintegration is not None:
            self.imu_preintegrations.append(imu_preintegration)
        
        # 添加特征观测
        frame_idx = len(self.states) - 1
        for feature_id, observation in feature_observations.items():
            if feature_id not in self.feature_observations:
                self.feature_observations[feature_id] = {}
            self.feature_observations[feature_id][frame_idx] = observation
        
        # 如果窗口已满，执行边缘化
        if len(self.states) > self.window_size:
            self._marginalize()
    
    def optimize(self, max_iterations=None):
        """
        执行滑动窗口优化
        
        参数:
            max_iterations: 最大迭代次数
            
        返回:
            优化后的状态
        """
        if len(self.states) < 2:
            return self.states
        
        # 记录开始时间
        start_time = time.time()
        
        # 使用传入的迭代次数或默认值
        max_iter = max_iterations if max_iterations is not None else self.max_iterations
        
        # 根据优化次数动态调整迭代次数，减少计算量
        if self.optimization_count > 0:
            # 如果是连续优化，可以减少迭代次数
            time_since_last = time.time() - self.last_optimization_time
            if time_since_last < 0.1:  # 如果距离上次优化时间很短
                max_iter = max(3, max_iter // 2)  # 减少迭代次数，但至少3次
        
        # 将状态转换为优化变量
        initial_params = self._states_to_params()
        
        # 定义目标函数
        def objective_function(params):
            # 将优化变量转换回状态
            states = self._params_to_states(params)
            
            # 计算总误差
            total_error = 0.0
            
            # 1. IMU误差
            imu_error = self._compute_imu_error(states)
            total_error += imu_error
            
            # 2. 视觉重投影误差
            reprojection_error = self._compute_reprojection_error(states)
            total_error += reprojection_error
            
            # 3. 先验误差
            if self.prior_info is not None:
                prior_error = self._compute_prior_error(states)
                total_error += prior_error
            
            return total_error
        
        # 执行优化
        result = minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',  # 使用L-BFGS-B方法，比默认的BFGS更快
            options={
                'maxiter': max_iter,
                'ftol': self.convergence_threshold,
                'disp': False
            }
        )
        
        # 更新状态
        optimized_states = self._params_to_states(result.x)
        self.states = optimized_states
        
        # 更新缓存
        self.rotation_cache = {}
        for i, state in enumerate(self.states):
            self.rotation_cache[i] = state['rotation']
        
        # 更新优化统计信息
        self.last_optimization_time = time.time()
        self.optimization_count += 1
        
        # 打印优化性能
        optimization_time = time.time() - start_time
        print(f"优化完成: {result.nit}次迭代, 耗时{optimization_time:.4f}秒")
        
        return optimized_states
    
    def _states_to_params(self):
        """
        将状态转换为优化变量
        
        返回:
            优化变量数组
        """
        params = []
        
        for state in self.states:
            # 位置
            params.extend(state['position'])
            
            # 旋转 (使用四元数)
            r = R.from_matrix(state['rotation'])
            quat = r.as_quat()  # [x, y, z, w]
            params.extend(quat)
            
            # 速度
            params.extend(state['velocity'])
            
            # 加速度计偏置
            params.extend(state['acc_bias'])
            
            # 陀螺仪偏置
            params.extend(state['gyro_bias'])
        
        return np.array(params)
    
    def _params_to_states(self, params):
        """
        将优化变量转换回状态
        
        参数:
            params: 优化变量数组
            
        返回:
            状态列表
        """
        states = []
        param_size_per_state = 16  # 3(位置) + 4(四元数) + 3(速度) + 3(加速度偏置) + 3(陀螺仪偏置)
        
        for i in range(len(self.states)):
            start_idx = i * param_size_per_state
            
            # 提取参数
            position = params[start_idx:start_idx+3]
            quat = params[start_idx+3:start_idx+7]
            velocity = params[start_idx+7:start_idx+10]
            acc_bias = params[start_idx+10:start_idx+13]
            gyro_bias = params[start_idx+13:start_idx+16]
            
            # 四元数转旋转矩阵
            r = R.from_quat(quat)
            rotation = r.as_matrix()
            
            # 创建状态
            state = {
                'timestamp': self.states[i]['timestamp'],
                'position': position,
                'rotation': rotation,
                'velocity': velocity,
                'acc_bias': acc_bias,
                'gyro_bias': gyro_bias
            }
            
            states.append(state)
        
        return states
    
    def _compute_imu_error(self, states):
        """
        计算IMU误差
        
        参数:
            states: 状态列表
            
        返回:
            IMU误差
        """
        total_error = 0.0
        
        for i in range(1, len(states)):
            if i-1 >= len(self.imu_preintegrations):
                continue
            
            preintegration = self.imu_preintegrations[i-1]
            
            # 提取状态
            prev_state = states[i-1]
            curr_state = states[i]
            
            # 计算预测值
            dt = curr_state['timestamp'] - prev_state['timestamp']
            
            # 预测位置
            prev_rot = prev_state['rotation']
            prev_pos = prev_state['position']
            prev_vel = prev_state['velocity']
            prev_acc_bias = prev_state['acc_bias']
            prev_gyro_bias = prev_state['gyro_bias']
            
            # 使用IMU预积分结果计算预测值
            delta_p = preintegration['delta_p']
            delta_v = preintegration['delta_v']
            delta_q = preintegration['delta_q']
            
            # 校正预积分结果
            # 这里简化实现，实际应该使用雅可比矩阵进行校正
            
            # 预测位置
            pred_pos = prev_pos + prev_vel * dt + 0.5 * (prev_rot @ (delta_p - self.gravity * dt * dt))
            
            # 预测速度
            pred_vel = prev_vel + prev_rot @ delta_v - self.gravity * dt
            
            # 预测旋转
            r_delta = R.from_quat(delta_q)
            pred_rot = prev_rot @ r_delta.as_matrix()
            
            # 计算误差
            pos_error = np.linalg.norm(curr_state['position'] - pred_pos) ** 2
            vel_error = np.linalg.norm(curr_state['velocity'] - pred_vel) ** 2
            
            # 旋转误差 (使用四元数差)
            r_curr = R.from_matrix(curr_state['rotation'])
            r_pred = R.from_matrix(pred_rot)
            q_curr = r_curr.as_quat()
            q_pred = r_pred.as_quat()
            q_error = 1.0 - np.abs(np.dot(q_curr, q_pred)) ** 2
            
            # 加权误差
            error = pos_error + vel_error + q_error
            
            # 添加偏置变化惩罚
            bias_error = (np.linalg.norm(curr_state['acc_bias'] - prev_state['acc_bias']) ** 2 +
                         np.linalg.norm(curr_state['gyro_bias'] - prev_state['gyro_bias']) ** 2)
            
            error += 0.1 * bias_error
            
            total_error += error
        
        return total_error
    
    def _compute_reprojection_error(self, states):
        """
        计算视觉重投影误差
        
        参数:
            states: 状态列表
            
        返回:
            重投影误差
        """
        total_error = 0.0
        feature_count = 0
        
        # 使用缓存的旋转矩阵
        if not self.rotation_cache:
            for i, state in enumerate(states):
                self.rotation_cache[i] = state['rotation']
        
        # 遍历特征点
        for feature_id, observations in self.feature_observations.items():
            if len(observations) < 2:
                continue
            
            # 选择一个关键帧作为参考帧
            ref_frame_idx = min(observations.keys())
            ref_state = states[ref_frame_idx]
            
            # 获取特征点在参考帧中的观测
            ref_observation = observations[ref_frame_idx]
            
            # 将特征点从图像平面反投影到3D空间
            # 假设深度为1，后续会优化
            ref_point_img = np.array([ref_observation[0], ref_observation[1], 1.0])
            ref_point_cam = np.linalg.inv(self.K) @ ref_point_img
            
            # 将特征点从相机坐标系转换到IMU坐标系
            ref_R_imu_cam = self.R_cam_imu.T
            ref_t_imu_cam = -ref_R_imu_cam @ self.t_cam_imu
            ref_point_imu = ref_R_imu_cam @ ref_point_cam + ref_t_imu_cam
            
            # 将特征点从IMU坐标系转换到世界坐标系
            ref_R_w_imu = ref_state['rotation']
            ref_t_w_imu = ref_state['position']
            point_world = ref_R_w_imu @ ref_point_imu + ref_t_w_imu
            
            # 计算其他帧的重投影误差
            for frame_idx, observation in observations.items():
                if frame_idx == ref_frame_idx:
                    continue
                
                state = states[frame_idx]
                
                # 将特征点从世界坐标系转换到当前帧的IMU坐标系
                R_imu_w = state['rotation'].T
                t_imu_w = -R_imu_w @ state['position']
                point_imu = R_imu_w @ point_world + t_imu_w
                
                # 将特征点从IMU坐标系转换到相机坐标系
                point_cam = self.R_cam_imu @ point_imu + self.t_cam_imu
                
                # 检查点是否在相机前方
                if point_cam[2] <= 0:
                    continue
                
                # 将特征点投影到图像平面
                point_img = self.K @ point_cam
                point_img = point_img[:2] / point_cam[2]
                
                # 计算重投影误差
                observation_vec = np.array(observation)
                error = np.linalg.norm(observation_vec - point_img)
                
                # 使用鲁棒核函数
                if self.use_robust_cost:
                    error = self._huber_loss(error, 1.0)
                
                total_error += error
                feature_count += 1
        
        # 归一化误差
        if feature_count > 0:
            total_error /= feature_count
        
        return total_error
    
    def _compute_prior_error(self, states):
        """
        计算先验误差
        
        参数:
            states: 状态列表
            
        返回:
            先验误差
        """
        # 简化实现，实际应该使用信息矩阵
        if self.prior_info is None:
            return 0.0
        
        total_error = 0.0
        
        # 计算与先验状态的差异
        for i, (state, prior_state) in enumerate(zip(states, self.prior_info['states'])):
            # 位置误差
            pos_error = np.linalg.norm(state['position'] - prior_state['position']) ** 2
            
            # 旋转误差
            r_state = R.from_matrix(state['rotation'])
            r_prior = R.from_matrix(prior_state['rotation'])
            q_state = r_state.as_quat()
            q_prior = r_prior.as_quat()
            q_error = 1.0 - np.abs(np.dot(q_state, q_prior)) ** 2
            
            # 速度误差
            vel_error = np.linalg.norm(state['velocity'] - prior_state['velocity']) ** 2
            
            # 偏置误差
            bias_error = (np.linalg.norm(state['acc_bias'] - prior_state['acc_bias']) ** 2 +
                         np.linalg.norm(state['gyro_bias'] - prior_state['gyro_bias']) ** 2)
            
            # 加权误差
            error = (self.prior_info['weights']['position'] * pos_error +
                    self.prior_info['weights']['rotation'] * q_error +
                    self.prior_info['weights']['velocity'] * vel_error +
                    self.prior_info['weights']['bias'] * bias_error)
            
            total_error += error
        
        return total_error
    
    def _triangulate_features(self, states):
        """
        三角化特征点
        
        参数:
            states: 状态列表
            
        返回:
            三角化的3D点
        """
        triangulated_points = {}
        
        for feature_id, observations in self.feature_observations.items():
            if len(observations) < 2:
                continue
            
            # 收集观测
            A = []
            
            for frame_idx, observation in observations.items():
                if frame_idx >= len(states):
                    continue
                
                # 提取状态
                state = states[frame_idx]
                
                # 计算投影矩阵
                R_world_imu = state['rotation']
                t_world_imu = state['position']
                
                # 从IMU坐标系到相机坐标系
                R_world_cam = self.R_cam_imu @ R_world_imu
                t_world_cam = self.R_cam_imu @ t_world_imu + self.t_cam_imu
                
                # 投影矩阵
                P = self.K @ np.hstack((R_world_cam, t_world_cam.reshape(3, 1)))
                
                # 构建方程
                x, y = observation
                A.append(x * P[2] - P[0])
                A.append(y * P[2] - P[1])
            
            if len(A) < 4:
                continue
            
            # 求解最小二乘问题
            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            
            # 归一化
            X = X / X[3]
            point_3d = X[:3]
            
            triangulated_points[feature_id] = point_3d
        
        return triangulated_points
    
    def _marginalize(self):
        """
        执行边缘化
        """
        if self.marginalization_flag == 0:
            # 边缘化最老帧
            
            # 保存最老帧的信息作为先验
            if self.prior_info is None:
                self.prior_info = {
                    'states': [self.states[0]],
                    'weights': {
                        'position': 100.0,
                        'rotation': 100.0,
                        'velocity': 100.0,
                        'bias': 100.0
                    }
                }
            else:
                self.prior_info['states'].append(self.states[0])
            
            # 移除最老帧
            self.states.pop(0)
            if len(self.imu_preintegrations) > 0:
                self.imu_preintegrations.pop(0)
            
            # 更新特征观测
            for feature_id in list(self.feature_observations.keys()):
                if 0 in self.feature_observations[feature_id]:
                    del self.feature_observations[feature_id][0]
                
                # 更新帧索引
                new_observations = {}
                for frame_idx, observation in self.feature_observations[feature_id].items():
                    new_observations[frame_idx - 1] = observation
                
                self.feature_observations[feature_id] = new_observations
                
                # 如果特征不再被观测，移除它
                if len(self.feature_observations[feature_id]) == 0:
                    del self.feature_observations[feature_id]
        else:
            # 边缘化次新帧
            # 简化实现，实际应该更复杂
            
            second_newest_idx = len(self.states) - 2
            
            # 移除次新帧
            self.states.pop(second_newest_idx)
            if second_newest_idx < len(self.imu_preintegrations):
                self.imu_preintegrations.pop(second_newest_idx)
            
            # 更新特征观测
            for feature_id in list(self.feature_observations.keys()):
                if second_newest_idx in self.feature_observations[feature_id]:
                    del self.feature_observations[feature_id][second_newest_idx]
                
                # 更新帧索引
                new_observations = {}
                for frame_idx, observation in self.feature_observations[feature_id].items():
                    if frame_idx > second_newest_idx:
                        new_observations[frame_idx - 1] = observation
                    else:
                        new_observations[frame_idx] = observation
                
                self.feature_observations[feature_id] = new_observations
                
                # 如果特征不再被观测，移除它
                if len(self.feature_observations[feature_id]) == 0:
                    del self.feature_observations[feature_id]
    
    def _huber_loss(self, error, delta):
        """
        Huber损失函数，用于降低离群值的影响
        
        参数:
            error: 误差值
            delta: 阈值
            
        返回:
            鲁棒化后的误差
        """
        if error <= delta:
            return 0.5 * error**2
        else:
            return delta * (error - 0.5 * delta)
