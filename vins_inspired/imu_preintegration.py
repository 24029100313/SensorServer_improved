import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUPreintegration:
    """
    IMU预积分模块
    负责处理IMU数据，执行预积分操作，类似于VINS-Mono中的initial/imu_factor模块
    """
    
    def __init__(self, acc_noise=0.01, gyro_noise=0.001, acc_bias_noise=0.0001, gyro_bias_noise=0.0001):
        """
        初始化IMU预积分
        
        参数:
            acc_noise: 加速度计噪声标准差
            gyro_noise: 陀螺仪噪声标准差
            acc_bias_noise: 加速度计偏置随机游走噪声
            gyro_bias_noise: 陀螺仪偏置随机游走噪声
        """
        # 噪声参数
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
        self.acc_bias_noise = acc_bias_noise
        self.gyro_bias_noise = gyro_bias_noise
        
        # 重力向量 (假设z轴向上)
        self.gravity = np.array([0, 0, 9.81])
        
        # 当前偏置估计
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        
        # 预积分状态
        self.reset()
    
    def reset(self):
        """重置预积分状态"""
        # 预积分值
        self.delta_p = np.zeros(3)  # 位置增量
        self.delta_v = np.zeros(3)  # 速度增量
        self.delta_q = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数增量 (w, x, y, z)
        
        # 协方差矩阵
        self.covariance = np.zeros((9, 9))  # [delta_p, delta_v, delta_theta]
        
        # 雅可比矩阵 (对偏置的导数)
        self.dp_dba = np.zeros((3, 3))  # 位置对加速度偏置的雅可比
        self.dp_dbg = np.zeros((3, 3))  # 位置对陀螺仪偏置的雅可比
        self.dv_dba = np.zeros((3, 3))  # 速度对加速度偏置的雅可比
        self.dv_dbg = np.zeros((3, 3))  # 速度对陀螺仪偏置的雅可比
        self.dq_dbg = np.zeros((3, 3))  # 姿态对陀螺仪偏置的雅可比
        
        # 积分时间
        self.sum_dt = 0
        
        # 存储积分过程中的所有IMU测量
        self.imu_measurements = []
    
    def integrate(self, acc, gyro, dt):
        """
        执行单步IMU预积分
        
        参数:
            acc: 加速度测量值 [ax, ay, az]
            gyro: 角速度测量值 [wx, wy, wz]
            dt: 时间间隔
        """
        # 存储IMU测量
        self.imu_measurements.append({
            'acc': acc.copy(),
            'gyro': gyro.copy(),
            'dt': dt
        })
        
        # 去除偏置
        acc_corrected = acc - self.acc_bias
        gyro_corrected = gyro - self.gyro_bias
        
        # 中点积分法
        # 首先进行半步积分得到中点估计
        half_dt = dt / 2.0
        
        # 角速度的模长
        gyro_norm = np.linalg.norm(gyro_corrected)
        
        # 避免除以零
        if gyro_norm > 1e-12:
            delta_theta_half = gyro_corrected * half_dt
            
            # 使用罗德里格斯公式计算旋转
            delta_q_half = self._small_rotation_quaternion(delta_theta_half)
            
            # 更新四元数
            delta_q_half = self._quaternion_multiply(self.delta_q, delta_q_half)
            
            # 从四元数获取旋转矩阵
            delta_R_half = self._quaternion_to_rotation_matrix(delta_q_half)
            
            # 计算中点加速度
            acc_mid = delta_R_half @ acc_corrected
            
            # 更新位置和速度
            self.delta_v += acc_mid * dt
            self.delta_p += self.delta_v * dt + 0.5 * acc_mid * dt * dt
            
            # 更新姿态
            delta_theta = gyro_corrected * dt
            delta_q_step = self._small_rotation_quaternion(delta_theta)
            self.delta_q = self._quaternion_multiply(self.delta_q, delta_q_step)
            
            # 归一化四元数
            self.delta_q = self.delta_q / np.linalg.norm(self.delta_q)
        else:
            # 如果角速度接近零，简化计算
            self.delta_v += self.delta_q @ acc_corrected * dt
            self.delta_p += self.delta_v * dt + 0.5 * self.delta_q @ acc_corrected * dt * dt
        
        # 更新雅可比矩阵 (简化版，完整版需要更复杂的推导)
        # 这里只是一个近似实现
        delta_R = self._quaternion_to_rotation_matrix(self.delta_q)
        
        # 更新位置对加速度偏置的雅可比
        self.dp_dba += self.dv_dba * dt - 0.5 * delta_R * dt * dt
        
        # 更新位置对陀螺仪偏置的雅可比
        self.dp_dbg += self.dv_dbg * dt
        
        # 更新速度对加速度偏置的雅可比
        self.dv_dba -= delta_R * dt
        
        # 更新速度对陀螺仪偏置的雅可比
        # 简化实现，实际应该考虑旋转对角速度偏置的影响
        self.dv_dbg -= delta_R @ self._skew_symmetric(acc_corrected) * dt
        
        # 更新姿态对陀螺仪偏置的雅可比
        self.dq_dbg -= delta_R * dt
        
        # 更新协方差矩阵 (简化版)
        # 实际应该使用状态转移矩阵和噪声协方差矩阵
        noise = np.zeros((6, 6))
        noise[:3, :3] = np.eye(3) * self.acc_noise * self.acc_noise * dt
        noise[3:, 3:] = np.eye(3) * self.gyro_noise * self.gyro_noise * dt
        
        # 简化的协方差更新
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt
        
        G = np.zeros((9, 6))
        G[:3, :3] = 0.5 * np.eye(3) * dt * dt
        G[3:6, :3] = np.eye(3) * dt
        G[6:, 3:] = np.eye(3) * dt
        
        self.covariance = F @ self.covariance @ F.T + G @ noise @ G.T
        
        # 更新总时间
        self.sum_dt += dt
    
    def reintegrate_with_new_bias(self, new_acc_bias, new_gyro_bias):
        """
        使用新的偏置重新进行预积分
        
        参数:
            new_acc_bias: 新的加速度计偏置
            new_gyro_bias: 新的陀螺仪偏置
            
        返回:
            新的预积分结果
        """
        # 保存旧的偏置
        old_acc_bias = self.acc_bias.copy()
        old_gyro_bias = self.gyro_bias.copy()
        
        # 设置新的偏置
        self.acc_bias = new_acc_bias
        self.gyro_bias = new_gyro_bias
        
        # 重置预积分状态
        self.reset()
        
        # 重新积分所有IMU测量
        for measurement in self.imu_measurements:
            self.integrate(measurement['acc'], measurement['gyro'], measurement['dt'])
        
        # 恢复旧的偏置
        self.acc_bias = old_acc_bias
        self.gyro_bias = old_gyro_bias
        
        return {
            'delta_p': self.delta_p.copy(),
            'delta_v': self.delta_v.copy(),
            'delta_q': self.delta_q.copy(),
            'covariance': self.covariance.copy()
        }
    
    def correct_with_bias_increment(self, delta_acc_bias, delta_gyro_bias):
        """
        使用偏置增量校正预积分结果
        
        参数:
            delta_acc_bias: 加速度计偏置增量
            delta_gyro_bias: 陀螺仪偏置增量
            
        返回:
            校正后的预积分结果
        """
        # 使用一阶泰勒展开校正
        corrected_delta_p = self.delta_p + self.dp_dba @ delta_acc_bias + self.dp_dbg @ delta_gyro_bias
        corrected_delta_v = self.delta_v + self.dv_dba @ delta_acc_bias + self.dv_dbg @ delta_gyro_bias
        
        # 校正四元数 (简化版)
        delta_theta = self.dq_dbg @ delta_gyro_bias
        delta_q_correction = self._small_rotation_quaternion(delta_theta)
        corrected_delta_q = self._quaternion_multiply(self.delta_q, delta_q_correction)
        
        return {
            'delta_p': corrected_delta_p,
            'delta_v': corrected_delta_v,
            'delta_q': corrected_delta_q
        }
    
    def _skew_symmetric(self, v):
        """
        计算向量的反对称矩阵
        
        参数:
            v: 3D向量
            
        返回:
            反对称矩阵
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def _small_rotation_quaternion(self, theta):
        """
        计算小旋转的四元数
        
        参数:
            theta: 旋转向量
            
        返回:
            四元数 [w, x, y, z]
        """
        theta_norm = np.linalg.norm(theta)
        
        if theta_norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        half_theta = theta_norm / 2.0
        axis = theta / theta_norm
        
        return np.array([
            np.cos(half_theta),
            axis[0] * np.sin(half_theta),
            axis[1] * np.sin(half_theta),
            axis[2] * np.sin(half_theta)
        ])
    
    def _quaternion_multiply(self, q1, q2):
        """
        四元数乘法
        
        参数:
            q1: 第一个四元数 [w, x, y, z]
            q2: 第二个四元数 [w, x, y, z]
            
        返回:
            乘积四元数 [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])
    
    def _quaternion_to_rotation_matrix(self, q):
        """
        四元数转旋转矩阵
        
        参数:
            q: 四元数 [w, x, y, z]
            
        返回:
            3x3旋转矩阵
        """
        w, x, y, z = q
        
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        ])
