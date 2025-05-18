import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import numba

# 使用Numba加速的独立函数
@numba.jit(nopython=True)
def compute_reprojection_error(point_3d, point_2d, P):
    """
    计算重投影误差
    
    参数:
        point_3d: 3D点
        point_2d: 2D点
        P: 投影矩阵
        
    返回:
        重投影误差
    """
    # 投影3D点到图像平面
    point_3d_h = np.append(point_3d, 1.0)
    point_proj = P @ point_3d_h
    point_proj = point_proj[:2] / point_proj[2]
    
    # 计算误差
    error = np.linalg.norm(point_2d - point_proj)
    return error

@numba.jit(nopython=True)
def filter_triangulated_points(points_3d, pts1, pts2, P1, P2, max_error=10.0, min_depth=0.01):
    """
    过滤三角化点
    
    参数:
        points_3d: 三角化得到的3D点
        pts1: 第一帧中的特征点
        pts2: 第二帧中的特征点
        P1: 第一个相机的投影矩阵
        P2: 第二个相机的投影矩阵
        max_error: 最大重投影误差
        min_depth: 最小深度
        
    返回:
        有效的3D点和对应的索引
    """
    valid_points = []
    valid_indices = []
    
    for i in range(len(points_3d)):
        pt = points_3d[i]
        
        # 检查点是否在相机前方
        if pt[2] > min_depth:
            # 计算重投影误差
            err1 = compute_reprojection_error(pt, pts1[i], P1)
            err2 = compute_reprojection_error(pt, pts2[i], P2)
            
            # 检查重投影误差
            if err1 < max_error and err2 < max_error:
                valid_points.append(pt)
                valid_indices.append(i)
    
    return np.array(valid_points), np.array(valid_indices)

class VIOInitializer:
    """
    VIO初始化器
    负责系统的初始化，包括尺度、重力方向、速度和偏置的估计
    类似于VINS-Mono中的initial/initial_ex_rotation.h和initial_sfm.h
    """
    
    def __init__(self, camera_matrix, min_frames=3, min_features=15):
        """
        初始化VIO初始化器
        
        参数:
            camera_matrix: 相机内参矩阵
            min_frames: 初始化所需的最小帧数
            min_features: 初始化所需的最小特征点数量
        """
        self.K = camera_matrix
        
        # 初始化所需的最小帧数和特征点数量
        self.min_frames = 2  # 降低为2帧
        self.min_features = 8  # 降低为8个特征点
        
        # 初始化状态
        self.initialized = False
        
        # 存储初始化过程中的帧
        self.init_frames = []
        
        # 重力向量 (初始假设为z轴方向)
        self.gravity = np.array([0, 0, 9.81])
        
        # 尺度因子
        self.scale = 1.0
        
        # 初始偏置
        self.init_acc_bias = np.zeros(3)
        self.init_gyro_bias = np.zeros(3)
        
        # 初始速度
        self.init_velocity = np.zeros(3)
        
        # 初始姿态 (旋转矩阵)
        self.init_rotation = np.eye(3)
    
    def add_frame(self, frame_data):
        """
        添加一帧用于初始化
        
        参数:
            frame_data: 包含图像特征和IMU数据的帧
                {
                    'timestamp': 时间戳,
                    'features': {feature_id: [x, y, z?]}, # z可能不存在
                    'imu_data': [{'timestamp': ts, 'acc': [ax, ay, az], 'gyro': [gx, gy, gz]}, ...],
                }
            
        返回:
            是否已初始化成功
        """
        # 添加帧到初始化队列
        self.init_frames.append(frame_data)
        
        # 如果帧数不足，继续等待
        if len(self.init_frames) < self.min_frames:
            return False
        
        # 检查是否有足够的视差和跟踪点
        if not self._check_initialization_conditions():
            # 如果运动不足，移除最旧的帧
            if len(self.init_frames) > self.min_frames * 2:
                self.init_frames.pop(0)
            return False
        
        # 尝试初始化
        success = self._initialize()
        
        if success:
            self.initialized = True
        
        return success
    
    def _check_initialization_conditions(self):
        """
        检查是否满足初始化条件
        
        返回:
            满足条件返回True，否则返回False
        """
        # 检查帧数是否足够
        if len(self.init_frames) < self.min_frames:
            print(f"初始化条件检查: 帧数不足 ({len(self.init_frames)}/{self.min_frames})")
            return False
        
        # 检查特征点数量是否足够
        common_features = self._find_common_features()
        if len(common_features) < self.min_features:
            print(f"初始化条件检查: 共同特征点不足 ({len(common_features)}/{self.min_features})")
            return False
        
        print(f"初始化条件检查: 通过 (帧数: {len(self.init_frames)}, 共同特征点: {len(common_features)})")
        return True
    
    def _find_common_features(self):
        """
        找到两帧中共同的特征点
        
        返回:
            共同特征点列表
        """
        if len(self.init_frames) < 2:
            return []
        
        # 获取第一帧和最后一帧
        first_frame = self.init_frames[0]
        last_frame = self.init_frames[-1]
        
        # 找到两帧中共同的特征点
        common_features = {}
        for feat_id, feat_pos in first_frame['features'].items():
            if feat_id in last_frame['features']:
                common_features[feat_id] = (
                    np.array(feat_pos[:2]),  # 第一帧中的位置
                    np.array(last_frame['features'][feat_id][:2])  # 最后一帧中的位置
                )
        
        return common_features
    
    def _initialize(self):
        """
        执行VIO初始化
        
        返回:
            初始化是否成功
        """
        # 1. 首先进行纯视觉SFM初始化
        success, R_list, t_list, points_3d = self._visual_sfm()
        
        if not success:
            return False
        
        # 2. 对齐视觉SFM结果与IMU预积分结果
        success = self._align_visual_imu(R_list, t_list)
        
        if not success:
            return False
        
        # 3. 估计重力方向和尺度
        success = self._estimate_gravity_and_scale(R_list, t_list, points_3d)
        
        return success
    
    def _visual_sfm(self):
        """
        执行纯视觉SFM初始化
        
        返回:
            success: 初始化是否成功
            R_list: 旋转矩阵列表
            t_list: 平移向量列表
            points_3d: 3D点云
        """
        # 提取所有帧中的特征点
        frames_features = []
        for frame in self.init_frames:
            features = {}
            for feat_id, feat_pos in frame['features'].items():
                features[feat_id] = feat_pos[:2]  # 只使用x,y坐标
            frames_features.append(features)
        
        # 构建特征跟踪
        feature_tracks = {}
        for frame_idx, frame_feats in enumerate(frames_features):
            for feat_id, feat_pos in frame_feats.items():
                if feat_id not in feature_tracks:
                    feature_tracks[feat_id] = {}
                feature_tracks[feat_id][frame_idx] = feat_pos
        
        # 筛选出在多帧中出现的特征
        good_tracks = {}
        for feat_id, track in feature_tracks.items():
            if len(track) >= 3:  # 至少在3帧中出现
                good_tracks[feat_id] = track
        
        if len(good_tracks) < self.min_features:
            return False, None, None, None
        
        # 选择两帧进行初始化
        # 这里简单选择第一帧和最后一帧，实际应该选择视差最大的两帧
        frame_i = 0
        frame_j = len(self.init_frames) - 1
        
        # 找到两帧中共同的特征点
        common_points = []
        for feat_id, track in good_tracks.items():
            if frame_i in track and frame_j in track:
                pt_i = track[frame_i]
                pt_j = track[frame_j]
                common_points.append((feat_id, pt_i, pt_j))
        
        if len(common_points) < 8:  # 至少需要8个点来计算基础矩阵
            return False, None, None, None
        
        # 提取点坐标
        pts_i = np.array([pt_i for _, pt_i, _ in common_points])
        pts_j = np.array([pt_j for _, _, pt_j in common_points])
        
        # 计算基础矩阵
        F, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 1.0, 0.99)
        
        if F is None or F.shape != (3, 3):
            return False, None, None, None
        
        # 计算本质矩阵
        E = self.K.T @ F @ self.K
        
        # 从本质矩阵恢复R和t
        _, R, t, _ = cv2.recoverPose(E, pts_i, pts_j, self.K)
        
        # 初始化相机位姿列表
        R_list = [np.eye(3)] * len(self.init_frames)
        t_list = [np.zeros(3)] * len(self.init_frames)
        
        # 设置第一帧和第j帧的位姿
        R_list[frame_i] = np.eye(3)
        t_list[frame_i] = np.zeros(3)
        R_list[frame_j] = R
        t_list[frame_j] = t.flatten()
        
        # 三角化共同点
        points_3d = {}
        for idx, (feat_id, pt_i, pt_j) in enumerate(common_points):
            if mask[idx]:
                # 构建投影矩阵
                P_i = self.K @ np.hstack((R_list[frame_i], t_list[frame_i].reshape(3, 1)))
                P_j = self.K @ np.hstack((R_list[frame_j], t_list[frame_j].reshape(3, 1)))
                
                # 三角化
                pt_i_homo = np.append(pt_i, 1)
                pt_j_homo = np.append(pt_j, 1)
                
                A = np.zeros((4, 4))
                A[0] = pt_i_homo[0] * P_i[2] - P_i[0]
                A[1] = pt_i_homo[1] * P_i[2] - P_i[1]
                A[2] = pt_j_homo[0] * P_j[2] - P_j[0]
                A[3] = pt_j_homo[1] * P_j[2] - P_j[1]
                
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                X = X / X[3]  # 归一化
                
                points_3d[feat_id] = X[:3]
        
        # 使用PnP估计其他帧的位姿
        for frame_idx in range(len(self.init_frames)):
            if frame_idx != frame_i and frame_idx != frame_j:
                # 收集3D-2D对应点
                obj_points = []
                img_points = []
                
                for feat_id, point_3d in points_3d.items():
                    if feat_id in frames_features[frame_idx]:
                        obj_points.append(point_3d)
                        img_points.append(frames_features[frame_idx][feat_id])
                
                if len(obj_points) < 6:  # PnP至少需要6个点
                    continue
                
                # 使用PnP估计位姿
                obj_points = np.array(obj_points).astype(np.float32)
                img_points = np.array(img_points).astype(np.float32)
                
                _, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.K, None)
                
                # 转换旋转向量为旋转矩阵
                R_mat, _ = cv2.Rodrigues(rvec)
                
                R_list[frame_idx] = R_mat
                t_list[frame_idx] = tvec.flatten()
        
        return True, R_list, t_list, points_3d
    
    def _triangulate_features(self, P1, P2, pts1, pts2):
        """
        三角化特征点
        
        参数:
            P1: 第一个相机的投影矩阵
            P2: 第二个相机的投影矩阵
            pts1: 第一帧中的特征点
            pts2: 第二帧中的特征点
            
        返回:
            三角化后的3D点
        """
        # 转换为numpy数组以提高性能
        pts1_np = np.array([pt[0] for pt in pts1], dtype=np.float32)
        pts2_np = np.array([pt[0] for pt in pts2], dtype=np.float32)
        
        # 归一化坐标
        pts1_n = cv2.undistortPoints(pts1, self.K, None)
        pts2_n = cv2.undistortPoints(pts2, self.K, None)
        
        # 提取点坐标
        pts1_n_flat = pts1_n.reshape(-1, 2).T
        pts2_n_flat = pts2_n.reshape(-1, 2).T
        
        # 三角化
        points_4d = cv2.triangulatePoints(P1, P2, pts1_n_flat, pts2_n_flat)
        
        # 转换为3D点
        points_3d_h = points_4d.T
        points_3d = points_3d_h[:, :3] / points_3d_h[:, 3:4]
        
        # 使用Numba加速的函数过滤点
        valid_points, valid_indices = filter_triangulated_points(
            points_3d, pts1_np, pts2_np, P1, P2, 10.0, 0.01
        )
        
        if len(valid_points) > 0:
            print(f"三角化: 有效点数量 {len(valid_points)}/{len(points_3d)}")
        else:
            print(f"三角化: 没有有效点")
        
        # 重新格式化为原始格式
        valid_points_reshaped = np.array([[[x, y, z]] for x, y, z in valid_points])
        
        return valid_points_reshaped, valid_indices.tolist()
    
    def _align_visual_imu(self, R_list, t_list):
        """
        对齐视觉SFM结果与IMU预积分结果
        
        参数:
            R_list: 旋转矩阵列表
            t_list: 平移向量列表
            
        返回:
            对齐是否成功
        """
        # 这里简化实现，实际应该使用手眼标定方法
        # 假设IMU和相机坐标系已经对齐，只需要估计旋转偏移
        
        # 计算相邻帧之间的视觉旋转
        visual_rotations = []
        for i in range(1, len(R_list)):
            R_prev = R_list[i-1]
            R_curr = R_list[i]
            R_rel = R_curr @ R_prev.T
            visual_rotations.append(R_rel)
        
        # 计算相邻帧之间的IMU旋转
        imu_rotations = []
        for i in range(1, len(self.init_frames)):
            prev_frame = self.init_frames[i-1]
            curr_frame = self.init_frames[i]
            
            # 提取IMU数据
            imu_data = []
            for data in prev_frame['imu_data']:
                if data['timestamp'] <= curr_frame['timestamp']:
                    imu_data.append(data)
            
            # 使用IMU数据计算旋转
            R_imu = np.eye(3)
            for j in range(1, len(imu_data)):
                prev_data = imu_data[j-1]
                curr_data = imu_data[j]
                
                dt = curr_data['timestamp'] - prev_data['timestamp']
                # 获取陈螺仪数据
                gyro_data = curr_data['gyro']
                
                # 处理不同格式的陈螺仪数据
                if isinstance(gyro_data, dict):
                    if 'values' in gyro_data:
                        # 从字典的values字段中提取值
                        gyro_values = [float(gyro_data['values'].get(str(i), 0.0)) for i in range(3)]
                        gyro = np.array(gyro_values) - self.init_gyro_bias
                    elif all(str(i) in gyro_data for i in range(3)):
                        # 如果字典直接包含0,1,2索引
                        gyro_values = [float(gyro_data.get(str(i), 0.0)) for i in range(3)]
                        gyro = np.array(gyro_values) - self.init_gyro_bias
                    else:
                        # 如果是vio_system处理过的格式，直接使用
                        gyro = np.array(gyro_data) - self.init_gyro_bias
                elif isinstance(gyro_data, (list, np.ndarray)):
                    # 如果是数组，直接使用
                    gyro = np.array(gyro_data) - self.init_gyro_bias
                else:
                    # 如果是其他类型，尝试转换为数组
                    gyro = np.array([0.0, 0.0, 0.0])  # 默认值
                
                # 简单积分
                angle = np.linalg.norm(gyro) * dt
                if angle > 1e-10:
                    axis = gyro / np.linalg.norm(gyro)
                    r = R.from_rotvec(axis * angle)
                    R_step = r.as_matrix()
                    R_imu = R_step @ R_imu
            
            imu_rotations.append(R_imu)
        
        # 如果没有足够的旋转对，返回失败
        if len(visual_rotations) < 3 or len(imu_rotations) < 3:
            return False
        
        # 使用Kabsch算法估计旋转对齐
        # 将旋转矩阵转换为四元数或旋转向量进行平均
        q_visual = []
        q_imu = []
        
        for R_vis, R_imu in zip(visual_rotations, imu_rotations):
            r_vis = R.from_matrix(R_vis)
            r_imu = R.from_matrix(R_imu)
            
            q_visual.append(r_vis.as_quat())
            q_imu.append(r_imu.as_quat())
        
        q_visual = np.array(q_visual)
        q_imu = np.array(q_imu)
        
        # 简化对齐，实际应该使用优化方法
        # 这里假设对齐已经完成，使用单位旋转
        self.R_cam_imu = np.eye(3)
        
        return True
    
    def _estimate_gravity_and_scale(self, R_list, t_list, points_3d):
        """
        估计重力方向和尺度
        
        参数:
            R_list: 旋转矩阵列表
            t_list: 平移向量列表
            points_3d: 3D点云
            
        返回:
            估计是否成功
        """
        # 简化实现，实际应该使用优化方法
        
        # 1. 估计重力方向
        # 假设初始静止，使用加速度计读数作为重力方向
        gravity_samples = []
        for frame in self.init_frames[:3]:  # 使用前几帧
            for imu_data in frame['imu_data']:
                # 正确处理加速度计数据
                acc_data = imu_data['acc']
                if isinstance(acc_data, dict):
                    if 'values' in acc_data:
                        # 从字典的values字段中提取值
                        acc_values = [float(acc_data['values'].get(str(i), 0.0)) for i in range(3)]
                        gravity_samples.append(np.array(acc_values))
                    elif all(str(i) in acc_data for i in range(3)):
                        # 如果字典直接包含0,1,2索引
                        acc_values = [float(acc_data.get(str(i), 0.0)) for i in range(3)]
                        gravity_samples.append(np.array(acc_values))
                    else:
                        # 如果是vio_system处理过的格式，直接使用
                        gravity_samples.append(np.array(acc_data))
                elif isinstance(acc_data, (list, np.ndarray)):
                    # 如果是数组，直接使用
                    gravity_samples.append(np.array(acc_data))
                else:
                    # 如果是其他类型，尝试转换为数组
                    gravity_samples.append(np.array([0.0, 0.0, 0.0]))  # 默认值
        
        if len(gravity_samples) < 10:
            return False
        
        # 平均加速度作为重力方向
        gravity_dir = np.mean(gravity_samples, axis=0)
        gravity_norm = np.linalg.norm(gravity_dir)
        
        if gravity_norm < 8.0 or gravity_norm > 11.0:  # 检查是否接近9.8
            return False
        
        self.gravity = gravity_dir
        
        # 2. 估计尺度
        # 使用IMU位移和视觉位移的比例
        
        # 计算视觉位移
        visual_displacements = []
        for i in range(1, len(t_list)):
            disp = np.linalg.norm(t_list[i] - t_list[i-1])
            visual_displacements.append(disp)
        
        # 计算IMU位移
        imu_displacements = []
        for i in range(1, len(self.init_frames)):
            prev_frame = self.init_frames[i-1]
            curr_frame = self.init_frames[i]
            
            # 提取IMU数据
            imu_data = []
            for data in prev_frame['imu_data']:
                if data['timestamp'] <= curr_frame['timestamp']:
                    imu_data.append(data)
            
            # 双重积分计算位移
            dt_total = curr_frame['timestamp'] - prev_frame['timestamp']
            
            if dt_total <= 0 or len(imu_data) < 2:
                continue
            
            # 简化计算，假设加速度恒定
            acc_samples = []
            for data in imu_data:
                # 正确处理加速度计数据
                acc_data = data['acc']
                
                # 处理不同格式的加速度计数据
                if isinstance(acc_data, dict):
                    if 'values' in acc_data:
                        # 从字典的values字段中提取值
                        acc_values = [float(acc_data['values'].get(str(i), 0.0)) for i in range(3)]
                        acc = np.array(acc_values) - self.gravity  # 减去重力
                    elif all(str(i) in acc_data for i in range(3)):
                        # 如果字典直接包含0,1,2索引
                        acc_values = [float(acc_data.get(str(i), 0.0)) for i in range(3)]
                        acc = np.array(acc_values) - self.gravity  # 减去重力
                    else:
                        # 如果是vio_system处理过的格式，直接使用
                        acc = np.array(acc_data) - self.gravity  # 减去重力
                elif isinstance(acc_data, (list, np.ndarray)):
                    # 如果是数组，直接使用
                    acc = np.array(acc_data) - self.gravity  # 减去重力
                else:
                    # 如果是其他类型，尝试转换为数组
                    acc = np.array([0.0, 0.0, 0.0])  # 默认值
                
                acc_samples.append(acc)
            
            mean_acc = np.mean(acc_samples, axis=0)
            
            # s = 1/2 * a * t^2
            disp = 0.5 * np.linalg.norm(mean_acc) * dt_total * dt_total
            imu_displacements.append(disp)
        
        # 计算尺度比例
        if len(visual_displacements) < 3 or len(imu_displacements) < 3:
            return False
        
        # 使用中位数避免异常值影响
        ratios = []
        for vis_disp, imu_disp in zip(visual_displacements, imu_displacements):
            if vis_disp > 1e-6 and imu_disp > 1e-6:
                ratios.append(imu_disp / vis_disp)
        
        if len(ratios) < 3:
            return False
        
        self.scale = np.median(ratios)
        
        # 检查尺度是否合理
        if self.scale <= 0 or self.scale > 100:
            return False
        
        # 3. 估计初始速度和偏置
        # 简化实现，假设初始速度为零，偏置已在之前计算
        self.init_velocity = np.zeros(3)
        
        # 设置初始旋转
        self.init_rotation = R_list[0]
        
        return True
    
    def get_initialization_result(self):
        """
        获取初始化结果
        
        返回:
            初始化结果字典
        """
        if not self.initialized:
            return None
        
        return {
            'gravity': self.gravity,
            'scale': self.scale,
            'init_rotation': self.init_rotation,
            'init_velocity': self.init_velocity,
            'init_acc_bias': self.init_acc_bias,
            'init_gyro_bias': self.init_gyro_bias,
            'R_cam_imu': self.R_cam_imu
        }
