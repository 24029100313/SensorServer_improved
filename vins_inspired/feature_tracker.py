import cv2
import numpy as np
from collections import defaultdict
import time
import threading

class FeatureTracker:
    """
    特征点跟踪器
    负责检测和跟踪图像中的特征点，类似于VINS-Mono中的feature_tracker模块
    """
    
    def __init__(self, max_features=150, min_distance=30.0, quality_level=0.01):
        """
        初始化特征跟踪器
        
        参数:
            max_features: 最大特征点数量
            min_distance: 特征点之间的最小欧氏距离
            quality_level: 特征点质量水平阈值
        """
        self.max_features = max_features
        self.min_distance = min_distance
        self.quality_level = quality_level
        
        # 特征点检测参数
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        
        # 光流跟踪参数 - 增大窗口大小和金字塔层数，以应对旋转运动
        self.lk_params = dict(
            winSize=(21, 21),  # 减小窗口大小以提高速度
            maxLevel=3,        # 减少金字塔层数以提高速度
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
            minEigThreshold=0.001
        )
        
        # 存储当前帧的特征点和ID
        self.curr_pts = None
        self.curr_ids = None
        self.next_id = 0
        
        # 存储当前帧的图像
        self.prev_img = None
        
        # 特征点生命周期跟踪
        self.feature_lifetime = defaultdict(int)
        
        # 特征点轨迹 (用于可视化和后端优化)
        self.feature_tracks = defaultdict(list)
        
        # 预分配缓冲区
        self.status_buffer = None
        self.fb_err_buffer = None
        
        # 性能统计
        self.detection_time = 0
        self.tracking_time = 0
        self.total_frames = 0
        
        # 使用ORB检测器作为备选
        self.orb = cv2.ORB_create(nfeatures=max_features, 
                                  scaleFactor=1.2, 
                                  nlevels=8,
                                  edgeThreshold=31)
    
    def detect_new_features(self, img, mask=None):
        """
        检测新的特征点
        
        参数:
            img: 输入图像
            mask: 掩码，指定不检测特征的区域
            
        返回:
            新的特征点
        """
        start_time = time.time()
        
        # 创建掩码，如果没有提供
        if mask is None:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        
        # 如果已有特征点，在现有特征点周围创建掩码
        if self.curr_pts is not None and len(self.curr_pts) > 0:
            for pt in self.curr_pts:
                x, y = int(pt[0][0]), int(pt[0][1])
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    cv2.circle(mask, (x, y), int(self.min_distance), 0, -1)
        
        # 使用Shi-Tomasi角点检测器
        new_pts = cv2.goodFeaturesToTrack(img, mask=mask, **self.feature_params)
        
        # 如果检测到的特征点过少，使用ORB检测器
        if new_pts is None or len(new_pts) < self.max_features // 2:
            keypoints = self.orb.detect(img, mask)
            if keypoints:
                # 将关键点转换为适合光流跟踪的格式
                new_pts_orb = np.array([[kp.pt] for kp in keypoints], dtype=np.float32)
                
                # 如果原来有一些特征点，合并结果
                if new_pts is not None and len(new_pts) > 0:
                    new_pts = np.vstack((new_pts, new_pts_orb))
                else:
                    new_pts = new_pts_orb
        
        # 如果仍然没有检测到特征点，创建一些人工特征点
        if new_pts is None or len(new_pts) == 0:
            h, w = img.shape
            grid_size = 50
            artificial_pts = []
            for x in range(grid_size, w, grid_size):
                for y in range(grid_size, h, grid_size):
                    artificial_pts.append([[float(x), float(y)]])
            new_pts = np.array(artificial_pts, dtype=np.float32)
        
        # 限制特征点数量
        if len(new_pts) > self.max_features:
            # 随机选择特征点，以确保分布均匀
            indices = np.random.choice(len(new_pts), self.max_features, replace=False)
            new_pts = new_pts[indices]
        
        self.detection_time += time.time() - start_time
        return new_pts
    
    def track_features(self, curr_img):
        """
        跟踪特征点
        
        参数:
            curr_img: 当前帧图像
            
        返回:
            curr_pts: 当前帧特征点
            curr_ids: 当前帧特征点ID
            feature_tracks: 特征点轨迹
        """
        self.total_frames += 1
        
        # 第一次调用，初始化
        if self.prev_img is None:
            start_time = time.time()
            
            # 第一帧，初始化特征点
            self.prev_img = curr_img.copy()  # 使用copy避免引用问题
            self.curr_pts = self.detect_new_features(curr_img)
            
            # 确保检测到了特征点
            if self.curr_pts is None or len(self.curr_pts) == 0:
                # 如果没有检测到特征点，尝试降低要求
                self.feature_params['qualityLevel'] = self.feature_params['qualityLevel'] / 2
                self.curr_pts = self.detect_new_features(curr_img)
                
                # 如果仍然没有特征点，创建一些人工特征点
                if self.curr_pts is None or len(self.curr_pts) == 0:
                    h, w = curr_img.shape
                    grid_size = 50
                    artificial_pts = []
                    for x in range(grid_size, w, grid_size):
                        for y in range(grid_size, h, grid_size):
                            artificial_pts.append([[float(x), float(y)]])
                    self.curr_pts = np.array(artificial_pts, dtype=np.float32)
            
            self.curr_ids = np.array([self.next_id + i for i in range(len(self.curr_pts))])
            self.next_id += len(self.curr_pts)
            
            # 初始化特征点生命周期和轨迹
            for i, pt in enumerate(self.curr_pts):
                self.feature_lifetime[self.curr_ids[i]] = 1
                self.feature_tracks[self.curr_ids[i]].append((pt[0][0], pt[0][1]))
            
            self.detection_time += time.time() - start_time
            return self.curr_pts, self.curr_ids, self.feature_tracks
        
        # 确保有足够的特征点进行跟踪
        if self.curr_pts is None or len(self.curr_pts) == 0:
            # 如果没有特征点，重新初始化
            self.prev_img = None
            return self.track_features(curr_img)
        
        start_time = time.time()
        
        try:
            # 预分配缓冲区
            if self.status_buffer is None or len(self.status_buffer) != len(self.curr_pts):
                self.status_buffer = np.zeros(len(self.curr_pts), dtype=np.uint8)
                self.fb_err_buffer = np.zeros(len(self.curr_pts), dtype=np.float32)
            
            # 使用LK光流法跟踪特征点
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_img, curr_img, self.curr_pts, None, **self.lk_params
            )
            
            # 反向跟踪验证
            prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                curr_img, self.prev_img, next_pts, None, **self.lk_params
            )
            
            # 计算前向-后向误差
            fb_err = np.linalg.norm(self.curr_pts - prev_pts_back, axis=2)
            # 确保status是一维布尔数组
            status_flat = status.flatten()
            status_bool = np.zeros(len(status_flat), dtype=bool)
            for i in range(len(status_flat)):
                status_bool[i] = bool(status_flat[i]) and fb_err[i] < 3.0
            
            # 过滤掉未成功跟踪的点
            curr_pts_filtered = []
            ids_filtered = []
            
            # 使用索引访问，确保是标量布尔值
            for i in range(len(next_pts)):
                if i < len(status_bool) and status_bool[i]:
                    pt = next_pts[i]
                    id_ = self.curr_ids[i]
                    
                    # 检查点是否在图像边界内
                    x, y = pt[0]
                    if 0 <= x < curr_img.shape[1] and 0 <= y < curr_img.shape[0]:
                        curr_pts_filtered.append(pt)
                        ids_filtered.append(id_)
                        
                        # 更新特征点生命周期和轨迹
                        self.feature_lifetime[id_] += 1
                        self.feature_tracks[id_].append((x, y))
            
            # 更新当前特征点和ID
            if curr_pts_filtered:
                self.curr_pts = np.array(curr_pts_filtered, dtype=np.float32)
                self.curr_ids = np.array(ids_filtered)
            else:
                self.curr_pts = None
                self.curr_ids = None
            
            # 如果特征点数量不足，检测新的特征点
            if self.curr_pts is None or len(self.curr_pts) < self.max_features // 2:
                # 创建掩码，避免在现有特征点附近检测新点
                mask = np.ones(curr_img.shape[:2], dtype=np.uint8) * 255
                if self.curr_pts is not None and len(self.curr_pts) > 0:
                    for pt in self.curr_pts:
                        x, y = int(pt[0][0]), int(pt[0][1])
                        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                            cv2.circle(mask, (x, y), int(self.min_distance), 0, -1)
                
                # 检测新的特征点
                new_pts = self.detect_new_features(curr_img, mask)
                
                if new_pts is not None and len(new_pts) > 0:
                    # 为新特征点分配ID
                    new_ids = np.array([self.next_id + i for i in range(len(new_pts))])
                    self.next_id += len(new_pts)
                    
                    # 初始化新特征点的生命周期和轨迹
                    for i, pt in enumerate(new_pts):
                        self.feature_lifetime[new_ids[i]] = 1
                        self.feature_tracks[new_ids[i]].append((pt[0][0], pt[0][1]))
                    
                    # 合并新旧特征点
                    if self.curr_pts is not None and len(self.curr_pts) > 0:
                        self.curr_pts = np.vstack((self.curr_pts, new_pts))
                        self.curr_ids = np.hstack((self.curr_ids, new_ids))
                    else:
                        self.curr_pts = new_pts
                        self.curr_ids = new_ids
            
            # 更新上一帧图像
            self.prev_img = curr_img.copy()
            
            self.tracking_time += time.time() - start_time
            
            # 每100帧输出一次性能统计
            if self.total_frames % 100 == 0:
                avg_detection_time = self.detection_time / max(1, self.total_frames)
                avg_tracking_time = self.tracking_time / max(1, self.total_frames)
                print(f"特征跟踪性能统计:")
                print(f"  平均检测时间: {avg_detection_time:.4f}秒/帧")
                print(f"  平均跟踪时间: {avg_tracking_time:.4f}秒/帧")
                print(f"  平均总时间: {(avg_detection_time + avg_tracking_time):.4f}秒/帧")
            
            return self.curr_pts, self.curr_ids, self.feature_tracks
            
        except cv2.error as e:
            # 如果光流跟踪失败，重新初始化
            print(f"光流跟踪失败，重新初始化: {e}")
            self.prev_img = None
            return self.track_features(curr_img)
    
    def draw_tracks(self, img, max_length=10):
        """
        在图像上绘制特征点轨迹
        
        参数:
            img: 输入图像
            max_length: 轨迹最大长度
            
        返回:
            vis_img: 带有特征点轨迹的可视化图像
        """
        vis_img = img.copy()
        
        if self.curr_pts is None or len(self.curr_pts) == 0:
            return vis_img
        
        # 绘制当前特征点
        for i, (pt, id_) in enumerate(zip(self.curr_pts, self.curr_ids)):
            x, y = int(pt[0][0]), int(pt[0][1])
            cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)
            
            # 绘制特征点ID
            cv2.putText(vis_img, str(id_), (x + 3, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
            # 绘制轨迹
            track = self.feature_tracks[id_]
            if len(track) > 1:
                # 只绘制最近的max_length个点
                track = track[-max_length:] if len(track) > max_length else track
                
                for j in range(len(track) - 1):
                    pt1 = (int(track[j][0]), int(track[j][1]))
                    pt2 = (int(track[j+1][0]), int(track[j+1][1]))
                    cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)
        
        return vis_img
    
    def reset(self):
        """重置特征跟踪器"""
        self.curr_pts = None
        self.curr_ids = None
        self.prev_img = None
        self.next_id = 0
        self.feature_lifetime.clear()
        self.feature_tracks.clear()
