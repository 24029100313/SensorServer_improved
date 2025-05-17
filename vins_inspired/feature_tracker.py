import cv2
import numpy as np
from collections import defaultdict

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
        
        # 光流跟踪参数
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
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
    
    def detect_new_features(self, img, mask=None):
        """
        在图像中检测新的特征点
        
        参数:
            img: 输入图像
            mask: 掩码，指定不检测特征的区域
            
        返回:
            new_pts: 新检测到的特征点
        """
        if mask is None:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            
        # 如果已有特征点，在现有特征点周围创建掩码
        if self.curr_pts is not None and len(self.curr_pts) > 0:
            for pt in self.curr_pts:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(mask, (x, y), int(self.min_distance), 0, -1)
        
        # 检测新特征点
        new_pts = cv2.goodFeaturesToTrack(img, mask=mask, **self.feature_params)
        
        return new_pts if new_pts is not None else []
    
    def track_features(self, curr_img):
        """
        跟踪特征点从上一帧到当前帧
        
        参数:
            curr_img: 当前帧图像
            
        返回:
            curr_pts: 当前帧中的特征点
            ids: 特征点对应的ID
            tracks: 特征点轨迹
        """
        if self.prev_img is None:
            # 第一帧，初始化特征点
            self.prev_img = curr_img
            self.curr_pts = self.detect_new_features(curr_img)
            self.curr_ids = np.array([self.next_id + i for i in range(len(self.curr_pts))])
            self.next_id += len(self.curr_pts)
            
            # 初始化特征点生命周期和轨迹
            for i, pt in enumerate(self.curr_pts):
                self.feature_lifetime[self.curr_ids[i]] = 1
                self.feature_tracks[self.curr_ids[i]].append((pt[0][0], pt[0][1]))
            
            return self.curr_pts, self.curr_ids, self.feature_tracks
        
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
        status = status.flatten() & (fb_err < 1.0)
        
        # 过滤掉未成功跟踪的点
        curr_pts_filtered = []
        ids_filtered = []
        
        for i, (pt, id_, st) in enumerate(zip(next_pts, self.curr_ids, status)):
            if st:
                curr_pts_filtered.append(pt)
                ids_filtered.append(id_)
                
                # 更新特征点生命周期和轨迹
                self.feature_lifetime[id_] += 1
                self.feature_tracks[id_].append((pt[0][0], pt[0][1]))
        
        curr_pts_filtered = np.array(curr_pts_filtered).reshape(-1, 1, 2)
        ids_filtered = np.array(ids_filtered)
        
        # 如果特征点数量不足，检测新的特征点
        if len(curr_pts_filtered) < self.max_features * 0.5:
            mask = np.ones(curr_img.shape[:2], dtype=np.uint8) * 255
            
            # 在现有特征点周围创建掩码
            for pt in curr_pts_filtered:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(mask, (x, y), int(self.min_distance), 0, -1)
            
            # 检测新特征点
            new_pts = self.detect_new_features(curr_img, mask)
            
            if len(new_pts) > 0:
                new_ids = np.array([self.next_id + i for i in range(len(new_pts))])
                self.next_id += len(new_pts)
                
                # 初始化新特征点的生命周期和轨迹
                for i, pt in enumerate(new_pts):
                    self.feature_lifetime[new_ids[i]] = 1
                    self.feature_tracks[new_ids[i]].append((pt[0][0], pt[0][1]))
                
                # 合并新旧特征点
                curr_pts_filtered = np.vstack([curr_pts_filtered, new_pts])
                ids_filtered = np.concatenate([ids_filtered, new_ids])
        
        # 更新状态
        self.prev_img = curr_img
        self.curr_pts = curr_pts_filtered
        self.curr_ids = ids_filtered
        
        return self.curr_pts, self.curr_ids, self.feature_tracks
    
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
