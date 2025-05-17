import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import time
from scipy.spatial.transform import Rotation as R

# 导入VIO系统
from vins_inspired.vio_system import VIOSystem

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

def process_movement_data(session_dir, movement_type, camera_matrix):
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
    
    # 创建VIO系统
    vio = VIOSystem(camera_matrix)
    
    # 处理数据
    result_frames = []
    trajectory = []
    
    for i, data_point in enumerate(synchronized_data):
        if i % 10 == 0:
            print(f"处理数据点 {i+1}/{len(synchronized_data)}")
        
        # 处理IMU和陀螺仪数据
        combined_imu = {
            'acc': data_point['imu']['values'],
            'gyro': data_point['gyro']['values']
        }
        
        vio.process_imu(combined_imu, data_point['timestamp'])
        
        # 处理视频帧
        result = vio.process_frame(data_point['frame'], data_point['timestamp'])
        
        # 可视化
        vis_frame = vio.visualize_current_frame()
        if vis_frame is not None:
            result_frames.append(vis_frame)
        
        # 记录轨迹
        if result['state'] is not None:
            trajectory.append((
                result['timestamp'],
                result['state']['position'],
                result['state']['rotation']
            ))
    
    # 保存结果
    results_dir = os.path.join(session_dir, movement_type, "vio_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存轨迹
    trajectory_file = os.path.join(results_dir, "trajectory.json")
    with open(trajectory_file, 'w') as f:
        json.dump([{
            'timestamp': float(t),
            'position': p.tolist(),
            'rotation': r.tolist()
        } for t, p, r in trajectory], f, indent=2)
    
    # 绘制轨迹
    if trajectory:
        positions = np.array([p for _, p, _ in trajectory])
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Estimated Trajectory')
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', marker='o', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', marker='o', s=100, label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Estimated Camera Trajectory - {movement_type}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "trajectory.png"))
    
    # 保存可视化视频
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
    print("===== VINS-Inspired VIO系统 =====")
    print("此程序将处理收集的数据并运行视觉惯性里程计算法")
    
    # 相机内参矩阵 (需要根据实际相机进行标定)
    # 这里使用一个估计值，实际应用中应该进行相机标定
    camera_matrix = np.array([
        [525.0, 0, 320.0],  # fx, 0, cx
        [0, 525.0, 240.0],  # 0, fy, cy
        [0, 0, 1.0]         # 0, 0, 1
    ])
    
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
        results[movement] = process_movement_data(latest_session, movement, camera_matrix)
    
    print("\n所有运动类型处理完成!")
    print("结果摘要:")
    for movement, result_dir in results.items():
        status = "成功" if result_dir else "失败"
        print(f"- {movement}: {status}")
    
    # 创建汇总报告
    report_file = os.path.join(latest_session, "vio_report.html")
    with open(report_file, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>VIO处理报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .movement {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>VIO处理报告</h1>
            <p>会话: {os.path.basename(latest_session)}</p>
            <p>处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>处理结果摘要</h2>
            <ul>
        """)
        
        for movement, result_dir in results.items():
            status_class = "success" if result_dir else "failure"
            status_text = "成功" if result_dir else "失败"
            f.write(f'<li><span class="{status_class}">{movement}: {status_text}</span></li>\n')
        
        f.write("</ul>\n")
        
        for movement, result_dir in results.items():
            if result_dir:
                trajectory_img = os.path.join(result_dir, "trajectory.png")
                trajectory_img_rel = os.path.relpath(trajectory_img, latest_session)
                
                f.write(f"""
                <div class="movement">
                    <h2>{movement}</h2>
                    <h3>轨迹</h3>
                    <img src="{trajectory_img_rel}" alt="{movement} 轨迹">
                    
                    <h3>视频</h3>
                    <p>可视化视频保存在: {os.path.join(result_dir, "vio_visualization.avi")}</p>
                </div>
                """)
        
        f.write("""
        </body>
        </html>
        """)
    
    print(f"\n汇总报告已保存到: {report_file}")
    print(f"请使用浏览器打开此文件查看详细结果")

if __name__ == "__main__":
    main()
