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
    """处理特定运动类型的数据并运行VIO
    
    参数:
        session_dir: 会话目录
        movement_type: 运动类型
        camera_matrix: 相机内参矩阵
    """
    print(f"处理 {movement_type} 运动数据...")
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载数据
    imu_data, gyro_data, frames = load_data_from_session(session_dir, movement_type)
    
    print(f"加载了 {len(imu_data)} 个IMU数据点, {len(gyro_data)} 个陀螺仪数据点, {len(frames)} 帧视频")
    print(f"数据加载时间: {time.time() - start_time:.2f}秒")
    
    # 同步数据
    sync_start_time = time.time()
    data_points = synchronize_data(imu_data, gyro_data, frames)
    print(f"同步后有 {len(data_points)} 个数据点")
    print(f"数据同步时间: {time.time() - sync_start_time:.2f}秒")
    
    # 限制处理的数据点数量为25个
    max_points = 25
    if len(data_points) > max_points:
        print(f"限制处理数据点数量为前{max_points}个")
        data_points = data_points[:max_points]
    
    # 创建VIO系统
    vio = VIOSystem(camera_matrix)
    
    # 轨迹和可视化结果
    trajectory = []
    result_frames = []
    
    # 处理每个数据点
    process_start_time = time.time()
    processed_frames = 0
    
    for i, data_point in enumerate(data_points):
        frame_start_time = time.time()
        print(f"处理数据点 {i+1}/{len(data_points)}")
        
        try:
            # 组合IMU和陀螺仪数据
            combined_imu = {
                'acc': data_point['imu']['values'],
                'gyro': data_point['gyro']['values']
            }
            
            # 记录IMU处理时间
            imu_start_time = time.time()
            vio.process_imu(combined_imu, data_point['timestamp'])
            imu_time = time.time() - imu_start_time
            
            # 记录帧处理时间
            frame_process_start = time.time()
            result = vio.process_frame(data_point['frame'], data_point['timestamp'])
            frame_process_time = time.time() - frame_process_start
            
            # 添加调试信息：检查result的内容
            print(f"  结果状态: {'有效' if result['state'] is not None else '无效'}")
            if result['state'] is not None:
                print(f"  位置: {result['state']['position'].tolist()}")
                print(f"  旋转: {[round(x, 3) for x in R.from_matrix(result['state']['rotation']).as_euler('xyz', degrees=True).tolist()]}度")
                print(f"  速度: {result['state']['velocity'].tolist()}")
            else:
                print(f"  初始化状态: {vio.initialized}")
                print(f"  特征点数量: {len(result['features']) if 'features' in result else 0}")
            
            # 记录可视化时间
            vis_start_time = time.time()
            vis_frame = vio.visualize_current_frame()
            vis_time = time.time() - vis_start_time
            
            if vis_frame is not None:
                result_frames.append(vis_frame)
            
            # 记录轨迹
            if result['state'] is not None:
                trajectory.append((
                    result['timestamp'],
                    result['state']['position'],
                    result['state']['rotation']
                ))
                print(f"  已添加轨迹点，当前轨迹长度: {len(trajectory)}")
            else:
                print(f"  未添加轨迹点，当前轨迹长度: {len(trajectory)}")
            
            processed_frames += 1
            
            # 每10帧输出一次性能统计
            if i % 10 == 0 and i > 0:
                frame_time = time.time() - frame_start_time
                print(f"  性能统计 [帧 {i}]:")  
                print(f"  - IMU处理时间: {imu_time:.4f}秒")  
                print(f"  - 帧处理时间: {frame_process_time:.4f}秒")  
                print(f"  - 可视化时间: {vis_time:.4f}秒")  
                print(f"  - 总帧处理时间: {frame_time:.4f}秒")  
                print(f"  - 平均每帧时间: {(time.time() - process_start_time) / processed_frames:.4f}秒")  
                
        except Exception as e:
            print(f"处理数据点 {i+1} 时出错: {e}")
            # 继续处理下一个数据点
            continue
    
    # 计算总处理时间
    total_process_time = time.time() - process_start_time
    print(f"总处理时间: {total_process_time:.2f}秒")
    if processed_frames > 0:
        print(f"平均每帧处理时间: {total_process_time / processed_frames:.4f}秒")
    
    # 保存结果
    results_dir = os.path.join(session_dir, movement_type, "vio_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存轨迹
    trajectory_file = os.path.join(results_dir, "trajectory.json")
    print(f"\n轨迹信息:")
    print(f"  轨迹点数量: {len(trajectory)}")
    if len(trajectory) > 0:
        print(f"  第一个点: 时间戳={trajectory[0][0]}, 位置={trajectory[0][1].tolist()}")
        print(f"  最后一个点: 时间戳={trajectory[-1][0]}, 位置={trajectory[-1][1].tolist()}")
    else:
        print("  轨迹为空，检查以下可能原因:")
        print("  1. 系统未能成功初始化")
        print("  2. 处理数据点时出现错误")
        print("  3. IMU或视觉数据质量问题")
        print("  4. 特征点数量不足")
    
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
    print("注意: 每种运动类型只处理前25个数据点")
    
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
    valid_movement_types = ["x_positive_negative", "x_rotation", "y_positive_negative", "y_rotation", "z_positive_negative", "z_rotation"]
    movement_types = [d for d in os.listdir(latest_session) 
                     if os.path.isdir(os.path.join(latest_session, d)) and d in valid_movement_types]
    
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
