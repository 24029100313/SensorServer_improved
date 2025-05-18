import websocket
import json
import os
import time
import threading
import cv2
import numpy as np
import base64
from datetime import datetime
import sys
import websocket  # 添加websocket库导入

# 配置参数
SERVER_IP = "192.168.43.132"  # 请修改为您的手机IP地址
SERVER_PORT = "8080"
VIDEO_PORT = "8080"  # 视频流服务端口

# 调试模式
DEBUG = False  # 设置为False以减少调试信息
DEBUG_DATA = False  # 是否显示数据内容
MOVEMENT_TYPES = [
    "x_positive_negative",  # x轴正向负向运动
    "y_positive_negative",  # y轴正向负向运动
    "z_positive_negative",  # z轴正向负向运动
    "x_rotation",           # 绕x轴旋转
    "y_rotation",           # 绕y轴旋转
    "z_rotation"            # 绕z轴旋转
]
RECORD_DURATION = 8  # 每种运动记录的秒数

# 运动描述
def get_movement_description(movement_type):
    descriptions = {
        "x_positive_negative": "请沿着手机X轴方向（左右）来回移动手机",
        "y_positive_negative": "请沿着手机Y轴方向（上下）来回移动手机",
        "z_positive_negative": "请沿着手机Z轴方向（前后）来回移动手机",
        "x_rotation": "请绕着手机X轴（左右）旋转手机",
        "y_rotation": "请绕着手机Y轴（上下）旋转手机",
        "z_rotation": "请绕着手机Z轴（平面内）旋转手机"
    }
    return descriptions.get(movement_type, "请按照指示移动手机")

# 创建数据存储目录
def create_directories():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "collected_data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, f"session_{timestamp}")
    
    # 创建基础目录
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 创建会话目录
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    
    # 为每种运动创建目录
    for movement in MOVEMENT_TYPES:
        movement_dir = os.path.join(session_dir, movement)
        
        # 创建IMU数据目录
        imu_dir = os.path.join(movement_dir, "imu_data")
        os.makedirs(imu_dir, exist_ok=True)
        
        # 创建陀螺仪数据目录
        gyro_dir = os.path.join(movement_dir, "gyro_data")
        os.makedirs(gyro_dir, exist_ok=True)
        
        # 创建视频帧目录
        frames_dir = os.path.join(movement_dir, "video_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # 创建元数据文件
        metadata = {
            "movement_type": movement,
            "description": get_movement_description(movement),
            "created_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(movement_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return session_dir

# 数据收集类
class DataCollector:
    def __init__(self, server_ip, server_port, session_dir):
        self.server_ip = server_ip
        self.server_port = server_port
        self.session_dir = session_dir
        self.current_movement = None
        self.is_recording = False
        self.imu_data = []
        self.gyro_data = []
        self.frame_count = 0
        self.start_time = 0
        
        # 创建锁以防止线程冲突
        self.lock = threading.Lock()
    
    def start_recording(self, movement_type):
        """开始记录特定类型的运动数据"""
        with self.lock:
            self.current_movement = movement_type
            self.is_recording = True
            self.imu_data = []
            self.gyro_data = []
            self.frame_count = 0
            self.start_time = time.time()
            print(f"\n开始记录 {movement_type} 运动数据...")
    
    def stop_recording(self):
        """停止记录并保存数据"""
        with self.lock:
            if not self.is_recording:
                return
                
            self.is_recording = False
            movement_dir = os.path.join(self.session_dir, self.current_movement)
            end_time = datetime.now().isoformat()
            
            # 保存IMU数据
            if self.imu_data:
                imu_file = os.path.join(movement_dir, "imu_data", "imu_data.json")
                imu_metadata = {
                    "data_type": "imu",
                    "movement": self.current_movement,
                    "record_start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "record_end_time": end_time,
                    "sample_count": len(self.imu_data)
                }
                
                # 保存IMU数据
                with open(imu_file, 'w') as f:
                    json.dump(self.imu_data, f, indent=2)
                    
                # 保存IMU元数据
                with open(os.path.join(movement_dir, "imu_data", "metadata.json"), 'w') as f:
                    json.dump(imu_metadata, f, indent=2)
                    
                print(f"已保存 {len(self.imu_data)} 条IMU数据记录到 {imu_file}")
            
            # 保存陀螺仪数据
            if self.gyro_data:
                gyro_file = os.path.join(movement_dir, "gyro_data", "gyro_data.json")
                gyro_metadata = {
                    "data_type": "gyroscope",
                    "movement": self.current_movement,
                    "record_start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                    "record_end_time": end_time,
                    "sample_count": len(self.gyro_data)
                }
                
                # 保存陀螺仪数据
                with open(gyro_file, 'w') as f:
                    json.dump(self.gyro_data, f, indent=2)
                    
                # 保存陀螺仪元数据
                with open(os.path.join(movement_dir, "gyro_data", "metadata.json"), 'w') as f:
                    json.dump(gyro_metadata, f, indent=2)
                    
                print(f"已保存 {len(self.gyro_data)} 条陀螺仪数据记录到 {gyro_file}")
            
            # 保存视频帧元数据
            frames_metadata = {
                "data_type": "video_frames",
                "movement": self.current_movement,
                "record_start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "record_end_time": end_time,
                "frame_count": self.frame_count,
                "format": "jpg"
            }
            
            with open(os.path.join(movement_dir, "video_frames", "metadata.json"), 'w') as f:
                json.dump(frames_metadata, f, indent=2)
            
            # 更新运动元数据
            movement_metadata = {
                "movement_type": self.current_movement,
                "description": get_movement_description(self.current_movement),
                "created_at": datetime.fromtimestamp(self.start_time).isoformat(),
                "completed_at": end_time,
                "imu_samples": len(self.imu_data),
                "gyro_samples": len(self.gyro_data),
                "video_frames": self.frame_count,
                "duration_seconds": time.time() - self.start_time
            }
            
            with open(os.path.join(movement_dir, "metadata.json"), 'w') as f:
                json.dump(movement_metadata, f, indent=2)
                
            print(f"已保存 {self.frame_count} 帧视频到 {os.path.join(movement_dir, 'video_frames')}")
            print(f"{self.current_movement} 运动数据记录完成")
            self.current_movement = None
    
    def record_imu_data(self, data):
        """记录IMU数据"""
        with self.lock:
            if self.is_recording and self.current_movement:
                # 处理数据格式，确保兼容VINS算法
                processed_data = {
                    'timestamp': time.time() - self.start_time
                }
                
                # 将字典格式的values转换为数组格式
                acc_values = [0.0, 0.0, 0.0]  # 默认值
                
                if 'values' in data:
                    # 从字典的values字段中提取值
                    for i in range(3):
                        acc_values[i] = float(data['values'].get(str(i), 0.0))
                elif all(str(i) in data for i in range(3)):
                    # 如果字典直接包含0,1,2索引
                    for i in range(3):
                        acc_values[i] = float(data.get(str(i), 0.0))
                elif 'x' in data and 'y' in data and 'z' in data:
                    # 如果字典包含x,y,z字段
                    acc_values[0] = float(data.get('x', 0.0))
                    acc_values[1] = float(data.get('y', 0.0))
                    acc_values[2] = float(data.get('z', 0.0))
                
                # 存储为数组格式，而不是字典
                processed_data['values'] = acc_values
                
                self.imu_data.append(processed_data)
    
    def record_gyro_data(self, data):
        """记录陀螺仪数据"""
        with self.lock:
            if self.is_recording and self.current_movement:
                # 处理数据格式，确保兼容VINS算法
                processed_data = {
                    'timestamp': time.time() - self.start_time
                }
                
                # 将字典格式的values转换为数组格式
                gyro_values = [0.0, 0.0, 0.0]  # 默认值
                
                if 'values' in data:
                    # 从字典的values字段中提取值
                    for i in range(3):
                        gyro_values[i] = float(data['values'].get(str(i), 0.0))
                elif all(str(i) in data for i in range(3)):
                    # 如果字典直接包含0,1,2索引
                    for i in range(3):
                        gyro_values[i] = float(data.get(str(i), 0.0))
                elif 'x' in data and 'y' in data and 'z' in data:
                    # 如果字典包含x,y,z字段
                    gyro_values[0] = float(data.get('x', 0.0))
                    gyro_values[1] = float(data.get('y', 0.0))
                    gyro_values[2] = float(data.get('z', 0.0))
                
                # 存储为数组格式，而不是字典
                processed_data['values'] = gyro_values
                
                self.gyro_data.append(processed_data)
    
    def record_video_frame(self, frame_data):
        """记录视频帧"""
        with self.lock:
            if self.is_recording and self.current_movement:
                try:
                    # 保存原始数据以便调试
                    movement_dir = os.path.join(self.session_dir, self.current_movement)
                    raw_dir = os.path.join(movement_dir, "video_frames", "raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    raw_path = os.path.join(raw_dir, f"frame_{self.frame_count:04d}_raw.bin")
                    
                    # 尝试不同的处理方法
                    success = False
                    
                    # 方法1: 假设是直接的JPEG数据
                    try:
                        import cv2
                        import numpy as np
                        
                        # 如果是字符串，尝试Base64解码
                        binary_data = frame_data
                        if isinstance(frame_data, str):
                            import base64
                            try:
                                binary_data = base64.b64decode(frame_data)
                            except:
                                print(f"警告: 无法进行Base64解码")
                        
                        # 保存原始数据
                        with open(raw_path, 'wb') as f:
                            if isinstance(binary_data, str):
                                f.write(binary_data.encode('utf-8'))
                            else:
                                f.write(binary_data)
                        
                        # 尝试直接解码为图像
                        nparr = np.frombuffer(binary_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None and img.size > 0:
                            # 成功解码，保存为JPEG
                            frame_path = os.path.join(movement_dir, "video_frames", f"frame_{self.frame_count:04d}.jpg")
                            cv2.imwrite(frame_path, img)
                            success = True
                            print(f"成功保存视频帧 {self.frame_count}, 大小: {img.shape}")
                    except Exception as e:
                        print(f"解码视频帧时出错(方法1): {e}")
                    
                    # 如果方法1失败，尝试方法2: 创建一个空白图像作为占位符
                    if not success:
                        try:
                            # 创建一个空白图像
                            img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # 白色图像
                            # 添加帧编号文本
                            cv2.putText(img, f"Frame {self.frame_count}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            # 保存为JPEG
                            frame_path = os.path.join(movement_dir, "video_frames", f"frame_{self.frame_count:04d}.jpg")
                            cv2.imwrite(frame_path, img)
                            success = True
                            print(f"创建占位图像作为视频帧 {self.frame_count}")
                        except Exception as e:
                            print(f"创建占位图像时出错: {e}")
                    
                    self.frame_count += 1
                    
                except Exception as e:
                    print(f"处理视频帧时出错: {e}")
                    self.frame_count += 1
    
    def connect_to_imu(self):
        """连接到IMU传感器"""
        ws_url = f"ws://{self.server_ip}:{self.server_port}/sensor/connect?type=android.sensor.accelerometer"
        
        if DEBUG:
            print(f"尝试连接到IMU传感器: {ws_url}")
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if DEBUG_DATA:
                    print(f"收到IMU数据: {data}")
                self.record_imu_data(data)
            except Exception as e:
                print(f"处理IMU数据时出错: {e}")
        
        def on_error(ws, error):
            print(f"IMU连接错误: {error}")
        
        def on_close(ws, close_code, reason):
            print(f"IMU连接关闭: {reason}")
        
        def on_open(ws):
            print("已成功连接到IMU传感器")
        
        ws = websocket.WebSocketApp(ws_url,
                                   on_open=on_open,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)
        
        # 设置连接超时时间
        websocket.setdefaulttimeout(10)  # 10秒超时
        
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
        thread.start()
        
        if DEBUG:
            print(f"IMU连接线程已启动")
        
        return ws
    
    def connect_to_gyro(self):
        """连接到陀螺仪传感器"""
        ws_url = f"ws://{self.server_ip}:{self.server_port}/sensor/connect?type=android.sensor.gyroscope"
        
        if DEBUG:
            print(f"尝试连接到陀螺仪传感器: {ws_url}")
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if DEBUG_DATA:
                    print(f"收到陀螺仪数据: {data}")
                self.record_gyro_data(data)
            except Exception as e:
                print(f"处理陀螺仪数据时出错: {e}")
        
        def on_error(ws, error):
            print(f"陀螺仪连接错误: {error}")
        
        def on_close(ws, close_code, reason):
            print(f"陀螺仪连接关闭: {reason}")
        
        def on_open(ws):
            print("已成功连接到陀螺仪传感器")
        
        ws = websocket.WebSocketApp(ws_url,
                                   on_open=on_open,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)
        
        # 设置连接超时时间
        websocket.setdefaulttimeout(10)  # 10秒超时
        
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
        thread.start()
        
        if DEBUG:
            print(f"陀螺仪连接线程已启动")
        
        return ws
    
    def connect_to_video(self):
        """连接到视频流"""
        ws_url = f"ws://{self.server_ip}:{VIDEO_PORT}/video"
        print(f"\n正在连接到视频流: {ws_url}")
        
        if DEBUG:
            print(f"尝试连接到视频流: {ws_url}")
        
        def on_message(ws, message):
            try:
                if DEBUG_DATA:
                    print(f"收到视频帧，大小: {len(message) if message else 0} 字节")
                self.record_video_frame(message)
            except Exception as e:
                print(f"处理视频帧时出错: {e}")
        
        def on_error(ws, error):
            print(f"视频连接错误: {error}")
        
        def on_close(ws, close_code, reason):
            print(f"视频连接关闭: {reason}")
        
        def on_open(ws):
            print("已成功连接到视频流")
        
        ws = websocket.WebSocketApp(ws_url,
                                   on_open=on_open,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)
        
        # 设置连接超时时间
        websocket.setdefaulttimeout(10)  # 10秒超时
        
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
        thread.start()
        
        if DEBUG:
            print(f"视频连接线程已启动")
        
        return ws

def main():
    print("===== 手机传感器数据自动采集工具 =====")
    print("此工具将自动记录手机在不同运动模式下的IMU数据、陀螺仪数据和视频帧")
    
    # 创建目录结构
    session_dir = create_directories()
    print(f"数据将保存到: {session_dir}")
    
    # 创建会话元数据文件
    session_metadata = {
        "session_id": os.path.basename(session_dir),
        "start_time": datetime.now().isoformat(),
        "movements": MOVEMENT_TYPES,
        "record_duration_per_movement": RECORD_DURATION,
        "server_ip": SERVER_IP,
        "server_port": SERVER_PORT,
        "system_info": {
            "python_version": sys.version,
            "os": sys.platform
        }
    }
    
    with open(os.path.join(session_dir, "session_metadata.json"), 'w') as f:
        json.dump(session_metadata, f, indent=2)
    
    print("\n请确保在手机上同时启动以下两个服务:")
    print("1. 主界面上点击'Start'启动传感器服务(端口{})" .format(SERVER_PORT))
    print("2. 在导航菜单中点击'Video Stream'并启动视频流服务(端口{})" .format(VIDEO_PORT))
    input("\n当两个服务都启动后，按Enter键继续...")
    
    # 创建数据采集器
    collector = DataCollector(SERVER_IP, SERVER_PORT, session_dir)
    
    print("\n正在连接到传感器...")
    imu_ws = collector.connect_to_imu()
    gyro_ws = collector.connect_to_gyro()
    
    print("\n正在连接到视频流...")
    video_ws = collector.connect_to_video()
    
    # 等待连接建立
    time.sleep(2)
    
    # 检查连接状态
    if not imu_ws or not gyro_ws:
        print("\n警告: 无法连接到传感器服务。请确保在主界面上点击了'Start'按钮。")
        input("按Enter键继续尝试...")
        
    if not video_ws:
        print("\n警告: 无法连接到视频流服务。请确保在'Video Stream'页面上点击了'Start'按钮。")
        input("按Enter键继续尝试...")
    
    try:
        # 对每种运动类型进行记录
        for movement in MOVEMENT_TYPES:
            print(f"\n准备记录 '{movement}' 运动:")
            print(get_movement_description(movement))
            input("按Enter键开始记录...")
            collector.start_recording(movement)
            
            # 倒计时
            for i in range(RECORD_DURATION, 0, -1):
                print(f"记录中... {i}秒", end="\r")
                time.sleep(1)
            
            collector.stop_recording()
            print(f"'{movement}' 运动记录完成!")
        
        # 更新会话元数据
        session_metadata["end_time"] = datetime.now().isoformat()
        session_metadata["status"] = "completed"
        session_metadata["completed_movements"] = MOVEMENT_TYPES
        
        with open(os.path.join(session_dir, "session_metadata.json"), 'w') as f:
            json.dump(session_metadata, f, indent=2)
        
        print("\n所有运动类型数据采集完成!")
        print(f"数据已保存到: {session_dir}")
        
    except KeyboardInterrupt:
        print("\n用户中断，停止记录...")
        collector.stop_recording()
        
        # 更新会话元数据，记录中断状态
        session_metadata["end_time"] = datetime.now().isoformat()
        session_metadata["status"] = "interrupted"
        completed_movements = MOVEMENT_TYPES[:MOVEMENT_TYPES.index(collector.current_movement)] if collector.current_movement else []
        session_metadata["completed_movements"] = completed_movements
        
        with open(os.path.join(session_dir, "session_metadata.json"), 'w') as f:
            json.dump(session_metadata, f, indent=2)
    finally:
        # 关闭所有连接
        print("\n关闭连接...")
        imu_ws.close()
        gyro_ws.close()
        video_ws.close()
        print("程序结束")

if __name__ == "__main__":
    main()
