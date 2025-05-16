#!/usr/bin/env python3
import asyncio
import websockets
import cv2
import numpy as np
import argparse
import io
from PIL import Image

async def receive_video_stream(uri):
    """
    接收视频流并显示
    
    参数:
    uri -- WebSocket服务器的URI，例如: ws://192.168.1.100:8086/video
    """
    print(f"正在连接到 {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("连接成功！正在接收视频流...")
            
            # 创建一个窗口来显示视频
            cv2.namedWindow("SensorServer Video Stream", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("SensorServer Video Stream", 640, 480)
            
            while True:
                try:
                    # 接收二进制数据（JPEG图像）
                    frame_data = await websocket.recv()
                    
                    # 将二进制数据转换为NumPy数组
                    image = Image.open(io.BytesIO(frame_data))
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # 显示图像
                    cv2.imshow("SensorServer Video Stream", frame)
                    
                    # 按下'q'键退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket连接已关闭")
                    break
                except Exception as e:
                    print(f"接收或处理帧时出错: {e}")
                    continue
            
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"连接到WebSocket服务器时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='SensorServer视频流客户端')
    parser.add_argument('--uri', type=str, default='ws://192.168.1.100:8086/video',
                        help='WebSocket服务器URI (默认: ws://192.168.1.100:8086/video)')
    
    args = parser.parse_args()
    
    # 运行异步WebSocket客户端
    asyncio.run(receive_video_stream(args.uri))

if __name__ == "__main__":
    main()
