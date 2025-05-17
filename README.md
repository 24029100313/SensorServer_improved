<div align="center">

# SensorServer Improved
![GitHub](https://img.shields.io/github/license/umer0586/SensorServer?style=for-the-badge) ![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/umer0586/SensorServer?style=for-the-badge) ![GitHub all releases](https://img.shields.io/github/downloads/umer0586/SensorServer/total?label=GitHub%20downloads&style=for-the-badge) ![Android](https://img.shields.io/badge/Android%205.0+-3DDC84?style=for-the-badge&logo=android&logoColor=white) ![F-Droid](https://img.shields.io/f-droid/v/github.umer0586.sensorserver?style=for-the-badge) ![Websocket](https://img.shields.io/badge/protocol-websocket-green?style=for-the-badge)

[<img src="https://github.com/user-attachments/assets/0f628053-199f-4587-a5b2-034cf027fb99" height="100">](https://github.com/umer0586/SensorServer/releases) [<img src="https://fdroid.gitlab.io/artwork/badge/get-it-on.png"
    alt="Get it on F-Droid"
    height="100">](https://f-droid.org/packages/github.umer0586.sensorserver)

 
### SensorServer Improved transforms Android device into a versatile sensor hub, providing real-time access to its entire array of sensors AND camera video stream. It allows multiple Websocket clients to simultaneously connect and retrieve live sensor data and video feed. The app exposes all available sensors of the Android device, enabling WebSocket clients to read sensor data related to device position, motion (e.g., accelerometer, gyroscope), environment (e.g., temperature, light, pressure), GPS location, touchscreen interactions, and now real-time camera video stream.

<img src="https://github.com/umer0586/SensorServer/blob/main/fastlane/metadata/android/en-US/images/phoneScreenshots/01.png" width="250" heigth="250"> <img src="https://github.com/umer0586/SensorServer/blob/main/fastlane/metadata/android/en-US/images/phoneScreenshots/02.png" width="250" heigth="250"> <img src="https://github.com/umer0586/SensorServer/blob/main/fastlane/metadata/android/en-US/images/phoneScreenshots/03.png" width="250" heigth="250"> <img src="https://github.com/umer0586/SensorServer/blob/main/fastlane/metadata/android/en-US/images/phoneScreenshots/04.png" width="250" heigth="250"> <img src="https://github.com/umer0586/SensorServer/blob/main/fastlane/metadata/android/en-US/images/phoneScreenshots/05.png" width="250" heigth="250"> <img src="https://github.com/umer0586/SensorServer/blob/main/fastlane/metadata/android/en-US/images/phoneScreenshots/06.png" width="250" heigth="250">
<img src="https://github.com/umer0586/SensorServer/blob/main/fastlane/metadata/android/en-US/images/phoneScreenshots/07.png" width="250" heigth="250">

</div>



Since this application functions as a Websocket Server, you will require a Websocket Client API to establish a connection with the application. To obtain a Websocket library for your preferred programming language click [here](https://github.com/facundofarias/awesome-websockets). 
 
 
 
 
 # Usage
 To receive sensor data, **Websocket client**  must connect to the app using following **URL**.
 
                 ws://<ip>:<port>/sensor/connect?type=<sensor type here> 
 
 
  Value for the `type` parameter can be found by navigating to **Available Sensors** in the app. 
 
 For example
 
 * For **accelerometer** `/sensor/connect?type=android.sensor.accelerometer` .
 
 * For **gyroscope** `/sensor/connect?type=android.sensor.gyroscope` .
 
 * For **step detector**  `/sensor/connect?type=android.sensor.step_detector`

 * so on... 
 
 Once connected, client will receive sensor data in `JSON Array` (float type values) through `websocket.onMessage`. 
 
 A snapshot from accelerometer.
 
 ```json
{
  "accuracy": 2,
  "timestamp": 3925657519043709,
  "values": [0.31892395,-0.97802734,10.049896]
}
 ```
![axis_device](https://user-images.githubusercontent.com/35717992/179351418-bf3b511a-ebea-49bb-af65-5afd5f464e14.png)

where

| Array Item  | Description |
| ------------- | ------------- |
| values[0]  | Acceleration force along the x axis (including gravity)  |
| values[1]  | Acceleration force along the y axis (including gravity)  |
| values[2]  | Acceleration force along the z axis (including gravity)  |

And [timestamp](https://developer.android.com/reference/android/hardware/SensorEvent#timestamp) is the time in nanoseconds at which the event happened

Use `JSON` parser to get these individual values.

 
**Note** : *Use  following links to know what each value in **values** array corresponds to*
- For motion sensors [/topics/sensors/sensors_motion](https://developer.android.com/guide/topics/sensors/sensors_motion)
- For position sensors [/topics/sensors/sensors_position](https://developer.android.com/guide/topics/sensors/sensors_position)
- For Environmental sensors [/topics/sensors/sensors_environment](https://developer.android.com/guide/topics/sensors/sensors_environment)

## Undocumented (mostly QTI) sensors on Android devices
Some Android devices have additional sensors like **Coarse Motion Classifier** `(com.qti.sensor.motion_classifier)`, **Basic Gesture** `(com.qti.sensor.basic_gestures)` etc  which are not documented on offical android docs. Please refer to this [Blog](https://louis993546.medium.com/quick-tech-support-undocumented-mostly-qti-sensors-on-android-devices-d7e2fb6c5064) for corresponding values in `values` array  

## Supports multiple connections to multiple sensors simultaneously

Multiple WebSocket clients can connect to a specific type of sensor. For example, by connecting to `/sensor/connect?type=android.sensor.accelerometer` multiple times, separate connections to the accelerometer sensor are created. Each connected client will receive accelerometer data simultaneously.

Additionally, it is possible to connect to different types of sensors from either the same or different machines. For instance, one WebSocket client object can connect to the accelerometer, while another WebSocket client object can connect to the gyroscope. To view all active connections, you can select the "Connections" navigation button.
 
## Example: Websocket client (Python) 
Here is a simple websocket client in python using [websocket-client api](https://github.com/websocket-client/websocket-client) which receives live data from accelerometer sensor.

```python
import websocket
import json


def on_message(ws, message):
    values = json.loads(message)['values']
    x = values[0]
    y = values[1]
    z = values[2]
    print("x = ", x , "y = ", y , "z = ", z )

def on_error(ws, error):
    print("error occurred ", error)
    
def on_close(ws, close_code, reason):
    print("connection closed : ", reason)
    
def on_open(ws):
    print("connected")
    

def connect(url):
    ws = websocket.WebSocketApp(url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever()
 
 
connect("ws://192.168.0.103:8080/sensor/connect?type=android.sensor.accelerometer") 

```
 *Your device's IP might be different when you tap start button, so make sure you are using correct IP address at client side*

 Also see [
Connecting to Multiple Sensors Using Threading in Python](https://github.com/umer0586/SensorServer/wiki/Connecting-to-Multiple-Sensors-Using-Threading-in-Python) 


## Connecting To The Server without hardcoding IP Address and Port No
In networks using DHCP (Dynamic Host Configuration Protocol), devices frequently receive different IP addresses upon reconnection, making it impractical to rely on hardcoded network configurations. To address this challenge, the app supports Zero-configuration networking (Zeroconf/mDNS), enabling automatic server discovery on local networks. This feature eliminates the need for clients to hardcode IP addresses and port numbers when connecting to the WebSocket server. When enabled by the app user, the server broadcasts its presence on the network using the service type `_websocket._tcp`, allowing clients to discover the server automatically. Clients can now implement service discovery to locate the server dynamically, rather than relying on hardcoded network configurations.

See complete python Example at [Connecting To the Server Using Service Discovery](https://github.com/umer0586/SensorServer/wiki/Connecting-To-the-Server-Using-Service-Discovery) 
 
## Using Multiple Sensors Over single Websocket Connection
You can also connect to multiple sensors over single websocket connection. To use multiple sensors over single websocket connection use following **URL**.

                 ws://<ip>:<port>/sensors/connect?types=["<type1>","<type2>","<type3>"...]

By connecting using above URL you will receive JSON response containing sensor data along with a type of sensor. See complete example at [Using Multiple Sensors On Single Websocket Connection](https://github.com/umer0586/SensorServer/wiki/Using-Multiple-Sensors-On-Single-Websocket-Connection). Avoid connecting too many sensors over single connection

## Reading Touch Screen Data
By connecting to the address `ws://<ip>:<port>/touchscreen`, clients can receive touch screen events in following JSON formate.

|   Key   |   Value                  |
|:-------:|:-----------------------:|
|   x     |         x coordinate of touch           |
|   y     |         y coordinate of touch          |
| action  | ACTION_MOVE or ACTION_UP or ACTION_DOWN |

"ACTION_DOWN" indicates that a user has touched the screen.
"ACTION_UP" means the user has removed their finger from the screen.
"ACTION_MOVE" implies the user is sliding their finger across the screen.
See [Controlling Mouse Movement Using SensorServer app](https://github.com/umer0586/SensorServer/wiki/Controlling-Mouse-Using-SensorServer-App)

## Getting Device Location Using GPS
You can access device location through GPS using following URL.

                 ws://<ip>:<port>/gps
                 
See [Getting Data From GPS](https://github.com/umer0586/SensorServer/wiki/Getting-Data-From-GPS) for more details



## Real Time plotting
See [Real Time Plot of Accelerometer (Python)](https://github.com/umer0586/SensorServer/wiki/Real-Time-Plot-Example-(-Python)) using this app

![result](https://user-images.githubusercontent.com/35717992/208961337-0f69757e-e85b-4637-8c39-fa5554d85921.gif)



https://github.com/umer0586/SensorServer/assets/35717992/2ebf865d-529e-4702-8254-347df98dc795

## Testing in a Web Browser
You can also view your phone's sensor data in a Web Browser. Open the app's navigation drawer menu and enable `Test in a Web Browser`.Once the web server is running, the app will display an address on your screen. This address will look something like `http://<ip>:<port>`.On your device or another computer on the same network, open a web browser and enter that address. The web browser will now display a list of all the sensors available on your device. The web interface have options to connect to and disconnect from individual sensors, allowing you to view their real-time data readings.

<img width="742" src="https://github.com/user-attachments/assets/6ddac5cc-dc88-4ab2-aca9-b53fbd57a9c2">

![plotting](https://github.com/user-attachments/assets/297a001a-ed88-4299-9cf1-31451fff2c18)




This web app is built using Flutter and its source could be found under [sensors_dashboard](https://github.com/umer0586/SensorServer/tree/main/sensors_dashboard). However, there's one current limitation to be aware of. The app is built with Flutter using the `--web-renderer canvaskit` option. This means that the resulting app will have some dependencies that need to be downloaded from the internet. This means that any device accessing the web app through a browser will require an internet connection to function properly.

The web app is built and deployed to Android's assets folder via `python deploy_web_app.py`


## Connecting over Hotspot :fire:
If you don't have Wifi network at your work place you can directly connect websocket clients to the app by enabling **Hotspot Option** from settings. Just make sure that websocket clients are connected to your phone's hotspot


## Connecting over USB (using ADB)
To connect over USB make sure `USB debugging` option is enable in your phone and `ADB` (android debug bridge) is available in your machine
* **Step 1 :** Enable `Local Host` option in app
* **Step 2** : Run adb command `adb forward tcp:8081 tcp:8081` (8081 is just for example) from client
* **Step 3** : use address `ws://localhost:8081:/sensor/connect?type=<sensor type here>` to connect 

Make sure you have installed your android device driver and `adb devices` command detects your connected android phone.

## Video Streaming Functionality

This improved version of SensorServer adds real-time video streaming capabilities, allowing you to access your Android device's camera feed over WebSocket. This feature enables various applications such as remote monitoring, computer vision projects, and integrating your phone's camera into other systems.

### Accessing the Video Stream

To access the video stream, connect to the following WebSocket endpoint:

```
ws://<ip>:<port>/video
```

The video stream is sent as Base64-encoded JPEG frames, which can be decoded and processed by your client application.

### Python Example for Video Stream

Here's a simple Python example that connects to the video stream and displays it:

```python
import websocket
import json
import cv2
import numpy as np
import base64

def on_message(ws, message):
    try:
        # Decode base64 image
        img_data = base64.b64decode(message)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Display image
        cv2.imshow("Video Stream", img)
        cv2.waitKey(1)
    except Exception as e:
        print(f"Error processing frame: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_code, reason):
    print(f"Connection closed: {reason}")
    cv2.destroyAllWindows()

def on_open(ws):
    print("Connected to video stream")

# Connect to video stream
ws = websocket.WebSocketApp("ws://192.168.0.103:8086/video",
                          on_open=on_open,
                          on_message=on_message,
                          on_error=on_error,
                          on_close=on_close)

ws.run_forever()
```

Make sure to install the required packages:
```bash
pip install websocket-client opencv-python numpy
```

## 视觉惯性里程计 (VIO) 实现

本项目还包含两种视觉惯性里程计(Visual-Inertial Odometry, VIO)系统的实现，用于整合IMU数据、陀螺仪数据和视频流来估计相机的运动轨迹。

### 什么是VIO？

视觉惯性里程计(VIO)是一种结合视觉信息和惯性测量单元(IMU)数据的定位技术，可以在没有GPS的环境中实现精确的位置跟踪。VIO通过融合两种互补的传感器数据源来获得更准确的位置估计：

- **视觉部分**：通过分析连续视频帧中的特征点移动来估计相机的运动
- **惯性部分**：使用IMU(加速度计和陀螺仪)数据来预测相机的短期运动

### 两种VIO实现

本项目提供了两种不同的VIO实现：

1. **简单VIO实现** (`simple_vio.py`)
   - 基础的视觉惯性融合
   - 使用ORB特征和简单加权融合策略

2. **基于VINS-Mono架构的VIO系统** (`vins_inspired/`)
   - 参考香港科技大学VINS-Mono项目的架构
   - 包含特征跟踪、IMU预积分、初始化、滑动窗口优化等模块
   - 更加完善的状态估计和融合策略

### 数据收集

在使用VIO系统前，需要先收集传感器数据：

```bash
python data_collector.py
```

该脚本会引导您完成六种不同运动模式的数据收集：
- x轴正向负向运动
- y轴正向负向运动
- z轴正向负向运动
- x轴旋转
- y轴旋转
- z轴旋转

每种运动模式会分别收集IMU数据、陀螺仪数据和视频帧，并保存在结构化的目录中，包含详细的元数据信息。

### 运行VIO系统

收集完数据后，可以选择运行以下任一VIO系统：

```bash
# 运行简单VIO实现
python simple_vio.py

# 运行基于VINS-Mono架构的VIO系统
python run_vins_inspired_vio.py
```

程序会自动处理最新收集的数据，并为每种运动类型生成轨迹和可视化结果。

### 结果输出

VIO系统会生成以下输出：

1. **轨迹图**：显示估计的3D相机运动轨迹
2. **可视化视频**：包含特征点跟踪和状态信息的视频
3. **轨迹数据**：保存的轨迹坐标数据
4. **HTML报告**：汇总所有运动类型的处理结果（VINS架构版本）

所有结果都保存在对应运动类型的`vio_results`子目录中。

### 技术细节

VINS架构的VIO系统包含以下核心模块：

1. **特征跟踪器** (`feature_tracker.py`)
   - 使用光流法跟踪特征点
   - 管理特征点ID和生命周期

2. **IMU预积分** (`imu_preintegration.py`)
   - 执行IMU数据预积分
   - 处理偏置校正

3. **初始化器** (`initializer.py`)
   - 执行视觉SFM初始化
   - 估计重力方向和尺度

4. **滑动窗口优化器** (`sliding_window_optimizer.py`)
   - 维护滑动窗口状态
   - 融合视觉和IMU约束

5. **VIO系统** (`vio_system.py`)
   - 整合所有模块
   - 处理传感器数据

### 系统要求

- Python 3.6+
- OpenCV
- NumPy
- SciPy
- Matplotlib
