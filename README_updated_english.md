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

## Visual-Inertial Odometry (VIO) Implementation

This project also includes two Visual-Inertial Odometry (VIO) system implementations for integrating IMU data, gyroscope data, and video streams to estimate camera motion trajectories.

### What is VIO?

Visual-Inertial Odometry (VIO) is a positioning technology that combines visual information with Inertial Measurement Unit (IMU) data to achieve precise position tracking in environments without GPS. VIO obtains more accurate position estimates by fusing two complementary sensor data sources:

- **Visual Component**: Estimates camera motion by analyzing feature point movements across consecutive video frames
- **Inertial Component**: Uses IMU (accelerometer and gyroscope) data to predict short-term camera motion

### Two VIO Implementations

This project provides two different VIO implementations:

1. **Simple VIO Implementation** (`simple_vio.py`)
   - Basic visual-inertial fusion
   - Uses ORB features and simple weighted fusion strategy

2. **VINS-Mono Architecture-based VIO System** (`vins_inspired/`)
   - References the architecture of the VINS-Mono project from Hong Kong University of Science and Technology
   - Includes feature tracking, IMU preintegration, initialization, sliding window optimization, and other modules
   - More comprehensive state estimation and fusion strategy

### Data Collection

Before using the VIO system, you need to collect sensor data:

```bash
python data_collector.py
```

This script will guide you through collecting data for six different motion patterns:
- Positive and negative movement along the x-axis
- Positive and negative movement along the y-axis
- Positive and negative movement along the z-axis
- Rotation around the x-axis
- Rotation around the y-axis
- Rotation around the z-axis

Each motion pattern will collect IMU data, gyroscope data, and video frames, saving them in a structured directory with detailed metadata.

### Running the VIO System

After collecting the data, you can choose to run either VIO system:

```bash
# Run the simple VIO implementation
python simple_vio.py

# Run the VINS-Mono architecture-based VIO system
python run_vins_inspired_vio.py
```

The program will automatically process the most recently collected data and generate trajectories and visualization results for each motion type.

### Output Results

The VIO system generates the following outputs:

1. **Trajectory Plots**: Displays the estimated 3D camera motion trajectory
2. **Visualization Videos**: Videos containing feature tracking and state information
3. **Trajectory Data**: Saved trajectory coordinate data
4. **HTML Report**: Summary of processing results for all motion types (VINS architecture version)

All results are saved in the `vio_results` subdirectory for the corresponding motion type.

### Technical Details

The VINS architecture VIO system includes the following core modules:

1. **Feature Tracker** (`feature_tracker.py`)
   - Tracks feature points using optical flow
   - Manages feature point IDs and lifecycles

2. **IMU Preintegration** (`imu_preintegration.py`)
   - Performs IMU data preintegration
   - Handles bias correction

3. **Initializer** (`initializer.py`)
   - Performs visual SFM initialization
   - Estimates gravity direction and scale

4. **Sliding Window Optimizer** (`sliding_window_optimizer.py`)
   - Maintains sliding window state
   - Fuses visual and IMU constraints

5. **VIO System** (`vio_system.py`)
   - Integrates all modules
   - Processes sensor data

### System Requirements

- Python 3.6+
- OpenCV
- NumPy
- SciPy
- Matplotlib
