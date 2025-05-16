import websocket
import json


def on_message(ws, message):
    data = json.loads(message)
    values = data['values']
    x = values[0]  # 绕X轴旋转的角速度（弧度/秒）
    y = values[1]  # 绕Y轴旋转的角速度（弧度/秒）
    z = values[2]  # 绕Z轴旋转的角速度（弧度/秒）
    print("陀螺仪数据（弧度/秒）:")
    print("x轴旋转 = ", x, "y轴旋转 = ", y, "z轴旋转 = ", z)


def on_error(ws, error):
    print("连接错误: ", error)


def on_close(ws, close_code, reason):
    print("连接关闭: ", reason)


def on_open(ws):
    print("已连接到陀螺仪传感器")


def connect(url):
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()


# 连接到陀螺仪传感器，注意修改IP地址为您的设备IP
connect("ws://192.168.43.132:8080/sensor/connect?type=android.sensor.gyroscope")
