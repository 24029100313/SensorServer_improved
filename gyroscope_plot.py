"""
Python code to show real time plot from live gyroscope's
data recieved via SensorServer app over websocket 
"""

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import websocket
import json
import threading

#shared data
x_data = []
y_data = []
z_data = []
time_data = []

x_data_color = "#d32f2f"   # red (x轴旋转)
y_data_color = "#7cb342"   # green (y轴旋转)
z_data_color = "#0288d1"   # blue (z轴旋转)

background_color = "#fafafa" # white (material)


class Sensor:
    #constructor
    def __init__(self,address,sensor_type):
        self.address = address
        self.sensor_type = sensor_type
    
    # called each time when sensor data is recieved
    def on_message(self,ws, message):
        values = json.loads(message)['values']
        timestamp = json.loads(message)['timestamp']

        x_data.append(values[0])
        y_data.append(values[1])
        z_data.append(values[2])

        time_data.append(float(timestamp/1000000))

    def on_error(self,ws, error):
        print("错误发生")
        print(error)

    def on_close(self,ws, close_code, reason):
        print("连接关闭")
        print("关闭代码: ", close_code)
        print("原因: ", reason)

    def on_open(self,ws):
        print(f"已连接到: {self.address}")

    # Call this method on seperate Thread
    def make_websocket_connection(self):
        ws = websocket.WebSocketApp(f"ws://{self.address}/sensor/connect?type={self.sensor_type}",
                                on_open=self.on_open,
                                on_message=self.on_message,
                                on_error=self.on_error,
                                on_close=self.on_close)

        # blocking call
        ws.run_forever() 
    
    # make connection and start recieving data on sperate thread
    def connect(self):
        thread = threading.Thread(target=self.make_websocket_connection)
        thread.start()           



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        self.graphWidget.setBackground(background_color)

        self.graphWidget.setTitle("陀螺仪数据实时图", color="#8d6e63", size="20pt")
        
        # Add Axis Labels
        styles = {"color": "#f00", "font-size": "15px"}
        self.graphWidget.setLabel("left", "角速度 (弧度/秒)", **styles)
        self.graphWidget.setLabel("bottom", "时间 (毫秒)", **styles)
        self.graphWidget.addLegend()

        self.x_data_line =  self.graphWidget.plot([],[], name="X轴旋转", pen=pg.mkPen(color=x_data_color))
        self.y_data_line =  self.graphWidget.plot([],[], name="Y轴旋转", pen=pg.mkPen(color=y_data_color))
        self.z_data_line =  self.graphWidget.plot([],[], name="Z轴旋转", pen=pg.mkPen(color=z_data_color))
      
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot_data) # call update_plot_data function every 50 milisec
        self.timer.start()

    def update_plot_data(self):
        
        # limit lists data to 1000 items 
        limit = -1000 

        # Update the data.
        self.x_data_line.setData(time_data[limit:], x_data[limit:])  
        self.y_data_line.setData(time_data[limit:], y_data[limit:])
        self.z_data_line.setData(time_data[limit:], z_data[limit:])


if __name__ == "__main__":
    # 在这里修改为您的设备IP地址和端口
    sensor = Sensor(address = "192.168.43.132:8080", sensor_type="android.sensor.gyroscope")
    sensor.connect() # asynchronous call

    app = QtWidgets.QApplication(sys.argv)

    # call on Main thread
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
