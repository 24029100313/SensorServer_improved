#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import websocket
import threading
import time
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='WebSocket Video Stream Client')
    parser.add_argument('--url', type=str, default='ws://192.168.1.100:8080/video',
                        help='WebSocket URL for video stream (default: ws://192.168.1.100:8080/video)')
    return parser.parse_args()

class VideoStreamClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.connected = False
        self.frame = None
        self.lock = threading.Lock()
        
    def on_message(self, ws, message):
        # Convert binary message to numpy array
        try:
            nparr = np.frombuffer(message, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            with self.lock:
                self.frame = img
        except Exception as e:
            print(f"Error decoding image: {e}")
            
    def on_error(self, ws, error):
        print(f"Error: {error}")
        
    def on_close(self, ws, close_status_code, close_msg):
        print(f"Connection closed: {close_status_code} - {close_msg}")
        self.connected = False
        
    def on_open(self, ws):
        print(f"Connected to {self.url}")
        self.connected = True
        
    def connect(self):
        print(f"Connecting to {self.url}...")
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Wait for connection or timeout
        timeout = 5
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
            
        if not self.connected:
            print(f"Failed to connect to {self.url} within {timeout} seconds")
            return False
            
        return True
        
    def disconnect(self):
        if self.ws:
            self.ws.close()
            
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

def main():
    args = parse_args()
    client = VideoStreamClient(args.url)
    
    if not client.connect():
        sys.exit(1)
    
    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = client.get_frame()
            if frame is not None:
                cv2.imshow("Video Stream", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        client.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
