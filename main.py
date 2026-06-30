#The Integrators - ROS2 Turltebot3 Independant Maneuvarability, collision avoidance and Object detection code (Single target)
# An autonomous ROS2 TurtleBot3 node that uses YOLOv8 for object detection (trash hunting), LiDAR for obstacle avoidance, and Flask for live video streaming.

#!/usr/bin/env python3  

# The Integrators: Turtlebot Functionality Code (Try30) 
# Author = Safwaan Syed , Joud Almomani, Keren Sara John.

import rclpy  

from rclpy.node import Node  

from geometry_msgs.msg import Twist  

from sensor_msgs.msg import LaserScan  

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  

 

import cv2  

 

import numpy as np  

 

from ultralytics import YOLO  

 

import threading  

 

import time  

 

import math  

 

import subprocess  

 

import shlex  

 

import sys  

 

from flask import Flask, Response  


# ═══════════════════════════════════════════════  

 
#  FLASK WEB SERVER  


# ═══════════════════════════════════════════════  

 

app = Flask(__name__)  

 

global_frame = None  

 

def generate_frames():  

 

    global global_frame  

 

    while True:  

 

        if global_frame is None:  

 

            time.sleep(0.1)  

 

            continue  

 

        ret, buffer = cv2.imencode('.jpg', global_frame)  

 

        if not ret: continue  

 

        yield (b'--frame\r\n'  

 

               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')  

 

  
@app.route('/')  

 

def video_feed():  

 

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  

 

  

 
def run_flask():  

 

    import logging  

 

    log = logging.getLogger('werkzeug')  

 

    log.setLevel(logging.ERROR)  

 

    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)  

 

# ═══════════════════════════════════════════════  


#  CONFIG & TUNING  

 
# ═══════════════════════════════════════════════  

 

CAMERA_WIDTH = 640  

 

CAMERA_HEIGHT = 480  

 

IMAGE_CENTER_X = CAMERA_WIDTH // 2  

 

  

 

BUMPER_Y_LIMIT = CAMERA_HEIGHT * 0.90  

 

BLIND_TRIGGER_Y = CAMERA_HEIGHT * 0.70  

 

  

 

LINEAR_SPEED = 0.12  

 

TURN_SPEED = 0.5  

 

OBSTACLE_DIST = 0.45  # LOWERED: Less paranoid  

 

CLEAR_DIST = 0.55     # LOWERED: Resumes faster  

 

  

 

WASTE_MAP = {  

 

    39: ("PLASTIC", "Bottle"),  

 

    41: ("PAPER", "Cup"),  

 

    73: ("PAPER", "Book"),  

 

    42: ("METAL", "Fork"),  

 

    43: ("METAL", "Knife"),  

 

    44: ("METAL", "Spoon"),  

 

    67: ("METAL", "Cell Phone")  

 

}  

 

  

 

CONFIDENCE_THRESHOLD = 0.60  

 

  

 

class State:  

 

    WANDER = "WANDER"  

 

    AVOID = "AVOID_OBSTACLE"  

 

    HUNT = "HUNT_TRASH"  

 

    BLIND_FINISH = "BLIND_FINISH"  

 

    COLLECT = "COLLECT_WAIT"  

 

    BACKUP = "BACKUP_MANEUVER"       # NEW: Back away from collected trash  

 

    TURN_AWAY = "TURN_AWAY_MANEUVER" # NEW: Face a new direction  

 

  

 

class Try21Node(Node):  

 

    def __init__(self):  

 

        super().__init__('try21_node')  

 

  

 

        print("🚀 Launching TurtleBot Hardware...")  

 

        self.bringup_proc = self.launch_bringup()  

 

        time.sleep(8.0)  

 

  

 

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)  

 

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)  

 

        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)  

 

  

 

        print("🧠 Loading YOLOv8n...")  

 

        self.model = YOLO('yolov8n.pt')  

 

  

 

        self.cap = None  

 

        for cam_idx in [0, 2, 4, 1]:  

 

            cap = cv2.VideoCapture(cam_idx)  

 

            if cap.isOpened():  

 

                ret, _ = cap.read()  

 

                if ret:  

 

                    self.cap = cap  

 

                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)  

 

                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)  

 

                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

 

                    print(f"✅ USB Camera LOCKED on index {cam_idx}!")  

 

                    break  

 

                else:  

 

                    cap.release()  

 

  

 

        if self.cap is None:  

 

            print("❌ CRITICAL: Could not find USB Camera!")  

 

            self.shutdown()  

 

            sys.exit(1)  

 

  

 

        self.state = State.WANDER  

 

        self.min_front_dist = float('inf')  

 

        self.latest_frame = None  

 

        self.shutdown_flag = False  

 

  

 

        self.ai_target_found = False  

 

        self.ai_target_cx = 0  

 

        self.ai_target_y2 = 0  

 

        self.ai_target_category = ""  

 

        self.ai_target_item = ""  

 

        self.ai_consecutive_hits = 0  

 

  

 

        self.collect_start_time = 0.0  

 

        self.blind_start_time = 0.0  

 

        self.backup_start_time = 0.0  

 

        self.turn_away_start_time = 0.0  

 

        self.last_seen_time = time.time()  

 

  

 

        threading.Thread(target=self.camera_loop, daemon=True).start()  

 

        threading.Thread(target=self.yolo_loop, daemon=True).start()  

 

        threading.Thread(target=run_flask, daemon=True).start()  

 

  

 

        self.timer = self.create_timer(0.1, self.motor_reflex_loop)  

 

        print("✅ Node Ready. State: WANDER")  

 

  

 

    def launch_bringup(self):  

 

        cmd = "ros2 launch turtlebot3_bringup robot.launch.py"  

 

        return subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  

 

  

 

    def camera_loop(self):  

 

        while rclpy.ok() and not self.shutdown_flag:  

 

            if self.cap.isOpened():  

 

                ret, frame = self.cap.read()  

 

                if ret:  

 

                    self.latest_frame = frame  

 

            time.sleep(0.01)  

 

  

 

    def yolo_loop(self):  

 

        global global_frame  

 

        while rclpy.ok() and not self.shutdown_flag:  

 

            if self.latest_frame is None:  

 

                time.sleep(0.1)  

 

                continue  

 

  

 

            frame = self.latest_frame.copy()  

 

            display_frame = frame.copy()  

 

  

 

            results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=320)[0]  

 

  

 

            detected_trash = []  

 

            for box in results.boxes:  

 

                cls_id = int(box.cls[0])  

 

                if cls_id in WASTE_MAP:  

 

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  

 

                    area = (x2 - x1) * (y2 - y1)  

 

                    cx = int((x1 + x2) / 2)  

 

                    category, item = WASTE_MAP[cls_id]  

 

                    detected_trash.append((cx, area, category, item, y2))  

 

  

 

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

 

                    cv2.line(display_frame, (0, int(BUMPER_Y_LIMIT)), (CAMERA_WIDTH, int(BUMPER_Y_LIMIT)), (0, 0, 255), 2)  

 

                    cv2.putText(display_frame, f"{item}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  

 

  

 

            global_frame = display_frame  

 

            found_this_frame = False  

 

  

 

            if detected_trash:  

 

                best_match = None  

 

                if self.state == State.HUNT and self.ai_target_found:  

 

                    min_dist = float('inf')  

 

                    for trash in detected_trash:  

 

                        dist = abs(trash[0] - self.ai_target_cx)  

 

                        if dist < min_dist:  

 

                            min_dist = dist  

 

                            best_match = trash  

 

                else:  

 

                    max_area = 0  

 

                    for trash in detected_trash:  

 

                        if trash[1] > max_area:  

 

                            max_area = trash[1]  

 

                            best_match = trash  

 

  

 

                if best_match:  

 

                    found_this_frame = True  

 

                    self.ai_target_cx = best_match[0]  

 

                    self.ai_target_category = best_match[2]  

 

                    self.ai_target_item = best_match[3]  

 

                    self.ai_target_y2 = best_match[4]  

 

                    self.last_seen_time = time.time()  

 

  

 

            if found_this_frame:  

 

                self.ai_consecutive_hits += 1  

 

                if self.ai_consecutive_hits >= 2:  

 

                    self.ai_target_found = True  

 

            else:  

 

                self.ai_consecutive_hits = 0  

 

                if time.time() - self.last_seen_time > 1.0:  

 

                    self.ai_target_found = False  

 

  

 

    def scan_cb(self, msg: LaserScan):  

 

        ranges = msg.ranges  

 

        n = len(ranges)  

 

        if n == 0: return  

 

        idx_half = int(math.radians(30) / msg.angle_increment)  

 

        front_ranges = []  

 

  

 

        # ⚡ LiDAR FIX: 0.15 ignores the chassis/dust, but catches hands/walls  

 

        for i in range(idx_half):  

 

            if 0.15 < ranges[i] < 5.0: front_ranges.append(ranges[i])  

 

        for i in range(n - idx_half, n):  

 

            if 0.15 < ranges[i] < 5.0: front_ranges.append(ranges[i])  

 

  

 

        self.min_front_dist = min(front_ranges) if front_ranges else float('inf')  

 

  

 

    def publish_cmd(self, linear, angular):  

 

        msg = Twist()  

 

        msg.linear.x = float(linear)  

 

        msg.angular.z = float(angular)  

 

        self.cmd_pub.publish(msg)  

 

  

 

    def motor_reflex_loop(self):  

 

        current_time = time.time()  

 

  

 

        if self.state == State.WANDER:  

 

            if self.ai_target_found:  

 

                print(f"🎯 TARGET LOCKED: {self.ai_target_item}. Hunting...")  

 

                self.state = State.HUNT  

 

            elif self.min_front_dist < OBSTACLE_DIST:  

 

                print("🧱 Wall detected! Pivoting...")  

 

                self.state = State.AVOID  

 

            else:  

 

                self.publish_cmd(LINEAR_SPEED, 0.0)  

 

  

 

        elif self.state == State.AVOID:  

 

            if self.min_front_dist > CLEAR_DIST:  

 

                print("✅ Path clear. Resuming wander.")  

 

                self.state = State.WANDER  

 

            elif self.ai_target_found and self.min_front_dist > 0.30:  

 

                print(f"🎯 Trash spotted while turning! Hunting...")  

 

                self.state = State.HUNT  

 

            else:  

 

                self.publish_cmd(0.0, TURN_SPEED)  

 

  

 

        elif self.state == State.HUNT:  

 

            if not self.ai_target_found:  

 

                if self.ai_target_y2 > BLIND_TRIGGER_Y:  

 

                    print("🙈 Object dipped under camera! Blind grab (0.4s)...")  

 

                    self.blind_start_time = current_time  

 

                    self.state = State.BLIND_FINISH  

 

                else:  

 

                    print("❌ Target lost entirely. Resuming wander.")  

 

                    self.state = State.WANDER  

 

            else:  

 

                if self.ai_target_y2 >= BUMPER_Y_LIMIT:  

 

                    self.publish_cmd(0.0, 0.0)  

 

                    print(f"♻️ BUMPER REACHED! Collecting {self.ai_target_category}... Waiting 2s.")  

 

                    self.collect_start_time = current_time  

 

                    self.state = State.COLLECT  

 

                else:  

 

                    box_width = 80  

 

                    if self.ai_target_cx < IMAGE_CENTER_X - box_width // 2:  

 

                        self.publish_cmd(LINEAR_SPEED, TURN_SPEED)  

 

                    elif self.ai_target_cx > IMAGE_CENTER_X + box_width // 2:  

 

                        self.publish_cmd(LINEAR_SPEED, -TURN_SPEED)  

 

                    else:  

 

                        self.publish_cmd(LINEAR_SPEED, 0.0)  

 

  

 

        elif self.state == State.BLIND_FINISH:  

 

            if current_time - self.blind_start_time < 0.4:  

 

                self.publish_cmd(LINEAR_SPEED, 0.0)  

 

            else:  

 

                self.publish_cmd(0.0, 0.0)  

 

                print(f"♻️ BLIND GRAB DONE! Collecting... Waiting 2s.")  

 

                self.collect_start_time = current_time  

 

                self.state = State.COLLECT  

 

  

 

        elif self.state == State.COLLECT:  

 

            self.publish_cmd(0.0, 0.0)  

 

            if current_time - self.collect_start_time > 2.0:  

 

                print("✅ Collection complete. Backing up...")  

 

                self.backup_start_time = current_time  

 

                self.state = State.BACKUP  

 

  

 

        elif self.state == State.BACKUP:  

 

            self.publish_cmd(-LINEAR_SPEED, 0.0) # Reverse  

 

            if current_time - self.backup_start_time > 1.0:  

 

                print("🔄 Turning away from collected trash...")  

 

                self.turn_away_start_time = current_time  

 

                self.state = State.TURN_AWAY  

 

  

 

        elif self.state == State.TURN_AWAY:  

 

            self.publish_cmd(0.0, TURN_SPEED) # Spin  

 

            if current_time - self.turn_away_start_time > 1.5:  

 

                print("✅ Resuming wander...")  

 

                self.ai_target_found = False  

 

                self.ai_consecutive_hits = 0  

 

                self.state = State.WANDER  

 

 

    def shutdown(self):  

 

        print("🛑 Shutting down...")  

 

        self.shutdown_flag = True  

 

        self.publish_cmd(0.0, 0.0)  

 

        if self.cap:  

 

            self.cap.release()  

 

            time.sleep(0.5)  

 

        if self.bringup_proc:  

 

            self.bringup_proc.kill()  

 

        time.sleep(0.5)  

 

 

def main(args=None):  

 

    rclpy.init(args=args)  

 

    node = Try21Node()  

    try: rclpy.spin(node)  

    except KeyboardInterrupt: pass  

    except SystemExit: pass  

 

    finally:  

 

        node.shutdown()  

 

        node.destroy_node()  

 

        rclpy.shutdown()  

 

 

if __name__ == '__main__':  

 

    main() 
