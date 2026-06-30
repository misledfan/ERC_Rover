#!/usr/bin/env python3  

# The Integrators: Perception Pipeline code 

# Code is the part of main.py that handles the object detection and classification on Raspberry pi (Turtlebot)

import cv2  

 

import numpy as np  

 

from ultralytics import YOLO  

 

import threading  

 

import time  

 

import sys  

 

from flask import Flask, Response  

 

  

 

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

 

  

 

WASTE_MAP = {  

 

    39: ("PLASTIC", "Bottle", (255, 0, 0)),  

    41: ("PAPER", "Cup",  (0, 255, 0)),  

    73: ("PAPER", "Book", (0, 255, 0)),  

    42: ("METAL", "Fork",  (0, 0, 255)),  

    43: ("METAL", "Knife", (0, 0, 255)),  

    44: ("METAL", "Spoon", (0, 0, 255)),  

 

}  

 

CONFIDENCE_THRESHOLD = 0.25  

 

def main():  

 

    global global_frame  

 

    print("🧠 Loading YOLOv8s (Small) Model...")  

 

    model = YOLO('yolov8s.pt')  

    cap = None  

 

    for cam_idx in [0, 2, 4, 1]:  

 

        cap = cv2.VideoCapture(cam_idx)  

 

        if cap.isOpened():  

 

            ret, _ = cap.read()  

 

            if ret:  

 

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  

 

                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

 

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  

 

                print(f"✅ USB Camera LOCKED on index {cam_idx}!")  

 

                break  

 

            else:  

 

                cap.release()  

 

    if cap is None:  

 

        print("❌ CRITICAL: Could not find USB Camera!")  

 

        sys.exit(1)  

 

    threading.Thread(target=run_flask, daemon=True).start()  

 

    print("🌐 Web server running! Open http://<RASPBERRY_PI_IP>:5000")  

 

  

 

    try:  

 

        while True:  

 

            ret, frame = cap.read()  

 

            if not ret:  

 

                time.sleep(0.01)  

 

                continue  

 

  

 

            display_frame = frame.copy()  

 

            yolo_boxes = []  

 

            # --- 1. YOLO AI DETECTION ---  

 

            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=320)[0]  

 

  

 

            for box in results.boxes:  

 

                cls_id = int(box.cls[0])  

 

                if cls_id in WASTE_MAP:  

 

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  

 

                    conf = float(box.conf[0])  

 

                    category, item, color = WASTE_MAP[cls_id]  

 

  

 

                    roi = frame[y1:y2, x1:x2]  

 

                    if roi.size == 0: continue  

 

  

 

                    # 🚀 FIX 1: Cutlery Core Cropping & Strict Colors 🚀  

 

                    if cls_id in [42, 43, 44]:  

 

                        # Crop to the dead center (20% to 80%) to avoid the carpet  

 

                        h, w = roi.shape[:2]  

 

                        cy1, cy2 = int(h * 0.2), int(h * 0.8)  

 

                        cx1, cx2 = int(w * 0.2), int(w * 0.8)  

 

                        core_roi = roi[cy1:cy2, cx1:cx2]  

 

  

 

                        if core_roi.size > 0:  

 

                            hsv_core = cv2.cvtColor(core_roi, cv2.COLOR_BGR2HSV)  

 

  

 

                            # STRICT White: Must be extremely bright, ignoring dull metallic glare  

 

                            white_mask = cv2.inRange(hsv_core, (0, 0, 180), (180, 30, 255))  

 

                            # STRICT Black: Must be pitch black, ignoring normal shadows  

 

                            black_mask = cv2.inRange(hsv_core, (0, 0, 0), (180, 255, 40))  

 

  

 

                            white_pixels = cv2.countNonZero(white_mask)  

 

                            black_pixels = cv2.countNonZero(black_mask)  

 

                            total_core = core_roi.shape[0] * core_roi.shape[1]  

 

  

 

                            # Check if the center is overwhelmingly matte white or black  

 

                            if (white_pixels / total_core) > 0.15:  

 

                                category, item, color = "PLASTIC", f"Wht Plastic {item}", (255, 0, 0)  

 

                            elif (black_pixels / total_core) > 0.15:  

 

                                category, item, color = "PLASTIC", f"Blk Plastic {item}", (255, 0, 0)  

 

                            else:  

 

                                category, item, color = "METAL", f"Metal {item}", (0, 0, 255)  

 

  

 

                    # 🚀 FIX 2: Painted Soda Cans vs Clear Bottles 🚀  

 

                    if cls_id == 39:  

 

                        width = x2 - x1  

 

                        height = y2 - y1  

 

                        longest = max(width, height)  

 

                        shortest = min(width, height)  

 

  

 

                        if shortest > 0:  

 

                            aspect_ratio = longest / shortest  

 

                            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  

 

                            # Measure how "colorful" the object is  

 

                            avg_saturation = np.mean(hsv_roi[:, :, 1])  

 

                            if aspect_ratio < 2.0:  

 

                                category, item, color = "METAL", "Fat Can", (0, 0, 255)  

 

                            elif 2.0 <= aspect_ratio < 3.5:  

 

                                # A painted Coke/Pepsi can has high saturation. Clear bottles are low.  

 

                                if avg_saturation > 85:  

 

                                    category, item, color = "METAL", "Sleek Can", (0, 0, 255)  

 

                                else:  

 

                                    category, item, color = "PLASTIC", "Bottle", (255, 0, 0)  

 

                    yolo_boxes.append((x1, y1, x2, y2))  

 

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)  

 

                    label = f"AI: {category} ({item}) {conf:.2f}"  

 

                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  

 

 

            # --- 2. WHITE PAPER DETECTOR ---  

 

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  

 

            lower_white = np.array([0, 0, 150])  

 

            upper_white = np.array([180, 50, 255])  

 

            paper_mask = cv2.inRange(hsv, lower_white, upper_white)  

 

            contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

 

  

 

            for cnt in contours:  

 

                area = cv2.contourArea(cnt)  

 

                if area > 200:  

 

                    wx1, wy1, ww, wh = cv2.boundingRect(cnt)  

 

                    wx2, wy2 = wx1 + ww, wy1 + wh  

 

                    is_duplicate = False  

 

                    for (yx1, yy1, yx2, yy2) in yolo_boxes:  

 

                        # Overlap check to prevent Can glare from becoming paper  

 

                        if not (wx2 < yx1 or wx1 > yx2 or wy2 < yy1 or wy1 > yy2):  

 

                            is_duplicate = True  

 

                            break  

 

  

 

                    if not is_duplicate:  

 

                        yolo_boxes.append((wx1, wy1, wx2, wy2))  

 

                        cv2.rectangle(display_frame, (wx1, wy1), (wx2, wy2), (0, 255, 0), 2)  

 

                        cv2.putText(display_frame, "CV: PAPER (White)", (wx1, wy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  

 

            # --- 3. CLEAR PLASTIC WRAPPER DETECTOR ---  

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

 

            edges = cv2.Canny(gray, 100, 200)  

 

            wrapper_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

 

            for cnt in wrapper_contours:  

 

                area = cv2.contourArea(cnt)  

 

                if 500 < area < 5000:  

 

                    px1, py1, pw, ph = cv2.boundingRect(cnt)  

 

                    px2, py2 = px1 + pw, py1 + ph  

 

  

 

                    is_duplicate = False  

 

                    for (yx1, yy1, yx2, yy2) in yolo_boxes:  

 

                        if not (px2 < yx1 or px1 > yx2 or py2 < yy1 or py1 > yy2):  

 

                            is_duplicate = True  

 

                            break  

  

 

                    if not is_duplicate:  

 

                        cv2.rectangle(display_frame, (px1, py1), (px2, py2), (255, 0, 0), 1)  

 

                        cv2.putText(display_frame, "CV: PLASTIC (Wrapper)", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  

 

 

            global_frame = display_frame  

            time.sleep(0.05)  

 

    except KeyboardInterrupt:  

 

        print("\n🛑 Stopping...")  

 

    finally:  

        cap.release()  

 

  

 

if __name__ == '__main__':  

 

    main() 

 
