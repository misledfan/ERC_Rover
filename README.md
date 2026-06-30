# ERC_Rover
An autonomous ROS2 TurtleBot3 node that uses YOLOv8 for object detection (trash hunting), LiDAR for obstacle avoidance, and Flask for live video streaming.


This script integrates vision, navigation, and web streaming into a single ROS2 node to autonomously hunt and collect waste.

Object Detection: Uses a YOLOv8 Nano model to identify and classify specific waste types (plastic, paper, metal) via a USB camera.

Autonomous Navigation: Implements a state machine (Wander, Avoid, Hunt, Collect, Backup, Turn Away) to track targets and maneuver the robot via /cmd_vel.

Obstacle Avoidance: Processes /scan LiDAR data to actively avoid walls and obstacles while wandering.

Live Web Feed: Runs a lightweight Flask server in a separate thread to broadcast the annotated camera feed (with YOLO bounding boxes and status text) over HTTP.

Process Management: Automatically launches the TurtleBot3 bringup process and safely handles multi-threaded hardware shutdown.
