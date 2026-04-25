# Prop Inspection

This project simulates an automated inspection system for propellers using ROS 2 and Gazebo.

An overhead camera observes a grid of props and classifies each as:
- good
- warped
- incomplete
- sinkage

The system compares predictions against ground truth and reports accuracy.

---

## Features

- Gazebo simulation with spawned prop models
- Overhead camera with ROS image bridge
- Grid-based image segmentation
- Template-based classification using contour matching
- Ground truth generation and accuracy evaluation

---

## Workspace Setup

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/prop_inspection_ws
cd ~/prop_inspection_ws

git clone https://github.com/Adelco24/prop_inspection_ws.git .
```
Alternatively, you can cd into the home directory, run the git clone without the dot at the end, and then cd into the prop_inspection_ws.

## Run Code

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch prop_inspection prop_inspection.launch.py
```

You should expect to see the props slowly spawn, and then after about 45 seconds (to ensure all are spawned) it will run the visual code and print accuracy results to terminal.
