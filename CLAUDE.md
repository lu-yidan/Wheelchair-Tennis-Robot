# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fully autonomous wheelchair tennis robot. It combines a motorized wheelchair base with a 7-DOF Barrett WAM arm and a distributed camera network to detect, track, and intercept tennis balls. The robot competes in wheelchair tennis by autonomously navigating the court, predicting ball trajectories, and executing swing motions.

## Repository Structure

```
├── firmware/          # Teensy 4.1 microcontroller code (Arduino/C++)
├── hardware/          # Mechanical and electrical design files
│   ├── electrical/    # Schematics, power system, ODrive config
│   └── mechanical/    # CAD files for drivetrain, arm mount, racket holder
├── software/          # ROS packages (C++/Python)
│   ├── ball_detection/       # Real-time ball detection with ZED + CUDA
│   ├── ball_localization/    # EKF/UKF trajectory estimation
│   ├── ball_calibration/     # Camera calibration utilities (AprilTag-based)
│   ├── wtr_navigation/       # Wheelchair motion control + TEB planner + EKF localization
│   ├── wtr_plan/             # High-level strategizer (ball interception, swing timing)
│   ├── wam_control/          # Barrett WAM arm trajectory planning + execution
│   ├── wam_moveit/           # MoveIt config, URDF, kinematics
│   ├── wam_model/            # Robot model (based on jhu-lcsr/barrett_model)
│   └── wtr_sim/              # Gazebo simulation (not fully functional)
└── docs/              # Jekyll website source
```

## Building the Software

All ROS packages live under `software/` and are built with catkin:

```bash
# From the catkin workspace root (create one if needed)
mkdir -p ~/catkin_ws/src
ln -s /path/to/Wheelchair-Tennis-Robot/software ~/catkin_ws/src/
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

**ROS version**: Melodic (Jetson Nanos, Ubuntu 18.04) or Noetic (master computer)  
**C++ standard**: C++17–C++20 depending on package  
**Key external deps**: ZED SDK, OpenCV 4.5.2+ with CUDA, MoveIt, TEB Local Planner, Eigen, yaml-cpp

Individual package build:
```bash
catkin_make --pkg ball_detection
```

## Running the System

### Firmware (Teensy)
Flash `firmware/teensy_interface/teensy_interface.ino` via Arduino IDE with Teensyduino. The Teensy connects via USB to the master computer and acts as a bridge between ROS commands and the ODrive motor controllers over UART.

### Vision Network (Jetsons)
Six Jetson Nano nodes on a 5 GHz WiFi network (`core-robotics-net-2`). Static IPs: `192.168.1.101`–`192.168.1.106`. Master: `192.168.1.100`.

Set up a Jetson node:
```bash
# From ball_calibration package
python3 setup_jetson.py  # configures WiFi, ROS_MASTER_URI, chrony NTP
roslaunch ball_detection ball_detection.launch
```

### Main Launch
```bash
# On master computer
roslaunch wtr_navigation wtr_navigation.launch
roslaunch wam_control wam_control.launch
roslaunch wtr_plan wtr_plan.launch
```

## Architecture Data Flow

```
ZED Cameras (x6 Jetsons)
    → ball_detection (CUDA, ZED SDK) → /ball_detection topic (3D positions)
    → ball_localization (EKF/UKF) → /ball_state (trajectory + prediction)
        → wtr_plan (strategizer) → intercept point + swing timing
            → wtr_navigation (TEB planner) → /cmd_vel → Teensy → ODrive → wheels
            → wam_control (MoveIt) → WAM joint trajectories → swing
```

**RC remote** (4-channel) connects to Teensy for manual override, E-stop, and calibration mode switching.

## Key Subsystem Details

### Ball Detection (`ball_detection`)
- Uses ZED 2 stereo cameras; outputs 3D ball positions in world frame
- CUDA-accelerated; requires ZED SDK
- Config: `ball_detection/config/ball_detection.yaml`

### Ball Localization (`ball_localization`)
- EKF and UKF implementations for ball trajectory prediction under ballistic dynamics
- Fuses detections from all 6 cameras
- Config: `ball_localization/config/`

### Navigation (`wtr_navigation`)
- EKF sensor fusion for wheelchair localization
- TEB Local Planner for dynamic obstacle avoidance
- Subscribes to `/ball_state`, publishes `/cmd_vel`
- Teensy receives velocity commands over USB serial (ROS serial bridge)

### WAM Arm Control (`wam_control`)
- Trapezoidal trajectory profiler for swing execution
- MoveIt-based planning for pre-swing positioning
- Joint limits and collision models in `wam_moveit/`

### Calibration (`ball_calibration`)
- AprilTag-based extrinsic calibration for camera-to-court coordinate transforms
- Scripts for auto-gain, exposure calibration
- Remote joystick calibration support

## Hardware Notes

**Power**: GBoost 48V battery → 48V bus → ODrive 3.6 controllers → D6374 150KV BLDC motors (2x). Encoders: CUI AMT-102, 8192 CPR.

**ODrive config**: `hardware/electrical/odrive_config.json` — uses UART communication at 115200 baud.

**Barrett WAM**: 7-DOF arm with custom racket holder. HEAD Graphite Instinct Power racket.
