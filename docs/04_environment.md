# External Dependency Map

This page is only a map of external dependencies. Detailed installation,
`colcon`, Docker, Conda, and launch commands belong in each workspace README.

## Left Control PC

Workspace README:

```text
left_ws/README.md
```

External dependencies:

- Ubuntu 22.04
- ROS 2 Humble
- libfranka / Franka ROS 2 support
- Python 3.10 virtual environment for retargeting and finger mapping
- dex-retargeting
- PyModbus
- evdev
- Vicon UDP handpoint source
- Inspire hand Modbus TCP network

## Right Control PC

Workspace README:

```text
right_ws/README.md
```

External dependencies:

- Ubuntu 22.04
- ROS 2 Humble
- libfranka / Franka ROS 2 support
- RealSense ROS packages
- Azure Kinect ROS/runtime packages
- Python packages used by recording and viewer nodes
- Inspire hand Modbus TCP network

## Train PC

Workspace READMEs:

```text
train_ws/docker_inspire/README.md
train_ws/ros2_ws/README.md
train_ws/train/README.md
```

External dependencies:

- Docker with NVIDIA GPU passthrough
- ROS 2 Humble inside the deploy container
- Conda environment for policy deploy
- Conda environment for policy training
- CUDA/PyTorch runtime matching the machine GPU
- RealSense and Azure Kinect runtime libraries for deploy cameras
- Dataset storage outside git
- Checkpoint/log/output storage outside git
