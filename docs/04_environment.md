# Environment and Dependencies

This page lists the environment and third-party dependencies for the left-control
workspace.

## System

Use:

```text
Ubuntu 22.04
ROS 2 Humble
Python 3.10
```

ROS 2 Humble install page:

```text
https://docs.ros.org/en/humble/Installation.html
```

Install ROS 2:

```bash
sudo apt update
sudo apt install -y ros-humble-desktop
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Install build tools:

```bash
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3-venv \
  python3-pip \
  build-essential \
  cmake \
  git
```

Initialize rosdep:

```bash
sudo rosdep init
rosdep update
```

Install ROS dependencies from the workspace:

```bash
cd <repo>/left_ws
rosdep install --from-paths src --ignore-src -r -y
```

If libfranka is installed manually instead of through `ros-humble-libfranka`, skip that rosdep key and provide the CMake path later:

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys libfranka
export LIBFRANKA_BUILD_DIR=/path/to/libfranka/build
```

Manual ROS packages sometimes needed:

```bash
sudo apt install -y \
  ros-humble-rclcpp \
  ros-humble-rclpy \
  ros-humble-std-msgs \
  ros-humble-std-srvs \
  ros-humble-sensor-msgs \
  ros-humble-geometry-msgs \
  ros-humble-tf2-ros \
  ros-humble-rosidl-default-generators \
  ros-humble-rosidl-default-runtime \
  ros-humble-rviz2
```

## libfranka

Used by:

```text
franka_control
```

Links:

```text
https://frankaemika.github.io/docs/installation_linux.html
https://support.franka.de/docs/franka_ros2.html
```

Make sure CMake can find `Franka::Franka`. One common way is:

```bash
export LIBFRANKA_BUILD_DIR=/path/to/libfranka/build
# or
export CMAKE_PREFIX_PATH=/path/to/libfranka/build:$CMAKE_PREFIX_PATH
```

Check by rebuilding:

```bash
cd <repo>/left_ws
colcon build --packages-select custom_msgs franka_control
```

## Python Virtual Environment

By default the launch file uses:

```text
~/finger_map_venv
```

Override it with:

```bash
export FINGER_MAP_VENV=/path/to/finger_map_venv
```

Create it:

```bash
python3 -m venv ~/finger_map_venv
source ~/finger_map_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install Python packages:

```bash
pip install \
  numpy \
  scipy \
  pymodbus \
  evdev \
  tqdm \
  tyro \
  pyyaml \
  lxml \
  pytransform3d \
  anytree \
  nlopt
```

PyModbus link:

```text
https://www.pymodbus.org/docs/installation
```

## dex-retargeting

Used by:

```text
inspire_retargeting
```

Link:

```text
https://github.com/dexsuite/dex-retargeting
```

Current install:

```bash
source ~/finger_map_venv/bin/activate
cd ~
git clone https://github.com/dexsuite/dex-retargeting.git
cd ~/dex-retargeting
pip install -e .
```

If example/visualization dependencies are needed:

```bash
pip install -e ".[example]"
```

Runtime code uses hand assets under:

```text
~/dex-retargeting/assets/robots/hands
```

`bringup_left.launch.py` passes this path through `dex_retargeting_robot_dir`. Override it with:

```bash
export DEX_RETARGETING_ROBOT_DIR=/path/to/dex-retargeting/assets/robots/hands
# or
ros2 launch inspire_launch bringup_left.launch.py dex_retargeting_robot_dir:=/path/to/dex-retargeting/assets/robots/hands
```

## Inspire Hand Modbus

Python package:

```bash
source ~/finger_map_venv/bin/activate
pip install pymodbus
```

Default endpoint:

```text
192.168.11.210:6000
```

Launch override:

```bash
ros2 launch inspire_launch bringup_left.launch.py \
  modbus_ip:=192.168.11.210 \
  modbus_port:=6000
```

## Foot Switch

Python package:

```bash
source ~/finger_map_venv/bin/activate
pip install evdev
```

or apt:

```bash
sudo apt install -y python3-evdev
```

Default device:

```text
/dev/input/footswitch_keyboard
```

Check:

```bash
ls -l /dev/input/footswitch_keyboard
```

## Vicon UDP

No Vicon SDK is required on the left-control PC for the current runtime path.
The node receives UDP JSON handpoints.

Default:

```text
0.0.0.0:5005
```

Launch override:

```bash
ros2 launch inspire_launch bringup_left.launch.py \
  vicon_udp_ip:=0.0.0.0 \
  vicon_udp_port:=5005
```

## Optional Camera Packages

Current left bringup does not require these. Install them only for optional camera/data-collection workflows.

RealSense:

```bash
sudo apt install -y ros-humble-realsense2-camera
```

Link:

```text
https://github.com/realsenseai/realsense-ros
```

Azure Kinect driver link:

```text
https://github.com/microsoft/Azure_Kinect_ROS_Driver
```

## Build Check

```bash
cd <repo>/left_ws
colcon build
source install/setup.bash
```

Quick C++ check:

```bash
colcon build --packages-select \
  custom_msgs inspire_interfaces franka_control franka_control_trans
```
