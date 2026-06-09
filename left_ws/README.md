# Left Control Workspace

This ROS 2 workspace runs the left Franka arm, left Inspire hand, Vicon handpoint
receiver, retargeting, finger mapping, foot switch, and bringup manager.

## Packages

- `custom_msgs`: shared custom message definitions.
- `inspire_interfaces`: Inspire hand command and feedback messages.
- `inspire_launch`: launch files for the left runtime.
- `teleop_manager`: starts/stops bringup from `/teleop/bringup_enable`.
- `foot_switch`: reads the pedal and publishes bringup, episode, and freeze
  commands.
- `franka_control`: left Franka Cartesian impedance controller.
- `franka_control_trans`: converts handpoints into desired Franka poses.
- `inspire_hand_modbus`: Inspire hand Modbus TCP command/feedback node.
- `inspire_retargeting`: converts hand keypoints into hand joint states.
- `finger_map`: maps hand joint states to Inspire angle commands.
- `vicon_udp_receiver`: receives Vicon handpoints over UDP.
- `inspire_description`: hand description and visualization assets.

## System Dependencies

Target system:

```text
Ubuntu 22.04
ROS 2 Humble
Python 3.10
```

Install ROS 2 Humble using the official instructions:

```text
https://docs.ros.org/en/humble/Installation.html
```

Install common build tools:

```bash
sudo apt update
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

Initialize `rosdep` once per machine:

```bash
sudo rosdep init
rosdep update
```

Install ROS package dependencies from this workspace:

```bash
cd <repo>/left_ws
rosdep install --from-paths src --ignore-src -r -y
```

If `libfranka` is installed manually instead of through the ROS packages, skip
that rosdep key and expose the CMake path:

```bash
cd <repo>/left_ws
rosdep install --from-paths src --ignore-src -r -y --skip-keys libfranka
export LIBFRANKA_BUILD_DIR=/path/to/libfranka/build
```

Uncertainty to verify on the actual left PC: whether `libfranka` is installed
from apt/rosdep or from a manual build. The workspace only needs CMake to find
`Franka::Franka`.

## Python Retargeting Environment

The launch file uses this venv by default:

```text
~/finger_map_venv
```

Create it:

```bash
python3 -m venv ~/finger_map_venv
source ~/finger_map_venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
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

Install `dex-retargeting` outside this repo:

```bash
source ~/finger_map_venv/bin/activate
cd ~
git clone https://github.com/dexsuite/dex-retargeting.git
cd ~/dex-retargeting
pip install -e .
```

The default hand asset directory is:

```text
~/dex-retargeting/assets/robots/hands
```

Override paths when needed:

```bash
export FINGER_MAP_VENV=/path/to/finger_map_venv
export DEX_RETARGETING_ROBOT_DIR=/path/to/dex-retargeting/assets/robots/hands
```

## Hardware Assumptions

- Left Franka robot IP: `12.1.1.5`
- Left Inspire hand Modbus endpoint: `192.168.11.210:6000`
- Vicon UDP bind address: `0.0.0.0:5005`
- Foot switch device: `/dev/input/footswitch_keyboard`

## Build

```bash
cd <repo>/left_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

Selective rebuild after source changes:

```bash
cd <repo>/left_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select \
  custom_msgs inspire_interfaces \
  franka_control franka_control_trans \
  inspire_hand_modbus inspire_retargeting \
  finger_map foot_switch teleop_manager \
  vicon_udp_receiver inspire_launch inspire_description
source install/setup.bash
```

## Normal Startup

Open two terminals and source the workspace in both:

```bash
cd <repo>/left_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Terminal 1:

```bash
ros2 run teleop_manager bringup_manager
```

Terminal 2:

```bash
ros2 run foot_switch footswitch_node
```

The foot switch publishes `/teleop/bringup_enable`; `bringup_manager` starts or
stops:

```bash
ros2 launch inspire_launch bringup_left.launch.py
```

Manual launch for testing:

```bash
ros2 launch inspire_launch bringup_left.launch.py \
  robot_ip:=12.1.1.5 \
  modbus_ip:=192.168.11.210 \
  modbus_port:=6000 \
  vicon_udp_ip:=0.0.0.0 \
  vicon_udp_port:=5005
```

## Foot Switch Controls

- Pedal C short press: toggle bringup on/off.
- Pedal A short press: call `/start_episode`.
- Pedal B short press: call `/stop_episode`.
- Pedal A long press: toggle `/teleop/left_hand_freeze`.
- Pedal B long press: toggle `/teleop/right_hand_freeze`.

## Launch Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `log_level` | `info` | ROS log level. |
| `robot_ip` | `12.1.1.5` | Left Franka robot IP. |
| `pose_input_topic` | `/Lefthandpoint` | Hand keypoint topic for pose generation. |
| `pose_output_topic` | `/left/desired_pose_matrix` | Desired pose topic for the Franka controller. |
| `initial_frame_count` | `21` | Number of hand frames collected before smoothed output. |
| `alpha_translation` | `0.3` | EMA smoothing factor for desired pose translation. |
| `alpha_rotation` | `0.2` | SLERP smoothing factor for desired pose rotation. |
| `t21_x_mm` | `66.0` | Left hand mounting offset x component, in mm. |
| `t21_y_mm` | `0.0` | Left hand mounting offset y component, in mm. |
| `t21_z_mm` | `66.0` | Left hand mounting offset z component, in mm. |
| `t0a_x_mm` | `405.0` | Tracking origin to left Franka base x offset, in mm. |
| `t0a_y_mm` | `-441.0` | Tracking origin to left Franka base y offset, in mm. |
| `t0a_z_mm` | `-90.0` | Tracking origin to left Franka base z offset, in mm. |
| `modbus_ip` | `192.168.11.210` | Inspire hand Modbus TCP IP. |
| `modbus_port` | `6000` | Inspire hand Modbus TCP port. |
| `vicon_udp_ip` | `0.0.0.0` | UDP bind address for hand keypoints. |
| `vicon_udp_port` | `5005` | UDP port for hand keypoints. |
| `vicon_filter_alpha` | `0.2` | EMA smoothing factor for marker positions. |
| `vicon_scale_m` | `0.001` | Conversion scale from UDP marker units to meters. |
| `dex_retargeting_robot_dir` | `~/dex-retargeting/assets/robots/hands` | dex-retargeting robot asset directory. |

## Launch Sequence

When bringup starts, `bringup_left.launch.py` starts:

1. `franka_control/franka_node`
   - subscribes `/left/desired_pose_matrix`
   - publishes `/frankaLeft/ee_pose_matrix`
2. `franka_control_trans/pose_publisher`
   - subscribes `/Lefthandpoint`
   - publishes `/left/desired_pose_matrix`
3. `inspire_hand_modbus/inspire_hand_modbus_topic`
   - namespace `left`
   - command topics `/left/set_angle_data`, `/left/set_force_data`,
     `/left/set_speed_data`
   - feedback topics `/left/angle_data`, `/left/force_data`,
     `/left/touch_data`
4. hand initialization commands
   - force, speed, and angle are initialized to `1000`
5. `inspire_retargeting/pub_csv`
   - subscribes `/Lefthandpoint`, `/Righthandpoint`
   - publishes `/left_hand/joint_states`, `/right_hand/joint_states`
6. two `finger_map/finger_mapper_node` instances
   - left: `/left_hand/joint_states` to `/left/set_angle_data`
   - right: `/right_hand/joint_states` to `/right/set_angle_data`
7. `vicon_udp_receiver/udp_lefthand_10`
   - UDP listen `0.0.0.0:5005`
   - publishes `/Lefthandpoint`, `/Righthandpoint`

## Shutdown

Preferred shutdown order:

1. Press Pedal C to stop bringup.
2. Stop `footswitch_node` with `Ctrl+C`.
3. Stop `bringup_manager` with `Ctrl+C`.

The bringup manager sends SIGINT to the launch process group and escalates only
if the launch process does not stop cleanly.
