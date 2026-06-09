# Right PC Setup

## Workspace

```bash
cd inspire_teleop_project/right_ws
```

## Build

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

If `libfranka` is installed outside rosdep:

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys libfranka
export LIBFRANKA_BUILD_DIR=/path/to/libfranka/build
colcon build --symlink-install
```

## Runtime Workflow

Use three terminals after building and sourcing `install/setup.bash`.

### Terminal 1: Cameras And Recorder

```bash
ros2 launch inspire_launch bringup_camera_and_record.launch.py \
  out_dir:=~/inspire_teleop_data/right \
  rate_hz:=15.0 \
  slop_sec:=0.3
```

This starts:

- Left wrist RealSense: `/camera_wrist_left/color/image_raw/compressed`
- Right wrist RealSense: `/camera_wrist_right/color/image_raw/compressed`
- Azure Kinect RGB: `/ak/rgb/image_raw/compressed`
- Timestamped recorder: `multi_sensor_data_collection_with_timestamps`

The recorder exposes:

- `/start_episode`
- `/stop_episode`
- `/episode_recording`
- `/recorder_status_text`
- `/recorder_event_text`

Keep `out_dir` outside this repo. Do not commit `multi_modal_data_collection/data`, `.npz`, bag files, exported images, or experiment CSVs.

### Terminal 2: Bringup Manager

```bash
ros2 run teleop_manager bringup_manager
```

The manager listens on:

```bash
/teleop/bringup_enable
```

Start the right control chain:

```bash
ros2 topic pub --once /teleop/bringup_enable std_msgs/msg/Bool "{data: true}"
```

Stop it:

```bash
ros2 topic pub --once /teleop/bringup_enable std_msgs/msg/Bool "{data: false}"
```

The manager starts this launch:

```bash
ros2 launch inspire_launch bringup_right.launch.py \
  robot_ip:=12.1.1.6 \
  modbus_ip:=192.168.11.210 \
  modbus_port:=6000 \
  hand_ns:=right
```

Run it directly only for debugging. Optional topic overrides:

```bash
ros2 launch inspire_launch bringup_right.launch.py \
  input_hand_topic:=/Righthandpoint \
  desired_pose_topic:=/right/desired_pose_matrix \
  actual_ee_pose_topic:=/frankaRight/ee_pose_matrix
```

### Terminal 3: Viewer

```bash
ros2 run teleop_viewer single_image_viewer
```

The viewer subscribes to the three camera streams, left/right touch data, hand freeze flags, and recorder status topics.

## Main Nodes

- `franka_control/franka_node`: connects to the right Franka, moves to the right home posture, then runs Cartesian impedance control.
- `franka_control_trans/pose_publisher`: converts `/Righthandpoint` into a desired 4x4 pose matrix.
- `franka_control_trans/show_desired_matrix`: publishes TF for visualizing the desired pose.
- `inspire_hand_modbus/inspire_hand_modbus_topic`: reads and commands the Inspire hand over Modbus TCP.
- `teleop_manager/bringup_manager`: optional start/stop manager for `bringup_right.launch.py`.
- `foot_switch/footswitch_node`: optional pedal node for start/stop recording services and host bringup.
- `multi_modal_data_collection/multi_sensor_data_collection_with_timestamps`: records synchronized episode data.
- `teleop_viewer/single_image_viewer`: shows camera, tactile, and recorder state.

## Topics

- Input: `/Righthandpoint`
- Cameras: `/ak/rgb/image_raw/compressed`, `/camera_wrist_right/color/image_raw/compressed`, `/camera_wrist_left/color/image_raw/compressed`
- Desired pose: `/right/desired_pose_matrix`
- Franka actual pose: `/frankaRight/ee_pose_matrix`
- Hand feedback: `/right/force_data`, `/right/angle_data`, `/right/touch_data`
- Hand commands: `/right/set_force_data`, `/right/set_speed_data`, `/right/set_angle_data`

## Hardware Parameters

- `robot_ip`: right Franka IP, default `12.1.1.6`.
- `modbus_ip`: Inspire hand Modbus IP, default `192.168.11.210`.
- `modbus_port`: Inspire hand Modbus TCP port, default `6000`.
- `hand_ns`: hand namespace, default `right`.
- `out_dir`: recorder output directory, default `~/inspire_teleop_data/right`.
- `wrist_left_serial`: left wrist RealSense serial, default `'213322073743'`.
- `wrist_right_serial`: right wrist RealSense serial, default `'828112071102'`.
