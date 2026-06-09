# Left Control PC Setup and Operation

This page describes how to use the left-control computer.

## Workspace Location

In the final repository, this workspace should be placed at:

```text
inspire_teleop_project/left_ws/src
```

After cloning the repository, the source workspace is:

```text
<repo>/left_ws/src
```

## Main Packages

- `teleop_manager`: starts and stops the bringup launch when commanded by a ROS topic.
- `foot_switch`: reads the foot pedal and publishes start/stop/freeze/bringup commands.
- `franka_control`: runs the left Franka Panda Cartesian impedance controller.
- `franka_control_trans`: converts left-hand marker points into desired Franka poses.
- `inspire_hand_modbus`: sends commands to and reads feedback from the Inspire hand.
- `inspire_retargeting`: converts hand keypoints into Inspire joint states.
- `finger_map`: maps Inspire joint states to hand angle commands.
- `vicon_udp_receiver`: receives hand keypoints over UDP and publishes ROS messages.
- `custom_msgs`, `inspire_interfaces`: shared message definitions.

## Build

Install system, Python, hardware, and third-party dependencies first:

- [Environment and third-party dependencies](04_environment.md)

From the ROS workspace root:

```bash
cd <repo>/left_ws
colcon build
source install/setup.bash
```

If only the left-control packages need to be rebuilt after source changes:

```bash
cd <repo>/left_ws
colcon build --packages-select \
  custom_msgs inspire_interfaces \
  franka_control franka_control_trans \
  inspire_hand_modbus inspire_retargeting \
  finger_map foot_switch teleop_manager \
  vicon_udp_receiver inspire_launch
source install/setup.bash
```

## Python Environment

By default the launch file overlays:

```text
~/finger_map_venv
```

This environment is used by retargeting/finger-mapping nodes that depend on
`dex-retargeting` and related Python packages.

## Start Procedure

Open two terminals after sourcing the workspace.

Terminal 1:

```bash
ros2 run teleop_manager bringup_manager
```

Terminal 2:

```bash
ros2 run foot_switch footswitch_node
```

The foot switch controls the actual bringup launch through the
`/teleop/bringup_enable` topic.

## Foot Switch Controls

- Pedal C short press: toggle bringup on/off.
- Pedal A short press: call `/start_episode`.
- Pedal B short press: call `/stop_episode`.
- Pedal A long press: toggle `/teleop/left_hand_freeze`.
- Pedal B long press: toggle `/teleop/right_hand_freeze`.

Default foot switch device:

```text
/dev/input/footswitch_keyboard
```

## Bringup Launch

The bringup manager starts:

```bash
ros2 launch inspire_launch bringup_left.launch.py
```

The same launch can also be started manually for testing.

Common overrides:

```bash
ros2 launch inspire_launch bringup_left.launch.py \
  robot_ip:=12.1.1.5 \
  modbus_ip:=192.168.11.210 \
  modbus_port:=6000 \
  vicon_udp_ip:=0.0.0.0 \
  vicon_udp_port:=5005
```

## Launch Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `log_level` | `info` | ROS log level. |
| `robot_ip` | `12.1.1.5` | Left Franka robot IP. |
| `pose_input_topic` | `/Lefthandpoint` | Hand keypoint topic used for desired pose generation. |
| `pose_output_topic` | `/left/desired_pose_matrix` | Desired pose topic published to the Franka controller. |
| `initial_frame_count` | `21` | Number of hand frames collected before publishing smoothed poses. |
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
| `vicon_filter_alpha` | `0.2` | EMA smoothing factor for incoming marker positions. |
| `vicon_scale_m` | `0.001` | Conversion scale from UDP marker units to meters. |

## Launch Sequence

When bringup starts, the launch file starts these components:

1. `franka_control/franka_node`
   - Subscribes: `/left/desired_pose_matrix`
   - Publishes: `/frankaLeft/ee_pose_matrix`
   - Runs left Franka Cartesian impedance control.

2. `franka_control_trans/pose_publisher`
   - Subscribes: `/Lefthandpoint`
   - Publishes: `/left/desired_pose_matrix`
   - Converts left-hand marker frame into desired end-effector pose.

3. `inspire_hand_modbus/inspire_hand_modbus_topic`
   - Namespace: `left`
   - Command topics: `/left/set_angle_data`, `/left/set_force_data`, `/left/set_speed_data`
   - Feedback topics: `/left/angle_data`, `/left/force_data`, `/left/touch_data`

4. Hand initialization commands
   - Force: all fingers `1000`
   - Speed: all fingers `1000`
   - Angle: all fingers `1000`

5. `inspire_retargeting/pub_csv`
   - Subscribes: `/Lefthandpoint`, `/Righthandpoint`
   - Publishes: `/left_hand/joint_states`, `/right_hand/joint_states`

6. Two `finger_map/finger_mapper_node` instances
   - Left: `/left_hand/joint_states` -> `/left/set_angle_data`
   - Right: `/right_hand/joint_states` -> `/right/set_angle_data`

7. `vicon_udp_receiver/udp_lefthand_10`
   - UDP listen: `0.0.0.0:5005`
   - Publishes: `/Lefthandpoint`, `/Righthandpoint`

## Shutdown

Preferred method:

1. Press Pedal C to stop bringup.
2. Stop `footswitch_node` with `Ctrl+C`.
3. Stop `bringup_manager` with `Ctrl+C`.

The bringup manager sends SIGINT to the launch process group and escalates to
SIGTERM/SIGKILL only if the process does not stop cleanly.
