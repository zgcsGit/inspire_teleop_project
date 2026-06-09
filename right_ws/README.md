# Right PC Workspace

This workspace contains the self-maintained ROS 2 packages required by the right control PC.

## Packages

- `inspire_launch`: right-PC launch entry point.
- `franka_control`: right Franka Cartesian impedance controller.
- `franka_control_trans`: converts right hand keypoints to desired Franka pose matrices.
- `inspire_hand_modbus`: Modbus TCP node for the Inspire right hand.
- `inspire_interfaces`: Inspire hand and hand-keypoint message definitions.
- `custom_msgs`: shared pose/control messages.
- `multi_modal_data_collection`: timestamped multimodal episode recorder.
- `teleop_manager`: optional topic-driven launch start/stop manager.
- `teleop_viewer`: image/touch/recording-status viewer.
- `foot_switch`: optional foot pedal node for recording services and host bringup.

## Camera Driver Dependencies

Install camera SDK dependencies only when needed for deployment.

Azure Kinect:

- SDK repo: `https://github.com/microsoft/Azure-Kinect-Sensor-SDK`
- Debian packages used here: `libk4a1.4`, `libk4a1.4-dev`

RealSense:

- ROS2 repo: `https://github.com/realsenseai/realsense-ros/tree/ros2-legacy`
- System packages used here: `librealsense2`, `librealsense2-dev`, `librealsense2-utils`
- ROS dependency: `ros-humble-diagnostic-updater`

After plugging in cameras:

```bash
lsusb
```

should show the two RealSense cameras and the Azure Kinect.

## Build

```bash
cd inspire_teleop_project/right_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

If `libfranka` is installed manually and rosdep cannot resolve it:

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys libfranka
```

If `FrankaConfig.cmake` is not on the CMake path:

```bash
export LIBFRANKA_BUILD_DIR=/path/to/libfranka/build
```

## Start Right PC Workflow

Terminal 1: cameras and recorder.

```bash
source install/setup.bash
ros2 launch inspire_launch bringup_camera_and_record.launch.py \
  out_dir:=~/inspire_teleop_data/right
```

Terminal 2: manager that starts/stops the right hand and Franka bringup from `/teleop/bringup_enable`.

```bash
source install/setup.bash
ros2 run teleop_manager bringup_manager
```

To start the right hand/arm launch through the manager:

```bash
ros2 topic pub --once /teleop/bringup_enable std_msgs/msg/Bool "{data: true}"
```

To stop it:

```bash
ros2 topic pub --once /teleop/bringup_enable std_msgs/msg/Bool "{data: false}"
```

Terminal 3: viewer.

```bash
source install/setup.bash
ros2 run teleop_viewer single_image_viewer
```

The manager launches `inspire_launch bringup_right.launch.py`, which can also be run directly for debugging:

```bash
ros2 launch inspire_launch bringup_right.launch.py \
  robot_ip:=12.1.1.6 \
  modbus_ip:=192.168.11.210 \
  modbus_port:=6000 \
  hand_ns:=right
```

Important topics:

- `/teleop/bringup_enable`: start/stop command for `teleop_manager`.
- `/start_episode`, `/stop_episode`: recording services provided by the recorder.
- `/episode_recording`, `/recorder_status_text`, `/recorder_event_text`: recorder status for the viewer.
- `/ak/rgb/image_raw/compressed`: Azure Kinect RGB stream.
- `/camera_wrist_right/color/image_raw/compressed`: right wrist RealSense stream.
- `/camera_wrist_left/color/image_raw/compressed`: left wrist RealSense stream.
- `/Righthandpoint`: input hand keypoints.
- `/right/desired_pose_matrix`: desired Franka pose from hand keypoints.
- `/frankaRight/ee_pose_matrix`: actual Franka end-effector pose.
- `/right/angle_data`, `/right/force_data`, `/right/touch_data`: Inspire hand feedback.
- `/right/set_angle_data`, `/right/set_force_data`, `/right/set_speed_data`: Inspire hand commands.

Environment-variable overrides:

- `RIGHT_FRANKA_ROBOT_IP`: default Franka IP if no launch parameter is passed.
- `INSPIRE_HAND_MODBUS_IP`: default Modbus IP if no launch parameter is passed.
- `INSPIRE_HAND_MODBUS_PORT`: default Modbus TCP port if no launch parameter is passed.
- `INSPIRE_TELEOP_DATA_DIR`: default recorder output directory. Keep this outside the repository.
