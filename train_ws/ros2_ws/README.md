# ROS2 Deployment Workspace

This workspace contains project-owned ROS2 packages:

- `src/inspire_policy_deploy`: policy inference and replay nodes.
- `src/custom_msgs`: robot pose/control message definitions.
- `src/inspire_interfaces`: Inspire hand message definitions.

Third-party camera drivers are installed separately:

- `src/realsense-ros`
- `src/Azure_Kinect_ROS_Driver`

## Build

Inside the Docker container:

```bash
cd /workspace/zhicheng_ws
source /opt/ros/humble/setup.bash
conda activate umi310
colcon build --symlink-install --base-paths ros2_ws/src
source ros2_ws/install/setup.bash
```

If the camera driver packages are present under `ros2_ws/src`, they will be built as part of the workspace.

## Start Cameras

Open one terminal per camera. In every terminal:

```bash
cd /workspace/zhicheng_ws
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
conda activate umi310
```

Left wrist RealSense:

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera_wrist_left \
  serial_no:="'213322073743'" \
  enable_depth:=false \
  rgb_camera.profile:="960,540,30"
```

Right wrist RealSense:

```bash
ros2 launch realsense2_camera rs_launch.py \
  camera_name:=camera_wrist_right \
  serial_no:="'828112071102'" \
  enable_depth:=false \
  rgb_camera.profile:="960,540,30"
```

Azure Kinect:

```bash
ros2 launch azure_kinect_ros_driver driver.launch.py \
  depth_enabled:=false \
  color_enabled:=true \
  point_cloud:=false \
  rgb_point_cloud:=false
```

Expected image topics:

```text
/camera_wrist_left/color/image_raw/compressed
/camera_wrist_right/color/image_raw/compressed
/ak/rgb/image_raw/compressed
```

## Start Policy Node

In another terminal:

```bash
cd /workspace/zhicheng_ws
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
conda activate umi310
```

Start policy deployment: 
before start deploy need to check every topic is published into ros topic space. include franka and inspire control nodes, which should be launch or run on ohter two computers. (default $ROS_DOMAIN_ID=7)

```bash
ros2 run inspire_policy_deploy policy_ros_node --ros-args \
  -p run_policy_after_init:=true \
  -p ckpt_path:=/workspace/zhicheng_ws/train/data/outputs/<date>/<run>/checkpoints/latest.ckpt \
  -p init_episode_path:=/mnt/data/zhicheng_ws/dataset/<session>/episode_xxx.npz
```

Direct policy mode without the initialization phase:

```bash
ros2 run inspire_policy_deploy policy_ros_node --ros-args \
  -p enable_init_pose:=false \
  -p ckpt_path:=/workspace/zhicheng_ws/train/data/outputs/<date>/<run>/checkpoints/latest.ckpt
```

## Quick Observation Check

```bash
ros2 run inspire_policy_deploy obs_monitor_node
```

## Replay Dataset Through Policy

This is useful before commanding the real robot:

```bash
ros2 run inspire_policy_deploy dataset_obs_replay_node --ros-args \
  -p ckpt_path:=/workspace/zhicheng_ws/train/data/outputs/<date>/<run>/checkpoints/latest.ckpt \
  -p dataset_path:=/workspace/zhicheng_ws/train/hand_inspire_dataset/dataset.zarr.zip
```

The replay node defaults to not publishing real commands.
