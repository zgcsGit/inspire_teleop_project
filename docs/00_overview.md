# Inspire Teleoperation Project Overview

This repository is the integration repo for the dual-arm Franka Panda and
Inspire dexterous-hand teleoperation system.

The three computers have different jobs. The left and right control computers
run ROS 2 control/data-collection workspaces. The train computer keeps the
Docker environment, deploy ROS 2 workspace, and policy training code together
under `train_ws/`.

## Repository Layout

After third-party packages are installed on the target machines, the repository
is expected to look like this:

```text
inspire_teleop_project/
  left_ws/
    src/
      custom_msgs/
      finger_map/
      foot_switch/
      franka_control/
      franka_control_trans/
      inspire_description/
      inspire_hand_modbus/
      inspire_interfaces/
      inspire_launch/
      inspire_retargeting/
      teleop_manager/
      vicon_udp_receiver/
    README.md

  right_ws/
    src/
      custom_msgs/
      franka_control/
      franka_control_trans/
      inspire_hand_modbus/
      inspire_interfaces/
      inspire_launch/
      multi_modal_data_collection/
      teleop_manager/
      teleop_viewer/
    README.md

  train_ws/
    docker_inspire/
    ros2_ws/
    train/

  docs/
    00_overview.md
    01_left_pc_setup.md
    02_right_pc_setup.md
    03_train_pc_setup.md
    04_environment.md
```

External libraries such as ROS 2 Humble, libfranka, dex-retargeting, RealSense,
Azure Kinect, CUDA, Conda, and Docker are installed on the machines themselves.
They are documented as dependencies, but they are not vendored into this repo.

## Computer Roles

- Left control PC: left Franka control, left Inspire hand Modbus control, Vicon
  UDP handpoint input, retargeting, finger mapping, foot switch, and bringup
  manager.
- Right control PC: right Franka control, right Inspire hand Modbus control,
  camera capture, episode recording, viewer, foot switch integration, and
  bringup manager.
- Train PC: replay-buffer generation, policy training, policy checkpoint
  loading, deploy Docker image/container, deploy ROS 2 node, and camera runtime
  support inside the container.

## Clone Only One Workspace

Each machine can clone only the part it needs with git sparse checkout.

Left control PC:

```bash
git clone --filter=blob:none --sparse git@github.com:zgcsGit/inspire_teleop_project.git
cd inspire_teleop_project
git sparse-checkout set left_ws docs
```

Right control PC:

```bash
git clone --filter=blob:none --sparse git@github.com:zgcsGit/inspire_teleop_project.git
cd inspire_teleop_project
git sparse-checkout set right_ws docs
```

Train PC:

```bash
git clone --filter=blob:none --sparse git@github.com:zgcsGit/inspire_teleop_project.git
cd inspire_teleop_project
git sparse-checkout set train_ws docs
```

The concrete build, environment, and launch commands live in the README of each
workspace, for example `left_ws/README.md`.
