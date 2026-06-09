# Inspire Teleoperation Project Overview

This project contains the code and operating notes for the dual-arm Franka Panda
and Inspire dexterous-hand teleoperation system.

## Repository Layout

The planned monorepo layout is:

```text
inspire_teleop_project/
  left_ws/
    src/
      custom_msgs/
      inspire_interfaces/
      inspire_launch/
      teleop_manager/
      foot_switch/
      finger_map/
      franka_control/
      franka_control_trans/
      inspire_hand_modbus/
      inspire_retargeting/
      vicon_udp_receiver/
      inspire_description/
      ...
    README.md

  right_ws/
    src/
      custom_msgs/
      inspire_interfaces/
      inspire_launch/
      teleop_manager/
      foot_switch/
      franka_control/
      franka_control_trans/
      inspire_hand_modbus/
      multi_modal_data_collection/
      teleop_viewer/
    README.md

  train/
    README.md
    scripts/
    configs/
    src/

  docs/
    00_overview.md
    01_left_pc_setup.md
    02_right_pc_setup.md
    03_train_pc_setup.md
    04_environment.md

  .gitignore
  README.md
```

## Computer Roles

- Left control PC: runs the left Franka controller, left Inspire Modbus node,
  hand retargeting, finger mapping, Vicon UDP receiver, bringup manager, and
  foot switch node.
- Right control PC: runs the right Franka controller, right Inspire Modbus node,
  hand-to-pose conversion, camera/episode recording, bringup manager, foot
  switch integration, and viewer.
- Train PC: stores training code, scripts, configs, and datasets that should not
  be mixed into the runtime workspaces.

## Current Status

The left-control workspace has been cleaned for upload and documented. The
right-control workspace has also been added from the right PC整理 result.
Runtime parameters that previously required source-code edits can now be passed
through launch arguments.

Right PC startup flow:

1. Terminal 1 runs `inspire_launch/bringup_camera_and_record.launch.py`.
2. That launch starts two RealSense cameras, Azure Kinect, and the timestamped
   recorder.
3. Terminal 2 runs `teleop_manager/bringup_manager`, which starts/stops
   `bringup_right.launch.py` from `/teleop/bringup_enable`.
4. `bringup_right.launch.py` starts the right Franka, right hand Modbus node,
   and hand-to-pose conversion.
5. Terminal 3 runs `teleop_viewer/single_image_viewer` to display cameras,
   touch data, and recording status.

See `04_environment.md` for system, Python, hardware, and third-party
dependency notes.
