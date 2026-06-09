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
      ...
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
- Right control PC: planned mirror of the left control PC for the right arm and
  right hand.
- Train PC: stores training code, scripts, configs, and datasets that should not
  be mixed into the runtime workspaces.

## Current Status

The left-control workspace has been cleaned for upload and documented. Runtime
parameters that previously required source-code edits can now be passed through
launch arguments.

See `04_environment.md` for system, Python, hardware, and third-party
dependency notes.
