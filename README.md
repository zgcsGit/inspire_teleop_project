# Inspire Teleoperation Project

This repository contains the runtime workspaces and documentation for the
Franka Panda + Inspire dexterous-hand teleoperation system.

Current uploaded content includes the cleaned left-control workspace and the
right-control workspace organized from the right PC.

## Layout

```text
inspire_teleop_project/
  left_ws/
    src/
  right_ws/
    src/
  train/
  docs/
```

## Documentation

- [Project overview](docs/00_overview.md)
- [Left control PC setup](docs/01_left_pc_setup.md)
- [Right control PC setup](docs/02_right_pc_setup.md)
- [Train PC setup](docs/03_train_pc_setup.md)
- [Environment and dependencies](docs/04_environment.md)

## Right PC Quick Start

```bash
cd inspire_teleop_project/right_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch inspire_launch bringup_camera_and_record.launch.py
```

The right PC normally uses three terminals: camera/record launch, bringup
manager, and viewer. See `docs/02_right_pc_setup.md` for the full workflow.

## Sparse Checkout

To download only the left-control workspace and documentation:

```bash
git clone --filter=blob:none --sparse <repo_url>
cd inspire_teleop_project
git sparse-checkout set left_ws docs
```

For the right-control PC:

```bash
git sparse-checkout set right_ws docs
```

For the training PC:

```bash
git sparse-checkout set train docs
```
