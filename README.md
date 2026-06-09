# Inspire Teleoperation Project

This repository contains the cleaned workspaces and documentation for the
Franka Panda + Inspire dexterous-hand teleoperation system.

The repository is split by computer role:

```text
inspire_teleop_project/
  left_ws/     # left control PC ROS 2 workspace
  right_ws/    # right control PC ROS 2 workspace and data collection
  train_ws/    # train/deploy computer: Docker, deploy ROS 2, training code
  docs/        # high-level structure and dependency maps
```

Third-party packages, datasets, checkpoints, logs, videos, bags, and generated
outputs are not vendored into this repository.

## Documentation

- [Project overview](docs/00_overview.md)
- [Left control PC](docs/01_left_pc_setup.md)
- [Right control PC](docs/02_right_pc_setup.md)
- [Train PC](docs/03_train_pc_setup.md)
- [External dependency map](docs/04_environment.md)

Concrete install, build, and startup commands live in the corresponding
workspace README:

- `left_ws/README.md`
- `right_ws/README.md`
- `train_ws/docker_inspire/README.md`
- `train_ws/ros2_ws/README.md`
- `train_ws/train/README.md`

## Clone Only One Workspace

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
