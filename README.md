# Inspire Teleoperation Project

This repository contains the runtime workspaces and documentation for the
Franka Panda + Inspire dexterous-hand teleoperation system.

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
