# Left Control Workspace

Left-control PC ROS 2 workspace. Source packages are under `src/`.

Main startup commands:

```bash
ros2 run teleop_manager bringup_manager
ros2 run foot_switch footswitch_node
```

The foot switch publishes `/teleop/bringup_enable`; `bringup_manager` starts or stops:

```bash
ros2 launch inspire_launch bringup_left.launch.py
```

Documentation:

- [Left PC setup](../docs/01_left_pc_setup.md)
- [Environment](../docs/04_environment.md)
