# Left Control Workspace Source

ROS 2 source packages for the left-control PC.

Main startup commands:

```bash
ros2 run teleop_manager bringup_manager
ros2 run foot_switch footswitch_node
```

The foot switch toggles `/teleop/bringup_enable`. The bringup manager starts:

```bash
ros2 launch inspire_launch bringup_left.launch.py
```

Documentation:

- [Project overview](../../docs/00_overview.md)
- [Left control PC setup](../../docs/01_left_pc_setup.md)
- [Environment and dependencies](../../docs/04_environment.md)
