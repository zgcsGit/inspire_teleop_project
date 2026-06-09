# Left Control PC

The left control PC runs the left-side ROS 2 runtime. This page only describes
what belongs on the machine and what each package does. Build, dependency, and
startup commands are kept in `left_ws/README.md`.

## Workspace

Only the left workspace and shared docs are needed on this computer:

```bash
git clone --filter=blob:none --sparse git@github.com:zgcsGit/inspire_teleop_project.git
cd inspire_teleop_project
git sparse-checkout set left_ws docs
```

Expected repo-local layout:

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
  docs/
```

Expected external dependencies on the machine:

```text
ROS 2 Humble
libfranka / Franka ROS 2 support
~/finger_map_venv
~/dex-retargeting
Vicon UDP handpoint source
Inspire hand Modbus TCP network
foot switch device
```

## Package Roles

- `custom_msgs`: shared custom message definitions used by control nodes.
- `inspire_interfaces`: Inspire hand command and feedback message definitions.
- `inspire_launch`: left-side bringup launch files.
- `teleop_manager`: starts and stops the left bringup launch from a ROS topic.
- `foot_switch`: reads the foot pedal and publishes episode/freeze/bringup
  commands.
- `franka_control`: left Franka Cartesian impedance control.
- `franka_control_trans`: converts left-hand marker points into the desired
  left Franka end-effector pose.
- `inspire_hand_modbus`: communicates with the left Inspire hand over Modbus
  TCP.
- `inspire_retargeting`: converts Vicon hand keypoints into hand joint states.
- `finger_map`: maps retargeted joint states into Inspire hand angle commands.
- `vicon_udp_receiver`: receives Vicon handpoint data over UDP and publishes ROS
  topics.
- `inspire_description`: Inspire hand description and visualization assets.

## Runtime Flow

The normal left-side runtime is:

1. `teleop_manager` waits for bringup enable/disable commands.
2. `foot_switch` publishes bringup, episode, and freeze commands.
3. `bringup_left.launch.py` starts the left Franka controller, Inspire hand
   Modbus node, Vicon UDP receiver, retargeting, and finger mapping nodes.
4. Vicon handpoints become desired left Franka poses and Inspire hand angles.
5. The foot switch stops bringup cleanly when the operator disables it.

See `left_ws/README.md` for the actual installation, build, and launch commands.
