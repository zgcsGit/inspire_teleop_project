# Right Control PC

The right control PC runs the right-side ROS 2 runtime and the data-collection
pipeline. This page only describes what belongs on the machine and what each
package does. Build, dependency, recording, viewer, and launch commands are kept
in `right_ws/README.md`.

## Workspace

Only the right workspace and shared docs are needed on this computer:

```bash
git clone --filter=blob:none --sparse git@github.com:zgcsGit/inspire_teleop_project.git
cd inspire_teleop_project
git sparse-checkout set right_ws docs
```

Expected repo-local layout:

```text
inspire_teleop_project/
  right_ws/
    src/
      custom_msgs/
      foot_switch/
      franka_control/
      franka_control_trans/
      inspire_hand_modbus/
      inspire_interfaces/
      inspire_launch/
      multi_modal_data_collection/
      teleop_manager/
      teleop_viewer/
    README.md
  docs/
```

Expected external dependencies on the machine:

```text
ROS 2 Humble
libfranka / Franka ROS 2 support
RealSense runtime and ROS packages
Azure Kinect runtime and ROS packages
Inspire hand Modbus TCP network
episode output directory outside git
```

## Package Roles

- `custom_msgs`: shared custom message definitions.
- `inspire_interfaces`: Inspire hand command and feedback message definitions.
- `inspire_launch`: right-side bringup, camera, and recording launch files.
- `teleop_manager`: starts and stops the right bringup launch from a ROS topic.
- `foot_switch`: optional pedal integration for recording and bringup commands.
- `franka_control`: right Franka Cartesian impedance control.
- `franka_control_trans`: converts right-hand marker points into the desired
  right Franka end-effector pose.
- `inspire_hand_modbus`: communicates with the right Inspire hand over Modbus
  TCP.
- `multi_modal_data_collection`: records timestamped multimodal episodes.
- `teleop_viewer`: displays camera, tactile, and recorder status.

## Runtime Flow

The normal right-side runtime is:

1. Camera and recorder launch starts the wrist RealSense cameras, Azure Kinect,
   and timestamped recorder.
2. `teleop_manager` waits for bringup enable/disable commands.
3. Right bringup starts the right Franka controller, right Inspire hand Modbus
   node, and handpoint-to-pose conversion.
4. The viewer subscribes to camera streams, hand feedback, freeze flags, and
   recorder status.
5. Episode files are written outside the repository and must not be committed.

See `right_ws/README.md` for the actual installation, build, recording, and
launch commands.
