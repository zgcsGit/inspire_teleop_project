# Train PC

The train PC is organized as one `train_ws/` directory. It is not a left/right
ROS control workspace. It contains the deploy Docker files, deploy ROS 2
workspace, and policy training repository in their original structure.

## Workspace

Only the train workspace and shared docs are needed on this computer:

```bash
git clone --filter=blob:none --sparse git@github.com:zgcsGit/inspire_teleop_project.git
cd inspire_teleop_project
git sparse-checkout set train_ws docs
```

Expected repo-local layout:

```text
inspire_teleop_project/
  train_ws/
    docker_inspire/
    ros2_ws/
      src/
        inspire_policy_deploy/
        camera packages installed separately or omitted from git
    train/
      conda_environment.yaml
      scripts_hand_umi/
      diffusion_policy/
      hand_inspire_dataset/
      umi_dataset/
  docs/
```

Expected external dependencies on the machine:

```text
Docker with NVIDIA GPU passthrough
ROS 2 Humble inside the deploy container
Conda environment for deploy
Conda environment for training
CUDA/PyTorch runtime
RealSense runtime libraries
Azure Kinect runtime libraries
dataset storage outside git
checkpoint/log/output storage outside git
```

## Directory Roles

- `train_ws/docker_inspire`: Docker image/container files for policy deploy.
- `train_ws/ros2_ws`: ROS 2 deploy workspace used inside the container.
- `train_ws/train`: policy training code, replay-buffer generation scripts,
  configs, model code, and empty dataset directory placeholders.

## Workflow

The train computer has two separate workflows:

1. Training workflow
   - Convert episode `.npz` data into replay-buffer format.
   - Train policy code from `train_ws/train`.
   - Save checkpoints/logs outside git.
2. Deploy workflow
   - Start the deploy Docker container.
   - Source ROS 2 and the deploy workspace inside the container.
   - Start camera nodes and the policy ROS node.
   - Load policy checkpoints from an external checkpoint path.

See these files for the concrete commands:

```text
train_ws/docker_inspire/README.md
train_ws/ros2_ws/README.md
train_ws/train/README.md
```
