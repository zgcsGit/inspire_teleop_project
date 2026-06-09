# Docker Environment

This folder builds and starts the train-PC deployment container. The container runs Ubuntu 22.04 + ROS2 Humble + CUDA runtime. The project root is mounted into the container at:

```text
/workspace/zhicheng_ws
```

## First Time: Build And Create Container

From the project root:

```bash
cd train_ws/docker_inspire
./build.sh
./run.sh
```

Inside the container, first check GPU passthrough:

```bash
nvidia-smi
```

If the GPU is not visible, fix NVIDIA container runtime on the host. Do not install a new NVIDIA driver inside the container.

## First Time: Conda Environment

Inside the container:

```bash
cd /workspace/zhicheng_ws/docker_inspire
conda create -n umi310 python=3.10 -y
conda activate umi310
```

Install core packages:

```bash
conda install -y -c pytorch -c nvidia -c conda-forge \
  pytorch=2.1.0 torchvision=0.16.0 pytorch-cuda=12.1 \
  numpy=1.24 scipy=1.11 numba=0.57 pandas=2.1 py-opencv=4.7 \
  zarr=2.16 numcodecs=0.11 hydra-core=1.2.0 dill=0.3.7 \
  einops=0.6.1 diffusers=0.18.2 timm=0.9.7 av=10.0 \
  pyyaml=6.0 tqdm=4.65 matplotlib=3.7 wandb=0.15.8 \
  python-lmdb=1.4 threadpoolctl=3.2
```

Then:

```bash
pip install imagecodecs==2023.9.18 huggingface_hub==0.16.4
pip install accelerate==0.24.0
```

ROS/Python support:

```bash
pip install empy==3.3.4 catkin_pkg lark
apt update
apt install -y \
  ros-humble-xacro \
  ros-humble-joint-state-publisher \
  ros-humble-compressed-image-transport
```

## Camera Driver Dependencies

Install camera SDK dependencies only when needed for deployment.

Azure Kinect:

- SDK repo: `https://github.com/microsoft/Azure-Kinect-Sensor-SDK`
- Debian packages used here: `libk4a1.4`, `libk4a1.4-dev`

RealSense:

- ROS2 repo: `https://github.com/realsenseai/realsense-ros/tree/ros2-legacy`
- System packages used here: `librealsense2`, `librealsense2-dev`, `librealsense2-utils`
- ROS dependency: `ros-humble-diagnostic-updater`

After plugging in cameras:

```bash
lsusb
```

should show the two RealSense cameras and the Azure Kinect.

## Verify Environment

Inside the container:

```bash
source /opt/ros/humble/setup.bash
python -c "import rclpy; print('rclpy ok')"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If a checkpoint is available:

```bash
cd /workspace/zhicheng_ws/train/inspire_scripts_deploy
INSPIRE_TEST_CKPT=/workspace/zhicheng_ws/train/data/outputs/<date>/<run>/checkpoints/latest.ckpt \
python test_deploy_core.py
```

## Exit

```bash
exit
```

## Later: Enter Existing Container

The container name defaults to `inspire_humble`.

```bash
docker start -ai inspire_humble
```

Then:

```bash
cd /workspace/zhicheng_ws
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
conda activate umi310
```

## Notes

`run.sh` mounts the parent project directory automatically. Override these when needed:

```bash
CONTAINER_NAME=inspire_humble DATA_ROOT=/mnt/data ./run.sh
```
