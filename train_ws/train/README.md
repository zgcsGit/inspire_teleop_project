# Train Workspace

This folder keeps the original train-computer training layout. It is a heavily modified UMI / diffusion-policy workspace for Inspire bimanual policy learning.

## Workflow

```text
episode_*.npz
  -> scripts_hand_umi/generate_replay_buffer_inspire_bimanual_hand.py
  -> hand_inspire_dataset/dataset.zarr.zip
  -> train.py + diffusion_policy/config
  -> data/outputs/<date>/<run>/checkpoints/*.ckpt
  -> inspire_scripts_deploy/deploy_core.py
  -> ROS2 policy node
```

Datasets and checkpoints are not included.

## Environment

On the train PC:

```bash
cd train_ws/train
conda activate umi
```

If the environment does not exist:

```bash
mamba env create -f conda_environment.yaml
conda activate umi
```

## Dataset Conversion

Convert raw bimanual Inspire episodes:

```bash
python scripts_hand_umi/generate_replay_buffer_inspire_bimanual_hand.py \
  /mnt/data/zhicheng_ws/dataset/<session>/episode_*.npz \
  -o hand_inspire_dataset/dataset.zarr.zip \
  -or 224,224 \
  -cl 99 \
  -n 16
```

The output zarr contains:

- `camera0_rgb`: left wrist camera
- `camera1_rgb`: right wrist camera
- `camera2_rgb`: Azure Kinect RGB
- `robot0_*`: left arm / left hand
- `robot1_*`: right arm / right hand
- tactile channels split by finger/palm region

## Training

Main entry point:

```bash
python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=hand_inspire_dataset/dataset.zarr.zip
```

Memory-friendly run:

```bash
python train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task.dataset_path=hand_inspire_dataset/dataset.zarr.zip \
  dataloader.batch_size=16 \
  val_dataloader.batch_size=16 \
  training.gradient_accumulate_every=4 \
  training.num_epochs=50 \
  logging.mode=online
```

Important configs:

- `diffusion_policy/config/train_diffusion_unet_timm_umi_workspace.yaml`
- `diffusion_policy/config/task/inspire_bimanual.yaml`
- `diffusion_policy/config/train_diffusion_unet_timm_inspire_bimanual_active_workspace.yaml`
- `diffusion_policy/config/task/inspire_bimanual_active.yaml`

Training outputs are written under:

```text
data/outputs/<date>/<time>_<name>_<task_name>/
```

Do not commit output contents.

## Deploy-Time Loader

ROS2 deployment imports:

```text
inspire_scripts_deploy/deploy_core.py
```

`InspirePolicyDeploy` loads a `.ckpt`, reconstructs the Hydra workspace, builds the same observation dictionary used in training, predicts relative actions, and decodes them back to world-frame target poses and Inspire hand angles.

## Runtime Directories

Place runtime files here as needed:

- `hand_inspire_dataset/`: full bimanual `dataset.zarr.zip`
- `hand_umi_dataset/`: legacy UMI-hand experiments
- `data/outputs/`: training runs and checkpoints
- `wandb/`: local wandb cache

These directories are placeholders in git.
