# Train PC Setup

Status: not整理 yet. The training machine code was not整理 from this right PC
workspace.

To fill after scanning the train computer:

- workspace/repo path
- Python/conda environment
- training scripts
- config location
- dataset location
- model output location
- commands for training/evaluation
- files that should not be committed

Expected repository locations:

```text
inspire_teleop_project/train/scripts
inspire_teleop_project/train/configs
inspire_teleop_project/train/src
```

Keep generated datasets, videos, large `.npz` files, checkpoints, and
experiment outputs outside git unless they are intentionally versioned through a
data or artifact system.
