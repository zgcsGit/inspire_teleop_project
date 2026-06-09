#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime
import os
import re
import sys
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


register_codecs()
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "now",
    lambda pattern: datetime.datetime.now().strftime(pattern),
    replace=True,
)


TRAIN_ROOT = os.environ.get("INSPIRE_TRAIN_ROOT", "/workspace/zhicheng_ws/train")
DEFAULT_RUN_DIR = os.environ.get(
    "INSPIRE_EVAL_RUN_DIR",
    os.path.join(
        TRAIN_ROOT,
        "data/outputs/2026.04.24/"
        "12.40.24_train_diffusion_unet_timm_inspire_bimanual",
    ),
)


ACTION_SLICES = {
    "robot0_pos_mse": slice(0, 3),
    "robot0_rot_mse": slice(3, 9),
    "robot0_hand0_mse": slice(9, 10),
    "robot0_hand1_mse": slice(10, 11),
    "robot0_hand2_mse": slice(11, 12),
    "robot0_hand3_mse": slice(12, 13),
    "robot0_hand4_mse": slice(13, 14),
    "robot0_hand5_mse": slice(14, 15),
    "robot1_pos_mse": slice(15, 18),
    "robot1_rot_mse": slice(18, 24),
    "robot1_hand0_mse": slice(24, 25),
    "robot1_hand1_mse": slice(25, 26),
    "robot1_hand2_mse": slice(26, 27),
    "robot1_hand3_mse": slice(27, 28),
    "robot1_hand4_mse": slice(28, 29),
    "robot1_hand5_mse": slice(29, 30),
}


OLD_WANDB_SLICES = {
    "old_action_mse_error": slice(0, 30),
    "old_action_mse_error_pos": slice(0, 3),
    "old_action_mse_error_rot": slice(3, 9),
    "old_action_mse_error_tail": slice(9, 30),
}


def parse_epoch(path):
    name = os.path.basename(path)
    if name == "latest.ckpt":
        return 999999
    match = re.search(r"epoch=(\d+)", name)
    if match is None:
        return -1
    return int(match.group(1))


def get_checkpoints(run_dir, include_latest):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    paths = [
        os.path.join(ckpt_dir, name)
        for name in os.listdir(ckpt_dir)
        if name.endswith(".ckpt")
    ]
    if not include_latest:
        paths = [p for p in paths if os.path.basename(p) != "latest.ckpt"]
    return sorted(paths, key=lambda p: (parse_epoch(p), os.path.basename(p)))


def update_metric(acc, name, err):
    acc[name]["sum"] += float(err.sum().item())
    acc[name]["count"] += int(err.numel())


def evaluate_checkpoint(policy, dataloader, device, max_batches):
    acc = defaultdict(lambda: {"sum": 0.0, "count": 0})

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(
            dataloader,
            desc="eval batches",
            leave=False,
        )):
            if max_batches and batch_idx >= max_batches:
                break

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            gt_action = batch["action"]
            pred_action = policy.predict_action(batch["obs"], None)["action_pred"]

            if pred_action.shape != gt_action.shape:
                raise RuntimeError(
                    f"pred_action shape {tuple(pred_action.shape)} != "
                    f"gt_action shape {tuple(gt_action.shape)}"
                )
            if pred_action.shape[-1] != 30:
                raise RuntimeError(
                    f"Expected bimanual action dim 30, got {pred_action.shape[-1]}"
                )

            sq = (pred_action - gt_action) ** 2
            update_metric(acc, "action_mse_total", sq)

            for name, sl in ACTION_SLICES.items():
                update_metric(acc, name, sq[..., sl])
            for name, sl in OLD_WANDB_SLICES.items():
                update_metric(acc, name, sq[..., sl])
            for dim in range(30):
                update_metric(acc, f"action_dim_{dim:02d}_mse", sq[..., dim])

    out = {}
    for name, item in acc.items():
        count = item["count"]
        out[name] = item["sum"] / count if count > 0 else np.nan
    return out


def save_csvs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    wide_path = os.path.join(out_dir, "bimanual_action_mse_by_checkpoint.csv")
    long_path = os.path.join(out_dir, "bimanual_action_mse_by_checkpoint_long.csv")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    ordered = ["checkpoint", "checkpoint_path", "epoch"]
    fieldnames = ordered + [key for key in fieldnames if key not in ordered]

    with open(wide_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(long_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "checkpoint", "checkpoint_path", "metric", "value"],
        )
        writer.writeheader()
        for row in rows:
            for key, value in row.items():
                if key in ordered:
                    continue
                writer.writerow({
                    "epoch": row["epoch"],
                    "checkpoint": row["checkpoint"],
                    "checkpoint_path": row["checkpoint_path"],
                    "metric": key,
                    "value": value,
                })

    print(f"Saved wide CSV: {wide_path}")
    print(f"Saved long CSV: {long_path}")
    return wide_path, long_path


def save_plot(rows, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable, skip plot: {exc}")
        return

    plot_specs = [
        ("pos_mse", ["robot0_pos_mse", "robot1_pos_mse"]),
        ("rot_mse", ["robot0_rot_mse", "robot1_rot_mse"]),
        ("hand0_mse", ["robot0_hand0_mse", "robot1_hand0_mse"]),
        ("hand1_mse", ["robot0_hand1_mse", "robot1_hand1_mse"]),
        ("hand2_mse", ["robot0_hand2_mse", "robot1_hand2_mse"]),
        ("hand3_mse", ["robot0_hand3_mse", "robot1_hand3_mse"]),
        ("hand4_mse", ["robot0_hand4_mse", "robot1_hand4_mse"]),
        ("hand5_mse", ["robot0_hand5_mse", "robot1_hand5_mse"]),
    ]

    rows = sorted(rows, key=lambda x: x["epoch"])
    epochs = [row["epoch"] for row in rows]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, (title, metrics) in zip(axes, plot_specs):
        for metric in metrics:
            values = [row.get(metric, np.nan) for row in rows]
            ax.plot(epochs, values, marker="o", label=metric.replace("_mse", ""))
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plot_path = os.path.join(out_dir, "bimanual_action_mse_8plots.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 means evaluate all batches. Use a small value for a quick smoke test.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--include-latest", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "action_mse_eval")
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)

    cfg.dataloader.num_workers = args.num_workers
    cfg.val_dataloader.num_workers = args.num_workers
    if args.num_workers == 0:
        cfg.dataloader.persistent_workers = False
        cfg.val_dataloader.persistent_workers = False

    print(f"Loading dataset from config: {cfg_path}")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    eval_dataset = dataset.get_validation_dataset() if args.split == "val" else dataset
    dataloader_cfg = cfg.val_dataloader if args.split == "val" else cfg.dataloader
    dataloader = DataLoader(eval_dataset, **dataloader_cfg)
    print(f"{args.split} dataset: {len(eval_dataset)}, dataloader: {len(dataloader)}")

    if args.checkpoint:
        ckpt_paths = [os.path.abspath(p) for p in args.checkpoint]
    else:
        ckpt_paths = get_checkpoints(run_dir, include_latest=args.include_latest)
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cls = hydra.utils.get_class(cfg._target_)
    rows = []
    for ckpt_path in ckpt_paths:
        epoch = parse_epoch(ckpt_path)
        ckpt_name = os.path.basename(ckpt_path)
        print(f"\n=== Evaluating {ckpt_name} ===")

        workspace = cls(cfg, output_dir=run_dir)
        workspace.load_checkpoint(path=ckpt_path, map_location=device)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to(device)
        policy.eval()

        metrics = evaluate_checkpoint(
            policy=policy,
            dataloader=dataloader,
            device=device,
            max_batches=args.max_batches,
        )
        row = {
            "epoch": epoch,
            "checkpoint": ckpt_name,
            "checkpoint_path": ckpt_path,
        }
        row.update(metrics)
        rows.append(row)

        print(
            f"{ckpt_name}: "
            f"robot0_pos={row['robot0_pos_mse']:.6g}, "
            f"robot1_pos={row['robot1_pos_mse']:.6g}, "
            f"robot0_hand_mean={np.mean([row[f'robot0_hand{i}_mse'] for i in range(6)]):.6g}, "
            f"robot1_hand_mean={np.mean([row[f'robot1_hand{i}_mse'] for i in range(6)]):.6g}"
        )

        del workspace
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_csvs(rows, out_dir)
    save_plot(rows, out_dir)


if __name__ == "__main__":
    main()
