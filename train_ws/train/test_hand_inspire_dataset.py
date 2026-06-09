#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
from omegaconf import OmegaConf
import hydra

from diffusion_policy.dataset.umi_dataset import UmiDataset
from torch.utils.data import DataLoader


def main():
    # 读取 task yaml，并手动包一层 task:
    cfg_task = OmegaConf.load("diffusion_policy/config/task/inspire_bimanual.yaml")
    cfg = OmegaConf.create({"task": cfg_task})
    OmegaConf.resolve(cfg)

    print("=== Task name ===")
    print(cfg.task.name)
    print()

    print("=== Dataset path ===")
    print(cfg.task.dataset_path)
    print()

    # 实例化 dataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    print("=== Dataset created successfully ===")
    print("len(dataset) =", len(dataset))
    print()

    # 取一个 sample
    sample = dataset[0]

    print("=== Sample keys ===")
    print(sample.keys())
    print()

    print("=== Obs keys ===")
    print(sample["obs"].keys())
    print()

    print("=== Obs shapes ===")
    for k, v in sample["obs"].items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    print()

    print("=== Action ===")
    print("action shape:", tuple(sample["action"].shape))
    print("action dtype:", sample["action"].dtype)
    print()

    print("=== Numeric sanity check ===")
    for k, v in sample["obs"].items():
        if torch.is_floating_point(v):
            print(f"{k}: min={v.min().item():.4f}, max={v.max().item():.4f}")
        else:
            print(f"{k}: min={v.min().item()}, max={v.max().item()}")
    print()

    if torch.is_floating_point(sample["action"]):
        print(f"action: min={sample['action'].min().item():.4f}, max={sample['action'].max().item():.4f}")
    else:
        print(f"action: min={sample['action'].min().item()}, max={sample['action'].max().item()}")


    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    print("\n=== Batch obs shapes ===")
    for k, v in batch["obs"].items():
        print(k, v.shape)

    print("batch action:", batch["action"].shape)

    for i in range(10):
        s = dataset[i]
        print(f"\n--- sample {i} ---")
        print("robot0_eef_pos:")
        print(s["obs"]["robot0_eef_pos"])
        print("robot1_eef_pos:")
        print(s["obs"]["robot1_eef_pos"]) 
    
    print("\n=== Action split check ===")
    print("robot0 action first step:", sample["action"][0, :15])
    print("robot1 action first step:", sample["action"][0, 15:])
        
if __name__ == "__main__":
    main()