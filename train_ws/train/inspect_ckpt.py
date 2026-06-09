#!/usr/bin/env python3
import argparse
import torch
import dill
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()

    payload = torch.load(open(args.input, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    print("\n========== checkpoint ==========")
    print(args.input)

    print("\n========== cfg._target_ ==========")
    print(cfg._target_)

    print("\n========== dataset_path ==========")
    try:
        print(cfg.task.dataset.dataset_path)
    except Exception as e:
        print("N/A:", e)

    print("\n========== shape_meta ==========")
    print(OmegaConf.to_yaml(cfg.task.shape_meta))

    print("\n========== pose_repr ==========")
    try:
        print(OmegaConf.to_yaml(cfg.task.pose_repr))
    except Exception as e:
        print("N/A:", e)

    print("\n========== policy ==========")
    print(OmegaConf.to_yaml(cfg.policy))

    print("\n========== training.use_ema ==========")
    try:
        print(cfg.training.use_ema)
    except Exception as e:
        print("N/A:", e)

if __name__ == "__main__":
    main()