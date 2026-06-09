#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quickly inspect the structure of a recorded episode (.npz)
to see which modalities are saved and their array shapes.

Usage:
    python3 visualize_episode.py /path/to/episode_xxxxx.npz
"""

import numpy as np
import sys
import os
import json

def to_rgb_image(x):
    """
    Convert stored frame to RGB ndarray (H, W, 3), uint8
    Supports:
      - JPEG dict: {"data": uint8[...], "format": "..."}
      - ndarray:  H×W×3 (BGR or RGB)
    """
    import cv2
    import numpy as np

    # --- Case 1: JPEG dict ---
    if isinstance(x, dict) and "data" in x:
        buf = x["data"]
        img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Case 2: already ndarray ---
    if isinstance(x, np.ndarray):
        # 防御性处理
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)

        # 假设 recorder 里 tactile 已经是 RGB
        return x

    raise TypeError(f"Unsupported image type: {type(x)}")


def decode_image_sequence(arr):
    """
    arr: object array of length N
    return: np.ndarray (N, H, W, 3)
    """
    imgs = []
    for i, x in enumerate(arr):
        try:
            img = to_rgb_image(x)
            imgs.append(img)
        except Exception as e:
            print(f"⚠️ frame {i} decode failed: {e}")
            break

    # stack
    imgs = np.stack(imgs, axis=0)
    return imgs




def visualize_npz(npz_path):
    if not os.path.exists(npz_path):
        print(f"❌ File not found: {npz_path}")
        return

    print(f"📦 Inspecting file: {npz_path}")
    print(f"📁 Size: {os.path.getsize(npz_path)/1e6:.2f} MB\n")

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"🔑 Keys in file: {keys}\n")

    # --- Meta info ---
    if "meta_json" in data:
        meta = json.loads(str(data["meta_json"]))
        print("🧾 Meta information:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()
    

    # --- Show each modality ---
    for k in keys:
        if k == "meta_json":
            continue
        arr = data[k]
        if k in ("rgb2", "rgb3", "tactile"):
            arr = data[k]
            imgs = decode_image_sequence(arr)
            print(f"🖼️ '{k}' decoded -> shape={imgs.shape}, dtype={imgs.dtype}")
        if isinstance(arr, np.ndarray):
            print(f"📍 '{k}': dtype={arr.dtype}, shape={arr.shape}")
            # If it's an object array (list of frames)
            if arr.dtype == object and len(arr) > 0:
                first = arr[0]
                if isinstance(first, np.ndarray):
                    print(f"    ↳ first element: ndarray, shape={first.shape}, dtype={first.dtype}")
                else:
                    print(f"    ↳ first element type: {type(first)}")
        else:
            print(f"📍 '{k}': type={type(arr)}")

    print("\n✅ Inspection finished.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_episode.py /path/to/episode_xxxxx.npz")
        sys.exit(0)

    visualize_npz(sys.argv[1])
