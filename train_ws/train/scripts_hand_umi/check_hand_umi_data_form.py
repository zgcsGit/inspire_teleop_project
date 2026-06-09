# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()

z = zarr.open("hand_umi_dataset/dataset.zarr.zip", mode="r")

print("Top keys:", list(z.group_keys()))
print("Data keys:", list(z["data"].array_keys()))

print(z["data/robot0_demo_start_pose"].shape, z["data/robot0_demo_start_pose"].dtype)
print(z["data/robot0_demo_end_pose"].shape, z["data/robot0_demo_end_pose"].dtype)
print(z["data/robot0_eef_pos"].shape, z["data/robot0_eef_pos"].dtype)
print(z["data/robot0_eef_rot_axis_angle"].shape, z["data/robot0_eef_rot_axis_angle"].dtype)
print(z["data/robot0_hand_nodes"].shape, z["data/robot0_hand_nodes"].dtype)
print(z["data/camera0_rgb"].shape, z["data/camera0_rgb"].dtype)
