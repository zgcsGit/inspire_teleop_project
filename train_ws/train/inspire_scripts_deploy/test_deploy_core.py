#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from deploy_core import InspirePolicyDeploy


CKPT = os.environ.get(
    "INSPIRE_TEST_CKPT",
    "/workspace/zhicheng_ws/train/data/outputs/2026.04.24/"
    "12.40.24_train_diffusion_unet_timm_inspire_bimanual/"
    "checkpoints/epoch=0020-train_loss=0.018.ckpt",
)


def make_fake_obs():
    T = 2

    obs = {
        "camera0_rgb": np.zeros((T, 224, 224, 3), dtype=np.uint8),
        "camera1_rgb": np.zeros((T, 224, 224, 3), dtype=np.uint8),
        "camera2_rgb": np.zeros((T, 224, 224, 3), dtype=np.uint8),

        "robot0_eef_pos": np.array([[0.4, 0.2, 0.3], [0.41, 0.2, 0.3]], dtype=np.float32),
        "robot0_eef_rot_axis_angle": np.zeros((T, 3), dtype=np.float32),
        "robot0_hand_angles": np.zeros((T, 6), dtype=np.float32),
        "robot0_touch": np.zeros((T, 533), dtype=np.float32),

        "robot1_eef_pos": np.array([[0.4, -0.2, 0.3], [0.41, -0.2, 0.3]], dtype=np.float32),
        "robot1_eef_rot_axis_angle": np.zeros((T, 3), dtype=np.float32),
        "robot1_hand_angles": np.zeros((T, 6), dtype=np.float32),
        "robot1_touch": np.zeros((T, 533), dtype=np.float32),
    }

    return obs


def main():
    deploy = InspirePolicyDeploy(CKPT, device="cuda")
    obs = make_fake_obs()

    action = deploy.predict(obs)

    print("\n========== action output ==========")
    for k, v in action.items():
        print(k, v.shape)
        print(v[:2])


if __name__ == "__main__":
    main()
