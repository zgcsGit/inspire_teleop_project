#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import dill
import hydra
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.pose_util import (
    pose_to_mat,
    mat_to_pose,
    mat_to_pose10d,
    pose10d_to_mat,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)

ACTIVE_HAND_ANGLE_IDXS = [3, 4, 5]


class InspirePolicyDeploy:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = os.path.expanduser(ckpt_path)
        self.device = torch.device(device)

        payload = torch.load(
            open(self.ckpt_path, "rb"),
            map_location="cpu",
            pickle_module=dill,
        )

        self.cfg = payload["cfg"]
        self.shape_meta = self.cfg.task.shape_meta
        self.obs_pose_repr = self.cfg.task.pose_repr.obs_pose_repr
        self.action_pose_repr = self.cfg.task.pose_repr.action_pose_repr

        cls = hydra.utils.get_class(self.cfg._target_)
        workspace = cls(self.cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy = workspace.model
        if self.cfg.training.use_ema:
            self.policy = workspace.ema_model

        self.policy.num_inference_steps = self.cfg.policy.num_inference_steps
        self.disable_random_image_transforms()
        self.policy.eval().to(self.device)

        self.num_robot = 2
        self.action_dim = int(self.shape_meta.action.shape[0])
        if self.action_dim % self.num_robot != 0:
            raise ValueError(
                f"Action dim {self.action_dim} is not divisible by {self.num_robot}"
            )
        self.action_dim_per_robot = self.action_dim // self.num_robot
        self.hand_dim_per_robot = self.action_dim_per_robot - 9
        if self.hand_dim_per_robot not in (3, 6):
            raise ValueError(
                f"Unsupported hand action dim per robot: {self.hand_dim_per_robot}"
            )
        self.obs_meta = self.shape_meta.obs

        print("Loaded checkpoint:", self.ckpt_path)
        print("obs_pose_repr:", self.obs_pose_repr)
        print("action_pose_repr:", self.action_pose_repr)
        print("action shape:", self.shape_meta.action.shape)
        print("action horizon:", self.shape_meta.action.horizon)
        print("action dim per robot:", self.action_dim_per_robot)
        print("hand dim per robot:", self.hand_dim_per_robot)

    def disable_random_image_transforms(self):
        obs_encoder = getattr(self.policy, "obs_encoder", None)
        if obs_encoder is None or not hasattr(obs_encoder, "key_transform_map"):
            print("No image transform map found; deploy image transforms unchanged.")
            return

        image_size = None
        for attr in self.shape_meta.obs.values():
            if attr.get("type", "low_dim") == "rgb":
                image_size = int(attr.shape[-1])
                break

        if image_size is None:
            print("No rgb obs found; deploy image transforms unchanged.")
            return

        crop_ratio = 1.0
        transforms_cfg = self.cfg.policy.obs_encoder.get("transforms", None)
        if transforms_cfg is not None and len(transforms_cfg) > 0:
            first = transforms_cfg[0]
            if first.get("type", None) == "RandomCrop":
                crop_ratio = float(first.get("ratio", 1.0))

        crop_size = max(1, int(image_size * crop_ratio))
        for key in list(obs_encoder.key_transform_map.keys()):
            obs_encoder.key_transform_map[key] = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=crop_size),
                torchvision.transforms.Resize(size=image_size, antialias=True),
            )

        print(
            "Deploy image transforms set to deterministic "
            f"CenterCrop({crop_size}) + Resize({image_size}); random ColorJitter disabled."
        )

    @staticmethod
    def split_touch(touch_533: np.ndarray):
        """
        touch_533: (T, 533)
        return six tactile parts.
        """
        assert touch_533.shape[-1] == 533, touch_533.shape

        return {
            "pinky": touch_533[..., 0:93],
            "ring": touch_533[..., 93:186],
            "middle": touch_533[..., 186:279],
            "index": touch_533[..., 279:372],
            "thumb": touch_533[..., 372:477],
            "palm": touch_533[..., 477:533],
        }

    def has_obs_key(self, key):
        return key in self.obs_meta

    def obs_dim(self, key):
        return int(self.obs_meta[key].shape[0])

    def build_obs_dict(self, obs_world: dict, episode_start_pose_mats=None):
        """
        obs_world 输入原始世界系观测。

        Required keys:
            camera0_rgb, camera1_rgb, camera2_rgb: (T,H,W,3), uint8 or float
            robot0_eef_pos: (T,3)
            robot0_eef_rot_axis_angle: (T,3)
            robot0_hand_angles: (T,6)
            robot0_touch: (T,533)
            robot1_eef_pos: (T,3)
            robot1_eef_rot_axis_angle: (T,3)
            robot1_hand_angles: (T,6)
            robot1_touch: (T,533)
        """
        obs_dict = {}

        # ---------- images ----------
        for cam_key in ["camera0_rgb", "camera1_rgb", "camera2_rgb"]:
            if not self.has_obs_key(cam_key):
                continue
            img = obs_world[cam_key]
            assert img.ndim == 4, (cam_key, img.shape)

            # expected input: T,H,W,C
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)

            # T,H,W,C -> T,C,H,W
            obs_dict[cam_key] = np.moveaxis(img, -1, 1)

        # ---------- direct low-dim ----------
        for robot_id in range(self.num_robot):
            hand_key = f"robot{robot_id}_hand_angles"
            if self.has_obs_key(hand_key):
                hand = obs_world[hand_key].astype(np.float32)
                expected_dim = self.obs_dim(hand_key)
                if expected_dim == 3 and hand.shape[-1] == 6:
                    hand = hand[..., ACTIVE_HAND_ANGLE_IDXS]
                elif hand.shape[-1] != expected_dim:
                    raise ValueError(
                        f"{hand_key} expected dim {expected_dim}, got {hand.shape}"
                    )
                obs_dict[hand_key] = hand.astype(np.float32)

            touch_parts = self.split_touch(
                obs_world[f"robot{robot_id}_touch"].astype(np.float32)
            )
            for name, value in touch_parts.items():
                touch_key = f"robot{robot_id}_touch_{name}"
                if self.has_obs_key(touch_key):
                    obs_dict[touch_key] = value.astype(np.float32)

        # ---------- eef pose: world pose6 -> relative pose10d ----------
        current_pose_mats = []

        for robot_id in range(self.num_robot):
            pos = obs_world[f"robot{robot_id}_eef_pos"].astype(np.float32)
            rot = obs_world[f"robot{robot_id}_eef_rot_axis_angle"].astype(np.float32)
            pose6 = np.concatenate([pos, rot], axis=-1)

            pose_mat = pose_to_mat(pose6)
            current_pose_mats.append(pose_mat[-1])

            obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False,
            )
            obs_pose10d = mat_to_pose10d(obs_pose_mat)

            pos_key = f"robot{robot_id}_eef_pos"
            rot_key = f"robot{robot_id}_eef_rot_axis_angle"
            if self.has_obs_key(pos_key):
                obs_dict[pos_key] = obs_pose10d[:, :3].astype(np.float32)
            if self.has_obs_key(rot_key):
                obs_dict[rot_key] = obs_pose10d[:, 3:].astype(np.float32)

        # ---------- robot0 wrt robot1, robot1 wrt robot0 ----------
        for robot_id, other_robot_id in [(0, 1), (1, 0)]:
            pos = obs_world[f"robot{robot_id}_eef_pos"].astype(np.float32)
            rot = obs_world[f"robot{robot_id}_eef_rot_axis_angle"].astype(np.float32)
            pose6 = np.concatenate([pos, rot], axis=-1)
            pose_mat = pose_to_mat(pose6)

            other_pos = obs_world[f"robot{other_robot_id}_eef_pos"].astype(np.float32)
            other_rot = obs_world[f"robot{other_robot_id}_eef_rot_axis_angle"].astype(np.float32)
            other_pose6 = np.concatenate([other_pos, other_rot], axis=-1)
            other_pose_mat = pose_to_mat(other_pose6)

            rel_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=other_pose_mat[-1],
                pose_rep="relative",
                backward=False,
            )
            rel_pose10d = mat_to_pose10d(rel_pose_mat)

            pos_key = f"robot{robot_id}_eef_pos_wrt{other_robot_id}"
            rot_key = f"robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}"
            if self.has_obs_key(pos_key):
                obs_dict[pos_key] = rel_pose10d[:, :3].astype(np.float32)
            if self.has_obs_key(rot_key):
                obs_dict[rot_key] = rel_pose10d[:, 3:].astype(np.float32)

        # ---------- wrt episode start ----------
        # Training uses the episode start pose as the reference for this feature.
        # Real deploy should pass a pose captured when policy execution starts.
        if episode_start_pose_mats is None:
            episode_start_pose_mats = current_pose_mats

        for robot_id in range(self.num_robot):
            pos = obs_world[f"robot{robot_id}_eef_pos"].astype(np.float32)
            rot = obs_world[f"robot{robot_id}_eef_rot_axis_angle"].astype(np.float32)
            pose6 = np.concatenate([pos, rot], axis=-1)
            pose_mat = pose_to_mat(pose6)

            rel_start_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=episode_start_pose_mats[robot_id],
                pose_rep="relative",
                backward=False,
            )
            rel_start10d = mat_to_pose10d(rel_start_mat)

            rot_key = f"robot{robot_id}_eef_rot_axis_angle_wrt_start"
            if self.has_obs_key(rot_key):
                obs_dict[rot_key] = rel_start10d[:, 3:].astype(np.float32)

        return obs_dict, current_pose_mats

    @staticmethod
    def active_hand_to_hand6(hand_active):
        hand_active = np.asarray(hand_active, dtype=np.float32)
        hand6 = np.full(
            hand_active.shape[:-1] + (6,),
            1000.0,
            dtype=np.float32,
        )
        hand6[..., ACTIVE_HAND_ANGLE_IDXS] = hand_active
        return hand6

    def decode_action_to_world(self, action_pred: np.ndarray, current_pose_mats):
        """
        action_pred:
            old full-hand model: (Ta, 30) = (pose9 + hand6) * 2
            active-hand model:   (Ta, 24) = (pose9 + hand3) * 2
        output dict always uses world pose6 and ROS hand6.
        """
        assert action_pred.ndim == 2, action_pred.shape
        assert action_pred.shape[-1] == self.action_dim, action_pred.shape

        output = {}

        for robot_id in range(self.num_robot):
            base = robot_id * self.action_dim_per_robot

            rel_pose10d = action_pred[:, base:base + 9]
            hand = action_pred[
                :,
                base + 9:base + 9 + self.hand_dim_per_robot,
            ]
            if self.hand_dim_per_robot == 3:
                hand6 = self.active_hand_to_hand6(hand)
            else:
                hand6 = hand

            rel_action_mat = pose10d_to_mat(rel_pose10d)

            world_action_mat = convert_pose_mat_rep(
                rel_action_mat,
                base_pose_mat=current_pose_mats[robot_id],
                pose_rep=self.action_pose_repr,
                backward=True,
            )

            world_pose6 = mat_to_pose(world_action_mat)

            output[f"robot{robot_id}_target_pose"] = world_pose6.astype(np.float32)
            output[f"robot{robot_id}_hand_angles"] = hand6.astype(np.float32)

        return output

    @torch.no_grad()
    def predict(
        self,
        obs_world: dict,
        current_pose_mats_override=None,
        episode_start_pose_mats=None,
    ):
        """
        输入 obs_world，输出 action_world。

        If current_pose_mats_override is provided, the policy still observes
        obs_world, but relative actions are decoded around the supplied live
        robot poses. This is useful for replaying dataset observations on the
        real robot without commanding the dataset's absolute world poses.
        """
        obs_dict_np, current_pose_mats = self.build_obs_dict(
            obs_world,
            episode_start_pose_mats=episode_start_pose_mats,
        )
        if current_pose_mats_override is not None:
            current_pose_mats = current_pose_mats_override

        obs_dict = dict_apply(
            obs_dict_np,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
        )

        self.policy.reset()
        result = self.policy.predict_action(obs_dict)

        action_pred = result["action_pred"][0].detach().cpu().numpy()

        action_world = self.decode_action_to_world(
            action_pred=action_pred,
            current_pose_mats=current_pose_mats,
        )

        return action_world
