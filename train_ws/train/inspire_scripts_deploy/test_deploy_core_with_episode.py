#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys

import numpy as np
import zarr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from deploy_core import InspirePolicyDeploy


register_codecs()


TRAIN_ROOT = os.environ.get("INSPIRE_TRAIN_ROOT", "/workspace/zhicheng_ws/train")
DEFAULT_CKPT = os.environ.get(
    "INSPIRE_TEST_CKPT",
    os.path.join(
        TRAIN_ROOT,
        "data/outputs/2026.04.24/"
        "12.40.24_train_diffusion_unet_timm_inspire_bimanual/"
        "checkpoints/epoch=0020-train_loss=0.018.ckpt",
    ),
)
DEFAULT_DATASET = os.environ.get(
    "INSPIRE_TEST_DATASET",
    os.path.join(TRAIN_ROOT, "hand_inspire_dataset/dataset.zarr.zip"),
)
DEFAULT_OUT_DIR = os.environ.get(
    "INSPIRE_TEST_OUT_DIR",
    os.path.join(TRAIN_ROOT, "inspire_scripts_deploy/episode_eval_outputs"),
)

DOWNSAMPLE = 3
OBS_HORIZON = 2
ACTION_HORIZON = 16
ACTION_KEYS = [
    "robot0_target_pose",
    "robot1_target_pose",
    "robot0_hand_angles",
    "robot1_hand_angles",
]


def split_episode_indices(replay_buffer, episode_idx):
    ends = replay_buffer.episode_ends[:]
    start = 0 if episode_idx == 0 else ends[episode_idx - 1]
    end = ends[episode_idx]
    return int(start), int(end)


def get_obs_indices(t_idx, ep_start):
    idx = t_idx - np.arange(OBS_HORIZON - 1, -1, -1) * DOWNSAMPLE
    idx = np.clip(idx, ep_start, t_idx)
    return idx.astype(np.int64)


def get_action_indices(t_idx, ep_end):
    idx = t_idx + np.arange(ACTION_HORIZON) * DOWNSAMPLE
    idx = np.clip(idx, t_idx, ep_end - 1)
    return idx.astype(np.int64)


def make_obs_world(rb, obs_idx):
    obs = {}

    for cam in ["camera0_rgb", "camera1_rgb", "camera2_rgb"]:
        obs[cam] = rb.data[cam][obs_idx]

    for rid in [0, 1]:
        obs[f"robot{rid}_eef_pos"] = rb.data[
            f"robot{rid}_eef_pos"
        ][obs_idx].astype(np.float32)
        obs[f"robot{rid}_eef_rot_axis_angle"] = rb.data[
            f"robot{rid}_eef_rot_axis_angle"
        ][obs_idx].astype(np.float32)
        obs[f"robot{rid}_hand_angles"] = rb.data[
            f"robot{rid}_hand_angles"
        ][obs_idx].astype(np.float32)

        touch = np.concatenate([
            rb.data[f"robot{rid}_touch_pinky"][obs_idx],
            rb.data[f"robot{rid}_touch_ring"][obs_idx],
            rb.data[f"robot{rid}_touch_middle"][obs_idx],
            rb.data[f"robot{rid}_touch_index"][obs_idx],
            rb.data[f"robot{rid}_touch_thumb"][obs_idx],
            rb.data[f"robot{rid}_touch_palm"][obs_idx],
        ], axis=-1).astype(np.float32)
        obs[f"robot{rid}_touch"] = touch

    return obs


def build_gt_action(rb, action_idx):
    result = {}
    for rid in [0, 1]:
        result[f"robot{rid}_target_pose"] = np.concatenate([
            rb.data[f"robot{rid}_eef_pos"][action_idx].astype(np.float32),
            rb.data[f"robot{rid}_eef_rot_axis_angle"][action_idx].astype(np.float32),
        ], axis=-1)
        result[f"robot{rid}_hand_angles"] = rb.data[
            f"robot{rid}_hand_angles"
        ][action_idx].astype(np.float32)
    return result


def cosine(a, b, eps=1e-8):
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < eps or b_norm < eps:
        return np.nan
    return float(np.dot(a, b) / (a_norm * b_norm))


def evaluate_one_timestep(deploy, rb, t_idx, ep_start, ep_end, compare_step):
    obs_idx = get_obs_indices(t_idx, ep_start)
    action_idx = get_action_indices(t_idx, ep_end)
    obs_world = make_obs_world(rb, obs_idx)
    gt_action = build_gt_action(rb, action_idx)
    pred_action = deploy.predict(obs_world)

    row = {
        "t_idx": int(t_idx),
        "obs_idx0": int(obs_idx[0]),
        "obs_idx1": int(obs_idx[-1]),
    }

    for rid in [0, 1]:
        pose_key = f"robot{rid}_target_pose"
        hand_key = f"robot{rid}_hand_angles"

        current_pos = obs_world[f"robot{rid}_eef_pos"][-1]
        pred_pose = pred_action[pose_key]
        gt_pose = gt_action[pose_key]
        pred_hand = pred_action[hand_key]
        gt_hand = gt_action[hand_key]

        horizon_pos_err = np.linalg.norm(
            pred_pose[:, :3] - gt_pose[:, :3], axis=-1
        )
        horizon_rot_err = np.linalg.norm(
            pred_pose[:, 3:6] - gt_pose[:, 3:6], axis=-1
        )
        horizon_hand_mae = np.mean(np.abs(pred_hand - gt_hand), axis=-1)

        step = min(compare_step, len(pred_pose) - 1)
        pred_delta = pred_pose[step, :3] - current_pos
        gt_delta = gt_pose[step, :3] - current_pos

        row[f"robot{rid}_step_pos_err"] = float(horizon_pos_err[step])
        row[f"robot{rid}_horizon_pos_err_mean"] = float(np.mean(horizon_pos_err))
        row[f"robot{rid}_horizon_rot_err_mean"] = float(np.mean(horizon_rot_err))
        row[f"robot{rid}_step_hand_mae"] = float(horizon_hand_mae[step])
        row[f"robot{rid}_horizon_hand_mae_mean"] = float(np.mean(horizon_hand_mae))
        row[f"robot{rid}_delta_cos"] = cosine(pred_delta, gt_delta)
        row[f"robot{rid}_pred_step_dx"] = float(pred_delta[0])
        row[f"robot{rid}_pred_step_dy"] = float(pred_delta[1])
        row[f"robot{rid}_pred_step_dz"] = float(pred_delta[2])
        row[f"robot{rid}_gt_step_dx"] = float(gt_delta[0])
        row[f"robot{rid}_gt_step_dy"] = float(gt_delta[1])
        row[f"robot{rid}_gt_step_dz"] = float(gt_delta[2])

    return row, pred_action, gt_action


def print_summary(rows):
    print("\n========== episode summary ==========")
    for rid in [0, 1]:
        prefix = f"robot{rid}"
        for key in [
            "step_pos_err",
            "horizon_pos_err_mean",
            "horizon_rot_err_mean",
            "step_hand_mae",
            "horizon_hand_mae_mean",
            "delta_cos",
        ]:
            values = np.array([
                row[f"{prefix}_{key}"] for row in rows
            ], dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                print(f"{prefix}_{key}: no finite values")
                continue
            print(
                f"{prefix}_{key}: "
                f"mean={np.mean(values):.6f}, "
                f"median={np.median(values):.6f}, "
                f"p90={np.percentile(values, 90):.6f}"
            )


def save_outputs(out_dir, rows, pred_samples, gt_samples):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    npz_data = {
        "t_idx": np.array([row["t_idx"] for row in rows], dtype=np.int64),
    }
    for key in ACTION_KEYS:
        npz_data[f"pred_{key}"] = np.stack([sample[key] for sample in pred_samples])
        npz_data[f"gt_{key}"] = np.stack([sample[key] for sample in gt_samples])

    np.savez_compressed(os.path.join(out_dir, "episode_eval.npz"), **npz_data)
    print(f"\nSaved metrics to {csv_path}")
    print(f"Saved arrays to {os.path.join(out_dir, 'episode_eval.npz')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-offset", type=int, default=DOWNSAMPLE)
    parser.add_argument("--end-margin", type=int, default=(ACTION_HORIZON - 1) * DOWNSAMPLE)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--compare-step", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
    )
    args = parser.parse_args()

    print("Loading policy...")
    deploy = InspirePolicyDeploy(args.ckpt, device=args.device)

    print("Loading replay buffer...")
    with zarr.ZipStore(args.dataset, mode="r") as store:
        rb = ReplayBuffer.create_from_group(zarr.group(store))

        ep_start, ep_end = split_episode_indices(rb, args.episode)
        first_t = ep_start + args.start_offset
        last_t = ep_end - args.end_margin
        if first_t >= last_t:
            raise RuntimeError(
                f"Episode too short for evaluation: start={ep_start}, end={ep_end}, "
                f"first_t={first_t}, last_t={last_t}"
            )

        t_indices = np.arange(first_t, last_t, args.stride, dtype=np.int64)
        if args.max_samples > 0:
            t_indices = t_indices[:args.max_samples]

        print(
            f"episode={args.episode}, start={ep_start}, end={ep_end}, "
            f"samples={len(t_indices)}, compare_step={args.compare_step}"
        )

        rows = []
        pred_samples = []
        gt_samples = []

        for sample_idx, t_idx in enumerate(t_indices):
            print(f"[{sample_idx + 1}/{len(t_indices)}] t_idx={int(t_idx)}")
            row, pred_action, gt_action = evaluate_one_timestep(
                deploy=deploy,
                rb=rb,
                t_idx=int(t_idx),
                ep_start=ep_start,
                ep_end=ep_end,
                compare_step=args.compare_step,
            )
            rows.append(row)
            pred_samples.append(pred_action)
            gt_samples.append(gt_action)

    print_summary(rows)
    save_outputs(args.out_dir, rows, pred_samples, gt_samples)


if __name__ == "__main__":
    main()
