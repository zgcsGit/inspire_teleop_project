#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import multiprocessing
import concurrent.futures

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import numpy as np
import zarr
import cv2
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from scripts_hand_umi.generate_replay_buffer_inspire_bimanual_hand import (
    pose_mat_to_pos_rotvec,
    decode_image_dict_to_rgb,
    extract_hand_angles,
    extract_touch_data,
    split_touch_by_region,
)

register_codecs()


# Active-finger overfit dataset contract.
# Keep the same left/right mapping as the full bimanual generator:
# robot0 = left, robot1 = right.
ACTIVE_HAND_ANGLE_IDXS = [3, 4, 5]
ACTIVE_TOUCH_PARTS = ["index", "thumb"]


@click.command()
@click.argument("input", nargs=-1)
@click.option("-o", "--output", required=True, help="Output zarr.zip path")
@click.option("-or", "--out_res", type=str, default="224,224", help="Output resolution H,W")
@click.option("-cl", "--compression_level", type=int, default=99, help="JPEG-XL compression level")
@click.option("-n", "--num_workers", type=int, default=None, help="Number of image decode workers")
def main(input, output, out_res, compression_level, num_workers):
    if len(input) == 0:
        raise click.ClickException("No input npz files provided")

    if os.path.isfile(output):
        if click.confirm(f"{output} exists. Overwrite?", abort=True):
            pass

    out_res = tuple(int(x) for x in out_res.split(","))
    if len(out_res) != 2:
        raise click.ClickException("--out_res must be like 224,224")
    out_h, out_w = out_res

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    cv2.setNumThreads(1)

    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    buffer_start = 0
    image_tasks = []

    input_paths = sorted([os.path.abspath(os.path.expanduser(p)) for p in input])

    for npz_path in input_paths:
        if not os.path.isfile(npz_path):
            print(f"Skipping missing file: {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)

        required_keys = [
            "actual_ee_pose_left", "actual_ee_pose_right",
            "angle_left", "angle_right",
            "touch_left", "touch_right",
            "rgb3", "rgb2", "kinect_rgb",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"{npz_path} missing '{key}'")

        pose_left = data["actual_ee_pose_left"].astype(np.float32)
        pose_right = data["actual_ee_pose_right"].astype(np.float32)

        # Raw collection stores translation in the last row. Convert back to
        # standard homogeneous matrices with translation in the last column.
        pose_left = np.transpose(pose_left, (0, 2, 1))
        pose_right = np.transpose(pose_right, (0, 2, 1))

        angle_left = data["angle_left"]
        angle_right = data["angle_right"]
        touch_left = data["touch_left"]
        touch_right = data["touch_right"]
        rgb_left = data["rgb3"]
        rgb_right = data["rgb2"]
        rgb_kinect = data["kinect_rgb"]

        T = pose_left.shape[0]
        if pose_left.shape != (T, 4, 4):
            raise RuntimeError(f"{npz_path}: actual_ee_pose_left shape invalid: {pose_left.shape}")
        if pose_right.shape != (T, 4, 4):
            raise RuntimeError(f"{npz_path}: actual_ee_pose_right shape invalid: {pose_right.shape}")

        for name, arr in [
            ("angle_left", angle_left),
            ("angle_right", angle_right),
            ("touch_left", touch_left),
            ("touch_right", touch_right),
            ("rgb3", rgb_left),
            ("rgb2", rgb_right),
            ("kinect_rgb", rgb_kinect),
        ]:
            if len(arr) != T:
                raise RuntimeError(f"{npz_path}: {name} length {len(arr)} != pose length {T}")

        robot0_eef_pos, robot0_eef_rotvec = pose_mat_to_pos_rotvec(pose_left)
        robot1_eef_pos, robot1_eef_rotvec = pose_mat_to_pos_rotvec(pose_right)

        robot0_pose6 = np.concatenate([robot0_eef_pos, robot0_eef_rotvec], axis=-1).astype(np.float32)
        robot1_pose6 = np.concatenate([robot1_eef_pos, robot1_eef_rotvec], axis=-1).astype(np.float32)

        robot0_hand_angles = extract_hand_angles(angle_left, T, npz_path, "angle_left")
        robot1_hand_angles = extract_hand_angles(angle_right, T, npz_path, "angle_right")
        robot0_hand_angles = robot0_hand_angles[:, ACTIVE_HAND_ANGLE_IDXS].astype(np.int32)
        robot1_hand_angles = robot1_hand_angles[:, ACTIVE_HAND_ANGLE_IDXS].astype(np.int32)

        robot0_touch_regions = split_touch_by_region(
            extract_touch_data(touch_left, T, npz_path, "touch_left")
        )
        robot1_touch_regions = split_touch_by_region(
            extract_touch_data(touch_right, T, npz_path, "touch_right")
        )

        episode_data = {
            "robot0_eef_pos": robot0_eef_pos.astype(np.float32),
            "robot0_eef_rot_axis_angle": robot0_eef_rotvec.astype(np.float32),
            "robot0_hand_angles": robot0_hand_angles,
            "robot0_demo_start_pose": np.repeat(robot0_pose6[0:1], T, axis=0).astype(np.float32),
            "robot0_demo_end_pose": np.repeat(robot0_pose6[-1:], T, axis=0).astype(np.float32),

            "robot1_eef_pos": robot1_eef_pos.astype(np.float32),
            "robot1_eef_rot_axis_angle": robot1_eef_rotvec.astype(np.float32),
            "robot1_hand_angles": robot1_hand_angles,
            "robot1_demo_start_pose": np.repeat(robot1_pose6[0:1], T, axis=0).astype(np.float32),
            "robot1_demo_end_pose": np.repeat(robot1_pose6[-1:], T, axis=0).astype(np.float32),
        }

        for part in ACTIVE_TOUCH_PARTS:
            episode_data[f"robot0_touch_{part}"] = robot0_touch_regions[part].astype(np.float32)
            episode_data[f"robot1_touch_{part}"] = robot1_touch_regions[part].astype(np.float32)

        replay_buffer.add_episode(data=episode_data, compressors=None)
        image_tasks.append({
            "npz_path": npz_path,
            "T": T,
            "buffer_start": buffer_start,
        })
        buffer_start += T

    total_frames = replay_buffer["robot0_eef_pos"].shape[0]
    print(f"Total frames: {total_frames}")
    print(f"Active hand angle indices: {ACTIVE_HAND_ANGLE_IDXS}")
    print(f"Active touch parts: {ACTIVE_TOUCH_PARTS}")

    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_name in ["camera0_rgb", "camera1_rgb", "camera2_rgb"]:
        replay_buffer.data.require_dataset(
            name=cam_name,
            shape=(total_frames, out_h, out_w, 3),
            chunks=(1, out_h, out_w, 3),
            compressor=img_compressor,
            dtype=np.uint8,
        )

    img_array0 = replay_buffer.data["camera0_rgb"]  # rgb3, left wrist
    img_array1 = replay_buffer.data["camera1_rgb"]  # rgb2, right wrist
    img_array2 = replay_buffer.data["camera2_rgb"]  # kinect_rgb

    def write_images(task):
        data = np.load(task["npz_path"], allow_pickle=True)
        rgb_left = data["rgb3"]
        rgb_right = data["rgb2"]
        rgb_kinect = data["kinect_rgb"]
        buffer_idx = task["buffer_start"]

        for i in range(task["T"]):
            img0 = cv2.resize(
                decode_image_dict_to_rgb(rgb_left[i]),
                (out_w, out_h),
                interpolation=cv2.INTER_AREA,
            )
            img1 = cv2.resize(
                decode_image_dict_to_rgb(rgb_right[i]),
                (out_w, out_h),
                interpolation=cv2.INTER_AREA,
            )
            img2 = cv2.resize(
                decode_image_dict_to_rgb(rgb_kinect[i]),
                (out_w, out_h),
                interpolation=cv2.INTER_AREA,
            )

            img_array0[buffer_idx] = img0
            img_array1[buffer_idx] = img1
            img_array2[buffer_idx] = img2
            buffer_idx += 1

    with tqdm(total=len(image_tasks), desc="Writing images") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(write_images, task) for task in image_tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)

    output_dir = os.path.dirname(os.path.abspath(output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving to {output}")
    with zarr.ZipStore(output, mode="w") as store:
        replay_buffer.save_to_store(store)
    print("Done.")


if __name__ == "__main__":
    main()
