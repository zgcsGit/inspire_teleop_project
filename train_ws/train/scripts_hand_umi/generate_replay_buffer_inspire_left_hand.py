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
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

register_codecs()



# ===============================
# Pose utils
# ===============================
def pose_mat_to_pos_rotvec(pose_mat: np.ndarray):
    """
    pose_mat: (T,4,4) float32/float64
    return:
        pos:    (T,3) float32
        rotvec: (T,3) float32
    """
    pos = pose_mat[:, :3, 3]
    rot_mats = pose_mat[:, :3, :3]
    rotvec = R.from_matrix(rot_mats).as_rotvec()
    return pos.astype(np.float32), rotvec.astype(np.float32)


# ===============================
# Image utils
# ===============================
def decode_image_dict_to_rgb(frame_obj):
    """
    frame_obj is expected to be:
        {"format": "...", "data": np.ndarray(uint8, shape=(N,))}
    Decode compressed image bytes into RGB uint8 image.
    """
    if not isinstance(frame_obj, dict):
        raise TypeError(f"Expected dict frame, got {type(frame_obj)}")

    if "data" not in frame_obj:
        raise KeyError("Image dict missing key 'data'")

    buf = frame_obj["data"]
    if isinstance(buf, np.ndarray):
        jpeg_bytes = buf.tobytes()
    elif isinstance(buf, (bytes, bytearray)):
        jpeg_bytes = bytes(buf)
    else:
        raise TypeError(f"Unsupported image buffer type: {type(buf)}")

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on image bytes")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ===============================
# CLI
# ===============================
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

    # ==========================================
    # 1) Create ReplayBuffer in memory
    # ==========================================
    replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore()
    )

    buffer_start = 0
    image_tasks = []

    input_paths = sorted([os.path.abspath(os.path.expanduser(p)) for p in input])

    for npz_path in input_paths:
        if not os.path.isfile(npz_path):
            print(f"Skipping missing file: {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)

        # ------------------------------
        # Required raw fields
        # ------------------------------
        if "desired_pose_left" not in data:
            raise KeyError(f"{npz_path} missing 'desired_pose_left'")
        if "angle_left" not in data:
            raise KeyError(f"{npz_path} missing 'angle_left'")
        if "rgb3" not in data:
            raise KeyError(f"{npz_path} missing 'rgb3'")

        pose_left = data["desired_pose_left"].astype(np.float32)   # (T,4,4)
        pose_left = np.transpose(pose_left, (0, 2, 1))            # 转回标准齐次矩阵
        angle_left = data["angle_left"]                            # object array, len T
        rgb3 = data["rgb3"]                                        # object array, len T

        T = pose_left.shape[0]

        if pose_left.shape != (T, 4, 4):
            raise RuntimeError(f"{npz_path}: desired_pose_left shape invalid: {pose_left.shape}")
        if len(angle_left) != T:
            raise RuntimeError(f"{npz_path}: angle_left length {len(angle_left)} != pose length {T}")
        if len(rgb3) != T:
            raise RuntimeError(f"{npz_path}: rgb3 length {len(rgb3)} != pose length {T}")

        # ------------------------------
        # Convert low-dim
        # ------------------------------
        eef_pos, eef_rotvec = pose_mat_to_pos_rotvec(pose_left)    # (T,3), (T,3)

        eef_pose6 = np.concatenate([eef_pos, eef_rotvec], axis=-1).astype(np.float32)  # (T,6)

        start_pose6 = eef_pose6[0]   # (6,)
        end_pose6 = eef_pose6[-1]    # (6,)

        demo_start_pose = np.repeat(start_pose6[None, :], T, axis=0).astype(np.float32)
        demo_end_pose = np.repeat(end_pose6[None, :], T, axis=0).astype(np.float32)

        # hand angles: keep int32 in zarr
        hand_angles = []
        for i in range(T):
            frame = angle_left[i]
            if not isinstance(frame, dict):
                raise TypeError(f"{npz_path}: angle_left[{i}] is not dict, got {type(frame)}")
            if "angles" not in frame:
                raise KeyError(f"{npz_path}: angle_left[{i}] missing 'angles'")

            angles = np.asarray(frame["angles"], dtype=np.int32)
            if angles.shape != (6,):
                raise RuntimeError(
                    f"{npz_path}: angle_left[{i}]['angles'] expected shape (6,), got {angles.shape}"
                )
            hand_angles.append(angles)

        hand_angles = np.stack(hand_angles, axis=0).astype(np.int32)   # (T,6)

        episode_data = {
            "robot0_eef_pos": eef_pos.astype(np.float32),
            "robot0_eef_rot_axis_angle": eef_rotvec.astype(np.float32),
            "robot0_hand_angles": hand_angles.astype(np.int32),
            "robot0_demo_start_pose": demo_start_pose,
            "robot0_demo_end_pose": demo_end_pose,
        }

        replay_buffer.add_episode(data=episode_data, compressors=None)

        image_tasks.append({
            "npz_path": npz_path,
            "T": T,
            "buffer_start": buffer_start
        })

        buffer_start += T

    total_frames = replay_buffer["robot0_eef_pos"].shape[0]
    print(f"Total frames: {total_frames}")

    # ==========================================
    # 2) Create camera dataset
    # ==========================================
    img_compressor = JpegXl(level=compression_level, numthreads=1)

    replay_buffer.data.require_dataset(
        name="camera0_rgb",
        shape=(total_frames, out_h, out_w, 3),
        chunks=(1, out_h, out_w, 3),
        compressor=img_compressor,
        dtype=np.uint8
    )

    img_array = replay_buffer.data["camera0_rgb"]

    # ==========================================
    # 3) Image writing (parallel)
    # ==========================================
    def write_images(task):
        data = np.load(task["npz_path"], allow_pickle=True)
        rgb_arr = data["rgb3"]

        buffer_idx = task["buffer_start"]
        T = task["T"]

        for i in range(T):
            frame_obj = rgb_arr[i]
            img = decode_image_dict_to_rgb(frame_obj)
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            img_array[buffer_idx] = img
            buffer_idx += 1

    with tqdm(total=len(image_tasks), desc="Writing images") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(write_images, task) for task in image_tasks]
            for f in concurrent.futures.as_completed(futures):
                f.result()
                pbar.update(1)

    # ==========================================
    # 4) Save to zip
    # ==========================================
    print(f"Saving to {output}")
    with zarr.ZipStore(output, mode="w") as store:
        replay_buffer.save_to_store(store)

    print("Done.")


if __name__ == "__main__":
    main()