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

CHANNEL_COUNTS = {
    1: 93,   # Pinky
    2: 93,   # Ring
    3: 93,   # Middle
    4: 93,   # Index
    5: 105,  # Thumb
    7: 56,   # Palm
}

CHANNEL_ORDER = [1, 2, 3, 4, 5, 7]

CHANNEL_LABELS = {
    1: "pinky",
    2: "ring",
    3: "middle",
    4: "index",
    5: "thumb",
    7: "palm",
}

TOUCH_TOTAL_DIM = sum(CHANNEL_COUNTS.values())  # 533

# ===============================
# Pose utils
# ===============================
def pose_mat_to_pos_rotvec(pose_mat: np.ndarray):
    """
    pose_mat: (T,4,4)
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
    frame_obj:
        {"format": "...", "data": np.ndarray(uint8, shape=(N,))}
    return:
        rgb uint8 image, shape (H,W,3)
    """
    if not isinstance(frame_obj, dict):
        raise TypeError(f"Expected dict frame, got {type(frame_obj)}")

    if "data" not in frame_obj:
        raise KeyError("Image dict missing key 'data'")

    buf = frame_obj["data"]
    if isinstance(buf, np.ndarray):
        img_bytes = buf.tobytes()
    elif isinstance(buf, (bytes, bytearray)):
        img_bytes = bytes(buf)
    else:
        raise TypeError(f"Unsupported image buffer type: {type(buf)}")

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def extract_hand_angles(angle_obj_arr, T, npz_path, field_name):
    """
    angle_obj_arr: object array of len T
    return: (T,6) int32
    """
    hand_angles = []
    for i in range(T):
        frame = angle_obj_arr[i]
        if not isinstance(frame, dict):
            raise TypeError(f"{npz_path}: {field_name}[{i}] is not dict, got {type(frame)}")
        if "angles" not in frame:
            raise KeyError(f"{npz_path}: {field_name}[{i}] missing 'angles'")

        angles = np.asarray(frame["angles"], dtype=np.int32)
        if angles.shape != (6,):
            raise RuntimeError(
                f"{npz_path}: {field_name}[{i}]['angles'] expected shape (6,), got {angles.shape}"
            )
        hand_angles.append(angles)

    return np.stack(hand_angles, axis=0).astype(np.int32)

def extract_touch_data(touch_obj_arr, T, npz_path, field_name):
    """
    touch_obj_arr: object array of len T
    each frame expected to contain 533-dim touch data
    return:
        touch_full: (T,533) float32
    """
    touch_data = []

    for i in range(T):
        frame = touch_obj_arr[i]

        if not isinstance(frame, dict):
            raise TypeError(f"{npz_path}: {field_name}[{i}] is not dict, got {type(frame)}")

        if "touch_values" in frame:
            arr = frame["touch_values"]
        elif "data" in frame:
            arr = frame["data"]
        elif "touch" in frame:
            arr = frame["touch"]
        elif "values" in frame:
            arr = frame["values"]
        else:
            raise KeyError(
                f"{npz_path}: {field_name}[{i}] missing touch array key. "
                f"available keys: {list(frame.keys())}"
            )

        arr = np.asarray(arr, dtype=np.float32).reshape(-1)

        if arr.shape != (TOUCH_TOTAL_DIM,):
            raise RuntimeError(
                f"{npz_path}: {field_name}[{i}] expected shape ({TOUCH_TOTAL_DIM},), got {arr.shape}"
            )

        touch_data.append(arr)

    return np.stack(touch_data, axis=0).astype(np.float32)


def split_touch_by_region(touch_full):
    result = {}
    start = 0

    for channel_id in CHANNEL_ORDER:
        count = CHANNEL_COUNTS[channel_id]
        end = start + count
        label = CHANNEL_LABELS[channel_id]
        result[label] = touch_full[:, start:end].astype(np.float32)
        start = end

    assert start == TOUCH_TOTAL_DIM
    return result


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
        required_keys = [
            "actual_ee_pose_left", "actual_ee_pose_right",
            "angle_left", "angle_right",
            "touch_left", "touch_right",
            "rgb3", "rgb2", "kinect_rgb"
        ]
        for k in required_keys:
            if k not in data:
                raise KeyError(f"{npz_path} missing '{k}'")

        pose_left = data["actual_ee_pose_left"].astype(np.float32)
        pose_right = data["actual_ee_pose_right"].astype(np.float32)

        # 转回标准齐次矩阵
        pose_left = np.transpose(pose_left, (0, 2, 1))
        pose_right = np.transpose(pose_right, (0, 2, 1))

        angle_left = data["angle_left"]        # object array
        angle_right = data["angle_right"]      # object array
        touch_left = data["touch_left"]
        touch_right = data["touch_right"]

        rgb_left = data["rgb3"]                # 左手相机
        rgb_right = data["rgb2"]               # 右手相机
        rgb_kinect = data["kinect_rgb"]        # 第三人称

        T = pose_left.shape[0]

        # shape consistency
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

        # ------------------------------
        # Convert low-dim: robot0 = left
        # ------------------------------
        robot0_eef_pos, robot0_eef_rotvec = pose_mat_to_pos_rotvec(pose_left)
        robot0_pose6 = np.concatenate([robot0_eef_pos, robot0_eef_rotvec], axis=-1).astype(np.float32)
        robot0_start_pose6 = np.repeat(robot0_pose6[0:1], T, axis=0).astype(np.float32)
        robot0_end_pose6 = np.repeat(robot0_pose6[-1:], T, axis=0).astype(np.float32)
        robot0_hand_angles = extract_hand_angles(angle_left, T, npz_path, "angle_left")
        robot0_touch_full = extract_touch_data(touch_left, T, npz_path, "touch_left")
        robot0_touch_regions = split_touch_by_region(robot0_touch_full)

        # ------------------------------
        # Convert low-dim: robot1 = right
        # ------------------------------
        robot1_eef_pos, robot1_eef_rotvec = pose_mat_to_pos_rotvec(pose_right)
        robot1_pose6 = np.concatenate([robot1_eef_pos, robot1_eef_rotvec], axis=-1).astype(np.float32)
        robot1_start_pose6 = np.repeat(robot1_pose6[0:1], T, axis=0).astype(np.float32)
        robot1_end_pose6 = np.repeat(robot1_pose6[-1:], T, axis=0).astype(np.float32)
        robot1_hand_angles = extract_hand_angles(angle_right, T, npz_path, "angle_right")
        robot1_touch_full = extract_touch_data(touch_right, T, npz_path, "touch_right")
        robot1_touch_regions = split_touch_by_region(robot1_touch_full)

        episode_data = {
            # left arm -> robot0
            "robot0_eef_pos": robot0_eef_pos.astype(np.float32),
            "robot0_eef_rot_axis_angle": robot0_eef_rotvec.astype(np.float32),
            "robot0_hand_angles": robot0_hand_angles.astype(np.int32),
            "robot0_demo_start_pose": robot0_start_pose6,
            "robot0_demo_end_pose": robot0_end_pose6,

            # right arm -> robot1
            "robot1_eef_pos": robot1_eef_pos.astype(np.float32),
            "robot1_eef_rot_axis_angle": robot1_eef_rotvec.astype(np.float32),
            "robot1_hand_angles": robot1_hand_angles.astype(np.int32),
            "robot1_demo_start_pose": robot1_start_pose6,
            "robot1_demo_end_pose": robot1_end_pose6,
        }
        for region_name, region_data in robot0_touch_regions.items():
            episode_data[f"robot0_touch_{region_name}"] = region_data.astype(np.float32)

        for region_name, region_data in robot1_touch_regions.items():
            episode_data[f"robot1_touch_{region_name}"] = region_data.astype(np.float32)

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
    # 2) Create camera datasets
    # ==========================================
    img_compressor = JpegXl(level=compression_level, numthreads=1)

    for cam_name in ["camera0_rgb", "camera1_rgb", "camera2_rgb"]:
        replay_buffer.data.require_dataset(
            name=cam_name,
            shape=(total_frames, out_h, out_w, 3),
            chunks=(1, out_h, out_w, 3),
            compressor=img_compressor,
            dtype=np.uint8
        )

    img_array0 = replay_buffer.data["camera0_rgb"]   # rgb3
    img_array1 = replay_buffer.data["camera1_rgb"]   # rgb2
    img_array2 = replay_buffer.data["camera2_rgb"]   # kinect_rgb

    # ==========================================
    # 3) Image writing (parallel)
    # ==========================================
    def write_images(task):
        data = np.load(task["npz_path"], allow_pickle=True)

        rgb_left = data["rgb3"]
        rgb_right = data["rgb2"]
        rgb_kinect = data["kinect_rgb"]

        buffer_idx = task["buffer_start"]
        T = task["T"]

        for i in range(T):
            img0 = decode_image_dict_to_rgb(rgb_left[i])
            img1 = decode_image_dict_to_rgb(rgb_right[i])
            img2 = decode_image_dict_to_rgb(rgb_kinect[i])

            img0 = cv2.resize(img0, (out_w, out_h), interpolation=cv2.INTER_AREA)
            img1 = cv2.resize(img1, (out_w, out_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (out_w, out_h), interpolation=cv2.INTER_AREA)

            img_array0[buffer_idx] = img0
            img_array1[buffer_idx] = img1
            img_array2[buffer_idx] = img2
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