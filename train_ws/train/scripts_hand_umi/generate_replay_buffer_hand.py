#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
import sys
import multiprocessing
import concurrent.futures

import click
import numpy as np
import zarr
import cv2
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from scipy.spatial.transform import Rotation as R


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

register_codecs()


# ===============================
# Quaternion → rotation vector
# ===============================
def quat_xyzw_to_rotvec(q, eps=1e-8):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    q = q / norm

    x, y, z, w = [q[..., i] for i in range(4)]
    w = np.clip(w, -1.0, 1.0)

    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    denom = np.maximum(sin_half, eps)

    axis = np.stack([x / denom, y / denom, z / denom], axis=-1)
    rotvec = axis * angle[..., None]

    small = sin_half < 1e-6
    if np.any(small):
        rotvec = rotvec.copy()
        rotvec[small] = 0.0

    return rotvec.astype(np.float32)

def quat_mul_xyzw(a, b):
    """
    Quaternion multiplication q = a * b
    a,b: (...,4) in xyzw
    return: (...,4) in xyzw
    """
    ax, ay, az, aw = np.split(a, 4, axis=-1)
    bx, by, bz, bw = np.split(b, 4, axis=-1)

    qx = aw*bx + ax*bw + ay*bz - az*by
    qy = aw*by - ax*bz + ay*bw + az*bx
    qz = aw*bz + ax*by - ay*bx + az*bw
    qw = aw*bw - ax*bx - ay*by - az*bz
    return np.concatenate([qx, qy, qz, qw], axis=-1)

def quat_from_axis_angle(axis, degrees):
    """axis: (3,) unit-ish, return (4,) xyzw"""
    rad = np.deg2rad(degrees)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(rad / 2.0)
    c = np.cos(rad / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, c], dtype=np.float64)

def rotate_quat_z_premul(quat_xyzw, degrees):
    """
    pre-multiply: q' = r_z * q
    quat_xyzw: (...,4)
    """
    r = quat_from_axis_angle([0, 0, 1], degrees)[None, :]
    return quat_mul_xyzw(r, quat_xyzw)

def rotate_quat_x_postmul(quat_xyzw, degrees):
    """
    post-multiply: q' = q * r_x
    (matches your rotate_quaternion_x implementation: quat_mul(q, r))
    """
    r = quat_from_axis_angle([1, 0, 0], degrees)[None, :]
    return quat_mul_xyzw(quat_xyzw, r)

def rotate_point_z(xyz, degrees):
    """xyz: (...,3)"""
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    xr = c*x - s*y
    yr = s*x + c*y
    return np.stack([xr, yr, z], axis=-1)

def rotate_point_x(xyz, degrees):
    """xyz: (...,3)"""
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    yr = c*y - s*z
    zr = s*y + c*z
    return np.stack([x, yr, zr], axis=-1)

def ultimate_to_palm_pose(ultimate_pose, mount="left"):
    """
    ultimate_pose: (T,7) [px,py,pz,qx,qy,qz,qw] with quat in xyzw
    returns palm_pose: (T,7) in same format, applying the SAME rule as your ROS node:
      (A) y=-y; x_new=y; y_new=-x; z_new=z
          q' = Rz(-90) * q
      (B) if mount == 'left':
          pos = Rz(+90) pos
          q = Rz(+90) * q
          pos = Rx(-90) pos
          q = q * Rx(-90)   (local X)
    """
    assert ultimate_pose.shape[-1] == 7
    pos = ultimate_pose[:, :3].astype(np.float64)
    quat = ultimate_pose[:, 3:7].astype(np.float64)

    # ---------- (A) base fix ----------
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    y = -y
    x_new = y
    y_new = -x
    z_new = z
    pos2 = np.stack([x_new, y_new, z_new], axis=-1)

    quat2 = rotate_quat_z_premul(quat, -90.0)

    # ---------- (B) mount option ----------
    mount = (mount or "top").lower()
    if mount == "left":
        pos2 = rotate_point_z(pos2, +90.0)
        quat2 = rotate_quat_z_premul(quat2, +90.0)

        pos2 = rotate_point_x(pos2, -90.0)
        quat2 = rotate_quat_x_postmul(quat2, -90.0)

    palm_pose = np.concatenate([pos2, quat2], axis=-1).astype(np.float32)
    return palm_pose


def decode_jpeg_bytes_to_rgb(jpeg_bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ===============================
# CLI
# ===============================
@click.command()
@click.argument("input", nargs=-1)
@click.option("-o", "--output", required=True)
@click.option("-or", "--out_res", type=str, default="224,224")
@click.option("-cl", "--compression_level", type=int, default=99)
@click.option("-n", "--num_workers", type=int, default=None)

def main(input, output, out_res, compression_level, num_workers):

    if len(input) == 0:
        raise click.ClickException("No input npz files provided")

    if os.path.isfile(output):
        if click.confirm(f"{output} exists. Overwrite?", abort=True):
            pass

    out_res = tuple(int(x) for x in out_res.split(","))
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    cv2.setNumThreads(1)

    # ==========================================
    # 1️⃣ Create ReplayBuffer in memory
    # ==========================================
    replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore()
    )

    buffer_start = 0
    image_tasks = []

    input_paths = sorted([os.path.abspath(os.path.expanduser(p)) for p in input])

    for npz_path in input_paths:
        if not os.path.isfile(npz_path):
            print(f"Skipping {npz_path}")
            continue

        data = np.load(npz_path, allow_pickle=True)

        ultimate_pose = data["ultimate_pose"].astype(np.float32)  # (T,7)
        manus_nodes = np.array(data["manus_right_nodes"], dtype=np.float32)  # (T,25,7)
        rgb_arr = data["rgb"]

        T = ultimate_pose.shape[0]

        # ===============================
        # Convert low-dim
        # ===============================

        # # eef（The original ultimate tracker pose）
        # eef_pos = ultimate_pose[:, :3]          # (T,3)
        # quat = ultimate_pose[:, 3:7]            # (T,4)
        # eef_rotvec = quat_xyzw_to_rotvec(quat)  # (T,3)

        # eef（Using Transformation to transform the ultimate tracker pose into palm pose, you can change the transformation matrix, and change it into wrist frame）
        palm_pose = ultimate_to_palm_pose(ultimate_pose, mount="left")  # or "top" if needed(It depends on how you mount the tracker.)

        eef_pos = palm_pose[:, :3]              # (T,3)
        quat = palm_pose[:, 3:7]                # (T,4) xyzw
        eef_rotvec = quat_xyzw_to_rotvec(quat)  # (T,3)

        # build 6D pose (pos + rotvec) for demo start/end
        eef_pose6 = np.concatenate([eef_pos, eef_rotvec], axis=-1).astype(np.float32)  # (T,6)

        start_pose6 = eef_pose6[0]    # (6,)
        end_pose6   = eef_pose6[-1]   # (6,)

        demo_start_pose = np.repeat(start_pose6[None, :], T, axis=0).astype(np.float32)  # (T,6)
        demo_end_pose   = np.repeat(end_pose6[None, :], T, axis=0).astype(np.float32)    # (T,6)

        # hand nodes xyz
        hand_xyz = manus_nodes[:, :, :3]          # (T,25,3)
        hand_flat = hand_xyz.reshape(T, -1)       # (T,75)

        episode_data = {
            "robot0_eef_pos": eef_pos.astype(np.float32),
            "robot0_eef_rot_axis_angle": eef_rotvec.astype(np.float32),
            "robot0_hand_nodes": hand_flat.astype(np.float32),
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
    # 2️⃣ Create camera dataset
    # ==========================================
    img_compressor = JpegXl(level=compression_level, numthreads=1)

    replay_buffer.data.require_dataset(
        name="camera0_rgb",
        shape=(total_frames,) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=img_compressor,
        dtype=np.uint8
    )

    img_array = replay_buffer.data["camera0_rgb"]

    # ==========================================
    # 3️⃣ Image writing (parallel)
    # ==========================================
    def write_images(task):
        data = np.load(task["npz_path"], allow_pickle=True)
        rgb_arr = data["rgb"]

        buffer_idx = task["buffer_start"]
        T = task["T"]

        for i in range(T):
            jpeg_bytes = rgb_arr[i]
            if isinstance(jpeg_bytes, np.void):
                jpeg_bytes = bytes(jpeg_bytes)

            img = decode_jpeg_bytes_to_rgb(jpeg_bytes)
            img = cv2.resize(img, out_res, interpolation=cv2.INTER_AREA)
            img_array[buffer_idx] = img
            buffer_idx += 1

    with tqdm(total=len(image_tasks)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(write_images, task) for task in image_tasks]
            for f in concurrent.futures.as_completed(futures):
                f.result()
                pbar.update(1)

    # ==========================================
    # 4️⃣ Save to zip
    # ==========================================
    print(f"Saving to {output}")
    with zarr.ZipStore(output, mode="w") as store:
        replay_buffer.save_to_store(store)

    print("Done.")


if __name__ == "__main__":
    main()
