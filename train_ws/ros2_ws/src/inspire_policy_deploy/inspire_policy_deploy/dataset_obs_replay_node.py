#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import sys
import time
from collections import deque


def add_conda_site_packages():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(conda_prefix, "lib", version, "site-packages")
    if os.path.isdir(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)


add_conda_site_packages()

import numpy as np
import rclpy
import zarr
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.spatial.transform import Rotation as R

from custom_msgs.msg import DesiredPose
from inspire_interfaces.msg import GetAngleAct1, SetAngle1

WORKSPACE_ROOT = os.environ.get("INSPIRE_WORKSPACE_ROOT", "/workspace/zhicheng_ws")
TRAIN_ROOT = os.environ.get(
    "INSPIRE_TRAIN_ROOT",
    os.path.join(WORKSPACE_ROOT, "train"),
)
DEPLOY_SCRIPT_DIR = os.environ.get(
    "INSPIRE_DEPLOY_SCRIPT_DIR",
    os.path.join(TRAIN_ROOT, "inspire_scripts_deploy"),
)
DEBUG_ROOT = os.environ.get("INSPIRE_DEBUG_ROOT", WORKSPACE_ROOT)
sys.path.insert(0, TRAIN_ROOT)
sys.path.insert(0, DEPLOY_SCRIPT_DIR)

from deploy_core import InspirePolicyDeploy
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import mat_to_pose10d, pose_to_mat


TOUCH_PARTS = ["pinky", "ring", "middle", "index", "thumb", "palm"]
DEFAULT_CKPT_PATH = os.environ.get(
    "INSPIRE_REPLAY_CKPT",
    os.path.join(
        TRAIN_ROOT,
        "data/outputs/2026.04.24/"
        "12.40.24_train_diffusion_unet_timm_inspire_bimanual/"
        "checkpoints/epoch=0020-train_loss=0.018.ckpt",
    ),
)
DEFAULT_DATASET_PATH = os.environ.get(
    "INSPIRE_REPLAY_DATASET",
    os.path.join(TRAIN_ROOT, "hand_inspire_dataset/dataset.zarr.zip"),
)


class DatasetObsReplayNode(Node):
    """
    Replay dataset observations through the policy, but decode relative actions
    around the live robot pose before publishing commands.
    """

    def __init__(self):
        super().__init__("dataset_obs_replay_node")

        self.ckpt_path = self.declare_parameter(
            "ckpt_path",
            DEFAULT_CKPT_PATH,
        ).value
        self.dataset_path = self.declare_parameter(
            "dataset_path",
            DEFAULT_DATASET_PATH,
        ).value
        self.device = self.declare_parameter("device", "cuda").value

        self.episode_idx = int(self.declare_parameter("episode_idx", 0).value)
        self.start_frame = int(self.declare_parameter("start_frame", -1).value)
        self.end_frame = int(self.declare_parameter("end_frame", -1).value)
        self.step_stride = int(self.declare_parameter("step_stride", 4).value)

        self.obs_horizon = int(self.declare_parameter("obs_horizon", 2).value)
        self.obs_downsample_steps = int(
            self.declare_parameter("obs_downsample_steps", 3).value
        )
        self.timer_period = float(self.declare_parameter("timer_period", 0.05).value)
        self.step_wait_after_reached = float(
            self.declare_parameter("step_wait_after_reached", 0.5).value
        )
        self.require_measured_reach = bool(
            self.declare_parameter("require_measured_reach", True).value
        )
        self.target_timeout_sec = float(
            self.declare_parameter("target_timeout_sec", 0.0).value
        )
        self.print_interval = float(self.declare_parameter("print_interval", 2.0).value)
        self.max_age = float(self.declare_parameter("max_age", 0.5).value)

        self.action_start_idx = int(self.declare_parameter("action_start_idx", 1).value)
        self.action_chunk_size = int(self.declare_parameter("action_chunk_size", 4).value)
        self.action_source = self.declare_parameter("action_source", "policy").value
        self.policy_decode_base = self.declare_parameter(
            "policy_decode_base", "live"
        ).value
        self.gt_decode_base = self.declare_parameter("gt_decode_base", "live").value
        self.pos_step_limit = float(self.declare_parameter("pos_step_limit", 0.004).value)
        self.rot_step_limit = float(self.declare_parameter("rot_step_limit", 0.04).value)
        self.rotation_limit_mode = self.declare_parameter(
            "rotation_limit_mode", "linear_unwrapped"
        ).value
        self.hand_step_limit = float(self.declare_parameter("hand_step_limit", 20.0).value)
        self.pose_reach_eps = float(self.declare_parameter("pose_reach_eps", 0.002).value)
        self.rot_reach_eps = float(self.declare_parameter("rot_reach_eps", 0.02).value)
        self.hand_reach_eps = float(self.declare_parameter("hand_reach_eps", 8.0).value)

        self.enable_real_publish = bool(
            self.declare_parameter("enable_real_publish", False).value
        )
        if self.action_source not in ["policy", "dataset_gt"]:
            raise ValueError(
                f"action_source must be policy or dataset_gt, got {self.action_source}"
            )
        if self.gt_decode_base not in ["live", "dataset"]:
            raise ValueError(
                f"gt_decode_base must be live or dataset, got {self.gt_decode_base}"
            )
        if self.policy_decode_base not in ["live", "dataset"]:
            raise ValueError(
                "policy_decode_base must be live or dataset, "
                f"got {self.policy_decode_base}"
            )

        run_name = f"dataset_replay_{time.strftime('%Y%m%d_%H%M%S')}"
        self.debug_action_dir = self.declare_parameter(
            "debug_action_dir",
            os.path.join(DEBUG_ROOT, "debug_deploy_actions", run_name),
        ).value
        self.debug_action_csv_path = os.path.join(
            self.debug_action_dir, "dataset_replay_action_debug.csv"
        )
        self.compare_csv_path = os.path.join(
            self.debug_action_dir, "dataset_replay_pred_gt_compare.csv"
        )
        self.debug_action_file = None
        self.debug_action_writer = None
        self.debug_action_seq = 0
        self.compare_file = None
        self.compare_writer = None

        self.buffers = {
            "robot0_pose": deque(maxlen=100),
            "robot1_pose": deque(maxlen=100),
            "robot0_hand_angles": deque(maxlen=100),
            "robot1_hand_angles": deque(maxlen=100),
        }
        self.action_queue = deque()
        self.last_command = None
        self.active_chunk = False
        self.finished = False
        self.target_start_time = None
        self.next_infer_time = 0.0
        self.last_print_time = 0.0

        self.open_dataset()
        self.episode_start_pose_mats = self.read_episode_start_pose_mats()
        self.frame_idx = self.resolve_start_frame()

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            DesiredPose,
            "/frankaLeft/ee_pose_matrix",
            lambda msg: self.pose_cb(msg, "robot0_pose"),
            state_qos,
        )
        self.create_subscription(
            DesiredPose,
            "/frankaRight/ee_pose_matrix",
            lambda msg: self.pose_cb(msg, "robot1_pose"),
            state_qos,
        )
        self.create_subscription(
            GetAngleAct1,
            "/left/angle_data",
            lambda msg: self.angle_cb(msg, "robot0_hand_angles"),
            state_qos,
        )
        self.create_subscription(
            GetAngleAct1,
            "/right/angle_data",
            lambda msg: self.angle_cb(msg, "robot1_hand_angles"),
            state_qos,
        )

        self.pub_left_pose_cmd = self.create_publisher(
            DesiredPose, "/left/desired_pose_matrix", 10
        )
        self.pub_right_pose_cmd = self.create_publisher(
            DesiredPose, "/right/desired_pose_matrix", 10
        )
        self.pub_left_hand_cmd = self.create_publisher(
            SetAngle1, "/left/set_angle_data", 10
        )
        self.pub_right_hand_cmd = self.create_publisher(
            SetAngle1, "/right/set_angle_data", 10
        )

        self.get_logger().info("Loading InspirePolicyDeploy...")
        self.deploy = InspirePolicyDeploy(self.ckpt_path, device=self.device)
        self.get_logger().info("InspirePolicyDeploy loaded.")

        self.init_action_debug_csv()
        self.init_compare_csv()
        self.timer = self.create_timer(self.timer_period, self.timer_cb)

        self.get_logger().info(
            "DatasetObsReplayNode started. "
            f"real_publish={self.enable_real_publish}, episode={self.episode_idx}, "
            f"frame={self.frame_idx}, episode_frames=[{self.episode_start}, {self.episode_end}), "
            f"action_source={self.action_source}, "
            f"policy_decode_base={self.policy_decode_base}, "
            f"gt_decode_base={self.gt_decode_base}, "
            f"require_measured_reach={self.require_measured_reach}, "
            f"target_timeout_sec={self.target_timeout_sec}"
        )

    def open_dataset(self):
        register_codecs()
        self.dataset_path = os.path.expanduser(self.dataset_path)
        self.zarr_store = zarr.ZipStore(self.dataset_path, mode="r")
        self.zarr_root = zarr.group(self.zarr_store)
        self.zarr_data = self.zarr_root["data"]
        self.episode_ends = np.asarray(self.zarr_root["meta"]["episode_ends"][:])

        if not (0 <= self.episode_idx < len(self.episode_ends)):
            raise ValueError(
                f"episode_idx {self.episode_idx} out of range [0, {len(self.episode_ends)})"
            )

        self.episode_start = 0
        if self.episode_idx > 0:
            self.episode_start = int(self.episode_ends[self.episode_idx - 1])
        self.dataset_episode_end = int(self.episode_ends[self.episode_idx])
        self.episode_end = self.dataset_episode_end
        if self.end_frame >= 0:
            self.episode_end = min(self.episode_end, self.episode_start + self.end_frame)

    def resolve_start_frame(self):
        min_local = (self.obs_horizon - 1) * self.obs_downsample_steps
        local = self.start_frame if self.start_frame >= 0 else min_local
        local = max(local, min_local)
        frame = self.episode_start + local
        if frame >= self.episode_end:
            raise ValueError(
                f"start frame {frame} is outside episode end {self.episode_end}"
            )
        return frame

    def read_episode_start_pose_mats(self):
        pose_mats = []
        for rid in [0, 1]:
            prefix = f"robot{rid}"
            pose6 = np.concatenate(
                [
                    self.read_frames(f"{prefix}_eef_pos", [self.episode_start]),
                    self.read_frames(
                        f"{prefix}_eef_rot_axis_angle",
                        [self.episode_start],
                    ),
                ],
                axis=-1,
            ).astype(np.float32)
            pose_mats.append(pose_to_mat(pose6)[0])
        return pose_mats

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def init_action_debug_csv(self):
        os.makedirs(self.debug_action_dir, exist_ok=True)
        self.debug_action_file = open(self.debug_action_csv_path, "w", newline="")
        fieldnames = [
            "seq", "ros_time", "event", "dataset_frame", "queue_len",
            "target_idx", "robot_id", "cur_x", "cur_y", "cur_z",
            "target_x", "target_y", "target_z", "command_x", "command_y",
            "command_z", "target_dx", "target_dy", "target_dz",
            "command_dx", "command_dy", "command_dz", "cur_rx", "cur_ry",
            "cur_rz", "target_rx", "target_ry", "target_rz", "command_rx",
            "command_ry", "command_rz", "target_hand0", "target_hand1",
            "target_hand2", "target_hand3", "target_hand4", "target_hand5",
            "command_hand0", "command_hand1", "command_hand2", "command_hand3",
            "command_hand4", "command_hand5",
        ]
        self.debug_action_writer = csv.DictWriter(self.debug_action_file, fieldnames=fieldnames)
        self.debug_action_writer.writeheader()
        self.debug_action_file.flush()
        self.get_logger().info(f"Writing action debug CSV to {self.debug_action_csv_path}")

    def write_action_debug_row(
        self,
        event,
        robot_id,
        current_pose,
        target_pose,
        command_pose=None,
        target_hand=None,
        command_hand=None,
        target_idx=-1,
    ):
        if self.debug_action_writer is None:
            return

        current_pose = np.asarray(current_pose, dtype=np.float32)
        target_pose = np.asarray(target_pose, dtype=np.float32)
        command_pose = (
            np.full(6, np.nan, dtype=np.float32)
            if command_pose is None
            else np.asarray(command_pose, dtype=np.float32)
        )
        target_hand = (
            np.full(6, np.nan, dtype=np.float32)
            if target_hand is None
            else np.asarray(target_hand, dtype=np.float32)
        )
        command_hand = (
            np.full(6, np.nan, dtype=np.float32)
            if command_hand is None
            else np.asarray(command_hand, dtype=np.float32)
        )

        target_delta = target_pose[:3] - current_pose[:3]
        command_delta = command_pose[:3] - current_pose[:3]
        row = {
            "seq": self.debug_action_seq,
            "ros_time": self.now_sec(),
            "event": event,
            "dataset_frame": self.frame_idx - self.episode_start,
            "queue_len": len(self.action_queue),
            "target_idx": target_idx,
            "robot_id": robot_id,
            "cur_x": current_pose[0],
            "cur_y": current_pose[1],
            "cur_z": current_pose[2],
            "target_x": target_pose[0],
            "target_y": target_pose[1],
            "target_z": target_pose[2],
            "command_x": command_pose[0],
            "command_y": command_pose[1],
            "command_z": command_pose[2],
            "target_dx": target_delta[0],
            "target_dy": target_delta[1],
            "target_dz": target_delta[2],
            "command_dx": command_delta[0],
            "command_dy": command_delta[1],
            "command_dz": command_delta[2],
            "cur_rx": current_pose[3],
            "cur_ry": current_pose[4],
            "cur_rz": current_pose[5],
            "target_rx": target_pose[3],
            "target_ry": target_pose[4],
            "target_rz": target_pose[5],
            "command_rx": command_pose[3],
            "command_ry": command_pose[4],
            "command_rz": command_pose[5],
        }
        for idx in range(6):
            row[f"target_hand{idx}"] = target_hand[idx]
            row[f"command_hand{idx}"] = command_hand[idx]
        self.debug_action_writer.writerow(row)
        self.debug_action_seq += 1
        self.debug_action_file.flush()

    def init_compare_csv(self):
        self.compare_file = open(self.compare_csv_path, "w", newline="")
        fieldnames = [
            "ros_time",
            "episode_idx",
            "dataset_frame",
            "target_idx",
            "gt_dataset_frame",
            "robot_id",
            "live_x",
            "live_y",
            "live_z",
            "live_rx",
            "live_ry",
            "live_rz",
            "pred_x",
            "pred_y",
            "pred_z",
            "pred_rx",
            "pred_ry",
            "pred_rz",
            "gt_x",
            "gt_y",
            "gt_z",
            "gt_rx",
            "gt_ry",
            "gt_rz",
            "pred_dx",
            "pred_dy",
            "pred_dz",
            "gt_dx",
            "gt_dy",
            "gt_dz",
            "pred_gt_pos_err",
            "pred_gt_rot_err",
            "pred_gt_hand_max_err",
        ]
        for prefix in ["pred_hand", "gt_hand"]:
            for idx in range(6):
                fieldnames.append(f"{prefix}{idx}")
        self.compare_writer = csv.DictWriter(self.compare_file, fieldnames=fieldnames)
        self.compare_writer.writeheader()
        self.compare_file.flush()
        self.get_logger().info(f"Writing pred/GT comparison CSV to {self.compare_csv_path}")

    def push_buffer(self, key, data):
        self.buffers[key].append((self.now_sec(), data))

    def get_latest(self, key):
        if len(self.buffers[key]) == 0:
            return None, None
        stamp, data = self.buffers[key][-1]
        age = self.now_sec() - stamp
        if age > self.max_age:
            return None, age
        return data, age

    def pose_cb(self, msg: DesiredPose, key: str):
        try:
            mat = np.asarray(msg.pose_matrix, dtype=np.float64).reshape(4, 4).T
            pos = mat[:3, 3].astype(np.float32)
            rotvec = R.from_matrix(mat[:3, :3]).as_rotvec().astype(np.float32)
            self.push_buffer(
                key,
                {"pos": pos, "rotvec": rotvec, "mat": mat.astype(np.float32)},
            )
        except Exception as exc:
            self.get_logger().warn(f"{key} pose_cb error: {exc}")

    def angle_cb(self, msg: GetAngleAct1, key: str):
        try:
            angles = np.asarray(msg.angles, dtype=np.float32).reshape(-1)
            if angles.shape[0] == 6:
                self.push_buffer(key, angles)
        except Exception as exc:
            self.get_logger().warn(f"{key} angle_cb error: {exc}")

    def get_latest_control_state(self):
        state = {}
        for rid, pose_key, hand_key in [
            (0, "robot0_pose", "robot0_hand_angles"),
            (1, "robot1_pose", "robot1_hand_angles"),
        ]:
            pose, _ = self.get_latest(pose_key)
            hand, _ = self.get_latest(hand_key)
            if pose is None or hand is None:
                return None

            state[f"robot{rid}_target_pose"] = np.concatenate(
                [pose["pos"], pose["rotvec"]], axis=-1
            ).astype(np.float32)
            state[f"robot{rid}_hand_angles"] = hand.astype(np.float32)
            state[f"robot{rid}_pose_mat"] = pose["mat"].astype(np.float32)
        return state

    def get_live_pose_mats(self):
        state = self.get_latest_control_state()
        if state is None:
            return None
        return [state["robot0_pose_mat"], state["robot1_pose_mat"]]

    @staticmethod
    def get_obs_current_pose_mats(obs_world):
        pose_mats = []
        for rid in [0, 1]:
            prefix = f"robot{rid}"
            pose6 = np.concatenate(
                [
                    obs_world[f"{prefix}_eef_pos"],
                    obs_world[f"{prefix}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
            pose_mats.append(pose_to_mat(pose6)[-1])
        return pose_mats

    def build_dataset_obs_world(self):
        idxs = [
            self.frame_idx - i * self.obs_downsample_steps
            for i in range(self.obs_horizon - 1, -1, -1)
        ]
        if idxs[0] < self.episode_start or idxs[-1] >= self.episode_end:
            raise IndexError(f"dataset obs window out of episode: {idxs}")

        obs_world = {}
        for key in ["camera0_rgb", "camera1_rgb", "camera2_rgb"]:
            obs_world[key] = self.read_frames(key, idxs)

        for rid in [0, 1]:
            prefix = f"robot{rid}"
            for key in ["eef_pos", "eef_rot_axis_angle", "hand_angles"]:
                data_key = f"{prefix}_{key}"
                obs_world[data_key] = self.read_frames(data_key, idxs).astype(np.float32)

            touch_arrays = []
            for part in TOUCH_PARTS:
                data_key = f"{prefix}_touch_{part}"
                touch_arrays.append(self.read_frames(data_key, idxs))
            obs_world[f"{prefix}_touch"] = np.concatenate(touch_arrays, axis=-1).astype(np.float32)

        return obs_world

    def get_dataset_action_indices(self, count):
        return [
            self.frame_idx + idx * self.obs_downsample_steps
            for idx in range(count)
        ]

    def build_dataset_gt_action_world(self, obs_world, live_pose_mats, count):
        action_indices = self.get_dataset_action_indices(count)
        if len(action_indices) == 0 or action_indices[-1] >= self.dataset_episode_end:
            return None, action_indices

        actions = []
        for rid in [0, 1]:
            prefix = f"robot{rid}"
            obs_pose6 = np.concatenate(
                [
                    obs_world[f"{prefix}_eef_pos"],
                    obs_world[f"{prefix}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
            obs_pose_mat = pose_to_mat(obs_pose6)

            action_pose6 = np.concatenate(
                [
                    self.read_frames(f"{prefix}_eef_pos", action_indices),
                    self.read_frames(f"{prefix}_eef_rot_axis_angle", action_indices),
                ],
                axis=-1,
            ).astype(np.float32)
            action_pose_mat = pose_to_mat(action_pose6)
            rel_action_mat = convert_pose_mat_rep(
                action_pose_mat,
                base_pose_mat=obs_pose_mat[-1],
                pose_rep=self.deploy.action_pose_repr,
                backward=False,
            )
            rel_pose10d = mat_to_pose10d(rel_action_mat).astype(np.float32)
            hand_angles = self.read_frames(
                f"{prefix}_hand_angles",
                action_indices,
            ).astype(np.float32)
            actions.append(np.concatenate([rel_pose10d, hand_angles], axis=-1))

        action_repr = np.concatenate(actions, axis=-1)
        gt_action_world = self.deploy.decode_action_to_world(action_repr, live_pose_mats)
        return gt_action_world, action_indices

    def write_pred_gt_rows(self, action_world, gt_action_world, action_indices, live_state):
        if self.compare_writer is None or gt_action_world is None or live_state is None:
            return

        start = self.action_start_idx
        end = min(start + self.action_chunk_size, action_world["robot0_target_pose"].shape[0])
        for idx in range(start, end):
            if idx >= len(action_indices):
                continue
            for rid in [0, 1]:
                pose_key = f"robot{rid}_target_pose"
                hand_key = f"robot{rid}_hand_angles"
                live_pose = live_state[pose_key]
                pred_pose = action_world[pose_key][idx]
                gt_pose = gt_action_world[pose_key][idx]
                pred_hand = action_world[hand_key][idx]
                gt_hand = gt_action_world[hand_key][idx]
                row = {
                    "ros_time": self.now_sec(),
                    "episode_idx": self.episode_idx,
                    "dataset_frame": self.frame_idx - self.episode_start,
                    "target_idx": idx,
                    "gt_dataset_frame": action_indices[idx] - self.episode_start,
                    "robot_id": rid,
                    "live_x": live_pose[0],
                    "live_y": live_pose[1],
                    "live_z": live_pose[2],
                    "live_rx": live_pose[3],
                    "live_ry": live_pose[4],
                    "live_rz": live_pose[5],
                    "pred_x": pred_pose[0],
                    "pred_y": pred_pose[1],
                    "pred_z": pred_pose[2],
                    "pred_rx": pred_pose[3],
                    "pred_ry": pred_pose[4],
                    "pred_rz": pred_pose[5],
                    "gt_x": gt_pose[0],
                    "gt_y": gt_pose[1],
                    "gt_z": gt_pose[2],
                    "gt_rx": gt_pose[3],
                    "gt_ry": gt_pose[4],
                    "gt_rz": gt_pose[5],
                    "pred_dx": pred_pose[0] - live_pose[0],
                    "pred_dy": pred_pose[1] - live_pose[1],
                    "pred_dz": pred_pose[2] - live_pose[2],
                    "gt_dx": gt_pose[0] - live_pose[0],
                    "gt_dy": gt_pose[1] - live_pose[1],
                    "gt_dz": gt_pose[2] - live_pose[2],
                    "pred_gt_pos_err": float(np.linalg.norm(pred_pose[:3] - gt_pose[:3])),
                    "pred_gt_rot_err": self.rotation_error(pred_pose[3:6], gt_pose[3:6]),
                    "pred_gt_hand_max_err": float(np.max(np.abs(pred_hand - gt_hand))),
                }
                for hand_idx in range(6):
                    row[f"pred_hand{hand_idx}"] = pred_hand[hand_idx]
                    row[f"gt_hand{hand_idx}"] = gt_hand[hand_idx]
                self.compare_writer.writerow(row)
        self.compare_file.flush()

    def log_pred_gt_first_target(self, action_world, gt_action_world):
        if gt_action_world is None:
            self.get_logger().warn("Skipping pred/GT comparison near episode end.")
            return

        idx = self.action_start_idx
        if idx >= action_world["robot0_target_pose"].shape[0]:
            return

        parts = []
        for rid in [0, 1]:
            pose_key = f"robot{rid}_target_pose"
            hand_key = f"robot{rid}_hand_angles"
            pred_pose = action_world[pose_key][idx]
            gt_pose = gt_action_world[pose_key][idx]
            pred_hand = action_world[hand_key][idx]
            gt_hand = gt_action_world[hand_key][idx]
            parts.append(
                f"r{rid}: pos={np.linalg.norm(pred_pose[:3] - gt_pose[:3]):.4f}, "
                f"rot={self.rotation_error(pred_pose[3:6], gt_pose[3:6]):.4f}, "
                f"hand={np.max(np.abs(pred_hand - gt_hand)):.1f}"
            )
        self.get_logger().info(f"pred/GT target[{idx}] error[{', '.join(parts)}]")

    def read_frames(self, key, idxs):
        arr = self.zarr_data[key]
        selection = (idxs,) + (slice(None),) * (arr.ndim - 1)
        return np.asarray(arr.get_orthogonal_selection(selection))

    def pose6_to_desired_pose_msg(self, pose6):
        pose6 = np.asarray(pose6, dtype=np.float32).reshape(6)
        mat = pose_to_mat(pose6[None, :])[0]
        msg = DesiredPose()
        msg.pose_matrix = mat.T.reshape(-1).astype(np.float64).tolist()
        return msg

    def hand6_to_set_angle_msg(self, hand6):
        hand6 = np.asarray(hand6, dtype=np.float32).reshape(6)
        hand6 = np.clip(np.round(hand6), 0, 1000).astype(np.int32)
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        msg.angles = hand6.tolist()
        return msg

    def publish_real_command(self, command):
        if not self.enable_real_publish:
            return
        self.pub_left_pose_cmd.publish(
            self.pose6_to_desired_pose_msg(command["robot0_target_pose"])
        )
        self.pub_right_pose_cmd.publish(
            self.pose6_to_desired_pose_msg(command["robot1_target_pose"])
        )
        self.pub_left_hand_cmd.publish(
            self.hand6_to_set_angle_msg(command["robot0_hand_angles"])
        )
        self.pub_right_hand_cmd.publish(
            self.hand6_to_set_angle_msg(command["robot1_hand_angles"])
        )

    @staticmethod
    def limit_vector_step(current, target, max_step):
        current = np.asarray(current, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        delta = target - current
        norm = float(np.linalg.norm(delta))
        if norm <= max_step or norm < 1e-8:
            return target.copy()
        return current + delta * (max_step / norm)

    @staticmethod
    def limit_elementwise_step(current, target, max_step):
        current = np.asarray(current, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        return current + np.clip(target - current, -max_step, max_step)

    @staticmethod
    def unwrap_rotvec_near(current_rotvec, target_rotvec):
        current = np.asarray(current_rotvec, dtype=np.float64)
        target = R.from_rotvec(np.asarray(target_rotvec, dtype=np.float64)).as_rotvec()
        angle = float(np.linalg.norm(target))
        if angle < 1e-8:
            return target.astype(np.float32)
        axis = target / angle
        candidates = [target, target + 2.0 * np.pi * axis, target - 2.0 * np.pi * axis]
        return min(candidates, key=lambda x: float(np.linalg.norm(x - current))).astype(np.float32)

    @staticmethod
    def limit_rotation_step_so3(current_rotvec, target_rotvec, max_step):
        current_rot = R.from_rotvec(np.asarray(current_rotvec, dtype=np.float64))
        target_rot = R.from_rotvec(np.asarray(target_rotvec, dtype=np.float64))
        delta_rotvec = (target_rot * current_rot.inv()).as_rotvec()
        delta_norm = float(np.linalg.norm(delta_rotvec))
        if delta_norm <= max_step or delta_norm < 1e-8:
            return np.asarray(target_rotvec, dtype=np.float32).copy()
        command_rot = R.from_rotvec(delta_rotvec * (max_step / delta_norm)) * current_rot
        return command_rot.as_rotvec().astype(np.float32)

    def limit_rotation_step(self, current_rotvec, target_rotvec, max_step):
        if self.rotation_limit_mode == "none":
            return np.asarray(target_rotvec, dtype=np.float32).copy()
        if self.rotation_limit_mode == "so3":
            return self.limit_rotation_step_so3(current_rotvec, target_rotvec, max_step)
        if self.rotation_limit_mode == "linear_unwrapped":
            target_rotvec = self.unwrap_rotvec_near(current_rotvec, target_rotvec)
        if self.rotation_limit_mode in ["linear", "linear_unwrapped"]:
            return self.limit_vector_step(current_rotvec, target_rotvec, max_step)
        raise ValueError(f"Unknown rotation_limit_mode: {self.rotation_limit_mode}")

    @staticmethod
    def rotation_error(current_rotvec, target_rotvec):
        current_rot = R.from_rotvec(np.asarray(current_rotvec, dtype=np.float64))
        target_rot = R.from_rotvec(np.asarray(target_rotvec, dtype=np.float64))
        return float(np.linalg.norm((target_rot * current_rot.inv()).as_rotvec()))

    def measured_state_reached_target(self, current_state, target):
        if current_state is None:
            return False

        errors = self.compute_target_errors(current_state, target)
        return all(
            item["pos_err"] < self.pose_reach_eps
            and item["rot_err"] < self.rot_reach_eps
            and item["hand_err"] < self.hand_reach_eps
            for item in errors.values()
        )

    def compute_target_errors(self, current_state, target):
        errors = {}
        reached = True
        for rid in [0, 1]:
            pose_key = f"robot{rid}_target_pose"
            hand_key = f"robot{rid}_hand_angles"
            cur_pose = current_state[pose_key]
            tgt_pose = target[pose_key]
            cur_hand = current_state[hand_key]
            tgt_hand = target[hand_key]

            pos_err = float(np.linalg.norm(tgt_pose[:3] - cur_pose[:3]))
            rot_err = self.rotation_error(cur_pose[3:6], tgt_pose[3:6])
            hand_err = float(np.max(np.abs(tgt_hand - cur_hand)))
            errors[rid] = {
                "pos_err": pos_err,
                "rot_err": rot_err,
                "hand_err": hand_err,
            }
        return errors

    def make_limited_command(self, target):
        if self.last_command is None:
            self.last_command = self.get_latest_control_state()
        if self.last_command is None:
            return None, False

        command = {}
        reached = True
        for rid in [0, 1]:
            pose_key = f"robot{rid}_target_pose"
            hand_key = f"robot{rid}_hand_angles"
            cur_pose = self.last_command[pose_key]
            tgt_pose = target[pose_key]
            cur_hand = self.last_command[hand_key]
            tgt_hand = target[hand_key]

            cmd_pos = self.limit_vector_step(cur_pose[:3], tgt_pose[:3], self.pos_step_limit)
            cmd_rot = self.limit_rotation_step(cur_pose[3:6], tgt_pose[3:6], self.rot_step_limit)
            cmd_hand = self.limit_elementwise_step(cur_hand, tgt_hand, self.hand_step_limit)

            command[pose_key] = np.concatenate([cmd_pos, cmd_rot], axis=-1)
            command[hand_key] = cmd_hand

            pos_err = float(np.linalg.norm(tgt_pose[:3] - cmd_pos))
            rot_err = self.rotation_error(cmd_rot, tgt_pose[3:6])
            hand_err = float(np.max(np.abs(tgt_hand - cmd_hand)))
            reached = reached and (
                pos_err < self.pose_reach_eps
                and rot_err < self.rot_reach_eps
                and hand_err < self.hand_reach_eps
            )

        self.last_command = command
        return command, reached

    def update_action_queue(self, action_world):
        start = self.action_start_idx
        end = min(start + self.action_chunk_size, action_world["robot0_target_pose"].shape[0])
        if start >= end:
            return

        current_state = self.get_latest_control_state()
        self.action_queue.clear()
        self.target_start_time = self.now_sec()
        for idx in range(start, end):
            self.action_queue.append({
                "robot0_target_pose": action_world["robot0_target_pose"][idx],
                "robot1_target_pose": action_world["robot1_target_pose"][idx],
                "robot0_hand_angles": action_world["robot0_hand_angles"][idx],
                "robot1_hand_angles": action_world["robot1_hand_angles"][idx],
            })
            if current_state is not None:
                for rid in [0, 1]:
                    pose_key = f"robot{rid}_target_pose"
                    hand_key = f"robot{rid}_hand_angles"
                    self.write_action_debug_row(
                        event="raw_target",
                        robot_id=rid,
                        current_pose=current_state[pose_key],
                        target_pose=action_world[pose_key][idx],
                        target_hand=action_world[hand_key][idx],
                        target_idx=idx,
                    )

    def publish_queued_action(self):
        if len(self.action_queue) == 0:
            return
        target = self.action_queue[0]
        command, reached = self.make_limited_command(target)
        if command is None:
            return

        current_state = self.get_latest_control_state()
        if current_state is not None:
            for rid in [0, 1]:
                pose_key = f"robot{rid}_target_pose"
                hand_key = f"robot{rid}_hand_angles"
                self.write_action_debug_row(
                    event="published_command",
                    robot_id=rid,
                    current_pose=current_state[pose_key],
                    target_pose=target[pose_key],
                    command_pose=command[pose_key],
                    target_hand=target[hand_key],
                    command_hand=command[hand_key],
                    target_idx=0,
                )

        self.publish_real_command(command)
        measured_reached = self.measured_state_reached_target(current_state, target)
        timeout_reached = False
        if self.target_timeout_sec > 0.0 and self.target_start_time is not None:
            timeout_reached = (self.now_sec() - self.target_start_time) >= self.target_timeout_sec

        can_advance = reached and (
            (not self.enable_real_publish)
            or (not self.require_measured_reach)
            or measured_reached
            or timeout_reached
        )
        if can_advance and len(self.action_queue) > 0:
            self.action_queue.popleft()
            self.target_start_time = self.now_sec() if len(self.action_queue) > 0 else None

    def run_policy_on_current_dataset_window(self):
        obs_world = self.build_dataset_obs_world()
        live_state = self.get_latest_control_state()
        live_pose_mats = self.get_live_pose_mats()
        if live_state is None or live_pose_mats is None:
            return False

        action_world = None
        gt_action_world = None
        gt_action_indices = None
        if self.action_source == "policy":
            decode_base_mats = live_pose_mats
            if self.policy_decode_base == "dataset":
                decode_base_mats = self.get_obs_current_pose_mats(obs_world)
            action_world = self.deploy.predict(
                obs_world,
                current_pose_mats_override=decode_base_mats,
                episode_start_pose_mats=self.episode_start_pose_mats,
            )
            gt_action_world, gt_action_indices = self.build_dataset_gt_action_world(
                obs_world,
                decode_base_mats,
                count=action_world["robot0_target_pose"].shape[0],
            )
        else:
            decode_base_mats = live_pose_mats
            if self.gt_decode_base == "dataset":
                decode_base_mats = self.get_obs_current_pose_mats(obs_world)
            gt_count = int(self.deploy.shape_meta.action.horizon)
            action_world, gt_action_indices = self.build_dataset_gt_action_world(
                obs_world,
                decode_base_mats,
                count=gt_count,
            )
            gt_action_world = action_world
            if action_world is None:
                return False

        self.write_pred_gt_rows(
            action_world,
            gt_action_world,
            gt_action_indices,
            live_state,
        )
        self.log_pred_gt_first_target(action_world, gt_action_world)
        self.update_action_queue(action_world)
        self.active_chunk = len(self.action_queue) > 0

        self.get_logger().info(
            f"dataset frame {self.frame_idx - self.episode_start}: "
            f"source={self.action_source}, "
            f"queued {len(self.action_queue)} targets; "
            f"robot0 target[0]={action_world['robot0_target_pose'][0]}, "
            f"robot1 target[0]={action_world['robot1_target_pose'][0]}"
        )
        return self.active_chunk

    def timer_cb(self):
        now = self.now_sec()

        if self.finished:
            return

        if self.frame_idx >= self.episode_end:
            self.get_logger().info("Dataset replay finished.")
            self.finished = True
            return

        state = self.get_latest_control_state()
        if state is None:
            if now - self.last_print_time > self.print_interval:
                self.get_logger().warn("Waiting for fresh robot pose and hand angle topics.")
                self.last_print_time = now
            return

        if (not self.active_chunk) and now >= self.next_infer_time:
            try:
                if not self.run_policy_on_current_dataset_window():
                    self.next_infer_time = now + self.step_wait_after_reached
            except Exception as exc:
                self.get_logger().error(f"Dataset replay inference failed: {exc}")
                self.next_infer_time = now + self.step_wait_after_reached
                return

        self.publish_queued_action()

        if self.active_chunk and len(self.action_queue) == 0:
            self.active_chunk = False
            self.frame_idx += self.step_stride
            self.next_infer_time = now + self.step_wait_after_reached
            self.last_command = self.get_latest_control_state()
            self.get_logger().info(
                f"Reached chunk; next dataset frame {self.frame_idx - self.episode_start}"
            )

        if now - self.last_print_time > self.print_interval:
            msg = (
                f"frame={self.frame_idx - self.episode_start}, "
                f"queue={len(self.action_queue)}, real_publish={self.enable_real_publish}"
            )
            if len(self.action_queue) > 0:
                errors = self.compute_target_errors(state, self.action_queue[0])
                err_msg = ", ".join(
                    [
                        f"r{rid}: pos={item['pos_err']:.4f}, "
                        f"rot={item['rot_err']:.4f}, hand={item['hand_err']:.1f}"
                        for rid, item in errors.items()
                    ]
                )
                msg += f", target_err[{err_msg}]"
            self.get_logger().info(msg)
            self.last_print_time = now

    def close(self):
        if self.debug_action_file is not None:
            self.debug_action_file.flush()
            self.debug_action_file.close()
            self.debug_action_file = None
        if self.compare_file is not None:
            self.compare_file.flush()
            self.compare_file.close()
            self.compare_file = None
        if hasattr(self, "zarr_store"):
            self.zarr_store.close()


def main(args=None):
    rclpy.init(args=args)
    node = DatasetObsReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
