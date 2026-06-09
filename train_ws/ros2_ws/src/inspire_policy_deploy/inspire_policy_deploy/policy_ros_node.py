#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import csv
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from scipy.spatial.transform import Rotation as R

from custom_msgs.msg import DesiredPose
from inspire_interfaces.msg import GetAngleAct1, GetTouchAct1, SetAngle1

WORKSPACE_ROOT = os.environ.get("INSPIRE_WORKSPACE_ROOT", "/workspace/zhicheng_ws")
TRAIN_ROOT = os.environ.get(
    "INSPIRE_TRAIN_ROOT",
    os.path.join(WORKSPACE_ROOT, "train"),
)
DEPLOY_SCRIPT_DIR = os.environ.get(
    "INSPIRE_DEPLOY_SCRIPT_DIR",
    os.path.join(TRAIN_ROOT, "inspire_scripts_deploy"),
)
DATA_ROOT = os.environ.get("INSPIRE_DATA_ROOT", "/mnt/data")
DEBUG_ROOT = os.environ.get("INSPIRE_DEBUG_ROOT", WORKSPACE_ROOT)


def add_conda_site_packages():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.join(conda_prefix, "lib", version, "site-packages")
    if os.path.isdir(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)


add_conda_site_packages()
sys.path.insert(0, TRAIN_ROOT)
sys.path.insert(0, DEPLOY_SCRIPT_DIR)

from umi.common.pose_util import pose_to_mat

DEFAULT_INIT_EPISODE_PATH = os.environ.get(
    "INSPIRE_INIT_EPISODE_PATH",
    os.path.join(
        DATA_ROOT,
        "zhicheng_ws/dataset/test_2/episode_20260522_181158_e8d340.npz",
    ),
)
DEFAULT_CKPT_PATH = os.environ.get(
    "INSPIRE_DEFAULT_CKPT",
    os.path.join(
        TRAIN_ROOT,
        "data/outputs/2026.05.26/"
        "11.18.52_train_diffusion_unet_timm_inspire_bimanual_active/"
        "checkpoints/latest.ckpt",
    ),
)
ACTIVE_HAND_ANGLE_IDXS = [3, 4, 5]


class PolicyRosNode(Node):
    def __init__(self):
        super().__init__("policy_ros_node")

        self.obs_horizon = 2
        self.obs_downsample_steps = 3
        self.image_size = 224
        self.max_age = 0.5

        self.timer_period = float(self.declare_parameter("timer_period", 0.05).value)
        self.infer_interval = float(
            self.declare_parameter("infer_interval", 0.5).value
        )   # 2Hz，第一轮实机保守
        self.print_interval = float(
            self.declare_parameter("print_interval", 2.0).value
        )

        # Receding-horizon execution. action[0] is often too close to the
        # current state, so execute a short chunk starting from action[1].
        self.action_start_idx = int(
            self.declare_parameter("action_start_idx", 1).value
        )
        self.action_chunk_size = int(
            self.declare_parameter("action_chunk_size", 4).value
        )
        self.pos_step_limit = float(
            self.declare_parameter("pos_step_limit", 0.004).value
        )       # meters per timer tick
        self.rot_step_limit = float(
            self.declare_parameter("rot_step_limit", 0.04).value
        )        # rotvec units per timer tick
        self.rotation_limit_mode = self.declare_parameter(
            "rotation_limit_mode", "linear_unwrapped"
        ).value  # linear, linear_unwrapped, so3, none
        self.hand_step_limit = float(
            self.declare_parameter("hand_step_limit", 20.0).value
        )       # Inspire angle units per timer tick
        self.pose_reach_eps = float(
            self.declare_parameter("pose_reach_eps", 0.002).value
        )
        self.rot_reach_eps = float(
            self.declare_parameter("rot_reach_eps", 0.02).value
        )
        self.hand_reach_eps = float(
            self.declare_parameter("hand_reach_eps", 8.0).value
        )

        self.enable_real_publish = bool(
            self.declare_parameter("enable_real_publish", True).value
        )
        self.enable_init_pose = bool(
            self.declare_parameter("enable_init_pose", True).value
        )
        self.run_policy_after_init = bool(
            self.declare_parameter("run_policy_after_init", False).value
        )
        self.init_episode_path = self.declare_parameter(
            "init_episode_path", DEFAULT_INIT_EPISODE_PATH
        ).value
        self.init_frame_idx = int(self.declare_parameter("init_frame_idx", 3).value)
        self.init_publish_period = float(
            self.declare_parameter("init_publish_period", 0.1).value
        )
        self.init_hold_sec = float(self.declare_parameter("init_hold_sec", 20.0).value)
        self.init_pos_tolerance = float(
            self.declare_parameter("init_pos_tolerance", 0.015).value
        )
        self.init_rot_tolerance = float(
            self.declare_parameter("init_rot_tolerance", 0.20).value
        )
        self.init_hand_tolerance = float(
            self.declare_parameter("init_hand_tolerance", 25.0).value
        )
        self.init_inactive_angle = int(
            self.declare_parameter("init_inactive_angle", 1000).value
        )

        self.save_initial_obs = True
        self.max_saved_obs = 5
        self.saved_obs_count = 0
        self.save_obs_dir = os.path.join(
            os.path.join(DEBUG_ROOT, "debug_deploy_obs"),
            f"run_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        self.record_kinect_video = bool(
            self.declare_parameter("record_kinect_video", True).value
        )
        self.kinect_video_fps = float(
            self.declare_parameter("kinect_video_fps", 30.0).value
        )
        self.kinect_video_path = self.declare_parameter(
            "kinect_video_path",
            os.path.join(self.save_obs_dir, "kinect_view.mp4"),
        ).value
        self.kinect_video_writer = None
        self.kinect_video_frame_count = 0
        self.debug_action_dir = os.path.join(
            os.path.join(DEBUG_ROOT, "debug_deploy_actions"),
            f"run_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        self.debug_action_csv_path = os.path.join(
            self.debug_action_dir,
            "action_debug.csv",
        )
        self.debug_action_file = None
        self.debug_action_writer = None
        self.debug_action_seq = 0

        self.last_infer_time = 0.0
        self.last_print_time = 0.0

        self.ckpt_path = self.declare_parameter("ckpt_path", DEFAULT_CKPT_PATH).value

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.buffers = {}
        self.obs_history = deque(
            maxlen=(self.obs_horizon - 1) * self.obs_downsample_steps + 1
        )
        self.action_queue = deque()
        self.last_command = None
        self.episode_start_pose_mats = None
        self.policy_phase = "init" if self.enable_init_pose else "policy"
        self.init_target = None
        self.init_last_publish_time = 0.0
        self.init_start_time = 0.0
        self.deploy = None

        for key in [
            "camera0_rgb",
            "camera1_rgb",
            "camera2_rgb",
            "robot0_pose",
            "robot1_pose",
            "robot0_hand_angles",
            "robot1_hand_angles",
            "robot0_touch",
            "robot1_touch",
        ]:
            self.buffers[key] = deque(maxlen=100)

        if self.enable_init_pose:
            self.init_target = self.load_init_target_frame()
            self.log_init_target()

        should_load_policy = (not self.enable_init_pose) or self.run_policy_after_init
        if should_load_policy:
            from deploy_core import InspirePolicyDeploy

            self.get_logger().info("Loading InspirePolicyDeploy...")
            self.deploy = InspirePolicyDeploy(self.ckpt_path, device="cuda")
            self.get_logger().info("InspirePolicyDeploy loaded.")
        else:
            self.get_logger().info(
                "Policy model loading skipped. Set run_policy_after_init:=true "
                "to start policy inference after init reaches the target."
            )

        self.create_subscription(
            CompressedImage,
            "/camera_wrist_left/color/image_raw/compressed",
            lambda msg: self.image_cb(msg, "camera0_rgb"),
            sensor_qos,
        )
        self.create_subscription(
            CompressedImage,
            "/camera_wrist_right/color/image_raw/compressed",
            lambda msg: self.image_cb(msg, "camera1_rgb"),
            sensor_qos,
        )
        self.create_subscription(
            CompressedImage,
            "/ak/rgb/image_raw/compressed",
            lambda msg: self.image_cb(msg, "camera2_rgb"),
            sensor_qos,
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

        self.create_subscription(
            GetTouchAct1,
            "/left/touch_data",
            lambda msg: self.touch_cb(msg, "robot0_touch"),
            state_qos,
        )
        self.create_subscription(
            GetTouchAct1,
            "/right/touch_data",
            lambda msg: self.touch_cb(msg, "robot1_touch"),
            state_qos,
        )

        # # Debug converted publishers
        # self.pub_robot0_pose_cmd_debug = self.create_publisher(
        #     DesiredPose, "/policy_debug/robot0_desired_pose_matrix", 10
        # )
        # self.pub_robot1_pose_cmd_debug = self.create_publisher(
        #     DesiredPose, "/policy_debug/robot1_desired_pose_matrix", 10
        # )
        # self.pub_robot0_hand_cmd_debug = self.create_publisher(
        #     SetAngle1, "/policy_debug/robot0_set_angle_data", 10
        # )
        # self.pub_robot1_hand_cmd_debug = self.create_publisher(
        #     SetAngle1, "/policy_debug/robot1_set_angle_data", 10
        # )

        # Real control publishers
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

        self.timer = self.create_timer(self.timer_period, self.timer_cb)

        self.get_logger().info("PolicyRosNode started.")
        self.get_logger().info(
            f"Real publish: {self.enable_real_publish}, phase={self.policy_phase}, "
            f"run_policy_after_init={self.run_policy_after_init}, "
            f"inference interval: {self.infer_interval}s"
        )
        self.get_logger().info(
            f"Obs horizon: {self.obs_horizon}, downsample steps: {self.obs_downsample_steps}, "
            f"history length: {self.obs_history.maxlen}"
        )
        self.get_logger().info(
            f"Execution params: action_start_idx={self.action_start_idx}, "
            f"action_chunk_size={self.action_chunk_size}, "
            f"pos_step_limit={self.pos_step_limit:.4f}, "
            f"rot_step_limit={self.rot_step_limit:.4f}, "
            f"hand_step_limit={self.hand_step_limit:.1f}"
        )
        if self.save_initial_obs:
            self.get_logger().info(
                f"Will save first {self.max_saved_obs} obs windows to {self.save_obs_dir}"
            )
        if self.record_kinect_video:
            self.get_logger().info(
                f"Will record Kinect RGB video to {self.kinect_video_path} "
                f"at {self.kinect_video_fps:.1f} FPS"
            )
        self.init_action_debug_csv()

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def init_action_debug_csv(self):
        os.makedirs(self.debug_action_dir, exist_ok=True)
        self.debug_action_file = open(self.debug_action_csv_path, "w", newline="")
        fieldnames = [
            "seq",
            "ros_time",
            "event",
            "queue_len",
            "target_idx",
            "robot_id",
            "cur_x",
            "cur_y",
            "cur_z",
            "target_x",
            "target_y",
            "target_z",
            "command_x",
            "command_y",
            "command_z",
            "target_dx",
            "target_dy",
            "target_dz",
            "command_dx",
            "command_dy",
            "command_dz",
            "cur_rx",
            "cur_ry",
            "cur_rz",
            "target_rx",
            "target_ry",
            "target_rz",
            "command_rx",
            "command_ry",
            "command_rz",
            "target_hand0",
            "target_hand1",
            "target_hand2",
            "target_hand3",
            "target_hand4",
            "target_hand5",
            "command_hand0",
            "command_hand1",
            "command_hand2",
            "command_hand3",
            "command_hand4",
            "command_hand5",
        ]
        self.debug_action_writer = csv.DictWriter(
            self.debug_action_file,
            fieldnames=fieldnames,
        )
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
        if command_pose is None:
            command_pose = np.full(6, np.nan, dtype=np.float32)
        else:
            command_pose = np.asarray(command_pose, dtype=np.float32)

        if target_hand is None:
            target_hand = np.full(6, np.nan, dtype=np.float32)
        else:
            target_hand = np.asarray(target_hand, dtype=np.float32)
        if command_hand is None:
            command_hand = np.full(6, np.nan, dtype=np.float32)
        else:
            command_hand = np.asarray(command_hand, dtype=np.float32)

        target_delta = target_pose[:3] - current_pose[:3]
        command_delta = command_pose[:3] - current_pose[:3]

        row = {
            "seq": self.debug_action_seq,
            "ros_time": self.now_sec(),
            "event": event,
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
        if self.debug_action_file is not None:
            self.debug_action_file.flush()

    def push_buffer(self, key, data):
        self.buffers[key].append((self.now_sec(), data))

    def get_latest(self, key):
        if len(self.buffers[key]) == 0:
            return None, None
        t, data = self.buffers[key][-1]
        age = self.now_sec() - t
        if age > self.max_age:
            return None, age
        return data, age

    def image_cb(self, msg: CompressedImage, key: str):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            if key == "camera2_rgb":
                self.write_kinect_video_frame(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(
                rgb,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            self.push_buffer(key, rgb.astype(np.uint8))
        except Exception as e:
            self.get_logger().warn(f"{key} image_cb error: {e}")

    def write_kinect_video_frame(self, bgr):
        if not self.record_kinect_video:
            return
        if self.kinect_video_writer is None:
            os.makedirs(os.path.dirname(self.kinect_video_path), exist_ok=True)
            height, width = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.kinect_video_writer = cv2.VideoWriter(
                self.kinect_video_path,
                fourcc,
                self.kinect_video_fps,
                (width, height),
            )
            if not self.kinect_video_writer.isOpened():
                self.get_logger().error(
                    f"Failed to open Kinect video writer: {self.kinect_video_path}"
                )
                self.kinect_video_writer = None
                self.record_kinect_video = False
                return
            self.get_logger().info(
                f"Recording Kinect RGB video: {self.kinect_video_path}, "
                f"size={width}x{height}, fps={self.kinect_video_fps:.1f}"
            )

        self.kinect_video_writer.write(bgr)
        self.kinect_video_frame_count += 1

    def pose_cb(self, msg: DesiredPose, key: str):
        try:
            mat = np.asarray(msg.pose_matrix, dtype=np.float64).reshape(4, 4).T
            pos = mat[:3, 3].astype(np.float32)
            rotvec = R.from_matrix(mat[:3, :3]).as_rotvec().astype(np.float32)
            self.push_buffer(
                key,
                {"pos": pos, "rotvec": rotvec, "mat": mat.astype(np.float32)},
            )
        except Exception as e:
            self.get_logger().warn(f"{key} pose_cb error: {e}")

    def angle_cb(self, msg: GetAngleAct1, key: str):
        try:
            angles = np.asarray(msg.angles, dtype=np.float32).reshape(-1)
            if angles.shape[0] != 6:
                return
            self.push_buffer(key, angles)
        except Exception as e:
            self.get_logger().warn(f"{key} angle_cb error: {e}")

    def touch_cb(self, msg: GetTouchAct1, key: str):
        try:
            touch = np.asarray(msg.touch_values, dtype=np.float32).reshape(-1)
            if touch.shape[0] != 533:
                return
            self.push_buffer(key, touch)
        except Exception as e:
            self.get_logger().warn(f"{key} touch_cb error: {e}")

    def build_snapshot(self):
        required = [
            "camera0_rgb",
            "camera1_rgb",
            "camera2_rgb",
            "robot0_pose",
            "robot1_pose",
            "robot0_hand_angles",
            "robot1_hand_angles",
            "robot0_touch",
            "robot1_touch",
        ]

        latest = {}
        stale_info = []

        for key in required:
            data, age = self.get_latest(key)
            if data is None:
                stale_info.append((key, age))
            else:
                latest[key] = data

        if stale_info:
            return None, stale_info

        snapshot = {
            "camera0_rgb": latest["camera0_rgb"],
            "camera1_rgb": latest["camera1_rgb"],
            "camera2_rgb": latest["camera2_rgb"],

            "robot0_eef_pos": latest["robot0_pose"]["pos"],
            "robot0_eef_rot_axis_angle": latest["robot0_pose"]["rotvec"],
            "robot0_hand_angles": latest["robot0_hand_angles"],
            "robot0_touch": latest["robot0_touch"],

            "robot1_eef_pos": latest["robot1_pose"]["pos"],
            "robot1_eef_rot_axis_angle": latest["robot1_pose"]["rotvec"],
            "robot1_hand_angles": latest["robot1_hand_angles"],
            "robot1_touch": latest["robot1_touch"],
        }

        return snapshot, None

    def stack_obs_world(self):
        required_history = (self.obs_horizon - 1) * self.obs_downsample_steps + 1
        if len(self.obs_history) < required_history:
            return None

        selected = [
            self.obs_history[-1 - i * self.obs_downsample_steps]
            for i in range(self.obs_horizon - 1, -1, -1)
        ]

        obs_world = {}
        for key in selected[0].keys():
            obs_world[key] = np.stack(
                [snap[key] for snap in selected],
                axis=0,
            )
        return obs_world

    def get_obs_start_pose_mats(self, obs_world):
        pose_mats = []
        for rid in [0, 1]:
            pose6 = np.concatenate(
                [
                    obs_world[f"robot{rid}_eef_pos"][-1],
                    obs_world[f"robot{rid}_eef_rot_axis_angle"][-1],
                ],
                axis=-1,
            )
            pose_mats.append(pose_to_mat(pose6[None, :])[0].astype(np.float32))
        return pose_mats

    @staticmethod
    def layout_error(mats, translation_in_column):
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        if translation_in_column:
            return float(np.max(np.abs(mats[:, 3, :] - expected)))
        return float(np.max(np.abs(mats[:, :, 3] - expected)))

    def raw_mats_to_standard(self, mats, key):
        column_layout_error = self.layout_error(mats, translation_in_column=True)
        row_layout_error = self.layout_error(mats, translation_in_column=False)
        if column_layout_error < 1e-4 and row_layout_error >= 1e-4:
            return "standard_column_translation", mats.copy()
        if row_layout_error < 1e-4 and column_layout_error >= 1e-4:
            return "raw_row_translation_transposed", np.transpose(mats, (0, 2, 1)).copy()
        raise ValueError(
            f"{key} matrix layout is unsupported or ambiguous: "
            f"column_error={column_layout_error:.3g}, row_error={row_layout_error:.3g}"
        )

    @staticmethod
    def matrix_to_pose6(mat):
        mat = np.asarray(mat, dtype=np.float64)
        pos = mat[:3, 3]
        rotvec = R.from_matrix(mat[:3, :3]).as_rotvec()
        return np.concatenate([pos, rotvec], axis=0).astype(np.float32)

    @staticmethod
    def extract_raw_hand6(angle_obj_arr, frame_idx, field_name):
        if frame_idx >= len(angle_obj_arr):
            raise ValueError(
                f"frame_idx {frame_idx} out of range for {field_name}, "
                f"len={len(angle_obj_arr)}"
            )
        frame = angle_obj_arr[frame_idx]
        if isinstance(frame, np.ndarray) and frame.shape == ():
            frame = frame.item()
        if "angles" not in frame:
            raise KeyError(f"{field_name}[{frame_idx}] missing 'angles'")
        angles = np.asarray(frame["angles"], dtype=np.int32).reshape(-1)
        if angles.shape[0] != 6:
            raise ValueError(
                f"{field_name}[{frame_idx}]['angles'] expected 6, got {angles.shape}"
            )
        return angles

    def active_hand_command_from_raw(self, raw_hand6):
        command = np.full(6, self.init_inactive_angle, dtype=np.int32)
        command[ACTIVE_HAND_ANGLE_IDXS] = np.asarray(raw_hand6, dtype=np.int32)[
            ACTIVE_HAND_ANGLE_IDXS
        ]
        return command

    def load_init_target_frame(self):
        episode_path = os.path.expanduser(self.init_episode_path)
        data = np.load(episode_path, allow_pickle=True)
        required = [
            "actual_ee_pose_left",
            "actual_ee_pose_right",
            "angle_left",
            "angle_right",
        ]
        for key in required:
            if key not in data:
                raise KeyError(f"{episode_path} missing '{key}'")

        left_raw = np.asarray(data["actual_ee_pose_left"], dtype=np.float32)
        right_raw = np.asarray(data["actual_ee_pose_right"], dtype=np.float32)
        if left_raw.ndim != 3 or left_raw.shape[1:] != (4, 4):
            raise ValueError(f"actual_ee_pose_left has invalid shape {left_raw.shape}")
        if right_raw.shape != left_raw.shape:
            raise ValueError(
                f"actual_ee_pose_right shape {right_raw.shape} != {left_raw.shape}"
            )
        if self.init_frame_idx < 0 or self.init_frame_idx >= left_raw.shape[0]:
            raise ValueError(
                f"init_frame_idx {self.init_frame_idx} not in [0, {left_raw.shape[0]})"
            )

        left_layout, left_std = self.raw_mats_to_standard(
            left_raw, "actual_ee_pose_left"
        )
        right_layout, right_std = self.raw_mats_to_standard(
            right_raw, "actual_ee_pose_right"
        )
        left_raw_hand = self.extract_raw_hand6(
            data["angle_left"], self.init_frame_idx, "angle_left"
        )
        right_raw_hand = self.extract_raw_hand6(
            data["angle_right"], self.init_frame_idx, "angle_right"
        )
        return {
            "episode_path": episode_path,
            "frame_idx": self.init_frame_idx,
            "left_layout": left_layout,
            "right_layout": right_layout,
            "robot0_target_pose": self.matrix_to_pose6(left_std[self.init_frame_idx]),
            "robot1_target_pose": self.matrix_to_pose6(right_std[self.init_frame_idx]),
            "robot0_hand_angles": self.active_hand_command_from_raw(left_raw_hand).astype(
                np.float32
            ),
            "robot1_hand_angles": self.active_hand_command_from_raw(right_raw_hand).astype(
                np.float32
            ),
            "robot0_raw_hand_angles": left_raw_hand,
            "robot1_raw_hand_angles": right_raw_hand,
        }

    def log_init_target(self):
        target = self.init_target
        self.get_logger().info(
            f"Loaded init target: {target['episode_path']}, frame={target['frame_idx']}, "
            f"layouts=({target['left_layout']}, {target['right_layout']})"
        )
        self.get_logger().info(
            f"init robot0 pose={target['robot0_target_pose']}, "
            f"raw_hand={target['robot0_raw_hand_angles']}, "
            f"cmd_hand={target['robot0_hand_angles']}"
        )
        self.get_logger().info(
            f"init robot1 pose={target['robot1_target_pose']}, "
            f"raw_hand={target['robot1_raw_hand_angles']}, "
            f"cmd_hand={target['robot1_hand_angles']}"
        )

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

    def save_obs_capture(self, obs_world):
        if (not self.save_initial_obs) or self.saved_obs_count >= self.max_saved_obs:
            return

        capture_idx = self.saved_obs_count
        out_dir = os.path.join(self.save_obs_dir, f"obs_{capture_idx:03d}")
        os.makedirs(out_dir, exist_ok=True)

        saved = {}
        for cam in ["camera0_rgb", "camera1_rgb", "camera2_rgb"]:
            saved[cam] = obs_world[cam]
            for t_idx, img in enumerate(obs_world[cam]):
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(out_dir, f"{cam}_t{t_idx}.png"), bgr)

        for key, value in obs_world.items():
            if key not in saved:
                saved[key] = value

        np.savez_compressed(
            os.path.join(out_dir, "obs_world.npz"),
            **saved,
        )

        self.saved_obs_count += 1
        self.get_logger().info(f"Saved deploy obs capture {capture_idx} to {out_dir}")

    # def publish_converted_debug_action(self, action_world):
    #     robot0_pose0 = action_world["robot0_target_pose"][0]
    #     robot1_pose0 = action_world["robot1_target_pose"][0]
    #     robot0_hand0 = action_world["robot0_hand_angles"][0]
    #     robot1_hand0 = action_world["robot1_hand_angles"][0]

    #     self.pub_robot0_pose_cmd_debug.publish(
    #         self.pose6_to_desired_pose_msg(robot0_pose0)
    #     )
    #     self.pub_robot1_pose_cmd_debug.publish(
    #         self.pose6_to_desired_pose_msg(robot1_pose0)
    #     )
    #     self.pub_robot0_hand_cmd_debug.publish(
    #         self.hand6_to_set_angle_msg(robot0_hand0)
    #     )
    #     self.pub_robot1_hand_cmd_debug.publish(
    #         self.hand6_to_set_angle_msg(robot1_hand0)
    #     )

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
                [pose["pos"], pose["rotvec"]],
                axis=-1,
            ).astype(np.float32)
            state[f"robot{rid}_hand_angles"] = hand.astype(np.float32)
        return state

    def build_init_state(self):
        state = self.get_latest_control_state()
        if state is None:
            return None
        return state

    def init_target_errors(self, current_state):
        if current_state is None:
            return None

        errors = {}
        for rid in [0, 1]:
            pose_key = f"robot{rid}_target_pose"
            hand_key = f"robot{rid}_hand_angles"
            cur_pose = current_state[pose_key]
            tgt_pose = self.init_target[pose_key]
            cur_hand = current_state[hand_key]
            tgt_hand = self.init_target[hand_key]
            errors[rid] = {
                "pos": float(np.linalg.norm(tgt_pose[:3] - cur_pose[:3])),
                "rot": self.rotation_error(cur_pose[3:6], tgt_pose[3:6]),
                "hand": float(np.max(np.abs(tgt_hand - cur_hand))),
            }
        return errors

    def init_target_reached(self, errors):
        if errors is None:
            return False
        return all(
            item["pos"] <= self.init_pos_tolerance
            and item["rot"] <= self.init_rot_tolerance
            and item["hand"] <= self.init_hand_tolerance
            for item in errors.values()
        )

    def format_init_errors(self, errors):
        if errors is None:
            return "feedback missing"
        return ", ".join(
            f"r{rid}: pos={item['pos']:.4f}, rot={item['rot']:.4f}, "
            f"hand={item['hand']:.1f}"
            for rid, item in errors.items()
        )

    def transition_from_init(self):
        self.action_queue.clear()
        self.obs_history.clear()
        self.last_command = None
        self.episode_start_pose_mats = None
        self.last_infer_time = 0.0

        if self.run_policy_after_init:
            self.policy_phase = "policy"
            self.get_logger().info(
                "Init target reached. Switching to policy phase; waiting for a fresh obs window."
            )
        else:
            self.policy_phase = "done"
            self.get_logger().info(
                "Init target reached. Policy inference is disabled; holding node idle."
            )

    def run_init_phase(self, now):
        if self.init_start_time <= 0.0:
            self.init_start_time = now

        current_state = self.build_init_state()
        errors = self.init_target_errors(current_state)
        reached = self.init_target_reached(errors)
        elapsed = now - self.init_start_time

        if reached:
            self.transition_from_init()
            return

        if elapsed >= self.init_hold_sec:
            self.policy_phase = "done"
            self.get_logger().warn(
                "Init hold_sec elapsed before reaching target. "
                f"Stopping init publish. errors=[{self.format_init_errors(errors)}]"
            )
            return

        if now - self.init_last_publish_time >= self.init_publish_period:
            self.publish_real_command(self.init_target)
            self.init_last_publish_time = now

        if now - self.last_print_time > self.print_interval:
            self.get_logger().info(
                f"Init phase publishing frame {self.init_target['frame_idx']}, "
                f"elapsed={elapsed:.1f}s, errors=[{self.format_init_errors(errors)}]"
            )
            self.last_print_time = now

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
        delta = np.clip(target - current, -max_step, max_step)
        return current + delta

    @staticmethod
    def unwrap_rotvec_near(current_rotvec, target_rotvec):
        current = np.asarray(current_rotvec, dtype=np.float64)
        target = np.asarray(target_rotvec, dtype=np.float64)
        target = R.from_rotvec(target).as_rotvec()
        angle = float(np.linalg.norm(target))
        if angle < 1e-8:
            return target.astype(np.float32)

        axis = target / angle
        candidates = [
            target,
            target + 2.0 * np.pi * axis,
            target - 2.0 * np.pi * axis,
        ]
        best = min(candidates, key=lambda x: float(np.linalg.norm(x - current)))
        return best.astype(np.float32)

    @staticmethod
    def limit_rotation_step_so3(current_rotvec, target_rotvec, max_step):
        current_rot = R.from_rotvec(np.asarray(current_rotvec, dtype=np.float64))
        target_rot = R.from_rotvec(np.asarray(target_rotvec, dtype=np.float64))
        delta_rotvec = (target_rot * current_rot.inv()).as_rotvec()
        delta_norm = float(np.linalg.norm(delta_rotvec))
        if delta_norm <= max_step or delta_norm < 1e-8:
            return np.asarray(target_rotvec, dtype=np.float32).copy()

        step_rot = R.from_rotvec(delta_rotvec * (max_step / delta_norm))
        command_rot = step_rot * current_rot
        return command_rot.as_rotvec().astype(np.float32)

    def limit_rotation_step(self, current_rotvec, target_rotvec, max_step):
        if self.rotation_limit_mode == "none":
            return np.asarray(target_rotvec, dtype=np.float32).copy()
        if self.rotation_limit_mode == "so3":
            return self.limit_rotation_step_so3(current_rotvec, target_rotvec, max_step)
        if self.rotation_limit_mode == "linear_unwrapped":
            unwrapped_target = self.unwrap_rotvec_near(current_rotvec, target_rotvec)
            return self.limit_vector_step(current_rotvec, unwrapped_target, max_step)
        if self.rotation_limit_mode == "linear":
            return self.limit_vector_step(current_rotvec, target_rotvec, max_step)
        raise ValueError(f"Unknown rotation_limit_mode: {self.rotation_limit_mode}")

    @staticmethod
    def rotation_error(current_rotvec, target_rotvec):
        current_rot = R.from_rotvec(np.asarray(current_rotvec, dtype=np.float64))
        target_rot = R.from_rotvec(np.asarray(target_rotvec, dtype=np.float64))
        return float(np.linalg.norm((target_rot * current_rot.inv()).as_rotvec()))

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

            cmd_pos = self.limit_vector_step(
                cur_pose[:3], tgt_pose[:3], self.pos_step_limit
            )
            cmd_rot = self.limit_rotation_step(
                cur_pose[3:6], tgt_pose[3:6], self.rot_step_limit
            )
            cmd_hand = self.limit_elementwise_step(
                cur_hand, tgt_hand, self.hand_step_limit
            )

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
        end = min(
            start + self.action_chunk_size,
            action_world["robot0_target_pose"].shape[0],
        )
        if start >= end:
            return

        current_state = self.get_latest_control_state()
        self.action_queue.clear()
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
        if not self.enable_real_publish or len(self.action_queue) == 0:
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
        if reached and len(self.action_queue) > 0:
            self.action_queue.popleft()

    def print_obs_debug(self, obs_world):
        self.get_logger().info(
            f"robot0 pos latest: {obs_world['robot0_eef_pos'][-1]}"
        )
        self.get_logger().info(
            f"robot1 pos latest: {obs_world['robot1_eef_pos'][-1]}"
        )

    def print_action_debug(self, action_world):
        self.get_logger().info(
            f"robot0 target pose[0]: {action_world['robot0_target_pose'][0]}"
        )
        self.get_logger().info(
            f"robot1 target pose[0]: {action_world['robot1_target_pose'][0]}"
        )
        self.get_logger().info(
            f"robot0 hand[0]: {action_world['robot0_hand_angles'][0]}"  
        )
        self.get_logger().info(
            f"robot1 hand[0]: {action_world['robot1_hand_angles'][0]}"
        )

    def close_debug_files(self):
        if self.debug_action_file is not None:
            self.debug_action_file.flush()
            self.debug_action_file.close()
            self.debug_action_file = None
        if self.kinect_video_writer is not None:
            self.kinect_video_writer.release()
            self.kinect_video_writer = None
            self.get_logger().info(
                f"Closed Kinect video: {self.kinect_video_path}, "
                f"frames={self.kinect_video_frame_count}"
            )

    def timer_cb(self):
        now = self.now_sec()

        if self.policy_phase == "done":
            return

        if self.policy_phase == "init":
            self.run_init_phase(now)
            return

        snapshot, stale_info = self.build_snapshot()

        if snapshot is None:
            if now - self.last_print_time > self.print_interval:
                msg = ", ".join(
                    [
                        f"{key}: empty" if age is None else f"{key}: age={age:.3f}s"
                        for key, age in stale_info
                    ]
                )
                self.get_logger().warn(f"Waiting for fresh topics: {msg}")
                self.last_print_time = now
            return

        self.obs_history.append(snapshot)

        obs_world = self.stack_obs_world()
        if obs_world is None:
            return

        if self.episode_start_pose_mats is None:
            self.episode_start_pose_mats = self.get_obs_start_pose_mats(obs_world)
            self.get_logger().info("Captured policy episode start poses for wrt_start obs.")

        self.save_obs_capture(obs_world)

        action_world = None

        if now - self.last_infer_time > self.infer_interval:
            if self.deploy is None:
                if now - self.last_print_time > self.print_interval:
                    self.get_logger().warn(
                        "Policy phase requested, but deploy model is not loaded."
                    )
                    self.last_print_time = now
                return
            try:
                action_world = self.deploy.predict(
                    obs_world,
                    episode_start_pose_mats=self.episode_start_pose_mats,
                )

                # self.publish_converted_debug_action(action_world)
                self.update_action_queue(action_world)

                self.last_infer_time = now

            except Exception as e:
                self.get_logger().error(f"Policy inference failed: {e}")
                return

        self.publish_queued_action()

        if now - self.last_print_time > self.print_interval:
            self.print_obs_debug(obs_world)
            if action_world is not None:
                self.print_action_debug(action_world)
            self.last_print_time = now


def main(args=None):
    rclpy.init(args=args)
    node = PolicyRosNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_debug_files()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
