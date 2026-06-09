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
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.spatial.transform import Rotation as R

from custom_msgs.msg import DesiredPose


WORKSPACE_ROOT = os.environ.get("INSPIRE_WORKSPACE_ROOT", "/workspace/zhicheng_ws")
DATA_ROOT = os.environ.get("INSPIRE_DATA_ROOT", "/mnt/data")
DEBUG_ROOT = os.environ.get("INSPIRE_DEBUG_ROOT", WORKSPACE_ROOT)
DEFAULT_EPISODE_PATH = os.environ.get(
    "INSPIRE_RAW_REPLAY_EPISODE",
    os.path.join(
        DATA_ROOT,
        "zhicheng_ws/dataset/test_80/episode_20260422_103247_085b92.npz",
    ),
)


class RawEpisodePoseReplayNode(Node):
    """Replay raw actual end-effector pose matrices from one episode."""

    def __init__(self):
        super().__init__("raw_episode_pose_replay_node")

        self.episode_path = self.declare_parameter(
            "episode_path", DEFAULT_EPISODE_PATH
        ).value
        self.left_pose_key = self.declare_parameter(
            "left_pose_key", "actual_ee_pose_left"
        ).value
        self.right_pose_key = self.declare_parameter(
            "right_pose_key", "actual_ee_pose_right"
        ).value
        self.start_frame = int(self.declare_parameter("start_frame", 0).value)
        self.end_frame = int(self.declare_parameter("end_frame", -1).value)
        self.frame_stride = int(self.declare_parameter("frame_stride", 1).value)
        self.publish_period = float(
            self.declare_parameter("publish_period", 0.1).value
        )
        self.use_episode_timestamps = bool(
            self.declare_parameter("use_episode_timestamps", False).value
        )
        self.playback_speed = float(self.declare_parameter("playback_speed", 1.0).value)
        self.enable_real_publish = bool(
            self.declare_parameter("enable_real_publish", False).value
        )
        self.require_start_near = bool(
            self.declare_parameter("require_start_near", True).value
        )
        self.start_pos_tolerance = float(
            self.declare_parameter("start_pos_tolerance", 0.03).value
        )
        self.start_rot_tolerance = float(
            self.declare_parameter("start_rot_tolerance", 0.25).value
        )
        self.max_age = float(self.declare_parameter("max_age", 0.5).value)
        self.print_interval = float(self.declare_parameter("print_interval", 2.0).value)

        if self.frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {self.frame_stride}")
        if self.publish_period <= 0.0:
            raise ValueError(f"publish_period must be positive, got {self.publish_period}")
        if self.playback_speed <= 0.0:
            raise ValueError(f"playback_speed must be positive, got {self.playback_speed}")

        self.load_episode()
        self.frame_idx = self.start_frame
        self.next_publish_time = 0.0
        self.first_episode_time = self.pose_times[0]
        self.first_wall_time = None
        self.last_print_time = 0.0
        self.finished = False
        self.start_check_passed = not self.require_start_near

        run_name = f"raw_episode_replay_{time.strftime('%Y%m%d_%H%M%S')}"
        self.debug_dir = self.declare_parameter(
            "debug_dir",
            os.path.join(DEBUG_ROOT, "debug_deploy_actions", run_name),
        ).value
        self.debug_csv_path = os.path.join(self.debug_dir, "raw_pose_replay.csv")
        self.debug_file = None
        self.debug_writer = None
        self.debug_seq = 0
        self.init_debug_csv()

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pose_buffers = {
            "left": deque(maxlen=100),
            "right": deque(maxlen=100),
        }
        self.create_subscription(
            DesiredPose,
            "/frankaLeft/ee_pose_matrix",
            lambda msg: self.pose_cb(msg, "left"),
            state_qos,
        )
        self.create_subscription(
            DesiredPose,
            "/frankaRight/ee_pose_matrix",
            lambda msg: self.pose_cb(msg, "right"),
            state_qos,
        )
        self.pub_left_pose_cmd = self.create_publisher(
            DesiredPose, "/left/desired_pose_matrix", 10
        )
        self.pub_right_pose_cmd = self.create_publisher(
            DesiredPose, "/right/desired_pose_matrix", 10
        )

        self.timer = self.create_timer(0.02, self.timer_cb)
        self.log_episode_summary()

    def load_episode(self):
        self.episode_path = os.path.expanduser(self.episode_path)
        data = np.load(self.episode_path, allow_pickle=True)
        if self.left_pose_key not in data or self.right_pose_key not in data:
            raise KeyError(
                f"Missing raw pose keys {self.left_pose_key}/{self.right_pose_key} "
                f"in {self.episode_path}"
            )

        left_raw = np.asarray(data[self.left_pose_key], dtype=np.float64)
        right_raw = np.asarray(data[self.right_pose_key], dtype=np.float64)
        if left_raw.ndim != 3 or left_raw.shape[1:] != (4, 4):
            raise ValueError(f"{self.left_pose_key} has invalid shape {left_raw.shape}")
        if right_raw.shape != left_raw.shape:
            raise ValueError(
                f"{self.right_pose_key} shape {right_raw.shape} != "
                f"{self.left_pose_key} shape {left_raw.shape}"
            )

        self.left_layout, self.left_std_mats = self.to_standard_mats(
            left_raw, self.left_pose_key
        )
        self.right_layout, self.right_std_mats = self.to_standard_mats(
            right_raw, self.right_pose_key
        )
        self.n_frames = left_raw.shape[0]

        if self.start_frame < 0 or self.start_frame >= self.n_frames:
            raise ValueError(f"start_frame {self.start_frame} not in [0, {self.n_frames})")
        self.end_frame = self.n_frames if self.end_frame < 0 else min(
            self.end_frame, self.n_frames
        )
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame {self.end_frame} must be after start_frame {self.start_frame}"
            )

        time_key = f"{self.left_pose_key}_t"
        if time_key in data:
            self.pose_times = np.asarray(data[time_key], dtype=np.float64)
        else:
            self.pose_times = np.arange(self.n_frames, dtype=np.float64) * self.publish_period
        if self.pose_times.shape != (self.n_frames,):
            raise ValueError(f"{time_key} has invalid shape {self.pose_times.shape}")
        self.pose_times = self.pose_times[self.start_frame:self.end_frame]

    @staticmethod
    def layout_error(mats, translation_in_column):
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        if translation_in_column:
            return float(np.max(np.abs(mats[:, 3, :] - expected)))
        return float(np.max(np.abs(mats[:, :, 3] - expected)))

    def to_standard_mats(self, mats, key):
        column_layout_error = self.layout_error(mats, translation_in_column=True)
        row_layout_error = self.layout_error(mats, translation_in_column=False)
        if column_layout_error < 1e-4 and row_layout_error >= 1e-4:
            layout = "standard_column_translation"
            standard = mats.copy()
        elif row_layout_error < 1e-4 and column_layout_error >= 1e-4:
            layout = "raw_row_translation_transposed"
            standard = np.transpose(mats, (0, 2, 1)).copy()
        elif row_layout_error < 1e-4 and column_layout_error < 1e-4:
            raise ValueError(
                f"{key} matrix layout is ambiguous: no non-zero translation found. "
                "Inspect the raw matrices before replay."
            )
        else:
            raise ValueError(
                f"{key} is not a supported homogeneous transform layout: "
                f"column_layout_error={column_layout_error:.3g}, "
                f"row_layout_error={row_layout_error:.3g}"
            )

        bottom_error = self.layout_error(standard, translation_in_column=True)
        if bottom_error >= 1e-4:
            raise ValueError(f"{key} failed standard-matrix conversion, error={bottom_error}")
        return layout, standard.astype(np.float32)

    def log_episode_summary(self):
        left_start = self.left_std_mats[self.start_frame]
        right_start = self.right_std_mats[self.start_frame]
        left_end = self.left_std_mats[self.end_frame - 1]
        right_end = self.right_std_mats[self.end_frame - 1]
        duration = float(self.pose_times[-1] - self.pose_times[0])
        self.get_logger().info(
            f"Raw episode loaded: {self.episode_path}, frames=[{self.start_frame}, "
            f"{self.end_frame}), stride={self.frame_stride}, raw_duration={duration:.3f}s"
        )
        self.get_logger().info(
            f"Matrix layout: left={self.left_layout}, right={self.right_layout}. "
            "Published DesiredPose uses standard_mat.T.reshape(-1)."
        )
        self.get_logger().info(
            f"Raw start pos: left={left_start[:3, 3]}, right={right_start[:3, 3]}"
        )
        self.get_logger().info(
            f"Raw end pos: left={left_end[:3, 3]}, right={right_end[:3, 3]}"
        )
        self.get_logger().info(
            f"Raw pose replay started. real_publish={self.enable_real_publish}, "
            f"require_start_near={self.require_start_near}, CSV={self.debug_csv_path}"
        )

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def init_debug_csv(self):
        os.makedirs(self.debug_dir, exist_ok=True)
        self.debug_file = open(self.debug_csv_path, "w", newline="")
        fieldnames = [
            "seq",
            "ros_time",
            "event",
            "frame",
            "side",
            "matrix_layout",
            "target_x",
            "target_y",
            "target_z",
            "target_rx",
            "target_ry",
            "target_rz",
            "actual_x",
            "actual_y",
            "actual_z",
            "actual_rx",
            "actual_ry",
            "actual_rz",
            "actual_target_pos_err",
            "actual_target_rot_err",
        ]
        self.debug_writer = csv.DictWriter(self.debug_file, fieldnames=fieldnames)
        self.debug_writer.writeheader()
        self.debug_file.flush()

    def pose_cb(self, msg, side):
        try:
            mat = np.asarray(msg.pose_matrix, dtype=np.float64).reshape(4, 4).T
            self.pose_buffers[side].append((self.now_sec(), mat.astype(np.float32)))
        except Exception as exc:
            self.get_logger().warn(f"{side} pose_cb error: {exc}")

    def latest_pose(self, side):
        if len(self.pose_buffers[side]) == 0:
            return None
        stamp, mat = self.pose_buffers[side][-1]
        if self.now_sec() - stamp > self.max_age:
            return None
        return mat

    @staticmethod
    def matrix_to_pose6(mat):
        pos = mat[:3, 3]
        rotvec = R.from_matrix(mat[:3, :3]).as_rotvec()
        return np.concatenate([pos, rotvec], axis=0).astype(np.float32)

    @staticmethod
    def rotation_error(mat_a, mat_b):
        rot_a = R.from_matrix(mat_a[:3, :3])
        rot_b = R.from_matrix(mat_b[:3, :3])
        return float(np.linalg.norm((rot_b * rot_a.inv()).as_rotvec()))

    def current_start_errors(self):
        live_left = self.latest_pose("left")
        live_right = self.latest_pose("right")
        if live_left is None or live_right is None:
            return None

        target_left = self.left_std_mats[self.start_frame]
        target_right = self.right_std_mats[self.start_frame]
        return {
            "left": (
                float(np.linalg.norm(live_left[:3, 3] - target_left[:3, 3])),
                self.rotation_error(live_left, target_left),
            ),
            "right": (
                float(np.linalg.norm(live_right[:3, 3] - target_right[:3, 3])),
                self.rotation_error(live_right, target_right),
            ),
        }

    def pass_start_check(self):
        if self.start_check_passed:
            return True
        errors = self.current_start_errors()
        if errors is None:
            return False

        self.start_check_passed = all(
            pos_err <= self.start_pos_tolerance and rot_err <= self.start_rot_tolerance
            for pos_err, rot_err in errors.values()
        )
        if self.start_check_passed:
            self.get_logger().info("Raw replay start-pose check passed.")
        return self.start_check_passed

    @staticmethod
    def mat_to_desired_pose_msg(mat):
        msg = DesiredPose()
        msg.pose_matrix = np.asarray(mat, dtype=np.float32).T.reshape(-1).astype(
            np.float64
        ).tolist()
        return msg

    def publish_frame(self):
        left_target = self.left_std_mats[self.frame_idx]
        right_target = self.right_std_mats[self.frame_idx]
        if self.enable_real_publish:
            self.pub_left_pose_cmd.publish(self.mat_to_desired_pose_msg(left_target))
            self.pub_right_pose_cmd.publish(self.mat_to_desired_pose_msg(right_target))
        self.write_debug_row("left", self.left_layout, left_target)
        self.write_debug_row("right", self.right_layout, right_target)

    def write_debug_row(self, side, layout, target_mat):
        target_pose = self.matrix_to_pose6(target_mat)
        actual_mat = self.latest_pose(side)
        if actual_mat is None:
            actual_pose = np.full(6, np.nan, dtype=np.float32)
            pos_err = np.nan
            rot_err = np.nan
        else:
            actual_pose = self.matrix_to_pose6(actual_mat)
            pos_err = float(np.linalg.norm(actual_pose[:3] - target_pose[:3]))
            rot_err = self.rotation_error(actual_mat, target_mat)

        self.debug_writer.writerow({
            "seq": self.debug_seq,
            "ros_time": self.now_sec(),
            "event": "published" if self.enable_real_publish else "dry_run",
            "frame": self.frame_idx,
            "side": side,
            "matrix_layout": layout,
            "target_x": target_pose[0],
            "target_y": target_pose[1],
            "target_z": target_pose[2],
            "target_rx": target_pose[3],
            "target_ry": target_pose[4],
            "target_rz": target_pose[5],
            "actual_x": actual_pose[0],
            "actual_y": actual_pose[1],
            "actual_z": actual_pose[2],
            "actual_rx": actual_pose[3],
            "actual_ry": actual_pose[4],
            "actual_rz": actual_pose[5],
            "actual_target_pos_err": pos_err,
            "actual_target_rot_err": rot_err,
        })
        self.debug_seq += 1
        self.debug_file.flush()

    def frame_due(self, now):
        if not self.use_episode_timestamps:
            return now >= self.next_publish_time
        if self.first_wall_time is None:
            self.first_wall_time = now
        local_idx = self.frame_idx - self.start_frame
        episode_time = self.pose_times[local_idx] - self.first_episode_time
        return (now - self.first_wall_time) >= episode_time / self.playback_speed

    def timer_cb(self):
        now = self.now_sec()
        if self.finished:
            return
        if self.frame_idx >= self.end_frame:
            self.finished = True
            self.get_logger().info("Raw episode pose replay finished.")
            return

        if self.enable_real_publish and not self.pass_start_check():
            if now - self.last_print_time > self.print_interval:
                errors = self.current_start_errors()
                error_msg = "feedback missing"
                if errors is not None:
                    error_msg = ", ".join(
                        f"{side}: pos={pos_err:.4f}, rot={rot_err:.4f}"
                        for side, (pos_err, rot_err) in errors.items()
                    )
                self.get_logger().warn(
                    "Waiting for fresh feedback near the raw start pose. "
                    "Set require_start_near:=false only if an absolute jump is intended. "
                    f"[{error_msg}]"
                )
                self.last_print_time = now
            return

        if not self.frame_due(now):
            return

        self.publish_frame()
        if now - self.last_print_time > self.print_interval:
            self.get_logger().info(
                f"raw frame={self.frame_idx}/{self.end_frame - 1}, "
                f"real_publish={self.enable_real_publish}"
            )
            self.last_print_time = now
        self.frame_idx += self.frame_stride
        self.next_publish_time = now + self.publish_period

    def close(self):
        if self.debug_file is not None:
            self.debug_file.flush()
            self.debug_file.close()
            self.debug_file = None


def main(args=None):
    rclpy.init(args=args)
    node = RawEpisodePoseReplayNode()
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
