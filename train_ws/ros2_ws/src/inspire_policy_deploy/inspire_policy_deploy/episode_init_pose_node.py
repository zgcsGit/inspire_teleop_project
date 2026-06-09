#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from collections import deque

import numpy as np
import rclpy
from custom_msgs.msg import DesiredPose
from inspire_interfaces.msg import GetAngleAct1, SetAngle1
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R


DATA_ROOT = os.environ.get("INSPIRE_DATA_ROOT", "/mnt/data")
DEFAULT_EPISODE_PATH = os.environ.get(
    "INSPIRE_INIT_EPISODE_PATH",
    os.path.join(
        DATA_ROOT,
        "zhicheng_ws/dataset/test_2/episode_20260522_181158_e8d340.npz",
    ),
)
ACTIVE_HAND_ANGLE_IDXS = [3, 4, 5]


class EpisodeInitPoseNode(Node):
    """Publish one raw episode frame as the robot initial state."""

    def __init__(self):
        super().__init__("episode_init_pose_node")

        self.episode_path = self.declare_parameter(
            "episode_path", DEFAULT_EPISODE_PATH
        ).value
        self.frame_idx = int(self.declare_parameter("frame_idx", 3).value)
        self.publish_period = float(
            self.declare_parameter("publish_period", 0.1).value
        )
        self.hold_sec = float(self.declare_parameter("hold_sec", 10.0).value)
        self.enable_real_publish = bool(
            self.declare_parameter("enable_real_publish", False).value
        )
        self.stop_when_reached = bool(
            self.declare_parameter("stop_when_reached", True).value
        )
        self.inactive_angle = int(self.declare_parameter("inactive_angle", 1000).value)
        self.pos_tolerance = float(self.declare_parameter("pos_tolerance", 0.015).value)
        self.rot_tolerance = float(self.declare_parameter("rot_tolerance", 0.20).value)
        self.hand_tolerance = float(self.declare_parameter("hand_tolerance", 25.0).value)
        self.max_age = float(self.declare_parameter("max_age", 0.5).value)
        self.print_interval = float(self.declare_parameter("print_interval", 1.0).value)

        if self.publish_period <= 0.0:
            raise ValueError("publish_period must be positive")
        if self.hold_sec < 0.0:
            raise ValueError("hold_sec must be non-negative")

        self.left_mat, self.right_mat, self.left_hand6, self.right_hand6 = (
            self.load_target_frame()
        )

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pose_buffers = {
            "left": deque(maxlen=50),
            "right": deque(maxlen=50),
        }
        self.hand_buffers = {
            "left": deque(maxlen=50),
            "right": deque(maxlen=50),
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
        self.create_subscription(
            GetAngleAct1,
            "/left/angle_data",
            lambda msg: self.angle_cb(msg, "left"),
            state_qos,
        )
        self.create_subscription(
            GetAngleAct1,
            "/right/angle_data",
            lambda msg: self.angle_cb(msg, "right"),
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

        self.start_time = self.now_sec()
        self.last_publish_time = 0.0
        self.last_print_time = 0.0
        self.finished = False
        self.timer = self.create_timer(0.02, self.timer_cb)

        self.log_target_summary()

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

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
            return "standard_column_translation", mats.copy()
        if row_layout_error < 1e-4 and column_layout_error >= 1e-4:
            return "raw_row_translation_transposed", np.transpose(mats, (0, 2, 1)).copy()
        raise ValueError(
            f"{key} matrix layout is unsupported or ambiguous: "
            f"column_error={column_layout_error:.3g}, row_error={row_layout_error:.3g}"
        )

    @staticmethod
    def extract_hand6(angle_obj_arr, frame_idx, field_name):
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
        command = np.full(6, 1000, dtype=np.int32)
        command[ACTIVE_HAND_ANGLE_IDXS] = angles[ACTIVE_HAND_ANGLE_IDXS]
        return angles, command

    def load_target_frame(self):
        self.episode_path = os.path.expanduser(self.episode_path)
        data = np.load(self.episode_path, allow_pickle=True)
        required = [
            "actual_ee_pose_left",
            "actual_ee_pose_right",
            "angle_left",
            "angle_right",
        ]
        for key in required:
            if key not in data:
                raise KeyError(f"{self.episode_path} missing '{key}'")

        left_raw = np.asarray(data["actual_ee_pose_left"], dtype=np.float32)
        right_raw = np.asarray(data["actual_ee_pose_right"], dtype=np.float32)
        if left_raw.ndim != 3 or left_raw.shape[1:] != (4, 4):
            raise ValueError(f"actual_ee_pose_left has invalid shape {left_raw.shape}")
        if right_raw.shape != left_raw.shape:
            raise ValueError(
                f"actual_ee_pose_right shape {right_raw.shape} != {left_raw.shape}"
            )
        if self.frame_idx < 0 or self.frame_idx >= left_raw.shape[0]:
            raise ValueError(
                f"frame_idx {self.frame_idx} not in [0, {left_raw.shape[0]})"
            )

        self.left_layout, left_std = self.to_standard_mats(
            left_raw, "actual_ee_pose_left"
        )
        self.right_layout, right_std = self.to_standard_mats(
            right_raw, "actual_ee_pose_right"
        )
        left_raw_hand, left_hand6 = self.extract_hand6(
            data["angle_left"], self.frame_idx, "angle_left"
        )
        right_raw_hand, right_hand6 = self.extract_hand6(
            data["angle_right"], self.frame_idx, "angle_right"
        )
        inactive_idxs = [
            idx for idx in range(6) if idx not in ACTIVE_HAND_ANGLE_IDXS
        ]
        left_hand6[inactive_idxs] = self.inactive_angle
        right_hand6[inactive_idxs] = self.inactive_angle
        self.left_raw_hand6 = left_raw_hand
        self.right_raw_hand6 = right_raw_hand
        return (
            left_std[self.frame_idx].astype(np.float32),
            right_std[self.frame_idx].astype(np.float32),
            left_hand6.astype(np.int32),
            right_hand6.astype(np.int32),
        )

    @staticmethod
    def mat_to_desired_pose_msg(mat):
        msg = DesiredPose()
        msg.pose_matrix = np.asarray(mat, dtype=np.float32).T.reshape(-1).astype(
            np.float64
        ).tolist()
        return msg

    @staticmethod
    def hand6_to_set_angle_msg(hand6):
        hand6 = np.clip(np.asarray(hand6, dtype=np.float32), 0, 1000).astype(np.int32)
        msg = SetAngle1()
        msg.finger_ids = [1, 2, 3, 4, 5, 6]
        msg.angles = hand6.tolist()
        return msg

    @staticmethod
    def matrix_to_pose6(mat):
        mat = np.asarray(mat, dtype=np.float64)
        pos = mat[:3, 3]
        rotvec = R.from_matrix(mat[:3, :3]).as_rotvec()
        return np.concatenate([pos, rotvec], axis=0).astype(np.float32)

    @staticmethod
    def rotation_error(current_mat, target_mat):
        cur = R.from_matrix(np.asarray(current_mat, dtype=np.float64)[:3, :3])
        tgt = R.from_matrix(np.asarray(target_mat, dtype=np.float64)[:3, :3])
        return float(np.linalg.norm((tgt * cur.inv()).as_rotvec()))

    def log_target_summary(self):
        left_pose = self.matrix_to_pose6(self.left_mat)
        right_pose = self.matrix_to_pose6(self.right_mat)
        self.get_logger().info(
            f"Loaded init target: {self.episode_path}, frame={self.frame_idx}, "
            f"real_publish={self.enable_real_publish}, hold_sec={self.hold_sec}"
        )
        self.get_logger().info(
            f"Matrix layout: left={self.left_layout}, right={self.right_layout}. "
            "Published DesiredPose uses standard_mat.T.reshape(-1)."
        )
        self.get_logger().info(
            f"left pose6={left_pose}, raw_hand6={self.left_raw_hand6}, "
            f"cmd_hand6={self.left_hand6}"
        )
        self.get_logger().info(
            f"right pose6={right_pose}, raw_hand6={self.right_raw_hand6}, "
            f"cmd_hand6={self.right_hand6}"
        )

    def pose_cb(self, msg, side):
        mat = np.asarray(msg.pose_matrix, dtype=np.float32).reshape(4, 4).T
        self.pose_buffers[side].append((self.now_sec(), mat))

    def angle_cb(self, msg, side):
        angles = np.asarray(msg.angles, dtype=np.float32).reshape(-1)
        if angles.shape[0] == 6:
            self.hand_buffers[side].append((self.now_sec(), angles))

    def latest(self, buffers, side):
        if not buffers[side]:
            return None
        stamp, value = buffers[side][-1]
        if self.now_sec() - stamp > self.max_age:
            return None
        return value

    def current_errors(self):
        errors = {}
        for side, target_mat, target_hand in [
            ("left", self.left_mat, self.left_hand6),
            ("right", self.right_mat, self.right_hand6),
        ]:
            pose = self.latest(self.pose_buffers, side)
            hand = self.latest(self.hand_buffers, side)
            if pose is None or hand is None:
                return None
            pos_err = float(np.linalg.norm(pose[:3, 3] - target_mat[:3, 3]))
            rot_err = self.rotation_error(pose, target_mat)
            hand_err = float(np.max(np.abs(hand - target_hand)))
            errors[side] = (pos_err, rot_err, hand_err)
        return errors

    def reached(self):
        errors = self.current_errors()
        if errors is None:
            return False
        return all(
            pos_err <= self.pos_tolerance
            and rot_err <= self.rot_tolerance
            and hand_err <= self.hand_tolerance
            for pos_err, rot_err, hand_err in errors.values()
        )

    def publish_target(self):
        if not self.enable_real_publish:
            return
        self.pub_left_pose_cmd.publish(self.mat_to_desired_pose_msg(self.left_mat))
        self.pub_right_pose_cmd.publish(self.mat_to_desired_pose_msg(self.right_mat))
        self.pub_left_hand_cmd.publish(self.hand6_to_set_angle_msg(self.left_hand6))
        self.pub_right_hand_cmd.publish(self.hand6_to_set_angle_msg(self.right_hand6))

    def timer_cb(self):
        if self.finished:
            return

        now = self.now_sec()
        elapsed = now - self.start_time
        is_reached = self.reached()
        should_stop = elapsed >= self.hold_sec
        if self.stop_when_reached and is_reached and elapsed >= 1.0:
            should_stop = True

        if should_stop:
            self.finished = True
            reason = "reached target" if is_reached else "hold_sec elapsed"
            self.get_logger().info(f"Episode init publish stopped: {reason}.")
            return

        if now - self.last_publish_time >= self.publish_period:
            self.publish_target()
            self.last_publish_time = now

        if now - self.last_print_time >= self.print_interval:
            errors = self.current_errors()
            if errors is None:
                err_msg = "feedback missing"
            else:
                err_msg = ", ".join(
                    f"{side}: pos={pos_err:.4f}, rot={rot_err:.4f}, hand={hand_err:.1f}"
                    for side, (pos_err, rot_err, hand_err) in errors.items()
                )
            self.get_logger().info(
                f"publishing init frame {self.frame_idx}, elapsed={elapsed:.1f}s, "
                f"reached={is_reached}, errors=[{err_msg}]"
            )
            self.last_print_time = now


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeInitPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
