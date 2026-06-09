#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from custom_msgs.msg import DesiredPose
from inspire_interfaces.msg import SetAngle1


class DebugActionListener(Node):
    def __init__(self):
        super().__init__("debug_action_listener")

        self.create_subscription(
            Float32MultiArray,
            "/policy_debug/robot0_target_pose",
            lambda msg: self.cb(msg, "robot0_target_pose"),
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "/policy_debug/robot1_target_pose",
            lambda msg: self.cb(msg, "robot1_target_pose"),
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "/policy_debug/robot0_hand_angles",
            lambda msg: self.cb(msg, "robot0_hand_angles"),
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            "/policy_debug/robot1_hand_angles",
            lambda msg: self.cb(msg, "robot1_hand_angles"),
            10,
        )

        self.create_subscription(
            DesiredPose,
            "/policy_debug/robot0_desired_pose_matrix",
            lambda msg: self.pose_matrix_cb(msg, "robot0_desired_pose_matrix"),
            10,
        )

        self.create_subscription(
            DesiredPose,
            "/policy_debug/robot1_desired_pose_matrix",
            lambda msg: self.pose_matrix_cb(msg, "robot1_desired_pose_matrix"),
            10,
        )

        self.create_subscription(
            SetAngle1,
            "/policy_debug/robot0_set_angle_data",
            lambda msg: self.set_angle_cb(msg, "robot0_set_angle_data"),
            10,
        )

        self.create_subscription(
            SetAngle1,
            "/policy_debug/robot1_set_angle_data",
            lambda msg: self.set_angle_cb(msg, "robot1_set_angle_data"),
            10,
        )

        self.get_logger().info("DebugActionListener started.")

    def cb(self, msg, name):
        arr = np.asarray(msg.data, dtype=np.float32)

        if arr.size != 96:
            self.get_logger().warn(f"{name}: expected 96 values, got {arr.size}")
            return

        arr = arr.reshape(16, 6)

        self.get_logger().info(
            f"{name}: shape={arr.shape}, first={arr[0]}"
        )
    def pose_matrix_cb(self, msg, name):
        mat = np.asarray(msg.pose_matrix, dtype=np.float64).reshape(4, 4)

        # 你的控制消息是 mat.T.flatten()，所以这里转回标准矩阵方便看
        mat_std = mat.T
        pos = mat_std[:3, 3]

        self.get_logger().info(
            f"{name}: pos={pos}, raw_matrix_flat_len={len(msg.pose_matrix)}"
        )


    def set_angle_cb(self, msg, name):
        self.get_logger().info(
            f"{name}: finger_ids={list(msg.finger_ids)}, angles={list(msg.angles)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DebugActionListener()
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