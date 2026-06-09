#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node

from inspire_interfaces.msg import HandKeypoint, GetAngleAct1
from custom_msgs.msg import DesiredPose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class MultiTopicCsvRecorder(Node):
    def __init__(self):
        super().__init__("multi_topic_csv_recorder")
        default_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        pose_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.declare_parameter("save_dir", "./csv_record")
        self.save_dir = self.get_parameter("save_dir").value

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(self.save_dir, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        self.files = {}
        self.writers = {}

        self.init_hand_csv("/Lefthandpoint", "left_handpoint.csv")
        self.init_hand_csv("/Righthandpoint", "right_handpoint.csv")

        self.init_angle_csv("/left/angle_data", "left_angle_data.csv")
        self.init_angle_csv("/right/angle_data", "right_angle_data.csv")

        self.init_pose_csv("/frankaLeft/ee_pose_matrix", "frankaLeft_ee_pose_matrix.csv")
        self.init_pose_csv("/frankaRight/ee_pose_matrix", "frankaRight_ee_pose_matrix.csv")

        self.create_subscription(
            HandKeypoint,
            "/Lefthandpoint",
            lambda msg: self.hand_callback(msg, "/Lefthandpoint"),
            default_qos
        )

        self.create_subscription(
            HandKeypoint,
            "/Righthandpoint",
            lambda msg: self.hand_callback(msg, "/Righthandpoint"),
            10
        )

        self.create_subscription(
            GetAngleAct1,
            "/left/angle_data",
            lambda msg: self.angle_callback(msg, "/left/angle_data"),
            10
        )

        self.create_subscription(
            GetAngleAct1,
            "/right/angle_data",
            lambda msg: self.angle_callback(msg, "/right/angle_data"),
            10
        )

        self.create_subscription(
            DesiredPose,
            "/frankaLeft/ee_pose_matrix",
            lambda msg: self.pose_callback(msg, "/frankaLeft/ee_pose_matrix"),
            pose_qos
        )

        self.create_subscription(
            DesiredPose,
            "/frankaRight/ee_pose_matrix",
            lambda msg: self.pose_callback(msg, "/frankaRight/ee_pose_matrix"),
            pose_qos
        )

        self.get_logger().info(f"Saving CSV files to: {self.save_dir}")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def open_csv(self, topic, filename, header):
        path = os.path.join(self.save_dir, filename)
        f = open(path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(header)

        self.files[topic] = f
        self.writers[topic] = writer

        self.get_logger().info(f"{topic} -> {path}")

    def init_hand_csv(self, topic, filename):
        header = ["time"]
        for i in range(21):
            header += [f"p{i}_x", f"p{i}_y", f"p{i}_z"]
        self.open_csv(topic, filename, header)

    def init_angle_csv(self, topic, filename):
        header = ["time"]

        # 按实际 GetAngleAct1 内容动态保存成字符串，最稳
        header += [
            "finger_ids",
            "angles",
            "finger_names"
        ]

        self.open_csv(topic, filename, header)

    def init_pose_csv(self, topic, filename):
        header = ["time"]
        for i in range(4):
            for j in range(4):
                header.append(f"m{i}{j}")
        self.open_csv(topic, filename, header)

    def hand_callback(self, msg: HandKeypoint, topic: str):
        row = [f"{self.now_sec():.9f}"]

        for p in msg.points:
            row += [
                f"{p.x:.9f}",
                f"{p.y:.9f}",
                f"{p.z:.9f}",
            ]

        self.writers[topic].writerow(row)
        self.files[topic].flush()

    def angle_callback(self, msg: GetAngleAct1, topic: str):
        row = [
            f"{self.now_sec():.9f}",
            ";".join(map(str, msg.finger_ids)),
            ";".join(map(str, msg.angles)),
            ";".join(map(str, msg.finger_names)),
        ]

        self.writers[topic].writerow(row)
        self.files[topic].flush()

    def pose_callback(self, msg: DesiredPose, topic: str):
        row = [f"{self.now_sec():.9f}"]

        for v in msg.pose_matrix:
            row.append(f"{v:.9f}")

        self.writers[topic].writerow(row)
        self.files[topic].flush()

    def destroy_node(self):
        for f in self.files.values():
            f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MultiTopicCsvRecorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()