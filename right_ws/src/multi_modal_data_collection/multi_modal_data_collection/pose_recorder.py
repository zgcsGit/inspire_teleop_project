#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from custom_msgs.msg import DesiredPose
import csv
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

class PoseRecorder(Node):
    def __init__(self):
        super().__init__('pose_recorder')

        self.max_samples = 100

        # ===== desired =====
        self.desired_records = []
        self.desired_started = False
        self.desired_start_time = None
        self.desired_first_received = False

        # ===== actual =====
        self.actual_records = []
        self.actual_started = False
        self.actual_start_time = None
        self.actual_first_received = False

        # subscribers
        self.sub_desired = self.create_subscription(
            DesiredPose,
            '/right/desired_pose_matrix',
            self.desired_callback,
            qos
        )

        self.sub_actual = self.create_subscription(
            DesiredPose,
            '/frankaRight/ee_pose_matrix',
            self.actual_callback,
            qos
        )

        self.get_logger().info("Pose recorder started.")

    # ===== desired =====
    def desired_callback(self, msg):
        now = time.time()

        if not self.desired_first_received:
            self.desired_first_received = True
            self.desired_start_time = now + 5.0
            self.get_logger().info("Desired: first message received, wait 5s...")

        if not self.desired_first_received:
            return

        if not self.desired_started:
            if now >= self.desired_start_time:
                self.desired_started = True
                self.get_logger().info("Desired: start recording")
            else:
                return

        if len(self.desired_records) < self.max_samples:
            row = [now] + list(msg.pose_matrix)
            self.desired_records.append(row)

        if len(self.desired_records) == self.max_samples:
            self.save_desired()

    # ===== actual =====
    def actual_callback(self, msg):
        now = time.time()

        if not self.actual_first_received:
            self.actual_first_received = True
            self.actual_start_time = now + 5.0
            self.get_logger().info("Actual: first message received, wait 5s...")

        if not self.actual_first_received:
            return

        if not self.actual_started:
            if now >= self.actual_start_time:
                self.actual_started = True
                self.get_logger().info("Actual: start recording")
            else:
                return

        if len(self.actual_records) < self.max_samples:
            row = [now] + list(msg.pose_matrix)
            self.actual_records.append(row)

        if len(self.actual_records) == self.max_samples:
            self.save_actual()

    # ===== save =====
    def save_desired(self):
        filename = "desired_pose.csv"
        header = ["timestamp"] + [f"d_{i}" for i in range(16)]

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.desired_records)

        self.get_logger().info(f"Saved desired to {filename}")

        self.check_done()

    def save_actual(self):
        filename = "actual_pose.csv"
        header = ["timestamp"] + [f"a_{i}" for i in range(16)]

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.actual_records)

        self.get_logger().info(f"Saved actual to {filename}")

        self.check_done()

    def check_done(self):
        if len(self.desired_records) >= self.max_samples and len(self.actual_records) >= self.max_samples:
            self.get_logger().info("Both recordings done. Shutting down.")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = PoseRecorder()
    rclpy.spin(node)


if __name__ == '__main__':
    main()