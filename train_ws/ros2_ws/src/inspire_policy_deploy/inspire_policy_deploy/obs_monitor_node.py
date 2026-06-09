#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import CompressedImage
from custom_msgs.msg import DesiredPose
from inspire_interfaces.msg import GetAngleAct1, GetTouchAct1


class TopicStats:
    def __init__(self):
        self.count = 0
        self.last_print_time = time.time()
        self.last_summary = "no data"

    def update(self, summary: str):
        self.count += 1
        self.last_summary = summary

    def report(self):
        now = time.time()
        dt = now - self.last_print_time
        if dt <= 0:
            return None
        hz = self.count / dt
        self.count = 0
        self.last_print_time = now
        return hz, self.last_summary


class ObsMonitorNode(Node):
    def __init__(self):
        super().__init__("obs_monitor_node")

        workspace_root = os.environ.get("INSPIRE_WORKSPACE_ROOT", "/workspace/zhicheng_ws")
        debug_root = os.environ.get("INSPIRE_DEBUG_ROOT", workspace_root)
        self.declare_parameter("print_period_sec", 2.0)
        self.save_images = bool(self.declare_parameter("save_images", False).value)
        self.save_period_sec = float(
            self.declare_parameter("save_period_sec", 2.0).value
        )
        self.save_dir = self.declare_parameter(
            "save_dir",
            os.path.join(
                debug_root,
                "debug_deploy_obs",
                f"camera_check_{time.strftime('%Y%m%d_%H%M%S')}",
            ),
        ).value
        self.last_image_save_time = {}
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)

        self.topics = {
            "camera0_left": "/camera_wrist_left/color/image_raw/compressed",
            "camera1_right": "/camera_wrist_right/color/image_raw/compressed",
            "camera2_kinect": "/ak/rgb/image_raw/compressed",

            "robot0_left_pose": "/frankaLeft/ee_pose_matrix",
            "robot1_right_pose": "/frankaRight/ee_pose_matrix",

            "robot0_left_angle": "/left/angle_data",
            "robot1_right_angle": "/right/angle_data",

            "robot0_left_touch": "/left/touch_data",
            "robot1_right_touch": "/right/touch_data",
        }

        self.stats = {k: TopicStats() for k in self.topics.keys()}

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )

        # cameras
        self.create_subscription(
            CompressedImage,
            self.topics["camera0_left"],
            lambda msg: self.image_cb(msg, "camera0_left"),
            10,
        )
        self.create_subscription(
            CompressedImage,
            self.topics["camera1_right"],
            lambda msg: self.image_cb(msg, "camera1_right"),
            10,
        )
        self.create_subscription(
            CompressedImage,
            self.topics["camera2_kinect"],
            lambda msg: self.image_cb(msg, "camera2_kinect"),
            10,
        )

        # poses
        self.create_subscription(
            DesiredPose,
            self.topics["robot0_left_pose"],
            lambda msg: self.pose_cb(msg, "robot0_left_pose"),
            best_effort_qos,
        )
        self.create_subscription(
            DesiredPose,
            self.topics["robot1_right_pose"],
            lambda msg: self.pose_cb(msg, "robot1_right_pose"),
            best_effort_qos,
        )

        # hand angles
        self.create_subscription(
            GetAngleAct1,
            self.topics["robot0_left_angle"],
            lambda msg: self.angle_cb(msg, "robot0_left_angle"),
            10,
        )
        self.create_subscription(
            GetAngleAct1,
            self.topics["robot1_right_angle"],
            lambda msg: self.angle_cb(msg, "robot1_right_angle"),
            10,
        )

        # touch
        self.create_subscription(
            GetTouchAct1,
            self.topics["robot0_left_touch"],
            lambda msg: self.touch_cb(msg, "robot0_left_touch"),
            10,
        )
        self.create_subscription(
            GetTouchAct1,
            self.topics["robot1_right_touch"],
            lambda msg: self.touch_cb(msg, "robot1_right_touch"),
            10,
        )

        period = float(self.get_parameter("print_period_sec").value)
        self.timer = self.create_timer(period, self.print_stats)

        self.get_logger().info("obs_monitor_node started.")
        for name, topic in self.topics.items():
            self.get_logger().info(f"{name}: {topic}")

    def image_cb(self, msg: CompressedImage, name: str):
        size_kb = len(msg.data) / 1024.0

        summary = f"CompressedImage format={msg.format}, size={size_kb:.1f}KB"
        if self.save_images:
            now = time.time()
            last = self.last_image_save_time.get(name, 0.0)
            if now - last >= self.save_period_sec:
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    path = os.path.join(self.save_dir, f"{name}.png")
                    cv2.imwrite(path, bgr)
                    summary += f", saved={path}, shape={bgr.shape}"
                else:
                    summary += ", decode_failed"
                self.last_image_save_time[name] = now
        self.stats[name].update(summary)

    def pose_cb(self, msg: DesiredPose, name: str):
        arr = np.asarray(msg.pose_matrix, dtype=np.float32)
        summary = f"DesiredPose pose_matrix shape={arr.shape}, first4={arr[:4]}"
        self.stats[name].update(summary)

    def angle_cb(self, msg: GetAngleAct1, name: str):
        angles = np.asarray(msg.angles, dtype=np.int32)
        finger_ids = np.asarray(msg.finger_ids, dtype=np.int32)
        summary = (
            f"GetAngleAct1 angles shape={angles.shape}, "
            f"angles={angles.tolist()}, finger_ids={finger_ids.tolist()}"
        )
        self.stats[name].update(summary)

    def touch_cb(self, msg: GetTouchAct1, name: str):
        touch = np.asarray(msg.touch_values, dtype=np.int32)
        finger_ids = np.asarray(msg.finger_ids, dtype=np.int32)
        summary = (
            f"GetTouchAct1 touch shape={touch.shape}, "
            f"min={touch.min() if touch.size else 'NA'}, "
            f"max={touch.max() if touch.size else 'NA'}, "
            f"finger_ids={finger_ids.tolist()}"
        )
        self.stats[name].update(summary)

    def print_stats(self):
        self.get_logger().info("========== OBS TOPIC STATUS ==========")
        for name, stat in self.stats.items():
            result = stat.report()
            if result is None:
                continue
            hz, summary = result
            self.get_logger().info(f"{name:20s} | {hz:6.2f} Hz | {summary}")


def main(args=None):
    rclpy.init(args=args)
    node = ObsMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
