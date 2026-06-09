#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from inspire_interfaces.msg import HandKeypoint

import numpy as np
from pathlib import Path
import os
from typing import List, Optional, Tuple

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from inspire_retargeting.single_hand_detector import SingleHandDetector


class DualRetargetSubscriber(Node):
    def __init__(self):
        super().__init__("dual_retarget_subscriber")

        # --------------------------
        # Publishers (分开发布，避免互相覆盖)
        # --------------------------
        self.pub_left = self.create_publisher(JointState, "/left_hand/joint_states", 10)
        self.pub_right = self.create_publisher(JointState, "/right_hand/joint_states", 10)

        # --------------------------
        # SingleHandDetector: 左右各一套
        # --------------------------
        self.det_left = SingleHandDetector(hand_type="Left", selfie=False)
        self.det_right = SingleHandDetector(hand_type="Right", selfie=False)

        # --------------------------
        # Retargeting: 左右各一套（通常配置不同）
        # --------------------------
        robot_name = RobotName.inspire
        retargeting_type = RetargetingType.dexpilot

        self.declare_parameter("dex_retargeting_robot_dir", "")
        robot_dir_param = self.get_parameter("dex_retargeting_robot_dir").value
        robot_dir = Path(
            robot_dir_param
            or os.environ.get("DEX_RETARGETING_ROBOT_DIR", "")
            or "~/dex-retargeting/assets/robots/hands"
        ).expanduser()
        if not robot_dir.exists():
            self.get_logger().error(f"URDF dir {robot_dir} not exists. Set dex_retargeting_robot_dir or DEX_RETARGETING_ROBOT_DIR.")
            raise FileNotFoundError(robot_dir)
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        self.retarget_left, self.left_joint_names = self._build_retarget(robot_name, retargeting_type, HandType.left)
        self.retarget_right, self.right_joint_names = self._build_retarget(robot_name, retargeting_type, HandType.right)

        # --------------------------
        # Subscriptions
        # --------------------------
        self.sub_left = self.create_subscription(
            HandKeypoint, "/Lefthandpoint", self.cb_left, 10
        )
        self.sub_right = self.create_subscription(
            HandKeypoint, "/Righthandpoint", self.cb_right, 10
        )

        self.get_logger().info("DualRetargetSubscriber ready: /Lefthandpoint & /Righthandpoint -> /left_hand/joint_states & /right_hand/joint_states")

        # Optional scaling hook; currently pass-through.
        self.scaler = 1.0
        self.scaler_thumb = 1.0
        self.thumb_start = 8

    def _build_retarget(
        self,
        robot_name: RobotName,
        retargeting_type: RetargetingType,
        hand_type: HandType
    ) -> Tuple[SeqRetargeting, List[str]]:
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
        retarget: SeqRetargeting = RetargetingConfig.load_from_file(config_path).build()
        joint_names: List[str] = retarget.optimizer.robot.dof_joint_names
        self.get_logger().info(f"Loaded retargeting config: hand_type={hand_type}, dof={len(joint_names)}")
        return retarget, joint_names

    def cb_left(self, msg: HandKeypoint):
        self._process_one(msg, hand="left")

    def cb_right(self, msg: HandKeypoint):
        self._process_one(msg, hand="right")

    def _process_one(self, msg: HandKeypoint, hand: str):
        if not msg.points or len(msg.points) != 21:
            self.get_logger().warn(f"[{hand}] HandKeypoint 点数不正确，跳过。")
            return

        frame_pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float64)

        if hand == "left":
            detector = self.det_left
            retarget = self.retarget_left
            joint_names = self.left_joint_names
            pub = self.pub_left
        else:
            detector = self.det_right
            retarget = self.retarget_right
            joint_names = self.right_joint_names
            pub = self.pub_right

        # SingleHandDetector：把 wrist 置原点 + 估计 wrist frame + 转 MANO
        num_box, joint_pos, _, wrist_rot = detector.detect_from_array(frame_pts)
        if num_box == 0 or joint_pos is None:
            self.get_logger().warn(f"[{hand}] SingleHandDetector 无有效手，跳过。")
            return

        # 构造 ref_value
        opt = retarget.optimizer
        indices = opt.target_link_human_indices

        if opt.retargeting_type == "POSITION":
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

        # retarget
        qpos = retarget.retarget(ref_value)

        # Optional scaling hook; currently pass-through.
        qpos_scaled = qpos.copy()
        # qpos_scaled[:self.thumb_start] *= self.scaler
        # qpos_scaled[self.thumb_start:] *= self.scaler_thumb

        # Optional joint-limit clamp.
        # joint_limits = opt.robot.joint_limits
        # qpos_scaled = np.clip(qpos_scaled, joint_limits[:, 0], joint_limits[:, 1])

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = joint_names
        out.position = qpos_scaled.tolist()
        pub.publish(out)


def main():
    rclpy.init()
    node = DualRetargetSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
