# finger_map/finger_mapper_node.py
import time
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from inspire_interfaces.msg import SetAngle1

from finger_map.finger_model import map_finger, map_thumb_pitch, map_thumb_yaw


def _clip_0_1000_int(x: float) -> int:
    return int(max(0, min(1000, int(round(float(x))))))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class FingerMapperNode(Node):
    def __init__(self):
        super().__init__("finger_mapper_node")

        self.declare_parameter("input_topic", "/left_hand/joint_states")
        self.declare_parameter("output_topic", "/set_angle_data")
        self.declare_parameter("freeze_topic", "/teleop/left_hand_freeze")
        self.declare_parameter("is_frozen_topic", "/left/hand_is_frozen")
        self.declare_parameter("debug", False)

        # smooth unfreeze
        self.declare_parameter("ramp_duration", 1.0)         # seconds
        self.declare_parameter("ramp_min_step", 0.0)         # optional minimum ramp time
        self.declare_parameter("publish_on_freeze_edge", True)

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        freeze_topic = self.get_parameter("freeze_topic").value
        is_frozen_topic = self.get_parameter("is_frozen_topic").value

        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.sub = self.create_subscription(
            JointState, input_topic, self.joint_callback, 10
        )
        self.freeze_sub = self.create_subscription(
            Bool, freeze_topic, self.freeze_callback, 10
        )
        self.pub = self.create_publisher(SetAngle1, output_topic, 10)
        self.is_frozen_pub = self.create_publisher(Bool, is_frozen_topic, state_qos)

        self.debug = bool(self.get_parameter("debug").value)
        self.ramp_duration = float(self.get_parameter("ramp_duration").value)
        self.ramp_min_step = float(self.get_parameter("ramp_min_step").value)
        self.publish_on_edge = bool(self.get_parameter("publish_on_freeze_edge").value)

        # State machine: TELE -> FROZEN -> RAMP -> TELE
        self.mode = "TELE"  # "TELE" | "FROZEN" | "RAMP"

        # freeze request from pedal / upstream logic
        self.frozen = False

        # frozen hold command
        self.hold_angles: Optional[List[int]] = None

        # ramp state
        self.ramp_t0: float = 0.0
        self.ramp_start_angles: Optional[List[int]] = None
        self.ramp_target_angles: Optional[List[int]] = None

        # latest computed angles from JointState
        self.last_angles: Optional[List[int]] = None

        self._dbg_count = 0

        # publish initial state once (latched by TRANSIENT_LOCAL)
        self._publish_frozen_state()

        self.get_logger().info(
            f"FingerMapper: {input_topic} -> {output_topic}, "
            f"freeze={freeze_topic}, is_frozen={is_frozen_topic}, "
            f"ramp_duration={self.ramp_duration:.3f}s"
        )

    # ---------- publish helpers ----------
    def _publish_angles(self, angles: List[int]):
        out = SetAngle1()
        out.finger_ids = [1, 2, 3, 4, 5, 6]
        out.angles = angles
        self.pub.publish(out)

    def _publish_frozen_state(self):
        msg = Bool()
        msg.data = (self.mode == "FROZEN")
        self.is_frozen_pub.publish(msg)

    def _set_mode(self, new_mode: str):
        if self.mode != new_mode:
            self.mode = new_mode
            self._publish_frozen_state()

    def _compute_angles_from_jointstate(self, msg: JointState) -> List[int]:
        joint_map = dict(zip(msg.name, msg.position))

        index = joint_map.get("index_proximal_joint", 0.0)
        middle = joint_map.get("middle_proximal_joint", 0.0)
        pinky = joint_map.get("pinky_proximal_joint", 0.0)
        ring = joint_map.get("ring_proximal_joint", 0.0)
        thumb_pitch = joint_map.get("thumb_proximal_pitch_joint", 0.0)
        thumb_yaw = joint_map.get("thumb_proximal_yaw_joint", 0.0)

        angles = [
            _clip_0_1000_int(map_finger(pinky)),
            _clip_0_1000_int(map_finger(ring)),
            _clip_0_1000_int(map_finger(middle)),
            _clip_0_1000_int(map_finger(index)),
            _clip_0_1000_int(map_thumb_pitch(thumb_pitch)),
            _clip_0_1000_int(map_thumb_yaw(thumb_yaw)),
        ]

        # if self.debug:
        #     self._dbg_count += 1
        #     if self._dbg_count % 30 == 0:
        #         self.get_logger().info(
        #             f"rad: idx={index:.3f} mid={middle:.3f} ring={ring:.3f} "
        #             f"pky={pinky:.3f} tp={thumb_pitch:.3f} ty={thumb_yaw:.3f} "
        #             f"-> cmd={angles} mode={self.mode}"
        #         )

        return angles

    # ---------- freeze logic ----------
    def freeze_callback(self, msg: Bool):
        new_frozen = bool(msg.data)
        if new_frozen == self.frozen:
            return

        self.frozen = new_frozen
        now = time.monotonic()

        if self.frozen:
            # Enter FROZEN: hold the latest available command
            if self.last_angles is None:
                # No jointstate yet; enter FROZEN and wait for first one
                self._set_mode("FROZEN")
                self.get_logger().info("HAND FREEZE ENABLED (no last_angles yet)")
                return

            self.hold_angles = list(self.last_angles)
            self._set_mode("FROZEN")
            self.get_logger().info("HAND FREEZE ENABLED")

            if self.publish_on_edge:
                self._publish_angles(self.hold_angles)
            return

        # Unfreeze: leave FROZEN immediately; viewer should show not frozen now.
        self.get_logger().info("HAND FREEZE DISABLED -> smoothing (RAMP)")

        if self.hold_angles is None:
            # Nothing to ramp from
            self._set_mode("TELE")
            return

        self.ramp_t0 = now
        self.ramp_start_angles = list(self.hold_angles)
        self.ramp_target_angles = list(self.last_angles) if self.last_angles is not None else None
        self._set_mode("RAMP")

        if self.publish_on_edge and self.ramp_target_angles is not None:
            # publish first ramp sample (t = 0), i.e. start_angles
            self._publish_angles(self.ramp_start_angles)

    def joint_callback(self, msg: JointState):
        # Always compute current target from teleop input
        angles = self._compute_angles_from_jointstate(msg)
        self.last_angles = list(angles)

        # If frozen but hold was missing (startup edge case), capture it now.
        if self.mode == "FROZEN" and self.hold_angles is None:
            self.hold_angles = list(self.last_angles)

        if self.mode == "FROZEN":
            if self.hold_angles is not None:
                self._publish_angles(self.hold_angles)
            return

        if self.mode == "RAMP":
            # update ramp target continuously
            if self.ramp_target_angles is None:
                self.ramp_target_angles = list(self.last_angles)

            now = time.monotonic()
            dt = now - self.ramp_t0

            dur = max(self.ramp_duration, self.ramp_min_step)
            if dur <= 1e-6:
                self._set_mode("TELE")
                self.hold_angles = None
                self.ramp_start_angles = None
                self.ramp_target_angles = None
                self._publish_angles(self.last_angles)
                return

            t = dt / dur
            if t >= 1.0:
                self._set_mode("TELE")
                self.hold_angles = None
                self.ramp_start_angles = None
                self.ramp_target_angles = None
                self._publish_angles(self.last_angles)
                return

            start = self.ramp_start_angles if self.ramp_start_angles is not None else self.last_angles
            target = self.ramp_target_angles if self.ramp_target_angles is not None else self.last_angles

            out_angles = [
                _clip_0_1000_int(_lerp(float(start[i]), float(target[i]), t))
                for i in range(min(len(start), len(target)))
            ]
            self._publish_angles(out_angles)
            return

        # TELE mode: normal passthrough
        self._set_mode("TELE")
        self._publish_angles(self.last_angles)


def main(args=None):
    rclpy.init(args=args)
    node = FingerMapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
