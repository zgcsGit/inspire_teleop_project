#!/usr/bin/env python3
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Bool
from evdev import InputDevice, ecodes

KEY_START = ecodes.KEY_A    # Pedal1
KEY_STOP = ecodes.KEY_B     # Pedal2
KEY_TOGGLE = ecodes.KEY_C   # Pedal3


class FootSwitchNode(Node):
    def __init__(self):
        super().__init__("footswitch_node")

        self.declare_parameter("device", "/dev/input/footswitch_keyboard")
        self.declare_parameter("start_service", "/start_episode")
        self.declare_parameter("stop_service", "/stop_episode")

        self.declare_parameter("bringup_topic", "/teleop/bringup_enable")
        self.declare_parameter("bringup_initial", False)

        self.declare_parameter("left_freeze_topic", "/teleop/left_hand_freeze")
        self.declare_parameter("right_freeze_topic", "/teleop/right_hand_freeze")
        self.declare_parameter("hold_threshold", 0.55)
        self.declare_parameter("debounce_s", 0.12)

        self.device_path = self.get_parameter("device").get_parameter_value().string_value
        self.start_srv = self.get_parameter("start_service").get_parameter_value().string_value
        self.stop_srv = self.get_parameter("stop_service").get_parameter_value().string_value

        self.bringup_topic = self.get_parameter("bringup_topic").get_parameter_value().string_value
        self.bringup_enabled = bool(self.get_parameter("bringup_initial").value)

        self.left_freeze_topic = self.get_parameter("left_freeze_topic").get_parameter_value().string_value
        self.right_freeze_topic = self.get_parameter("right_freeze_topic").get_parameter_value().string_value

        self.hold_threshold = float(self.get_parameter("hold_threshold").value)
        self.debounce_s = float(self.get_parameter("debounce_s").value)

        self.get_logger().info(f"Using device: {self.device_path}")
        self.get_logger().info(f"Start service: {self.start_srv}")
        self.get_logger().info(f"Stop  service: {self.stop_srv}")
        self.get_logger().info(
            f"Bringup topic: {self.bringup_topic} (initial={self.bringup_enabled})"
        )
        self.get_logger().info(f"Left  freeze topic: {self.left_freeze_topic}")
        self.get_logger().info(f"Right freeze topic: {self.right_freeze_topic}")
        self.get_logger().info(
            f"hold_threshold={self.hold_threshold:.3f}s, "
            f"debounce_s={self.debounce_s:.3f}s"
        )

        self.dev = InputDevice(self.device_path)

        try:
            self.dev.grab()
            self.get_logger().info("Device grabbed (exclusive).")
        except Exception as e:
            self.get_logger().warn(f"Could not grab device: {e}")

        self.start_cli = self.create_client(Trigger, self.start_srv)
        self.stop_cli = self.create_client(Trigger, self.stop_srv)

        self.bringup_pub = self.create_publisher(Bool, self.bringup_topic, 10)
        self.left_freeze_pub = self.create_publisher(Bool, self.left_freeze_topic, 10)
        self.right_freeze_pub = self.create_publisher(Bool, self.right_freeze_topic, 10)

        self.is_recording = False
        self.last_edge_time = {}  # code -> monotonic time

        self.pressed = {
            KEY_START: False,
            KEY_STOP: False,
        }
        self.press_time = {
            KEY_START: 0.0,
            KEY_STOP: 0.0,
        }
        self.long_active = {
            KEY_START: False,  # long-hold already triggered?
            KEY_STOP: False,
        }
        self.freeze_latched = {
            KEY_START: False,  # A -> left freeze latched?
            KEY_STOP: False,   # B -> right freeze latched?
        }

        self.create_timer(0.01, self.poll_once)

    def _debounce_edge(self, code: int) -> bool:
        """Return True if this edge is allowed (not debounced)."""
        now = time.monotonic()
        last = self.last_edge_time.get(code, 0.0)
        if (now - last) < self.debounce_s:
            return False
        self.last_edge_time[code] = now
        return True

    def call_trigger(self, client, name: str):
        if not client.service_is_ready():
            self.get_logger().warn(f"Service not ready: {name}")
            return

        fut = client.call_async(Trigger.Request())

        def _done_cb(f):
            if f.exception() is not None:
                self.get_logger().error(f"{name} failed: {f.exception()}")
                return
            res = f.result()
            self.get_logger().info(f"{name} -> success={res.success}, msg='{res.message}'")

        fut.add_done_callback(_done_cb)

    @staticmethod
    def _bool_msg(value: bool) -> Bool:
        msg = Bool()
        msg.data = bool(value)
        return msg

    def publish_bringup(self, enabled: bool):
        self.bringup_pub.publish(self._bool_msg(enabled))

    def publish_freeze(self, left: Optional[bool] = None, right: Optional[bool] = None):
        if left is not None:
            self.left_freeze_pub.publish(self._bool_msg(left))
        if right is not None:
            self.right_freeze_pub.publish(self._bool_msg(right))

    def on_key_press(self, code: int):
        if not self._debounce_edge(code):
            return

        if code == KEY_TOGGLE:
            self.bringup_enabled = not self.bringup_enabled
            self.publish_bringup(self.bringup_enabled)
            return

        if code not in (KEY_START, KEY_STOP):
            return

        self.pressed[code] = True
        self.press_time[code] = time.monotonic()
        self.long_active[code] = False

    def on_key_release(self, code: int):
        if not self._debounce_edge(code):
            return

        if code not in (KEY_START, KEY_STOP):
            return

        was_pressed = self.pressed.get(code, False)
        self.pressed[code] = False

        if not was_pressed:
            return

        if self.long_active.get(code, False):
            self.long_active[code] = False
            return

        if code == KEY_START:
            if self.is_recording:
                return
            self.call_trigger(self.start_cli, self.start_srv)
            self.is_recording = True

        elif code == KEY_STOP:
            if not self.is_recording:
                return
            self.call_trigger(self.stop_cli, self.stop_srv)
            self.is_recording = False

    def check_long_holds(self):
        now = time.monotonic()

        if self.pressed.get(KEY_START, False) and not self.long_active.get(KEY_START, False):
            dt = now - self.press_time.get(KEY_START, 0.0)
            if dt >= self.hold_threshold:
                self.long_active[KEY_START] = True

                self.freeze_latched[KEY_START] = not self.freeze_latched[KEY_START]
                new_state = self.freeze_latched[KEY_START]
                self.publish_freeze(left=new_state)

        if self.pressed.get(KEY_STOP, False) and not self.long_active.get(KEY_STOP, False):
            dt = now - self.press_time.get(KEY_STOP, 0.0)
            if dt >= self.hold_threshold:
                self.long_active[KEY_STOP] = True

                self.freeze_latched[KEY_STOP] = not self.freeze_latched[KEY_STOP]
                new_state = self.freeze_latched[KEY_STOP]
                self.publish_freeze(right=new_state)

    def poll_once(self):
        try:
            for event in self.dev.read():
                if event.type != ecodes.EV_KEY:
                    continue

                # press=1, release=0, repeat=2
                if event.value == 1:
                    self.on_key_press(event.code)
                elif event.value == 0:
                    self.on_key_release(event.code)
                else:
                    pass
        except BlockingIOError:
            pass

        self.check_long_holds()


def main():
    rclpy.init()
    node = FootSwitchNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
