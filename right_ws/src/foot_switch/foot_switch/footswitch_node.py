#!/usr/bin/env python3
import time
import subprocess
import shlex
import os
import signal

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from evdev import InputDevice, ecodes

KEY_START = ecodes.KEY_A   # Pedal1
KEY_STOP = ecodes.KEY_B    # Pedal2
KEY_TOGGLE = ecodes.KEY_C  # Pedal3

class FootSwitchNode(Node):
    def __init__(self):
        super().__init__("footswitch_node")

        self.declare_parameter("device", "/dev/input/footswitch_keyboard")
        self.declare_parameter("start_service", "/start_episode")
        self.declare_parameter("stop_service", "/stop_episode")

        # --- host bringup launch (for pedal C) ---
        self.declare_parameter("host_launch_pkg", "inspire_launch")
        self.declare_parameter("host_launch_file", "bringup_right.launch.py")
        self.declare_parameter("host_launch_args", "")  # e.g. "robot_ip:=12.1.1.6 log_level:=info"

        self.device_path = self.get_parameter("device").get_parameter_value().string_value
        self.start_srv = self.get_parameter("start_service").get_parameter_value().string_value
        self.stop_srv  = self.get_parameter("stop_service").get_parameter_value().string_value

        self.host_launch_pkg = self.get_parameter("host_launch_pkg").value
        self.host_launch_file = self.get_parameter("host_launch_file").value
        self.host_launch_args = self.get_parameter("host_launch_args").value

        self.get_logger().info(f"Using device: {self.device_path}")
        self.get_logger().info(f"Start service: {self.start_srv}")
        self.get_logger().info(f"Stop  service: {self.stop_srv}")
        self.get_logger().info(f"Host bringup: ros2 launch {self.host_launch_pkg} {self.host_launch_file} {self.host_launch_args}".rstrip())

        self.dev = InputDevice(self.device_path)

        # prevent keys A/B/C from going to the system
        try:
            self.dev.grab()
            self.get_logger().info("Device grabbed (exclusive).")
        except Exception as e:
            self.get_logger().warn(f"Could not grab device: {e}")

        # service clients
        self.start_cli = self.create_client(Trigger, self.start_srv)
        self.stop_cli  = self.create_client(Trigger, self.stop_srv)

        # debounce
        self.debounce_s = 0.20
        self.last_press_time = {}

        self.is_recording = False

        # pedal C: host launch process handle
        self.host_proc = None

        self.create_timer(0.01, self.poll_once)

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

    def _host_running(self) -> bool:
        return self.host_proc is not None and self.host_proc.poll() is None

    def start_host_launch(self):
        if self._host_running():
            self.get_logger().info("Host bringup already running.")
            return

        cmd = ["ros2", "launch", self.host_launch_pkg, self.host_launch_file]
        if str(self.host_launch_args).strip():
            cmd += shlex.split(str(self.host_launch_args))

        self.get_logger().info(f"Starting host bringup: {' '.join(cmd)}")

        # start in its own process group so we can stop whole tree
        self.host_proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid
        )

    def stop_host_launch(self):
        if not self._host_running():
            self.get_logger().info("Host bringup not running.")
            self.host_proc = None
            return

        self.get_logger().info("Stopping host bringup...")

        try:
            # send SIGINT to the whole process group (more like Ctrl+C, ROS-friendly)
            os.killpg(os.getpgid(self.host_proc.pid), signal.SIGINT)
        except Exception as e:
            self.get_logger().warn(f"Failed to send SIGINT: {e}. Trying terminate().")
            self.host_proc.terminate()

        try:
            self.host_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.get_logger().warn("Host bringup did not exit, killing...")
            try:
                os.killpg(os.getpgid(self.host_proc.pid), signal.SIGKILL)
            except Exception:
                self.host_proc.kill()

        self.host_proc = None

    def on_key_pressed(self, code: int):
        now = time.monotonic()
        last = self.last_press_time.get(code, 0.0)
        if (now - last) < self.debounce_s:
            return
        self.last_press_time[code] = now

        if code == KEY_START:
            if self.is_recording:
                self.get_logger().info("Pedal1 -> already recording (ignored)")
                return
            self.get_logger().info("Pedal1 (KEY_A) -> call start service")
            self.call_trigger(self.start_cli, self.start_srv)
            self.is_recording = True

        elif code == KEY_STOP:
            if not self.is_recording:
                self.get_logger().info("Pedal2 -> not recording yet (ignored)")
                return
            self.get_logger().info("Pedal2 (KEY_B) -> call stop service")
            self.call_trigger(self.stop_cli, self.stop_srv)
            self.is_recording = False

        elif code == KEY_TOGGLE:
            if self._host_running():
                self.get_logger().info("Pedal3 (KEY_C) -> STOP host bringup")
                self.stop_host_launch()
            else:
                self.get_logger().info("Pedal3 (KEY_C) -> START host bringup")
                self.start_host_launch()

    def poll_once(self):
        try:
            for event in self.dev.read():
                if event.type != ecodes.EV_KEY:
                    continue
                if event.value != 1:  # only press; ignore release(0) and repeat(2)
                    continue
                self.on_key_pressed(event.code)
        except BlockingIOError:
            pass

def main():
    rclpy.init()
    node = FootSwitchNode()
    try:
        rclpy.spin(node)
    finally:
        # make sure we don't leave host bringup running
        try:
            node.stop_host_launch()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
