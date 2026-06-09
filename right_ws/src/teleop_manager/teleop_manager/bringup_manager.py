#!/usr/bin/env python3
import os
import signal
import subprocess
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class BringupManager(Node):
    """
    Subscribe to a Bool topic:
      - True  => start bringup launch (if not running)
      - False => stop bringup launch (if running)

    This node starts: ros2 launch <launch_pkg> <launch_file> ...
    and stops it by sending SIGINT to the whole process group.
    """

    def __init__(self):
        super().__init__('bringup_manager')

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter('enable_topic', '/teleop/bringup_enable')

        self.declare_parameter('launch_pkg', 'inspire_launch')
        self.declare_parameter('launch_file', 'bringup_right.launch.py')
        # launch_args as a string, e.g. "robot_ip:=12.1.1.6 log_level:=info"
        self.declare_parameter('launch_args', '')

        # extra: ignore repeated same command
        self.declare_parameter('ignore_same_state', True)

        self.enable_topic = self.get_parameter('enable_topic').get_parameter_value().string_value
        self.launch_pkg = self.get_parameter('launch_pkg').get_parameter_value().string_value
        self.launch_file = self.get_parameter('launch_file').get_parameter_value().string_value
        self.launch_args_str = self.get_parameter('launch_args').get_parameter_value().string_value
        self.ignore_same_state = self.get_parameter('ignore_same_state').get_parameter_value().bool_value

        # -------------------------
        # State
        # -------------------------
        self._proc: Optional[subprocess.Popen] = None
        self._desired_enabled: Optional[bool] = None  # remember last command

        # -------------------------
        # Sub
        # -------------------------
        self.sub = self.create_subscription(Bool, self.enable_topic, self._on_enable, 10)

        self.get_logger().info(
            f"BringupManager listening on {self.enable_topic}, "
            f"launch: ros2 launch {self.launch_pkg} {self.launch_file} {self.launch_args_str}"
        )

    # -------------------------
    # Helpers
    # -------------------------
    def _build_launch_cmd(self) -> List[str]:
        cmd = ['ros2', 'launch', self.launch_pkg, self.launch_file]
        if self.launch_args_str.strip():
            # split by whitespace: "a:=1 b:=2"
            cmd += self.launch_args_str.strip().split()
        return cmd

    def _is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _start(self):
        if self._is_running():
            self.get_logger().info("Bringup already running. Ignore start.")
            return

        cmd = self._build_launch_cmd()
        self.get_logger().info("Starting bringup: " + " ".join(cmd))

        # Start as a new process group so we can SIGINT the whole tree
        self._proc = subprocess.Popen(
            cmd,
            stdout=None,  # inherit terminal output
            stderr=None,
            preexec_fn=os.setsid  # new process group
        )

        self.get_logger().info(f"Bringup started (pid={self._proc.pid}).")

    def _stop(self):
        if not self._is_running():
            self.get_logger().info("Bringup not running. Ignore stop.")
            self._proc = None
            return

        pid = self._proc.pid
        self.get_logger().info(f"Stopping bringup (pid={pid}) with SIGINT...")

        try:
            os.killpg(os.getpgid(pid), signal.SIGINT)
        except ProcessLookupError:
            self.get_logger().warn("Process already exited.")
            self._proc = None
            return

        # wait a bit for clean shutdown
        try:
            self._proc.wait(timeout=6.0)
            self.get_logger().info("Bringup stopped cleanly.")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("SIGINT timeout, sending SIGTERM...")
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                self._proc = None
                return

            try:
                self._proc.wait(timeout=3.0)
                self.get_logger().info("Bringup stopped with SIGTERM.")
            except subprocess.TimeoutExpired:
                self.get_logger().error("SIGTERM timeout, sending SIGKILL!")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self._proc = None

    # -------------------------
    # Callback
    # -------------------------
    def _on_enable(self, msg: Bool):
        enabled = bool(msg.data)

        if self.ignore_same_state and self._desired_enabled is not None and enabled == self._desired_enabled:
            # same command repeated
            self.get_logger().debug(f"Repeated enable={enabled}, ignored.")
            return

        self._desired_enabled = enabled
        if enabled:
            self._start()
        else:
            self._stop()

    def destroy_node(self):
        # Ensure bringup is stopped when this manager node exits
        try:
            self._stop()
        except Exception as e:
            self.get_logger().error(f"Exception while stopping bringup on shutdown: {e}")
        super().destroy_node()


def main():
    rclpy.init()
    node = BringupManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
