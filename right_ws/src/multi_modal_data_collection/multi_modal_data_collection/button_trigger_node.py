#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import serial
import serial.tools.list_ports as lp
import time
import threading


class ButtonTriggerNode(Node):
    def __init__(self):
        super().__init__('button_trigger_node')
        self.get_logger().info("🔘 ButtonTriggerNode started.")

        # --- Serial config ---
        self.baudrate = 115200
        self.description = "FT232R USB UART"
        self.serial_port = None
        self.ser = None
        self.last_state = None
        self.is_recording = False
        self.last_change_time = time.time()
        self.debounce_sec = 0.15  # prevent bouncing

        # --- Open serial (non-blocking mode) ---
        self.open_serial()

        # --- ROS2 service clients ---
        self.start_client = self.create_client(Trigger, '/start_episode')
        self.stop_client = self.create_client(Trigger, '/stop_episode')
        while not self.start_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/start_episode not available, waiting...')
        while not self.stop_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/stop_episode not available, waiting...')
        self.get_logger().info("✅ Connected to /start_episode and /stop_episode services.")

        # --- Background thread for serial reading ---
        self.running = True
        self.serial_thread = threading.Thread(target=self.serial_loop, daemon=True)
        self.serial_thread.start()

    # ------------------------------------------------------
    # Connect to Arduino serial port
    # ------------------------------------------------------
    def open_serial(self):
        try_paths = [
            "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5917006280-if00",
            "/dev/ttyACM0", "/dev/ttyUSB0",
        ]
        for p in lp.comports():
            if self.description in (p.description or ""):
                try_paths.append(p.device)
        seen = set()
        try_paths = [p for p in try_paths if not (p in seen or seen.add(p))]

        for path in try_paths:
            try:
                self.ser = serial.Serial(path, self.baudrate, timeout=0)  # 非阻塞
                self.serial_port = path
                self.get_logger().info(f"✅ Connected to Arduino at {path}")
                time.sleep(1)
                self.ser.reset_input_buffer()
                return
            except Exception:
                continue
        raise RuntimeError("❌ No Arduino serial port found.")

    # ------------------------------------------------------
    # serial reading loop
    # ------------------------------------------------------
    def serial_loop(self):
        buffer = b""
        while self.running:
            try:
                # Read available data
                chunk = self.ser.read(64)
                if not chunk:
                    time.sleep(0.01)
                    continue
                buffer += chunk
                if b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    self.process_line(line.decode(errors="ignore").strip())
            except Exception as e:
                self.get_logger().warn(f"Serial read error: {e}")
                time.sleep(0.1)

    # ------------------------------------------------------
    # Process a line from serial
    # ------------------------------------------------------
    def process_line(self, line: str):
        parts = line.split(",")
        if len(parts) != 2:
            return
        try:
            _, button_state = parts
            button_state = int(button_state)
        except ValueError:
            return

        if self.last_state is None:
            self.last_state = button_state
            return

        now = time.time()
        if now - self.last_change_time < self.debounce_sec:
            return

        # Detect transitions
        if self.last_state == 0 and button_state == 1 and not self.is_recording:
            self.get_logger().info("🎬 Button pressed → Start recording")
            self.call_service_async(self.start_client, "start")
            self.is_recording = True
            self.last_change_time = now

        elif self.last_state == 1 and button_state == 0 and self.is_recording:
            self.get_logger().info("🛑 Button released → Stop recording")
            self.call_service_async(self.stop_client, "stop")
            self.is_recording = False
            self.last_change_time = now

        self.last_state = button_state

    # ------------------------------------------------------
    # Call service asynchronously
    # ------------------------------------------------------
    def call_service_async(self, client, action):
        req = Trigger.Request()
        future = client.call_async(req)

        def done_callback(fut):
            if fut.result() is not None:
                res = fut.result()
                msg = res.message if res.message else "(no message)"
                self.get_logger().info(f"✅ {action.upper()} done: {msg}")
            else:
                self.get_logger().error(f"❌ {action.upper()} failed")

        future.add_done_callback(done_callback)

    # ------------------------------------------------------
    # Cleanup on shutdown
    # ------------------------------------------------------
    def destroy_node(self):
        self.running = False
        time.sleep(0.1)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.get_logger().info("Serial closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ButtonTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
