#!/usr/bin/env python3
import socket
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from inspire_interfaces.msg import HandKeypoint


class DualHandUDPReceiver(Node):
    def __init__(self):
        super().__init__('dual_hand_udp_receiver')
        self.declare_parameter("udp_ip", "0.0.0.0")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("filter_alpha", 0.2)
        self.declare_parameter("scale_m", 0.001)

        self.pub_left = self.create_publisher(HandKeypoint, '/Lefthandpoint', 10)
        self.pub_right = self.create_publisher(HandKeypoint, '/Righthandpoint', 10)
        self.get_logger().info("DualHand UDP Receiver started. Publishing to /Lefthandpoint and /Righthandpoint")

        self.UDP_IP = self.get_parameter("udp_ip").get_parameter_value().string_value
        self.UDP_PORT = self.get_parameter("udp_port").get_parameter_value().integer_value
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.sock.settimeout(0.0001)
        self.get_logger().info(f"UDP listening on {self.UDP_IP}:{self.UDP_PORT}")

        self.timer = self.create_timer(0.001, self.timer_callback)

        self.alpha = self.get_parameter("filter_alpha").get_parameter_value().double_value
        self.scale_m = self.get_parameter("scale_m").get_parameter_value().double_value

        # 只保留这些点：0/5/9/13/17 (掌基点) + 4/8/12/16/20 (指尖)
        self.keep_indices = {0, 5, 9, 13, 17, 4, 8, 12, 16, 20}

        # 每只手各自的滤波状态与启动标志
        self.filtered = {
            "left": {i: [0.0, 0.0, 0.0] for i in range(21)},
            "right": {i: [0.0, 0.0, 0.0] for i in range(21)},
        }
        self.started = {"left": False, "right": False}

    def timer_callback(self):
        try:
            data, addr = self.sock.recvfrom(65536)
        except socket.timeout:
            return

        try:
            payload = json.loads(data.decode('utf-8'))
        except Exception as e:
            self.get_logger().warn(f"JSON decode error: {e}")
            return

        # 新结构：payload["hands"]["left"] / payload["hands"]["right"]
        hands = payload.get("hands", {})
        left_dict = hands.get("left", None)
        right_dict = hands.get("right", None)

        # 分别处理（某只手缺失就跳过那只）
        if isinstance(left_dict, dict):
            self._process_one_hand("left", left_dict, self.pub_left)

        if isinstance(right_dict, dict):
            self._process_one_hand("right", right_dict, self.pub_right)

    def _process_one_hand(self, hand_name: str, msg_dict: dict, publisher):
        """
        hand_name: "left" or "right"
        msg_dict: {"0":[x,y,z], ... "20":[x,y,z]} (mm)
        publisher: ROS publisher for HandKeypoint
        """

        # 第一次启动：等待该手至少有一个 keep_indices 不是全0
        if not self.started[hand_name]:
            any_valid = False
            for idx in self.keep_indices:
                key = str(idx)
                if key in msg_dict:
                    x, y, z = msg_dict[key]
                    if not (x == 0.0 and y == 0.0 and z == 0.0):
                        any_valid = True
                        break
            if not any_valid:
                return

            # 初始化滤波状态为当前帧（避免从0慢慢爬）
            for idx in self.keep_indices:
                key = str(idx)
                if key in msg_dict:
                    x, y, z = msg_dict[key]
                    self.filtered[hand_name][idx] = [float(x), float(y), float(z)]
            self.started[hand_name] = True

        # 正常发布
        msg = self._build_hand_keypoint(hand_name, msg_dict)
        publisher.publish(msg)

    def _build_hand_keypoint(self, hand_name: str, msg_dict: dict) -> HandKeypoint:
        points_list = []

        for idx in range(21):
            out_x, out_y, out_z = 0.0, 0.0, 0.0

            if idx in self.keep_indices:
                key = str(idx)
                fx_prev, fy_prev, fz_prev = self.filtered[hand_name][idx]

                valid_raw = False
                if key in msg_dict:
                    x, y, z = msg_dict[key]
                    # Treat [0, 0, 0] as an invalid/missing marker.
                    if not (x == 0.0 and y == 0.0 and z == 0.0):
                        valid_raw = True

                if valid_raw:
                    # EMA（mm）
                    fx = self.alpha * float(x) + (1.0 - self.alpha) * fx_prev
                    fy = self.alpha * float(y) + (1.0 - self.alpha) * fy_prev
                    fz = self.alpha * float(z) + (1.0 - self.alpha) * fz_prev
                    self.filtered[hand_name][idx] = [fx, fy, fz]
                    out_x, out_y, out_z = fx, fy, fz
                else:
                    # 丢点：沿用上一帧滤波值（mm）
                    out_x, out_y, out_z = fx_prev, fy_prev, fz_prev

            # 输出转米
            points_list.append(Point(
                x=float(out_x) * self.scale_m,
                y=float(out_y) * self.scale_m,
                z=float(out_z) * self.scale_m
            ))

        msg = HandKeypoint()
        msg.points = points_list
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = DualHandUDPReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
