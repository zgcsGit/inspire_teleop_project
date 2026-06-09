#!/usr/bin/env python3
import socket
import json
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from inspire_interfaces.msg import HandKeypoint

# ------------------------------
# Fixed Vicon Marker Order (21 points)
# ------------------------------
Order = [
    "palm1",
    "thumb_0", "thumb_root", "thumb_mid", "thumb_tip",
    "index_0", "index_root", "index_mid", "index_tip",
    "middle_0", "middle_root", "middle_mid", "middle_tip",
    "ring_0", "ring_root", "ring_mid", "ring_tip",
    "pinky_0", "pinky_root", "pinky_mid", "pinky_tip"
]

# 想要读取的 subject 名字（需要和发送端 Target_Subject 对应）
Target_Subject = "lefthand"


class LeftHandUDPReceiver(Node):
    def __init__(self):
        super().__init__('left_hand_udp_receiver')

        # --------------------------
        # ROS Publisher
        # --------------------------
        self.pub = self.create_publisher(HandKeypoint, '/Lefthandpoint', 10)
        self.get_logger().info("LeftHand UDP Receiver started. Publishing to /Lefthandpoint")

        # --------------------------
        # UDP Setup
        # --------------------------
        # 如果你想监听所有网卡，可以写成 "" 或 "0.0.0.0"
        self.UDP_IP = "192.168.10.3"   # 当前绑定 IP
        self.UDP_PORT = 5005           # MUST match sender!

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))
        self.get_logger().info(f"UDP listening on {self.UDP_IP}:{self.UDP_PORT}")

        # --------------------------
        # Timer for polling UDP
        # --------------------------
        self.timer = self.create_timer(0.001, self.timer_callback)

        # --------------------------
        # Store last valid points
        # --------------------------
        self.last_valid = {name: [0.0, 0.0, 0.0] for name in Order}

        # --------------------------
        # Filter state (EMA)
        # --------------------------
        self.alpha = 0.2  # 滤波强度：0~1，越大越“跟得紧”，越小越平滑
        self.filtered = {name: [0.0, 0.0, 0.0] for name in Order}

        # 跳变相关参数
        self.max_step = 2.0   # 每帧允许的最大运动量（mm），比如 0.02 = 2cm
        
        # 只保留这些点：0/5/9/13/17 (掌基点) + 4/8/12/16/20 (指尖)
        self.keep_indices = {0, 5, 9, 13, 17, 4, 8, 12, 16, 20}
        self.keep_names = [Order[i] for i in range(len(Order)) if i in self.keep_indices]


        # --------------------------
        # Control first-time publishing
        # --------------------------
        self.started = False

    def timer_callback(self):
        """
        Try receiving UDP packet.
        If no data, ignore and continue.
        """
        try:
            self.sock.settimeout(0.0001)
            data, addr = self.sock.recvfrom(65536)
        except socket.timeout:
            return

        try:
            msg_dict = json.loads(data.decode('utf-8'))
        except Exception as e:
            self.get_logger().warn(f"JSON decode error: {e}")
            return

        # # === DEBUG: print raw UDP data ===
        # self.get_logger().info(f"UDP raw frame: {msg_dict}")

        # -----------------------------
        # 这里开始：按 subject 取子字典
        # -----------------------------
        if Target_Subject not in msg_dict:
            # 可能这一帧没有这个 subject，直接等下一帧
            self.get_logger().warn_once(
                f"Target subject '{Target_Subject}' not found in UDP frame keys: {list(msg_dict.keys())}"
            )
            return

        subject_dict = msg_dict[Target_Subject]  # {markerName: [x, y, z], ...}
        
        # self.get_logger().info(f"UDP raw frame for subject '{Target_Subject}': {subject_dict}")

        # 第一次接收：检查所有点是否有效
        if not self.started:
            all_valid = True
            for name in self.keep_names:
                if name not in subject_dict:
                    all_valid = False
                    break
                x, y, z = subject_dict[name]
                if x == 0.0 and y == 0.0 and z == 0.0:
                    all_valid = False
                    break


            if not all_valid:
                # self.get_logger().info("Waiting for first valid UDP frame...")
                return
            else:
                self.started = True
                # 初始化滤波状态为首帧，避免从0爬行
                for name in self.keep_names:
                    x, y, z = subject_dict[name]
                    self.filtered[name] = [float(x), float(y), float(z)]


                # self.get_logger().info(
                #     f"First valid UDP frame received for subject '{Target_Subject}'. "
                #     "Start publishing /Lefthandpoint."
                # )

        # 发布 topic（只用这个 subject 的数据）
        self.publish_hand_points(subject_dict)

    def publish_hand_points(self, subject_dict):
        points_list = []
        scale_m = 0.001  # mm -> m

        for idx, name in enumerate(Order):
            # 默认输出 0
            out_x, out_y, out_z = 0.0, 0.0, 0.0

            # 只处理保留点
            if idx in self.keep_indices:
                valid_raw = False

                if name in subject_dict:
                    x, y, z = subject_dict[name]
                    if not (x == 0.0 and y == 0.0 and z == 0.0):
                        valid_raw = True
                else:
                    valid_raw = False

                fx_prev, fy_prev, fz_prev = self.filtered[name]

                if valid_raw:
                    # EMA
                    fx_temp = self.alpha * x + (1.0 - self.alpha) * fx_prev
                    fy_temp = self.alpha * y + (1.0 - self.alpha) * fy_prev
                    fz_temp = self.alpha * z + (1.0 - self.alpha) * fz_prev

                    # 限制每帧最大位移（mm）
                    dx = fx_temp - fx_prev
                    dy = fy_temp - fy_prev
                    dz = fz_temp - fz_prev
                    step = math.sqrt(dx * dx + dy * dy + dz * dz)

                    if step > self.max_step:
                        s = self.max_step / (step + 1e-9)
                        fx = fx_prev + dx * s
                        fy = fy_prev + dy * s
                        fz = fz_prev + dz * s
                    else:
                        fx, fy, fz = fx_temp, fy_temp, fz_temp

                    self.filtered[name] = [fx, fy, fz]
                    out_x, out_y, out_z = fx, fy, fz
                else:
                    # 缺失：沿用上一帧滤波值（mm）
                    out_x, out_y, out_z = fx_prev, fy_prev, fz_prev

            # 发布时统一转成米
            points_list.append(Point(
                x=float(out_x) * scale_m,
                y=float(out_y) * scale_m,
                z=float(out_z) * scale_m
            ))

        msg = HandKeypoint()
        msg.points = points_list
        self.pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = LeftHandUDPReceiver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

