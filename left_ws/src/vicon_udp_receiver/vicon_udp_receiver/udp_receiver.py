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
ORDER = [
    "palm1",
    "thumb_0", "thumb_root", "thumb_mid", "thumb_tip",
    "index_0", "index_root", "index_mid", "index_tip",
    "middle_0", "middle_root", "middle_mid", "middle_tip",
    "ring_0", "ring_root", "ring_mid", "ring_tip",
    "pinky_0", "pinky_root", "pinky_mid", "pinky_tip"
]

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
        self.UDP_IP = "192.168.10.3"
        self.UDP_PORT = 5005
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
        self.last_valid = {name: [0.0, 0.0, 0.0] for name in ORDER}

        # --------------------------
        # Filter state (EMA)
        # --------------------------
        self.alpha = 0.2  # EMA strength
        self.filtered = {name: [0.0, 0.0, 0.0] for name in ORDER}
        self.max_step = 2.0  # max allowed movement per frame (mm)

        # --------------------------
        # Control first-time publishing
        # --------------------------
        self.started = False

    def timer_callback(self):
        """Try receiving UDP packet. If no data, ignore and continue."""
        try:
            self.sock.settimeout(0.0001)
            data, addr = self.sock.recvfrom(65536)
        except socket.timeout:
            return

        # JSON decode
        try:
            msg_dict = json.loads(data.decode('utf-8'))
        except Exception as e:
            self.get_logger().warn(f"JSON decode error: {e}")
            return
        
        # === DEBUG: print raw UDP data ===
        self.get_logger().info(f"UDP raw frame: {msg_dict}")


        # First frame must contain all valid points
        if not self.started:
            all_valid = True
            for name in ORDER:
                if name not in msg_dict:
                    all_valid = False
                    break
                x, y, z = msg_dict[name]
                if x == 0.0 and y == 0.0 and z == 0.0:
                    all_valid = False
                    break

            # if not all_valid:
            #     self.get_logger().info("Waiting for first valid UDP frame...")
            #     return
            # else:
            #     self.started = True
            #     self.get_logger().info("First valid UDP frame received. Start publishing /Lefthandpoint.")

        # Publish filtered points
        self.publish_hand_points(msg_dict)

    def publish_hand_points(self, msg_dict):
        points_list = []

        for name in ORDER:
            valid_raw = False

            # 1) Raw measurement
            if name in msg_dict:
                x, y, z = msg_dict[name]
                if not (x == 0.0 and y == 0.0 and z == 0.0):
                    valid_raw = True

            fx_prev, fy_prev, fz_prev = self.filtered[name]

            if valid_raw:
                # Update last valid
                self.last_valid[name] = [x, y, z]

                # 2) EMA smoothing
                fx_temp = self.alpha * x + (1.0 - self.alpha) * fx_prev
                fy_temp = self.alpha * y + (1.0 - self.alpha) * fy_prev
                fz_temp = self.alpha * z + (1.0 - self.alpha) * fz_prev

                # 3) Limit maximum movement
                dx = fx_temp - fx_prev
                dy = fy_temp - fy_prev
                dz = fz_temp - fz_prev
                step = math.sqrt(dx*dx + dy*dy + dz*dz)

                if step > self.max_step:
                    scale = self.max_step / (step + 1e-9)
                    fx = fx_prev + dx * scale
                    fy = fy_prev + dy * scale
                    fz = fz_prev + dz * scale
                else:
                    fx, fy, fz = fx_temp, fy_temp, fz_temp

                self.filtered[name] = [fx, fy, fz]
                out_x, out_y, out_z = fx, fy, fz

            else:
                # Missing frame → use last filtered value
                out_x, out_y, out_z = fx_prev, fy_prev, fz_prev

            points_list.append(Point(x=float(out_x), y=float(out_y), z=float(out_z)))

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
