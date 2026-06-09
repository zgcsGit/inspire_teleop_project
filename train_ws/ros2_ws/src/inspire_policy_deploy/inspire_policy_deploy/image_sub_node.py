#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import time


class ImageSubNode(Node):
    def __init__(self):
        super().__init__('image_sub_node')

        self.last_time = time.time()
        self.count = 0

        self.sub = self.create_subscription(
            CompressedImage,
            '/ak/rgb/image_raw/compressed',
            self.callback,
            10
        )

        self.get_logger().info("Listening /ak/rgb/image_raw/compressed")

    def callback(self, msg):
        self.count += 1

        now = time.time()
        dt = now - self.last_time

        if dt >= 1.0:
            hz = self.count / dt
            size_kb = len(msg.data) / 1024.0

            self.get_logger().info(
                f"Hz={hz:.1f}, size={size_kb:.1f} KB, format={msg.format}"
            )

            self.count = 0
            self.last_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()