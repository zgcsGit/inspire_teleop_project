#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import os
import rclpy
from rclpy.node import Node
from pymodbus.client import ModbusTcpClient
from pymodbus.pdu import ExceptionResponse
from inspire_interfaces.msg import (
    GetForceAct1,
    GetAngleAct1,
    GetTouchAct1,
    SetAngle1,
    SetForce1,
    SetSpeed1,
)

FORCE_SENSOR_RANGES = {
    1: (1582,),
    2: (1584,),
    3: (1586,),
    4: (1588,),
    5: (1590,),
    6: (1592,),
}

TOUCH_ACT_RANGES = {
    1: (3000, 3369),
    2: (3370, 3739),
    3: (3740, 4109),
    4: (4110, 4479),
    5: (4480, 4899),
    7: (4900, 5123),
}

FORCE_SET_RANGES = {
    1: (1498,),
    2: (1500,),
    3: (1502,),
    4: (1504,),
    5: (1506,),
    6: (1508,),
}

SPEED_SET_RANGES = {
    1: (1522,),
    2: (1524,),
    3: (1526,),
    4: (1528,),
    5: (1530,),
    6: (1532,),
}

ANGLE_ACT_RANGES = {
    1: (1546,),
    2: (1548,),
    3: (1550,),
    4: (1552,),
    5: (1554,),
    6: (1556,),
}

ANGLE_SET_RANGES = {
    1: (1486,),
    2: (1488,),
    3: (1490,),
    4: (1492,),
    5: (1494,),
    6: (1496,),
}

FINGER_NAMES = {
    1: "Pinky",
    2: "Ring Finger",
    3: "Middle Finger",
    4: "Index Finger",
    5: "Thumb Flexion",
    6: "Thumb Abduction",
    7: "Palm",
}


def read_signed_register(client, address):
    response = client.read_holding_registers(address=address, count=1)
    if isinstance(response, ExceptionResponse) or response.isError():
        print(f"[ERROR] read register {address} failed: {response}")
        return 0
    value = response.registers[0]
    if value > 32767:
        value -= 65536
    return value


def read_register_range(client, start_addr, end_addr):
    registers = []
    for addr in range(start_addr, end_addr + 1, 125 * 2):
        current_count = min(125, (end_addr - addr) // 2 + 1)
        response = client.read_holding_registers(address=addr, count=current_count)
        if isinstance(response, ExceptionResponse) or response.isError():
            registers.extend([0] * current_count)
        else:
            registers.extend(response.registers)
    return registers


def read_touch_data(client):
    touch_data = {}
    for finger_id, (start_addr, end_addr) in TOUCH_ACT_RANGES.items():
        reg_values = read_register_range(client, start_addr, end_addr)
        touch_data[finger_id] = [int(reg_values[i]) for i in range(0, len(reg_values), 2)]
        #print(f"[INFO] Finger ID: {finger_id}, Touch Values: {touch_data[finger_id]}")
    return touch_data


def write_signed_register(client, address, value):
    if value < 0:
        value += 65536
    response = client.write_register(address, value)
    if isinstance(response, ExceptionResponse) or response.isError():
        print(f"[ERROR] write register {address} failed: {response}")


class SensorPublisherNode(Node):
    def __init__(self):
        super().__init__("sensor_data_publisher")
        default_ip = os.environ.get("INSPIRE_HAND_MODBUS_IP", "192.168.11.210")
        default_port = int(os.environ.get("INSPIRE_HAND_MODBUS_PORT", "6000"))

        self.declare_parameter("modbus_ip", default_ip)
        self.declare_parameter("modbus_port", default_port)
        self.modbus_ip = self.get_parameter("modbus_ip").value
        self.modbus_port = int(self.get_parameter("modbus_port").value)

        self.force_pub = self.create_publisher(GetForceAct1, "force_data", 10)
        self.angle_pub = self.create_publisher(GetAngleAct1, "angle_data", 10)
        self.touch_pub = self.create_publisher(GetTouchAct1, "touch_data", 10)

        self.create_subscription(SetAngle1, "set_angle_data", self.angle_callback, 10)
        self.create_subscription(SetForce1, "set_force_data", self.force_callback, 10)
        self.create_subscription(SetSpeed1, "set_speed_data", self.speed_callback, 10)

        self.modbus_client = ModbusTcpClient(self.modbus_ip, port=self.modbus_port)
        if not self.modbus_client.connect():
            self.get_logger().error(
                f"Failed to connect to Modbus server {self.modbus_ip}:{self.modbus_port}"
            )
            raise RuntimeError("Modbus connection failed")
        self.get_logger().info(
            f"Modbus connected successfully: {self.modbus_ip}:{self.modbus_port}"
        )

        # start reading thread
        self._stop_thread = False
        self.read_thread = threading.Thread(target=self.data_reading_thread)
        self.read_thread.start()

    def publish_data(self):
        start_time = time.time()

        # Force data
        if self.force_pub.get_subscription_count() > 0:
            force_msg = GetForceAct1()
            force_msg.finger_ids = []
            force_msg.force_values = []
            force_msg.finger_names = []

            for finger_id, (addr,) in FORCE_SENSOR_RANGES.items():
                val = read_signed_register(self.modbus_client, addr)
                force_msg.finger_ids.append(finger_id)
                force_msg.force_values.append(val)
                force_msg.finger_names.append(FINGER_NAMES.get(finger_id, "Unknown Finger"))

            self.force_pub.publish(force_msg)
            #self.get_logger().info(f"Published force data, freq: {1/(time.time()-start_time):.2f} Hz")

        # Angle data
        if self.angle_pub.get_subscription_count() > 0:
            angle_msg = GetAngleAct1()
            angle_msg.finger_ids = []
            angle_msg.angles = []
            angle_msg.finger_names = []

            for finger_id, (addr,) in ANGLE_ACT_RANGES.items():
                val = read_signed_register(self.modbus_client, addr)
                angle_msg.finger_ids.append(finger_id)
                angle_msg.angles.append(val)
                angle_msg.finger_names.append(FINGER_NAMES.get(finger_id, "Unknown Finger"))

            self.angle_pub.publish(angle_msg)
            #self.get_logger().info(f"Published angle data, freq: {1/(time.time()-start_time):.2f} Hz")

        # Touch data
        if self.touch_pub.get_subscription_count() > 0:
            touch_data = read_touch_data(self.modbus_client)
            touch_msg = GetTouchAct1()
            touch_msg.finger_ids = list(touch_data.keys())
            touch_msg.touch_values = []
            touch_msg.finger_names = []

            for fid in touch_msg.finger_ids:
                touch_msg.touch_values.extend(touch_data[fid])
                touch_msg.finger_names.append(FINGER_NAMES.get(fid, "Unknown Finger"))

            self.touch_pub.publish(touch_msg)
            #self.get_logger().info(f"Published touch data, freq: {1/(time.time()-start_time):.2f} Hz")

    # Callbacks for commands
    def angle_callback(self, msg):
        for fid, angle in zip(msg.finger_ids, msg.angles):
            if 0 <= angle <= 1000:
                addr = ANGLE_SET_RANGES.get(fid, (None,))[0]
                if addr is not None:
                    write_signed_register(self.modbus_client, addr, angle)
                else:
                    self.get_logger().warn(f"No address for finger ID {fid}")

    def force_callback(self, msg):
        for fid, force in zip(msg.finger_ids, msg.forces):
            if 0 <= force <= 3000:
                addr = FORCE_SET_RANGES.get(fid, (None,))[0]
                if addr is not None:
                    write_signed_register(self.modbus_client, addr, force)
                else:
                    self.get_logger().warn(f"No address for finger ID {fid}")

    def speed_callback(self, msg):
        for fid, speed in zip(msg.finger_ids, msg.speeds):
            if 0 <= speed <= 1000:
                addr = SPEED_SET_RANGES.get(fid, (None,))[0]
                if addr is not None:
                    write_signed_register(self.modbus_client, addr, speed)
                else:
                    self.get_logger().warn(f"No address for finger ID {fid}")

    def data_reading_thread(self):
        while not self._stop_thread and rclpy.ok():
            self.publish_data()
            time.sleep(0.05)

    def destroy_node(self):
        self._stop_thread = True
        self.read_thread.join()
        self.modbus_client.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
