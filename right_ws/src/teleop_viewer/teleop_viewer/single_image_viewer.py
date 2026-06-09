#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import site

# 1) 先把用户目录 site-packages 从 sys.path 里移除
#    防止导入 ~/.local/... 里的 pip 版 cv2
try:
    user_site = site.getusersitepackages()
    if user_site and user_site in sys.path:
        sys.path.remove(user_site)
except Exception:
    pass

# 再保险：过滤掉 .local 下面的 Python site-packages
sys.path = [
    p for p in sys.path
    if not ("/.local/lib/python" in p and "site-packages" in p)
]

# 2) 清理 Qt 相关环境变量
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ["QT_QPA_PLATFORM"] = "xcb"

import threading
import numpy as np
import cv2

print("[DEBUG] cv2 from:", cv2.__file__)

from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QFrame,
)

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
from inspire_interfaces.msg import GetTouchAct1
from std_msgs.msg import Bool, String


CHANNEL_COUNTS = {
    1: 93,   # Pinky
    2: 93,   # Ring
    3: 93,   # Middle
    4: 93,   # Index
    5: 105,  # Thumb Flexion
    7: 56,   # Palm
}

FINGER_ORDER = [3, 4, 5]   # Middle, Index, Thumb
FINGER_LABELS = {
    1: "Pinky",
    2: "Ring",
    3: "Middle",
    4: "Index",
    5: "Thumb",
    7: "Palm",
}

TOTAL_EXPECTED = sum(CHANNEL_COUNTS.values())
TOUCH_MAX_VALUE = 4000.0


class ImageBuffer:
    def __init__(self, name: str):
        self.name = name
        self.latest_frame = None
        self.lock = threading.Lock()


class TouchBuffer:
    def __init__(self):
        self.latest_values = {}
        self.lock = threading.Lock()


class TextBuffer:
    def __init__(self, initial_text: str = ""):
        self.text = initial_text
        self.lock = threading.Lock()


class MultiImageSubscriber(Node):
    def __init__(self):
        super().__init__("multi_image_viewer_node")

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        latched_text_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.buffers = {
            "azure_kinect": ImageBuffer("Azure Kinect"),
            "wrist_right": ImageBuffer("Wrist Right"),
            "wrist_left": ImageBuffer("Wrist Left"),
        }

        self.left_touch = TouchBuffer()
        self.right_touch = TouchBuffer()

        self.left_freeze = False
        self.right_freeze = False
        self.left_freeze_lock = threading.Lock()
        self.right_freeze_lock = threading.Lock()

        self.episode_recording = False
        self.episode_recording_lock = threading.Lock()

        self.recorder_status = TextBuffer("")
        self.recorder_event = TextBuffer("")

        self.sub_ak = self.create_subscription(
            CompressedImage,
            "/ak/rgb/image_raw/compressed",
            lambda msg: self.image_callback(msg, "azure_kinect"),
            image_qos,
        )

        self.sub_right = self.create_subscription(
            CompressedImage,
            "/camera_wrist_right/color/image_raw/compressed",
            lambda msg: self.image_callback(msg, "wrist_right"),
            image_qos,
        )

        self.sub_left = self.create_subscription(
            CompressedImage,
            "/camera_wrist_left/color/image_raw/compressed",
            lambda msg: self.image_callback(msg, "wrist_left"),
            image_qos,
        )

        self.sub_left_touch = self.create_subscription(
            GetTouchAct1,
            "/left/touch_data",
            self.left_touch_callback,
            10,
        )

        self.sub_right_touch = self.create_subscription(
            GetTouchAct1,
            "/right/touch_data",
            self.right_touch_callback,
            10,
        )

        latched_bool_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub_left_freeze = self.create_subscription(
            Bool,
            "/left/hand_is_frozen",
            self.left_freeze_callback,
            latched_bool_qos,
        )

        self.sub_right_freeze = self.create_subscription(
            Bool,
            "/right/hand_is_frozen",
            self.right_freeze_callback,
            latched_bool_qos,
        )

        self.sub_episode_recording = self.create_subscription(
            Bool,
            "/episode_recording",
            self.episode_recording_callback,
            latched_text_qos,
        )

        self.sub_recorder_status = self.create_subscription(
            String,
            "/recorder_status_text",
            self.recorder_status_callback,
            latched_text_qos,
        )

        self.sub_recorder_event = self.create_subscription(
            String,
            "/recorder_event_text",
            self.recorder_event_callback,
            latched_text_qos,
        )

        self.get_logger().info(
            "Subscribed to image + touch + freeze + episode_recording + recorder text topics"
        )

    def image_callback(self, msg: CompressedImage, key: str):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        buf = self.buffers[key]
        with buf.lock:
            buf.latest_frame = frame

    def parse_touch_msg(self, msg: GetTouchAct1):
        if len(msg.touch_values) != TOTAL_EXPECTED:
            self.get_logger().warn(
                f"touch_values length mismatch: expected {TOTAL_EXPECTED}, got {len(msg.touch_values)}"
            )

        idx = 0
        parsed = {}

        for fid, name in zip(msg.finger_ids, msg.finger_names):
            n = CHANNEL_COUNTS.get(fid, 0)
            if n <= 0:
                self.get_logger().warn(f"Unknown finger id {fid}, skip")
                continue

            if idx + n > len(msg.touch_values):
                self.get_logger().warn(
                    f"Not enough data for finger {fid}: need {n}, available {len(msg.touch_values) - idx}"
                )
                break

            vals = list(msg.touch_values[idx: idx + n])
            idx += n
            parsed[fid] = (name, vals)

        return parsed

    def left_touch_callback(self, msg: GetTouchAct1):
        parsed = self.parse_touch_msg(msg)
        with self.left_touch.lock:
            self.left_touch.latest_values = parsed

    def right_touch_callback(self, msg: GetTouchAct1):
        parsed = self.parse_touch_msg(msg)
        with self.right_touch.lock:
            self.right_touch.latest_values = parsed

    def left_freeze_callback(self, msg: Bool):
        with self.left_freeze_lock:
            self.left_freeze = bool(msg.data)

    def right_freeze_callback(self, msg: Bool):
        with self.right_freeze_lock:
            self.right_freeze = bool(msg.data)

    def episode_recording_callback(self, msg: Bool):
        with self.episode_recording_lock:
            self.episode_recording = bool(msg.data)

    def recorder_status_callback(self, msg: String):
        with self.recorder_status.lock:
            self.recorder_status.text = msg.data

    def recorder_event_callback(self, msg: String):
        with self.recorder_event.lock:
            self.recorder_event.text = msg.data


class TouchWidget(QWidget):
    def __init__(self, ros_node: MultiImageSubscriber, touch_side: str):
        super().__init__()
        self.ros_node = ros_node
        self.touch_side = touch_side
        self.setMinimumHeight(260)
        self.setStyleSheet("background-color: #111111;")

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        rect = self.rect()
        painter.fillRect(rect, QColor(17, 17, 17))

        touch_buffer = self.ros_node.left_touch if self.touch_side == "left" else self.ros_node.right_touch

        with touch_buffer.lock:
            touch_data = dict(touch_buffer.latest_values)

        margin_left = 80
        margin_right = 10
        margin_top = 10
        margin_bottom = 10
        gap_y = 6

        usable_w = max(10, rect.width() - margin_left - margin_right)
        usable_h = max(10, rect.height() - margin_top - margin_bottom)

        rows = len(FINGER_ORDER)
        row_h = max(10, (usable_h - gap_y * (rows - 1)) / rows)

        label_pen = QPen(QColor(220, 220, 220))
        border_pen = QPen(QColor(90, 90, 90))

        for i, fid in enumerate(FINGER_ORDER):
            y = margin_top + i * (row_h + gap_y)
            band_rect = QRectF(margin_left, y, usable_w, row_h)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(35, 35, 35))
            painter.drawRect(band_rect)

            painter.setPen(label_pen)
            painter.drawText(8, int(y + row_h * 0.65), FINGER_LABELS.get(fid, str(fid)))

            painter.setPen(border_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(band_rect)

            if fid not in touch_data:
                continue

            _, vals = touch_data[fid]
            if not vals:
                continue

            target_bins = 40
            n = len(vals)
            bin_size = max(1, int(np.ceil(n / target_bins)))

            binned = []
            for k in range(0, n, bin_size):
                chunk = vals[k:k + bin_size]
                if not chunk:
                    continue
                binned.append(max(chunk))

            if not binned:
                continue

            nb = len(binned)
            bar_w = usable_w / nb

            for j, v in enumerate(binned):
                vv = max(0.0, min(float(v), TOUCH_MAX_VALUE))
                ratio = vv / TOUCH_MAX_VALUE

                g = int(80 + 175 * ratio)
                color = QColor(30, g, 50)

                x = margin_left + j * bar_w
                h = row_h * ratio
                y0 = y + (row_h - h)

                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawRect(QRectF(x, y0, max(2.0, bar_w - 1.0), h))

        painter.end()


class StatusPanel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #171717;
                border: 1px solid #444444;
                border-radius: 6px;
            }
            QLabel {
                color: white;
            }
        """)
        self.setMinimumHeight(135)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(11)
        self.title_label.setFont(title_font)

        self.text_label = QLabel("")
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        body_font = QFont()
        body_font.setPointSize(10)
        self.text_label.setFont(body_font)

        layout.addWidget(self.title_label)
        layout.addWidget(self.text_label, stretch=1)
        self.setLayout(layout)

    def set_text(self, text: str):
        self.text_label.setText(text)


class ImageDisplayWidget(QWidget):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.current_pixmap = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: black;")

    def setPixmap(self, pixmap: QPixmap):
        self.current_pixmap = pixmap
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self.current_pixmap is not None:
            scaled = self.current_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            x = (self.width() - scaled.width()) / 2
            y = (self.height() - scaled.height()) / 2
            painter.drawPixmap(int(x), int(y), scaled)

        freeze_state = self.parent_window.get_freeze_state()
        if freeze_state is not None:
            freeze_color = QColor(255, 60, 60) if freeze_state else QColor(0, 220, 0)
            painter.setPen(Qt.NoPen)
            painter.setBrush(freeze_color)

            diameter = 42
            margin = 18
            painter.drawEllipse(margin, margin, diameter, diameter)

        if self.parent_window.show_recording_overlay():
            if self.parent_window.recording_blink_on():
                margin_x = 18
                margin_y = 18

                rec_diameter = 26
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(255, 60, 60))
                painter.drawEllipse(margin_x, margin_y, rec_diameter, rec_diameter)

                painter.setPen(QColor(255, 255, 255))
                font = QFont()
                font.setPointSize(16)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(
                    margin_x + rec_diameter + 12,
                    margin_y + 21,
                    "REC"
                )

        painter.end()


class CameraWindow(QMainWindow):
    def __init__(self, ros_node, buffer_key, window_title,
                 with_touch=False, touch_side=None, freeze_side=None,
                 show_recording=False, show_recorder_text=False):
        super().__init__()
        self.ros_node = ros_node
        self.buffer_key = buffer_key
        self.with_touch = with_touch
        self.touch_side = touch_side
        self.freeze_side = freeze_side
        self.show_recording = show_recording
        self.show_recorder_text = show_recorder_text

        self._recording_blink_on = False
        self._prev_recording_state = False
        self._local_event_text = ""

        self.setWindowTitle(window_title)
        if with_touch:
            self.resize(960, 960)
        elif show_recorder_text:
            self.resize(960, 720)
        else:
            self.resize(960, 540)

        central = QWidget()
        self.setCentralWidget(central)

        self.title_label = QLabel(window_title)
        self.title_label.setAlignment(Qt.AlignCenter)

        self.image_widget = ImageDisplayWidget(self)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_widget, stretch=1)

        self.recorder_status_panel = None
        self.recorder_event_panel = None
        if self.show_recorder_text:
            text_row = QWidget()
            text_row_layout = QHBoxLayout()
            text_row_layout.setContentsMargins(0, 0, 0, 0)
            text_row_layout.setSpacing(8)

            self.recorder_status_panel = StatusPanel("Status")
            self.recorder_event_panel = StatusPanel("Event")

            text_row_layout.addWidget(self.recorder_status_panel, stretch=1)
            text_row_layout.addWidget(self.recorder_event_panel, stretch=1)
            text_row.setLayout(text_row_layout)

            layout.addWidget(text_row, stretch=0)

        self.touch_widget = None
        if self.with_touch and self.touch_side is not None:
            self.touch_widget = TouchWidget(ros_node, self.touch_side)
            layout.addWidget(self.touch_widget, stretch=0)

        central.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)

        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.toggle_recording_blink)
        self.blink_timer.start(500)

    def update_view(self):
        buf = self.ros_node.buffers[self.buffer_key]
        with buf.lock:
            frame = None if buf.latest_frame is None else buf.latest_frame.copy()

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.image_widget.setPixmap(pixmap)

        if self.touch_widget is not None:
            self.touch_widget.update()

        if self.show_recorder_text:
            self.update_recorder_text_panels()

        self.image_widget.update()

    def update_recorder_text_panels(self):
        # 左侧 status
        with self.ros_node.recorder_status.lock:
            status_text = self.ros_node.recorder_status.text

        if self.recorder_status_panel is not None:
            self.recorder_status_panel.set_text(status_text if status_text else "Episode:\nSensors:")

        # 右侧 event
        current_recording = self.current_recording_state()
        if (not self._prev_recording_state) and current_recording:
            self._local_event_text = "Started recording"
        elif self._prev_recording_state and (not current_recording):
            self._local_event_text = "Stopped recording"

        self._prev_recording_state = current_recording

        with self.ros_node.recorder_event.lock:
            recorder_event_text = self.ros_node.recorder_event.text

        event_text = recorder_event_text if recorder_event_text else self._local_event_text
        if self.recorder_event_panel is not None:
            self.recorder_event_panel.set_text(event_text)

    def toggle_recording_blink(self):
        self._recording_blink_on = not self._recording_blink_on
        if self.show_recording:
            self.image_widget.update()

    def recording_blink_on(self):
        return self._recording_blink_on

    def current_recording_state(self):
        with self.ros_node.episode_recording_lock:
            return self.ros_node.episode_recording

    def get_freeze_state(self):
        if self.freeze_side == "left":
            with self.ros_node.left_freeze_lock:
                return self.ros_node.left_freeze
        elif self.freeze_side == "right":
            with self.ros_node.right_freeze_lock:
                return self.ros_node.right_freeze
        return None

    def show_recording_overlay(self):
        if not self.show_recording:
            return False
        return self.current_recording_state()


def main(args=None):
    rclpy.init(args=args)

    node = MultiImageSubscriber()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)

    win_ak = CameraWindow(
        node,
        "azure_kinect",
        "Azure Kinect",
        with_touch=False,
        freeze_side=None,
        show_recording=True,
        show_recorder_text=True,
    )

    win_right = CameraWindow(
        node,
        "wrist_right",
        "Wrist Right",
        with_touch=True,
        touch_side="right",
        freeze_side="right",
        show_recording=False,
        show_recorder_text=False,
    )

    win_left = CameraWindow(
        node,
        "wrist_left",
        "Wrist Left",
        with_touch=True,
        touch_side="left",
        freeze_side="left",
        show_recording=False,
        show_recorder_text=False,
    )

    win_ak.show()
    win_right.show()
    win_left.show()

    ret = app.exec_()

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(ret)


if __name__ == "__main__":
    main()