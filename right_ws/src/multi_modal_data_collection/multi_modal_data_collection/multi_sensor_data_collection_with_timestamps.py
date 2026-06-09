#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Multimodal Data Recorder (Humble) — with per-modality timestamps and dt.
"""

import os
import time
import uuid
import json
import threading
import queue
import zipfile
import numpy as np
import cv2
import collections

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool
from std_msgs.msg import String
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


# ========== Utility functions ==========

def default_data_dir():
    return os.path.expanduser(
        os.environ.get("INSPIRE_TELEOP_DATA_DIR", "~/inspire_teleop_data/right")
    )

def t_now(node):
    """Return current ROS time in seconds."""
    return node.get_clock().now().nanoseconds / 1e9


def list_to_object_array(lst):
    arr = np.empty((len(lst),), dtype=object)
    arr[:] = lst
    return arr


# ========== General RecorderWorker ==========

class RecorderWorker:
    """Subscribe to a topic and buffer parsed data."""
    def __init__(self, node, topic, msg_type, parse_fn, name='worker', maxlen=1000):
        self.node = node
        self.name = name
        self.topic = topic
        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=maxlen)
        self.parse_fn = parse_fn
        self.count = 0
        self.sub = node.create_subscription(msg_type, topic, self._cb, 10)
        self.last_msg_time = None   # ROS time (sec)
        self.first_msg_time = None
        node.get_logger().info(f"[{name}] Subscribed to {topic}")

    def _cb(self, msg):
        try:
            if hasattr(msg, 'header'):
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            else:
                t = t_now(self.node)
            data = self.parse_fn(msg)

            with self.lock:
                self.buf.append((t, data))
                self.count += 1
                self.last_msg_time = t
                if self.first_msg_time is None:
                    self.first_msg_time = t

        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] parse error: {e}")

    def nearest(self, t_star, max_slop):
        with self.lock:
            if not self.buf:
                return None
            best, best_dt = None, 1e9
            for (t, d) in self.buf:
                dt = abs(t - t_star)
                if dt < best_dt:
                    best, best_dt = (t, d), dt
            return best if best_dt <= max_slop else None

    def seconds_since_last_msg(self, now):
        if self.last_msg_time is None:
            return None
        return now - self.last_msg_time


# ========== Dedicated Recorder ==========

def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    """Encode RGB uint8 image to JPEG bytes."""
    assert rgb.dtype == np.uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


class RealSenseRGBRecorder(RecorderWorker):
    """Store compressed JPEG bytes (no decode)."""
    def __init__(self, node, topic='/camera_third_view/color/image_raw/compressed'):
        def parse_fn(msg: CompressedImage):
            return {
                "format": msg.format,
                "data": np.frombuffer(msg.data, dtype=np.uint8)
            }
        super().__init__(node, topic, CompressedImage, parse_fn, name='RealSenseRGB')


class TactileSensorRecorder(RecorderWorker):
    """Tactile sensor (GelSight) → compressed JPEG."""
    def __init__(self, node, topic='/gelsight/image_raw', jpeg_quality=85):
        bridge = CvBridge()

        def parse_fn(msg: Image):
            img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            jpeg_bytes = encode_jpeg(img_rgb, quality=jpeg_quality)
            return {
                "format": "jpeg",
                "data": np.frombuffer(jpeg_bytes, dtype=np.uint8)
            }

        super().__init__(node, topic, Image, parse_fn, name='TactileSensor')


class ViveTrackerRecorder(RecorderWorker):
    """Vive tracker pose recorder."""
    def __init__(self, node, topic='/vive_tracker/pose'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        super().__init__(node, topic, PoseStamped, parse_fn, name='ViveTracker')


class ManusFingertipRecorder(RecorderWorker):
    """Record fingertip poses (5 fingers × 7 values)."""
    def __init__(self, node, topic, name):
        from manus_msg.msg import PoseStampedArray

        def parse_fn(msg: PoseStampedArray):
            arr = []
            for p in msg.poses:
                pos = p.pose.position
                ori = p.pose.orientation
                arr.append([
                    pos.x, pos.y, pos.z,
                    ori.x, ori.y, ori.z, ori.w
                ])
            return np.array(arr, dtype=np.float32)

        super().__init__(node, topic, PoseStampedArray, parse_fn, name=name)


class ManusGloveDataRecorder(RecorderWorker):
    """Record 20D glove flexion sensor values."""
    def __init__(self, node, topic, name):
        from std_msgs.msg import Float64MultiArray

        def parse_fn(msg: Float64MultiArray):
            return np.array(msg.data, dtype=np.float32)

        super().__init__(node, topic, Float64MultiArray, parse_fn, name=name)


class ForceTorqueRecorder(RecorderWorker):
    """Record 6D force–torque vector."""
    def __init__(self, node, topic='/inertial_wrench_compensation/wrench_compensated'):
        from geometry_msgs.msg import WrenchStamped

        def parse_fn(msg: WrenchStamped):
            f = msg.wrench.force
            t = msg.wrench.torque
            return np.array([f.x, f.y, f.z, t.x, t.y, t.z], dtype=np.float32)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.node = node
        self.name = "ForceTorque"
        self.lock = threading.Lock()
        self.topic = topic
        self.buf = collections.deque(maxlen=1000)
        self.parse_fn = parse_fn
        self.count = 0
        self.last_msg_time = None
        self.first_msg_time = None
        self.sub = node.create_subscription(WrenchStamped, topic, self._cb, qos)

        node.get_logger().info(f"[ForceTorque] Subscribed to {topic} with BEST_EFFORT QoS")


class ViconTcpRecorder(RecorderWorker):
    """Record Vicon TCP pose (same format as ViveTracker)."""
    def __init__(self, node, topic, name='ViconTcp'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array(
                [p.x, p.y, p.z, q.x, q.y, q.z, q.w],
                dtype=np.float32
            )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.node = node
        self.name = name
        self.topic = topic
        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=1000)
        self.parse_fn = parse_fn
        self.count = 0
        self.last_msg_time = None
        self.first_msg_time = None
        self.sub = node.create_subscription(PoseStamped, topic, self._cb, qos)

        node.get_logger().info(f"[{name}] Subscribed to {topic} with BEST_EFFORT QoS")


class FingertipWidthRecorder(RecorderWorker):
    """Record fingertip opening width (scalar)."""
    def __init__(self, node, topic, name):
        from std_msgs.msg import Float64

        def parse_fn(msg: Float64):
            return float(msg.data)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.node = node
        self.name = name
        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=1000)
        self.parse_fn = parse_fn
        self.count = 0
        self.topic = topic
        self.last_msg_time = None
        self.first_msg_time = None
        self.sub = node.create_subscription(Float64, topic, self._cb, qos)

        node.get_logger().info(f"[{name}] Subscribed to {topic}")


class KinectRGBRecorder(RecorderWorker):
    """Kinect RGB (store compressed JPEG bytes, no decode)."""
    def __init__(self, node, topic='/ak/rgb/image_raw/compressed'):
        def parse_fn(msg: CompressedImage):
            return {
                "format": msg.format,
                "data": np.frombuffer(msg.data, dtype=np.uint8)
            }
        super().__init__(node, topic, CompressedImage, parse_fn, name='KinectRGB')


class DesiredPoseMatrixRecorder(RecorderWorker):
    """Record 4x4 pose matrix from DesiredPose."""
    def __init__(self, node, topic, name="DesiredPoseMatrix"):
        from custom_msgs.msg import DesiredPose

        def parse_fn(msg: DesiredPose):
            return np.array(msg.pose_matrix, dtype=np.float32).reshape(4, 4)

        super().__init__(node, topic, DesiredPose, parse_fn, name=name)

class ActualEePoseMatrixRecorder(RecorderWorker):
    """Record actual end-effector 4x4 pose matrix from DesiredPose."""
    def __init__(self, node, topic, name="ActualEePoseMatrix"):
        from custom_msgs.msg import DesiredPose
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        def parse_fn(msg: DesiredPose):
            return np.array(msg.pose_matrix, dtype=np.float32).reshape(4, 4)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.node = node
        self.name = name
        self.topic = topic
        self.lock = threading.Lock()
        self.buf = collections.deque(maxlen=1000)
        self.parse_fn = parse_fn
        self.count = 0
        self.last_msg_time = None
        self.first_msg_time = None
        self.sub = node.create_subscription(DesiredPose, topic, self._cb, qos)

        node.get_logger().info(f"[{name}] Subscribed to {topic} with BEST_EFFORT QoS")


class AngleDataRecorder(RecorderWorker):
    """Record inspire angle data."""
    def __init__(self, node, topic, name="AngleData"):
        from inspire_interfaces.msg import GetAngleAct1

        def parse_fn(msg: GetAngleAct1):
            return {
                "finger_ids": np.array(msg.finger_ids, dtype=np.int32),
                "angles": np.array(msg.angles, dtype=np.int32),
                "finger_names": list(msg.finger_names)
            }

        super().__init__(node, topic, GetAngleAct1, parse_fn, name=name)


class TouchDataRecorder(RecorderWorker):
    """Record inspire touch data."""
    def __init__(self, node, topic, name="TouchData"):
        from inspire_interfaces.msg import GetTouchAct1

        def parse_fn(msg: GetTouchAct1):
            return {
                "finger_ids": np.array(msg.finger_ids, dtype=np.int32),
                "touch_values": np.array(msg.touch_values, dtype=np.int32),
                "finger_names": list(msg.finger_names)
            }

        super().__init__(node, topic, GetTouchAct1, parse_fn, name=name)


# ========== Aggregator ==========

class Aggregator:
    """Sample all workers periodically and aggregate synchronized samples."""
    def __init__(self, node, workers, rate_hz=5.0, slop_sec=0.1, on_sample=None):
        self.node = node
        self.workers = workers
        self.rate_hz = float(rate_hz)
        self.slop = float(slop_sec)
        self.on_sample = on_sample
        self.timer = node.create_timer(1.0 / rate_hz, self._tick)
        self.active = False
        self._last_warn_time = 0.0
        self.warn_interval = 1.0

    def start(self):
        self.active = True
        self.node.get_logger().info(f"Aggregator started ({self.rate_hz} Hz)")

    def stop(self):
        self.active = False
        self.node.get_logger().info("Aggregator stopped")

    def build_sensor_status_lines(self):
        now = t_now(self.node)

        ref_worker = self.workers.get("kinect_rgb")
        if ref_worker is None and self.workers:
            ref_worker = next(iter(self.workers.values()))
        if not ref_worker or not ref_worker.buf:
            return ["reference sensor has no data yet"]

        with ref_worker.lock:
            t_star = ref_worker.buf[-1][0]

        missing_lines = []
        for name, worker in self.workers.items():
            item = worker.nearest(t_star, self.slop)
            if item is None:
                if worker.last_msg_time is None:
                    missing_lines.append(f"{name}: never received data")
                else:
                    dt = now - worker.last_msg_time
                    missing_lines.append(f"{name}: last msg {dt:.2f}s ago")

        if not missing_lines:
            return ["All sensors OK"]

        return missing_lines

    def _tick(self):
        if not self.active:
            return
        now = t_now(self.node)

        # 按你的要求，这段暂时不改
        ref_worker = self.workers.get("kinect_rgb")
        ref_worker = next(iter(self.workers.values()))
        if not ref_worker or not ref_worker.buf:
            return

        with ref_worker.lock:
            t_star = ref_worker.buf[-1][0]

        picks = {}
        missing = []

        for name, worker in self.workers.items():
            item = worker.nearest(t_star, self.slop)
            if item:
                picks[name] = item
            else:
                missing.append(name)

        if missing:
            if now - self._last_warn_time > self.warn_interval:
                self._last_warn_time = now

                msgs = []
                for name in missing:
                    w = self.workers[name]
                    if w.last_msg_time is None:
                        msgs.append(f"{name}: never received data")
                    else:
                        dt = now - w.last_msg_time
                        msgs.append(f"{name}: last msg {dt:.2f}s ago")

                self.node.get_logger().warn(
                    "[Aggregator] Waiting for sensors:\n  " +
                    "\n  ".join(msgs)
                )
            return

        if len(picks) < len(self.workers):
            return

        sample = {
            "t_ref": float(t_star),
            "t": float(np.mean([p[0] for p in picks.values()])),
        }

        if "realsense_rgb" in picks:
            t_rgb, data_rgb = picks["realsense_rgb"]
            sample["rgb"] = data_rgb
            sample["rgb_t"] = float(t_rgb)

        if "realsense_rgb2" in picks:
            t_rgb2, data_rgb2 = picks["realsense_rgb2"]
            sample["rgb2"] = data_rgb2
            sample["rgb2_t"] = float(t_rgb2)

        if "realsense_rgb3" in picks:
            t_rgb3, data_rgb3 = picks["realsense_rgb3"]
            sample["rgb3"] = data_rgb3
            sample["rgb3_t"] = float(t_rgb3)

        if "vive_tracker" in picks:
            t_pose, data_pose = picks["vive_tracker"]
            sample["pose"] = data_pose
            sample["pose_t"] = float(t_pose)

        if "tactile_sensor" in picks:
            t_tact, data_tact = picks["tactile_sensor"]
            sample["tactile"] = data_tact
            sample["tactile_t"] = float(t_tact)

        if "force_torque" in picks:
            t_ft, data_ft = picks["force_torque"]
            sample["force_torque"] = data_ft
            sample["force_torque_t"] = float(t_ft)

        if "fingertip_left" in picks:
            t_ftl, data_ftl = picks["fingertip_left"]
            sample["fingertip_left"] = data_ftl
            sample["fingertip_left_t"] = float(t_ftl)

        if "fingertip_right" in picks:
            t_ftr, data_ftr = picks["fingertip_right"]
            sample["fingertip_right"] = data_ftr
            sample["fingertip_right_t"] = float(t_ftr)

        if "glove_left" in picks:
            t_gl, data_gl = picks["glove_left"]
            sample["glove_left"] = data_gl
            sample["glove_left_t"] = float(t_gl)

        if "glove_right" in picks:
            t_gr, data_gr = picks["glove_right"]
            sample["glove_right"] = data_gr
            sample["glove_right_t"] = float(t_gr)

        if "vicon_tcp_right" in picks:
            t_vr, data_vr = picks["vicon_tcp_right"]
            sample["vicon_tcp_right"] = data_vr
            sample["vicon_tcp_right_t"] = float(t_vr)

        if "vicon_tcp_left" in picks:
            t_vl, data_vl = picks["vicon_tcp_left"]
            sample["vicon_tcp_left"] = data_vl
            sample["vicon_tcp_left_t"] = float(t_vl)

        if "fingertip_width_left" in picks:
            t_wl, data_wl = picks["fingertip_width_left"]
            sample["fingertip_width_left"] = data_wl
            sample["fingertip_width_left_t"] = float(t_wl)

        if "fingertip_width_right" in picks:
            t_wr, data_wr = picks["fingertip_width_right"]
            sample["fingertip_width_right"] = data_wr
            sample["fingertip_width_right_t"] = float(t_wr)

        if "kinect_rgb" in picks:
            t_krgb, data_krgb = picks["kinect_rgb"]
            sample["kinect_rgb"] = data_krgb
            sample["kinect_rgb_t"] = float(t_krgb)

        if "desired_pose_left" in picks:
            t_dp, data_dp = picks["desired_pose_left"]
            sample["desired_pose_left"] = data_dp
            sample["desired_pose_left_t"] = float(t_dp)

        if "desired_pose_right" in picks:
            t_dp_r, data_dp_r = picks["desired_pose_right"]
            sample["desired_pose_right"] = data_dp_r
            sample["desired_pose_right_t"] = float(t_dp_r)

        if "actual_ee_pose_left" in picks:
            t_ael, data_ael = picks["actual_ee_pose_left"]
            sample["actual_ee_pose_left"] = data_ael
            sample["actual_ee_pose_left_t"] = float(t_ael)

        if "actual_ee_pose_right" in picks:
            t_aer, data_aer = picks["actual_ee_pose_right"]
            sample["actual_ee_pose_right"] = data_aer
            sample["actual_ee_pose_right_t"] = float(t_aer)

        if "angle_left" in picks:
            t_ang_l, data_ang_l = picks["angle_left"]
            sample["angle_left"] = data_ang_l
            sample["angle_left_t"] = float(t_ang_l)

        if "angle_right" in picks:
            t_ar, data_ar = picks["angle_right"]
            sample["angle_right"] = data_ar
            sample["angle_right_t"] = float(t_ar)

        if "touch_right" in picks:
            t_tr, data_tr = picks["touch_right"]
            sample["touch_right"] = data_tr
            sample["touch_right_t"] = float(t_tr)

        if "touch_left" in picks:
            t_tl, data_tl = picks["touch_left"]
            sample["touch_left"] = data_tl
            sample["touch_left_t"] = float(t_tl)

        if self.on_sample:
            self.on_sample(sample)


# ========== Asynchronous save thread ==========

class SaverWorker:
    def __init__(self, node, out_dir, on_save_result=None):
        self.node = node
        self.out_dir = out_dir
        self.on_save_result = on_save_result
        os.makedirs(out_dir, exist_ok=True)
        self.q = queue.Queue()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def save_async(self, payload):
        self.q.put(payload)

    def join(self):
        self.q.join()

    @staticmethod
    def _is_zipfile(path):
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    def _loop(self):
        while True:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                if not rclpy.ok():
                    break
                continue

            try:
                saved_path = self._save_npz(payload)
                if self.on_save_result is not None:
                    try:
                        self.on_save_result(True, payload, saved_path, "")
                    except Exception as cb_e:
                        self.node.get_logger().error(f"Save callback error: {cb_e}")
            except Exception as e:
                self.node.get_logger().error(f"Save error: {e}")
                if self.on_save_result is not None:
                    try:
                        self.on_save_result(False, payload, None, str(e))
                    except Exception as cb_e:
                        self.node.get_logger().error(f"Save callback error: {cb_e}")
            finally:
                self.q.task_done()

    def _save_npz(self, payload):
        path = payload["path"]
        arrays = dict(payload["arrays"])
        meta = payload.get("meta", {})
        arrays["meta_json"] = np.array(json.dumps(meta), dtype=object)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        tmp_path = path + ".tmp.npz"
        with open(tmp_path, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

        if not self._is_zipfile(path):
            bad_path = path + ".bad"
            try:
                os.rename(path, bad_path)
            except Exception:
                pass
            raise IOError(f"Invalid NPZ file (moved to {bad_path})")

        self.node.get_logger().info(f"Saved episode to {path}")
        return path


# ========== Episode Recorder ==========

class EpisodeRecorder:
    def __init__(self, node, workers, out_dir='/tmp/data', rate_hz=5.0, slop_sec=0.1, on_save_result=None):
        self.node = node
        self.workers = workers
        self.saver = SaverWorker(node, out_dir, on_save_result=on_save_result)
        self.agg = Aggregator(node, workers, rate_hz, slop_sec, self._on_sample)
        self._reset_buffers()
        self.recording = False
        self.out_dir = out_dir
        self.episode_id = ""

    def _reset_buffers(self):
        self.samples = []

    def _on_sample(self, sample):
        self.samples.append(sample)

    def start(self):
        if self.recording:
            return False, "Already recording"
        self.recording = True
        self.episode_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self._reset_buffers()
        self.agg.start()
        return True, f"Started episode {self.episode_id}"

    def stop(self):
        if not self.recording:
            return False, "Not recording"

        self.recording = False
        self.agg.stop()
        os.makedirs(self.out_dir, exist_ok=True)

        path = os.path.join(self.out_dir, f"episode_{self.episode_id}.npz")
        filename = os.path.basename(path)
        n_samples = len(self.samples)

        t_ref_list = []
        t_list = []

        rgb_list = []
        rgb_t_list = []
        rgb2_list = []
        rgb2_t_list = []
        rgb3_list = []
        rgb3_t_list = []
        tactile_list = []
        tactile_t_list = []
        pose_list = []
        pose_t_list = []
        ft_list = []
        ft_t_list = []
        fingertip_left_list = []
        fingertip_left_t_list = []
        fingertip_right_list = []
        fingertip_right_t_list = []
        glove_left_list = []
        glove_left_t_list = []
        glove_right_list = []
        glove_right_t_list = []

        vicon_tcp_right_list = []
        vicon_tcp_right_t_list = []
        vicon_tcp_left_list = []
        vicon_tcp_left_t_list = []

        fingertip_width_left_list = []
        fingertip_width_left_t_list = []
        fingertip_width_right_list = []
        fingertip_width_right_t_list = []

        desired_pose_left_list = []
        desired_pose_left_t_list = []
        desired_pose_right_list = []
        desired_pose_right_t_list = []

        angle_left_list = []
        angle_left_t_list = []
        angle_right_list = []
        angle_right_t_list = []

        kinect_rgb_list = []
        kinect_rgb_t_list = []

        touch_right_list = []
        touch_right_t_list = []

        touch_left_list = []
        touch_left_t_list = []

        actual_ee_pose_left_list = []
        actual_ee_pose_left_t_list = []
        actual_ee_pose_right_list = []
        actual_ee_pose_right_t_list = []

        for s in self.samples:
            t_ref_list.append(s.get("t_ref", np.nan))
            t_list.append(s.get("t", np.nan))

            if "rgb" in s:
                rgb_list.append(s["rgb"])
                rgb_t_list.append(s.get("rgb_t", np.nan))

            if "rgb2" in s:
                rgb2_list.append(s["rgb2"])
                rgb2_t_list.append(s.get("rgb2_t", np.nan))

            if "rgb3" in s:
                rgb3_list.append(s["rgb3"])
                rgb3_t_list.append(s.get("rgb3_t", np.nan))

            if "tactile" in s:
                tactile_list.append(s["tactile"])
                tactile_t_list.append(s.get("tactile_t", np.nan))

            if "pose" in s:
                pose_list.append(s["pose"])
                pose_t_list.append(s.get("pose_t", np.nan))

            if "force_torque" in s:
                ft_list.append(s["force_torque"])
                ft_t_list.append(s.get("force_torque_t", np.nan))

            if "fingertip_left" in s:
                fingertip_left_list.append(s["fingertip_left"])
                fingertip_left_t_list.append(s.get("fingertip_left_t", np.nan))

            if "fingertip_right" in s:
                fingertip_right_list.append(s["fingertip_right"])
                fingertip_right_t_list.append(s.get("fingertip_right_t", np.nan))

            if "glove_left" in s:
                glove_left_list.append(s["glove_left"])
                glove_left_t_list.append(s.get("glove_left_t", np.nan))

            if "glove_right" in s:
                glove_right_list.append(s["glove_right"])
                glove_right_t_list.append(s.get("glove_right_t", np.nan))

            if "vicon_tcp_right" in s:
                vicon_tcp_right_list.append(s["vicon_tcp_right"])
                vicon_tcp_right_t_list.append(s.get("vicon_tcp_right_t", np.nan))

            if "vicon_tcp_left" in s:
                vicon_tcp_left_list.append(s["vicon_tcp_left"])
                vicon_tcp_left_t_list.append(s.get("vicon_tcp_left_t", np.nan))

            if "fingertip_width_left" in s:
                fingertip_width_left_list.append(s["fingertip_width_left"])
                fingertip_width_left_t_list.append(s.get("fingertip_width_left_t", np.nan))

            if "fingertip_width_right" in s:
                fingertip_width_right_list.append(s["fingertip_width_right"])
                fingertip_width_right_t_list.append(s.get("fingertip_width_right_t", np.nan))

            if "kinect_rgb" in s:
                kinect_rgb_list.append(s["kinect_rgb"])
                kinect_rgb_t_list.append(s.get("kinect_rgb_t", np.nan))

            if "desired_pose_left" in s:
                desired_pose_left_list.append(s["desired_pose_left"])
                desired_pose_left_t_list.append(s.get("desired_pose_left_t", np.nan))

            if "desired_pose_right" in s:
                desired_pose_right_list.append(s["desired_pose_right"])
                desired_pose_right_t_list.append(s.get("desired_pose_right_t", np.nan))

            if "angle_left" in s:
                angle_left_list.append(s["angle_left"])
                angle_left_t_list.append(s.get("angle_left_t", np.nan))

            if "angle_right" in s:
                angle_right_list.append(s["angle_right"])
                angle_right_t_list.append(s.get("angle_right_t", np.nan))

            if "touch_right" in s:
                touch_right_list.append(s["touch_right"])
                touch_right_t_list.append(s.get("touch_right_t", np.nan))

            if "touch_left" in s:
                touch_left_list.append(s["touch_left"])
                touch_left_t_list.append(s.get("touch_left_t", np.nan))

            if "actual_ee_pose_left" in s:
                actual_ee_pose_left_list.append(s["actual_ee_pose_left"])
                actual_ee_pose_left_t_list.append(s.get("actual_ee_pose_left_t", np.nan))

            if "actual_ee_pose_right" in s:
                actual_ee_pose_right_list.append(s["actual_ee_pose_right"])
                actual_ee_pose_right_t_list.append(s.get("actual_ee_pose_right_t", np.nan))

        arrays = {
            "t_ref": np.array(t_ref_list, dtype=np.float64),
            "t": np.array(t_list, dtype=np.float64),
            "pose": np.array(pose_list, dtype=np.float32) if pose_list else np.zeros((0, 7), np.float32),
        }

        t_ref_arr = arrays["t_ref"]

        if rgb_list:
            arrays["rgb"] = np.array(rgb_list, dtype=object)
            if rgb_t_list:
                rgb_t_arr = np.array(rgb_t_list, dtype=np.float64)
                arrays["rgb_t"] = rgb_t_arr
                arrays["rgb_dt"] = rgb_t_arr - t_ref_arr

        if rgb2_list:
            arrays["rgb2"] = np.array(rgb2_list, dtype=object)
            if rgb2_t_list:
                rgb2_t_arr = np.array(rgb2_t_list, dtype=np.float64)
                arrays["rgb2_t"] = rgb2_t_arr
                arrays["rgb2_dt"] = rgb2_t_arr - t_ref_arr

        if rgb3_list:
            arrays["rgb3"] = np.array(rgb3_list, dtype=object)
            if rgb3_t_list:
                rgb3_t_arr = np.array(rgb3_t_list, dtype=np.float64)
                arrays["rgb3_t"] = rgb3_t_arr
                arrays["rgb3_dt"] = rgb3_t_arr - t_ref_arr

        if tactile_list:
            arrays["tactile"] = np.array(tactile_list, dtype=object)
            if tactile_t_list:
                tactile_t_arr = np.array(tactile_t_list, dtype=np.float64)
                arrays["tactile_t"] = tactile_t_arr
                arrays["tactile_dt"] = tactile_t_arr - t_ref_arr

        if pose_list:
            pose_t_arr = np.array(pose_t_list, dtype=np.float64)
            arrays["pose_t"] = pose_t_arr
            arrays["pose_dt"] = pose_t_arr - t_ref_arr

        if ft_list:
            arrays["force_torque"] = np.array(ft_list, dtype=np.float32)
            if ft_t_list:
                ft_t_arr = np.array(ft_t_list, dtype=np.float64)
                arrays["force_torque_t"] = ft_t_arr
                arrays["force_torque_dt"] = ft_t_arr - t_ref_arr

        if fingertip_left_list:
            arrays["fingertip_left"] = np.array(fingertip_left_list, dtype=np.float32)
            if fingertip_left_t_list:
                ftl_t_arr = np.array(fingertip_left_t_list, dtype=np.float64)
                arrays["fingertip_left_t"] = ftl_t_arr
                arrays["fingertip_left_dt"] = ftl_t_arr - t_ref_arr

        if fingertip_right_list:
            arrays["fingertip_right"] = np.array(fingertip_right_list, dtype=np.float32)
            if fingertip_right_t_list:
                ftr_t_arr = np.array(fingertip_right_t_list, dtype=np.float64)
                arrays["fingertip_right_t"] = ftr_t_arr
                arrays["fingertip_right_dt"] = ftr_t_arr - t_ref_arr

        if glove_left_list:
            arrays["glove_left"] = np.array(glove_left_list, dtype=np.float32)
            if glove_left_t_list:
                gl_t_arr = np.array(glove_left_t_list, dtype=np.float64)
                arrays["glove_left_t"] = gl_t_arr
                arrays["glove_left_dt"] = gl_t_arr - t_ref_arr

        if glove_right_list:
            arrays["glove_right"] = np.array(glove_right_list, dtype=np.float32)
            if glove_right_t_list:
                gr_t_arr = np.array(glove_right_t_list, dtype=np.float64)
                arrays["glove_right_t"] = gr_t_arr
                arrays["glove_right_dt"] = gr_t_arr - t_ref_arr

        if vicon_tcp_right_list:
            arrays["vicon_tcp_right"] = np.array(vicon_tcp_right_list, dtype=np.float32)
            if vicon_tcp_right_t_list:
                vr_t_arr = np.array(vicon_tcp_right_t_list, dtype=np.float64)
                arrays["vicon_tcp_right_t"] = vr_t_arr
                arrays["vicon_tcp_right_dt"] = vr_t_arr - t_ref_arr

        if vicon_tcp_left_list:
            arrays["vicon_tcp_left"] = np.array(vicon_tcp_left_list, dtype=np.float32)
            if vicon_tcp_left_t_list:
                vl_t_arr = np.array(vicon_tcp_left_t_list, dtype=np.float64)
                arrays["vicon_tcp_left_t"] = vl_t_arr
                arrays["vicon_tcp_left_dt"] = vl_t_arr - t_ref_arr

        if fingertip_width_left_list:
            arrays["fingertip_width_left"] = np.array(fingertip_width_left_list, dtype=np.float32)
            if fingertip_width_left_t_list:
                wl_t_arr = np.array(fingertip_width_left_t_list, dtype=np.float64)
                arrays["fingertip_width_left_t"] = wl_t_arr
                arrays["fingertip_width_left_dt"] = wl_t_arr - t_ref_arr

        if fingertip_width_right_list:
            arrays["fingertip_width_right"] = np.array(fingertip_width_right_list, dtype=np.float32)
            if fingertip_width_right_t_list:
                wr_t_arr = np.array(fingertip_width_right_t_list, dtype=np.float64)
                arrays["fingertip_width_right_t"] = wr_t_arr
                arrays["fingertip_width_right_dt"] = wr_t_arr - t_ref_arr

        if kinect_rgb_list:
            arrays["kinect_rgb"] = np.array(kinect_rgb_list, dtype=object)
            if kinect_rgb_t_list:
                kinect_rgb_t_arr = np.array(kinect_rgb_t_list, dtype=np.float64)
                arrays["kinect_rgb_t"] = kinect_rgb_t_arr
                arrays["kinect_rgb_dt"] = kinect_rgb_t_arr - t_ref_arr

        if desired_pose_left_list:
            arrays["desired_pose_left"] = np.array(desired_pose_left_list, dtype=np.float32)
            if desired_pose_left_t_list:
                dp_t_arr = np.array(desired_pose_left_t_list, dtype=np.float64)
                arrays["desired_pose_left_t"] = dp_t_arr
                arrays["desired_pose_left_dt"] = dp_t_arr - t_ref_arr

        if desired_pose_right_list:
            arrays["desired_pose_right"] = np.array(desired_pose_right_list, dtype=np.float32)
            if desired_pose_right_t_list:
                dpr_t_arr = np.array(desired_pose_right_t_list, dtype=np.float64)
                arrays["desired_pose_right_t"] = dpr_t_arr
                arrays["desired_pose_right_dt"] = dpr_t_arr - t_ref_arr

        if angle_left_list:
            arrays["angle_left"] = np.array(angle_left_list, dtype=object)
            if angle_left_t_list:
                al_t_arr = np.array(angle_left_t_list, dtype=np.float64)
                arrays["angle_left_t"] = al_t_arr
                arrays["angle_left_dt"] = al_t_arr - t_ref_arr

        if angle_right_list:
            arrays["angle_right"] = np.array(angle_right_list, dtype=object)
            if angle_right_t_list:
                ar_t_arr = np.array(angle_right_t_list, dtype=np.float64)
                arrays["angle_right_t"] = ar_t_arr
                arrays["angle_right_dt"] = ar_t_arr - t_ref_arr

        if touch_right_list:
            arrays["touch_right"] = np.array(touch_right_list, dtype=object)
            if touch_right_t_list:
                tr_t_arr = np.array(touch_right_t_list, dtype=np.float64)
                arrays["touch_right_t"] = tr_t_arr
                arrays["touch_right_dt"] = tr_t_arr - t_ref_arr

        if touch_left_list:
            arrays["touch_left"] = np.array(touch_left_list, dtype=object)
            if touch_left_t_list:
                tl_t_arr = np.array(touch_left_t_list, dtype=np.float64)
                arrays["touch_left_t"] = tl_t_arr
                arrays["touch_left_dt"] = tl_t_arr - t_ref_arr

        if actual_ee_pose_left_list:
            arrays["actual_ee_pose_left"] = np.array(actual_ee_pose_left_list, dtype=np.float32)
            if actual_ee_pose_left_t_list:
                ael_t_arr = np.array(actual_ee_pose_left_t_list, dtype=np.float64)
                arrays["actual_ee_pose_left_t"] = ael_t_arr
                arrays["actual_ee_pose_left_dt"] = ael_t_arr - t_ref_arr

        if actual_ee_pose_right_list:
            arrays["actual_ee_pose_right"] = np.array(actual_ee_pose_right_list, dtype=np.float32)
            if actual_ee_pose_right_t_list:
                aer_t_arr = np.array(actual_ee_pose_right_t_list, dtype=np.float64)
                arrays["actual_ee_pose_right_t"] = aer_t_arr
                arrays["actual_ee_pose_right_dt"] = aer_t_arr - t_ref_arr

        meta = {
            "episode_id": self.episode_id,
            "n_samples": n_samples,
            "keys": list(arrays.keys())
        }

        self.saver.save_async({
            "path": path,
            "arrays": arrays,
            "meta": meta,
            "episode_id": self.episode_id,
            "n_samples": n_samples,
            "filename": filename,
        })

        return True, f"Save queued: {filename} ({n_samples} samples)"

    def shutdown(self):
        self.agg.stop()
        self.saver.join()


# ========== Main Node ==========

class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_data_collection_with_timestamps')

        self.declare_parameter('out_dir', default_data_dir())
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('slop_sec', 0.10)
        self.declare_parameter('enable_rgb', False)
        self.declare_parameter('enable_rgb2', True)
        self.declare_parameter('enable_rgb3', True)
        self.declare_parameter('enable_vive', False)
        self.declare_parameter('enable_tactile', True)
        self.declare_parameter('rgb_topic', '/camera_third_view/color/image_raw/compressed')
        self.declare_parameter('rgb2_topic', '/camera_wrist_right/color/image_raw/compressed')
        self.declare_parameter('rgb3_topic', '/camera_wrist_left/color/image_raw/compressed')
        self.declare_parameter('vive_topic', '/vive_tracker/pose')
        self.declare_parameter('tactile_topic', '/gelsight/image_raw')
        self.declare_parameter('enable_ft', True)
        self.declare_parameter('ft_topic', '/inertial_wrench_compensation/wrench_compensated')
        self.declare_parameter('enable_angle_left', True)
        self.declare_parameter('angle_left_topic', '/left/angle_data')
        self.declare_parameter('enable_angle_right', True)
        self.declare_parameter('angle_right_topic', '/right/angle_data')

        self.declare_parameter('enable_desired_pose_left', True)
        self.declare_parameter('desired_pose_left_topic', '/left/desired_pose_matrix')
        self.declare_parameter('enable_desired_pose_right', True)
        self.declare_parameter('desired_pose_right_topic', '/right/desired_pose_matrix')

        self.declare_parameter('enable_actual_ee_pose_left', True)
        self.declare_parameter('actual_ee_pose_left_topic', '/frankaLeft/ee_pose_matrix')
        self.declare_parameter('enable_actual_ee_pose_right', True)
        self.declare_parameter('actual_ee_pose_right_topic', '/frankaRight/ee_pose_matrix')

        self.declare_parameter('enable_fingertip_left', False)
        self.declare_parameter('enable_glove_left', False)
        self.declare_parameter('enable_fingertip_right', False)
        self.declare_parameter('enable_glove_right', False)
        self.declare_parameter('fingertip_left_topic', '/manus_fingertip_left')
        self.declare_parameter('fingertip_right_topic', '/manus_fingertip_right')
        self.declare_parameter('glove_left_topic', '/manus_glove_data_left')
        self.declare_parameter('glove_right_topic', '/manus_glove_data_right')

        self.declare_parameter('enable_vicon_tcp_right', False)
        self.declare_parameter('enable_vicon_tcp_left', False)
        self.declare_parameter('vicon_tcp_right_topic', '/tcp_pose_right')
        self.declare_parameter('vicon_tcp_left_topic', '/tcp_pose_left')

        self.declare_parameter('enable_fingertip_width_left', False)
        self.declare_parameter('enable_fingertip_width_right', False)
        self.declare_parameter('fingertip_width_left_topic', '/fingertip_width_left')
        self.declare_parameter('fingertip_width_right_topic', '/fingertip_width_right')

        self.declare_parameter('enable_kinect_rgb', False)
        self.declare_parameter('kinect_rgb_topic', '/ak/rgb/image_raw/compressed')

        self.declare_parameter('enable_touch_right', True)
        self.declare_parameter('touch_right_topic', '/right/touch_data')
        self.declare_parameter('enable_touch_left', False)
        self.declare_parameter('touch_left_topic', '/left/touch_data')

        out_dir = os.path.expanduser(self.get_parameter('out_dir').value)
        rate_hz = self.get_parameter('rate_hz').value
        slop_sec = self.get_parameter('slop_sec').value
        enable_rgb = self.get_parameter('enable_rgb').value
        enable_rgb2 = self.get_parameter('enable_rgb2').value
        enable_rgb3 = self.get_parameter('enable_rgb3').value
        enable_vive = self.get_parameter('enable_vive').value
        enable_tactile = self.get_parameter('enable_tactile').value
        enable_ft = self.get_parameter('enable_ft').value
        rgb_topic = self.get_parameter('rgb_topic').value
        rgb2_topic = self.get_parameter('rgb2_topic').value
        rgb3_topic = self.get_parameter('rgb3_topic').value
        vive_topic = self.get_parameter('vive_topic').value
        tactile_topic = self.get_parameter('tactile_topic').value
        ft_topic = self.get_parameter('ft_topic').value

        enable_desired_pose_left = self.get_parameter('enable_desired_pose_left').value
        desired_pose_left_topic = self.get_parameter('desired_pose_left_topic').value
        enable_desired_pose_right = self.get_parameter('enable_desired_pose_right').value
        desired_pose_right_topic = self.get_parameter('desired_pose_right_topic').value

        enable_actual_ee_pose_left = self.get_parameter('enable_actual_ee_pose_left').value
        actual_ee_pose_left_topic = self.get_parameter('actual_ee_pose_left_topic').value
        enable_actual_ee_pose_right = self.get_parameter('enable_actual_ee_pose_right').value
        actual_ee_pose_right_topic = self.get_parameter('actual_ee_pose_right_topic').value

        enable_angle_left = self.get_parameter('enable_angle_left').value
        angle_left_topic = self.get_parameter('angle_left_topic').value
        enable_angle_right = self.get_parameter('enable_angle_right').value
        angle_right_topic = self.get_parameter('angle_right_topic').value

        enable_touch_right = self.get_parameter('enable_touch_right').value
        touch_right_topic = self.get_parameter('touch_right_topic').value
        enable_touch_left = self.get_parameter('enable_touch_left').value
        touch_left_topic = self.get_parameter('touch_left_topic').value

        enable_kinect_rgb = self.get_parameter('enable_kinect_rgb').value
        kinect_rgb_topic = self.get_parameter('kinect_rgb_topic').value

        enable_fingertip_left = self.get_parameter('enable_fingertip_left').value
        enable_glove_left = self.get_parameter('enable_glove_left').value
        enable_fingertip_right = self.get_parameter('enable_fingertip_right').value
        enable_glove_right = self.get_parameter('enable_glove_right').value
        fingertip_left_topic = self.get_parameter('fingertip_left_topic').value
        fingertip_right_topic = self.get_parameter('fingertip_right_topic').value
        glove_left_topic = self.get_parameter('glove_left_topic').value
        glove_right_topic = self.get_parameter('glove_right_topic').value

        enable_vicon_tcp_right = self.get_parameter('enable_vicon_tcp_right').value
        enable_vicon_tcp_left = self.get_parameter('enable_vicon_tcp_left').value
        vicon_tcp_right_topic = self.get_parameter('vicon_tcp_right_topic').value
        vicon_tcp_left_topic = self.get_parameter('vicon_tcp_left_topic').value

        enable_fingertip_width_left = self.get_parameter('enable_fingertip_width_left').value
        enable_fingertip_width_right = self.get_parameter('enable_fingertip_width_right').value
        fingertip_width_left_topic = self.get_parameter('fingertip_width_left_topic').value
        fingertip_width_right_topic = self.get_parameter('fingertip_width_right_topic').value

        self.workers = {}

        if enable_rgb:
            self.workers["realsense_rgb"] = RealSenseRGBRecorder(self, rgb_topic)
        if enable_rgb2:
            self.workers["realsense_rgb2"] = RealSenseRGBRecorder(self, rgb2_topic)
        if enable_rgb3:
            self.workers["realsense_rgb3"] = RealSenseRGBRecorder(self, rgb3_topic)
        if enable_vive:
            self.workers["vive_tracker"] = ViveTrackerRecorder(self, vive_topic)
        if enable_tactile:
            self.workers["tactile_sensor"] = TactileSensorRecorder(self, tactile_topic)
        if enable_ft:
            self.workers["force_torque"] = ForceTorqueRecorder(self, ft_topic)
        if enable_kinect_rgb:
            self.workers["kinect_rgb"] = KinectRGBRecorder(self, kinect_rgb_topic)

        if enable_fingertip_left:
            self.workers["fingertip_left"] = ManusFingertipRecorder(
                self, fingertip_left_topic, "ManusFingertipLeft"
            )
        if enable_fingertip_right:
            self.workers["fingertip_right"] = ManusFingertipRecorder(
                self, fingertip_right_topic, "ManusFingertipRight"
            )

        if enable_glove_left:
            self.workers["glove_left"] = ManusGloveDataRecorder(
                self, glove_left_topic, "ManusGloveLeft"
            )
        if enable_glove_right:
            self.workers["glove_right"] = ManusGloveDataRecorder(
                self, glove_right_topic, "ManusGloveRight"
            )

        if enable_vicon_tcp_right:
            self.workers["vicon_tcp_right"] = ViconTcpRecorder(
                self, vicon_tcp_right_topic, name="ViconTcpRight"
            )

        if enable_vicon_tcp_left:
            self.workers["vicon_tcp_left"] = ViconTcpRecorder(
                self, vicon_tcp_left_topic, name="ViconTcpLeft"
            )

        if enable_fingertip_width_left:
            self.workers["fingertip_width_left"] = FingertipWidthRecorder(
                self, fingertip_width_left_topic, "FingertipWidthLeft"
            )

        if enable_fingertip_width_right:
            self.workers["fingertip_width_right"] = FingertipWidthRecorder(
                self, fingertip_width_right_topic, "FingertipWidthRight"
            )

        if enable_desired_pose_left:
            self.workers["desired_pose_left"] = DesiredPoseMatrixRecorder(
                self, desired_pose_left_topic, name="DesiredPoseLeft"
            )

        if enable_desired_pose_right:
            self.workers["desired_pose_right"] = DesiredPoseMatrixRecorder(
                self, desired_pose_right_topic, name="DesiredPoseRight"
            )

        if enable_angle_left:
            self.workers["angle_left"] = AngleDataRecorder(
                self, angle_left_topic, name="AngleLeft"
            )

        if enable_angle_right:
            self.workers["angle_right"] = AngleDataRecorder(
                self, angle_right_topic, name="AngleRight"
            )

        if enable_touch_right:
            self.workers["touch_right"] = TouchDataRecorder(
                self, touch_right_topic, name="TouchRight"
            )

        if enable_touch_left:
            self.workers["touch_left"] = TouchDataRecorder(
                self, touch_left_topic, name="TouchLeft"
            )
        
        if enable_actual_ee_pose_left:
            self.workers["actual_ee_pose_left"] = ActualEePoseMatrixRecorder(
                self, actual_ee_pose_left_topic, name="ActualEePoseLeft"
            )

        if enable_actual_ee_pose_right:
            self.workers["actual_ee_pose_right"] = ActualEePoseMatrixRecorder(
                self, actual_ee_pose_right_topic, name="ActualEePoseRight"
            )

        if not self.workers:
            self.get_logger().warn("No sensor workers enabled! Nothing will be recorded.")

        self.episode = EpisodeRecorder(
            self,
            self.workers,
            out_dir,
            rate_hz,
            slop_sec,
            on_save_result=self._handle_save_result,
        )

        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.recording_pub = self.create_publisher(
            Bool,
            "/episode_recording",
            qos_latched,
        )

        self.status_text_pub = self.create_publisher(
            String,
            "/recorder_status_text",
            qos_latched,
        )

        self.event_text_pub = self.create_publisher(
            String,
            "/recorder_event_text",
            qos_latched,
        )

        self.status_timer = self.create_timer(1.0, self._status_timer_cb)

        self.publish_recording_state(False)
        self.publish_event_text("")
        self.publish_status_text()

        self.start_srv = self.create_service(Trigger, 'start_episode', self._srv_start)
        self.stop_srv = self.create_service(Trigger, 'stop_episode', self._srv_stop)

        active = ', '.join(self.workers.keys())
        self.get_logger().info(f"Recorder with timestamps ready. Active modalities: {active or 'None'}")

    def publish_recording_state(self, is_recording: bool):
        msg = Bool()
        msg.data = bool(is_recording)
        self.recording_pub.publish(msg)

    def publish_event_text(self, text: str):
        msg = String()
        msg.data = text
        self.event_text_pub.publish(msg)

    def build_status_text(self):
        episode_str = ""
        if self.episode.recording:
            episode_str = getattr(self.episode, "episode_id", "")

        sensor_lines = self.episode.agg.build_sensor_status_lines()

        text = "Episode: " + episode_str + "\n"
        text += "Sensors:\n"
        for line in sensor_lines:
            text += f"  {line}\n"

        return text.rstrip()

    def publish_status_text(self):
        msg = String()
        msg.data = self.build_status_text()
        self.status_text_pub.publish(msg)

    def _status_timer_cb(self):
        self.publish_status_text()

    def _handle_save_result(self, success: bool, payload: dict, saved_path: str, error_text: str):
        filename = payload.get("filename", "")
        n_samples = payload.get("n_samples", 0)

        if success:
            text = f"Saved OK: {filename} ({n_samples} samples)"
        else:
            text = f"Save error: {error_text}"

        self.publish_event_text(text)

    def _srv_start(self, request, response):
        ok, msg = self.episode.start()
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)

        if ok:
            self.publish_recording_state(True)
            self.publish_event_text("")
            self.publish_status_text()

        return response

    def _srv_stop(self, request, response):
        ok, msg = self.episode.stop()
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)

        if ok:
            self.publish_recording_state(False)
            self.publish_event_text(msg)   # Save queued: ...
            self.publish_status_text()

        return response

    def destroy_node(self):
        try:
            self.publish_recording_state(False)
        except Exception:
            pass
        self.episode.shutdown()
        super().destroy_node()


# ========== Main ==========

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down...")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError:
            pass


if __name__ == '__main__':
    main()
