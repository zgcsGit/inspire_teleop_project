#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Multimodal Data Recorder (Humble)
Record RealSense RGB and Vive Tracker pose only.
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
import re
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from typing import Optional, Tuple
from sensor_msgs.msg import Image, CompressedImage
import traceback
import glob

# ========== Utils ==========

def default_data_dir():
    return os.path.expanduser(
        os.environ.get("INSPIRE_TELEOP_DATA_DIR", "~/inspire_teleop_data/right")
    )

# def t_now(node):
#     """Return current ROS time in seconds."""
#     return node.get_clock().now().nanoseconds / 1e9

def t_now(node: Optional[Node] = None) -> float:
    """Return current ROS time in seconds."""
    if node is not None:
        t = node.get_clock().now().to_msg()
        return t.sec + t.nanosec * 1e-9
    else:
        return rclpy.clock.Clock().now().nanoseconds * 1e-9

def safe_stack(arr_list, axis=0):
    """Stack a list of NumPy arrays if they have the same shape,"""
    if len(arr_list) == 0:
        return np.array([])
    first_shape = np.shape(arr_list[0])
    same = all(np.shape(a) == first_shape for a in arr_list)
    if same:
        return np.stack(arr_list, axis=axis)
    else:
        return np.array(arr_list, dtype=object)

def list_to_object_array(lst):
    """Convert a list of arbitrary objects to a NumPy object array."""
    return np.array(lst, dtype=object)


# ========== Base Worker ==========

class RecorderWorker:
    """Subscribe to a topic and buffer parsed data."""
    def __init__(self, node, topic, msg_type, parse_fn, name='worker', maxlen=1000):
        self.name = name
        self.topic = topic
        self.msg_type = msg_type
        self.parse_fn = parse_fn
        self.node = node
        self.buf = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.sub = node.create_subscription(msg_type, topic, self._cb, 10)
        self.count = 0
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
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] parse error: {e}")

    def nearest(self, t_star, max_slop):
        """Find the buffered data nearest to t_star within max_slop."""
        with self.lock:
            if not self.buf:
                return None
            best, best_dt = None, 1e9
            for (t, d) in self.buf:
                dt = abs(t - t_star)
                if dt < best_dt:
                    best, best_dt = (t, d), dt
            return best if best_dt <= max_slop else None


# ========== Specific Workers ==========

# class RealSenseRGBRecorder(RecorderWorker):
#     """RealSense RGB image recorder."""
#     def __init__(self, node, topic='/camera/color/image_rect_raw'):
#         bridge = CvBridge()

#         def parse_fn(msg: Image):
#             img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#         super().__init__(node, topic, Image, parse_fn, name='RealSenseRGB')


class RGBDRecorder:
    """ROS2 RGB-D recorder with dual buffer + nearest-neighbor sync."""
    def __init__(self, node: Node,
                 color_topic: str = '/camera/color/image_rect_raw',
                 depth_topic: str = '/camera/depth/image_rect_raw',
                 name: str = 'RGBDRecorder',
                 maxlen: int = 200,
                 bgr_to_rgb: bool = True,
                 img_stamp_mode: str = 'auto',
                 max_stamp_skew: float = 1.0,
                 want_color: bool = True,
                 want_depth: bool = False):

        self.node = node
        self.name = name
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.bgr_to_rgb = bgr_to_rgb

        # double buffer
        self.color_buf = collections.deque(maxlen=maxlen)
        self.depth_buf = collections.deque(maxlen=maxlen)

        # color /depth
        self.want_color = bool(want_color)
        self.want_depth = bool(want_depth)

        # topic 
        self.color_is_comp = color_topic.endswith('/compressed')
        self.depth_is_comp = depth_topic.endswith('/compressedDepth') or depth_topic.endswith('/compressed')

        self.color_msg_type = CompressedImage if self.color_is_comp else Image
        self.depth_msg_type = CompressedImage if self.depth_is_comp else Image

        self.color_count = 0
        self.depth_count = 0

        # subscriptions
        if self.want_color:
            self.color_sub = node.create_subscription(self.color_msg_type, color_topic, self._cb_color, 10)
            node.get_logger().info(f"[{self.name}] Subscribed to {color_topic}")
        if self.want_depth:
            self.depth_sub = node.create_subscription(self.depth_msg_type, depth_topic, self._cb_depth, 10)
            node.get_logger().info(f"[{self.name}] Subscribed to {depth_topic}")

        # timestamp mode
        self.img_stamp_mode = img_stamp_mode.lower()
        self.max_stamp_skew = float(max_stamp_skew)
        self._warned_skew = False

        

    # ---------- timestamp picking ----------
    def _pick_ts(self, msg):
        now = t_now(self.node)
        if self.img_stamp_mode == 'now':
            return now
        tmsg = None
        if hasattr(msg, 'header'):
            tmsg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if tmsg is None:
            return now
        if self.img_stamp_mode == 'msg':
            return tmsg
        # auto mode
        if abs(tmsg - now) > self.max_stamp_skew:
            if not self._warned_skew:
                self.node.get_logger().warn(
                    f"[{self.name}] image stamp skew {abs(tmsg - now):.3f}s > {self.max_stamp_skew:.3f}s, "
                    f"fallback to wall-time (once).")
                self._warned_skew = True
            return now
        return tmsg

    # ---------- color callback ----------
    def _cb_color(self, msg):
        t = self._pick_ts(msg)
        try:
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    img_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if img_bgr is None:
                self.node.get_logger().warn_once(f"[{self.name}] color decode returned None, dropping.")
                return

            img = img_bgr[:, :, ::-1] if self.bgr_to_rgb else img_bgr
            # ensure uint8
            if img.dtype == object:
                img = np.array(img.tolist(), dtype=np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)

            with self.lock:
                self.color_buf.append((t, img))
                self.color_count += 1
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] color parse error: {e}")

    # ---------- depth callback ----------
    def _cb_depth(self, msg):
        t = self._pick_ts(msg)
        try:
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    depth = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if depth is None:
                self.node.get_logger().warn_once(f"[{self.name}] depth decode returned None, dropping.")
                return

            with self.lock:
                self.depth_buf.append((t, depth))
                self.depth_count += 1
        except Exception as e:
            self.node.get_logger().warn(f"[{self.name}] depth parse error: {e}")

    # ---------- nearest ----------
    def nearest(self, t_star: float, max_slop: float = 0.05) -> Optional[Tuple[float, Tuple[np.ndarray, Optional[np.ndarray]]]]:
        """Return (t, (rgb, depth)) nearest to t_star within max_slop."""
        with self.lock:
            if self.want_color and not self.want_depth:
                if not self.color_buf:
                    return None
                best, best_dt = None, 1e9
                for (t, img) in self.color_buf:
                    dt = abs(t - t_star)
                    if dt < best_dt:
                        best, best_dt = (t, img), dt
                return (best[0], (best[1], None)) if best_dt <= max_slop else None

            if not self.color_buf or not self.depth_buf:
                return None

            # find the nearest color
            best_color, best_dt = None, 1e9
            for (t, img) in self.color_buf:
                dt = abs(t - t_star)
                if dt < best_dt:
                    best_color, best_dt = (t, img), dt
            if best_dt > max_slop:
                return None

            t_color = best_color[0]
            # then find the nearest depth
            best_depth, best_dt2 = None, 1e9
            for (td, dep) in self.depth_buf:
                dt2 = abs(td - t_color)
                if dt2 < best_dt2:
                    best_depth, best_dt2 = (td, dep), dt2
            if best_dt2 > max_slop:
                return None

            return (t_color, (best_color[1], best_depth[1]))

class ViveTrackerRecorder(RecorderWorker):
    """Vive tracker pose recorder."""
    def __init__(self, node, topic='/vive_tracker/pose'):
        def parse_fn(msg: PoseStamped):
            p = msg.pose.position
            q = msg.pose.orientation
            return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

        super().__init__(node, topic, PoseStamped, parse_fn, name='ViveTracker')


# ========== Aggregator ==========

class Aggregator:
    """Sample all workers periodically and aggregate synchronized samples (ROS2 version)."""
    def __init__(self, node, workers, rate_hz=5.0, slop_sec=0.5, on_sample=None):
        self.node = node
        self.workers = workers              # dict: name -> RecorderWorker
        self.rate_hz = float(rate_hz)
        self.slop = float(slop_sec)
        self.on_sample = on_sample

        self.active = False
        self.drop_counter = collections.Counter()
        self.stats_lock = threading.Lock()
        self.n_ticks = 0
        self.n_kept = 0
        self._last_warn_ts = 0.0

        # --- timer ---
        self.timer = node.create_timer(1.0 / self.rate_hz, self._tick)

    # --------------------------------------------------------
    def start(self):
        self.active = True
        self.node.get_logger().info(f"[Aggregator] started ({self.rate_hz:.1f} Hz)")

    def stop(self):
        self.active = False
        self.node.get_logger().info("[Aggregator] stopped")

    # --------------------------------------------------------
    def _tick(self):
        if not self.active:
            return

        t_star = t_now(self.node)
        picks = {}
        missing = []

        # --- Collect nearest neighbor data for each worker ---
        for name, worker in self.workers.items():
            item = worker.nearest(t_star, self.slop)
            if item is None:
                missing.append(name)
            else:
                picks[name] = item

        with self.stats_lock:
            self.n_ticks += 1

        # --- Check for missing frames ---
        if missing:
            for k in missing:
                self.drop_counter[k] += 1

            now = time.time()
            # print warning at most once per second
            if now - self._last_warn_ts > 1.0:
                self.node.get_logger().warn(
                    f"[Aggregator] drop@t={t_star:.3f}s missing={missing} counts={dict(self.drop_counter)}"
                )
                self._last_warn_ts = now
            return

        with self.stats_lock:
            self.n_kept += 1

        # --- Assemble sample ---
        sample = {"t": float(np.mean([p[0] for p in picks.values()]))}

        # --- Fill in data from each worker ---
        if "rgbd" in picks:
            rgb, depth = picks["rgbd"][1]
            sample["rgb"] = rgb
            if depth is not None:
                sample["depth"] = depth

        if "vive_tracker" in picks:
            sample["pose"] = picks["vive_tracker"][1]

        # --- Call the callback ---
        if self.on_sample:
            try:
                self.on_sample(sample)
            except Exception as e:
                self.node.get_logger().warn(f"[Aggregator] on_sample error: {e}")


# ========= Saver (async, atomic) ==========
class SaverWorker:
    """Asynchronous NPZ saver with atomic write and shard merging."""
    def __init__(self, node, out_dir):
        self.node = node
        self.out_dir = out_dir
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
        """Check if the file at path is a valid zip file (NPZ)."""
        try:
            return zipfile.is_zipfile(path)
        except Exception:
            return False

    @staticmethod
    def _part_idx(filename):
        """Extract part index from filename like ..._partXX.npz."""
        m = re.search(r"_part(\d+)\.npz$", filename)
        return int(m.group(1)) if m else 1 << 30

    # ============================ main Loop ============================

    def _loop(self):
        while True:
            try:
                payload = self.q.get(timeout=0.5)
            except queue.Empty:
                if not rclpy.ok():
                    break
                continue

            try:
                kind = payload.get("kind", "save")
                if kind == "save":
                    self._save_npz(payload)
                elif kind == "merge":
                    self._merge_shards(
                        base_path=payload["base_path"],
                        shard_keys=payload.get("shard_keys", []),
                        delete_parts=payload.get("delete_parts", True)
                    )
                else:
                    self.node.get_logger().warn(f"[SaverWorker] Unknown job kind: {kind}")
            except Exception as e:
                tb = traceback.format_exc()
                self.node.get_logger().error(f"[SaverWorker] Task '{payload.get('kind', '?')}' failed: {e}\n{tb}")
            finally:
                self.q.task_done()

    
    def _save_npz(self, payload):
        """Save arrays and meta to an NPZ file atomically."""
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

        self.node.get_logger().info(
            f"[SaverWorker] Saved episode to {path} "
            f"({os.path.getsize(path)/1e6:.2f} MB, keys={list(arrays.keys())})"
        )

    
    def _merge_shards(self, base_path, shard_keys, delete_parts=True):
        """
        Merge shard NPZ files into a single NPZ file atomically.
        """
        def load_npz(path):
            with np.load(path, allow_pickle=True) as d:
                return {k: d[k] for k in d.files}

        merged = {}

        # 1. load base file if exists
        if os.path.exists(base_path):
            merged.update(load_npz(base_path))

        # 2. load shards
        for key in shard_keys:
            pattern = base_path.replace(".npz", f"_{key}_part*.npz")
            parts = sorted(glob.glob(pattern), key=self._part_idx)
            if not parts:
                continue
            frames = []
            for p in parts:
                try:
                    with np.load(p, allow_pickle=True) as d:
                        if key not in d.files:
                            continue
                        arr = d[key]
                        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.ndim == 1:
                            frames.extend(list(arr))
                        elif isinstance(arr, np.ndarray) and arr.ndim >= 1:
                            frames.extend([arr[i] for i in range(arr.shape[0])])
                except Exception as e:
                    self.node.get_logger().warn(f"[SaverWorker] Failed to read shard {p}: {e}")
                    continue
            
            # # 一维 object 数组写回
            # out = np.empty((len(frames),), dtype=object)
            # out[:] = frames
            # merged[key] = out
            merged[key] = np.stack(frames, axis=0)  

        # 3. ensure meta_json exists
        if "meta_json" not in merged:
            merged["meta_json"] = np.array(json.dumps({}), dtype=object)

        # 4. write merged file atomically
        tmp_path = base_path + ".merge.tmp.npz"
        with open(tmp_path, "wb") as f:
            np.savez(f, **merged)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, base_path)

        # 5. delete parts if needed
        if delete_parts:
            for key in shard_keys:
                for p in glob.glob(base_path.replace(".npz", f"_{key}_part*.npz")):
                    try:
                        os.remove(p)
                    except Exception as e:
                        self.node.get_logger().warn(f"[SaverWorker] Failed to remove shard {p}: {e}")

        self.node.get_logger().info(
            f"[SaverWorker] Merged shards into {base_path} "
            f"(keys={list(merged.keys())}, size={os.path.getsize(base_path)/1e6:.2f} MB)"
        )

# ========== Episode Recorder ==========

class EpisodeRecorder:

    def __init__(self, node, workers, out_dir="/tmp/data", rate_hz=5.0, slop_sec=0.5,
                 rs_prewarm_frames=30):
        self.node = node
        self.workers = workers
        self.out_dir = out_dir
        self.rate_hz = rate_hz
        self.slop_sec = slop_sec
        self.rs_prewarm_frames = int(rs_prewarm_frames)

        self.saver = SaverWorker(node, out_dir)
        self.agg = Aggregator(node, workers, rate_hz=rate_hz, slop_sec=slop_sec, on_sample=self._on_sample)

        self._reset_buffers()
        self._state_lock = threading.Lock()
        self.recording = False
        self.episode_id = None
        self.meta = {}
        

    # ---------------------------- Public Methods ----------------------------
    def _save_sharded(self, base_path, arrays, shard=1000):
        
        small_keys = ("t", "pose")
        small = {k: arrays[k] for k in small_keys if k in arrays}
        self.saver.save_async({"path": base_path, "arrays": small, "meta": self.meta})

        def push(key):
            v = arrays.get(key)
            if not (isinstance(v, np.ndarray) and v.dtype == object):
                return
            n = len(v)
            for s in range(0, n, shard):
                sub = v[s:s + shard]
                part = base_path.replace(".npz", f"_{key}_part{s // shard:03d}.npz")
                self.saver.save_async({"path": part, "arrays": {key: sub}, "meta": self.meta})

        for key in ("rgb", "depth"):
            push(key)
            if key == "rgb":
                arrays[key] = []  # 清空内存缓存
                self.rgb_list.clear()

    def start_episode(self, meta=None):
        with self._state_lock:
            if self.recording:
                self.node.get_logger().warn("Episode already recording.")
                return False, "already recording"

            # -------- RealSense prewarm --------
            if "rgbd" in self.workers and hasattr(self.workers["rgbd"], "count"):
                rs = self.workers["rgbd"]
                c0 = rs.count
                need = max(1, self.rs_prewarm_frames)
                while (rs.count - c0) < need and self._ok():
                    self.node.get_logger().info(
                        f"[EpisodeRecorder] Warming up RealSense: frames={rs.count - c0}/{need}"
                    )
                    time.sleep(0.05)

            
            self._reset_buffers()
            self.episode_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            self.meta = meta or {}
            self.meta.update({
                "episode_id": self.episode_id,
                "rate_hz": self.rate_hz,
                "slop_sec": self.slop_sec,
                "ros_time": t_now(self.node),
                "topics": list(self.workers.keys())
            })

            self.recording = True
            self.agg.start()
            self.node.get_logger().info(f"[EpisodeRecorder] Episode {self.episode_id} started.")
            return True, f"started {self.episode_id}"

    def stop_episode(self, success=True):

        with self._state_lock:
            if not self.recording:
                return False, "not recording"

            self.agg.stop()
            self.recording = False

            arrays = self._pack_arrays()
            tag = "ok" if success else "abort"
            date_dir = time.strftime("%Y%m%d")
            save_dir = os.path.join(self.out_dir, date_dir)
            os.makedirs(save_dir, exist_ok=True)

            path = os.path.join(save_dir, f"episode_{self.episode_id}_{tag}.npz")
            shard = 1000  

            self._save_sharded(path, arrays, shard=shard)
            self.saver.join()
            # merge shards
            self.saver.save_async({
                "kind": "merge",
                "base_path": path,
                "shard_keys": ["rgb"],   
                "delete_parts": True
            })

            # reset buffers
            self._reset_buffers()
            self.node.get_logger().info(f"[EpisodeRecorder] Episode {self.episode_id} stopped, queued to save {path}")
            return True, "stopped"

    def shutdown(self):
        
        self.agg.stop()
        self.saver.join()
        self.node.get_logger().info("[EpisodeRecorder] Saver thread joined, shutdown complete.")

    # ---------------------------- Internal Methods ----------------------------

    def _ok(self):
        import rclpy
        return rclpy.ok()

    def _reset_buffers(self):
        
        self.t_list = []
        self.rgb_list = []
        self.pose_list = []

    def _on_sample(self, sample):
        """Callback for each aggregated sample."""
        self.t_list.append(float(sample["t"]))
        if "rgb" in sample:
            self.rgb_list.append(sample["rgb"])
        if "pose" in sample:
            self.pose_list.append(np.asarray(sample["pose"], dtype=np.float32))

    def _pack_arrays(self):
        
        arrays = {
            "t": np.asarray(self.t_list, dtype=np.float64),
            "pose": np.stack(self.pose_list, axis=0) if self.pose_list else np.zeros((0, 7), np.float32),
            "rgb": self._to_object_array(self.rgb_list)
        }
        return arrays

    def _to_object_array(self, lst):
        return np.array(lst, dtype=object)

    

# ========== ROS Node ==========

class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_data_recording')

        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('slop_sec', 0.5)
        self.declare_parameter('out_dir', default_data_dir())

        # RealSense
        self.declare_parameter('enable_rs', True)
        self.declare_parameter('rs_color_topic', '/camera/color/image_rect_raw')
        self.declare_parameter('rs_prewarm_frames', 30)

        # Vive Tracker
        self.declare_parameter('enable_vive', True)
        self.declare_parameter('vive_topic', '/vive_tracker/pose')


        self.declare_parameter('img_stamp_mode', 'auto')   # auto|msg|now
        self.declare_parameter('max_stamp_skew', 1.0)

        rate_hz = self.get_parameter('rate_hz').value
        slop_sec = self.get_parameter('slop_sec').value
        out_dir = os.path.expanduser(self.get_parameter('out_dir').value)
        enable_rs = self.get_parameter('enable_rs').value
        enable_vive = self.get_parameter('enable_vive').value
        rs_color_topic = self.get_parameter('rs_color_topic').value
        rs_prewarm = self.get_parameter('rs_prewarm_frames').value
        vive_topic = self.get_parameter('vive_topic').value
        img_stamp_mode = self.get_parameter('img_stamp_mode').value
        max_stamp_skew = self.get_parameter('max_stamp_skew').value


        self.workers = {}

        if enable_rs:
            self.workers["rgbd"] = RGBDRecorder(
                self,
                color_topic=rs_color_topic,
                depth_topic="",              
                name="RealSense",
                bgr_to_rgb=True,
                img_stamp_mode=img_stamp_mode,
                max_stamp_skew=max_stamp_skew,
                want_color=True,
                want_depth=False
            )

        if enable_vive:
            self.workers["vive_tracker"] = ViveTrackerRecorder(self, vive_topic)

        
        self.rec = EpisodeRecorder(
            node=self,
            workers=self.workers,
            out_dir=out_dir,
            rate_hz=rate_hz,
            slop_sec=slop_sec,
            rs_prewarm_frames=rs_prewarm,
        )

        # =========================
        # ---- ROS service ----
        # =========================
        self.start_srv = self.create_service(Trigger, 'start_episode', self._srv_start)
        self.stop_srv = self.create_service(Trigger, 'stop_episode', self._srv_stop)

        # =========================
        # ---- Logger ----
        # =========================
        self.get_logger().info(
            f"Recorder ready. rate={rate_hz:.2f} Hz slop={slop_sec:.3f} s, output={out_dir}"
        )
        self.get_logger().info(
            f"Enabled sensors: RealSense={enable_rs}, ViveTracker={enable_vive}"
        )
        self.get_logger().info(
            f"RealSense prewarm={rs_prewarm} frames, img_stamp_mode={img_stamp_mode}, max_skew={max_stamp_skew:.2f}s"
        )
        self.get_logger().info(
            f"Call services: ros2 service call /start_episode std_srvs/srv/Trigger '{{}}'; "
            f"ros2 service call /stop_episode std_srvs/srv/Trigger '{{}}'"
        )

    # =========================
    # ---- service callback ----
    # =========================
    def _srv_start(self, request, response):
        ok, msg = self.rec.start_episode(meta={"node": self.get_name()})
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)
        return response

    def _srv_stop(self, request, response):
        ok, msg = self.rec.stop_episode(success=True)
        response.success = ok
        response.message = msg
        self.get_logger().info(msg)
        return response

    # =========================
    # ---- destroy the node ----
    # =========================
    def destroy_node(self):
        self.rec.shutdown()
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
            pass  # already shut down



if __name__ == '__main__':
    main()
