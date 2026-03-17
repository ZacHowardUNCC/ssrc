"""
ROS2 trajectory collector for NoMaD fine-tuning datasets.

This node subscribes to camera and odometry topics, saves RGB images as
sequential files, and writes `traj_data.pkl` containing:
  - "position": np.ndarray [T, 2]
  - "yaw": np.ndarray [T]
"""

import os
import pickle
import shutil
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Convert quaternion orientation to yaw (radians)."""
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(t3, t4))


def stamp_to_sec(stamp) -> float:
    """Convert builtin_interfaces/Time to seconds."""
    if stamp is None:
        return 0.0
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def msg_to_pil(msg: Image) -> PILImage.Image:
    """Convert ROS2 sensor_msgs/Image to a PIL RGB image."""
    encoding = (msg.encoding.lower() if msg.encoding else "rgb8").replace("-", "_")
    raw = bytes(msg.data)
    h = msg.height
    w = msg.width
    step = int(msg.step) if getattr(msg, "step", 0) else 0

    def rows_2d(bytes_per_row: int) -> np.ndarray:
        flat = np.frombuffer(raw, dtype=np.uint8)
        expected = h * bytes_per_row
        if flat.size < expected:
            raise ValueError(
                f"Image buffer too small: have {flat.size} bytes, need {expected}"
            )
        return flat[:expected].reshape(h, bytes_per_row)

    def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        y = y.astype(np.float32)
        u = u.astype(np.float32) - 128.0
        v = v.astype(np.float32) - 128.0
        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u
        rgb = np.stack([r, g, b], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def yuyv_to_rgb(step_bytes: int) -> PILImage.Image:
        if w % 2 != 0:
            raise ValueError("YUYV decoding requires even image width")
        row = rows_2d(step_bytes)
        packed = row[:, : w * 2].reshape(h, w // 2, 4)
        y0 = packed[:, :, 0]
        u = packed[:, :, 1]
        y1 = packed[:, :, 2]
        v = packed[:, :, 3]
        y = np.empty((h, w), dtype=np.uint8)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        u_full = np.repeat(u, 2, axis=1)
        v_full = np.repeat(v, 2, axis=1)
        return PILImage.fromarray(yuv_to_rgb(y, u_full, v_full), "RGB")

    def uyvy_to_rgb(step_bytes: int) -> PILImage.Image:
        if w % 2 != 0:
            raise ValueError("UYVY decoding requires even image width")
        row = rows_2d(step_bytes)
        packed = row[:, : w * 2].reshape(h, w // 2, 4)
        u = packed[:, :, 0]
        y0 = packed[:, :, 1]
        v = packed[:, :, 2]
        y1 = packed[:, :, 3]
        y = np.empty((h, w), dtype=np.uint8)
        y[:, 0::2] = y0
        y[:, 1::2] = y1
        u_full = np.repeat(u, 2, axis=1)
        v_full = np.repeat(v, 2, axis=1)
        return PILImage.fromarray(yuv_to_rgb(y, u_full, v_full), "RGB")

    if encoding in ("rgb8",):
        row = rows_2d(step or (w * 3))
        img = row[:, : w * 3].reshape(h, w, 3)
        return PILImage.fromarray(img, "RGB")
    if encoding in ("bgr8",):
        row = rows_2d(step or (w * 3))
        img = row[:, : w * 3].reshape(h, w, 3)
        return PILImage.fromarray(img[:, :, ::-1].copy(), "RGB")
    if encoding in ("rgba8",):
        row = rows_2d(step or (w * 4))
        img = row[:, : w * 4].reshape(h, w, 4)
        return PILImage.fromarray(img[:, :, :3], "RGB")
    if encoding in ("bgra8",):
        row = rows_2d(step or (w * 4))
        img = row[:, : w * 4].reshape(h, w, 4)
        return PILImage.fromarray(img[:, :, 2::-1].copy(), "RGB")
    if encoding in ("mono8",):
        row = rows_2d(step or w)
        img = row[:, :w]
        return PILImage.fromarray(img, "L").convert("RGB")
    if encoding in ("yuyv", "yuy2", "yuv422", "yuv422_yuy2", "yuv422_yuyv"):
        return yuyv_to_rgb(step or (w * 2))
    if encoding in ("uyvy", "yuv422_uyvy"):
        return uyvy_to_rgb(step or (w * 2))

    if step >= w * 3:
        row = rows_2d(step)
        img = row[:, : w * 3].reshape(h, w, 3)
        return PILImage.fromarray(img, "RGB")
    row = rows_2d(step or w)
    img = row[:, :w]
    return PILImage.fromarray(img, "L").convert("RGB")


class CollectTrajectoryNode(Node):
    def __init__(self):
        super().__init__("nomad_collect_trajectory")

        self.declare_parameter("output_dir", "trajectory")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("sample_rate_hz", 4.0)
        self.declare_parameter("image_ext", "jpg")
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("overwrite", False)
        self.declare_parameter("zero_origin", True)
        self.declare_parameter("sync_tolerance_s", 0.25)
        self.declare_parameter("max_samples", -1)

        output_dir = str(self.get_parameter("output_dir").value).strip()
        if not output_dir:
            raise ValueError("output_dir must not be empty")
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self.image_ext = str(self.get_parameter("image_ext").value).lower().lstrip(".")
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.overwrite = bool(self.get_parameter("overwrite").value)
        self.zero_origin = bool(self.get_parameter("zero_origin").value)
        self.sync_tolerance_s = float(self.get_parameter("sync_tolerance_s").value)
        self.max_samples = int(self.get_parameter("max_samples").value)

        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if self.image_ext not in ("jpg", "jpeg", "png"):
            raise ValueError("image_ext must be one of: jpg, jpeg, png")
        self.jpeg_quality = max(1, min(100, self.jpeg_quality))

        self._prepare_output_dir()

        self.latest_image: Optional[PILImage.Image] = None
        self.latest_image_stamp: Tuple[int, int] = (-1, -1)
        self.latest_image_stamp_sec = -1.0

        self.latest_odom_xy: Optional[np.ndarray] = None
        self.latest_odom_yaw: Optional[float] = None
        self.latest_odom_stamp_sec = -1.0

        self.origin_xy: Optional[np.ndarray] = None
        self.positions = []
        self.yaws = []
        self.frame_index = 0
        self.last_saved_image_stamp: Tuple[int, int] = (-1, -1)
        self._warn_last_times = {}
        self._written = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, self.image_topic, self._image_cb, sensor_qos)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, sensor_qos)
        self.timer = self.create_timer(1.0 / self.sample_rate_hz, self._save_tick)

        self.get_logger().info(
            f"Collecting trajectory into '{self.output_dir}' "
            f"at {self.sample_rate_hz:.2f} Hz"
        )
        self.get_logger().info(
            f"Topics: image='{self.image_topic}', odom='{self.odom_topic}'"
        )

    def _prepare_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            return

        existing = os.listdir(self.output_dir)
        if not existing:
            return
        if not self.overwrite:
            raise RuntimeError(
                f"Output directory is not empty: {self.output_dir}. "
                "Set overwrite:=true to clear it."
            )

        for name in existing:
            path = os.path.join(self.output_dir, name)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)

    def _warn_throttled(self, key: str, message: str, period_s: float = 2.0):
        now = time.monotonic()
        last = self._warn_last_times.get(key, float("-inf"))
        if now - last >= period_s:
            self.get_logger().warn(message)
            self._warn_last_times[key] = now

    def _image_cb(self, msg: Image):
        try:
            self.latest_image = msg_to_pil(msg)
        except Exception as exc:  # pylint: disable=broad-except
            self._warn_throttled("img_decode", f"Image decode error: {exc}", 1.0)
            return

        stamp = msg.header.stamp
        self.latest_image_stamp = (int(stamp.sec), int(stamp.nanosec))
        self.latest_image_stamp_sec = stamp_to_sec(stamp)

    def _odom_cb(self, msg: Odometry):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.latest_odom_xy = np.array([position.x, position.y], dtype=np.float32)
        self.latest_odom_yaw = quat_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        self.latest_odom_stamp_sec = stamp_to_sec(msg.header.stamp)

    def _save_tick(self):
        if self.latest_image is None:
            self._warn_throttled("no_image", f"Waiting for image topic {self.image_topic}")
            return
        if self.latest_odom_xy is None or self.latest_odom_yaw is None:
            self._warn_throttled("no_odom", f"Waiting for odom topic {self.odom_topic}")
            return
        if self.latest_image_stamp == self.last_saved_image_stamp:
            return

        if (
            self.sync_tolerance_s > 0.0
            and self.latest_image_stamp_sec > 0.0
            and self.latest_odom_stamp_sec > 0.0
        ):
            skew_s = abs(self.latest_image_stamp_sec - self.latest_odom_stamp_sec)
            if skew_s > self.sync_tolerance_s:
                self._warn_throttled(
                    "skew",
                    (
                        f"Image/odom timestamp skew {skew_s:.3f}s exceeds "
                        f"sync_tolerance_s={self.sync_tolerance_s:.3f}s"
                    ),
                )
                return

        xy = self.latest_odom_xy.copy()
        if self.zero_origin:
            if self.origin_xy is None:
                self.origin_xy = xy.copy()
            xy = xy - self.origin_xy

        image_path = os.path.join(self.output_dir, f"{self.frame_index}.{self.image_ext}")
        if self.image_ext in ("jpg", "jpeg"):
            self.latest_image.save(image_path, quality=self.jpeg_quality)
        else:
            self.latest_image.save(image_path)

        self.positions.append([float(xy[0]), float(xy[1])])
        self.yaws.append(float(self.latest_odom_yaw))
        self.last_saved_image_stamp = self.latest_image_stamp

        self.frame_index += 1
        if self.frame_index == 1 or self.frame_index % 20 == 0:
            self.get_logger().info(f"Saved {self.frame_index} frames")

        if self.max_samples > 0 and self.frame_index >= self.max_samples:
            self.get_logger().info(
                f"Reached max_samples={self.max_samples}. Stopping collector."
            )
            rclpy.shutdown()

    def write_traj_data(self):
        if self._written:
            return
        self._written = True

        if self.positions:
            position_arr = np.asarray(self.positions, dtype=np.float32).reshape(-1, 2)
        else:
            position_arr = np.zeros((0, 2), dtype=np.float32)
        yaw_arr = np.asarray(self.yaws, dtype=np.float32)

        traj_data = {"position": position_arr, "yaw": yaw_arr}
        out_path = os.path.join(self.output_dir, "traj_data.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(traj_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.get_logger().info(
            f"Wrote {out_path} with {position_arr.shape[0]} synchronized samples"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CollectTrajectoryNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.write_traj_data()
            finally:
                node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
