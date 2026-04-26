"""
Topomap creation node (ROS2 Humble).

Subscribes to a camera topic and periodically saves images to disk.
These images form the topomap used by the navigate node for
goal-conditioned navigation.

Usage:
  ros2 run nomad_nav create_topomap --ros-args \
      -p dir:=my_topomap -p dt:=1.0 -p image_topic:=/camera/image_raw
"""

import os
import shutil
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

from nomad_nav.path_utils import get_default_topomap_images_dir
from nomad_nav.utils import msg_to_pil


class CreateTopomapNode(Node):
    def __init__(self):
        super().__init__("nomad_create_topomap")

        self.declare_parameter("dir", "topomap")
        self.declare_parameter("dt", 1.0)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("topomap_images_dir", "")

        topomap_subdir = self.get_parameter("dir").value
        self.dt = self.get_parameter("dt").value
        image_topic = self.get_parameter("image_topic").value
        topomap_base = self.get_parameter("topomap_images_dir").value
        if not topomap_base:
            topomap_base = get_default_topomap_images_dir()

        assert self.dt > 0, "dt must be positive"

        self.save_dir = os.path.join(topomap_base, topomap_subdir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            self.get_logger().warn(
                f"{self.save_dir} already exists. Removing previous images."
            )
            for f in os.listdir(self.save_dir):
                fp = os.path.join(self.save_dir, f)
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)

        self.obs_img = None
        self.frame_index = 0
        self.last_save_time = float("inf")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, image_topic, self._image_callback, sensor_qos)

        self.timer = self.create_timer(self.dt, self._save_tick)
        self.get_logger().info(
            f"Topomap creator ready. Saving to {self.save_dir} every {self.dt}s."
        )

    def _image_callback(self, msg: Image):
        self.obs_img = msg_to_pil(msg)

    def _save_tick(self):
        if self.obs_img is not None:
            path = os.path.join(self.save_dir, f"{self.frame_index}.png")
            self.obs_img.save(path)
            self.get_logger().info(f"Saved image {self.frame_index}")
            self.frame_index += 1
            self.last_save_time = time.monotonic()
            self.obs_img = None
        else:
            elapsed = time.monotonic() - self.last_save_time
            if self.last_save_time != float("inf") and elapsed > 2 * self.dt:
                self.get_logger().warn(
                    "Camera topic stopped publishing. Shutting down."
                )
                raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = CreateTopomapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
