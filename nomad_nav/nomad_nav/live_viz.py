"""
Live navigation visualization for NoMaD (ROS2 Humble).

Shows the current camera frame in a popup window and renders sampled action
trajectories in a bird's-eye inset for quick debugging during navigation.
"""

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

from nomad_nav.utils import msg_to_pil


def _to_bgr(msg: Image) -> np.ndarray:
    pil_img = msg_to_pil(msg)
    rgb = np.asarray(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class LiveVizNode(Node):
    def __init__(self):
        super().__init__("nomad_live_viz")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("sampled_actions_topic", "/sampled_actions")
        self.declare_parameter("waypoint_topic", "/waypoint")
        self.declare_parameter("goal_topic", "/topoplan/reached_goal")
        self.declare_parameter("num_samples", 8)
        self.declare_parameter("window_name", "NoMaD Live View")
        self.declare_parameter("panel_size", 260)
        self.declare_parameter("meters_forward", 3.0)
        self.declare_parameter("meters_lateral", 1.5)
        self.declare_parameter("refresh_hz", 12.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.sampled_actions_topic = str(self.get_parameter("sampled_actions_topic").value)
        self.waypoint_topic = str(self.get_parameter("waypoint_topic").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.num_samples = int(self.get_parameter("num_samples").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.panel_size = int(self.get_parameter("panel_size").value)
        self.meters_forward = float(self.get_parameter("meters_forward").value)
        self.meters_lateral = float(self.get_parameter("meters_lateral").value)
        refresh_hz = float(self.get_parameter("refresh_hz").value)

        self.window_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not self.window_enabled:
            self.get_logger().warn("No DISPLAY/WAYLAND_DISPLAY found. Live popup disabled.")

        self.latest_bgr: Optional[np.ndarray] = None
        self.latest_waypoint: Optional[np.ndarray] = None
        self.latest_trajs: Optional[np.ndarray] = None
        self.reached_goal = False
        self.window_created = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(Image, self.image_topic, self._image_cb, sensor_qos)
        self.create_subscription(
            Float32MultiArray, self.sampled_actions_topic, self._sampled_actions_cb, 10
        )
        self.create_subscription(Float32MultiArray, self.waypoint_topic, self._waypoint_cb, 10)
        self.create_subscription(Bool, self.goal_topic, self._goal_cb, 10)

        self.timer = self.create_timer(1.0 / refresh_hz, self._render_tick)
        self.get_logger().info(
            f"Live viz ready. image='{self.image_topic}', sampled_actions='{self.sampled_actions_topic}'"
        )

    def _image_cb(self, msg: Image):
        try:
            self.latest_bgr = _to_bgr(msg)
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"Image decode failed: {exc}")

    def _sampled_actions_cb(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size <= 1 or self.num_samples <= 0:
            self.latest_trajs = None
            return
        flat = data[1:]
        denom = 2 * self.num_samples
        if flat.size % denom != 0:
            self.get_logger().warn(
                f"sampled_actions size {flat.size} is incompatible with num_samples={self.num_samples}"
            )
            self.latest_trajs = None
            return
        horizon = flat.size // denom
        self.latest_trajs = flat.reshape(self.num_samples, horizon, 2)

    def _waypoint_cb(self, msg: Float32MultiArray):
        if msg.data:
            self.latest_waypoint = np.asarray(msg.data, dtype=np.float32)
        else:
            self.latest_waypoint = None

    def _goal_cb(self, msg: Bool):
        self.reached_goal = bool(msg.data)

    def _render_tick(self):
        if not self.window_enabled or self.latest_bgr is None:
            return

        frame = self.latest_bgr.copy()
        self._draw_action_overlay(frame)
        self._draw_status(frame)

        try:
            cv2.imshow(self.window_name, frame)
            self.window_created = True
            cv2.waitKey(1)
        except cv2.error as exc:
            self.window_enabled = False
            self.get_logger().error(f"Live popup disabled after OpenCV GUI failure: {exc}")

    def _draw_status(self, frame: np.ndarray):
        status = "GOAL REACHED" if self.reached_goal else "RUNNING"
        color = (0, 180, 0) if self.reached_goal else (0, 200, 255)
        cv2.putText(
            frame,
            status,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )
        if self.latest_waypoint is not None and self.latest_waypoint.size >= 2:
            dx = float(self.latest_waypoint[0])
            dy = float(self.latest_waypoint[1])
            cv2.putText(
                frame,
                f"waypoint dx={dx:.2f} dy={dy:.2f}",
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_action_overlay(self, frame: np.ndarray):
        size = self.panel_size
        margin = 20
        frame_h, frame_w = frame.shape[:2]
        panel_h = min(size, max(frame_h - 2 * margin, 80))
        panel_w = min(size, max(frame_w - 2 * margin, 80))
        y0 = max(margin, frame_h - panel_h - margin)
        x0 = max(margin, frame_w - panel_w - margin)

        roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w - 1, panel_h - 1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.35, roi, 0.65, 0.0, roi)
        cv2.rectangle(roi, (0, 0), (panel_w - 1, panel_h - 1), (210, 210, 210), 1)

        center_x = panel_w // 2
        robot_y = panel_h - 24
        px_per_meter_x = (panel_w - 40) / max(self.meters_lateral * 2.0, 1e-6)
        px_per_meter_y = (panel_h - 40) / max(self.meters_forward, 1e-6)

        cv2.line(roi, (center_x, panel_h - 10), (center_x, 10), (170, 170, 170), 1)
        cv2.line(roi, (10, robot_y), (panel_w - 10, robot_y), (170, 170, 170), 1)
        cv2.putText(roi, "actions", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        robot_pts = np.array(
            [[center_x, robot_y - 14], [center_x - 10, robot_y + 8], [center_x + 10, robot_y + 8]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(roi, robot_pts, (240, 240, 240))

        if self.latest_trajs is not None:
            for traj in self.latest_trajs:
                pixels = self._traj_to_pixels(traj, center_x, robot_y, px_per_meter_x, px_per_meter_y)
                if len(pixels) >= 2:
                    cv2.polylines(roi, [pixels], False, (0, 215, 255), 2, cv2.LINE_AA)

        if self.latest_waypoint is not None and self.latest_waypoint.size >= 2:
            point = self.latest_waypoint[:2].reshape(1, 2)
            pixels = self._traj_to_pixels(point, center_x, robot_y, px_per_meter_x, px_per_meter_y)
            if len(pixels) == 1:
                px, py = pixels[0]
                cv2.line(roi, (center_x, robot_y), (int(px), int(py)), (255, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(roi, (int(px), int(py)), 5, (255, 0, 0), -1)

    def _traj_to_pixels(
        self,
        traj: np.ndarray,
        center_x: int,
        robot_y: int,
        px_per_meter_x: float,
        px_per_meter_y: float,
    ) -> np.ndarray:
        pts = []
        for x, y in traj:
            px = int(round(center_x + float(y) * px_per_meter_x))
            py = int(round(robot_y - float(x) * px_per_meter_y))
            pts.append([px, py])
        return np.asarray(pts, dtype=np.int32)

    def destroy_node(self):
        try:
            if self.window_enabled and self.window_created:
                cv2.destroyWindow(self.window_name)
        except Exception:  # pylint: disable=broad-except
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LiveVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
