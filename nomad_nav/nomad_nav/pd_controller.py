"""
PD controller node (ROS2 Humble).

Subscribes to /waypoint (Float32MultiArray) from the navigate node,
converts waypoints to Twist commands, and publishes to /cmd_vel.

Also subscribes to /topoplan/reached_goal to stop the robot when
the goal is reached.
"""

import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from nomad_nav.path_utils import get_default_robot_config_path
from nomad_nav.ros_data import ROSData


EPS = 1e-8
WAYPOINT_TIMEOUT = 5.0  # seconds (inference takes ~3s on CPU)
WAYPOINT_WARN_PERIOD_SEC = 5.0


def clip_angle(theta: float) -> float:
    """Clip angle to [-pi, pi]."""
    theta %= 2 * np.pi
    if -np.pi < theta < np.pi:
        return theta
    return theta - 2 * np.pi


def pd_control(waypoint: np.ndarray, dt: float, max_v: float, max_w: float):
    """Convert a waypoint [dx, dy] or [dx, dy, hx, hy] into (v, w)."""
    assert len(waypoint) in (2, 4), "waypoint must be 2D or 4D"
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint

    # Minimum forward distance for angular calculation.  When the
    # subgoal is closer than this, use MIN_DX so that the heading
    # correction stays proportional instead of blowing up.
    MIN_DX = 0.10  # metres

    if len(waypoint) == 4 and abs(dx) < EPS and abs(dy) < EPS:
        v = 0.0
        w = clip_angle(np.arctan2(hy, hx)) / dt
    elif abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / max(dx, MIN_DX)) / dt

    v = float(np.clip(v, 0.0, max_v))
    w = float(np.clip(w, -max_w, max_w))
    return v, w


class PDControllerNode(Node):
    def __init__(self):
        super().__init__("nomad_pd_controller")

        # -- Parameters --
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("rate", 9.0)

        robot_config_path = self.get_parameter("robot_config_path").value
        if not robot_config_path:
            robot_config_path = get_default_robot_config_path()
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        rate_hz = self.get_parameter("rate").value

        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]
        self.dt = 1.0 / robot_config["frame_rate"]

        # -- State --
        self.waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
        self.reached_goal = False
        self._goal_streak = 0        # consecutive goal=True messages received
        self._startup_time = time.monotonic()
        self._wp_arrival_time = 0.0  # monotonic time when last waypoint arrived
        self._last_v = 0.0           # cached linear velocity from last pd_control
        self._waypoint_stale = False
        self._waypoint_stale_since = 0.0
        self._last_waypoint_warn_time = 0.0

        # -- Subscribers --
        self.create_subscription(
            Float32MultiArray, "/waypoint", self._waypoint_callback, 1
        )
        self.create_subscription(
            Bool, "/topoplan/reached_goal", self._goal_callback, 1
        )

        # -- Publisher --
        self.vel_pub = self.create_publisher(Twist, cmd_vel_topic, 1)

        # -- Timer --
        self.timer = self.create_timer(1.0 / rate_hz, self._control_tick)
        self.get_logger().info(
            f"PD controller ready. Publishing to '{cmd_vel_topic}' at {rate_hz} Hz."
        )

    def _waypoint_callback(self, msg: Float32MultiArray):
        self.waypoint.set(list(msg.data))
        self._wp_arrival_time = time.monotonic()

    def _goal_callback(self, msg: Bool):
        # Require 3 consecutive goal=True messages before latching.
        # Each message arrives once per inference (~3s), so this means
        # the model must confirm goal reached 3 times in a row (~9s).
        if msg.data:
            self._goal_streak += 1
            if self._goal_streak >= 3 and not self.reached_goal:
                self.reached_goal = True
                self.get_logger().info("Reached goal. Publishing zero velocity.")
        else:
            self._goal_streak = 0

    def _control_tick(self):
        vel_msg = Twist()
        now = time.monotonic()

        if self.reached_goal:
            self.vel_pub.publish(vel_msg)
            return

        if self.waypoint.is_valid():
            if self._waypoint_stale:
                self.get_logger().info(
                    "Waypoint stream resumed after "
                    f"{now - self._waypoint_stale_since:.1f}s without updates."
                )
                self._waypoint_stale = False
                self._waypoint_stale_since = 0.0
                self._last_waypoint_warn_time = 0.0

            v, w = pd_control(
                np.array(self.waypoint.get()),
                self.dt,
                self.max_v,
                self.max_w,
            )
            self._last_v = v

            # w is a heading correction meant for one dt frame, but we
            # need multiple controller ticks to actually execute it.
            # Apply w for up to 1s (≈9 ticks at 9Hz), then drive
            # straight until the next waypoint arrives.
            age = time.monotonic() - self._wp_arrival_time
            if age > 1.0:
                w = 0.0

            vel_msg.linear.x = v
            vel_msg.angular.z = w
            self.get_logger().debug(f"cmd_vel: v={v:.3f}, w={w:.3f}")
        else:
            if self.waypoint.get() is None:
                stale_age = now - self._startup_time
                if stale_age < self.waypoint.timeout:
                    self.vel_pub.publish(vel_msg)
                    return
                reason = (
                    f"No /waypoint messages received yet after {stale_age:.1f}s. "
                    "Holding zero velocity while waiting for navigate."
                )
            else:
                stale_age = now - self._wp_arrival_time
                reason = (
                    f"No fresh /waypoint for {stale_age:.1f}s "
                    f"(timeout: {self.waypoint.timeout:.1f}s). "
                    "Holding zero velocity while waiting for navigate."
                )

            if (
                not self._waypoint_stale
                or now - self._last_waypoint_warn_time >= WAYPOINT_WARN_PERIOD_SEC
            ):
                if not self._waypoint_stale:
                    self._waypoint_stale_since = now
                self.get_logger().warn(reason)
                self._last_waypoint_warn_time = now
            self._waypoint_stale = True

        self.vel_pub.publish(vel_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
