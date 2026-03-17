"""
PD controller node (ROS2 Humble).

Subscribes to /waypoint (Float32MultiArray) from the navigate node,
converts waypoints to Twist commands, and publishes to /cmd_vel.

Also subscribes to /topoplan/reached_goal to stop the robot when
the goal is reached.
"""

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from nomad_nav.path_utils import get_default_robot_config_path
from nomad_nav.ros_data import ROSData


EPS = 1e-8
WAYPOINT_TIMEOUT = 1.0  # seconds


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

    if len(waypoint) == 4 and abs(dx) < EPS and abs(dy) < EPS:
        v = 0.0
        w = clip_angle(np.arctan2(hy, hx)) / dt
    elif abs(dx) < EPS:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt

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

        # -- Subscribers --
        self.create_subscription(
            Float32MultiArray, "/waypoint", self._waypoint_callback, 10
        )
        self.create_subscription(
            Bool, "/topoplan/reached_goal", self._goal_callback, 10
        )

        # -- Publisher --
        self.vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # -- Timer --
        self.timer = self.create_timer(1.0 / rate_hz, self._control_tick)
        self.get_logger().info(
            f"PD controller ready. Publishing to '{cmd_vel_topic}' at {rate_hz} Hz."
        )

    def _waypoint_callback(self, msg: Float32MultiArray):
        self.waypoint.set(list(msg.data))

    def _goal_callback(self, msg: Bool):
        self.reached_goal = msg.data

    def _control_tick(self):
        vel_msg = Twist()

        if self.reached_goal:
            self.vel_pub.publish(vel_msg)
            self.get_logger().info("Reached goal. Publishing zero velocity.")
            return

        if self.waypoint.is_valid(verbose=True):
            v, w = pd_control(
                np.array(self.waypoint.get()),
                self.dt,
                self.max_v,
                self.max_w,
            )
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            self.get_logger().debug(f"cmd_vel: v={v:.3f}, w={w:.3f}")

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
