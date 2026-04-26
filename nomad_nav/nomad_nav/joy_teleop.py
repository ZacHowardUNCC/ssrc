"""
Joystick teleop override node (ROS2 Humble).

When the deadman switch button is held, joystick axes are mapped to Twist
and published on a teleop velocity topic. This allows manual override of
autonomous navigation.

Assumption: The Scout Mini base driver accepts /cmd_vel directly. If you
use a twist_mux to prioritize teleop over nav, remap this node's output
topic to the mux teleop input.
"""

import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool

from nomad_nav.path_utils import get_default_joy_config_path, get_default_robot_config_path


class JoyTeleopNode(Node):
    def __init__(self):
        super().__init__("nomad_joy_teleop")

        # -- Parameters --
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("joy_config_path", "")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_teleop")
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("max_v", 0.4)
        self.declare_parameter("max_w", 0.8)
        self.declare_parameter("rate", 9.0)

        robot_config_path = self.get_parameter("robot_config_path").value
        if not robot_config_path:
            robot_config_path = get_default_robot_config_path()

        joy_config_path = self.get_parameter("joy_config_path").value
        if not joy_config_path:
            joy_config_path = get_default_joy_config_path()
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        joy_topic = self.get_parameter("joy_topic").value
        self.max_v = self.get_parameter("max_v").value
        self.max_w = self.get_parameter("max_w").value
        rate_hz = self.get_parameter("rate").value

        with open(joy_config_path, "r") as f:
            joy_config = yaml.safe_load(f)
        self.deadman_switch = joy_config["deadman_switch"]
        self.lin_vel_button = joy_config["lin_vel_button"]
        self.ang_vel_button = joy_config["ang_vel_button"]

        # -- State --
        self.vel_msg = Twist()
        self.deadman_held = False
        self.bumper = False

        # -- Sub / Pub --
        self.create_subscription(Joy, joy_topic, self._joy_callback, 10)
        self.vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.bumper_pub = self.create_publisher(Bool, "/joy_bumper", 10)

        self.timer = self.create_timer(1.0 / rate_hz, self._teleop_tick)
        self.get_logger().info(
            f"Joy teleop ready. Publishing to '{cmd_vel_topic}' when deadman held."
        )

    def _joy_callback(self, msg: Joy):
        if len(msg.buttons) <= self.deadman_switch:
            return
        self.deadman_held = bool(msg.buttons[self.deadman_switch])
        if self.deadman_held:
            self.vel_msg.linear.x = self.max_v * msg.axes[self.lin_vel_button]
            self.vel_msg.angular.z = self.max_w * msg.axes[self.ang_vel_button]
        else:
            self.vel_msg = Twist()

        # Bumper button is one index below the deadman switch
        bumper_idx = self.deadman_switch - 1
        if bumper_idx >= 0 and bumper_idx < len(msg.buttons):
            self.bumper = bool(msg.buttons[bumper_idx])
        else:
            self.bumper = False

    def _teleop_tick(self):
        if self.deadman_held:
            self.vel_pub.publish(self.vel_msg)
            self.get_logger().debug(
                f"Teleop: v={self.vel_msg.linear.x:.2f}, "
                f"w={self.vel_msg.angular.z:.2f}"
            )

        bumper_msg = Bool()
        bumper_msg.data = self.bumper
        self.bumper_pub.publish(bumper_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JoyTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
