import sys
import termios
import threading
import tty
from select import select

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class KeyboardControlNode(Node):
    def __init__(self):
        super().__init__("keyboard_control_node")

        self.declare_parameter("arm_topic", "/decoder/arm")
        self.declare_parameter("estop_topic", "/decoder/estop")
        self.declare_parameter("repeat_rate_hz", 2.0)

        self.arm_topic = self.get_parameter("arm_topic").value
        self.estop_topic = self.get_parameter("estop_topic").value

        self.arm_pub = self.create_publisher(Bool, self.arm_topic, 10)
        self.estop_pub = self.create_publisher(Bool, self.estop_topic, 10)

        self.armed = False
        self.estop_active = False
        self._running = True
        self._stdin_fd = None
        self._stdin_state = None

        self._print_help()

        if sys.stdin.isatty():
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_state = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self._thread.start()
        else:
            self.get_logger().error("stdin is not a TTY; keyboard control disabled")

        rate_hz = float(self.get_parameter("repeat_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate_hz, self.publish_state)

        self.publish_state()

    def _print_help(self):
        self.get_logger().info("Keyboard controls:")
        self.get_logger().info("  a: arm decoder")
        self.get_logger().info("  d: disarm decoder")
        self.get_logger().info("  e: toggle emergency stop")
        self.get_logger().info("  q: exit")

    def publish_state(self):
        arm_msg = Bool()
        arm_msg.data = self.armed
        self.arm_pub.publish(arm_msg)

        estop_msg = Bool()
        estop_msg.data = self.estop_active
        self.estop_pub.publish(estop_msg)

    def _read_key(self, timeout_s: float):
        if self._stdin_fd is None:
            return None
        rlist, _, _ = select([self._stdin_fd], [], [], timeout_s)
        if rlist:
            return sys.stdin.read(1)
        return None

    def _keyboard_loop(self):
        while rclpy.ok() and self._running:
            key = self._read_key(0.1)
            if not key:
                continue

            key = key.lower()
            if key == "a":
                self.armed = True
                self.get_logger().info("Arm -> true")
                self.publish_state()
            elif key == "d":
                self.armed = False
                self.get_logger().info("Arm -> false")
                self.publish_state()
            elif key == "e":
                self.estop_active = not self.estop_active
                self.get_logger().info(f"E-Stop -> {self.estop_active}")
                self.publish_state()
            elif key == "q":
                self.get_logger().info("Exit requested")
                self.armed = False
                self.estop_active = True
                self.publish_state()
                self._running = False
                rclpy.shutdown()

    def destroy_node(self):
        self._running = False
        if self._stdin_fd is not None and self._stdin_state is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_state)
        super().destroy_node()


def main():
    rclpy.init()
    node = KeyboardControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
