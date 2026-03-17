import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Float32MultiArray
from nav_msgs.msg import Odometry


class AutoControllerNode(Node):
    def __init__(self):
        super().__init__("auto_controller_node")

        self.declare_parameter("instruction", "go forward")
        self.declare_parameter("start_delay_s", 2.0)
        self.declare_parameter("arm_after_embedding", True)
        self.declare_parameter("embedding_timeout_s", 2.0)
        self.declare_parameter("run_duration_s", 3.0)
        self.declare_parameter("auto_disarm", True)
        self.declare_parameter("set_estop_on_exit", False)

        self.declare_parameter("instruction_topic", "/vla/instruction")
        self.declare_parameter("arm_topic", "/decoder/arm")
        self.declare_parameter("estop_topic", "/decoder/estop")
        self.declare_parameter("embedding_topic", "/vla/goal_embedding")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("wait_for_odom", False)
        self.declare_parameter("odom_timeout_s", 2.0)

        self.instruction_pub = self.create_publisher(
            String, self.get_parameter("instruction_topic").value, 10
        )
        self.arm_pub = self.create_publisher(
            Bool, self.get_parameter("arm_topic").value, 10
        )
        self.estop_pub = self.create_publisher(
            Bool, self.get_parameter("estop_topic").value, 10
        )

        self.embedding_received = threading.Event()
        self.odom_received = threading.Event()

        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("embedding_topic").value,
            self.embedding_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self.odom_callback,
            10,
        )

        self.publish_estop(False)
        self.publish_arm(False)

        self._sequence_thread = threading.Thread(target=self.run_sequence, daemon=True)
        self._sequence_thread.start()

    def embedding_callback(self, _msg: Float32MultiArray):
        self.embedding_received.set()

    def odom_callback(self, _msg: Odometry):
        self.odom_received.set()

    def publish_arm(self, value: bool):
        msg = Bool()
        msg.data = bool(value)
        self.arm_pub.publish(msg)
        self.get_logger().info(f"/decoder/arm -> {msg.data}")

    def publish_estop(self, value: bool):
        msg = Bool()
        msg.data = bool(value)
        self.estop_pub.publish(msg)
        self.get_logger().info(f"/decoder/estop -> {msg.data}")

    def publish_instruction(self, text: str):
        msg = String()
        msg.data = text
        self.instruction_pub.publish(msg)
        self.get_logger().info(f"/vla/instruction -> {text}")

    def wait_for_odom(self):
        if not bool(self.get_parameter("wait_for_odom").value):
            return

        timeout = float(self.get_parameter("odom_timeout_s").value)
        if not self.odom_received.wait(timeout=timeout):
            self.get_logger().warn(
                f"/odom not received within {timeout:.2f}s; continuing anyway"
            )
        else:
            self.get_logger().info("/odom received")

    def wait_for_embedding(self):
        timeout = float(self.get_parameter("embedding_timeout_s").value)
        if not self.embedding_received.wait(timeout=timeout):
            self.get_logger().warn(
                f"/vla/goal_embedding not received within {timeout:.2f}s; continuing anyway"
            )
        else:
            self.get_logger().info("/vla/goal_embedding received")

    def run_sequence(self):
        try:
            self.wait_for_odom()

            start_delay = float(self.get_parameter("start_delay_s").value)
            if start_delay > 0.0:
                time.sleep(start_delay)

            instruction = str(self.get_parameter("instruction").value)
            self.publish_instruction(instruction)

            if bool(self.get_parameter("arm_after_embedding").value):
                self.wait_for_embedding()

            self.publish_arm(True)

            run_duration = float(self.get_parameter("run_duration_s").value)
            if run_duration > 0.0:
                time.sleep(run_duration)

            if bool(self.get_parameter("auto_disarm").value):
                self.publish_arm(False)

            if bool(self.get_parameter("set_estop_on_exit").value):
                self.publish_estop(True)
        finally:
            self.publish_arm(False)
            rclpy.shutdown()


def main():
    rclpy.init()
    node = AutoControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
