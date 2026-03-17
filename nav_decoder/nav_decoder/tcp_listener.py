import socket
import json
import select
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class TCPActionListener(Node):
    def __init__(self):
        super().__init__("tcp_action_listener")

        # Declare parameters
        self.declare_parameter("port", 5555)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("arm_topic", "/decoder/arm")
        self.declare_parameter("estop_topic", "/decoder/estop")
        self.declare_parameter("max_lin_vel", 0.3)
        self.declare_parameter("max_ang_vel", 0.6)

        # Publishers
        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter("cmd_vel_topic").value,
            10
        )

        # State
        self.armed = False
        self.estop = False

        # Subscribers
        self.create_subscription(
            Bool,
            self.get_parameter("arm_topic").value,
            self.arm_cb,
            10
        )
        self.create_subscription(
            Bool,
            self.get_parameter("estop_topic").value,
            self.estop_cb,
            10
        )

        # TCP setup
        self.port = int(self.get_parameter("port").value)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(False)
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.listen(5)

        self.get_logger().info(f"TCP listener ready on port {self.port}")
        self.get_logger().info("Action mapping: 0=stop, 1=forward, 2=left, 3=right")
        self.get_logger().info("Waiting for /decoder/arm=true to accept commands")

        # Timer for polling connections
        self.create_timer(0.01, self.poll_socket)

    def arm_cb(self, msg):
        prev = self.armed
        self.armed = msg.data
        if prev != self.armed:
            self.get_logger().info(f"Armed: {self.armed}")
        if not self.armed:
            self.publish_stop()

    def estop_cb(self, msg):
        prev = self.estop
        self.estop = msg.data
        if prev != self.estop:
            self.get_logger().warn(f"E-STOP: {self.estop}")
        if self.estop:
            self.publish_stop()

    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def poll_socket(self):
        # Check if connection is ready
        readable, _, _ = select.select([self.sock], [], [], 0.0)
        if not readable:
            return

        try:
            conn, addr = self.sock.accept()
            conn.settimeout(1.0)
            
            # Receive data
            data = conn.recv(1024)
            conn.close()

            if not data:
                return

            # Parse message
            msg = json.loads(data.decode("utf-8"))
            action = int(msg["action"])

            self.get_logger().info(f"Received action {action} from {addr[0]}")

            # Safety check
            if not self.armed:
                self.get_logger().warn("Not armed - ignoring action")
                self.publish_stop()
                return

            if self.estop:
                self.get_logger().warn("E-STOP active - ignoring action")
                self.publish_stop()
                return

            # Execute action
            cmd = self.action_to_cmd(action)
            self.cmd_pub.publish(cmd)
            self.get_logger().info(f"Published: lin={cmd.linear.x:.2f}, ang={cmd.angular.z:.2f}")

        except socket.timeout:
            self.get_logger().warn("Connection timeout")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid JSON: {e}")
        except KeyError:
            self.get_logger().error("Missing 'action' field in JSON")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

    def action_to_cmd(self, action: int) -> Twist:
        """Convert discrete action to velocity command.
        
        Action space:
        0: stop
        1: forward
        2: turn left
        3: turn right
        """
        max_lin = float(self.get_parameter("max_lin_vel").value)
        max_ang = float(self.get_parameter("max_ang_vel").value)

        cmd = Twist()

        if action == 0:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif action == 1:
            cmd.linear.x = max_lin
            cmd.angular.z = 0.0
        elif action == 2:
            cmd.linear.x = 0.0
            cmd.angular.z = max_ang
        elif action == 3:
            cmd.linear.x = 0.0
            cmd.angular.z = -max_ang
        else:
            self.get_logger().warn(f"Unknown action {action}, stopping")
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def destroy_node(self):
        self.publish_stop()
        self.sock.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = TCPActionListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()