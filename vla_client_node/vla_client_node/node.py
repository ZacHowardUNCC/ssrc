import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
from PIL import Image as PILImage

import numpy as np
import requests
import time
import base64
import io


class VLAClientNode(Node):
    def __init__(self):
        super().__init__('vla_client_node')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('server_url', 'http://127.0.0.1:8000/predict')
        self.declare_parameter('max_linear_vel', 0.2)
        self.declare_parameter('max_angular_vel', 0.5)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('watchdog_timeout', 0.5)

        self.image_topic = self.get_parameter('image_topic').value
        self.server_url = self.get_parameter('server_url').value
        self.max_linear_vel = float(self.get_parameter('max_linear_vel').value)
        self.max_angular_vel = float(self.get_parameter('max_angular_vel').value)
        self.confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self.watchdog_timeout = float(self.get_parameter('watchdog_timeout').value)

        self.bridge = CvBridge()
        self.latest_image = None

        # Inference state
        self.last_action = None
        self.last_action_time = None
        self.action_history = []

        # ROS Functions
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.inference_timer = self.create_timer(0.5, self.inference_callback)
        self.control_timer = self.create_timer(0.05, self.control_callback)

        self.get_logger().info('vla_client_node started successfully')


    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image[:, :, ::-1]  # BGR -> RGB
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')


    def inference_callback(self):
        if self.latest_image is None:
            return

        try:
            pil_image = PILImage.fromarray(self.latest_image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')

            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            payload = {
                'image': image_base64,
                'instruction': 'Explore the environment safely and move through open space.',
                'action_history': self.action_history,
            }

            response = requests.post(
                self.server_url,
                json=payload,
                timeout=0.3,
            )

            if response.status_code != 200:
                self.get_logger().warn(f'Inference server returned {response.status_code}')
                return

            result = response.json()

            v = float(result.get('v', 0.0))
            w = float(result.get('w', 0.0))
            dt = float(result.get('dt', 0.0))
            confidence = float(result.get('confidence', 0.0))

            self.last_action = {
                'v': v,
                'w': w,
                'dt': dt,
                'confidence': confidence,
            }
            self.last_action_time = time.time()

            self.action_history.append({'v': v, 'w': w})
            if len(self.action_history) > 3:
                self.action_history.pop(0)

        except Exception as e:
            self.get_logger().warn(f'Inference failed: {e}')


    def control_callback(self):
        twist = Twist()

        # No action yet
        if self.last_action is None or self.last_action_time is None:
            self.cmd_vel_pub.publish(twist)
            return

        # Watchdog timeout
        if (time.time() - self.last_action_time) > self.watchdog_timeout:
            self.cmd_vel_pub.publish(twist)
            return

        # Confidence gating
        if self.last_action['confidence'] < self.confidence_threshold:
            self.cmd_vel_pub.publish(twist)
            return

        v = max(
            -self.max_linear_vel,
            min(self.max_linear_vel, self.last_action['v'])
        )
        w = max(
            -self.max_angular_vel,
            min(self.max_angular_vel, self.last_action['w'])
        )

        twist.linear.x = v
        twist.angular.z = w

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = VLAClientNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
