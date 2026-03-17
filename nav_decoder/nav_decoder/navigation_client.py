#!/usr/bin/env python3
"""
Navigation Client (Jetson)
Captures images from camera, sends to Desktop for UniVLA inference,
receives discrete actions, and publishes them for robot control.

Wire protocol: 4-byte big-endian length prefix + JSON payload
  (matches image_inference_server.py on the Desktop side)
"""

import struct
import socket
import json
import base64
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge


class NavigationClient(Node):
    """ROS2 node: camera -> Desktop inference -> /discrete_action."""

    def __init__(self):
        super().__init__('navigation_client')

        # ---- Parameters ----
        self.declare_parameter('desktop_ip', '10.107.157.5')
        self.declare_parameter('desktop_port', 5556)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')
        self.declare_parameter('instruction', 'Navigate forward and avoid obstacles')
        self.declare_parameter('inference_rate_hz', 2.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('socket_timeout', 5.0)

        self.desktop_ip = self.get_parameter('desktop_ip').value
        self.desktop_port = self.get_parameter('desktop_port').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.instruction = self.get_parameter('instruction').value
        self.inference_rate = self.get_parameter('inference_rate_hz').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.max_retries = self.get_parameter('max_retries').value
        self.socket_timeout = self.get_parameter('socket_timeout').value

        self.get_logger().info('=' * 70)
        self.get_logger().info('Navigation Client Starting')
        self.get_logger().info(f'  Desktop:    {self.desktop_ip}:{self.desktop_port}')
        self.get_logger().info(f'  Camera:     {self.camera_topic}')
        self.get_logger().info(f'  Rate:       {self.inference_rate} Hz')
        self.get_logger().info(f'  Instruction: "{self.instruction}"')
        self.get_logger().info('=' * 70)

        # ---- CV Bridge ----
        self.bridge = CvBridge()

        # ---- State ----
        self.latest_image = None
        self.is_armed = False

        # ---- Publishers ----
        self.action_pub = self.create_publisher(Int32, '/discrete_action', 10)

        # ---- Subscribers ----
        self.image_sub = self.create_subscription(
            Image, self.camera_topic, self._image_cb, 10
        )
        self.arm_sub = self.create_subscription(
            Bool, '/decoder/arm', self._arm_cb, 10
        )

        # ---- Inference timer ----
        self.inference_timer = self.create_timer(
            1.0 / self.inference_rate, self._inference_loop
        )

        # ---- Statistics ----
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0

        self.get_logger().info('Ready. Waiting for camera images...')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def _arm_cb(self, msg: Bool):
        self.is_armed = msg.data
        state = 'Armed - navigation active' if msg.data else 'Disarmed - navigation paused'
        self.get_logger().info(state)

    # ------------------------------------------------------------------
    # Wire protocol helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recv_exactly(sock: socket.socket, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError('Connection closed')
            buf += chunk
        return buf

    @staticmethod
    def _send_msg(sock: socket.socket, data: bytes):
        sock.sendall(struct.pack('!I', len(data)) + data)

    @classmethod
    def _recv_msg(cls, sock: socket.socket) -> bytes:
        header = cls._recv_exactly(sock, 4)
        length = struct.unpack('!I', header)[0]
        return cls._recv_exactly(sock, length)

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _encode_jpeg_b64(self, image_rgb: np.ndarray) -> str:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            '.jpg', image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return base64.b64encode(encoded.tobytes()).decode('utf-8')

    # ------------------------------------------------------------------
    # Inference request
    # ------------------------------------------------------------------

    def _request_inference(self, image_rgb: np.ndarray) -> dict:
        """Send image to Desktop, return response dict (or dict with 'error')."""
        last_err = 'unknown'
        for attempt in range(1, self.max_retries + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.socket_timeout)
            try:
                sock.connect((self.desktop_ip, self.desktop_port))

                payload = json.dumps({
                    'image': self._encode_jpeg_b64(image_rgb),
                    'instruction': self.instruction,
                }).encode('utf-8')

                self._send_msg(sock, payload)
                raw = self._recv_msg(sock)
                return json.loads(raw.decode('utf-8'))

            except socket.timeout:
                last_err = 'timeout'
                self.get_logger().warning(
                    f'Timeout (attempt {attempt}/{self.max_retries})')
            except Exception as e:
                last_err = str(e)
                self.get_logger().warning(
                    f'Error (attempt {attempt}/{self.max_retries}): {e}')
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

        return {'error': f'Failed after {self.max_retries} retries: {last_err}'}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _publish_stop(self):
        msg = Int32()
        msg.data = 0
        self.action_pub.publish(msg)

    def _inference_loop(self):
        if self.latest_image is None:
            return

        if not self.is_armed:
            self._publish_stop()
            return

        t0 = time.time()
        response = self._request_inference(self.latest_image)
        latency = time.time() - t0

        self.total_requests += 1
        self.total_latency += latency

        if 'error' in response:
            self.get_logger().error(f'Inference error: {response["error"]}')
            self.failed_requests += 1
            self._publish_stop()
            return

        self.successful_requests += 1
        action = response['action']
        action_name = response.get('action_name', '?')
        confidence = response.get('confidence', 0.0)

        msg = Int32()
        msg.data = action
        self.action_pub.publish(msg)

        self.get_logger().info(
            f'Action: {action} ({action_name}) '
            f'[conf={confidence:.3f}] [latency={latency:.3f}s]'
        )

        if self.total_requests % 20 == 0:
            self._print_stats()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _print_stats(self):
        if self.total_requests == 0:
            return
        avg = self.total_latency / self.total_requests
        rate = (self.successful_requests / self.total_requests) * 100
        self.get_logger().info('=' * 70)
        self.get_logger().info(f'  Total:        {self.total_requests}')
        self.get_logger().info(f'  Success rate: {rate:.1f}%')
        self.get_logger().info(f'  Avg latency:  {avg:.3f}s')
        self.get_logger().info('=' * 70)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
        node._print_stats()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
