import os
import numpy as np

import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import Twist


class SimpleNavDecoder(nn.Module):
    def __init__(self, dim=4096, num_classes=4):
        super().__init__()
        self.vis_proj = nn.Linear(dim, 512)
        self.lat_proj = nn.Linear(dim, 512)
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, visual_embed, latent_action):
        vis = visual_embed.mean(dim=1)
        lat = latent_action.mean(dim=1)
        vis = self.vis_proj(vis)
        lat = self.lat_proj(lat)
        x = torch.cat([vis, lat], dim=-1)
        return self.mlp(x)


class DecoderInferenceNode(Node):
    def __init__(self):
        super().__init__("decoder_inference_node")

        self.declare_parameter("checkpoint_name", "decoder_nav_v1.pt")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("latent_topic", "/vla/goal_embedding")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("arm_topic", "/decoder/arm")
        self.declare_parameter("estop_topic", "/decoder/estop")

        self.declare_parameter("num_actions", 4)
        self.declare_parameter("embed_dim", 4096)
        self.declare_parameter("stub_visual", True)
        self.declare_parameter("visual_T", 1)
        self.declare_parameter("rate_hz", 10.0)

        self.declare_parameter("max_lin_vel", 0.2)
        self.declare_parameter("max_ang_vel", 0.3)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.latest_state = None
        self.latent_vec = None
        self.armed = False
        self.estop_active = False

        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter("cmd_vel_topic").value,
            10,
        )

        self.create_subscription(
            Odometry,
            self.get_parameter("odom_topic").value,
            self.odom_callback,
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            self.get_parameter("latent_topic").value,
            self.latent_callback,
            10,
        )

        self.create_subscription(
            Bool,
            self.get_parameter("arm_topic").value,
            self.arm_callback,
            10,
        )

        self.create_subscription(
            Bool,
            self.get_parameter("estop_topic").value,
            self.estop_callback,
            10,
        )

        self.model = self.load_model()

        rate_hz = float(self.get_parameter("rate_hz").value)
        self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)

        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info("Waiting for /odom and /vla/goal_embedding")
        self.get_logger().info("Robot will not move until /decoder/arm is true")
        self.get_logger().info("Emergency stop topic: /decoder/estop (true = stop)")

    def load_model(self):
        pkg_share = get_package_share_directory("nav_decoder")
        ckpt_name = self.get_parameter("checkpoint_name").value
        ckpt_path = os.path.join(pkg_share, "weights", ckpt_name)

        self.get_logger().info(f"Loading checkpoint: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        num_actions = int(self.get_parameter("num_actions").value)
        embed_dim = int(self.get_parameter("embed_dim").value)

        model = SimpleNavDecoder(dim=embed_dim, num_classes=num_actions).to(self.device)

        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.get_logger().info("Decoder loaded successfully")
        return model

    def odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        wz = msg.twist.twist.angular.z
        self.latest_state = np.array([vx, wz], dtype=np.float32)

    def latent_callback(self, msg: Float32MultiArray):
        self.latent_vec = np.array(msg.data, dtype=np.float32)

    def arm_callback(self, msg: Bool):
        self.armed = bool(msg.data)
        self.get_logger().info(f"Armed: {self.armed}")
        if not self.armed:
            self.publish_stop()

    def estop_callback(self, msg: Bool):
        self.estop_active = bool(msg.data)
        self.get_logger().info(f"E-Stop: {self.estop_active}")
        if self.estop_active:
            self.publish_stop()

    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def action_to_cmd(self, action: int) -> Twist:
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
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        return cmd

    def control_loop(self):
        if self.estop_active:
            self.publish_stop()
            return

        if not self.armed:
            return

        if self.latest_state is None:
            return

        if self.latent_vec is None:
            return

        embed_dim = int(self.get_parameter("embed_dim").value)
        if self.latent_vec.shape[0] != embed_dim:
            self.get_logger().error(
                f"Latent dim mismatch. Got {self.latent_vec.shape[0]} expected {embed_dim}"
            )
            self.publish_stop()
            return

        T = int(self.get_parameter("visual_T").value)

        if bool(self.get_parameter("stub_visual").value):
            visual_embed = torch.zeros((1, T, embed_dim), device=self.device, dtype=torch.float32)
        else:
            self.get_logger().error("stub_visual is false but no visual embedding source is wired yet")
            self.publish_stop()
            return

        latent_action = torch.from_numpy(self.latent_vec).to(self.device).float().view(1, 1, embed_dim)

        with torch.no_grad():
            logits = self.model(visual_embed, latent_action)
            action = 1

        cmd = self.action_to_cmd(action)
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = DecoderInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
