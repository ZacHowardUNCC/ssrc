from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="nav_decoder",
            executable="decoder_inference",
            name="decoder_inference_node",
            output="screen",
            parameters=[{
                "checkpoint_name": "decoder_nav_v1.pt",
                "odom_topic": "/odom",
                "latent_topic": "/vla/goal_embedding",
                "cmd_vel_topic": "/cmd_vel",
                "arm_topic": "/decoder/arm",
                "estop_topic": "/decoder/estop",
                "rate_hz": 10.0,
                "embed_dim": 4096,
                "num_actions": 4,
                "max_lin_vel": 0.2,
                "max_ang_vel": 0.3
            }]
        )
    ])
