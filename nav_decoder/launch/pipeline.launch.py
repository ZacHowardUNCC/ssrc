from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_auto_controller = LaunchConfiguration("enable_auto_controller")
    enable_keyboard_control = LaunchConfiguration("enable_keyboard_control")

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_auto_controller",
            default_value="true",
            description="Start headless automation controller",
        ),
        DeclareLaunchArgument(
            "enable_keyboard_control",
            default_value="false",
            description="Start keyboard controller (TTY required)",
        ),
        Node(
            package="vla_client_node",
            executable="instruction_to_embedding",
            name="instruction_to_embedding_node",
            output="screen",
            parameters=[{
                "instruction_topic": "/vla/instruction",
                "embedding_topic": "/vla/goal_embedding",
                "embedding_dim": 4096,
            }],
        ),
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
                "stub_visual": True,
                "visual_T": 1,
                "max_lin_vel": 0.2,
                "max_ang_vel": 0.3,
            }],
        ),
        Node(
            package="nav_decoder",
            executable="auto_controller",
            name="auto_controller_node",
            output="screen",
            condition=IfCondition(enable_auto_controller),
            parameters=[{
                "instruction": "go forward",
                "start_delay_s": 2.0,
                "arm_after_embedding": True,
                "embedding_timeout_s": 2.0,
                "run_duration_s": 3.0,
                "auto_disarm": True,
                "set_estop_on_exit": False,
                "instruction_topic": "/vla/instruction",
                "arm_topic": "/decoder/arm",
                "estop_topic": "/decoder/estop",
                "embedding_topic": "/vla/goal_embedding",
                "odom_topic": "/odom",
                "wait_for_odom": False,
                "odom_timeout_s": 2.0,
            }],
        ),
        Node(
            package="nav_decoder",
            executable="keyboard_control",
            name="keyboard_control_node",
            output="screen",
            condition=IfCondition(enable_keyboard_control),
            parameters=[{
                "arm_topic": "/decoder/arm",
                "estop_topic": "/decoder/estop",
                "repeat_rate_hz": 2.0,
            }],
        ),
    ])
