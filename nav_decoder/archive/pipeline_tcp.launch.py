from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_keyboard = LaunchConfiguration("enable_keyboard")

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_keyboard",
            default_value="true",
            description="Enable keyboard control for arm/estop",
        ),
        
        # TCP action listener - receives actions from cluster
        Node(
            package="nav_decoder",
            executable="tcp_action_listener",
            name="tcp_action_listener",
            output="screen",
            parameters=[{
                "port": 5555,
                "arm_topic": "/decoder/arm",
                "estop_topic": "/decoder/estop",
                "cmd_vel_topic": "/cmd_vel",
            }],
        ),
        
        # Optional keyboard control for safety
        Node(
            package="nav_decoder",
            executable="keyboard_control",
            name="keyboard_control_node",
            output="screen",
            condition=IfCondition(enable_keyboard),
            parameters=[{
                "arm_topic": "/decoder/arm",
                "estop_topic": "/decoder/estop",
                "repeat_rate_hz": 2.0,
            }],
        ),
    ])