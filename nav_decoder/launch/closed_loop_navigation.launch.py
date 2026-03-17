"""ROS2 Launch File for Closed-Loop UniVLA Navigation."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    args = [
        DeclareLaunchArgument('desktop_ip', default_value='10.107.157.5',
                              description='Desktop inference server IP'),
        DeclareLaunchArgument('desktop_port', default_value='5556',
                              description='Desktop inference server port'),
        DeclareLaunchArgument('camera_topic',
                              default_value='/camera/color/image_raw',
                              description='ROS2 camera image topic'),
        DeclareLaunchArgument('instruction',
                              default_value='Navigate forward and avoid obstacles',
                              description='Navigation instruction for UniVLA'),
        DeclareLaunchArgument('inference_rate_hz', default_value='2.0',
                              description='Inference request rate (Hz)'),
        DeclareLaunchArgument('jpeg_quality', default_value='80',
                              description='JPEG compression quality (0-100)'),
    ]

    navigation_client_node = Node(
        package='nav_decoder',
        executable='navigation_client',
        name='navigation_client',
        output='screen',
        parameters=[{
            'desktop_ip':       LaunchConfiguration('desktop_ip'),
            'desktop_port':     LaunchConfiguration('desktop_port'),
            'camera_topic':     LaunchConfiguration('camera_topic'),
            'instruction':      LaunchConfiguration('instruction'),
            'inference_rate_hz': LaunchConfiguration('inference_rate_hz'),
            'jpeg_quality':     LaunchConfiguration('jpeg_quality'),
        }],
    )

    nav_decoder_node = Node(
        package='nav_decoder',
        executable='nav_decoder_node',
        name='nav_decoder_node',
        output='screen',
    )

    return LaunchDescription(args + [navigation_client_node, nav_decoder_node])
