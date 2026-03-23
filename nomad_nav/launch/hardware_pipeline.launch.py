#!/usr/bin/env python3
"""
Launch file for robot hardware pipeline:
- Scout base (robot control)
- RealSense D435 camera driver

This file does NOT collect data. It only starts the hardware.
Data collection runs separately via collect_nomad_dataset.sh
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get package directories
    scout_base_dir = FindPackageShare('scout_base')

    return LaunchDescription([
        # Scout base launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([scout_base_dir, 'launch', 'scout_base.launch.py'])
            ),
        ),

        # RealSense camera driver
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ),
            launch_arguments={
                'camera_namespace': '/',
                'camera_name': 'camera',
            }.items(),
        ),

        # Live camera visualization
        Node(
            package='nomad_nav',
            executable='live_viz',
            name='nomad_live_viz',
            output='screen',
            parameters=[{
                'image_topic': '/camera/color/image_raw',
            }],
        ),
    ])
