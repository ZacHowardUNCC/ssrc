#!/usr/bin/env python3
"""
Launch file for robot hardware pipeline:
- CAN interface setup (can0)
- Scout Mini base (robot control)
- RealSense D435 camera driver

This file does NOT collect data. It only starts the hardware.
Data collection runs separately via collect_nomad_dataset.sh

NOTE: CAN setup requires passwordless sudo for ip commands.
Add to /etc/sudoers (run: sudo visudo):
  <user> ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can0 type can bitrate 500000
  <user> ALL=(ALL) NOPASSWD: /usr/sbin/ip link set can0 up
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scout_base_dir = FindPackageShare('scout_base')

    # Bring up CAN interface before starting the Scout base node
    can_setup = ExecuteProcess(
        cmd=['sudo', 'bash', '-c',
             'ip link set can0 type can bitrate 500000 && ip link set can0 up'],
        output='screen',
    )

    # Scout Mini base — delayed 1s to ensure CAN is up before the driver starts
    scout_mini = TimerAction(
        period=1.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([scout_base_dir, 'launch', 'scout_mini_base.launch.py'])
                ),
            ),
        ],
    )

    # RealSense camera driver
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py',
            ])
        ),
        launch_arguments={
            'camera_namespace': '/',
            'camera_name': 'camera',
            'enable_depth': 'false',
        }.items(),
    )

    return LaunchDescription([
        can_setup,
        scout_mini,
        realsense,
    ])
