"""
ROS2 launch file for NoMaD navigation on Scout Mini.

Launches the navigate node, PD controller, and optionally a joystick teleop node.

Usage (topomap mode):
  ros2 launch nomad_nav nomad_navigate.launch.py topomap_dir:=my_route

Usage (goal image mode):
  ros2 launch nomad_nav nomad_navigate.launch.py goal_image_path:=/path/to/goal.png
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _iter_ancestors(path: str):
    current = os.path.abspath(path)
    while True:
        yield current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


def _is_visualnav_root(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "deployment", "config"))
        and os.path.isdir(os.path.join(path, "train"))
    )


def _find_visualnav_root() -> str:
    env_root = os.environ.get("VISUALNAV_ROOT", "").strip()
    if env_root:
        expanded = os.path.abspath(os.path.expanduser(env_root))
        if _is_visualnav_root(expanded):
            return expanded

    seen = set()
    for start in (os.path.dirname(os.path.abspath(__file__)), os.getcwd()):
        for ancestor in _iter_ancestors(start):
            for candidate in (
                ancestor,
                os.path.join(ancestor, "visualnav-transformer"),
                os.path.join(ancestor, "src", "visualnav-transformer"),
            ):
                candidate = os.path.abspath(candidate)
                if candidate in seen:
                    continue
                seen.add(candidate)
                if _is_visualnav_root(candidate):
                    return candidate

    return ""


def _default_deploy_path(*parts: str) -> str:
    root = _find_visualnav_root()
    if not root:
        return ""
    return os.path.join(root, "deployment", *parts)


_DEFAULT_ROBOT_CONFIG = _default_deploy_path("config", "robot.yaml")
_DEFAULT_MODEL_CONFIG = _default_deploy_path("config", "models.yaml")
_DEFAULT_JOY_CONFIG = _default_deploy_path("config", "joystick.yaml")
_DEFAULT_TOPOMAP_DIR = _default_deploy_path("topomaps", "images")


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument("model", default_value="nomad",
            description="Model name (must match key in models.yaml)"),
        DeclareLaunchArgument("goal_image_path", default_value="",
            description="Path to a single goal image. If set, topomap is not used."),
        DeclareLaunchArgument("topomap_dir", default_value="topomap",
            description="Subdirectory under topomaps/images/"),
        DeclareLaunchArgument("goal_node", default_value="-1",
            description="Goal node index (-1 = last node)"),
        DeclareLaunchArgument("waypoint_index", default_value="2",
            description="Which predicted waypoint to follow (0..len_traj_pred-1)"),
        DeclareLaunchArgument("close_threshold", default_value="3",
            description="Distance threshold for advancing to next topomap node"),
        DeclareLaunchArgument("radius", default_value="4",
            description="Number of topomap nodes to consider for localization"),
        DeclareLaunchArgument("num_samples", default_value="8",
            description="Number of diffusion action samples"),
        DeclareLaunchArgument("enable_nwm_ranking", default_value="false",
            description="Send sampled actions to an external NWM ranker before selecting one"),
        DeclareLaunchArgument("nwm_ranking_timeout_sec", default_value="-1.0",
            description="How long navigate waits for an NWM ranking reply before falling back to sample 0. Set <= 0 to wait forever"),
        DeclareLaunchArgument("nwm_request_topic", default_value="/nwm/ranking/request",
            description="Topic used to publish serialized NWM ranking requests"),
        DeclareLaunchArgument("nwm_result_topic", default_value="/nwm/ranking/result",
            description="Topic used to receive [request_id, best_index] NWM ranking results"),
        DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw",
            description="Camera image topic"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel",
            description="Velocity command topic for the base driver"),
        DeclareLaunchArgument("robot_config_path",
            default_value=_DEFAULT_ROBOT_CONFIG,
            description="Path to robot.yaml"),
        DeclareLaunchArgument("model_config_path",
            default_value=_DEFAULT_MODEL_CONFIG,
            description="Path to models.yaml"),
        DeclareLaunchArgument("joy_config_path",
            default_value=_DEFAULT_JOY_CONFIG,
            description="Path to joystick.yaml"),
        DeclareLaunchArgument("topomap_images_dir",
            default_value=_DEFAULT_TOPOMAP_DIR,
            description="Base directory for topomap images"),
        DeclareLaunchArgument("enable_live_viz", default_value="false",
            description="Open a live popup with camera feed and predicted actions"),
        DeclareLaunchArgument("viz_window_name", default_value="NoMaD Live View",
            description="Window title for the live visualization popup"),

        # Navigate node
        Node(
            package="nomad_nav",
            executable="navigate",
            name="nomad_navigate",
            output="screen",
            parameters=[{
                "model": LaunchConfiguration("model"),
                "goal_image_path": LaunchConfiguration("goal_image_path"),
                "waypoint_index": LaunchConfiguration("waypoint_index"),
                "topomap_dir": LaunchConfiguration("topomap_dir"),
                "goal_node": LaunchConfiguration("goal_node"),
                "close_threshold": LaunchConfiguration("close_threshold"),
                "radius": LaunchConfiguration("radius"),
                "num_samples": LaunchConfiguration("num_samples"),
                "enable_nwm_ranking": LaunchConfiguration("enable_nwm_ranking"),
                "nwm_ranking_timeout_sec": LaunchConfiguration("nwm_ranking_timeout_sec"),
                "nwm_request_topic": LaunchConfiguration("nwm_request_topic"),
                "nwm_result_topic": LaunchConfiguration("nwm_result_topic"),
                "image_topic": LaunchConfiguration("image_topic"),
                "robot_config_path": LaunchConfiguration("robot_config_path"),
                "model_config_path": LaunchConfiguration("model_config_path"),
                "topomap_images_dir": LaunchConfiguration("topomap_images_dir"),
            }],
        ),

        # PD controller node
        Node(
            package="nomad_nav",
            executable="pd_controller",
            name="nomad_pd_controller",
            output="screen",
            parameters=[{
                "robot_config_path": LaunchConfiguration("robot_config_path"),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
            }],
        ),

        # Joy teleop node (harmless if no joystick is connected)
        Node(
            package="nomad_nav",
            executable="joy_teleop",
            name="nomad_joy_teleop",
            output="screen",
            parameters=[{
                "joy_config_path": LaunchConfiguration("joy_config_path"),
                "cmd_vel_topic": "/cmd_vel_teleop",
            }],
        ),

        # Navigation diagnostic logger (always runs)
        Node(
            package="nomad_nav",
            executable="nav_logger",
            name="nomad_nav_logger",
            output="screen",
            parameters=[{
                "robot_config_path": LaunchConfiguration("robot_config_path"),
            }],
        ),

        Node(
            package="nomad_nav",
            executable="live_viz",
            name="nomad_live_viz",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_live_viz")),
            parameters=[{
                "image_topic": LaunchConfiguration("image_topic"),
                "num_samples": LaunchConfiguration("num_samples"),
                "window_name": LaunchConfiguration("viz_window_name"),
            }],
        ),
    ])
