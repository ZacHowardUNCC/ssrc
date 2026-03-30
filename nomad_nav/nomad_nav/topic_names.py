# Topic names for ROS2 NoMaD navigation on Scout Mini.
# These are default values. All topics used by nodes are exposed as
# ROS2 parameters or launch arguments so they can be remapped without
# editing this file.

# Camera
IMAGE_TOPIC = "/camera/color/image_raw"

# Navigation internals (between navigate and pd_controller)
WAYPOINT_TOPIC = "/waypoint"
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"

# Robot base
ODOM_TOPIC = "/odom"
CMD_VEL_TOPIC = "/cmd_vel"

# Joystick teleop override
JOY_TOPIC = "/joy"
JOY_CMD_VEL_TOPIC = "/cmd_vel_teleop"
JOY_BUMPER_TOPIC = "/joy_bumper"
