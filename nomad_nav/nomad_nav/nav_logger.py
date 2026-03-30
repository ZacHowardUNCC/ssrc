"""
Navigation diagnostic logger node (ROS2 Humble).

Logs a side-by-side timeline of:
  MODEL OUTPUT  →  CONTROLLER CMD  →  ROBOT ODOM

Each row in timeline.csv is triggered by a /waypoint message and
snapshots the latest cmd_vel and odom alongside it, so you can
see the full pipeline in one row.

A separate odometry.csv captures full-rate odom for trajectory
reconstruction.

Output:
  ~/ros2_ws/logs/nav_runs/run_NNN/
    timeline.csv   — one row per waypoint, all three stages
    odometry.csv   — full-rate odom for trajectory plots
    summary.txt    — human-readable run summary
"""

import csv
import math
import os
import re

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray

from nomad_nav.path_utils import get_default_robot_config_path


EPS = 1e-8

TIMELINE_COLUMNS = [
    "time (s)",
    # --- MODEL OUTPUT (navigate.py /waypoint) ---
    "wp_dx (m)",
    "wp_dy (m)",
    "expected_v (m/s)",
    "expected_w (rad/s)",
    # --- CONTROLLER CMD (pd_controller /cmd_vel) ---
    "cmd_v (m/s)",
    "cmd_w (rad/s)",
    # --- ROBOT ODOM (/odom) ---
    "odom_v (m/s)",
    "odom_w (rad/s)",
    "pos_x (m)",
    "pos_y (m)",
    "yaw (rad)",
    # --- DELTAS ---
    "v_error (m/s)",
    "w_error (rad/s)",
]

ODOM_COLUMNS = [
    "time (s)",
    "odom_v (m/s)",
    "odom_w (rad/s)",
    "pos_x (m)",
    "pos_y (m)",
    "yaw (rad)",
]


def _clip_angle(theta: float) -> float:
    """Clip angle to [-pi, pi]. Same as pd_controller.clip_angle."""
    theta %= 2 * math.pi
    if -math.pi < theta < math.pi:
        return theta
    return theta - 2 * math.pi


def _expected_velocity(waypoint_data, dt, max_v, max_w):
    """Replicate pd_controller.pd_control math to compute expected v, w."""
    n = len(waypoint_data)
    if n < 2:
        return 0.0, 0.0

    dx = waypoint_data[0]
    dy = waypoint_data[1]
    hx = waypoint_data[2] if n >= 4 else 0.0
    hy = waypoint_data[3] if n >= 4 else 0.0

    if n >= 4 and abs(dx) < EPS and abs(dy) < EPS:
        v = 0.0
        w = _clip_angle(math.atan2(hy, hx)) / dt
    elif abs(dx) < EPS:
        v = 0.0
        w = (1.0 if dy > 0 else -1.0) * math.pi / (2 * dt)
    else:
        v = dx / dt
        w = math.atan(dy / dx) / dt

    v = float(np.clip(v, 0.0, max_v))
    w = float(np.clip(w, -max_w, max_w))
    return v, w


def _yaw_from_quaternion(q):
    """Extract yaw from a geometry_msgs Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _next_run_dir(base_dir):
    """Find next run_NNN folder number."""
    os.makedirs(base_dir, exist_ok=True)
    max_n = 0
    for name in os.listdir(base_dir):
        m = re.match(r"^run_(\d+)$", name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _open_csv(path, columns):
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(columns)
    return f, writer


class NavLoggerNode(Node):
    def __init__(self):
        super().__init__("nomad_nav_logger")

        # Load robot config
        robot_config_path = get_default_robot_config_path()
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.max_v = robot_config["max_v"]
        self.max_w = robot_config["max_w"]
        self.dt = 1.0 / robot_config["frame_rate"]

        # Create run directory: run_001, run_002, ...
        base_dir = os.path.expanduser("~/ros2_ws/logs/nav_runs")
        run_num = _next_run_dir(base_dir)
        self.run_name = f"run_{run_num:03d}"
        self.run_dir = os.path.join(base_dir, self.run_name)
        os.makedirs(self.run_dir)

        # Open CSV files
        self.tl_file, self.tl_writer = _open_csv(
            os.path.join(self.run_dir, "timeline.csv"), TIMELINE_COLUMNS
        )
        self.odom_file, self.odom_writer = _open_csv(
            os.path.join(self.run_dir, "odometry.csv"), ODOM_COLUMNS
        )

        # Latest snapshot from each source (updated by callbacks)
        self.last_wp_dx = 0.0
        self.last_wp_dy = 0.0
        self.last_exp_v = 0.0
        self.last_exp_w = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.last_odom_v = 0.0
        self.last_odom_w = 0.0
        self.last_pos_x = 0.0
        self.last_pos_y = 0.0
        self.last_yaw = 0.0

        # Don't log until the first real waypoint arrives (skips zero preamble)
        self._logging_active = False

        # Stats
        self.waypoint_count = 0
        self.cmd_vel_count = 0
        self.odom_count = 0
        self.goal_reached = False
        self.v_errors = []
        self.w_errors = []
        self.wp_dxs = []
        self.wp_dys = []
        self.expected_vs = []
        self.expected_ws = []
        self.cmd_vs = []
        self.cmd_ws = []
        self.start_time = self.get_clock().now()

        # Subscribers
        self.create_subscription(
            Float32MultiArray, "/waypoint", self._waypoint_cb, 10
        )
        self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_cb, 10
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Odometry, "/odom", self._odom_cb, odom_qos
        )
        self.create_subscription(
            Bool, "/topoplan/reached_goal", self._goal_cb, 10
        )

        # Timeline rows are written on every cmd_vel arrival (not a timer),
        # so odom is always fresh (at most one odom period = ~20ms stale).

        print(f"NAV LOGGER: {self.run_name} -> {self.run_dir}/")

    def _elapsed(self):
        """Seconds since node start."""
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    # ── /waypoint: update latest, accumulate stats ──

    def _waypoint_cb(self, msg: Float32MultiArray):
        data = list(msg.data)
        dx = data[0] if len(data) >= 1 else 0.0
        dy = data[1] if len(data) >= 2 else 0.0

        exp_v, exp_w = _expected_velocity(
            data, self.dt, self.max_v, self.max_w
        )

        self.last_wp_dx = dx
        self.last_wp_dy = dy
        self.last_exp_v = exp_v
        self.last_exp_w = exp_w

        # Activate logging on first non-zero waypoint
        if not self._logging_active and (abs(dx) > 1e-6 or abs(dy) > 1e-6):
            self._logging_active = True
            self.start_time = self.get_clock().now()  # reset clock so t=0 is navigation start

        self.waypoint_count += 1
        self.wp_dxs.append(dx)
        self.wp_dys.append(dy)
        self.expected_vs.append(exp_v)
        self.expected_ws.append(exp_w)

    # ── /cmd_vel: update latest, write timeline row immediately ──

    def _cmd_vel_cb(self, msg: Twist):
        self.last_cmd_v = msg.linear.x
        self.last_cmd_w = msg.angular.z
        if not self._logging_active:
            return
        self.cmd_vel_count += 1
        self.cmd_vs.append(self.last_cmd_v)
        self.cmd_ws.append(self.last_cmd_w)
        # Write timeline row now — odom is at 50Hz so last_odom is at most ~20ms stale
        self._timeline_tick()

    # ── /odom: update latest, write full-rate file ──

    def _odom_cb(self, msg: Odometry):
        tw = msg.twist.twist
        pos = msg.pose.pose.position
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)

        self.last_odom_v = tw.linear.x
        self.last_odom_w = tw.angular.z
        self.last_pos_x = pos.x
        self.last_pos_y = pos.y
        self.last_yaw = yaw
        if not self._logging_active:
            return
        self.odom_count += 1

        self.odom_writer.writerow([
            f"{self._elapsed():.3f}",
            f"{tw.linear.x:.4f}",
            f"{tw.angular.z:.4f}",
            f"{pos.x:.4f}",
            f"{pos.y:.4f}",
            f"{yaw:.4f}",
        ])
        self.odom_file.flush()

    # ── /topoplan/reached_goal ──

    def _goal_cb(self, msg: Bool):
        if msg.data:
            self.goal_reached = True

    # ── write one timeline row (called on every cmd_vel) ──

    def _timeline_tick(self):
        v_err = self.last_exp_v - self.last_cmd_v
        w_err = self.last_exp_w - self.last_cmd_w

        self.tl_writer.writerow([
            f"{self._elapsed():.3f}",
            f"{self.last_wp_dx:.4f}",
            f"{self.last_wp_dy:.4f}",
            f"{self.last_exp_v:.4f}",
            f"{self.last_exp_w:.4f}",
            f"{self.last_cmd_v:.4f}",
            f"{self.last_cmd_w:.4f}",
            f"{self.last_odom_v:.4f}",
            f"{self.last_odom_w:.4f}",
            f"{self.last_pos_x:.4f}",
            f"{self.last_pos_y:.4f}",
            f"{self.last_yaw:.4f}",
            f"{v_err:.4f}",
            f"{w_err:.4f}",
        ])
        self.tl_file.flush()

        self.v_errors.append(v_err)
        self.w_errors.append(w_err)

    # ── Summary ──

    def _build_summary(self):
        elapsed = self._elapsed()

        lines = [
            "=" * 55,
            f"  NAV RUN SUMMARY — {self.run_name}",
            "=" * 55,
            f"  Duration:        {elapsed:.1f}s",
            f"  Waypoints:       {self.waypoint_count}",
            f"  CMD_VEL msgs:    {self.cmd_vel_count}",
            f"  Odom msgs:       {self.odom_count}",
            f"  Goal reached:    {self.goal_reached}",
            "",
        ]

        if self.waypoint_count == 0:
            lines += [
                "  !! NO WAYPOINTS RECEIVED !!",
                "  navigate node likely crashed — check its logs.",
                "",
            ]
        else:
            ve = np.array(self.v_errors)
            we = np.array(self.w_errors)
            lines += [
                "  EXPECTED vs ACTUAL CMD_VEL:",
                f"    v error — mean: {np.mean(ve):+.4f}  std: {np.std(ve):.4f}  max: {np.max(np.abs(ve)):.4f}",
                f"    w error — mean: {np.mean(we):+.4f}  std: {np.std(we):.4f}  max: {np.max(np.abs(we)):.4f}",
                "",
            ]

            dxs = np.array(self.wp_dxs)
            dys = np.array(self.wp_dys)
            lines += [
                "  WAYPOINT STATS:",
                f"    dx — mean: {np.mean(dxs):.4f}  std: {np.std(dxs):.4f}  range: [{np.min(dxs):.4f}, {np.max(dxs):.4f}]",
                f"    dy — mean: {np.mean(dys):.4f}  std: {np.std(dys):.4f}  range: [{np.min(dys):.4f}, {np.max(dys):.4f}]",
                "",
            ]

        lines += [
            f"  Log: {self.run_dir}/",
            "=" * 55,
        ]
        return "\n".join(lines)

    def destroy_node(self):
        summary = self._build_summary()

        summary_path = os.path.join(self.run_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary + "\n")

        print("\n" + summary)

        self.tl_file.close()
        self.odom_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
