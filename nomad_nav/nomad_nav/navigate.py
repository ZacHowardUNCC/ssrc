"""
NoMaD topomap navigation node (ROS2 Humble).

Loads a topomap, localizes via the distance head,
selects subgoals, and runs diffusion toward each.

Dataflow:
  /camera/image_raw -> vision_encoder -> diffusion -> /waypoint
"""

import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import yaml

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

from nomad_nav.path_utils import (
    ensure_visualnav_python_paths,
    get_default_model_config_path,
    get_default_robot_config_path,
    get_default_topomap_images_dir,
)

ensure_visualnav_python_paths()

from nomad_nav.utils import msg_to_pil, to_numpy, transform_images, load_model
from nomad_nav.topic_names import (IMAGE_TOPIC,
                                    WAYPOINT_TOPIC,
                                    SAMPLED_ACTIONS_TOPIC,
                                    REACHED_GOAL_TOPIC)
from vint_train.training.train_utils import get_action

from PIL import Image as PILImage
import argparse
import time


# CONSTANTS
TOPOMAP_IMAGES_DIR = get_default_topomap_images_dir()
MODEL_CONFIG_PATH = get_default_model_config_path()
ROBOT_CONFIG_PATH = get_default_robot_config_path()
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"]

# GLOBALS
context_queue = []
context_size = None
model = None
model_params = None
topomap = None
closest_node = 0
goal_node = 0
reached_goal = False
num_diffusion_iters = None
noise_scheduler = None
subgoal = []

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        if len(context_queue) < context_size + 1:
            context_queue.append(obs_img)
        else:
            context_queue.pop(0)
            context_queue.append(obs_img)


class NavigateNode(Node):
    """Minimal ROS2 node — pub/sub/timer only. All logic uses globals."""

    def __init__(self, args):
        super().__init__("nomad_navigate")
        self.args = args

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, callback_obs, sensor_qos
        )
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, 1
        )
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )
        self.goal_pub = self.create_publisher(
            Bool, REACHED_GOAL_TOPIC, 1
        )

        self.timer = self.create_timer(1.0 / RATE, self.timer_callback)
        self.get_logger().info(
            "Registered with master node. Waiting for image observations..."
        )

    def timer_callback(self):
        global closest_node, reached_goal

        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(
                    context_queue, model_params["image_size"], center_crop=False
                )
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1)
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)

                start = max(closest_node - self.args.radius, 0)
                end = min(closest_node + self.args.radius + 1, goal_node)
                goal_image = [
                    transform_images(
                        g_img, model_params["image_size"], center_crop=False
                    ).to(device)
                    for g_img in topomap[start : end + 1]
                ]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = model(
                    "vision_encoder",
                    obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                    goal_img=goal_image,
                    input_goal_mask=mask.repeat(len(goal_image)),
                )
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                sg_idx = min(
                    min_idx + int(dists[min_idx] < self.args.close_threshold),
                    len(obsgoal_cond) - 1,
                )
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)

                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (self.args.num_samples, model_params["len_traj_pred"], 2),
                        device=device,
                    )
                    naction = noisy_action

                    # init scheduler
                    noise_scheduler.set_timesteps(num_diffusion_iters)

                    start_time = time.time()
                    for k in noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = model(
                            "noise_pred_net",
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond,
                        )
                        # inverse diffusion step (remove noise)
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction,
                        ).prev_sample
                    print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate(
                    (np.array([0]), naction.flatten())
                ).tolist()
                print("published sampled actions")
                self.sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0]
                chosen_waypoint = naction[self.args.waypoint]
            else:
                start = max(closest_node - self.args.radius, 0)
                end = min(closest_node + self.args.radius + 1, goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(topomap[start : end + 1]):
                    transf_obs_img = transform_images(
                        context_queue, model_params["image_size"]
                    )
                    goal_data = transform_images(
                        sg_img, model_params["image_size"]
                    )
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)

                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                if distances[min_dist_idx] > self.args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][self.args.waypoint]
                    closest_node = start + min_dist_idx
                else:
                    chosen_waypoint = waypoints[
                        min(min_dist_idx + 1, len(waypoints) - 1)
                    ][self.args.waypoint]
                    closest_node = min(start + min_dist_idx + 1, goal_node)

        # RECOVERY MODE
        if model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint.tolist()
        self.waypoint_pub.publish(waypoint_msg)
        reached_goal = closest_node == goal_node
        goal_msg = Bool()
        goal_msg.data = bool(reached_goal)
        self.goal_pub.publish(goal_msg)
        if reached_goal:
            print("Reached goal! Stopping...")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,  # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    return parser


def _read_launch_params():
    """Read ROS2 launch parameters and return as argparse-compatible Namespace."""
    tmp = rclpy.create_node("nomad_navigate")
    tmp.declare_parameter("model", "nomad")
    tmp.declare_parameter("waypoint_index", 2)
    tmp.declare_parameter("topomap_dir", "topomap")
    tmp.declare_parameter("goal_node", -1)
    tmp.declare_parameter("close_threshold", 3)
    tmp.declare_parameter("radius", 4)
    tmp.declare_parameter("num_samples", 8)

    args = argparse.Namespace(
        model=tmp.get_parameter("model").value,
        waypoint=tmp.get_parameter("waypoint_index").value,
        dir=tmp.get_parameter("topomap_dir").value,
        goal_node=tmp.get_parameter("goal_node").value,
        close_threshold=tmp.get_parameter("close_threshold").value,
        radius=tmp.get_parameter("radius").value,
        num_samples=tmp.get_parameter("num_samples").value,
    )
    tmp.destroy_node()
    return args


def main(args=None):
    global context_size, model, model_params, topomap
    global closest_node, goal_node, reached_goal
    global num_diffusion_iters, noise_scheduler

    # Init ROS2 first so we can read launch parameters
    rclpy.init()

    if args is None:
        import sys
        # Strip ROS2-injected args (--ros-args ...) before parsing
        stripped = rclpy.utilities.remove_ros_args(sys.argv)
        cli_args = stripped[1:]
        if cli_args:
            # Direct CLI invocation: python navigate.py --dir test_route_001
            args = _build_arg_parser().parse_args(cli_args)
        else:
            # Launched via ros2 launch — read from ROS2 parameters
            args = _read_launch_params()

    print(f"Args: model={args.model}, dir={args.dir}, "
          f"goal_node={args.goal_node}, waypoint={args.waypoint}")

    # load model parameters
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    model_config_path = model_paths[args.model]["config_path"]
    if not os.path.isabs(model_config_path):
        model_config_path = os.path.normpath(
            os.path.join(os.path.dirname(MODEL_CONFIG_PATH), model_config_path)
        )
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    context_size = model_params["context_size"]

    # load model weights
    ckpt_path = model_paths[args.model]["ckpt_path"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(
            os.path.join(os.path.dirname(MODEL_CONFIG_PATH), ckpt_path)
        )
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
    model = load_model(
        ckpt_path,
        model_params,
        device,
    )
    model = model.to(device)
    model.eval()

    # load topomap
    topomap_filenames = sorted(
        os.listdir(os.path.join(TOPOMAP_IMAGES_DIR, args.dir)),
        key=lambda x: int(x.split(".")[0]),
    )
    topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    num_nodes = len(os.listdir(topomap_dir))
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"

    if args.goal_node == -1:
        goal_node = len(topomap) - 1
    else:
        goal_node = args.goal_node
    reached_goal = False

    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    # ROS2 — already initialized above
    node = NavigateNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    print(f"Using {device}")
    main()
