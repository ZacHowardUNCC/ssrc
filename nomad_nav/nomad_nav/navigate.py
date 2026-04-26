"""
NoMaD topomap navigation node (ROS2 Humble).

Loads a topomap, localizes via the distance head,
selects subgoals, and runs diffusion toward each.

Dataflow:
  /camera/image_raw -> vision_encoder -> diffusion -> /waypoint
"""

import os
import threading
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import yaml

# ROS2
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray, UInt8MultiArray

from nomad_nav.path_utils import (
    ensure_visualnav_python_paths,
    get_default_model_config_path,
    get_default_robot_config_path,
    get_default_topomap_images_dir,
)

ensure_visualnav_python_paths()

from nomad_nav.nwm_request import serialize_ranking_request
from nomad_nav.utils import msg_to_pil, to_numpy, transform_images, load_model
from nomad_nav.topic_names import (IMAGE_TOPIC,
                                    WAYPOINT_TOPIC,
                                    SAMPLED_ACTIONS_TOPIC,
                                    REACHED_GOAL_TOPIC,
                                    NWM_RANKING_REQUEST_TOPIC,
                                    NWM_RANKING_RESULT_TOPIC)
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
context_lock = threading.Lock()
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
NWM_REQUEST_REPUBLISH_SEC = 1.0
STATUS_LOG_PERIOD_SEC = 5.0

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    obs_img = msg_to_pil(msg)
    if context_size is not None:
        with context_lock:
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class NavigateNode(Node):
    """Minimal ROS2 node — pub/sub/timer only. All logic uses globals."""

    def __init__(self, args):
        super().__init__("nomad_navigate")
        self.args = args
        self.image_topic = getattr(self.args, "image_topic", IMAGE_TOPIC)
        self.nwm_request_topic = getattr(
            self.args, "nwm_request_topic", NWM_RANKING_REQUEST_TOPIC
        )
        self.nwm_result_topic = getattr(
            self.args, "nwm_result_topic", NWM_RANKING_RESULT_TOPIC
        )
        self.last_context_wait_log_time = 0.0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_sub = self.create_subscription(
            Image, self.image_topic, callback_obs, sensor_qos
        )
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, WAYPOINT_TOPIC, 1
        )
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1
        )
        self.nwm_request_pub = self.create_publisher(
            UInt8MultiArray, self.nwm_request_topic, 1
        )
        self.goal_pub = self.create_publisher(
            Bool, REACHED_GOAL_TOPIC, 1
        )

        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.nwm_result_group = MutuallyExclusiveCallbackGroup()
        self.pending_ranking_request_id: Optional[int] = None
        self.pending_ranking_best_index: Optional[int] = None
        self.pending_ranking_event = threading.Event()
        self.ranking_lock = threading.Lock()
        self.next_ranking_request_id = 1

        self.nwm_result_sub = self.create_subscription(
            Int32MultiArray,
            self.nwm_result_topic,
            self._nwm_ranking_result_cb,
            10,
            callback_group=self.nwm_result_group,
        )
        self.timer = self.create_timer(
            1.0 / RATE,
            self.timer_callback,
            callback_group=self.timer_group,
        )
        self.get_logger().info(
            "Navigate node ready. "
            f"image='{self.image_topic}', "
            f"nwm_request='{self.nwm_request_topic}', "
            f"nwm_result='{self.nwm_result_topic}', "
            f"enable_nwm_ranking={self.args.enable_nwm_ranking}. "
            "Waiting for image observations..."
        )

    def _nwm_ranking_result_cb(self, msg: Int32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn("Ignoring malformed NWM ranking result.")
            return

        request_id = int(msg.data[0])
        best_index = int(msg.data[1])
        with self.ranking_lock:
            if self.pending_ranking_request_id != request_id:
                return
            self.pending_ranking_best_index = best_index
            self.pending_ranking_event.set()
        self.get_logger().info(
            f"Received NWM ranking result {request_id} with best_index={best_index}."
        )

    def _wait_for_nwm_ranking_result(self, request_id: int, msg: UInt8MultiArray) -> bool:
        timeout_sec = float(self.args.nwm_ranking_timeout_sec)
        deadline = None if timeout_sec <= 0 else time.monotonic() + timeout_sec
        republish_count = 0
        last_status_log_time = time.monotonic()

        if deadline is None:
            self.get_logger().info(
                f"Waiting indefinitely for NWM ranking result {request_id}; "
                f"republishing every {NWM_REQUEST_REPUBLISH_SEC:.1f}s until a reply arrives."
            )
        else:
            self.get_logger().info(
                f"Waiting up to {timeout_sec:.3f}s for NWM ranking result {request_id}; "
                f"republishing every {NWM_REQUEST_REPUBLISH_SEC:.1f}s until timeout."
            )

        while rclpy.ok():
            wait_sec = NWM_REQUEST_REPUBLISH_SEC
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait_sec = min(wait_sec, remaining)

            if self.pending_ranking_event.wait(wait_sec):
                return True

            if deadline is not None and time.monotonic() >= deadline:
                return False

            self.nwm_request_pub.publish(msg)
            republish_count += 1

            now = time.monotonic()
            if republish_count == 1 or now - last_status_log_time >= STATUS_LOG_PERIOD_SEC:
                subscriber_count = self.nwm_request_pub.get_subscription_count()
                log_fn = self.get_logger().warn if subscriber_count == 0 else self.get_logger().info
                log_fn(
                    f"Still waiting for NWM ranking result {request_id}; "
                    f"republished {republish_count} time(s) on '{self.nwm_request_topic}' "
                    f"(subscribers={subscriber_count})."
                )
                last_status_log_time = now

        return False

    def _choose_action_index_with_nwm(
        self,
        context_images,
        goal_image,
        sampled_actions_metric: np.ndarray,
    ) -> int:
        request_id = self.next_ranking_request_id
        self.next_ranking_request_id += 1

        try:
            payload = serialize_ranking_request(
                request_id=request_id,
                context_images=context_images,
                goal_image=goal_image,
                sampled_actions=sampled_actions_metric,
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().warn(f"Failed to serialize NWM ranking request: {exc}")
            return 0

        msg = UInt8MultiArray()
        msg.data = np.frombuffer(payload, dtype=np.uint8).tolist()

        with self.ranking_lock:
            self.pending_ranking_request_id = request_id
            self.pending_ranking_best_index = None
            self.pending_ranking_event.clear()

        self.nwm_request_pub.publish(msg)
        subscriber_count = self.nwm_request_pub.get_subscription_count()
        log_fn = self.get_logger().warn if subscriber_count == 0 else self.get_logger().info
        log_fn(
            f"Published NWM ranking request {request_id} on '{self.nwm_request_topic}' "
            f"({sampled_actions_metric.shape[0]} samples, {len(context_images)} context frames, "
            f"{len(msg.data)} bytes, subscribers={subscriber_count})."
        )
        got_result = self._wait_for_nwm_ranking_result(request_id, msg)

        with self.ranking_lock:
            best_index = self.pending_ranking_best_index
            self.pending_ranking_request_id = None
            self.pending_ranking_best_index = None
            self.pending_ranking_event.clear()

        if not got_result or best_index is None:
            self.get_logger().warn(
                f"NWM ranking timed out after {self.args.nwm_ranking_timeout_sec:.3f}s. "
                "Falling back to sample 0."
            )
            return 0

        if best_index < 0 or best_index >= sampled_actions_metric.shape[0]:
            self.get_logger().warn(
                f"NWM returned invalid sample index {best_index}. Falling back to sample 0."
            )
            return 0

        return best_index

    def timer_callback(self):
        global closest_node, reached_goal

        # EXPLORATION MODE
        chosen_waypoint = np.zeros(4)
        with context_lock:
            local_context_queue = list(context_queue)

        required_context_frames = model_params["context_size"] + 1
        if len(local_context_queue) >= required_context_frames:
            if model_params["model_type"] == "nomad":
                obs_images = transform_images(
                    local_context_queue, model_params["image_size"], center_crop=False
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
                selected_goal_img = topomap[start + sg_idx]

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
                ranking_actions_metric = np.array(naction, copy=True)
                if model_params["normalize"]:
                    ranking_actions_metric[..., :2] *= MAX_V / RATE

                chosen_index = 0
                if self.args.enable_nwm_ranking and ranking_actions_metric.shape[0] > 1:
                    chosen_index = self._choose_action_index_with_nwm(
                        context_images=local_context_queue,
                        goal_image=selected_goal_img,
                        sampled_actions_metric=ranking_actions_metric,
                    )
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate(
                    (np.array([chosen_index], dtype=np.float32), naction.flatten())
                ).tolist()
                print("published sampled actions")
                self.sampled_actions_pub.publish(sampled_actions_msg)
                print(naction) # DEBUG
                naction = naction[chosen_index]
                print(naction) # DEBUG
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
                        local_context_queue, model_params["image_size"]
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
        else:
            now = time.monotonic()
            if now - self.last_context_wait_log_time >= STATUS_LOG_PERIOD_SEC:
                self.get_logger().info(
                    f"Waiting for images on '{self.image_topic}' before navigation can start "
                    f"({len(local_context_queue)}/{required_context_frames} context frames)."
                )
                self.last_context_wait_log_time = now

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
    parser.add_argument(
        "--enable-nwm-ranking",
        action="store_true",
        help="Request an external NWM ranker to choose the best sampled action.",
    )
    parser.add_argument(
        "--nwm-ranking-timeout-sec",
        default=-1.0,
        type=float,
        help="How long to wait for an NWM ranking reply before falling back to sample 0. Set <= 0 to wait forever.",
    )
    parser.add_argument(
        "--image-topic",
        default=IMAGE_TOPIC,
        type=str,
        help="Camera image topic.",
    )
    parser.add_argument(
        "--nwm-request-topic",
        default=NWM_RANKING_REQUEST_TOPIC,
        type=str,
        help="Topic used to publish serialized NWM ranking requests.",
    )
    parser.add_argument(
        "--nwm-result-topic",
        default=NWM_RANKING_RESULT_TOPIC,
        type=str,
        help="Topic used to receive [request_id, best_index] NWM ranking results.",
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
    tmp.declare_parameter("enable_nwm_ranking", False)
    tmp.declare_parameter("nwm_ranking_timeout_sec", -1.0)
    tmp.declare_parameter("image_topic", IMAGE_TOPIC)
    tmp.declare_parameter("nwm_request_topic", NWM_RANKING_REQUEST_TOPIC)
    tmp.declare_parameter("nwm_result_topic", NWM_RANKING_RESULT_TOPIC)

    args = argparse.Namespace(
        model=tmp.get_parameter("model").value,
        waypoint=tmp.get_parameter("waypoint_index").value,
        dir=tmp.get_parameter("topomap_dir").value,
        goal_node=tmp.get_parameter("goal_node").value,
        close_threshold=tmp.get_parameter("close_threshold").value,
        radius=tmp.get_parameter("radius").value,
        num_samples=tmp.get_parameter("num_samples").value,
        enable_nwm_ranking=_as_bool(tmp.get_parameter("enable_nwm_ranking").value),
        nwm_ranking_timeout_sec=float(tmp.get_parameter("nwm_ranking_timeout_sec").value),
        image_topic=str(tmp.get_parameter("image_topic").value),
        nwm_request_topic=str(tmp.get_parameter("nwm_request_topic").value),
        nwm_result_topic=str(tmp.get_parameter("nwm_result_topic").value),
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
          f"goal_node={args.goal_node}, waypoint={args.waypoint}, "
          f"enable_nwm_ranking={args.enable_nwm_ranking}, "
          f"nwm_ranking_timeout_sec={args.nwm_ranking_timeout_sec}, "
          f"image_topic={args.image_topic}, "
          f"nwm_request_topic={args.nwm_request_topic}, "
          f"nwm_result_topic={args.nwm_result_topic}")

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
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    print(f"Using {device}")
    main()
