"""
NoMaD topomap navigation node (ROS2 Humble).

Loads a topomap, localizes via the distance head,
selects subgoals, and runs diffusion toward each.

Dataflow:
  /camera/image_raw -> vision_encoder -> diffusion -> /waypoint
"""

import os
import time
import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

from nomad_nav.path_utils import (
    ensure_visualnav_python_paths,
    find_visualnav_root,
    get_default_model_config_path,
    get_default_robot_config_path,
    get_default_topomap_images_dir,
)

ensure_visualnav_python_paths()

from nomad_nav.utils import msg_to_pil, to_numpy, transform_images, load_model


def _load_action_stats() -> dict:
    data_config_path = os.path.join(
        find_visualnav_root(), "train", "vint_train", "data", "data_config.yaml"
    )
    with open(data_config_path, "r") as f:
        data_config = yaml.safe_load(f)
    action_stats = data_config["action_stats"]
    return {
        "min": np.array(action_stats["min"]),
        "max": np.array(action_stats["max"]),
    }


def _diffusion_output_to_actions(diffusion_output: torch.Tensor, action_stats: dict) -> np.ndarray:
    ndeltas = diffusion_output.reshape(diffusion_output.shape[0], -1, 2)
    ndeltas_np = to_numpy(ndeltas)
    deltas = (ndeltas_np + 1.0) / 2.0
    deltas = deltas * (action_stats["max"] - action_stats["min"]) + action_stats["min"]
    return np.cumsum(deltas, axis=1)


class NavigateNode(Node):
    def __init__(self):
        super().__init__("nomad_navigate")

        # Declare parameters
        self.declare_parameter("model", "nomad")
        self.declare_parameter("waypoint_index", 2)
        self.declare_parameter("num_samples", 8)
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("model_config_path", "")
        # Topomap mode parameters
        self.declare_parameter("topomap_dir", "topomap")
        self.declare_parameter("topomap_images_dir", "")
        self.declare_parameter("goal_node", -1)
        self.declare_parameter("close_threshold", 3)
        self.declare_parameter("radius", 4)

        # Read parameters
        self.model_name = self.get_parameter("model").value
        self.waypoint_index = self.get_parameter("waypoint_index").value
        self.num_samples = self.get_parameter("num_samples").value
        image_topic = self.get_parameter("image_topic").value
        robot_config_path = self.get_parameter("robot_config_path").value
        model_config_path = self.get_parameter("model_config_path").value
        self.close_threshold = self.get_parameter("close_threshold").value
        self.radius = self.get_parameter("radius").value

        if not robot_config_path:
            robot_config_path = get_default_robot_config_path()
        if not model_config_path:
            model_config_path = get_default_model_config_path()

        # Load robot config
        with open(robot_config_path, "r") as f:
            robot_config = yaml.safe_load(f)
        self.max_v = robot_config["max_v"]
        self.rate_hz = robot_config["frame_rate"]

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        with open(model_config_path, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_file = model_paths[self.model_name]["config_path"]
        if not os.path.isabs(model_config_file):
            model_config_file = os.path.normpath(
                os.path.join(os.path.dirname(model_config_path), model_config_file)
            )
        with open(model_config_file, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]
        self.action_stats = _load_action_stats()

        ckpt_path = model_paths[self.model_name]["ckpt_path"]
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.normpath(
                os.path.join(os.path.dirname(model_config_path), ckpt_path)
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model weights not found at {ckpt_path}")
        self.get_logger().info(f"Loading model from {ckpt_path}")

        self.model = load_model(ckpt_path, self.model_params, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Diffusion scheduler (NoMaD)
        self.noise_scheduler = None
        self.num_diffusion_iters = None
        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon",
            )

        # Topomap state
        self.topomap = None
        self.closest_node = 0
        self.goal_node = 0
        self.reached_goal = False

        self._init_topomap_mode()

        # Context queue for observation images
        self.context_queue = []

        # ROS2 pub/sub
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, sensor_qos
        )
        self.waypoint_pub = self.create_publisher(Float32MultiArray, "/waypoint", 10)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, "/sampled_actions", 10
        )
        self.goal_pub = self.create_publisher(Bool, "/topoplan/reached_goal", 10)

        self.timer = self.create_timer(1.0 / self.rate_hz, self._navigate_tick)
        self.get_logger().info(
            f"Navigate node ready. Waiting for images on '{image_topic}'"
        )

    def _init_topomap_mode(self):
        """Load a topomap directory for multi-node navigation."""
        topomap_images_dir = self.get_parameter("topomap_images_dir").value
        if not topomap_images_dir:
            topomap_images_dir = get_default_topomap_images_dir()
        topomap_subdir = self.get_parameter("topomap_dir").value
        topomap_full_dir = os.path.join(topomap_images_dir, topomap_subdir)
        if not os.path.isdir(topomap_full_dir):
            raise FileNotFoundError(
                f"Topomap directory not found: {topomap_full_dir}"
            )
        filenames = sorted(
            os.listdir(topomap_full_dir), key=lambda x: int(x.split(".")[0])
        )
        self.topomap = [
            PILImage.open(os.path.join(topomap_full_dir, f)) for f in filenames
        ]
        goal_node_param = self.get_parameter("goal_node").value
        if goal_node_param == -1:
            self.goal_node = len(self.topomap) - 1
        else:
            self.goal_node = goal_node_param
        assert 0 <= self.goal_node < len(self.topomap), (
            f"Invalid goal_node {self.goal_node} for topomap "
            f"with {len(self.topomap)} nodes"
        )
        self.get_logger().info(
            f"Topomap mode: {len(self.topomap)} nodes from {topomap_full_dir}, "
            f"goal_node={self.goal_node}"
        )

    def _image_callback(self, msg: Image):
        obs_img = msg_to_pil(msg)
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(obs_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(obs_img)

    def _navigate_tick(self):
        """Main loop called at RATE Hz."""
        if len(self.context_queue) <= self.context_size:
            return

        mp = self.model_params

        if mp["model_type"] == "nomad":
            chosen_waypoint = self._navigate_nomad(mp)
        else:
            chosen_waypoint = self._navigate_vint(mp)

        if mp.get("normalize", False):
            chosen_waypoint[:2] *= self.max_v / self.rate_hz

        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = [float(x) for x in chosen_waypoint]
        self.waypoint_pub.publish(waypoint_msg)

        self.reached_goal = bool(self.closest_node == self.goal_node)
        goal_msg = Bool()
        goal_msg.data = bool(self.reached_goal)
        self.goal_pub.publish(goal_msg)

        if self.reached_goal:
            self.get_logger().info("Reached goal. Stopping.")

    def _prepare_obs(self, mp: dict) -> torch.Tensor:
        """Transform and prepare observation images for the model."""
        obs_images = transform_images(
            self.context_queue, mp["image_size"], center_crop=False
        )
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)
        return obs_images

    def _run_diffusion(self, obs_cond: torch.Tensor, mp: dict) -> np.ndarray:
        """Run the diffusion denoising loop and return unnormalized actions."""
        with torch.no_grad():
            if len(obs_cond.shape) == 2:
                obs_cond = obs_cond.repeat(self.num_samples, 1)
            else:
                obs_cond = obs_cond.repeat(self.num_samples, 1, 1)

            naction = torch.randn(
                (self.num_samples, mp["len_traj_pred"], 2), device=self.device
            )
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            t0 = time.time()
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=naction
                ).prev_sample
            self.get_logger().debug(f"Diffusion: {time.time() - t0:.3f}s")

        naction = _diffusion_output_to_actions(naction, self.action_stats)

        sampled_msg = Float32MultiArray()
        sampled_msg.data = [0.0] + naction.flatten().tolist()
        self.sampled_actions_pub.publish(sampled_msg)

        return naction

goal_cond=obsgoal_cond).flatten())

        min_idx = int(np.argmin(dists))
        self.closest_node = int(min_idx + start)
        self.get_logger().debug(f"Closest node: {self.closest_node}")

        sg_idx = min(
            min_idx + int(dists[min_idx] < self.close_threshold),
            len(obsgoal_cond) - 1,
        )
        obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

        naction = self._run_diffusion(obs_cond, mp)
        return naction[0][self.waypoint_index]

    def _navigate_vint(self, mp: dict) -> np.ndarray:
        """ViNT/GNM path (non-diffusion). Kept for compatibility."""
        start = max(self.closest_node - self.radius, 0)
        end = min(self.closest_node + self.radius + 1, self.goal_node)

        batch_obs = []
        batch_goal = []
        for sg_img in self.topomap[start:end + 1]:
            batch_obs.append(transform_images(self.context_queue, mp["image_size"]))
            batch_goal.append(transform_images(sg_img, mp["image_size"]))

        batch_obs = torch.cat(batch_obs, dim=0).to(self.device)
        batch_goal = torch.cat(batch_goal, dim=0).to(self.device)

        distances, waypoints = self.model(batch_obs, batch_goal)
        distances = to_numpy(distances)
        waypoints = to_numpy(waypoints)

        min_dist_idx = int(np.argmin(distances))
        if distances[min_dist_idx] > self.close_threshold:
            chosen = waypoints[min_dist_idx][self.waypoint_index]
            self.closest_node = int(start + min_dist_idx)
        else:
            chosen = waypoints[min(min_dist_idx + 1, len(waypoints) - 1)][
                self.waypoint_index
            ]
            self.closest_node = int(min(start + min_dist_idx + 1, self.goal_node))
        return chosen


def main(args=None):
    rclpy.init(args=args)
    node = NavigateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
