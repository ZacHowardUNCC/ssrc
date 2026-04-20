"""
Offline inference evaluation for NoMaD.

Runs model inference on a recorded trajectory and compares predicted
waypoints against ground truth positions — no ROS, no robot movement.

Usage:
    python3 eval_inference.py
    python3 eval_inference.py --traj traj_000
    python3 eval_inference.py --traj traj_000 --goal-dist 0.5 --num-samples 4
    python3 eval_inference.py --list-trajs
"""

import argparse
import csv
import math
import os
import pickle
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage

# ── resolve visualnav paths the same way navigate.py does ────────────────────
_pkg_parent = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

from nomad_nav.path_utils import (
    ensure_visualnav_python_paths,
    get_default_model_config_path,
    get_default_robot_config_path,
)

ensure_visualnav_python_paths()

from nomad_nav.utils import load_model, to_numpy, transform_images
from vint_train.training.train_utils import get_action

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR = "/home/charml/ros2_ws/data/nomad_finetune"
DEFAULT_TRAJ_DIR    = os.path.join(DEFAULT_DATASET_DIR, "traj_000")
LOG_DIR             = os.path.expanduser("~/ros2_ws/logs/eval_runs")


# ── helpers ───────────────────────────────────────────────────────────────────

def _list_trajs(dataset_dir: str) -> list:
    if not os.path.isdir(dataset_dir):
        return []
    return sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    )


def _load_images(traj_dir: str) -> list:
    files = [f for f in os.listdir(traj_dir) if f.endswith(".jpg") or f.endswith(".png")]
    indices = sorted(int(os.path.splitext(f)[0]) for f in files)
    images = []
    for idx in indices:
        for ext in ("jpg", "png"):
            p = os.path.join(traj_dir, f"{idx}.{ext}")
            if os.path.exists(p):
                images.append(PILImage.open(p).convert("RGB"))
                break
    return images


def _load_traj_data(traj_dir: str):
    """Load traj_data.pkl → (positions [N,2], yaws [N])."""
    pkl_path = os.path.join(traj_dir, "traj_data.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    positions = np.asarray(data["position"], dtype=np.float32)
    yaws      = np.asarray(data["yaw"],      dtype=np.float32)
    return positions, yaws


def _find_goal_frame(t: int, positions: np.ndarray, goal_dist_m: float) -> int:
    """Return the first frame index j > t where cumulative distance from t >= goal_dist_m.
    If the trajectory ends before reaching that distance, returns the last valid index."""
    cum = 0.0
    j = t + 1
    while j < len(positions):
        cum += float(np.linalg.norm(positions[j] - positions[j - 1]))
        if cum >= goal_dist_m:
            return j
        j += 1
    return len(positions) - 1


def _world_to_local(dx_world: float, dy_world: float, yaw: float):
    """Rotate world-frame displacement into robot-local frame (forward=x, left=y)."""
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    dx_local =  dx_world * cos_y + dy_world * sin_y
    dy_local = -dx_world * sin_y + dy_world * cos_y
    return dx_local, dy_local


def _make_plots(rows: list, traj_dir: str, waypoint_idx: int,
                goal_dist: float, num_samples: int, png_path: str):
    """Save a 4-panel matplotlib figure to png_path."""
    t_arr   = np.array([r["t"]               for r in rows])
    gt_dx   = np.array([r["gt_dx"]           for r in rows])
    gt_dy   = np.array([r["gt_dy"]           for r in rows])
    pred_dx = np.array([r["pred_dx"]         for r in rows])
    pred_dy = np.array([r["pred_dy"]         for r in rows])

    traj_name = os.path.basename(traj_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f"NoMaD Offline Eval — {traj_name}   "
        f"(waypoint={waypoint_idx}, goal_dist={goal_dist}m, samples={num_samples})",
        fontsize=11, fontweight="bold",
    )

    # ── dx time series ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_arr, gt_dx,   label="ground truth", color="steelblue", linewidth=1.4)
    ax.plot(t_arr, pred_dx, label="predicted",    color="tomato",    linewidth=1.2, alpha=0.85)
    ax.fill_between(t_arr, gt_dx, pred_dx, alpha=0.12, color="gray")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(f"dx — forward displacement over {waypoint_idx+1} steps (m)")
    ax.set_xlabel("frame t")
    ax.set_ylabel("meters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── dy time series ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_arr, gt_dy,   label="ground truth", color="steelblue", linewidth=1.4)
    ax.plot(t_arr, pred_dy, label="predicted",    color="tomato",    linewidth=1.2, alpha=0.85)
    ax.fill_between(t_arr, gt_dy, pred_dy, alpha=0.12, color="gray")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(f"dy — lateral displacement over {waypoint_idx+1} steps (m)  [+ left / − right]")
    ax.set_xlabel("frame t")
    ax.set_ylabel("meters")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── dx scatter: predicted vs GT ───────────────────────────────────────────
    ax = axes[1, 0]
    all_dx = np.concatenate([gt_dx, pred_dx])
    lim_dx = max(abs(all_dx).max() * 1.15, 0.01)
    sign_match_dx = np.mean(np.sign(pred_dx) == np.sign(gt_dx)) * 100
    ax.scatter(gt_dx, pred_dx, s=12, alpha=0.5, color="steelblue")
    ax.plot([-lim_dx, lim_dx], [-lim_dx, lim_dx], "k--", linewidth=0.9, label="ideal (y=x)")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)
    ax.set_title(f"dx scatter  (sign match: {sign_match_dx:.1f}%)")
    ax.set_xlabel("gt_dx (m)")
    ax.set_ylabel("pred_dx (m)")
    ax.set_xlim(-lim_dx, lim_dx)
    ax.set_ylim(-lim_dx, lim_dx)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── dy scatter: predicted vs GT ───────────────────────────────────────────
    ax = axes[1, 1]
    all_dy = np.concatenate([gt_dy, pred_dy])
    lim_dy = max(abs(all_dy).max() * 1.15, 0.01)
    sign_match_dy = np.mean(np.sign(pred_dy) == np.sign(gt_dy)) * 100
    ax.scatter(gt_dy, pred_dy, s=12, alpha=0.5, color="tomato")
    ax.plot([-lim_dy, lim_dy], [-lim_dy, lim_dy], "k--", linewidth=0.9, label="ideal (y=x)")
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)
    ax.set_title(f"dy scatter  (sign match: {sign_match_dy:.1f}%)")
    ax.set_xlabel("gt_dy (m)")
    ax.set_ylabel("pred_dy (m)")
    ax.set_xlim(-lim_dy, lim_dy)
    ax.set_ylim(-lim_dy, lim_dy)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {png_path}")


# ── per-trajectory evaluation ─────────────────────────────────────────────────

def _eval_traj(traj_dir, model, noise_scheduler, model_params,
               device, args, MAX_V, RATE, context_size):
    """Run inference on a single trajectory and save results."""
    traj_dir = os.path.abspath(traj_dir)
    if not os.path.isdir(traj_dir):
        raise FileNotFoundError(f"Trajectory folder not found: {traj_dir}")

    print(f"\nLoading trajectory: {traj_dir}")
    images = _load_images(traj_dir)
    positions, yaws = _load_traj_data(traj_dir)

    N = len(images)
    if N != len(positions):
        print(f"WARNING: {N} images but {len(positions)} position entries — using min")
        N = min(N, len(positions))
        images    = images[:N]
        positions = positions[:N]
        yaws      = yaws[:N]

    step_dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    print(f"Frames:          {N}")
    print(f"Position range:  x=[{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]  "
          f"y=[{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]")
    print(f"Yaw range:       [{yaws.min():.3f}, {yaws.max():.3f}] rad")
    print(f"Avg step dist:   {step_dists.mean():.4f}m")

    t_start = context_size
    # t_end: need t + waypoint + 1 < N, and goal frame < N.
    # We don't know max goal frame upfront so conservatively stop at N-2
    # and skip frames where either bound is violated inside the loop.
    t_end = N - 2

    if t_end <= t_start:
        raise ValueError(f"Trajectory too short (need >{context_size + 2} frames, have {N})")

    print(f"\nEvaluating frames {t_start}–{t_end} ({t_end - t_start + 1} steps)...\n")

    rows = []

    for t in range(t_start, t_end + 1):
        # gt requires positions[t + waypoint + 1]
        if t + args.waypoint + 1 >= N:
            break

        # dynamic goal frame: first frame at least goal_dist meters away
        goal_frame = _find_goal_frame(t, positions, args.goal_dist)
        if goal_frame >= N:
            break

        context_window = images[t - context_size : t + 1]
        obs_images = transform_images(context_window, model_params["image_size"], center_crop=False)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(device)

        goal_tensor = transform_images(
            [images[goal_frame]], model_params["image_size"], center_crop=False
        ).to(device)

        mask = torch.zeros(1).long().to(device)

        gt_end   = t + args.waypoint + 1
        dx_world = float(positions[gt_end][0] - positions[t][0])
        dy_world = float(positions[gt_end][1] - positions[t][1])
        gt_dx, gt_dy = _world_to_local(dx_world, dy_world, float(yaws[t]))
        gt_dist  = float(np.linalg.norm([dx_world, dy_world]))

        with torch.no_grad():
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=obs_images,
                goal_img=goal_tensor,
                input_goal_mask=mask,
            )

            pred_dist_tensor = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            pred_dist = float(to_numpy(pred_dist_tensor.flatten())[0])

            if obsgoal_cond.dim() == 2:
                obs_cond = obsgoal_cond.repeat(args.num_samples, 1)
            else:
                obs_cond = obsgoal_cond.repeat(args.num_samples, 1, 1)

            noisy_action = torch.randn(
                (args.num_samples, model_params["len_traj_pred"], 2), device=device
            )
            naction = noisy_action

            noise_scheduler.set_timesteps(model_params["num_diffusion_iters"])
            for k in noise_scheduler.timesteps:
                noise_pred = model(
                    "noise_pred_net",
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond,
                )
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction,
                ).prev_sample

        naction_out = to_numpy(get_action(naction))
        chosen_waypoint = naction_out[0][args.waypoint].copy()  # sample 0 only, matches navigate.py
        if model_params["normalize"]:
            chosen_waypoint[:2] *= MAX_V / RATE

        pred_dx = float(chosen_waypoint[0])
        pred_dy = float(chosen_waypoint[1])

        error_dx        = pred_dx - gt_dx
        error_dy        = pred_dy - gt_dy
        euclidean_error = math.sqrt(error_dx**2 + error_dy**2)
        sign_dx = int(math.copysign(1, pred_dx)) == int(math.copysign(1, gt_dx))
        sign_dy = int(math.copysign(1, pred_dy)) == int(math.copysign(1, gt_dy))

        rows.append({
            "t":               t,
            "gt_dx":           gt_dx,
            "gt_dy":           gt_dy,
            "pred_dx":         pred_dx,
            "pred_dy":         pred_dy,
            "error_dx":        error_dx,
            "error_dy":        error_dy,
            "euclidean_error": euclidean_error,
            "pred_dist":       pred_dist,
            "gt_dist":         gt_dist,
            "sign_match_dx":   int(sign_dx),
            "sign_match_dy":   int(sign_dy),
        })

        if (t - t_start) % 10 == 0 or t == t_end:
            n_done  = t - t_start + 1
            n_total = t_end - t_start + 1
            print(
                f"Frame {t}/{t_end} ({n_done}/{n_total}) | "
                f"gt=[{gt_dx:+.3f}, {gt_dy:+.3f}] | "
                f"pred=[{pred_dx:+.3f}, {pred_dy:+.3f}] | "
                f"err={euclidean_error:.3f}m | "
                f"pred_dist={pred_dist:.2f}"
            )

    ee   = np.array([r["euclidean_error"] for r in rows])
    sdx  = np.array([r["sign_match_dx"]   for r in rows])
    sdy  = np.array([r["sign_match_dy"]   for r in rows])
    pd_  = np.array([r["pred_dist"]       for r in rows])

    sdx_pct = 100.0 * sdx.mean()
    sdy_pct = 100.0 * sdy.mean()

    verdict_dx = (
        "GOOD"     if sdx_pct > 80 else
        "MARGINAL" if sdx_pct > 60 else
        "POOR — model confused"
    )
    verdict_dy = (
        "GOOD"     if sdy_pct > 70 else
        "MARGINAL" if sdy_pct > 55 else
        "POOR — model confused"
    )

    notes = []
    if args.num_samples == 1:
        notes.append("NOTE: num_samples=1 — diffusion is stochastic; try --num-samples 4 for stable results")

    summary = f"""
================================================
INFERENCE EVALUATION SUMMARY
Trajectory:      {traj_dir}
Frames evaluated:{len(rows)}
Waypoint index:  {args.waypoint}  (cumulative {args.waypoint+1}-step prediction)
Goal dist:       {args.goal_dist}m  (dynamic per-frame, matches trajectory step size)
Num samples:     {args.num_samples}
================================================
WAYPOINT PREDICTION  (gt covers same {args.waypoint+1} steps as prediction):
  Mean euclidean error:  {ee.mean():.3f}m
  Std  euclidean error:  {ee.std():.3f}m
  Max  euclidean error:  {ee.max():.3f}m

DIRECTION ACCURACY:
  Forward (dx) sign match:  {sdx_pct:.1f}%  → {verdict_dx}
  Lateral (dy) sign match:  {sdy_pct:.1f}%  → {verdict_dy}

DISTANCE HEAD OUTPUT:
  Mean pred_dist:  {pd_.mean():.2f}  Std: {pd_.std():.2f}  (model's temporal-step estimate)

VERDICT:
  Forward direction: {verdict_dx}
  Turning direction: {verdict_dy}
{chr(10).join(notes)}
================================================"""
    print(summary)

    traj_name = os.path.basename(traj_dir)
    run_dir   = os.path.join(LOG_DIR, traj_name)
    os.makedirs(run_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(run_dir, f"eval_{ts}.csv")
    png_path = os.path.join(run_dir, f"eval_{ts}.png")
    txt_path = os.path.join(run_dir, f"eval_{ts}_summary.txt")

    fieldnames = [
        "t", "gt_dx", "gt_dy", "pred_dx", "pred_dy",
        "error_dx", "error_dy", "euclidean_error",
        "pred_dist", "gt_dist", "sign_match_dx", "sign_match_dy",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"CSV saved:     {csv_path}")

    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"Summary saved: {txt_path}")

    _make_plots(rows, traj_dir, args.waypoint, args.goal_dist, args.num_samples, png_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Offline NoMaD inference evaluation")
    parser.add_argument(
        "--traj-dir", default=None,
        help="Full path to a single trajectory folder",
    )
    parser.add_argument(
        "--traj", nargs="+", default=None, metavar="TRAJ",
        help="One or more trajectory names (e.g. --traj traj_000 traj_001)",
    )
    parser.add_argument(
        "--trajs", nargs="+", default=None, metavar="TRAJ",
        help="One or more trajectory names (e.g. --trajs traj_000 traj_001 traj_005)",
    )
    parser.add_argument(
        "--all-trajs", action="store_true",
        help="Run evaluation on every trajectory in the dataset dir",
    )
    parser.add_argument(
        "--waypoint", type=int, default=2,
        help="Waypoint index into predicted action sequence (default: 2)",
    )
    parser.add_argument(
        "--goal-dist", type=float, default=0.5,
        help=(
            "Target distance in meters for the goal image (default: 0.5m). "
            "For each frame t, the goal is the first frame at least this far away — "
            "automatically adjusts to trajectory step size so short and long trajs "
            "are evaluated fairly. Mirrors topomap node spacing in real navigation."
        ),
    )
    parser.add_argument(
        "--num-samples", type=int, default=1,
        help=(
            "Number of diffusion rollouts to average per frame (default: 1). "
            "Higher values reduce stochasticity at the cost of speed."
        ),
    )
    parser.add_argument(
        "--model", default="nomad",
        help="Model name from models.yaml (default: nomad)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: cuda, cpu, or auto (default: auto)",
    )
    parser.add_argument(
        "--list-trajs", action="store_true",
        help="List available trajectories in default dataset dir and exit",
    )
    args = parser.parse_args()

    # ── list trajectories and exit ────────────────────────────────────────────
    if args.list_trajs:
        trajs = _list_trajs(DEFAULT_DATASET_DIR)
        if not trajs:
            print(f"No trajectories found in: {DEFAULT_DATASET_DIR}")
        else:
            print(f"Trajectories in {DEFAULT_DATASET_DIR}:")
            for t in trajs:
                print(f"  {t}")
        return

    # ── resolve which trajectories to run ────────────────────────────────────
    all_available = _list_trajs(DEFAULT_DATASET_DIR)

    if args.all_trajs:
        traj_dirs = [os.path.join(DEFAULT_DATASET_DIR, t) for t in all_available]
    elif args.trajs:
        traj_dirs = [os.path.join(DEFAULT_DATASET_DIR, t) for t in args.trajs]
    elif args.traj:
        traj_dirs = [os.path.join(DEFAULT_DATASET_DIR, t) for t in args.traj]
    elif args.traj_dir:
        traj_dirs = [args.traj_dir]
    else:
        traj_dirs = [DEFAULT_TRAJ_DIR]

    selected_names = {os.path.basename(d) for d in traj_dirs}
    print(f"\nAvailable trajectories in {DEFAULT_DATASET_DIR}:")
    for t in all_available:
        marker = " <-- selected" if t in selected_names else ""
        print(f"  {t}{marker}")
    print(f"\nWill evaluate {len(traj_dirs)} trajectory/trajectories.")

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── load config (identical to navigate.py) ────────────────────────────────
    model_config_path = get_default_model_config_path()
    robot_config_path = get_default_robot_config_path()

    with open(model_config_path, "r") as f:
        model_paths = yaml.safe_load(f)

    model_cfg_rel = model_paths[args.model]["config_path"]
    if not os.path.isabs(model_cfg_rel):
        model_cfg_rel = os.path.normpath(
            os.path.join(os.path.dirname(model_config_path), model_cfg_rel)
        )
    with open(model_cfg_rel, "r") as f:
        model_params = yaml.safe_load(f)

    with open(robot_config_path, "r") as f:
        robot_config = yaml.safe_load(f)

    MAX_V        = robot_config["max_v"]
    RATE         = robot_config["frame_rate"]
    context_size = model_params["context_size"]
    num_diffusion_iters = model_params["num_diffusion_iters"]

    print(f"Model config:    {model_cfg_rel}")
    print(f"context_size:    {context_size}")
    print(f"len_traj_pred:   {model_params['len_traj_pred']}")
    print(f"num_diff_iters:  {num_diffusion_iters}")
    print(f"normalize:       {model_params['normalize']}")
    print(f"MAX_V/RATE:      {MAX_V}/{RATE} = {MAX_V/RATE:.4f}")
    print(f"waypoint index:  {args.waypoint}  (cumulative {args.waypoint+1}-step prediction)")
    print(f"goal dist:       {args.goal_dist}m  (dynamic per-frame)")
    print(f"num samples:     {args.num_samples}")

    # ── load model weights ────────────────────────────────────────────────────
    ckpt_path = model_paths[args.model]["ckpt_path"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(
            os.path.join(os.path.dirname(model_config_path), ckpt_path)
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"\nLoading weights: {ckpt_path}")

    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
    from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

    if model_params["vision_encoder"] == "nomad_vint":
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=model_params["encoding_size"],
            context_size=model_params["context_size"],
            mha_num_attention_heads=model_params["mha_num_attention_heads"],
            mha_num_attention_layers=model_params["mha_num_attention_layers"],
            mha_ff_dim_factor=model_params["mha_ff_dim_factor"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
    else:
        raise ValueError(f"Unsupported vision encoder: {model_params['vision_encoder']}")

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=model_params["encoding_size"],
        down_dims=model_params["down_dims"],
        cond_predict_scale=model_params["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=model_params["encoding_size"])
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"MISSING KEYS ({len(missing)}):    {missing if missing else 'none'}")
    print(f"UNEXPECTED KEYS ({len(unexpected)}): {unexpected if unexpected else 'none'}")
    model = model.to(device)
    model.eval()

    # ── noise scheduler (identical to navigate.py) ────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # ── run evaluation on each trajectory ────────────────────────────────────
    failed = []
    for i, traj_dir in enumerate(traj_dirs):
        print(f"\n{'='*60}")
        print(f"Trajectory {i+1}/{len(traj_dirs)}: {os.path.basename(traj_dir)}")
        print(f"{'='*60}")
        try:
            _eval_traj(
                traj_dir, model, noise_scheduler, model_params,
                device, args, MAX_V, RATE, context_size,
            )
        except Exception as exc:
            print(f"ERROR on {os.path.basename(traj_dir)}: {exc}")
            failed.append(os.path.basename(traj_dir))

    if len(traj_dirs) > 1:
        print(f"\nDone. {len(traj_dirs) - len(failed)}/{len(traj_dirs)} trajectories succeeded.")
        if failed:
            print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
