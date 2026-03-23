# Nomad Nav - Quick Start

Two simple options for controlling the Scout Mini robot with RealSense D435 camera.

## Prerequisites (One Time Setup)

Enable CAN interface:
```bash
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

In ~/ros2_ws before any command:
```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

## Option 1: Start Hardware to Teleop

Launch Scout base + RealSense camera (no data collection):

```bash
source install/setup.bash
ros2 launch nomad_nav hardware_pipeline.launch.py
```

Then in another terminal:
```bash
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Control robot with keys on screen.

## Option 2: Collect Training Data

Collect teleoperated trajectories for model finetuning:

**Terminal 1** - Start hardware:
```bash
source install/setup.bash
ros2 launch nomad_nav hardware_pipeline.launch.py
```

**Terminal 2** - Start data collection:
```bash
source install/setup.bash
./src/nomad_nav/collect_nomad_dataset.sh \
  --num-trajs 50 \
  --output-root ~/ros2_ws/data/nomad_finetune \
  --image-topic /camera/color/image_raw \
  --record-rosbag true
```

**Collection workflow:**
```
For each of (X) trajectories:
  [Script prompts] Press Enter to start recording
  [You press Enter & start driving]
  [Drive robot ~15-20 seconds with varied movements]
  [Press Enter to stop]
  → Next trajectory
```

Output: `~/ros2_ws/data/nomad_finetune/traj_000/...traj_099/`

Each trajectory folder contains:
- Sequential images: `0.jpg`, `1.jpg`, ..., `N.jpg`
- Metadata: `traj_data.pkl` (position + yaw)

---

## Inspect Collected Data

View metadata from a trajectory:
```bash
cd ~/ros2_ws/data/nomad_finetune/traj_000
python ../../inspect_trajectory.py
```

## Camera Details

- Topic: `/camera/color/image_raw` (RGB8, 640×480, 30 FPS)
- Driver: Intel RealSense D435 (realsense2_camera)
- Odometry topic: `/odom` (from Scout base)
- Sample rate: 4 Hz (4 images saved per second)

## Troubleshooting

**Camera not publishing:**
```bash
ros2 topic list | grep camera
ros2 topic hz /camera/color/image_raw
```

**Scout not responding:**
- Check CAN: `ip link show can0` (should say `state UP`)
- Check odometry: `ros2 topic hz /odom`

**Collection stuck:**
- Press Ctrl+C to stop
- Check terminal 1 for hardware errors
- Restart hardware and retry


Notes:
Auto stop traj?..