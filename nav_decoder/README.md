# Nav Decoder (Jetson Side)

Single-command startup scripts for the Jetson side of the UniVLA navigation system.

Quick start:
```
./start_navigation.sh <DESKTOP_IP> [CAMERA_TOPIC] [INSTRUCTION] [INFERENCE_RATE_HZ]
```

Stop:
```
./stop_navigation.sh
```

Monitor:
```
./monitor_navigation.sh <DESKTOP_IP> [CAMERA_TOPIC]
```

Rebuild package:
```
cd ~/ros2_ws
colcon build --packages-select nav_decoder
```

Notes:
- `start_navigation.sh` will auto-detect a camera topic if the default (`/camera/color/image_raw`) is not available.
- Default instruction is `Navigate forward and avoid obstacles`.
- Default inference rate is `2.0` Hz.
- Logs are written under `~/ros2_ws/src/nav_decoder/logs/`.

Troubleshooting:
- Package not found: ensure you sourced the workspace in your terminal:
  ```
  source /opt/ros/humble/setup.bash
  source ~/ros2_ws/install/setup.bash
  ```
- Camera topic not found: pass the camera topic explicitly as the second argument.
- Low or missing FPS: verify the camera is publishing and the correct topic is selected.
- Desktop unreachable: check IP connectivity and ensure port `5556` is open on the Desktop.
