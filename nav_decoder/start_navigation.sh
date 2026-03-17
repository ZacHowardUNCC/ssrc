#!/usr/bin/env bash
set -euo pipefail

if [[ -t 1 ]]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  BOLD=''
  NC=''
fi

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_TS=$(date +%s)

DESKTOP_IP="${1:-}"
CAMERA_TOPIC="${2:-/camera/color/image_raw}"
INSTRUCTION="${3:-Navigate forward and avoid obstacles}"
INFER_HZ="${4:-2.0}"

if [[ -z "$DESKTOP_IP" ]]; then
  err "Desktop IP is required."
  echo "Usage: $0 <DESKTOP_IP> [CAMERA_TOPIC] [INSTRUCTION] [INFERENCE_RATE_HZ]"
  exit 1
fi

if [[ ! -f "/opt/ros/humble/setup.bash" ]]; then
  err "ROS2 Humble not found at /opt/ros/humble/setup.bash"
  exit 1
fi

if [[ ! -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  err "Workspace not built."
  echo "Run: cd ~/ros2_ws && colcon build"
  exit 1
fi

# Source ROS2 and workspace
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source "$HOME/ros2_ws/install/setup.bash"

info "Configuration:"
info "  Desktop IP     : $DESKTOP_IP"
info "  Camera topic   : $CAMERA_TOPIC"
info "  Instruction    : $INSTRUCTION"
info "  Inference rate : ${INFER_HZ} Hz"

# Auto-detect camera topic if default doesn't exist
TOPIC_LIST=$(ros2 topic list 2>/dev/null || true)
if [[ -n "$TOPIC_LIST" ]]; then
  if ! echo "$TOPIC_LIST" | grep -Fxq "$CAMERA_TOPIC"; then
    warn "Default camera topic not found: $CAMERA_TOPIC"
    IMAGE_TOPICS=$(echo "$TOPIC_LIST" | grep image || true)
    if [[ -z "$IMAGE_TOPICS" ]]; then
      err "No image topics found. Please specify a camera topic."
      exit 1
    fi
    info "Available image topics:"
    echo "$IMAGE_TOPICS"
    CAMERA_TOPIC=$(echo "$IMAGE_TOPICS" | head -n 1)
    warn "Using first image topic: $CAMERA_TOPIC"
  fi
else
  warn "Could not query ROS2 topics. Is ROS running?"
fi

# Check camera is publishing
HZ_LINE=$(timeout 2 ros2 topic hz "$CAMERA_TOPIC" 2>/dev/null | awk '/average rate/ {print $0}' | tail -n 1 || true)
if [[ -n "$HZ_LINE" ]]; then
  RATE=$(echo "$HZ_LINE" | awk '{print $3}')
  info "Camera frame rate: ${RATE} Hz"
  if awk "BEGIN {exit !(${RATE} < 15.0)}"; then
    warn "Camera frame rate is below 15 Hz"
  else
    ok "Camera frame rate looks good"
  fi
else
  warn "Unable to measure camera frame rate (no messages within 2s?)"
fi

# Test Desktop connectivity
if ping -c 1 -W 1 "$DESKTOP_IP" >/dev/null 2>&1; then
  ok "Ping to Desktop succeeded"
else
  warn "Ping to Desktop failed"
fi

if timeout 1 bash -c "cat </dev/null >/dev/tcp/$DESKTOP_IP/5556" >/dev/null 2>&1; then
  ok "TCP port 5556 is reachable"
else
  warn "TCP port 5556 is not reachable (continuing)"
fi

# Kill any existing instances
info "Stopping existing navigation processes (if any)"
pkill -f navigation_client >/dev/null 2>&1 || true
pkill -f scout_mini_base >/dev/null 2>&1 || true
pkill -f nav_decoder_node >/dev/null 2>&1 || true
pkill -f closed_loop_navigation.launch.py >/dev/null 2>&1 || true

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SCOUT_LOG="$LOG_DIR/${STAMP}.scout_base.log"
NAV_LOG="$LOG_DIR/${STAMP}.closed_loop.log"

cleanup() {
  echo
  info "Shutting down..."
  ros2 topic pub --once /decoder/arm std_msgs/msg/Bool "{data: false}" >/dev/null 2>&1 || true
  if [[ -n "${SCOUT_PID:-}" ]]; then
    kill "$SCOUT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${NAV_PID:-}" ]]; then
    kill "$NAV_PID" >/dev/null 2>&1 || true
  fi
  pkill -f navigation_client >/dev/null 2>&1 || true
  pkill -f scout_mini_base >/dev/null 2>&1 || true
  pkill -f nav_decoder_node >/dev/null 2>&1 || true
  pkill -f closed_loop_navigation.launch.py >/dev/null 2>&1 || true
  wait >/dev/null 2>&1 || true
  END_TS=$(date +%s)
  DURATION=$((END_TS - START_TS))
  info "Uptime: ${DURATION}s"
  info "Logs:"
  info "  $SCOUT_LOG"
  info "  $NAV_LOG"
  ok "Clean shutdown"
}
trap cleanup INT TERM

info "Starting scout_base (background)"
ros2 launch scout_base scout_mini_base.launch.py >"$SCOUT_LOG" 2>&1 &
SCOUT_PID=$!

info "Starting closed-loop navigation (background)"
ROS2_LAUNCH_ARGS=(
  "desktop_ip:=${DESKTOP_IP}"
  "camera_topic:=${CAMERA_TOPIC}"
  "instruction:=${INSTRUCTION}"
  "inference_rate_hz:=${INFER_HZ}"
)
ros2 launch nav_decoder closed_loop_navigation.launch.py "${ROS2_LAUNCH_ARGS[@]}" >"$NAV_LOG" 2>&1 &
NAV_PID=$!

sleep 2

info "Auto-arming system"
if ros2 topic pub --once /decoder/arm std_msgs/msg/Bool "{data: true}" >/dev/null 2>&1; then
  ok "System armed"
else
  warn "Failed to arm system"
fi

info "Monitoring commands:"
info "  ros2 topic echo /discrete_action"
info "  ros2 topic echo /decoder/arm"
info "  $SCRIPT_DIR/monitor_navigation.sh $DESKTOP_IP $CAMERA_TOPIC"
info "  tail -f $NAV_LOG"

ok "Navigation started. Press Ctrl+C to stop."
wait "$SCOUT_PID" "$NAV_PID"
