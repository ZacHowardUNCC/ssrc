#!/usr/bin/env bash
set -euo pipefail

if [[ -t 1 ]]; then
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'
else
  GREEN=''
  YELLOW=''
  BLUE=''
  NC=''
fi

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

DESKTOP_IP="${1:-}"
CAMERA_TOPIC="${2:-}"

if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/ros2_ws/install/setup.bash"
fi

if [[ -z "$CAMERA_TOPIC" ]]; then
  TOPICS=$(ros2 topic list 2>/dev/null || true)
  CAMERA_TOPIC=$(echo "$TOPICS" | grep image | head -n 1 || true)
fi

if [[ -z "$CAMERA_TOPIC" ]]; then
  warn "No camera topic detected. You can pass one as arg 2."
  CAMERA_TOPIC="N/A"
fi

while true; do
  clear
  echo -e "${GREEN}UniVLA Navigation Monitor${NC}"
  echo "Timestamp      : $(date)"
  echo "Camera topic   : $CAMERA_TOPIC"

  # Camera FPS
  FPS="N/A"
  if [[ "$CAMERA_TOPIC" != "N/A" ]]; then
    HZ_LINE=$(timeout 1 ros2 topic hz "$CAMERA_TOPIC" 2>/dev/null | awk '/average rate/ {print $0}' | tail -n 1 || true)
    if [[ -n "$HZ_LINE" ]]; then
      FPS=$(echo "$HZ_LINE" | awk '{print $3}')
    fi
  fi
  echo "Camera FPS     : $FPS"

  # Latest discrete action
  ACTION_RAW=$(timeout 1 ros2 topic echo -n 1 /discrete_action 2>/dev/null | tail -n 1 || true)
  if [[ -z "$ACTION_RAW" ]]; then
    ACTION_RAW="N/A"
  fi
  echo "Last action    : $ACTION_RAW"

  # Arm status
  ARM_RAW=$(timeout 1 ros2 topic echo -n 1 /decoder/arm 2>/dev/null | tail -n 1 || true)
  if [[ -z "$ARM_RAW" ]]; then
    ARM_RAW="N/A"
  fi
  echo "Arm status     : $ARM_RAW"

  # Network latency
  if [[ -n "$DESKTOP_IP" ]]; then
    PING_LINE=$(ping -c 1 -W 1 "$DESKTOP_IP" 2>/dev/null | awk -F'time=' '/time=/{print $2}' | awk '{print $1" ms"}' | head -n 1)
    if [[ -z "$PING_LINE" ]]; then
      PING_LINE="unreachable"
    fi
    echo "Latency        : $PING_LINE"
  else
    echo "Latency        : N/A (no desktop IP provided)"
  fi

  echo
  echo "Ctrl+C to exit"
  sleep 1
done
