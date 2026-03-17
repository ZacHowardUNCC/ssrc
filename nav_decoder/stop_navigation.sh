#!/usr/bin/env bash
set -euo pipefail

if [[ -t 1 ]]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  NC=''
fi

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
ok() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# Source ROS2 if available
if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/ros2_ws/install/setup.bash"
fi

PATTERNS=(
  "navigation_client"
  "scout_mini_base"
  "nav_decoder_node"
  "closed_loop_navigation.launch.py"
  "ros2 launch scout_base"
  "ros2 launch nav_decoder"
)

info "Stopping navigation processes"
FOUND=0
for pat in "${PATTERNS[@]}"; do
  MATCHES=$(pgrep -af "$pat" || true)
  if [[ -n "$MATCHES" ]]; then
    FOUND=1
    echo "$MATCHES"
  fi
done

for pat in "${PATTERNS[@]}"; do
  pkill -f "$pat" >/dev/null 2>&1 || true
done

sleep 1

if [[ $FOUND -eq 0 ]]; then
  warn "No matching navigation processes were running"
else
  ok "Processes stopped"
fi

if command -v ros2 >/dev/null 2>&1; then
  if ros2 topic pub --once /decoder/arm std_msgs/msg/Bool "{data: false}" >/dev/null 2>&1; then
    ok "System disarmed"
  else
    warn "Failed to disarm system"
  fi
else
  warn "ros2 not available; could not disarm"
fi

ok "Clean shutdown complete"
