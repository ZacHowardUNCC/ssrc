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
err() { echo -e "${RED}[ERROR]${NC} $*" 1>&2; }

normalize_bool() {
  case "${1,,}" in
    true|1|yes|y|on) echo "true" ;;
    false|0|no|n|off) echo "false" ;;
    *)
      err "Invalid boolean value: $1"
      exit 1
      ;;
  esac
}

wait_for_enter() {
  if ! IFS= read -r </dev/tty; then
    err "Failed to read from terminal input (/dev/tty)."
    exit 1
  fi
}

source_setup_safe() {
  local setup_file="$1"
  local had_nounset="false"
  if [[ $- == *u* ]]; then
    had_nounset="true"
    set +u
  fi

  # shellcheck disable=SC1090
  source "$setup_file"

  if [[ "$had_nounset" == "true" ]]; then
    set -u
  fi
}

usage() {
  cat <<EOF
Usage:
  $0 [options]

Options:
  --output-root <path>      Dataset directory (<dataset_name>) that will contain
                            only trajectory folders.
                            Default: \$HOME/ros2_ws/data/nomad_dataset
  --num-trajs <N>           Number of trajectories to collect. Default: 100
  --start-index <N>         Starting index for naming trajectories. Default: 0
  --traj-prefix <name>      Prefix for trajectory folder names. Default: traj
  --sample-rate <hz>        Sampling rate for image+odom capture. Default: 4.0
  --image-topic <topic>     Camera topic. Default: /camera/image_raw
  --odom-topic <topic>      Odom topic. Default: /odom
  --logs-root <path>        Log directory (kept outside dataset tree by default).
                            Default: <output-root>_logs
  --rosbag-root <path>      Rosbag directory (kept outside dataset tree by default).
                            Default: <output-root>_rosbags
  --record-rosbag <bool>    Whether to record rosbag per trajectory. Default: true
  --zero-origin <bool>      Subtract first odom sample from all positions. Default: true
  --sync-tolerance <sec>    Max image/odom timestamp skew for sample. Default: 1.5
  --overwrite <bool>        Overwrite existing trajectory folders. Default: false
  -h, --help                Show this help

Example:
  $0 --output-root ~/ros2_ws/data/nomad_finetune --num-trajs 100 --sample-rate 4.0
EOF
}

OUTPUT_ROOT="$HOME/ros2_ws/data/nomad_dataset"
NUM_TRAJS=100
START_INDEX=0
TRAJ_PREFIX="traj"
SAMPLE_RATE="4.0"
IMAGE_TOPIC="/camera/image_raw"
ODOM_TOPIC="/odom"
LOGS_ROOT=""
ROSBAG_ROOT=""
RECORD_ROSBAG="true"
ZERO_ORIGIN="true"
SYNC_TOLERANCE="1.5"
OVERWRITE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --num-trajs)
      NUM_TRAJS="$2"
      shift 2
      ;;
    --start-index)
      START_INDEX="$2"
      shift 2
      ;;
    --traj-prefix)
      TRAJ_PREFIX="$2"
      shift 2
      ;;
    --sample-rate)
      SAMPLE_RATE="$2"
      shift 2
      ;;
    --image-topic)
      IMAGE_TOPIC="$2"
      shift 2
      ;;
    --odom-topic)
      ODOM_TOPIC="$2"
      shift 2
      ;;
    --logs-root)
      LOGS_ROOT="$2"
      shift 2
      ;;
    --rosbag-root)
      ROSBAG_ROOT="$2"
      shift 2
      ;;
    --record-rosbag)
      RECORD_ROSBAG="$2"
      shift 2
      ;;
    --zero-origin)
      ZERO_ORIGIN="$2"
      shift 2
      ;;
    --sync-tolerance)
      SYNC_TOLERANCE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

RECORD_ROSBAG="$(normalize_bool "$RECORD_ROSBAG")"
ZERO_ORIGIN="$(normalize_bool "$ZERO_ORIGIN")"
OVERWRITE="$(normalize_bool "$OVERWRITE")"

if [[ -z "$LOGS_ROOT" ]]; then
  LOGS_ROOT="${OUTPUT_ROOT}_logs"
fi
if [[ -z "$ROSBAG_ROOT" ]]; then
  ROSBAG_ROOT="${OUTPUT_ROOT}_rosbags"
fi

if [[ ! -t 0 ]]; then
  err "Interactive terminal required (this script uses Enter-to-start/stop prompts)."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "/opt/ros/humble/setup.bash" ]]; then
  source_setup_safe /opt/ros/humble/setup.bash
fi

if [[ -f "${WS_ROOT}/install/setup.bash" ]]; then
  source_setup_safe "${WS_ROOT}/install/setup.bash"
elif [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  source_setup_safe "$HOME/ros2_ws/install/setup.bash"
else
  err "install/setup.bash not found. Build workspace first: colcon build"
  exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
  err "ros2 CLI not found in PATH."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOGS_ROOT"
if [[ "$RECORD_ROSBAG" == "true" ]]; then
  mkdir -p "$ROSBAG_ROOT"
fi

CURRENT_COLLECTOR_PID=""
CURRENT_BAG_PID=""

is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

stop_job_group() {
  local pid="$1"
  local name="$2"
  local pgid="-$pid"

  if ! is_running "$pid"; then
    return
  fi

  # Try graceful interrupt first.
  kill -INT "$pgid" >/dev/null 2>&1 || true
  for _ in {1..30}; do
    if ! is_running "$pid"; then
      wait "$pid" >/dev/null 2>&1 || true
      return
    fi
    sleep 0.1
  done

  warn "${name} did not stop after SIGINT; sending SIGTERM."
  kill -TERM "$pgid" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! is_running "$pid"; then
      wait "$pid" >/dev/null 2>&1 || true
      return
    fi
    sleep 0.1
  done

  warn "${name} still running; sending SIGKILL."
  kill -KILL "$pgid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
}

stop_current_recording() {
  stop_job_group "$CURRENT_BAG_PID" "rosbag recorder"
  CURRENT_BAG_PID=""

  stop_job_group "$CURRENT_COLLECTOR_PID" "trajectory collector"
  CURRENT_COLLECTOR_PID=""
}

on_interrupt() {
  echo
  warn "Interrupted. Stopping active recording."
  stop_current_recording
  exit 130
}
trap on_interrupt INT TERM

info "Output root      : $OUTPUT_ROOT"
info "Trajectories     : $NUM_TRAJS (starting at index $START_INDEX)"
info "Sample rate      : ${SAMPLE_RATE} Hz"
info "Image topic      : $IMAGE_TOPIC"
info "Odom topic       : $ODOM_TOPIC"
info "Logs root        : $LOGS_ROOT"
info "Rosbag root      : $ROSBAG_ROOT"
info "Record rosbag    : $RECORD_ROSBAG"
info "Zero origin      : $ZERO_ORIGIN"
info "Sync tolerance   : ${SYNC_TOLERANCE}s"
info "Overwrite existing: $OVERWRITE"

for ((i = 0; i < NUM_TRAJS; i++)); do
  idx=$((START_INDEX + i))
  traj_name=$(printf "%s_%03d" "$TRAJ_PREFIX" "$idx")
  traj_dir="${OUTPUT_ROOT}/${traj_name}"
  collector_log="${LOGS_ROOT}/${traj_name}.collector.log"
  bag_log="${LOGS_ROOT}/${traj_name}.rosbag.log"

  if [[ -d "$traj_dir" && "$OVERWRITE" != "true" ]]; then
    if [[ -f "${traj_dir}/traj_data.pkl" ]]; then
      warn "Skipping existing trajectory ${traj_name} (traj_data.pkl already exists)."
      continue
    fi
    if find "$traj_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      warn "Skipping non-empty folder ${traj_name}. Use --overwrite true to replace it."
      continue
    fi
  fi

  if [[ -d "$traj_dir" && "$OVERWRITE" == "true" ]]; then
    rm -rf "$traj_dir"
  fi
  mkdir -p "$traj_dir"

  echo
  info "Trajectory $((i + 1))/$NUM_TRAJS -> ${traj_name}"
  info "Drive with intention and safe coverage. Press Enter to start recording."
  wait_for_enter

  setsid ros2 run nomad_nav collect_trajectory --ros-args \
    -p output_dir:="${traj_dir}" \
    -p image_topic:="${IMAGE_TOPIC}" \
    -p odom_topic:="${ODOM_TOPIC}" \
    -p sample_rate_hz:="${SAMPLE_RATE}" \
    -p zero_origin:="${ZERO_ORIGIN}" \
    -p sync_tolerance_s:="${SYNC_TOLERANCE}" \
    -p overwrite:="true" \
    >"$collector_log" 2>&1 </dev/null &
  CURRENT_COLLECTOR_PID=$!

  if [[ "$RECORD_ROSBAG" == "true" ]]; then
    setsid ros2 bag record -o "${ROSBAG_ROOT}/${traj_name}" "$IMAGE_TOPIC" "$ODOM_TOPIC" >"$bag_log" 2>&1 </dev/null &
    CURRENT_BAG_PID=$!
  else
    CURRENT_BAG_PID=""
  fi

  sleep 1
  if ! is_running "$CURRENT_COLLECTOR_PID"; then
    err "Collector exited early for ${traj_name}. Check ${collector_log}"
    stop_current_recording
    exit 1
  fi

  info "Recording ${traj_name}. Press Enter to stop."
  wait_for_enter

  stop_current_recording

  img_count=$(find "$traj_dir" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')
  if [[ -f "${traj_dir}/traj_data.pkl" ]]; then
    ok "${traj_name}: saved ${img_count} images + traj_data.pkl"
  else
    warn "${traj_name}: traj_data.pkl missing. Check ${collector_log}"
  fi
done

echo
ok "Collection loop finished."
info "If you need exactly 100 trajectories, rerun with --start-index to fill any skipped indices."
