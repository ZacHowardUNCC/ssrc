#!/usr/bin/env bash
set -euo pipefail

if [[ -t 1 ]]; then
  BLUE='\033[0;34m'
  RED='\033[0;31m'
  NC='\033[0m'
else
  BLUE=''
  RED=''
  NC=''
fi

info() { echo -e "${BLUE}[INFO]${NC} $*"; }
err() { echo -e "${RED}[ERROR]${NC} $*" 1>&2; }

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
  --video-device <path>   Camera device (default: /dev/video4)
  --image-topic <topic>   Output image topic (default: /camera/image_raw)
  --pixel-format <fmt>    Pixel format (default: yuyv)
  --width <px>            Image width (default: 640)
  --height <px>           Image height (default: 480)
  --fps <hz>              Frame rate (default: 15.0)
  --frame-id <id>         frame_id for output image (default: usb_cam)
  --params-file <path>    usb_cam params file
                          (default: /opt/ros/humble/share/usb_cam/config/params_1.yaml)
  -h, --help              Show this help

Example:
  $0
  $0 --video-device /dev/video0 --image-topic /camera/image_raw --fps 30
EOF
}

VIDEO_DEVICE="/dev/video4"
IMAGE_TOPIC="/camera/image_raw"
PIXEL_FORMAT="yuyv"
IMAGE_WIDTH="640"
IMAGE_HEIGHT="480"
FPS="15.0"
FRAME_ID="usb_cam"
PARAMS_FILE="/opt/ros/humble/share/usb_cam/config/params_1.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-device)
      VIDEO_DEVICE="$2"
      shift 2
      ;;
    --image-topic)
      IMAGE_TOPIC="$2"
      shift 2
      ;;
    --pixel-format)
      PIXEL_FORMAT="$2"
      shift 2
      ;;
    --width)
      IMAGE_WIDTH="$2"
      shift 2
      ;;
    --height)
      IMAGE_HEIGHT="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --frame-id)
      FRAME_ID="$2"
      shift 2
      ;;
    --params-file)
      PARAMS_FILE="$2"
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

if [[ ! -f "/opt/ros/humble/setup.bash" ]]; then
  err "ROS2 Humble setup not found at /opt/ros/humble/setup.bash"
  exit 1
fi
source_setup_safe /opt/ros/humble/setup.bash

if [[ -f "$HOME/ros2_ws/install/setup.bash" ]]; then
  source_setup_safe "$HOME/ros2_ws/install/setup.bash"
fi

if [[ ! -e "$VIDEO_DEVICE" ]]; then
  err "Video device not found: $VIDEO_DEVICE"
  exit 1
fi
if [[ ! -f "$PARAMS_FILE" ]]; then
  err "Params file not found: $PARAMS_FILE"
  exit 1
fi

info "Starting usb_cam:"
info "  device:      $VIDEO_DEVICE"
info "  topic:       $IMAGE_TOPIC"
info "  format:      $PIXEL_FORMAT"
info "  resolution:  ${IMAGE_WIDTH}x${IMAGE_HEIGHT}"
info "  fps:         $FPS"
info "  frame_id:    $FRAME_ID"

exec ros2 run usb_cam usb_cam_node_exe --ros-args \
  --params-file "$PARAMS_FILE" \
  -p video_device:="$VIDEO_DEVICE" \
  -p pixel_format:="$PIXEL_FORMAT" \
  -p image_width:="$IMAGE_WIDTH" \
  -p image_height:="$IMAGE_HEIGHT" \
  -p framerate:="$FPS" \
  -p frame_id:="$FRAME_ID" \
  -r image_raw:="$IMAGE_TOPIC"
