#!/usr/bin/env bash
set -eo pipefail

BASE=${BASE:-$PWD}
DEPTH_PREPROCESS_SCRIPT=${DEPTH_PREPROCESS_SCRIPT:?set DEPTH_PREPROCESS_SCRIPT}
START_LINGBOT_SCRIPT=${START_LINGBOT_SCRIPT:-$BASE/scripts/start_lingbot_depth.sh}

source /opt/ros/jazzy/setup.bash
[[ -n "${ROBOT_ENV:-}" ]] && source "$ROBOT_ENV"
source "$BASE/install/setup.bash"

python3 "$DEPTH_PREPROCESS_SCRIPT" \
  --input-topic /hal/chest_front_d457_camera/depth_lingbot \
  --processed-topic /hal/chest_front_d457_camera/depth_parkour_processed \
  --depth-max 2.5 \
  --print-every-sec 5.0 &
PREPROCESS_PID=$!

"$START_LINGBOT_SCRIPT" &
LINGBOT_PID=$!

cleanup() {
  kill "$LINGBOT_PID" "$PREPROCESS_PID" 2>/dev/null || true
}
trap cleanup EXIT

wait "$LINGBOT_PID"
