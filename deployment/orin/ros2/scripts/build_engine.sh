#!/usr/bin/env bash
set -eo pipefail

BASE=${BASE:-$PWD}
PRECISION=${PRECISION:-int8}
ONNX=${ONNX:-$BASE/onnx/lingbot_tokens128_36x64.onnx}
CALIB=${CALIB:-$BASE/calib/lingbot_calib_36x64.bin}
ENGINE=${ENGINE:-$BASE/trt/lingbot_tokens128_36x64_${PRECISION}.engine}
CACHE=${CACHE:-$BASE/trt/lingbot_tokens128_36x64_int8_calib.cache}
WORKSPACE_MB=${WORKSPACE_MB:-4096}

source /opt/ros/jazzy/setup.bash
[[ -n "${ROBOT_ENV:-}" ]] && source "$ROBOT_ENV"
source "$BASE/install/setup.bash"

if [[ "$PRECISION" == "int8" ]]; then
  exec ros2 run a3_lingbot_depth_ros build_lingbot_trt_engine \
    --onnx "$ONNX" \
    --engine "$ENGINE" \
    --precision int8 \
    --calib-bin "$CALIB" \
    --calib-cache "$CACHE" \
    --workspace-mb "$WORKSPACE_MB"
else
  exec ros2 run a3_lingbot_depth_ros build_lingbot_trt_engine \
    --onnx "$ONNX" \
    --engine "$ENGINE" \
    --precision fp16 \
    --workspace-mb "$WORKSPACE_MB"
fi
