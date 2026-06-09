#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-inspire_humble}"
DATA_ROOT="${DATA_ROOT:-/mnt/data}"

docker run -it \
  --gpus all \
  --net=host \
  --ipc=host \
  --shm-size=8g \
  --privileged \
  -v /dev:/dev \
  -v /run/udev:/run/udev:ro \
  -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7} \
  -e RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp} \
  -v "${PROJECT_ROOT}:/workspace/zhicheng_ws" \
  -v "${DATA_ROOT}:/mnt/data" \
  --name "${CONTAINER_NAME}" \
  inspire_humble:base
