#!/usr/bin/env bash
set -euo pipefail

cd /lus/eagle/projects/datascience_collab/eku/Carla
module load conda
conda activate carla

# Configuration (override with env vars)
CARLA_PORT=${CARLA_PORT:-2000}
CARLA_HOST=${CARLA_HOST:-localhost}
FRAME_DIR=${CARLA_FRAME_DIR:-frames}
OUTPUT_VIDEO=${OUTPUT_VIDEO:-run.mp4}
FPS=${CARLA_FRAME_FPS:-10}

# Launch CARLA server headless
DISPLAY=${DISPLAY-} ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=${CARLA_PORT} &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT
echo "Started CARLA server (pid ${SERVER_PID}) on ${CARLA_HOST}:${CARLA_PORT}"

# Give the server a moment to come up
sleep 5

# Run the driver (set --train True to train)
python Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/continuous_driver.py --exp-name ppo --train False --town Town02 --test-timesteps 100

# Assemble frames into a video without ffmpeg (uses OpenCV)
python assemble_video.py "$FRAME_DIR" "$OUTPUT_VIDEO" "$FPS"