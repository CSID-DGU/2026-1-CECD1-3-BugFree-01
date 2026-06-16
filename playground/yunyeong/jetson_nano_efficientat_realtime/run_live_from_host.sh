#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_live_from_host.sh plughw:2,0 mn10_as
#   ./run_live_from_host.sh plughw:2,0 mn04_as
#
# Default:
#   mic device = plughw:2,0
#   model      = mn10_as

MIC_DEV="${1:-plughw:2,0}"
MODEL_NAME="${2:-mn10_as}"
CHUNK_SECONDS="${3:-2}"

cd "$HOME/efficientat_ws"
mkdir -p audio

sudo docker run --runtime nvidia -it --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$HOME/efficientat_ws:/workspace" \
  -e MIC_DEV="$MIC_DEV" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e CHUNK_SECONDS="$CHUNK_SECONDS" \
  nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3 \
  bash -lc '
    set -e

    apt update
    apt install -y alsa-utils ffmpeg usbutils git curl

    cd /workspace/EfficientAT
    mkdir -p /workspace/audio

    echo "=== Audio devices ==="
    arecord -l

    echo "=== CUDA check ==="
    python3 -c "import torch; print(\"cuda:\", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\")"

    echo "=== Start EfficientAT realtime inference ==="
    echo "device: ${MIC_DEV}"
    echo "model: ${MODEL_NAME}"

    python3 jetson_live.py \
      --device "$MIC_DEV" \
      --model_name "$MODEL_NAME" \
      --rate 16000 \
      --channels 6 \
      --channel-index 0 \
      --seconds "$CHUNK_SECONDS" \
      --topk 5 \
      --threshold 0.10 \
      | tee "/workspace/audio/live_${MODEL_NAME}.txt"
  '
