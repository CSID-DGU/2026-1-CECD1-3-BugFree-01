#!/usr/bin/env bash
set -euo pipefail

MIC_DEV="${1:-plughw:2,0}"
PORT="${2:-8765}"
CHUNK_SECONDS="${3:-2}"

# UI / inference policy
THRESHOLD="${THRESHOLD:-0.70}"
MIN_DB="${MIN_DB:-45}"
TOPK="${TOPK:-5}"
DISPLAY_SCORE_THRESHOLD="${DISPLAY_SCORE_THRESHOLD:-0.70}"
DISPLAY_MIN_DB="${DISPLAY_MIN_DB:-45}"
DOA_ENABLE="${DOA_ENABLE:-1}"
DOA_NORTH_OFFSET="${DOA_NORTH_OFFSET:-0}"

HOST_WS="${HOST_WS:-$HOME/efficientat_ws}"
HOST_EFFAT="$HOST_WS/EfficientAT"
HOST_AUDIO="$HOST_WS/audio"
HOST_CKPT="${HOST_CKPT:-$HOST_EFFAT/checkpoints/best_model_jetson.pt}"
CONTAINER_CKPT="${CONTAINER_CKPT:-/workspace/EfficientAT/checkpoints/best_model_jetson.pt}"

mkdir -p "$HOST_AUDIO" "$HOST_EFFAT/checkpoints"

need_file() {
  if [ ! -f "$1" ]; then
    echo "[ERROR] missing file: $1" >&2
    exit 1
  fi
}

need_file "$HOST_EFFAT/jetson_wifi_bridge.py"
need_file "$HOST_EFFAT/jetson_wifi_bridge_finetuned.py"
need_file "$HOST_EFFAT/jetson_live_finetuned.py"
need_file "$HOST_CKPT"

cd "$HOST_WS"

echo "=== Host paths ==="
echo "HOST_WS=$HOST_WS"
echo "HOST_EFFAT=$HOST_EFFAT"
echo "HOST_CKPT=$HOST_CKPT"
echo "MIC_DEV=$MIC_DEV"
echo "PORT=$PORT"
echo "THRESHOLD=$THRESHOLD"
echo "MIN_DB=$MIN_DB"
echo "DOA_ENABLE=$DOA_ENABLE"

sudo docker run --runtime nvidia -it --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$HOST_WS:/workspace" \
  -e MIC_DEV="$MIC_DEV" \
  -e PORT="$PORT" \
  -e CHUNK_SECONDS="$CHUNK_SECONDS" \
  -e THRESHOLD="$THRESHOLD" \
  -e MIN_DB="$MIN_DB" \
  -e TOPK="$TOPK" \
  -e DISPLAY_SCORE_THRESHOLD="$DISPLAY_SCORE_THRESHOLD" \
  -e DISPLAY_MIN_DB="$DISPLAY_MIN_DB" \
  -e DOA_ENABLE="$DOA_ENABLE" \
  -e DOA_NORTH_OFFSET="$DOA_NORTH_OFFSET" \
  -e CONTAINER_CKPT="$CONTAINER_CKPT" \
  nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3 \
  bash -lc '
set -euo pipefail
export PYTHONUNBUFFERED=1

for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
  if [ -f "$f" ]; then
    sed -i.bak "/apt.kitware.com/s/^/# disabled /" "$f" || true
  fi
done

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  alsa-utils ffmpeg usbutils git curl python3-usb

cd /workspace/EfficientAT
mkdir -p /workspace/audio /workspace/EfficientAT/checkpoints

echo "=== Check required files inside Docker ==="
python3 -m py_compile /workspace/EfficientAT/jetson_wifi_bridge.py
python3 -m py_compile /workspace/EfficientAT/jetson_wifi_bridge_finetuned.py
python3 -m py_compile /workspace/EfficientAT/jetson_live_finetuned.py
ls -lah /workspace/EfficientAT/jetson_wifi_bridge.py
ls -lah /workspace/EfficientAT/jetson_wifi_bridge_finetuned.py
ls -lah /workspace/EfficientAT/jetson_live_finetuned.py
ls -lah "$CONTAINER_CKPT"

echo "=== Audio devices ==="
arecord -l || true

echo "=== Network addresses ==="
hostname -I || true
echo "Phone browser URL: http://<JETSON_IP>:${PORT}"

echo "=== Start Wi-Fi bridge + fine-tuned EfficientAT realtime inference ==="
python3 -u jetson_wifi_bridge_finetuned.py \
  --port "$PORT" \
  --log "/workspace/audio/wifi_bridge_finetuned.txt" \
  -- \
  python3 -u jetson_live_finetuned.py \
    --checkpoint "$CONTAINER_CKPT" \
    --efficientat-dir /workspace/EfficientAT \
    --device "$MIC_DEV" \
    --rate 16000 \
    --channels 6 \
    --channel-index 0 \
    --seconds "$CHUNK_SECONDS" \
    --topk "$TOPK" \
    --threshold "$THRESHOLD" \
    --min-db "$MIN_DB"
'
