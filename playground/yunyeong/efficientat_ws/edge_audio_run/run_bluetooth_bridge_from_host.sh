#!/usr/bin/env bash
set -euo pipefail

MIC_DEV="${1:-plughw:2,0}"
MODEL_NAME="${2:-mn04_as}"
CHUNK_SECONDS="${3:-2}"
BT_CHANNEL="${4:-1}"

cd "$HOME/efficientat_ws"
mkdir -p audio

sudo docker run --runtime nvidia -it --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /var/run/dbus:/var/run/dbus \
  -v "$HOME/efficientat_ws:/workspace" \
  -e MIC_DEV="$MIC_DEV" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e CHUNK_SECONDS="$CHUNK_SECONDS" \
  -e BT_CHANNEL="$BT_CHANNEL" \
  nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3 \
  bash -lc '
    set -e

    echo "=== Disable broken Kitware apt repo inside container, if present ==="
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
      if [ -f "$f" ]; then
        sed -i.bak "/apt.kitware.com/s/^/# disabled /" "$f" || true
      fi
    done

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      alsa-utils ffmpeg usbutils git curl bluez bluetooth rfkill

    cd /workspace/EfficientAT
    mkdir -p /workspace/audio

    echo "=== Audio devices ==="
    arecord -l || true

    echo "=== Bluetooth adapter ==="
    rfkill unblock bluetooth || true
    hciconfig -a || true
    bluetoothctl show || true

    echo "=== Start Bluetooth bridge + EfficientAT realtime inference ==="
    echo "Bluetooth Classic RFCOMM/SPP channel: ${BT_CHANNEL}"
    echo "Pair the phone/PC with the Jetson host first, then connect with a Bluetooth serial client."

    python3 -u jetson_bluetooth_bridge.py \
      --adapter hci0 \
      --channel "${BT_CHANNEL}" \
      --log "/workspace/audio/bluetooth_bridge_${MODEL_NAME}.txt" \
      -- \
      python3 -u jetson_live.py \
        --device "${MIC_DEV}" \
        --model_name "${MODEL_NAME}" \
        --rate 16000 \
        --channels 6 \
        --channel-index 0 \
        --seconds "${CHUNK_SECONDS}" \
        --topk 5 \
        --threshold 0.10
  '
