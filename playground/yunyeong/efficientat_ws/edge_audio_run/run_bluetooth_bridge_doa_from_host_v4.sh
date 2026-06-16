#!/usr/bin/env bash
set -euo pipefail

# Host wrapper. Run from Jetson host, not inside Docker.
# Usage: ./run_bluetooth_bridge_doa_from_host_v4.sh [mic_dev] [model] [seconds] [bt_channel] [min_db] [north_offset] [db_offset] [bt_bind_addr]

MIC_DEV="${1:-plughw:2,0}"
MODEL_NAME="${2:-mn04_as}"
CHUNK_SECONDS="${3:-2}"
BT_CHANNEL="${4:-3}"
MIN_DB="${5:-45}"
NORTH_OFFSET="${6:-0}"
DB_OFFSET="${7:-80}"
BT_BIND_ADDR="${8:-}"
if [ -z "$BT_BIND_ADDR" ]; then
  BT_BIND_ADDR="$(hciconfig hci0 2>/dev/null | awk '/BD Address/ {print $3; exit}' || true)"
fi
if [ -z "$BT_BIND_ADDR" ]; then
  BT_BIND_ADDR="74:D8:3E:48:1F:F1"
fi

cd "$HOME/efficientat_ws"
mkdir -p audio

SDP_MOUNT=()
if [ -S /var/run/sdp ]; then
  SDP_MOUNT=(-v /var/run/sdp:/var/run/sdp)
elif [ -e /var/run/sdp ]; then
  echo "ERROR: /var/run/sdp exists but is not a socket. Run fix_bt_spp_host_v2.sh first." >&2
  ls -ld /var/run/sdp >&2 || true
  exit 2
else
  echo "WARN: /var/run/sdp socket is missing. Bluetooth payload will still print as [BT_JSON], but SPP advertising may fail." >&2
fi

sudo docker run --runtime nvidia -it --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /var/run/dbus:/var/run/dbus \
  "${SDP_MOUNT[@]}" \
  -v "$HOME/efficientat_ws:/workspace" \
  -e MIC_DEV="$MIC_DEV" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e CHUNK_SECONDS="$CHUNK_SECONDS" \
  -e BT_CHANNEL="$BT_CHANNEL" \
  -e MIN_DB="$MIN_DB" \
  -e NORTH_OFFSET="$NORTH_OFFSET" \
  -e DB_OFFSET="$DB_OFFSET" \
  -e BT_BIND_ADDR="$BT_BIND_ADDR" \
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
      alsa-utils ffmpeg usbutils git curl \
      bluez bluetooth rfkill python3-usb

    cd /workspace/EfficientAT
    mkdir -p /workspace/audio

    echo "=== Audio devices ==="
    arecord -l || true

    echo "=== USB devices, for ReSpeaker DOA control ==="
    lsusb || true

    echo "=== Bluetooth adapter ==="
    rfkill unblock bluetooth || true
    hciconfig -a || true
    sdptool browse local | sed -n "/Serial Port/,+18p" || true

    echo "=== Start Bluetooth bridge + DOA + EfficientAT realtime inference ==="
    echo "Bluetooth Classic RFCOMM/SPP channel: ${BT_CHANNEL}"
    echo "Minimum displayed dB: ${MIN_DB} dB"
    echo "North offset: ${NORTH_OFFSET} degrees"
    echo "RFCOMM bind address: ${BT_BIND_ADDR}"

    python3 -u jetson_bluetooth_bridge_doa.py \
      --adapter hci0 \
      --channel "${BT_CHANNEL}" \
      --bind-addr "${BT_BIND_ADDR}" \
      --min-db "${MIN_DB}" \
      --north-offset "${NORTH_OFFSET}" \
      --log "/workspace/audio/bluetooth_bridge_${MODEL_NAME}_doa_db.txt" \
      -- \
      python3 -u jetson_live.py \
        --device "${MIC_DEV}" \
        --model_name "${MODEL_NAME}" \
        --rate 16000 \
        --channels 6 \
        --channel-index 0 \
        --seconds "${CHUNK_SECONDS}" \
        --topk 5 \
        --threshold 0.10 \
        --min-db "${MIN_DB}" \
        --db-offset "${DB_OFFSET}"
  '
