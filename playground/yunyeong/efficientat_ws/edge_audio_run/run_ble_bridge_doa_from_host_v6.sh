#!/usr/bin/env bash
set -euo pipefail

MIC_DEV="${1:-plughw:2,0}"
MODEL_NAME="${2:-mn04_as}"
CHUNK_SECONDS="${3:-2}"
MIN_DB="${4:-30}"
NORTH_OFFSET="${5:-0}"
DB_OFFSET="${6:-80}"
BT_ADAPTER="${7:-hci0}"
BLE_NAME="${8:-EdgeAudio-Jetson}"

WORKDIR="$HOME/efficientat_ws"
IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3"
mkdir -p "$WORKDIR/audio"

printf '\n=== Stop stale EdgeAudio containers/processes ===\n'
sudo pkill -f 'jetson_ble_bridge_doa.py' 2>/dev/null || true
sudo pkill -f 'jetson_bluetooth_bridge_doa.py' 2>/dev/null || true
sudo pkill -f 'jetson_live.py' 2>/dev/null || true
sudo rfcomm release all 2>/dev/null || true

printf '\n=== Run BLE bridge + DOA + EfficientAT realtime inference ===\n'
echo "BLE name: ${BLE_NAME}"
echo "BLE adapter: ${BT_ADAPTER}"
echo "Minimum displayed dB: ${MIN_DB} dB"
echo "North offset: ${NORTH_OFFSET} degrees"

echo "Host log: $WORKDIR/audio/ble_debug_console.txt"

sudo docker run --runtime nvidia -i --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /run/dbus:/run/dbus \
  -v /var/run/dbus:/var/run/dbus \
  -v "$WORKDIR:/workspace" \
  "$IMAGE" \
  bash -lc "
set -euo pipefail
export DBUS_SYSTEM_BUS_ADDRESS=unix:path=/run/dbus/system_bus_socket

# Disable broken Kitware apt repo inside container, if present.
if [ -d /etc/apt/sources.list.d ]; then
  for f in /etc/apt/sources.list.d/*kitware* /etc/apt/sources.list.d/*cmake*; do
    [ -e \"\$f\" ] && mv \"\$f\" \"\$f.disabled\" || true
  done
fi

apt-get update
apt-get install -y alsa-utils ffmpeg usbutils git curl python3-usb python3-dbus python3-gi gir1.2-glib-2.0 dbus bluez bluetooth rfkill

cd /workspace/EfficientAT
mkdir -p /workspace/audio

echo '=== Audio devices ==='
arecord -l || true

echo '=== USB devices, for ReSpeaker DOA control ==='
lsusb || true

echo '=== Bluetooth adapter ==='
hciconfig -a '${BT_ADAPTER}' || hciconfig -a || true

python3 -u jetson_ble_bridge_doa.py \
  --adapter '${BT_ADAPTER}' \
  --name '${BLE_NAME}' \
  --min-db '${MIN_DB}' \
  --north-offset '${NORTH_OFFSET}' \
  --log '/workspace/audio/ble_bridge_${MODEL_NAME}_doa_db.txt' \
  -- \
  python3 -u jetson_live.py \
    --device '${MIC_DEV}' \
    --model_name '${MODEL_NAME}' \
    --rate 16000 \
    --channels 6 \
    --channel-index 0 \
    --seconds '${CHUNK_SECONDS}' \
    --topk 5 \
    --threshold 0.10 \
    --min-db '${MIN_DB}' \
    --db-offset '${DB_OFFSET}'
" | tee "$WORKDIR/audio/ble_debug_console.txt"
