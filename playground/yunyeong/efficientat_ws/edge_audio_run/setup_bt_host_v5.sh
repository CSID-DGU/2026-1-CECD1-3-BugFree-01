#!/usr/bin/env bash
set -euo pipefail

# Run on Jetson host, not inside Docker.
# Usage: ./setup_bt_host_v5.sh [alias] [adapter]

ALIAS="${1:-EdgeAudio-Jetson}"
ADAPTER="${2:-hci0}"

sudo apt-get update
sudo apt-get install -y bluez bluetooth rfkill

BTD="/usr/lib/bluetooth/bluetoothd"
if [ ! -x "$BTD" ]; then
  BTD="$(command -v bluetoothd)"
fi

sudo mkdir -p /etc/systemd/system/bluetooth.service.d
sudo tee /etc/systemd/system/bluetooth.service.d/zz-edgeaudio-spp.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=${BTD} --compat
EOF

# Avoid stale /run/sdp directories from previous failed attempts.
sudo systemctl daemon-reload
sudo systemctl restart bluetooth
sleep 1
sudo rfkill unblock bluetooth
sudo hciconfig hci1 down 2>/dev/null || true
sudo hciconfig "$ADAPTER" up
sudo hciconfig "$ADAPTER" name "$ALIAS" || true
sudo hciconfig "$ADAPTER" class 0x00010c || true
sudo hciconfig "$ADAPTER" piscan

# Delete stale Serial Port SDP records. The v5 Python server creates one after bind.
HANDLES="$(sudo sdptool browse local 2>/dev/null | awk '
  /Service Name: Serial Port/ {in_sp=1; next}
  in_sp && /Service RecHandle:/ {print $3; in_sp=0}
' || true)"
if [ -n "$HANDLES" ]; then
  for h in $HANDLES; do
    sudo sdptool del "$h" 2>/dev/null || true
  done
fi

ADDR="$(hciconfig "$ADAPTER" | awk '/BD Address/ {print $3; exit}')"
bluetoothctl <<EOF || true
select ${ADDR}
power on
system-alias ${ALIAS}
agent KeyboardDisplay
default-agent
pairable on
discoverable on
show
EOF

echo "=== bluetoothd ==="
ps -ef | grep '[b]luetoothd' || true

echo "=== ${ADAPTER} ==="
hciconfig -a "$ADAPTER"

echo "=== SDP Serial Port records before server start ==="
sudo sdptool browse local | sed -n '/Serial Port/,+18p' || true

echo "DONE: Start run_bluetooth_bridge_doa_from_host_v5.sh next."
