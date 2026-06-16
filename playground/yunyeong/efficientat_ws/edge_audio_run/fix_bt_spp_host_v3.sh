#!/usr/bin/env bash
# Robust Bluetooth Classic RFCOMM/SPP setup for Jetson Nano + BlueZ 5.x.
# Usage: ./fix_bt_spp_host_v3.sh [channel] [alias] [preferred_controller_mac]
# Example: ./fix_bt_spp_host_v3.sh 3 EdgeAudio-Jetson 74:D8:3E:48:1F:F1

set -u

CHANNEL="${1:-3}"
ALIAS="${2:-EdgeAudio-Jetson}"
PREFERRED_ADDR="${3:-74:D8:3E:48:1F:F1}"
BTD="/usr/lib/bluetooth/bluetoothd"

section() {
  echo
  echo "=== $* ==="
}

need_root_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1"
    return 1
  fi
}

section "Install/check required packages"
sudo apt-get update
sudo apt-get install -y bluez bluetooth rfkill bluez-tools || sudo apt-get install -y bluez bluetooth rfkill

if [ ! -x "$BTD" ]; then
  BTD="$(command -v bluetoothd || true)"
fi
if [ -z "$BTD" ] || [ ! -x "$BTD" ]; then
  echo "ERROR: bluetoothd not found"
  exit 1
fi

section "Force bluetoothd to run with --compat"
sudo mkdir -p /etc/systemd/system/bluetooth.service.d
sudo tee /etc/systemd/system/bluetooth.service.d/zz-edgeaudio-spp.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=${BTD} --compat
EOF

# Some systems activate BlueZ through the D-Bus alias. Creating the same drop-in is harmless
# on most Ubuntu/Jetson installs and helps when an alias-specific override was used before.
sudo mkdir -p /etc/systemd/system/dbus-org.bluez.service.d
sudo tee /etc/systemd/system/dbus-org.bluez.service.d/zz-edgeaudio-spp.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=${BTD} --compat
EOF

sudo systemctl daemon-reload
sudo systemctl stop bluetooth || true
sudo pkill -x bluetoothd || true
sleep 1
sudo rm -rf /run/sdp /var/run/sdp
sudo rfkill unblock bluetooth || true
sudo modprobe btusb || true
sudo systemctl start bluetooth
sleep 3

section "Verify bluetoothd process"
ps -ef | grep '[b]luetoothd' || true
if ! ps -ef | grep '[b]luetoothd' | grep -q -- '--compat'; then
  echo
  echo "ERROR: bluetoothd is still not running with --compat. Current unit follows:"
  systemctl cat bluetooth || true
  echo
  echo "Try a full reboot, then run this script again: sudo reboot"
  exit 2
fi

section "List current HCI controllers"
# Wait for controllers to appear after restart.
for _ in $(seq 1 10); do
  if hciconfig -a 2>/dev/null | grep -q '^hci[0-9]'; then
    break
  fi
  sleep 1
done

HCIOUT="$(hciconfig -a 2>/dev/null || true)"
printf '%s\n' "$HCIOUT"

if ! printf '%s\n' "$HCIOUT" | grep -q '^hci[0-9]'; then
  echo
  echo "ERROR: no hci controller found after restarting bluetooth."
  echo "Check: lsusb | grep -i -E 'bluetooth|intel|csr|realtek'"
  echo "Then replug the Bluetooth dongle or reboot Jetson."
  exit 3
fi

section "Choose target controller without assuming hci0"
TARGET_DEV="$(printf '%s\n' "$HCIOUT" | awk -v addr="$(printf '%s' "$PREFERRED_ADDR" | tr '[:lower:]' '[:upper:]')" '
  /^hci[0-9]+:/ { dev=$1; sub(":", "", dev) }
  /BD Address:/ {
    cur=toupper($3)
    if (cur == addr) { print dev; exit }
  }
')"

if [ -z "$TARGET_DEV" ]; then
  TARGET_DEV="$(printf '%s\n' "$HCIOUT" | awk -F: '/^hci[0-9]+:/ {print $1; exit}')"
fi

if [ -z "$TARGET_DEV" ]; then
  echo "ERROR: could not select an HCI controller."
  exit 4
fi

echo "Target controller: ${TARGET_DEV}"

section "Bring up target and disable other controllers for this test"
for DEV in $(printf '%s\n' "$HCIOUT" | awk -F: '/^hci[0-9]+:/ {print $1}'); do
  if [ "$DEV" = "$TARGET_DEV" ]; then
    sudo hciconfig "$DEV" up || true
  else
    sudo hciconfig "$DEV" down || true
  fi
done
sleep 1
sudo hciconfig "$TARGET_DEV" up
sudo hciconfig "$TARGET_DEV" name "$ALIAS" || true
sudo hciconfig "$TARGET_DEV" class 0x00010c || true
sudo hciconfig "$TARGET_DEV" piscan || true

TARGET_ADDR="$(hciconfig -a "$TARGET_DEV" | awk '/BD Address:/ {print $3; exit}')"
if [ -z "$TARGET_ADDR" ]; then
  echo "ERROR: could not read BD Address from ${TARGET_DEV}."
  hciconfig -a "$TARGET_DEV" || true
  exit 5
fi

echo "Target address: ${TARGET_ADDR}"

section "Make adapter pairable/discoverable"
bluetoothctl <<EOF || true
select ${TARGET_ADDR}
power on
system-alias ${ALIAS}
agent KeyboardDisplay
default-agent
pairable on
discoverable on
show
quit
EOF

section "Register SPP/Serial Port Profile on RFCOMM channel ${CHANNEL}"
# Clear stale local cache symptoms by calling browse once, then add SP.
sudo sdptool browse local >/tmp/edgeaudio_sdp_before.txt 2>&1 || true
sudo sdptool add --channel="${CHANNEL}" SP
sleep 1
sudo sdptool browse local > /tmp/edgeaudio_sdp_after.txt 2>&1 || true

if grep -qi 'Serial Port' /tmp/edgeaudio_sdp_after.txt; then
  sed -n '/Serial Port/,+18p' /tmp/edgeaudio_sdp_after.txt
  echo
  echo "SUCCESS: SPP is advertised. Use RFCOMM channel ${CHANNEL}."
else
  echo
  echo "ERROR: SPP was not found in local SDP after sdptool add."
  echo "--- /run/sdp and /var/run/sdp ---"
  ls -l /run/sdp /var/run/sdp 2>/dev/null || true
  echo "--- bluetooth unit ---"
  systemctl cat bluetooth || true
  echo "--- bluetooth journal tail ---"
  journalctl -u bluetooth -n 80 --no-pager || true
  exit 6
fi

section "Final HCI status"
hciconfig -a "$TARGET_DEV" || true

echo
cat <<EOF
Next:
  1) Pair Mac/Android with ${ALIAS}.
  2) Run bridge with the same channel:
     ./run_bluetooth_bridge_doa_from_host_v2.sh plughw:2,0 mn04_as 2 ${CHANNEL} 30 0 | tee ~/efficientat_ws/audio/bt_debug_console.txt
  3) Check JSON without app:
     grep 'BT_JSON' ~/efficientat_ws/audio/bt_debug_console.txt
EOF
