#!/usr/bin/env bash
set -euo pipefail

ALIAS="${1:-EdgeAudio-Jetson}"
ADAPTER="${2:-hci0}"
BT_ADDR="${3:-74:D8:3E:48:1F:F1}"

printf '\n=== Stop stale EdgeAudio Bluetooth/Docker processes ===\n'
sudo pkill -f 'jetson_ble_bridge_doa.py' 2>/dev/null || true
sudo pkill -f 'jetson_bluetooth_bridge_doa.py' 2>/dev/null || true
sudo pkill -f 'jetson_live.py' 2>/dev/null || true
sudo rfcomm release all 2>/dev/null || true

printf '\n=== Install/verify host Bluetooth packages ===\n'
sudo apt-get update
sudo apt-get install -y bluez bluetooth rfkill dbus

BTD="/usr/lib/bluetooth/bluetoothd"
if [ ! -x "$BTD" ]; then
  BTD="$(command -v bluetoothd)"
fi

printf '\n=== Enable bluetoothd --compat --experimental for BLE GATT peripheral ===\n'
sudo mkdir -p /etc/systemd/system/bluetooth.service.d
sudo tee /etc/systemd/system/bluetooth.service.d/zz-edgeaudio-ble.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=${BTD} --compat --experimental
EOF

sudo systemctl daemon-reload
sudo systemctl restart bluetooth
sudo rfkill unblock bluetooth
sleep 1

printf '\n=== Configure Bluetooth adapter ===\n'
# Keep only the target adapter active during the demo; hci1 has caused confusion on this Jetson.
for h in /sys/class/bluetooth/hci*; do
  [ -e "$h" ] || continue
  name="$(basename "$h")"
  if [ "$name" != "$ADAPTER" ]; then
    sudo hciconfig "$name" down || true
  fi
done

sudo hciconfig "$ADAPTER" up
sudo hciconfig "$ADAPTER" name "$ALIAS" || true
sudo hciconfig "$ADAPTER" class 0x00010c || true
sudo hciconfig "$ADAPTER" piscan || true

bluetoothctl <<EOF || true
select ${BT_ADDR}
power on
system-alias ${ALIAS}
agent KeyboardDisplay
default-agent
pairable on
discoverable on
show
quit
EOF

printf '\n=== Verify BlueZ D-Bus interfaces ===\n'
dbus-send --system --dest=org.bluez --print-reply / org.freedesktop.DBus.ObjectManager.GetManagedObjects \
  | grep -E 'GattManager1|LEAdvertisingManager1' || {
    echo '[ERROR] GattManager1/LEAdvertisingManager1 not visible. Reboot once, then run this script again.' >&2
    exit 1
  }

printf '\n=== Adapter status ===\n'
hciconfig -a "$ADAPTER"
ps -ef | grep '[b]luetoothd' || true

echo
printf 'DONE. Next: ./run_ble_bridge_doa_from_host_v6.sh plughw:2,0 mn04_as 2 30 0 80 %s %s\n' "$ADAPTER" "$ALIAS"
