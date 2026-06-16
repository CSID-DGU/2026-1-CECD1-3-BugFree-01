#!/usr/bin/env bash
set -euo pipefail

# Jetson HOST-side Bluetooth Classic SPP repair script.
# Run on Jetson host, not inside Docker.
# Usage: ./fix_bt_spp_host_v2.sh [channel] [alias] [adapter]
# Example: ./fix_bt_spp_host_v2.sh 3 EdgeAudio-Jetson hci0

CHANNEL="${1:-3}"
ALIAS_NAME="${2:-EdgeAudio-Jetson}"
ADAPTER="${3:-hci0}"
INDEX="${ADAPTER#hci}"

if [ "${EUID}" -ne 0 ]; then
  exec sudo bash "$0" "$@"
fi

log() { printf '\n=== %s ===\n' "$*"; }

log "Install packages"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y bluez bluetooth rfkill

BTD="/usr/lib/bluetooth/bluetoothd"
if [ ! -x "$BTD" ]; then
  BTD="$(command -v bluetoothd || true)"
fi
if [ -z "${BTD:-}" ] || [ ! -x "$BTD" ]; then
  echo "ERROR: bluetoothd not found" >&2
  exit 1
fi

log "Enable bluetoothd --compat"
mkdir -p /etc/systemd/system/bluetooth.service.d
cat > /etc/systemd/system/bluetooth.service.d/compat.conf <<EOF
[Service]
ExecStart=
ExecStart=${BTD} --compat
EOF

log "Patch /etc/bluetooth/main.conf visibility defaults"
python3 - <<PY
from pathlib import Path
p = Path('/etc/bluetooth/main.conf')
text = p.read_text(encoding='utf-8') if p.exists() else ''
if '[General]' not in text:
    text = '[General]\n' + text
pairs = {
    'Name': '${ALIAS_NAME}',
    'Class': '0x00010C',
    'DiscoverableTimeout': '0',
    'PairableTimeout': '0',
}
lines = text.splitlines()
out = []
in_general = False
seen = set()
for line in lines:
    stripped = line.strip()
    if stripped.startswith('[') and stripped.endswith(']'):
        if in_general:
            for k, v in pairs.items():
                if k not in seen:
                    out.append(f'{k} = {v}')
                    seen.add(k)
        in_general = (stripped == '[General]')
        out.append(line)
        continue
    if in_general:
        key = None
        raw = stripped[1:].strip() if stripped.startswith('#') else stripped
        if '=' in raw:
            key = raw.split('=', 1)[0].strip()
        if key in pairs:
            if key not in seen:
                out.append(f'{key} = {pairs[key]}')
                seen.add(key)
            continue
    out.append(line)
if in_general:
    for k, v in pairs.items():
        if k not in seen:
            out.append(f'{k} = {v}')
            seen.add(k)
p.write_text('\n'.join(out) + '\n', encoding='utf-8')
PY

log "Restart bluetooth and clean stale SDP path"
systemctl stop bluetooth || true
pkill bluetoothd || true
# If Docker previously mounted /var/run/sdp before the SDP socket existed,
# Docker may have created it as a directory. That breaks sdptool/SDP.
rm -rf /run/sdp /var/run/sdp
systemctl daemon-reload
systemctl enable bluetooth >/dev/null 2>&1 || true
systemctl start bluetooth
sleep 2
rfkill unblock bluetooth || true

log "bluetoothd process and SDP socket"
ps -ef | grep '[b]luetoothd' || true
ls -l /run/sdp /var/run/sdp 2>/dev/null || true
if [ -e /var/run/sdp ] && [ ! -S /var/run/sdp ]; then
  echo "ERROR: /var/run/sdp exists but is not a Unix socket. Remove it and restart bluetooth." >&2
  exit 2
fi

log "Keep only ${ADAPTER} active during test"
for devpath in /sys/class/bluetooth/hci*; do
  [ -e "$devpath" ] || continue
  dev="$(basename "$devpath")"
  idx="${dev#hci}"
  if [ "$dev" != "$ADAPTER" ]; then
    btmgmt --index "$idx" power off >/dev/null 2>&1 || true
    hciconfig "$dev" down >/dev/null 2>&1 || true
  fi
done

log "Configure ${ADAPTER} / index ${INDEX}"
btmgmt --index "$INDEX" power off >/dev/null 2>&1 || true
btmgmt --index "$INDEX" bredr on >/dev/null 2>&1 || true
btmgmt --index "$INDEX" le on >/dev/null 2>&1 || true
btmgmt --index "$INDEX" ssp on >/dev/null 2>&1 || true
btmgmt --index "$INDEX" bondable on >/dev/null 2>&1 || true
btmgmt --index "$INDEX" power on >/dev/null 2>&1 || true
btmgmt --index "$INDEX" connectable on >/dev/null 2>&1 || true
btmgmt --index "$INDEX" discov yes >/dev/null 2>&1 || true
btmgmt --index "$INDEX" name "$ALIAS_NAME" >/dev/null 2>&1 || true

hciconfig "$ADAPTER" up
hciconfig "$ADAPTER" name "$ALIAS_NAME" || true
hciconfig "$ADAPTER" class 0x00010c || true
hciconfig "$ADAPTER" piscan || true
ADDR="$(cat "/sys/class/bluetooth/${ADAPTER}/address")"

log "bluetoothctl pairing agent"
bluetoothctl <<EOF || true
select ${ADDR}
power on
agent KeyboardDisplay
default-agent
pairable on
discoverable on
show
quit
EOF

log "Register SPP Serial Port Profile on RFCOMM channel ${CHANNEL}"
set +e
SDP_OUTPUT="$(sdptool add --channel="${CHANNEL}" SP 2>&1)"
SDP_RC=$?
set -e
printf '%s\n' "$SDP_OUTPUT"
echo "sdptool_exit=${SDP_RC}"

log "Local SDP browse result"
SDP_BROWSE="$(sdptool browse local 2>&1 || true)"
printf '%s\n' "$SDP_BROWSE" | sed -n '/Serial Port/,+18p'

if ! printf '%s\n' "$SDP_BROWSE" | grep -qi 'Serial Port'; then
  echo
  echo "ERROR: Serial Port Profile is still not visible in local SDP." >&2
  echo "Check these diagnostics:" >&2
  echo "  systemctl cat bluetooth | grep ExecStart" >&2
  echo "  ps -ef | grep '[b]luetoothd'" >&2
  echo "  ls -l /var/run/sdp /run/sdp" >&2
  echo "  journalctl -u bluetooth -n 80 --no-pager" >&2
  exit 3
fi

log "Adapter status"
hciconfig -a "$ADAPTER" || true
btmgmt --index "$INDEX" info || true

echo
echo "OK: ${ALIAS_NAME} is discoverable and SPP is advertised on channel ${CHANNEL}."
echo "Use the same channel when running run_bluetooth_bridge_doa_from_host_v2.sh."
