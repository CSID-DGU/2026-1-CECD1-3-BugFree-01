#!/usr/bin/env bash
set -euxo pipefail

CHANNEL="${1:-1}"
ALIAS="${2:-EdgeAudio-Jetson}"
CTRL_ADDR="74:D8:3E:48:1F:F1"

sudo apt-get update
sudo apt-get install -y bluez bluetooth rfkill

BTD="/usr/lib/bluetooth/bluetoothd"
if [ ! -x "$BTD" ]; then
  BTD="$(command -v bluetoothd)"
fi

sudo mkdir -p /etc/systemd/system/bluetooth.service.d

sudo tee /etc/systemd/system/bluetooth.service.d/compat.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=${BTD} --compat
EOF

sudo systemctl daemon-reload
sudo systemctl restart bluetooth
sudo rfkill unblock bluetooth

# hci1이 같이 떠 있어서 테스트 중에는 hci0만 쓰게 정리
sudo hciconfig hci1 down || true

sudo hciconfig hci0 up
sudo hciconfig hci0 name "${ALIAS}"
sudo hciconfig hci0 class 0x00010c || true
sudo hciconfig hci0 piscan

bluetoothctl <<EOF || true
select ${CTRL_ADDR}
power on
system-alias ${ALIAS}
agent KeyboardDisplay
default-agent
pairable on
discoverable on
discoverable-timeout 0
show
EOF

# 여기서 실패 메시지가 나오면 그게 진짜 원인
sudo sdptool add --channel="${CHANNEL}" SP

echo
echo "=== Local SDP Serial Port entry: 아래에 Serial Port / RFCOMM / Channel ${CHANNEL}가 보여야 정상 ==="
sudo sdptool browse local | sed -n '/Serial Port/,+18p'

echo
echo "=== hci0 status ==="
hciconfig -a hci0

echo
echo "DONE"
