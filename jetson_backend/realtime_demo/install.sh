#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "EfficientAT" ]; then
  git clone https://github.com/fschmid56/EfficientAT.git
else
  echo "EfficientAT already exists; skipping clone."
fi

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3-dbus python3-gi gir1.2-glib-2.0 bluez
fi

pip install torch torchaudio sounddevice "numpy<2"

# EfficientAT's MobileNet model imports torchvision.ops.misc.
pip install torchvision
