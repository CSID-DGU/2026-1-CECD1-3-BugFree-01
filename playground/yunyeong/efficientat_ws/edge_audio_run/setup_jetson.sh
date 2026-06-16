#!/usr/bin/env bash
set -e
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-dev \
  python3-numpy \
  python3-scipy \
  python3-pycuda \
  alsa-utils \
  ffmpeg \
  unzip \
  git \
  htop \
  libopenblas-dev
python3 -m pip install --upgrade pip
python3 -m pip install --user tqdm
python3 - <<'PY'
import numpy, scipy, pycuda.driver, tensorrt
print('numpy', numpy.__version__)
print('scipy', scipy.__version__)
print('TensorRT', tensorrt.__version__)
print('Jetson Python dependency check: OK')
PY
