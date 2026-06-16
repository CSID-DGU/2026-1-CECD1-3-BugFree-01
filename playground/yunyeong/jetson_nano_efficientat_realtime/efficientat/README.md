# [MacBook] Jetson Nano에 SSH 접속 + 8765 포트포워딩
ip 확인

# [Jetson Nano] Docker + EfficientAT Wi-Fi bridge 실행
cd ~/efficientat_ws

sudo docker run --runtime nvidia -it --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$HOME/efficientat_ws:/workspace" \
  nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3 \
  bash -lc '
    set -e

    for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do
      if [ -f "$f" ]; then
        sed -i.bak "/apt.kitware.com/s/^/# disabled /" "$f"
      fi
    done

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y alsa-utils ffmpeg usbutils git curl

    cd /workspace/EfficientAT
    mkdir -p /workspace/audio

    arecord -l || true

    python3 -u jetson_wifi_bridge.py \
      --port 8765 \
      --log /workspace/audio/wifi_bridge_mn04_as.txt \
      -- \
      python3 -u jetson_live.py \
        --device plughw:2,0 \
        --model_name mn04_as \
        --rate 16000 \
        --channels 6 \
        --channel-index 0 \
        --seconds 2 \
        --topk 5 \
        --threshold 0.10
  '

# [MacBook] 브라우저에서 결과 확인
open http://localhost:8765
