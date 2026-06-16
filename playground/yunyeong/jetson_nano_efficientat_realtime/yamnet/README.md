# YAMNet TFLite on Jetson Nano

Jetson Nano + reSpeaker Mic Array 환경에서 YAMNet TFLite 모델을 이용해 실시간 소리 분류를 수행하고, 기존 Wi-Fi bridge를 통해 브라우저로 결과를 확인한다.

## 1. Docker 실행

Host에서 실행한다.

```bash
cd ~/efficientat_ws
mkdir -p audio

sudo docker run --runtime nvidia -it --rm \
  --network=host \
  --ipc=host \
  --privileged \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  -v "$HOME/efficientat_ws:/workspace" \
  nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
