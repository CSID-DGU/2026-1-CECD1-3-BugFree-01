# Jetson Nano 실시간 추론 및 시각화 실행 코드
```
cd ~/efficientat_ws
./run_wifi_bridge_finetuned_from_host.sh plughw:2,0 8765
```

# Jetson Nano EfficientAT Realtime Test

Jetson Nano + reSpeaker Mic Array v3.0 환경에서 EfficientAT pretrained 모델의 온디바이스 실시간 추론 가능성을 테스트한 코드입니다.

## 목적

- Jetson Nano에서 EfficientAT inference 가능 여부 확인
- reSpeaker Mic Array v3.0으로 실시간 오디오 입력 수집
- `mn04_as`, `mn05_as`, `mn10_as` 모델별 추론 시간 비교
- Jetson Nano에서는 학습하지 않고, 녹음/전처리/inference/classification만 수행

## 현재 Jetson 파일 구조

Host 기준:

```bash
~/efficientat_ws/
├── EfficientAT
├── audio
└── edge_audio_run
```

실행 명령어
```
python3 -u jetson_wifi_bridge_doa.py \
  --port 8765 \
  --log /workspace/audio/wifi_bridge_mn04_as_doa.txt \
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
```
