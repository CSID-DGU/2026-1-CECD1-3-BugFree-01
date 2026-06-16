## 파일 위치 정리

현재 `efficientat_ws` 루트에는 크게 `EfficientAT/`, `audio/`, `edge_audio_run/`, 그리고 Host 실행 스크립트 2개가 있음. 

```
efficientat_ws/
├── run_wifi_bridge_finetuned_from_host.sh   # 최종 실행용
├── run_wifi_bridge_from_host.sh           # pretrained 비교용
├── EfficientAT/                             # 모델/추론/UI 코드
├── audio/                                   # 실행 로그 저장
└── edge_audio_run/                          # 이전 실험 코드
```


## 최종 시연에 필요한 파일

| 경로                                             | 역할                                                                                      |
| ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| `run_wifi_bridge_finetuned_from_host.sh`       | **최종 실행 스크립트**. Host에서 실행하면 Docker를 띄우고 fine-tuned EfficientAT 실시간 추론 + Wi-Fi 웹 UI까지 실행 |
| [`EfficientAT/jetson_live_finetuned.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_live_finetuned.py)         | reSpeaker로 오디오 녹음하고 fine-tuned checkpoint로 실시간 소리 분류                                    |
| [`EfficientAT/jetson_wifi_bridge_finetuned.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_wifi_bridge_finetuned.py)  | 추론 결과를 웹 UI로 보여주고, reSpeaker DOA 방향값을 붙여서 표시                                            |
| [`EfficientAT/jetson_wifi_bridge.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_wifi_bridge.py)            | 기본 웹 UI 서버. fine-tuned bridge가 내부적으로 이 파일을 import해서 사용                                  |
| [`EfficientAT/checkpoints/best_model_jetson.pt`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/checkpoints/best_model_jetson.pt) | fine-tuned 모델 checkpoint.                             |
| [`EfficientAT/checkpoints/labels.json`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/checkpoints/labels.json)          | fine-tuned 모델 라벨 정보 확인용                                                                 |

`run_wifi_bridge_finetuned_from_host.sh` 자체가 필요한 파일로 `jetson_live_finetuned.py`, `jetson_wifi_bridge.py`, `jetson_wifi_bridge_finetuned.py`, `best_model_jetson.pt`를 검사하고 있음. 

## 실행 흐름

```text
run_wifi_bridge_finetuned_from_host.sh
  -> Docker 실행
  -> /workspace/EfficientAT 로 이동
  -> jetson_wifi_bridge_finetuned.py 실행
  -> 내부에서 jetson_live_finetuned.py 실행
  -> reSpeaker 입력 추론
  -> 휴대폰 브라우저에서 결과 확인
```

## 비교/테스트용 파일

| 경로                                                                    | 역할                                                     |
| --------------------------------------------------------------------- | ------------------------------------------------------ |
| [`run_wifi_bridge_from_host.sh`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/run_wifi_bridge_finetuned_from_host.sh)                                        | fine-tuned가 아니라 pretrained EfficientAT 모델로 Wi-Fi UI 실행 |
| [`EfficientAT/jetson_live.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_live.py)                                          | pretrained `mn04_as`, `mn05_as`, `mn10_as` 실시간 추론용     |
| [`EfficientAT/jetson_infer.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_infer.py)                                         | pretrained 모델로 단일 wav 파일 추론 테스트                        |
| [`EfficientAT/jetson_infer_finetuned.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_infer_finetuned.py)                               | fine-tuned checkpoint로 단일 wav 파일 추론 테스트                |
| [`EfficientAT/jetson_live_with_doa.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_live_with_doa.py)                                 | DOA 붙인 이전 실험용                                          |
| [`EfficientAT/jetson_wifi_bridge_doa.py`](https://github.com/26-DGU-CECD/EdgeAudioRecognition/blob/playground/playground/yunyeong/efficientat_ws/EfficientAT/jetson_wifi_bridge_doa.py)                               | DOA 붙인 이전 Wi-Fi bridge 실험용                             |


`run_wifi_bridge_from_host.sh`는 pretrained 모델명을 받아 `jetson_wifi_bridge.py`와 `jetson_live.py`를 실행하는 비교용 스크립트로 보면 됨.

## EfficientAT 기본 폴더

```text
EfficientAT/
├── models/       # EfficientAT 모델 구조
├── helpers/      # 모델 로딩, label 등 유틸
├── metadata/     # AudioSet 등 메타데이터
├── resources/    # 원본 예제/리소스
└── requirements.txt
```

## 로그/결과 파일 위치

```text
audio/
├── wifi_bridge_finetuned.txt
├── wifi_bridge_mn04_as.txt
├── live_mn04_as.txt
├── live_mn10_as.txt
├── live_yamnet.txt
└── bluetooth/ble 관련 로그들
```

`audio/`는 실행 결과 로그 저장용 `wifi_bridge_finetuned.txt`, `live_mn04_as.txt`, `live_mn10_as.txt`, `live_yamnet.txt` 같은 로그들이 들어 있음. 

## 예전 실험/백업으로 분리해서 보면 되는 파일

```text
edge_audio_run/
├── build_engine.sh
├── infer_wav.py
├── realtime_arecord.py
├── run_bluetooth_bridge_*.sh
├── setup_bt_*.sh
└── setup_ble_*.sh
```

`edge_audio_run/`은 TensorRT, Bluetooth, BLE, arecord 기반 이전 실험 코드 모음으로 보면 됨. 
