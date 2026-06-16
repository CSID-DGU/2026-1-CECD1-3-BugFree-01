# live_inference_refactored_ble_independent

`realtime_inference.py`, `realtime_inference_ble.py`, `realtime_inference_ble_doa.py`에 의존하지 않도록 기존 live inference + BLE 코드를 객체/파일 단위로 분리한 버전입니다.

## 실행

BLE GATT 서버까지 열고 실행:

```bash
python3 main.py --efficientat-dir /workspace/EfficientAT
```

장치 목록 확인:

```bash
python3 main.py --list-devices
```

콘솔 추론만 실행하고 BLE를 열지 않기:

```bash
python3 main.py --no-ble --efficientat-dir /workspace/EfficientAT
```

저음량 청크는 모델 추론까지 가지 않고 스킵:

```bash
python3 main.py --min-db 30 --skip-low-db
```

BLE 이름 변경:

```bash
python3 main.py --ble-name JHello
```

## 주요 구조

- `main.py`: 전체 객체 생성, BLE 서버 시작/종료, 실행 흐름 제어
- `cli.py`: 실행 인자 처리
- `constants.py`: 추론/전처리/라벨 상수
- `device_finder.py`: 입력 장치 탐색
- `microphone_module.py`: 마이크 스트림 관리
- `audio_queue.py`: 입력 콜백 데이터 큐
- `audio_buffer.py`: 청크 단위 버퍼링
- `audio_level_meter.py`: dBFS 계산
- `db_threshold_gate.py`: 임계값 판단
- `audio_preprocessor.py`: 작은 신호 감소 / 큰 신호 강조
- `efficientat_loader.py`: EfficientAT 모델 및 Mel frontend 로딩
- `label_mapper.py`: AudioSet 라벨을 사용자 정의 라벨로 매핑
- `model_inference.py`: 모델 입력 길이 조정 및 추론
- `audio_stream_controller.py`: 실시간 처리 루프 제어 및 BLE publish 호출
- `ble_constants.py`: BLE/DBus UUID 및 인터페이스 상수
- `ble_utils.py`: DBus 유틸리티
- `ble_gatt_objects.py`: GATT Application / Service / Characteristic / Advertisement 객체
- `ble_adapter.py`: Bluetooth adapter 탐색 및 전원 활성화
- `ble_inference_server.py`: BLE GATT 서버 시작, 광고 등록, notify 전송
- `ble_result_builder.py`: 앱으로 보낼 JSON payload 생성

## 필요 외부 요소

- EfficientAT 저장소
- `sounddevice`, `numpy`, `torch`, `torchaudio`
- BLE 사용 시 `dbus`, `PyGObject`, BlueZ, Bluetooth adapter 권한
