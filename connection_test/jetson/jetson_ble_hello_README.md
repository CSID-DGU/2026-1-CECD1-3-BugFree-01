# Jetson Orin Nano BLE hello

이 예제는 Jetson Orin Nano Developer Kit을 BLE Peripheral, 즉 광고하는 장치로 만들고 휴대폰이 연결해서 GATT characteristic을 읽거나 notify 구독하면 `안녕`을 받게 합니다.

## 왜 BLE 방식인가

iPhone은 일반적인 Bluetooth SPP/RFCOMM 시리얼 장치처럼 임의의 문자열을 바로 받는 구조가 아닙니다. iOS에서는 BLE GATT가 가장 쉬운 테스트 경로입니다.

Android도 BLE GATT 테스트 앱이 많아서 같은 코드로 확인할 수 있습니다. 휴대폰의 시스템 Bluetooth 설정에서 페어링만 한다고 문자열이 화면에 뜨는 것은 아니고, 앱이 BLE service/characteristic을 읽거나 notify 구독해야 합니다.

## Jetson에서 실행

필요 패키지가 없다면 설치합니다.

```bash
sudo apt update
sudo apt install -y bluez python3-dbus python3-gi
```

Bluetooth를 켭니다.

```bash
sudo rfkill unblock bluetooth
sudo systemctl enable --now bluetooth
```

서버를 실행합니다.

```bash
sudo python3 /home/bugless/jetson_blue/jetson_ble_hello.py
```

다른 문구를 보내려면:

```bash
sudo python3 /home/bugless/jetson_blue/jetson_ble_hello.py --message "안녕" --name JHello
```

구독 중에 3초마다 반복해서 보내려면:

```bash
sudo python3 /home/bugless/jetson_blue/jetson_ble_hello.py --repeat 3
```

기본 UUID는 다음과 같습니다.

```text
Service UUID:        12345678-1234-5678-1234-56789abcdef0
Characteristic UUID: 12345678-1234-5678-1234-56789abcdef1
```

## Android로 실험하는 방법

1. Android 휴대폰에 `nRF Connect for Mobile`을 설치합니다.
2. Jetson에서 `sudo python3 /home/bugless/jetson_blue/jetson_ble_hello.py`를 실행합니다.
3. nRF Connect에서 Scan을 누르고 `JHello` 장치를 찾습니다.
4. `CONNECT`를 누릅니다.
5. `12345678-1234-5678-1234-56789abcdef0` service를 엽니다.
6. `12345678-1234-5678-1234-56789abcdef1` characteristic에서 Read 버튼을 누르거나 Notify/Subscribe 버튼을 누릅니다.
7. 앱 표시 형식을 UTF-8/Text로 바꾸면 `안녕`이 보입니다. Hex로 보이면 `EC 95 88 EB 85 95`가 `안녕`의 UTF-8 바이트입니다.

## iPhone으로 실험하는 방법

1. iPhone에 `nRF Connect` 또는 `LightBlue`를 설치합니다.
2. Jetson에서 같은 스크립트를 실행합니다.
3. 앱에서 `JHello`를 스캔해서 연결합니다.
4. 같은 characteristic을 Read 또는 Notify 구독합니다.

## Flutter 앱으로 이어가기

Flutter 앱에서는 BLE central/client 역할을 맡기면 됩니다. 추천 패키지는 다음 중 하나입니다.

- `flutter_blue_plus`: 예제가 많고 Android/iOS BLE scan, connect, read, notify 구독 흐름이 단순합니다.
- `flutter_reactive_ble`: 연결 상태 스트림과 permission 처리가 명확해서 앱 구조를 크게 만들 때 좋습니다.

기본 흐름은 이렇습니다.

1. 앱에서 BLE scan을 시작합니다.
2. 광고 이름 `JHello` 또는 Service UUID `12345678-1234-5678-1234-56789abcdef0`로 Jetson을 찾습니다.
3. Jetson에 connect 합니다.
4. Characteristic UUID `12345678-1234-5678-1234-56789abcdef1`을 찾습니다.
5. `read`로 현재 메시지를 읽거나 `notify`를 구독합니다.
6. 받은 바이트를 UTF-8로 decode하면 `안녕`이 됩니다.

알림 기능을 만들려면 Jetson이 상황 발생 시 characteristic value를 바꾸고 notify를 보내면 됩니다. Flutter 앱은 notify 스트림을 듣다가 메시지를 받으면 화면 알림, 진동, 로컬 notification으로 이어 붙이면 됩니다.

## Flutter 개발 중 검증 방법

개발 중에는 두 가지를 분리해서 테스트하는 것이 좋습니다.

1. Flutter 앱 로직 검증

Android Emulator에서 BLE를 실제로 Jetson에 연결하지 말고, BLE 수신부를 mock/simulator로 바꿔서 테스트합니다. 예를 들어 앱 안에 `BleMessageSource` 같은 인터페이스를 만들고 실제 빌드에서는 BLE notify stream을 연결하고, 개발 빌드에서는 2초마다 `안녕`, `위험 감지`, `소리 감지` 같은 문자열을 emit하게 만듭니다.

이 방식으로 확인할 수 있는 것:

- 메시지를 받으면 화면 상태가 바뀌는지
- 알림 리스트에 기록되는지
- local notification 호출이 되는지
- 연결/끊김/재연결 UI가 의도대로 동작하는지

2. 실제 BLE 무선 검증

Jetson의 BLE advertising, scan, connect, read, notify가 실제로 되는지는 Android Emulator보다 실제 Android 폰에서 확인하는 것이 맞습니다. 개발 PC에 Bluetooth가 내장되어 있어도 일반 Android Emulator가 그 어댑터를 그대로 사용해서 주변 BLE 장치에 붙는 구조로 보기 어렵습니다.

중간 점검용으로는 개발 PC에서 Python `bleak` 같은 BLE central 라이브러리로 Jetson을 스캔하고 notify를 받아볼 수 있습니다. 이건 Flutter 앱 테스트는 아니지만 Jetson 쪽 BLE 서버가 제대로 동작하는지 빠르게 확인하는 용도입니다.

권장 개발 흐름:

1. Emulator에서는 mock BLE source로 Flutter UI와 알림 로직을 개발합니다.
2. Jetson 쪽은 nRF Connect 또는 PC BLE central 스크립트로 따로 확인합니다.
3. 실제 폰에서 마지막으로 Flutter 앱과 Jetson을 end-to-end로 붙입니다.

## 자주 나는 문제

`Operation not permitted`가 나오면 `sudo`로 실행합니다.

`No Bluetooth adapter with BLE GATT/advertising support was found`가 나오면 Bluetooth 서비스가 꺼져 있거나, 어댑터가 BLE peripheral advertising을 지원하지 않는 상태입니다.

`InvalidLength`가 나오면 BLE 광고 패킷이 너무 긴 것입니다. `--name JH`처럼 더 짧은 이름으로 실행합니다.
