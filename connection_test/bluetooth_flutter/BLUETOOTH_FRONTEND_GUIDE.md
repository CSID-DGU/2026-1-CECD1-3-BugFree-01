# bluetooth_flutter 구현 설명서

이 문서는 `bluetooth_flutter` 폴더의 Android BLE 클라이언트 구현을 프론트엔드 담당자나 AI 코드 생성 도구가 이해할 수 있도록 정리한 설명서입니다.

중요: 폴더 이름은 `bluetooth_flutter`지만, 현재 프로젝트는 Flutter가 아니라 **네이티브 Android Kotlin + Jetpack Compose 앱**입니다.

## 목표

Jetson은 BLE peripheral/server 역할을 하고, Android 앱은 BLE central/client 역할을 합니다.

전체 흐름은 다음과 같습니다.

1. Jetson에서 BLE 서버 실행
2. Android 앱에서 BLE scan
3. 앱이 Jetson 장치 `JHello`만 목록에 표시
4. 사용자가 `Connect` 버튼 클릭
5. 앱이 Jetson GATT service/characteristic 탐색
6. characteristic 값을 한 번 `read`
7. `notify`를 구독
8. Jetson이 주기적으로 보내는 메시지를 화면에 표시

## BLE 설정값

Jetson과 Android 앱은 아래 값을 기준으로 통신합니다.

```text
Device name: JHello
Service UUID: 12345678-1234-5678-1234-56789abcdef0
Characteristic UUID: 12345678-1234-5678-1234-56789abcdef1
Characteristic flags: read, notify
```

이 값들은 Android 코드의 `MainActivity.kt` 상단에 정의되어 있습니다.

```kotlin
private val JetsonServiceUuid: UUID = UUID.fromString("12345678-1234-5678-1234-56789abcdef0")
private val JetsonCharacteristicUuid: UUID =
    UUID.fromString("12345678-1234-5678-1234-56789abcdef1")
```

## 주요 파일

### `app/build.gradle.kts`

Android 앱 빌드 설정 파일입니다.

현재 중요한 설정:

```kotlin
minSdk = 26
targetSdk = 36
```

`minSdk = 26`으로 낮춘 이유는 Android 9 테스트폰에서 앱을 설치하기 위해서입니다. Android 9는 API 28이므로 설치 가능합니다.

### `app/src/main/AndroidManifest.xml`

Bluetooth 권한과 BLE 하드웨어 사용 선언이 들어 있습니다.

주요 권한:

```xml
<uses-feature
    android:name="android.hardware.bluetooth_le"
    android:required="true" />

<uses-permission
    android:name="android.permission.BLUETOOTH"
    android:maxSdkVersion="30" />
<uses-permission
    android:name="android.permission.BLUETOOTH_ADMIN"
    android:maxSdkVersion="30" />
<uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
<uses-permission
    android:name="android.permission.ACCESS_FINE_LOCATION"
    android:maxSdkVersion="30" />
```

Android 9에서는 BLE scan을 하려면 위치 권한과 위치 기능이 필요합니다.

실기기 테스트 시 확인할 것:

- Bluetooth 켜기
- 위치/GPS 켜기
- 앱의 위치 권한 허용

### `app/src/main/java/com/example/bluetoothtest/MainActivity.kt`

BLE 로직과 UI가 모두 들어 있는 핵심 파일입니다.

현재는 작은 테스트 앱이라 파일을 분리하지 않았습니다. 나중에 구조를 정리한다면 다음처럼 나눌 수 있습니다.

```text
MainActivity.kt
BleClient.kt
BleDeviceItem.kt
BleScreen.kt
```

하지만 현재 상태에서는 `MainActivity.kt` 하나만 보면 전체 흐름을 이해할 수 있습니다.

## 코드 구조

### `MainActivity`

Android 앱의 entry point입니다.

역할:

- `JetsonBleClient` 생성
- Compose UI 실행
- 앱 종료 시 BLE 연결 정리

```kotlin
class MainActivity : ComponentActivity() {
    private lateinit var bleClient: JetsonBleClient

    override fun onCreate(savedInstanceState: Bundle?) {
        bleClient = JetsonBleClient(this)
        setContent {
            BluetoothTestTheme {
                BleApp(bleClient)
            }
        }
    }

    override fun onDestroy() {
        bleClient.close()
        super.onDestroy()
    }
}
```

### `BleDeviceItem`

스캔된 BLE 장치를 UI에 표시하기 위한 단순 데이터 모델입니다.

```kotlin
data class BleDeviceItem(
    val name: String,
    val address: String,
    val rssi: Int,
)
```

각 필드 의미:

- `name`: BLE 광고 이름. Jetson은 `JHello`
- `address`: Android에서 보이는 BLE MAC 주소
- `rssi`: 신호 세기. 숫자가 0에 가까울수록 강함

### `JetsonBleClient`

실제 BLE 통신을 담당하는 클래스입니다.

담당 기능:

- 권한 확인
- BLE scan 시작/중지
- `JHello` 장치만 필터링
- BLE connect/disconnect
- GATT service discovery
- characteristic read
- notify subscribe
- 수신 메시지 저장

UI에서 사용하는 상태값:

```kotlin
var status by mutableStateOf("Idle")
var latestMessage by mutableStateOf("")
val devices = mutableStateListOf<BleDeviceItem>()
val messages = mutableStateListOf<String>()
```

각 상태값 의미:

- `status`: 현재 연결/스캔 상태 문자열
- `latestMessage`: 가장 최근에 받은 메시지
- `devices`: 화면에 표시할 Jetson 후보 목록
- `messages`: 받은 메시지 히스토리

Compose UI는 이 값들이 바뀌면 자동으로 다시 그려집니다.

## 권한 처리

Android 버전에 따라 필요한 권한이 다릅니다.

Android 12 이상:

```kotlin
Manifest.permission.BLUETOOTH_SCAN
Manifest.permission.BLUETOOTH_CONNECT
```

Android 11 이하, Android 9 포함:

```kotlin
Manifest.permission.ACCESS_FINE_LOCATION
```

코드에서는 `requiredPermissions()`가 기기 버전에 맞는 권한 목록을 반환합니다.

```kotlin
fun requiredPermissions(): Array<String> =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        arrayOf(
            Manifest.permission.BLUETOOTH_SCAN,
            Manifest.permission.BLUETOOTH_CONNECT,
        )
    } else {
        arrayOf(Manifest.permission.ACCESS_FINE_LOCATION)
    }
```

프론트 담당자가 UI를 바꿀 때도 권한 요청 흐름은 유지해야 합니다.

## 스캔 로직

스캔은 `startScan()`에서 시작합니다.

현재는 Jetson만 뜨도록 Android scan filter를 걸었습니다.

```kotlin
val filters = listOf(
    ScanFilter.Builder()
        .setDeviceName("JHello")
        .build(),
    ScanFilter.Builder()
        .setServiceUuid(ParcelUuid(JetsonServiceUuid))
        .build(),
)
activeScanner.startScan(filters, settings, scanCallback)
```

그리고 `onScanResult()`에서도 한 번 더 필터링합니다.

```kotlin
val name = result.scanRecord?.deviceName ?: device.name ?: "Unknown"
val serviceUuids = result.scanRecord?.serviceUuids.orEmpty()
if (name != "JHello" && !serviceUuids.contains(ParcelUuid(JetsonServiceUuid))) {
    return
}
```

이중 필터를 둔 이유:

- 주변 BLE 장치가 너무 많이 뜨는 것을 방지
- 일부 기기에서 service UUID 광고가 불안정해도 이름 `JHello`로 잡기 위해
- 이름이 안 잡히고 UUID만 잡히는 경우도 대응하기 위해

## 연결 로직

사용자가 UI에서 `Connect` 버튼을 누르면 `connect(address)`가 호출됩니다.

```kotlin
bluetoothGatt = device.connectGatt(
    context,
    false,
    gattCallback,
    BluetoothDevice.TRANSPORT_LE
)
```

연결 성공 후 흐름:

1. `onConnectionStateChange()`
2. `gatt.discoverServices()`
3. `onServicesDiscovered()`
4. Jetson service 찾기
5. Jetson characteristic 찾기
6. notify 구독
7. read 한 번 수행

## Notify 구독

Jetson의 characteristic은 `read`, `notify`를 지원합니다.

Android 쪽에서는 아래 순서로 notify를 켭니다.

```kotlin
gatt.setCharacteristicNotification(characteristic, true)
val descriptor = characteristic.getDescriptor(ClientCharacteristicConfigUuid)
```

그 다음 CCCD descriptor에 `ENABLE_NOTIFICATION_VALUE`를 씁니다.

Android 13 이상과 그 이전 버전의 API가 달라서 분기 처리되어 있습니다.

```kotlin
if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
    gatt.writeDescriptor(descriptor, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE)
} else {
    descriptor.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
    gatt.writeDescriptor(descriptor)
}
```

이 부분은 Android 9 테스트폰 호환을 위해 중요합니다.

## 메시지 수신

Jetson에서 notify가 오면 Android callback이 호출됩니다.

신형 callback:

```kotlin
override fun onCharacteristicChanged(
    gatt: BluetoothGatt,
    characteristic: BluetoothGattCharacteristic,
    value: ByteArray,
) {
    acceptMessage(value)
}
```

구형 Android 호환 callback:

```kotlin
override fun onCharacteristicChanged(
    gatt: BluetoothGatt,
    characteristic: BluetoothGattCharacteristic,
) {
    acceptMessage(characteristic.value)
}
```

받은 byte array는 UTF-8 문자열로 변환합니다.

```kotlin
private fun acceptMessage(value: ByteArray) {
    val decoded = value.toString(StandardCharsets.UTF_8)
    onMain {
        latestMessage = decoded
        messages.add(0, decoded)
    }
}
```

`onMain { ... }`을 쓰는 이유:

- BLE callback은 main thread가 아닌 곳에서 올 수 있음
- Compose state는 main thread에서 바꾸는 것이 안전함

## UI 구조

UI는 `BleApp()` composable 함수에 있습니다.

화면 구성:

1. 상단 앱바: `Jetson BLE Client`
2. Status 섹션
3. 버튼 Row
   - `Scan`
   - `Stop`
   - `Disconnect`
4. Nearby Jetson devices 목록
5. Latest message
6. History
7. Jetson 실행 명령 안내 문구

주요 UI 코드 위치:

```kotlin
@Composable
fun BleApp(bleClient: JetsonBleClient) {
    ...
}
```

프론트 담당자가 디자인을 바꾸려면 대부분 `BleApp()` 내부만 수정하면 됩니다.

BLE 로직은 가능하면 `JetsonBleClient` 안에 그대로 두는 것이 좋습니다.

## 현재 앱에서 표시되는 정보

### Status

예시:

```text
Idle
Scanning for JHello
Connected, discovering services
Subscribed to Jetson messages
Disconnected
```

### Device list

예시:

```text
JHello
AA:BB:CC:DD:EE:FF  RSSI -54
Connect
```

### Latest message

Jetson을 아래처럼 실행하면:

```bash
sudo python3 jetson_ble_hello.py --name JHello --message "test" --repeat 2
```

앱에는 다음처럼 표시됩니다.

```text
test 1
test 2
test 3
...
```

### History

새 메시지가 위에 쌓입니다.

```text
test 3
test 2
test 1
```

## Jetson 서버 실행

현재 테스트용 Jetson 서버는 notify 때마다 카운터를 붙입니다.

```bash
sudo python3 jetson_ble_hello.py --name JHello --message "test" --repeat 2
```

정상 로그:

```text
GATT application registered
Advertising as JHello
Service: 12345678-1234-5678-1234-56789abcdef0
Characteristic: 12345678-1234-5678-1234-56789abcdef1
Connect with nRF Connect, then read or subscribe to the characteristic.
```

앱이 notify를 구독하면:

```text
client subscribed; sending hello
sent notification: test 1
sent notification: test 2
sent notification: test 3
```

## Android 9 테스트 주의사항

Android 9에서 `JHello`가 안 뜨면 아래를 먼저 확인합니다.

1. 폰 Bluetooth 켜짐
2. 폰 위치/GPS 켜짐
3. 앱 위치 권한 허용됨
4. Jetson 서버가 `Advertising as JHello` 상태
5. nRF Connect에서는 `JHello`가 보이는지 확인

Android 9는 BLE scan이 위치 권한/위치 기능에 묶여 있습니다. 위치 기능이 꺼져 있으면 앱 권한을 허용해도 scan 결과가 안 나올 수 있습니다.

## 설치 및 실행

PowerShell에서:

```powershell
cd C:\Users\Gamzadole\OneDrive\Desktop\EdgeAudioRecognition\connection_test\bluetooth_flutter

$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
$env:Path="$env:JAVA_HOME\bin;$env:LOCALAPPDATA\Android\Sdk\platform-tools;$env:Path"

adb devices
.\gradlew.bat installDebug
```

에뮬레이터와 실기기가 둘 다 연결되어 있으면 특정 기기에 직접 설치합니다.

```powershell
adb devices
.\gradlew.bat assembleDebug
adb -s <PHONE_SERIAL> install -r .\app\build\outputs\apk\debug\app-debug.apk
```

## 프론트엔드 담당자가 주로 수정할 곳

### 화면 문구 변경

`BleApp()` 안의 `Text(...)`를 수정합니다.

예:

```kotlin
TopAppBar(title = { Text("Jetson BLE Client") })
```

### 버튼 스타일 변경

현재는 Material3 `Button`을 사용합니다.

```kotlin
Button(onClick = { bleClient.startScan() }) {
    Text("Scan")
}
```

색상, 크기, 배치, 아이콘을 바꾸고 싶으면 이 부분을 수정합니다.

### 메시지 리스트 디자인 변경

`History` 섹션의 `LazyColumn`을 수정합니다.

```kotlin
LazyColumn(modifier = Modifier.fillMaxWidth()) {
    items(bleClient.messages) { message ->
        Text(message, modifier = Modifier.padding(vertical = 6.dp))
        HorizontalDivider()
    }
}
```

### 연결 상태에 따른 UI 분기

현재는 `status` 문자열만 표시합니다. 더 좋은 구조로 바꾸려면 enum 상태를 만들 수 있습니다.

예:

```kotlin
enum class ConnectionState {
    Idle,
    Scanning,
    Connecting,
    Connected,
    Subscribed,
    Disconnected,
    Error,
}
```

지금은 빠른 테스트용이라 문자열 기반으로 충분합니다.

## AI에게 줄 수 있는 프롬프트 예시

### UI만 예쁘게 바꾸기

```text
이 프로젝트는 Flutter가 아니라 Kotlin Jetpack Compose Android 앱이다.
파일은 connection_test/bluetooth_flutter/app/src/main/java/com/example/bluetoothtest/MainActivity.kt 하나에 BLE 로직과 UI가 같이 있다.
JetsonBleClient 클래스의 BLE 로직은 건드리지 말고, BleApp composable 내부 UI만 수정해라.
목표는 BLE 테스트 앱처럼 보이게 하는 것이다.
Status, Scan/Stop/Disconnect 버튼, Nearby Jetson devices 목록, Latest message, History 섹션은 유지해라.
Android 9 작은 화면에서도 버튼과 텍스트가 겹치지 않게 해라.
```

### 메시지 카드 형태로 바꾸기

```text
Kotlin Jetpack Compose 앱의 BleApp UI를 수정해라.
messages 리스트를 단순 Text 목록이 아니라 카드 목록으로 보여줘라.
각 카드에는 메시지 내용과 수신 순번을 표시해라.
JetsonBleClient의 BLE scan/connect/notify 로직은 변경하지 마라.
```

### 연결 상태 배지 추가

```text
MainActivity.kt의 BleApp composable에 연결 상태 배지를 추가해라.
bleClient.status 문자열에 따라 색상을 다르게 보여줘라.
Scanning은 파란색, Connected/Subscribed는 초록색, Error/failed는 빨간색, Idle/Disconnected는 회색으로 처리해라.
BLE 로직은 수정하지 마라.
```

### 화면 분리 리팩터링

```text
현재 MainActivity.kt에 BLE 로직과 UI가 모두 들어 있다.
동작을 바꾸지 말고 파일만 분리해라.
JetsonBleClient는 BleClient.kt로 옮기고, BleDeviceItem은 BleDeviceItem.kt로 옮기고, BleApp composable은 BleScreen.kt로 옮겨라.
패키지는 com.example.bluetoothtest를 유지해라.
빌드가 통과해야 한다.
```

### Mock 모드 추가

```text
Android Emulator에서는 실제 Jetson BLE scan이 안 되므로 Mock 모드를 추가해라.
BleApp에 Mock Connect 버튼을 만들고, 누르면 2초마다 test 1, test 2, test 3 메시지가 latestMessage/messages에 들어오게 해라.
실제 BLE scan/connect 기능은 그대로 유지해라.
```

## 절대 건드리면 안 되는 것

프론트/UI 작업만 할 때는 아래를 함부로 바꾸지 않는 것이 좋습니다.

- `JetsonServiceUuid`
- `JetsonCharacteristicUuid`
- `ClientCharacteristicConfigUuid`
- `requiredPermissions()`
- `startScan()`
- `connect()`
- `subscribeToNotifications()`
- `onCharacteristicChanged()`
- `acceptMessage()`

이 부분을 잘못 바꾸면 UI는 보여도 BLE 통신이 깨질 수 있습니다.

## 현재 검증된 상태

검증된 것:

- Jetson이 nRF Connect에서 `JHello`로 보임
- nRF Connect에서 characteristic read 가능
- nRF Connect에서 notify 구독 시 `test 1`, `test 2`, `test 3` 수신 가능
- Android 앱이 debug build 성공
- Android 9 설치 가능하도록 `minSdk = 26` 적용
- Android 앱 스캔 목록에서 Jetson 후보만 보이도록 필터 적용

에뮬레이터 제한:

- 일반 Android Emulator는 실제 Jetson BLE를 직접 scan/connect 하기 어렵습니다.
- 실제 BLE 테스트는 물리 Android 폰으로 해야 합니다.
- 에뮬레이터에서는 Mock 모드를 넣어 UI만 테스트하는 방식이 적합합니다.

