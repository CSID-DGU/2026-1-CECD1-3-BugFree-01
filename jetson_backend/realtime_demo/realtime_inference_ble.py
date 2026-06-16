#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import signal
import sys
import threading
import warnings
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from gi.repository import GLib

# Import the existing inference helpers instead of changing the core inference
# file. This keeps model loading, preprocessing, label mapping, and prediction
# behavior aligned with realtime_inference.py.
from jetson_ef.realtime_demo.realtime_inference import (  # noqa: E402
    ANSI_GREEN,
    ANSI_RED,
    AUDIOSET_CLASS_COUNT,
    CHUNK_SAMPLES,
    CHUNK_SECONDS,
    DB_EPSILON,
    DEFAULT_ENHANCE_SHARPNESS,
    DEFAULT_ENHANCE_THRESHOLD_DB,
    DEFAULT_MAIN_GAIN_DB,
    DEFAULT_MIN_DB,
    DEFAULT_MIN_SCORE,
    DEFAULT_NOISE_REDUCTION_DB,
    FMAX,
    HOP_SIZE,
    LABEL_MAPPING,
    MIC_CHANNEL_INDEX,
    MIC_NAME_KEYWORDS,
    MODEL_INPUT_SAMPLES,
    MODEL_INPUT_SECONDS,
    MODEL_NAME,
    MODEL_SAMPLE_RATE,
    N_FFT,
    N_MELS,
    REQUIRED_INPUT_CHANNELS,
    SAMPLE_RATE,
    WINDOW_SIZE,
    build_custom_label_indices,
    colorize,
    db_gate_threshold,
    enhance_chunk,
    find_respeaker_device,
    format_scores,
    input_devices,
    load_efficientat,
    predict_chunk,
    print_input_devices,
    rms_dbfs,
)


BLUEZ_SERVICE_NAME = "org.bluez"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
ADAPTER_IFACE = "org.bluez.Adapter1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"

APP_PATH = "/com/bugless/bleinference"
ADVERTISEMENT_PATH = "/com/bugless/bleinference/advertisement0"

# Match the UUIDs used by EdgeAudioRecognition/connection_test/jetson.
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
INFERENCE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


def byte_array(data: bytes) -> dbus.Array:
    return dbus.Array([dbus.Byte(byte) for byte in data], signature="y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime EfficientAT inference from ReSpeaker Array V3 over BLE."
    )
    parser.add_argument(
        "--efficientat-dir",
        default=str(Path(__file__).resolve().parent / "EfficientAT"),
        help="Path to cloned fschmid56/EfficientAT repository.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="Optional sounddevice input device index. Defaults to automatic ReSpeaker search.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=MIC_CHANNEL_INDEX,
        help=(
            "Input channel to use. ReSpeaker 4 Mic Array 6-channel firmware is usually "
            "ch0=processed audio, ch1-4=raw microphones, ch5=playback."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input devices and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print waveform, mel, logits, and top AudioSet sigmoid scores for each chunk.",
    )
    parser.add_argument(
        "--min-db",
        type=float,
        default=DEFAULT_MIN_DB,
        help=(
            "Mark chunks quieter than this level as low signal. Positive values are treated as "
            "dB below full scale, so 30 means -30 dBFS. Use 0 or a negative value "
            "to pass an explicit dBFS threshold."
        ),
    )
    parser.add_argument(
        "--enhance-threshold-db",
        type=float,
        default=DEFAULT_ENHANCE_THRESHOLD_DB,
        help=(
            "Sample-level enhancement threshold. Positive values are treated as "
            "dB below full scale, so 35 means -35 dBFS."
        ),
    )
    parser.add_argument(
        "--noise-reduction-db",
        type=float,
        default=DEFAULT_NOISE_REDUCTION_DB,
        help="Reduce quieter waveform parts by this many dB before inference.",
    )
    parser.add_argument(
        "--main-gain-db",
        type=float,
        default=DEFAULT_MAIN_GAIN_DB,
        help="Boost louder waveform parts by this many dB before inference.",
    )
    parser.add_argument("--gain-db", type=float, dest="main_gain_db", help=argparse.SUPPRESS)
    parser.add_argument(
        "--enhance-sharpness",
        type=float,
        default=DEFAULT_ENHANCE_SHARPNESS,
        help="Higher values separate quiet noise and loud events more aggressively.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Mark predictions whose best custom sigmoid score is below this value as low confidence.",
    )
    parser.add_argument(
        "--ble-name",
        default="JHello",
        help="BLE advertising name. Keep JHello to match the Flutter reference scanner.",
    )
    parser.add_argument(
        "--ble-chunk-bytes",
        type=int,
        default=180,
        help=(
            "Maximum bytes per BLE notification frame. Flutter requests MTU 247, "
            "so 180 leaves room for the chunk header."
        ),
    )
    return parser.parse_args()


class Application(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus):
        self.path = APP_PATH
        self.services = []
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_service(self, service: "Service") -> None:
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self) -> dict:
        managed_objects = {}
        for service in self.services:
            managed_objects[service.get_path()] = service.get_properties()
            for characteristic in service.characteristics:
                managed_objects[characteristic.get_path()] = characteristic.get_properties()
        return managed_objects


class Service(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, index: int, uuid: str, primary: bool = True):
        self.path = f"{APP_PATH}/service{index}"
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic: "InferenceCharacteristic") -> None:
        self.characteristics.append(characteristic)

    def get_properties(self) -> dict:
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": dbus.Boolean(self.primary),
            }
        }

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict:
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_SERVICE_IFACE]


class InferenceCharacteristic(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, index: int, service: Service, chunk_bytes: int):
        self.path = f"{service.path}/char{index}"
        self.service = service
        self.uuid = INFERENCE_CHAR_UUID
        self.flags = ["read", "notify"]
        self.chunk_bytes = max(20, int(chunk_bytes))
        self.latest_payload = json.dumps(
            {"status": "starting", "message": "waiting_for_inference"},
            separators=(",", ":"),
            ensure_ascii=True,
        )
        self.notifying = False
        self.sequence = 0
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def get_properties(self) -> dict:
        return {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": dbus.Array(self.flags, signature="s"),
            }
        }

    def _frames_for_payload(self, payload: str) -> List[bytes]:
        # BLE notify payloads are MTU-limited. The Flutter client reassembles
        # these ASCII frames into the original JSON message.
        payload_bytes = payload.encode("ascii")
        total = 1
        while True:
            parts: List[bytes] = []
            offset = 0
            index = 1
            while offset < len(payload_bytes) or (offset == 0 and not payload_bytes):
                prefix = f"#{self.sequence}:{index}/{total}:".encode("ascii")
                content_size = max(1, self.chunk_bytes - len(prefix))
                parts.append(payload_bytes[offset: offset + content_size])
                offset += content_size
                index += 1
            if len(parts) == total:
                return [
                    f"#{self.sequence}:{index}/{total}:".encode("ascii") + part
                    for index, part in enumerate(parts, start=1)
                ]
            total = len(parts)

    def notify_text(self, payload: str) -> None:
        self.latest_payload = payload
        GLib.idle_add(self._notify_latest)

    def _notify_latest(self) -> bool:
        if not self.notifying:
            return False

        self.sequence += 1
        for frame in self._frames_for_payload(self.latest_payload):
            self.PropertiesChanged(
                GATT_CHRC_IFACE,
                dbus.Dictionary({"Value": byte_array(frame)}, signature="sv"),
                dbus.Array([], signature="s"),
            )
        print(
            f"sent BLE inference notification seq={self.sequence} bytes={len(self.latest_payload)}",
            flush=True,
        )
        return False

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict:
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options: dict) -> dbus.Array:
        print("read request: latest inference payload", flush=True)
        return byte_array(self.latest_payload.encode("utf-8"))

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self) -> None:
        if self.notifying:
            return
        self.notifying = True
        print("client subscribed; sending latest inference payload", flush=True)
        self._notify_latest()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self) -> None:
        self.notifying = False
        print("client unsubscribed", flush=True)

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, interface: str, changed: dict, invalidated: list) -> None:
        pass


class Advertisement(dbus.service.Object):
    def __init__(self, bus: dbus.SystemBus, local_name: str, service_uuid: str):
        self.path = ADVERTISEMENT_PATH
        self.local_name = local_name
        self.service_uuid = service_uuid
        super().__init__(bus, self.path)

    def get_path(self) -> dbus.ObjectPath:
        return dbus.ObjectPath(self.path)

    def get_properties(self) -> dict:
        return {
            LE_ADVERTISEMENT_IFACE: {
                "Type": "peripheral",
                "LocalName": dbus.String(self.local_name),
                "ServiceUUIDs": dbus.Array([self.service_uuid], signature="s"),
            }
        }

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface: str) -> dict:
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE)
    def Release(self) -> None:
        print("advertisement released", flush=True)


def find_adapter(bus: dbus.SystemBus) -> str | None:
    object_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
    objects = object_manager.GetManagedObjects()
    for path, interfaces in objects.items():
        if GATT_MANAGER_IFACE in interfaces and LE_ADVERTISING_MANAGER_IFACE in interfaces:
            return path
    return None


def set_adapter_powered(bus: dbus.SystemBus, adapter_path: str) -> None:
    properties = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), DBUS_PROP_IFACE)
    properties.Set(ADAPTER_IFACE, "Powered", dbus.Boolean(True))


class BleInferenceServer:
    def __init__(self, local_name: str, chunk_bytes: int):
        self.local_name = local_name
        self.chunk_bytes = chunk_bytes
        self.loop: GLib.MainLoop | None = None
        self.thread: threading.Thread | None = None
        self.service_manager = None
        self.advertising_manager = None
        self.application: Application | None = None
        self.advertisement: Advertisement | None = None
        self.characteristic: InferenceCharacteristic | None = None
        self.ready_event = threading.Event()
        self.startup_error: str | None = None
        self.app_ready = False
        self.advertisement_ready = False

    def start(self) -> None:
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SystemBus()
        adapter_path = find_adapter(bus)
        if not adapter_path:
            raise RuntimeError("No Bluetooth adapter with BLE GATT/advertising support was found.")

        set_adapter_powered(bus, adapter_path)
        self.service_manager = dbus.Interface(
            bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
            GATT_MANAGER_IFACE,
        )
        self.advertising_manager = dbus.Interface(
            bus.get_object(BLUEZ_SERVICE_NAME, adapter_path),
            LE_ADVERTISING_MANAGER_IFACE,
        )

        self.application = Application(bus)
        inference_service = Service(bus, 0, SERVICE_UUID, primary=True)
        self.characteristic = InferenceCharacteristic(
            bus,
            0,
            inference_service,
            self.chunk_bytes,
        )
        inference_service.add_characteristic(self.characteristic)
        self.application.add_service(inference_service)
        self.advertisement = Advertisement(bus, self.local_name, SERVICE_UUID)
        self.loop = GLib.MainLoop()

        self.service_manager.RegisterApplication(
            self.application.get_path(),
            {},
            reply_handler=self._app_registered,
            error_handler=self._registration_failed("Failed to register GATT application"),
        )
        self.advertising_manager.RegisterAdvertisement(
            self.advertisement.get_path(),
            {},
            reply_handler=self._advertisement_registered,
            error_handler=self._registration_failed("Failed to register advertisement"),
        )

        self.thread = threading.Thread(target=self._run_loop, name="ble-gatt-loop", daemon=True)
        self.thread.start()
        if not self.ready_event.wait(timeout=10):
            self.stop()
            raise RuntimeError("Timed out while registering BLE GATT application.")
        if self.startup_error:
            self.stop()
            raise RuntimeError(self.startup_error)

    def _app_registered(self) -> None:
        self.app_ready = True
        print("GATT application registered", flush=True)
        self._mark_ready()

    def _advertisement_registered(self) -> None:
        self.advertisement_ready = True
        print(f"Advertising as {self.local_name}", flush=True)
        print(f"Service: {SERVICE_UUID}", flush=True)
        print(f"Characteristic: {INFERENCE_CHAR_UUID}", flush=True)
        print("Connect from the Flutter app and subscribe to notifications.", flush=True)
        self._mark_ready()

    def _registration_failed(self, prefix: str):
        def handler(error: dbus.exceptions.DBusException) -> None:
            self.startup_error = f"{prefix}: {error}"
            print(self.startup_error, file=sys.stderr, flush=True)
            if "InvalidLength" in str(error):
                print("Try a shorter --ble-name value, for example --ble-name JH.", file=sys.stderr)
            self.ready_event.set()
            if self.loop is not None:
                self.loop.quit()

        return handler

    def _mark_ready(self) -> None:
        if self.app_ready and self.advertisement_ready:
            self.ready_event.set()

    def _run_loop(self) -> None:
        if self.loop is None:
            return
        try:
            self.loop.run()
        finally:
            if self.advertising_manager and self.advertisement:
                try:
                    self.advertising_manager.UnregisterAdvertisement(self.advertisement.get_path())
                except dbus.exceptions.DBusException:
                    pass
            if self.service_manager and self.application:
                try:
                    self.service_manager.UnregisterApplication(self.application.get_path())
                except dbus.exceptions.DBusException:
                    pass
            print("Stopped BLE inference server", flush=True)

    def publish(self, data: dict) -> None:
        if self.characteristic is None:
            return
        # Compact ASCII JSON makes the BLE chunk protocol deterministic and easy
        # to reassemble in Flutter without depending on locale-specific text.
        payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True)
        self.characteristic.notify_text(payload)

    def stop(self) -> None:
        if self.loop is not None:
            GLib.idle_add(self.loop.quit)
        if self.thread is not None:
            self.thread.join(timeout=3)


def build_ble_result(
    timestamp: str,
    best_label: str,
    best_probability: float,
    scores: Dict[str, float],
    status_key: str,
    status_text: str,
    chunk_dbfs: float,
    enhanced_dbfs: float,
    quiet_gain: float,
    loud_gain: float,
    clipped: bool,
    raw_line: str,
) -> dict:
    return {
        "source": "realtime_inference_ble",
        "time": timestamp,
        "label": best_label,
        "score": round(float(best_probability), 6),
        "status": status_key,
        "status_text": status_text,
        "level_dbfs": round(float(chunk_dbfs), 2),
        "enhanced_dbfs": round(float(enhanced_dbfs), 2),
        "quiet_gain": round(float(quiet_gain), 4),
        "loud_gain": round(float(loud_gain), 4),
        "clipped": bool(clipped),
        "scores": {label: round(float(score), 6) for label, score in scores.items()},
        "raw": raw_line,
    }


def run_stream_ble(
    device_index: int,
    device_info: dict,
    stream_channels: int,
    channel_index: int,
    model: torch.nn.Module,
    mel: torch.nn.Module,
    resampler: torch.nn.Module | None,
    custom_indices: Dict[str, List[int]],
    audioset_labels: Sequence[str],
    device: torch.device,
    debug: bool,
    min_db: float,
    enhance_threshold_db: float,
    noise_reduction_db: float,
    main_gain_db: float,
    enhance_sharpness: float,
    min_score: float,
    ble_server: BleInferenceServer,
) -> None:
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    if channel_index < 0 or channel_index >= stream_channels:
        raise RuntimeError(
            f"?좏깮??梨꾨꼸 index媛 踰붿쐞瑜?踰쀬뼱?ъ뒿?덈떎: "
            f"channel={channel_index}, available=0..{stream_channels - 1}"
        )

    if stream_channels < REQUIRED_INPUT_CHANNELS:
        print(
            f"寃쎄퀬: ?좏깮???붾컮?댁뒪媛 {stream_channels}媛??낅젰 梨꾨꼸留?蹂닿퀬?⑸땲?? "
            "?ъ슜 媛?ν븳 梨꾨꼸 0?쇰줈 怨꾩냽 吏꾪뻾?⑸땲??",
            file=sys.stderr,
        )

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"?ㅻ뵒???낅젰 ?곹깭: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    print(
        f"?낅젰 ?붾컮?댁뒪: [{device_index}] {device_info.get('name')} | "
        f"channels={stream_channels}, mic_sr={SAMPLE_RATE}, "
        f"model_sr={MODEL_SAMPLE_RATE}, chunk={CHUNK_SECONDS}s, "
        f"model_input={MODEL_INPUT_SECONDS}s, channel={channel_index}, "
        f"min_dbfs={db_gate_threshold(min_db):+.1f}, "
        f"enhance_threshold_dbfs={db_gate_threshold(enhance_threshold_db):+.1f}, "
        f"noise_reduction_db={noise_reduction_db:.1f}, main_gain_db={main_gain_db:+.1f}, "
        f"min_score={min_score:.1%}"
    )
    print("Ctrl+C濡?醫낅즺?⑸땲??")

    pending_blocks: List[np.ndarray] = []
    pending_samples = 0

    with sd.InputStream(
        device=device_index,
        samplerate=SAMPLE_RATE,
        channels=stream_channels,
        dtype="float32",
        callback=callback,
    ):
        while True:
            block = audio_queue.get()
            mono = block[:, channel_index].astype(np.float32, copy=True)
            pending_blocks.append(mono)
            pending_samples += mono.shape[0]

            if pending_samples < CHUNK_SAMPLES:
                continue

            joined = np.concatenate(pending_blocks)
            offset = 0
            while joined.shape[0] - offset >= CHUNK_SAMPLES:
                chunk = joined[offset: offset + CHUNK_SAMPLES]
                offset += CHUNK_SAMPLES
                timestamp = datetime.now().strftime("%H:%M:%S")
                chunk_dbfs = rms_dbfs(chunk)
                min_dbfs = db_gate_threshold(min_db)

                inference_chunk, clipped, quiet_gain, loud_gain = enhance_chunk(
                    chunk,
                    enhance_threshold_db,
                    noise_reduction_db,
                    main_gain_db,
                    enhance_sharpness,
                )
                enhanced_dbfs = rms_dbfs(inference_chunk)

                try:
                    best_label, best_probability, scores = predict_chunk(
                        inference_chunk,
                        model,
                        mel,
                        resampler,
                        custom_indices,
                        audioset_labels,
                        device,
                        debug=debug,
                    )
                except Exception as exc:  # Keep the stream alive on bad chunks.
                    print(
                        f"[{timestamp}] 異붾줎 ?ㅻ쪟: {exc} | ?대떦 泥?겕 skip",
                        file=sys.stderr,
                    )
                    continue

                status_reasons = []
                if chunk_dbfs < min_dbfs:
                    status_reasons.append(f"?뚮━?묒쓬 {chunk_dbfs:+.1f}<{min_dbfs:+.1f}dBFS")
                if best_probability < min_score:
                    status_reasons.append(f"?먯닔??쓬 {best_probability:.1%}<{min_score:.1%}")

                if status_reasons:
                    status = "??쓬(" + ", ".join(status_reasons) + ")"
                    status_key = "low_signal"
                    line_color = ANSI_RED
                else:
                    status = "媛먯?"
                    status_key = "detected"
                    line_color = ANSI_GREEN

                line = (
                    f"[{timestamp}] ?덉륫: {best_label} ({best_probability:.1%}) | "
                    f"status={status} | "
                    f"level={chunk_dbfs:+.1f} dBFS | "
                    f"enhanced={enhanced_dbfs:+.1f} dBFS | "
                    f"quiet_gain={quiet_gain:.2f}x loud_gain={loud_gain:.2f}x"
                    f"{' clipped' if clipped else ''} | ?꾩껜: {format_scores(scores)}"
                )
                print(colorize(line, line_color), flush=True)
                ble_server.publish(
                    build_ble_result(
                        timestamp=timestamp,
                        best_label=best_label,
                        best_probability=best_probability,
                        scores=scores,
                        status_key=status_key,
                        status_text=status,
                        chunk_dbfs=chunk_dbfs,
                        enhanced_dbfs=enhanced_dbfs,
                        quiet_gain=quiet_gain,
                        loud_gain=loud_gain,
                        clipped=clipped,
                        raw_line=line,
                    )
                )

            remainder = joined[offset:]
            pending_blocks = [remainder] if remainder.size else []
            pending_samples = remainder.shape[0]


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print_input_devices()
        return 0

    try:
        device_index, device_info, stream_channels = find_respeaker_device(args.device_index)
    except Exception as exc:
        print(f"?ㅻ쪟: {exc}", file=sys.stderr)
        print_input_devices()
        return 1

    ble_server = BleInferenceServer(args.ble_name, args.ble_chunk_bytes)
    try:
        ble_server.start()
    except Exception as exc:
        print(f"BLE 珥덇린???ㅻ쪟: {exc}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"異붾줎 ?붾컮?댁뒪: {device}")
    resampler = None
    if SAMPLE_RATE != MODEL_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=SAMPLE_RATE,
            new_freq=MODEL_SAMPLE_RATE,
        ).to(device).eval()

    try:
        model, mel, audioset_labels = load_efficientat(Path(args.efficientat_dir), device)
        custom_indices = build_custom_label_indices(audioset_labels)
    except Exception as exc:
        print(f"紐⑤뜽 珥덇린???ㅻ쪟: {exc}", file=sys.stderr)
        ble_server.stop()
        return 1

    stop_requested = False

    def stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        run_stream_ble(
            device_index=device_index,
            device_info=device_info,
            stream_channels=stream_channels,
            channel_index=args.channel_index,
            model=model,
            mel=mel,
            resampler=resampler,
            custom_indices=custom_indices,
            audioset_labels=audioset_labels,
            device=device,
            debug=args.debug,
            min_db=args.min_db,
            enhance_threshold_db=args.enhance_threshold_db,
            noise_reduction_db=args.noise_reduction_db,
            main_gain_db=args.main_gain_db,
            enhance_sharpness=args.enhance_sharpness,
            min_score=args.min_score,
            ble_server=ble_server,
        )
    except KeyboardInterrupt:
        print("\n醫낅즺?⑸땲??")
        return 0
    except Exception as exc:
        print(f"?ㅻ뵒???ㅽ듃由??ㅻ쪟: {exc}", file=sys.stderr)
        return 1
    finally:
        if stop_requested:
            print("Stopping BLE server...", flush=True)
        ble_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
