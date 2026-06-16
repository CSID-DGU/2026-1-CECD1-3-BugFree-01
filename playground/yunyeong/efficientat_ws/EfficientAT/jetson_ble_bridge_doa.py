#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano EfficientAT BLE GATT bridge with dB filtering and ReSpeaker DOA.

Flow:
  jetson_live.py stdout -> parse result -> add DOA direction -> BLE GATT notify JSON lines

BLE profile:
  Nordic UART Service compatible service
  Service: 6e400001-b5a3-f393-e0a9-e50e24dcca9e
  TX notify: 6e400003-b5a3-f393-e0a9-e50e24dcca9e  (Jetson -> Mac/phone)
  RX write:  6e400002-b5a3-f393-e0a9-e50e24dcca9e  (client -> Jetson, optional)

Payload:
  UTF-8 JSON object + '\n', chunked into small BLE notifications.
"""
from __future__ import print_function

import argparse
import io
import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

LINE_RE = re.compile(
    r"^\[(?P<time>[0-9:]+)\]\s+infer=(?P<infer>[0-9.]+)s(?P<middle>.*?)\|\s+(?P<items>.*)$"
)

DANGER_WORDS = [
    "horn", "siren", "alarm", "scream", "screaming", "gun", "gunshot",
    "explosion", "glass", "fire", "smoke", "crash", "shout",
    "경적", "사이렌", "경보", "비명", "총", "화재", "폭발", "유리",
]
CAUTION_WORDS = [
    "vehicle", "car", "truck", "dog", "cry", "baby", "knock", "door",
    "water", "engine", "cat", "bark",
    "차", "자동차", "트럭", "개", "고양이", "아기", "울음", "노크", "문", "물",
]

_doa_lock = threading.Lock()
_doa_started = False
_doa_angle = None
_doa_direction = None
_doa_status = "not_started"


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def to_float(text, default=None):
    try:
        return float(text)
    except Exception:
        return default


def risk_level(label):
    text = str(label).lower()
    for word in DANGER_WORDS:
        if word.lower() in text:
            return "danger"
    for word in CAUTION_WORDS:
        if word.lower() in text:
            return "caution"
    return "info"


def angle_to_cardinal(angle, north_offset=0.0):
    corrected = (float(angle) - float(north_offset)) % 360.0
    if corrected < 45 or corrected >= 315:
        return "북"
    if corrected < 135:
        return "동"
    if corrected < 225:
        return "남"
    return "서"


class InlineTuning(object):
    """Minimal DOA reader for ReSpeaker USB 4 Mic Array when tuning.py is absent."""
    TIMEOUT = 100000

    def __init__(self, dev, usb_util):
        self.dev = dev
        self.usb_util = usb_util

    @property
    def direction(self):
        # ReSpeaker tuning.py convention: DOAANGLE id=21.
        response = self.dev.ctrl_transfer(
            self.usb_util.CTRL_IN
            | self.usb_util.CTRL_TYPE_VENDOR
            | self.usb_util.CTRL_RECIPIENT_DEVICE,
            0,
            0xC0,
            21,
            8,
            self.TIMEOUT,
        )
        try:
            data = response.tobytes()
        except AttributeError:
            data = response.tostring()
        value, _exponent = struct.unpack(b"ii", data)
        return int(value)


def create_tuning(dev):
    try:
        from tuning import Tuning  # type: ignore
        return Tuning(dev)
    except Exception:
        import usb.util  # type: ignore
        return InlineTuning(dev, usb.util)


def set_doa_status(status):
    global _doa_status
    with _doa_lock:
        _doa_status = status


def doa_loop(north_offset):
    global _doa_angle, _doa_direction, _doa_status
    try:
        import usb.core  # type: ignore
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev is None:
            set_doa_status("usb_control_not_found")
            print("[DOA] ReSpeaker USB control device not found. Check lsusb and Docker USB mount.", file=sys.stderr)
            return
        tuning = create_tuning(dev)
        set_doa_status("enabled")
        print("[DOA] ReSpeaker DOA reader enabled.", file=sys.stderr)
        while True:
            try:
                angle = int(float(tuning.direction)) % 360
                direction = angle_to_cardinal(angle, north_offset)
                with _doa_lock:
                    _doa_angle = angle
                    _doa_direction = direction
                    _doa_status = "enabled"
            except Exception as e:
                set_doa_status("read_error")
                print("[DOA] read error:", repr(e), file=sys.stderr)
            time.sleep(0.2)
    except Exception as e:
        set_doa_status("disabled")
        print("[DOA] disabled:", repr(e), file=sys.stderr)


def ensure_doa_thread(enable=True, north_offset=0.0):
    global _doa_started
    if not enable:
        set_doa_status("disabled_by_option")
        return
    if _doa_started:
        return
    _doa_started = True
    t = threading.Thread(target=doa_loop, args=(north_offset,))
    t.daemon = True
    t.start()


def get_latest_doa(wait_sec=0.5):
    deadline = time.time() + max(0.0, float(wait_sec))
    while True:
        with _doa_lock:
            angle = _doa_angle
            direction = _doa_direction
            status = _doa_status
        if direction is not None or time.time() >= deadline:
            return angle, direction, status
        time.sleep(0.05)


def parse_items(items_text):
    # Drop a trailing DOA segment if an older wrapper already appended it.
    items_text = str(items_text).split(" | DOA ", 1)[0]
    items = []
    for part in items_text.split(" / "):
        part = part.strip()
        if not part:
            continue
        try:
            label, score_text = part.rsplit(" ", 1)
            score = float(score_text)
        except Exception:
            continue
        items.append({"label": label.strip(), "score": score})
    return items


def parse_line(line, min_db=None, add_doa=True, north_offset=0.0, doa_wait_sec=0.5,
               label_suffix=True, include_angle=False):
    stripped = line.strip()
    m = LINE_RE.search(stripped)
    if not m:
        return None

    middle = m.group("middle") or ""
    total_sec = None
    total_match = re.search(r"total=([0-9.]+)s", middle)
    if total_match:
        total_sec = to_float(total_match.group(1))

    display_db = None
    db_match = re.search(r"(?:^|\s)db=([-+]?[0-9.]+)dB", middle)
    if db_match:
        display_db = to_float(db_match.group(1))

    if min_db is not None and display_db is not None and display_db < float(min_db):
        return None

    items = parse_items(m.group("items"))
    if not items:
        return None

    top = dict(items[0])
    base_label = top["label"]
    direction = None
    angle = None
    doa_status = "disabled"

    if add_doa:
        ensure_doa_thread(enable=True, north_offset=north_offset)
        angle, direction, doa_status = get_latest_doa(wait_sec=doa_wait_sec)

    label = base_label
    out_items = [dict(x) for x in items]
    raw = stripped
    if direction is not None:
        if label_suffix:
            label = "%s [%s]" % (base_label, direction)
            out_items[0]["label"] = "%s [%s]" % (out_items[0]["label"], direction)
        raw = "%s | DOA %s" % (raw, direction)
    elif add_doa:
        raw = "%s | DOA unavailable:%s" % (raw, doa_status)

    result = {
        "type": "sound_result",
        "status": "ok",
        "source": "jetson_nano",
        "sent_at": now_iso(),
        "time": m.group("time"),
        "base_label": base_label,
        "label": label,
        "score": top["score"],
        "infer_sec": to_float(m.group("infer")),
        "total_sec": total_sec,
        "db": display_db,
        "display_db": display_db,
        "min_db": min_db,
        "level": risk_level(base_label),
        "direction": direction,
        "doa_status": doa_status,
        "items": out_items,
        "raw": raw,
    }
    if include_angle:
        result["angle"] = angle
    return result


# ----------------------------- BlueZ BLE GATT -----------------------------
# The run script installs these packages in the Docker container:
#   python3-dbus python3-gi gir1.2-glib-2.0
try:
    import dbus  # type: ignore
    import dbus.exceptions  # type: ignore
    import dbus.mainloop.glib  # type: ignore
    import dbus.service  # type: ignore
    from gi.repository import GLib  # type: ignore
except Exception as e:  # pragma: no cover - shown on Jetson if packages are missing
    print("[BLE] missing Python D-Bus/GI packages. Install python3-dbus python3-gi gir1.2-glib-2.0", file=sys.stderr)
    raise

NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # client -> Jetson
NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Jetson -> client notifications

BLUEZ_SERVICE_NAME = "org.bluez"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"


class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.freedesktop.DBus.Error.InvalidArgs"


class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.NotSupported"


class NotPermittedException(dbus.exceptions.DBusException):
    _dbus_error_name = "org.bluez.Error.NotPermitted"


def bytes_to_dbus_array(data):
    return dbus.Array([dbus.Byte(x) for x in bytearray(data)], signature="y")


def find_adapter(bus, adapter_name=None):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()
    for path, ifaces in objects.items():
        if adapter_name and not str(path).endswith("/" + adapter_name):
            continue
        if GATT_MANAGER_IFACE in ifaces and LE_ADVERTISING_MANAGER_IFACE in ifaces:
            return path
    return None


class Application(dbus.service.Object):
    PATH_BASE = "/org/edgeaudio/ble"

    def __init__(self, bus):
        self.path = self.PATH_BASE
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):  # noqa: N802
        response = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            for chrc in service.get_characteristics():
                response[chrc.get_path()] = chrc.get_properties()
        return response


class Service(dbus.service.Object):
    PATH_BASE = "/org/edgeaudio/ble/service"

    def __init__(self, bus, index, uuid, primary):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": dbus.Array([chrc.get_path() for chrc in self.characteristics], signature="o"),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    def get_characteristics(self):
        return self.characteristics

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):  # noqa: N802
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_SERVICE_IFACE]


class Characteristic(dbus.service.Object):
    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.path + "/char" + str(index)
        self.bus = bus
        self.uuid = uuid
        self.flags = flags
        self.service = service
        self.value = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": dbus.Array(self.flags, signature="s"),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):  # noqa: N802
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):  # noqa: N802
        return bytes_to_dbus_array(bytearray(self.value))

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):  # noqa: N802
        self.value = list(value)

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):  # noqa: N802
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):  # noqa: N802
        raise NotSupportedException()

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, interface, changed, invalidated):  # noqa: N802
        pass


class TxCharacteristic(Characteristic):
    def __init__(self, bus, index, service, chunk_size=20):
        Characteristic.__init__(self, bus, index, NUS_TX_UUID, ["read", "notify"], service)
        self.notifying = False
        self.chunk_size = int(chunk_size)

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):  # noqa: N802
        if self.notifying:
            return
        self.notifying = True
        print("[BLE] client subscribed to notifications")
        hello = {
            "type": "status",
            "status": "connected",
            "source": "jetson_nano",
            "sent_at": now_iso(),
            "message": "BLE notification connected. Sound results are sent as newline-delimited JSON.",
        }
        self.notify_text(json.dumps(hello, ensure_ascii=False, separators=(",", ":")) + "\n")

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):  # noqa: N802
        self.notifying = False
        print("[BLE] client unsubscribed from notifications")

    def notify_bytes(self, data):
        if not self.notifying:
            return False
        self.value = list(bytearray(data))
        changed = {"Value": bytes_to_dbus_array(data)}
        self.PropertiesChanged(GATT_CHRC_IFACE, changed, [])
        return True

    def notify_text(self, text):
        data = text.encode("utf-8")
        sent = False
        # Conservative 20-byte chunks work on old BLE 4.x stacks and Chrome.
        for start in range(0, len(data), self.chunk_size):
            chunk = data[start:start + self.chunk_size]
            sent = self.notify_bytes(chunk) or sent
            time.sleep(0.01)
        return sent


class RxCharacteristic(Characteristic):
    def __init__(self, bus, index, service):
        Characteristic.__init__(self, bus, index, NUS_RX_UUID, ["write", "write-without-response"], service)

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):  # noqa: N802
        try:
            text = bytes(bytearray(value)).decode("utf-8", "replace").strip()
        except Exception:
            text = repr(value)
        print("[BLE_RX]", text)


class UartService(Service):
    def __init__(self, bus, index, chunk_size=20):
        Service.__init__(self, bus, index, NUS_SERVICE_UUID, True)
        self.tx = TxCharacteristic(bus, 0, self, chunk_size=chunk_size)
        self.rx = RxCharacteristic(bus, 1, self)
        self.add_characteristic(self.tx)
        self.add_characteristic(self.rx)


class Advertisement(dbus.service.Object):
    PATH_BASE = "/org/edgeaudio/ble/advertisement"

    def __init__(self, bus, index, local_name):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = "peripheral"
        self.service_uuids = [NUS_SERVICE_UUID]
        self.local_name = local_name
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            LE_ADVERTISEMENT_IFACE: {
                "Type": self.ad_type,
                "ServiceUUIDs": dbus.Array(self.service_uuids, signature="s"),
                "LocalName": dbus.String(self.local_name),
                "Includes": dbus.Array(["tx-power"], signature="s"),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):  # noqa: N802
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE)
    def Release(self):  # noqa: N802
        print("[BLE] advertisement released")


class BleNotifier(object):
    def __init__(self, adapter="hci0", name="EdgeAudio-Jetson", chunk_size=20):
        self.adapter = adapter
        self.name = name
        self.chunk_size = int(chunk_size)
        self.queue = queue.Queue()
        self.loop = None
        self.loop_thread = None
        self.tx = None
        self.bus = None
        self.service_manager = None
        self.ad_manager = None
        self.app = None
        self.adv = None
        self.app_ok = False
        self.adv_ok = False
        self.registered = threading.Event()
        self.register_error = []

    def start(self):
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SystemBus()
        adapter_path = find_adapter(self.bus, self.adapter)
        if adapter_path is None:
            raise RuntimeError(
                "Bluetooth adapter with GattManager1/LEAdvertisingManager1 not found. "
                "Run setup_ble_host_v6.sh and make sure bluetoothd uses --experimental."
            )

        self.service_manager = dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), GATT_MANAGER_IFACE)
        self.ad_manager = dbus.Interface(self.bus.get_object(BLUEZ_SERVICE_NAME, adapter_path), LE_ADVERTISING_MANAGER_IFACE)

        self.app = Application(self.bus)
        uart = UartService(self.bus, 0, chunk_size=self.chunk_size)
        self.tx = uart.tx
        self.app.add_service(uart)
        self.adv = Advertisement(self.bus, 0, self.name)
        self.loop = GLib.MainLoop()

        def check_ready():
            if self.app_ok and self.adv_ok:
                self.registered.set()

        def register_app_cb():
            self.app_ok = True
            print("[BLE] GATT application registered")
            check_ready()

        def register_app_error_cb(error):
            self.register_error.append("GATT register error: %s" % str(error))
            self.registered.set()

        def register_adv_cb():
            self.adv_ok = True
            print("[BLE] advertising started: name=%s service=%s" % (self.name, NUS_SERVICE_UUID))
            check_ready()

        def register_adv_error_cb(error):
            self.register_error.append("Advertisement register error: %s" % str(error))
            self.registered.set()

        self.service_manager.RegisterApplication(
            self.app.get_path(), {}, reply_handler=register_app_cb, error_handler=register_app_error_cb
        )
        self.ad_manager.RegisterAdvertisement(
            self.adv.get_path(), {}, reply_handler=register_adv_cb, error_handler=register_adv_error_cb
        )

        GLib.timeout_add(50, self._pump_queue)
        self.loop_thread = threading.Thread(target=self.loop.run)
        self.loop_thread.daemon = True
        self.loop_thread.start()

        if not self.registered.wait(8.0):
            print("[BLE] warning: registration callbacks did not both arrive yet; continuing", file=sys.stderr)
        if self.register_error:
            raise RuntimeError("; ".join(self.register_error))

        print("[BLE] BLE GATT notifier started")
        print("[BLE] adapter=%s name=%s" % (self.adapter, self.name))
        print("[BLE] service_uuid=%s" % NUS_SERVICE_UUID)
        print("[BLE] tx_notify_uuid=%s" % NUS_TX_UUID)
        print("[BLE] payload format: newline-delimited JSON, UTF-8, chunked notifications")

    def _pump_queue(self):
        while True:
            try:
                text = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                if self.tx is not None:
                    self.tx.notify_text(text)
            except Exception as e:
                print("[BLE] notify error:", repr(e), file=sys.stderr)
        return True

    def broadcast(self, data):
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.queue.put(payload)

    def close(self):
        try:
            if self.ad_manager is not None and self.adv is not None:
                self.ad_manager.UnregisterAdvertisement(self.adv.get_path())
        except Exception:
            pass
        try:
            if self.service_manager is not None and self.app is not None:
                self.service_manager.UnregisterApplication(self.app.get_path())
        except Exception:
            pass
        try:
            if self.loop is not None:
                self.loop.quit()
        except Exception:
            pass


def open_log(path):
    if not path:
        return None
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def run_inference_command(cmd, log_f, ble, args):
    print("[RUN] starting inference command:")
    print("[RUN] " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            parsed = parse_line(
                line,
                min_db=args.min_db,
                add_doa=(not args.no_doa),
                north_offset=args.north_offset,
                doa_wait_sec=args.doa_wait_sec,
                label_suffix=(not args.no_label_suffix),
                include_angle=args.include_angle,
            )
            out_line = parsed.get("raw", line.strip()) if parsed is not None else line.rstrip("\n")
            print(out_line)
            if log_f is not None:
                log_f.write(out_line + "\n")
                log_f.flush()
            if parsed is None:
                continue
            print("[BLE_JSON] " + json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))
            ble.broadcast(parsed)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Send Jetson EfficientAT DOA JSON over BLE GATT notifications.")
    parser.add_argument("--adapter", default="hci0", help="Bluetooth adapter name. Default: hci0")
    parser.add_argument("--name", default="EdgeAudio-Jetson", help="BLE advertised local name")
    parser.add_argument("--chunk-size", type=int, default=20, help="BLE notification chunk size. Default: 20")
    parser.add_argument("--log", default=None, help="Optional log path for inference stdout")
    parser.add_argument("--min-db", type=float, default=45.0, help="BLE-side guard threshold; should match jetson_live.py --min-db")
    parser.add_argument("--no-doa", action="store_true", help="Disable ReSpeaker DOA lookup")
    parser.add_argument("--north-offset", type=float, default=0.0, help="Direction calibration offset in degrees")
    parser.add_argument("--doa-wait-sec", type=float, default=0.5, help="Wait this long for a DOA value per sound result")
    parser.add_argument("--include-angle", action="store_true", help="Include raw DOA angle in JSON")
    parser.add_argument("--no-label-suffix", action="store_true", help="Keep label unchanged; direction remains in JSON")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Inference command after --")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise RuntimeError("No inference command given. Put it after --")

    ensure_doa_thread(enable=(not args.no_doa), north_offset=args.north_offset)
    ble = BleNotifier(adapter=args.adapter, name=args.name, chunk_size=args.chunk_size)
    log_f = None
    try:
        log_f = open_log(args.log)
        ble.start()
        run_inference_command(cmd, log_f, ble, args)
    except KeyboardInterrupt:
        print("\n[RUN] stopped by user")
    finally:
        ble.close()
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
