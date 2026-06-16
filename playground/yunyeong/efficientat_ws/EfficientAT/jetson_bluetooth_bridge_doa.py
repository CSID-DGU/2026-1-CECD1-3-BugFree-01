#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano EfficientAT Bluetooth bridge with dB filtering metadata and DOA.

This is the Bluetooth/RFCOMM version of the current Wi-Fi DOA flow:
  jetson_live.py -> parse result line -> add ReSpeaker DOA direction -> send JSON

Payload format:
  one UTF-8 JSON object per line

Target:
  Jetson Nano + reSpeaker USB Mic Array + BlueZ + Android/PC Bluetooth SPP client
"""
from __future__ import print_function

import argparse
import io
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )

# New jetson_live.py output example:
# [12:00:01] infer=0.123s total=2.345s db=52.1dB | Siren 0.821 / Car horn 0.432
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

latest_lock = threading.Lock()
latest_result = {
    "type": "status",
    "status": "waiting",
    "source": "jetson_nano",
    "message": "Bluetooth bridge is running. Waiting for sound over min_db.",
    "sent_at": "",
}

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
    """Return the same 4-way Korean direction label used by the Wi-Fi DOA demo."""
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
        # ReSpeaker tuning.py convention:
        #   DOAANGLE = id 21, offset 0, type int
        #   read cmd = 0x80 | offset | 0x40 = 0xC0
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
            print(
                "[DOA] ReSpeaker USB control device not found. Check lsusb and Docker USB mount.",
                file=sys.stderr,
            )
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
    items = []
    for part in str(items_text).split(" / "):
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


def parse_line(line, min_db=None, add_doa=True, north_offset=0.0, doa_wait_sec=0.5, label_suffix=True, include_angle=False):
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

    # Safety net: current jetson_live.py already skips below --min-db before inference.
    # Keep this check here so Bluetooth also obeys the visible dB policy if older logs are replayed.
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


def run_quiet(cmd):
    try:
        return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return -1


def read_adapter_address(adapter):
    """Return the BD_ADDR for hciN, or None when it cannot be read."""
    candidates = []
    if adapter:
        candidates.append("/sys/class/bluetooth/%s/address" % adapter)
    for path in candidates:
        try:
            value = open(path, "r").read().strip()
            if value and value.count(":") == 5:
                return value.upper()
        except Exception:
            pass

    try:
        out = subprocess.check_output(["hciconfig", adapter], stderr=subprocess.STDOUT)
        if not isinstance(out, str):
            out = out.decode("utf-8", "replace")
        m = re.search(r"BD Address:\s*([0-9A-Fa-f:]{17})", out)
        if m:
            return m.group(1).upper()
    except Exception:
        pass
    return None


class BluetoothBroadcaster(object):
    def __init__(self, adapter="hci0", channel=1, backlog=1, advertise=True, bind_addr=None):
        self.adapter = adapter
        self.channel = int(channel)
        self.backlog = int(backlog)
        self.advertise_enabled = bool(advertise)
        self.bind_addr = bind_addr
        self.bound_addr = None
        self.server = None
        self.clients = []
        self.clients_lock = threading.Lock()
        self.stop_event = threading.Event()

    def prepare_adapter(self):
        # Keep adapter visible/connectable.  Do not register SDP here.
        # SDP is registered only after the RFCOMM socket has successfully bound;
        # otherwise stale Serial Port records can point clients to a dead channel.
        run_quiet(["hciconfig", self.adapter, "up"])
        run_quiet(["hciconfig", self.adapter, "piscan"])

    def _delete_stale_serial_port_records(self):
        if not self.advertise_enabled:
            return
        try:
            out = subprocess.check_output(["sdptool", "browse", "local"], stderr=subprocess.STDOUT)
            if not isinstance(out, str):
                out = out.decode("utf-8", "replace")
        except Exception:
            return

        handles = []
        in_serial = False
        for line in out.splitlines():
            if "Service Name: Serial Port" in line:
                in_serial = True
                continue
            if in_serial and "Service RecHandle:" in line:
                parts = line.strip().split()
                if parts:
                    handles.append(parts[-1])
                in_serial = False

        for handle in handles:
            run_quiet(["sdptool", "del", handle])

    def _advertise_serial_port(self):
        if not self.advertise_enabled:
            return
        self._delete_stale_serial_port_records()
        rc = run_quiet(["sdptool", "add", "--channel=%d" % self.channel, "SP"])
        if rc != 0:
            print(
                "[BT] warning: sdptool advertisement failed. Pairing may still work, but some clients may not find SPP.",
                file=sys.stderr,
            )
        else:
            print("[BT] SDP Serial Port advertised on RFCOMM channel %d" % self.channel)

    def _candidate_bind_addresses(self):
        # Some Python/BlueZ builds accept "" as BDADDR_ANY, but Jetson/Ubuntu 18.04
        # can raise OSError("bad bluetooth address") for it. Prefer an explicit
        # local adapter address, then fall back to Linux BDADDR_ANY.
        seen = set()
        addresses = []
        for value in [self.bind_addr, os.environ.get("BT_BIND_ADDR"), read_adapter_address(self.adapter), "00:00:00:00:00:00", ""]:
            if value is None:
                continue
            value = str(value).strip()
            if value not in seen:
                seen.add(value)
                addresses.append(value)
        return addresses

    def _make_server_socket(self):
        s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s

    def _candidate_channels(self):
        # RFCOMM user channels are usually 1..30.  Try the requested channel first,
        # then fall back to other channels so a stale server on channel 3 does not
        # block the whole demo.
        preferred = int(self.channel)
        channels = []
        if 1 <= preferred <= 30:
            channels.append(preferred)
        for ch in range(1, 31):
            if ch not in channels:
                channels.append(ch)
        return channels

    def start(self):
        if not hasattr(socket, "AF_BLUETOOTH") or not hasattr(socket, "BTPROTO_RFCOMM"):
            raise RuntimeError("This Python build does not expose AF_BLUETOOTH/BTPROTO_RFCOMM.")

        self.prepare_adapter()

        errors = []
        for ch in self._candidate_channels():
            for addr in self._candidate_bind_addresses():
                s = self._make_server_socket()
                try:
                    s.bind((addr, ch))
                    self.server = s
                    self.bound_addr = addr
                    self.channel = ch
                    break
                except OSError as e:
                    errors.append("ch%d %s -> %s" % (ch, repr(addr), repr(e)))
                    try:
                        s.close()
                    except Exception:
                        pass
            if self.server is not None:
                break

        if self.server is None:
            raise OSError("Could not bind RFCOMM server. Tried: %s" % ("; ".join(errors)))

        self.server.listen(self.backlog)
        self._advertise_serial_port()

        t = threading.Thread(target=self._accept_loop)
        t.daemon = True
        t.start()

        print("[BT] RFCOMM/SPP server started")
        print("[BT] adapter=%s channel=%d bind_addr=%s" % (self.adapter, self.channel, self.bound_addr))
        print("[BT] payload format: one JSON object per line")

    def _accept_loop(self):
        while not self.stop_event.is_set():
            try:
                client, address = self.server.accept()
                client.settimeout(5.0)
                with self.clients_lock:
                    self.clients.append(client)
                print("[BT] client connected:", address)

                hello = {
                    "type": "status",
                    "status": "connected",
                    "source": "jetson_nano",
                    "sent_at": now_iso(),
                    "message": "Bluetooth bridge connected. Sound results are sent only when jetson_live.py prints an above-min-db event.",
                }
                self._send_one(client, hello)
            except Exception as e:
                if not self.stop_event.is_set():
                    print("[BT] accept error:", repr(e), file=sys.stderr)
                time.sleep(0.5)

    def _send_one(self, client, data):
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        client.sendall(payload.encode("utf-8"))

    def broadcast(self, data):
        dead = []
        with self.clients_lock:
            clients = list(self.clients)

        for client in clients:
            try:
                self._send_one(client, data)
            except Exception:
                dead.append(client)

        if dead:
            with self.clients_lock:
                for client in dead:
                    try:
                        self.clients.remove(client)
                    except ValueError:
                        pass
                    try:
                        client.close()
                    except Exception:
                        pass
            print("[BT] removed disconnected client(s):", len(dead))

    def close(self):
        self.stop_event.set()
        with self.clients_lock:
            clients = list(self.clients)
            self.clients = []
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        if self.server is not None:
            try:
                self.server.close()
            except Exception:
                pass


def heartbeat_loop(bt, interval):
    if interval <= 0:
        return
    while True:
        time.sleep(interval)
        bt.broadcast({
            "type": "heartbeat",
            "status": "alive",
            "source": "jetson_nano",
            "sent_at": now_iso(),
        })


def open_log(path):
    if not path:
        return None
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def run_inference_command(cmd, log_f, bt, args):
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

            if parsed is not None:
                out_line = parsed.get("raw", line.strip())
            else:
                out_line = line.rstrip("\n")

            print(out_line)
            if log_f is not None:
                log_f.write(out_line + "\n")
                log_f.flush()

            if parsed is None:
                continue

            with latest_lock:
                latest_result.clear()
                latest_result.update(parsed)

            # Local debug mirror: lets you verify the exact Bluetooth payload
            # from the Jetson terminal/log even before a mobile app exists.
            print("[BT_JSON] " + json.dumps(parsed, ensure_ascii=False, separators=(",", ":")))
            bt.broadcast(parsed)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Send Jetson EfficientAT above-min-db DOA results over Bluetooth RFCOMM/SPP."
    )
    parser.add_argument("--adapter", default="hci0", help="Bluetooth adapter name. Default: hci0")
    parser.add_argument("--channel", type=int, default=1, help="RFCOMM channel. Default: 1")
    parser.add_argument("--bind-addr", default=None, help="Optional local Bluetooth BD_ADDR to bind. Default: auto-detect adapter address.")
    parser.add_argument("--backlog", type=int, default=1, help="Bluetooth client backlog. Default: 1")
    parser.add_argument("--no-advertise", action="store_true", help="Do not run hciconfig/sdptool advertisement commands.")
    parser.add_argument("--log", default=None, help="Optional log path for inference stdout.")
    parser.add_argument("--heartbeat-sec", type=float, default=0.0, help="Send lightweight heartbeat every N seconds. 0 disables it.")
    parser.add_argument("--min-db", type=float, default=45.0, help="Bluetooth-side guard threshold; should match jetson_live.py --min-db.")
    parser.add_argument("--no-doa", action="store_true", help="Disable ReSpeaker DOA lookup.")
    parser.add_argument("--north-offset", type=float, default=0.0, help="Direction calibration offset in degrees.")
    parser.add_argument("--doa-wait-sec", type=float, default=0.5, help="Wait this long for a DOA value per sound result.")
    parser.add_argument("--include-angle", action="store_true", help="Include raw DOA angle in JSON. Default sends only 북/동/남/서.")
    parser.add_argument("--no-label-suffix", action="store_true", help="Keep label unchanged; direction remains in the JSON direction field.")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Inference command after --")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise RuntimeError("No inference command given. Put it after --")

    ensure_doa_thread(enable=(not args.no_doa), north_offset=args.north_offset)

    bt = BluetoothBroadcaster(
        adapter=args.adapter,
        channel=args.channel,
        backlog=args.backlog,
        advertise=(not args.no_advertise),
        bind_addr=args.bind_addr,
    )
    log_f = None

    try:
        log_f = open_log(args.log)
        bt.start()

        hb = threading.Thread(target=heartbeat_loop, args=(bt, args.heartbeat_sec))
        hb.daemon = True
        hb.start()

        run_inference_command(cmd, log_f, bt, args)
    except KeyboardInterrupt:
        print("\n[RUN] stopped by user")
    finally:
        bt.close()
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
