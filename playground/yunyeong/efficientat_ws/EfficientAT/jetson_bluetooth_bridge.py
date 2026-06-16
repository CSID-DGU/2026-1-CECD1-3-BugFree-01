#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Nano EfficientAT Bluetooth bridge

- Runs an inference command, usually jetson_live.py.
- Parses the same output format used by jetson_wifi_bridge.py.
- Sends each parsed result to connected Bluetooth Classic RFCOMM/SPP clients
  as one UTF-8 JSON object per line.

Tested design target:
  Jetson Nano + Ubuntu/L4T + BlueZ + Android/PC Bluetooth serial client.
"""
from __future__ import print_function

import argparse
import io
import json
import os
import re
import socket
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
    "explosion", "glass", "fire", "smoke", "crash", "shout"
]
CAUTION_WORDS = [
    "vehicle", "car", "truck", "dog", "cry", "baby", "knock", "door",
    "water", "engine"
]

latest_lock = threading.Lock()
latest_result = {
    "type": "status",
    "status": "starting",
    "source": "jetson_nano",
    "raw": "",
    "time": "",
    "label": "",
    "score": 0.0,
    "infer_sec": None,
    "total_sec": None,
    "level": "info",
    "items": [],
}


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def risk_level(label):
    text = str(label).lower()
    for word in DANGER_WORDS:
        if word in text:
            return "danger"
    for word in CAUTION_WORDS:
        if word in text:
            return "caution"
    return "info"


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


def parse_line(line):
    m = LINE_RE.search(line.strip())
    if not m:
        return None

    total_sec = None
    total_match = re.search(r"total=([0-9.]+)s", m.group("middle"))
    if total_match:
        try:
            total_sec = float(total_match.group(1))
        except Exception:
            total_sec = None

    items = parse_items(m.group("items"))
    if not items:
        return None

    top = items[0]
    return {
        "type": "sound_result",
        "status": "ok",
        "source": "jetson_nano",
        "sent_at": now_iso(),
        "time": m.group("time"),
        "label": top["label"],
        "score": top["score"],
        "infer_sec": float(m.group("infer")),
        "total_sec": total_sec,
        "level": risk_level(top["label"]),
        "items": items,
        "raw": line.strip(),
    }


def run_quiet(cmd):
    try:
        return subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return -1


class BluetoothBroadcaster(object):
    def __init__(self, adapter="hci0", channel=1, backlog=1, advertise=True):
        self.adapter = adapter
        self.channel = int(channel)
        self.backlog = int(backlog)
        self.advertise_enabled = bool(advertise)
        self.server = None
        self.clients = []
        self.clients_lock = threading.Lock()
        self.stop_event = threading.Event()

    def prepare_adapter(self):
        if not self.advertise_enabled:
            return

        # Classic Bluetooth discoverable/connectable mode. This may fail inside
        # Docker on some setups; pairing can still be done on the host with bluetoothctl.
        run_quiet(["hciconfig", self.adapter, "up"])
        run_quiet(["hciconfig", self.adapter, "piscan"])

        # SDP advertisement for Serial Port Profile. Android serial-terminal apps
        # normally find the service via this record.
        rc = run_quiet(["sdptool", "add", "--channel=%d" % self.channel, "SP"])
        if rc != 0:
            print("[BT] warning: sdptool advertisement failed. Pairing may still work, but some clients may not find SPP.", file=sys.stderr)

    def start(self):
        if not hasattr(socket, "AF_BLUETOOTH") or not hasattr(socket, "BTPROTO_RFCOMM"):
            raise RuntimeError("This Python build does not expose AF_BLUETOOTH/BTPROTO_RFCOMM.")

        self.prepare_adapter()

        self.server = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("", self.channel))
        self.server.listen(self.backlog)

        t = threading.Thread(target=self._accept_loop)
        t.daemon = True
        t.start()

        print("[BT] RFCOMM/SPP server started")
        print("[BT] adapter=%s channel=%d" % (self.adapter, self.channel))
        print("[BT] payload format: one JSON object per line")

    def _accept_loop(self):
        while not self.stop_event.is_set():
            try:
                client, address = self.server.accept()
                client.settimeout(5.0)
                with self.clients_lock:
                    self.clients.append(client)
                print("[BT] client connected:", address)

                with latest_lock:
                    initial = dict(latest_result)
                    initial["sent_at"] = now_iso()
                self._send_one(client, initial)
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
        with latest_lock:
            data = dict(latest_result)
            data["type"] = "heartbeat" if data.get("status") != "ok" else "sound_result"
            data["sent_at"] = now_iso()
        bt.broadcast(data)


def open_log(path):
    if not path:
        return None
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def run_inference_command(cmd, log_f, bt):
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
            print(line, end="")
            if log_f is not None:
                log_f.write(line)
                log_f.flush()

            parsed = parse_line(line)
            if parsed is None:
                continue

            with latest_lock:
                latest_result.clear()
                latest_result.update(parsed)

            bt.broadcast(parsed)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Send Jetson EfficientAT results over Bluetooth RFCOMM/SPP.")
    parser.add_argument("--adapter", default="hci0", help="Bluetooth adapter name. Default: hci0")
    parser.add_argument("--channel", type=int, default=1, help="RFCOMM channel. Default: 1")
    parser.add_argument("--backlog", type=int, default=1, help="Bluetooth client backlog. Default: 1")
    parser.add_argument("--no-advertise", action="store_true", help="Do not run hciconfig/sdptool advertisement commands.")
    parser.add_argument("--log", default=None, help="Optional log path for inference stdout.")
    parser.add_argument("--heartbeat-sec", type=float, default=10.0, help="Send latest state every N seconds. 0 disables heartbeat.")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Inference command after --")
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise RuntimeError("No inference command given. Put it after --")

    bt = BluetoothBroadcaster(
        adapter=args.adapter,
        channel=args.channel,
        backlog=args.backlog,
        advertise=(not args.no_advertise),
    )
    log_f = None

    try:
        log_f = open_log(args.log)
        bt.start()

        hb = threading.Thread(target=heartbeat_loop, args=(bt, args.heartbeat_sec))
        hb.daemon = True
        hb.start()

        run_inference_command(cmd, log_f, bt)
    except KeyboardInterrupt:
        print("\n[RUN] stopped by user")
    finally:
        bt.close()
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
