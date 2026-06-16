#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wi-Fi web bridge for fine-tuned EfficientAT on Jetson Nano.

This script launches an inference command, parses lines printed by
jetson_live_finetuned.py, and exposes the latest prediction through:
  - http://<JETSON_IP>:8765/        mobile-friendly dashboard
  - http://<JETSON_IP>:8765/latest  JSON snapshot
  - http://<JETSON_IP>:8765/events  Server-Sent Events stream

It is intentionally stdlib-only. DOA support is optional and uses pyusb only
when --enable-doa is passed.
"""
from __future__ import print_function

import argparse
import io
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
except ImportError:  # Python 2 fallback, not expected here
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn

# Keep Korean labels / degree symbols safe when Docker stdout is redirected.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

LINE_RE = re.compile(
    r"^\[(?P<time>[0-9:]+)\]\s+infer=(?P<infer>[0-9.]+)s\s+(?P<middle>.*?)\|\s+(?P<items>.+)$"
)

LABEL_KO = {
    "alarm": "경보",
    "baby_cry": "아기 울음",
    "bicycle": "자전거",
    "cat_meow": "고양이",
    "crying": "울음",
    "dog_bark": "강아지",
    "fire_alarm": "화재경보",
    "glass_break": "유리 깨짐",
    "gunshot": "총성",
    "scream": "비명",
    "water": "물소리",
}

DANGER_WORDS = [
    "fire_alarm", "fire", "alarm", "scream", "gunshot", "gun", "glass_break",
    "glass", "siren", "horn", "explosion", "crash", "shout",
]
CAUTION_WORDS = [
    "baby_cry", "crying", "cry", "dog_bark", "cat_meow", "dog", "cat",
    "water", "bicycle", "vehicle", "car", "truck", "knock", "door",
]

latest_lock = threading.Lock()
latest_result = {
    "status": "starting",
    "raw": "",
    "time": "",
    "label": "",
    "label_ko": "",
    "score": 0.0,
    "infer_sec": None,
    "total_sec": None,
    "db": None,
    "level": "info",
    "items": [],
    "direction": None,
    "angle": None,
    "doa_status": "disabled",
    "updated_at": time.time(),
}

_doa_lock = threading.Lock()
_doa_started = False
_doa_angle = None
_doa_direction = None
_doa_status = "disabled"
_doa_north_offset = 0.0


def label_to_ko(label):
    return LABEL_KO.get(str(label), str(label))


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
    for part in items_text.split(" / "):
        part = part.strip()
        if not part:
            continue
        try:
            label, score_text = part.rsplit(" ", 1)
            score = float(score_text)
        except Exception:
            continue
        label = label.strip()
        items.append({
            "label": label,
            "label_ko": label_to_ko(label),
            "score": score,
            "level": risk_level(label),
        })
    return items


def angle_to_cardinal(angle, north_offset=0.0):
    corrected = (float(angle) - float(north_offset)) % 360.0
    if corrected < 45.0 or corrected >= 315.0:
        return "북"
    if corrected < 135.0:
        return "동"
    if corrected < 225.0:
        return "남"
    return "서"


class InlineTuning(object):
    """Minimal ReSpeaker USB 4 Mic Array DOA reader.

    Reads DOAANGLE from the USB control interface. Used only when the official
    tuning.py is not available.
    """

    TIMEOUT = 100000

    def __init__(self, dev, usb_util):
        self.dev = dev
        self.usb_util = usb_util

    @property
    def direction(self):
        response = self.dev.ctrl_transfer(
            self.usb_util.CTRL_IN | self.usb_util.CTRL_TYPE_VENDOR | self.usb_util.CTRL_RECIPIENT_DEVICE,
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
        value, exponent = struct.unpack(b"ii", data)
        return int(value)


def _set_doa_status(status):
    global _doa_status
    with _doa_lock:
        _doa_status = status


def _create_tuning(dev):
    try:
        from tuning import Tuning
        return Tuning(dev)
    except Exception:
        import usb.util
        return InlineTuning(dev, usb.util)


def _doa_loop():
    global _doa_angle, _doa_direction
    try:
        import usb.core
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if dev is None:
            _set_doa_status("usb_control_not_found")
            print("[DOA] ReSpeaker USB control device not found. Check lsusb and Docker USB mount.", file=sys.stderr)
            return
        tuning = _create_tuning(dev)
        _set_doa_status("enabled")
        print("[DOA] ReSpeaker DOA reader enabled.", file=sys.stderr)
        while True:
            try:
                angle = int(float(tuning.direction)) % 360
                direction = angle_to_cardinal(angle, _doa_north_offset)
                with _doa_lock:
                    _doa_angle = angle
                    _doa_direction = direction
            except Exception as e:
                _set_doa_status("read_error")
                print("[DOA] read error: %r" % (e,), file=sys.stderr)
            time.sleep(0.20)
    except Exception as e:
        _set_doa_status("disabled: %r" % (e,))
        print("[DOA] disabled: %r" % (e,), file=sys.stderr)


def ensure_doa_thread(enable_doa, north_offset):
    global _doa_started, _doa_north_offset
    if not enable_doa:
        _set_doa_status("disabled")
        return
    _doa_north_offset = float(north_offset)
    if _doa_started:
        return
    _doa_started = True
    t = threading.Thread(target=_doa_loop)
    t.daemon = True
    t.start()


def current_doa_snapshot(wait_sec=0.0):
    deadline = time.time() + max(0.0, float(wait_sec))
    while True:
        with _doa_lock:
            angle = _doa_angle
            direction = _doa_direction
            status = _doa_status
        if direction is not None or time.time() >= deadline:
            return angle, direction, status
        time.sleep(0.05)


def parse_line(line, enable_doa=False):
    text = line.strip()
    m = LINE_RE.search(text)
    if not m:
        return None

    middle = m.group("middle") or ""
    total_sec = None
    db_value = None

    total_match = re.search(r"total=([0-9.]+)s", middle)
    if total_match:
        try:
            total_sec = float(total_match.group(1))
        except Exception:
            total_sec = None

    db_match = re.search(r"db=([0-9.]+)dB", middle)
    if db_match:
        try:
            db_value = float(db_match.group(1))
        except Exception:
            db_value = None

    items = parse_items(m.group("items"))
    if len(items) == 0:
        return None
    top = items[0]

    angle, direction, doa_status = current_doa_snapshot(wait_sec=0.35 if enable_doa else 0.0)
    label = top["label"]
    label_ko = top["label_ko"]
    display_label = label_ko
    if direction:
        display_label = "%s [%s]" % (display_label, direction)

    raw = text
    if direction:
        raw = "%s | DOA %s" % (raw, direction)
    elif enable_doa:
        raw = "%s | DOA unavailable:%s" % (raw, doa_status)

    parsed = {
        "status": "ok",
        "time": m.group("time"),
        "label": label,
        "label_ko": label_ko,
        "display_label": display_label,
        "score": top["score"],
        "infer_sec": float(m.group("infer")),
        "total_sec": total_sec,
        "db": db_value,
        "level": top["level"],
        "items": items,
        "raw": raw,
        "direction": direction,
        "angle": angle,
        "doa_status": doa_status,
        "updated_at": time.time(),
    }
    return parsed


HTML = r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Jetson Sound Recognition</title>
<style>
:root { color-scheme: dark; --bg:#101114; --card:#1b1d22; --muted:#9ba3af; --line:#2a2d35; --ok:#3ecf8e; --caution:#facc15; --danger:#ff5a5f; }
* { box-sizing: border-box; }
body { margin:0; background: radial-gradient(circle at top, #20232b, var(--bg)); color:#f6f7fb; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.wrap { max-width:760px; margin:0 auto; padding:22px 16px 36px; }
.header { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:16px; }
h1 { font-size:22px; margin:0; letter-spacing:-0.02em; }
.status { font-size:13px; color:var(--muted); }
.card { background:rgba(27,29,34,.94); border:1px solid var(--line); border-radius:22px; padding:22px; box-shadow:0 18px 50px rgba(0,0,0,.28); }
.badge { display:inline-flex; align-items:center; border-radius:999px; padding:8px 12px; font-size:13px; font-weight:800; background:#2a2d35; color:#e5e7eb; }
.badge.danger { background:rgba(255,90,95,.16); color:var(--danger); }
.badge.caution { background:rgba(250,204,21,.16); color:var(--caution); }
.badge.info { background:rgba(62,207,142,.14); color:var(--ok); }
.main-label { font-size:42px; line-height:1.05; margin:18px 0 10px; font-weight:850; letter-spacing:-0.04em; }
.score { font-size:20px; color:#e5e7eb; margin-bottom:18px; }
.metrics { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px; margin:16px 0; }
.metric { border:1px solid var(--line); border-radius:16px; padding:12px; background:#15171c; }
.metric .k { font-size:12px; color:var(--muted); margin-bottom:4px; }
.metric .v { font-size:16px; font-weight:700; }
ul { list-style:none; padding:0; margin:18px 0 0; border-top:1px solid var(--line); }
li { display:flex; justify-content:space-between; gap:10px; padding:13px 0; border-bottom:1px solid var(--line); font-size:15px; }
.raw { margin-top:16px; color:var(--muted); font-size:12px; line-height:1.45; word-break:break-word; }
.small { color:var(--muted); font-size:12px; margin-top:12px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <h1>Jetson Nano Sound Recognition</h1>
    <div id="conn" class="status">connecting...</div>
  </div>
  <div class="card">
    <div id="badge" class="badge info">WAITING</div>
    <div id="label" class="main-label">Waiting...</div>
    <div id="score" class="score">score: -</div>
    <div class="metrics">
      <div class="metric"><div class="k">time</div><div id="time" class="v">-</div></div>
      <div class="metric"><div class="k">dB</div><div id="db" class="v">-</div></div>
      <div class="metric"><div class="k">infer</div><div id="infer" class="v">-</div></div>
      <div class="metric"><div class="k">total</div><div id="total" class="v">-</div></div>
    </div>
    <ul id="items"></ul>
    <div id="raw" class="raw"></div>
    <div id="doa" class="small"></div>
  </div>
</div>
<script>
function levelText(level){ if(level==='danger') return 'DANGER'; if(level==='caution') return 'CAUTION'; return 'INFO'; }
function pct(x){ return (Number(x || 0) * 100).toFixed(1) + '%'; }
function sec(x){ return x === null || x === undefined ? '-' : Number(x).toFixed(3) + 's'; }
function db(x){ return x === null || x === undefined ? '-' : Number(x).toFixed(1) + 'dB'; }
function render(d){
  document.getElementById('conn').textContent = d.status || 'ok';
  const badge = document.getElementById('badge');
  badge.className = 'badge ' + (d.level || 'info');
  badge.textContent = levelText(d.level || 'info');
  document.getElementById('label').textContent = d.display_label || d.label_ko || d.label || 'Waiting...';
  document.getElementById('score').textContent = 'score: ' + pct(d.score);
  document.getElementById('time').textContent = d.time || '-';
  document.getElementById('db').textContent = db(d.db);
  document.getElementById('infer').textContent = sec(d.infer_sec);
  document.getElementById('total').textContent = sec(d.total_sec);
  document.getElementById('raw').textContent = d.raw || '';
  document.getElementById('doa').textContent = d.direction ? ('direction: ' + d.direction + ' / angle: ' + d.angle + '°') : ('DOA: ' + (d.doa_status || 'disabled'));
  const ul = document.getElementById('items');
  ul.innerHTML = '';
  (d.items || []).forEach(function(it, idx){
    const li = document.createElement('li');
    li.innerHTML = '<span>' + (idx+1) + '. ' + (it.label_ko || it.label) + '</span><strong>' + pct(it.score) + '</strong>';
    ul.appendChild(li);
  });
}
const es = new EventSource('/events');
es.onopen = function(){ document.getElementById('conn').textContent = 'connected'; };
es.onerror = function(){ document.getElementById('conn').textContent = 'reconnecting...'; };
es.onmessage = function(ev){ try { render(JSON.parse(ev.data)); } catch(e){} };
fetch('/latest').then(r => r.json()).then(render).catch(()=>{});
</script>
</body>
</html>
"""


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last = None
            while True:
                try:
                    with latest_lock:
                        data = json.dumps(latest_result, ensure_ascii=False, sort_keys=True)
                    if data != last:
                        self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
                        self.wfile.flush()
                        last = data
                    time.sleep(0.2)
                except Exception:
                    break
            return

        if path == "/latest":
            with latest_lock:
                data = json.dumps(latest_result, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))


def run_server(port):
    server = ThreadedHTTPServer(("0.0.0.0", int(port)), Handler)
    print("Wi-Fi web server started.")
    print("Open this on phone browser: http://<JETSON_IP>:%d" % int(port))
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log", default=None)
    parser.add_argument("--enable-doa", action="store_true", help="Read ReSpeaker DOA direction through pyusb")
    parser.add_argument("--north-offset", type=float, default=0.0, help="DOA angle treated as North. Default: 0")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if len(cmd) > 0 and cmd[0] == "--":
        cmd = cmd[1:]
    if len(cmd) == 0:
        raise RuntimeError("No command given after --")

    ensure_doa_thread(args.enable_doa, args.north_offset)

    log_f = None
    if args.log:
        log_dir = os.path.dirname(args.log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.log, "a", encoding="utf-8")

    server_thread = threading.Thread(target=run_server, args=(args.port,))
    server_thread.daemon = True
    server_thread.start()

    print("Starting inference command:")
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    return_code = None
    try:
        for line in proc.stdout:
            parsed = parse_line(line, enable_doa=args.enable_doa)
            out_line = parsed.get("raw", line.rstrip("\n")) if parsed is not None else line.rstrip("\n")
            print(out_line)
            if log_f is not None:
                log_f.write(out_line + "\n")
                log_f.flush()
            if parsed is not None:
                with latest_lock:
                    latest_result.clear()
                    latest_result.update(parsed)
        return_code = proc.wait()
        if return_code != 0:
            msg = "inference process exited with code %s" % return_code
            print("[bridge] " + msg, file=sys.stderr)
            with latest_lock:
                latest_result.clear()
                latest_result.update({"status": "error", "raw": msg, "level": "danger", "items": [], "updated_at": time.time()})
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        if log_f is not None:
            log_f.close()
    if return_code is None:
        return_code = 0
    sys.exit(return_code)


if __name__ == "__main__":
    main()
