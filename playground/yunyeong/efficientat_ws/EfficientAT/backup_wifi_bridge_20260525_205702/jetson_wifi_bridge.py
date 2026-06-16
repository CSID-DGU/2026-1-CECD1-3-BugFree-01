from __future__ import print_function

import argparse
import json
import os
import re
import subprocess
import threading
import time

try:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn
except ImportError:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn


latest_lock = threading.Lock()
latest_result = {
    "status": "starting",
    "raw": "",
    "time": "",
    "label": "",
    "score": 0.0,
    "infer_sec": None,
    "total_sec": None,
    "level": "info",
    "items": []
}

LINE_RE = re.compile(
    r'^\[(?P<time>[0-9:]+)\]\s+infer=(?P<infer>[0-9.]+)s(?P<middle>.*?)\|\s+(?P<items>.*)$'
)


def risk_level(label):
    text = label.lower()

    danger_words = [
        "horn", "siren", "alarm", "scream", "screaming",
        "gun", "gunshot", "explosion", "glass", "fire",
        "smoke", "crash", "shout"
    ]

    caution_words = [
        "vehicle", "car", "truck", "dog", "cry", "baby",
        "knock", "door", "water", "engine"
    ]

    for word in danger_words:
        if word in text:
            return "danger"

    for word in caution_words:
        if word in text:
            return "caution"

    return "info"


def parse_items(items_text):
    items = []

    parts = items_text.split(" / ")

    for part in parts:
        part = part.strip()

        if not part:
            continue

        try:
            label, score_text = part.rsplit(" ", 1)
            score = float(score_text)
        except Exception:
            continue

        items.append({
            "label": label.strip(),
            "score": score
        })

    return items


def parse_line(line):
    m = LINE_RE.search(line.strip())

    if not m:
        return None

    mid = m.group("middle")
    total_sec = None

    total_match = re.search(r'total=([0-9.]+)s', mid)
    if total_match:
        total_sec = float(total_match.group(1))

    items = parse_items(m.group("items"))

    if len(items) == 0:
        return None

    top = items[0]

    return {
        "status": "ok",
        "time": m.group("time"),
        "label": top["label"],
        "score": top["score"],
        "infer_sec": float(m.group("infer")),
        "total_sec": total_sec,
        "level": risk_level(top["label"]),
        "items": items,
        "raw": line.strip()
    }


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Jetson Sound Alert</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 0;
      background: #111;
      color: #f5f5f5;
    }
    .wrap {
      padding: 24px;
      max-width: 760px;
      margin: auto;
    }
    .card {
      border-radius: 20px;
      padding: 24px;
      background: #222;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .level {
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      font-weight: 700;
      margin-bottom: 16px;
    }
    .danger { background: #ff3b30; }
    .caution { background: #ffcc00; color: #111; }
    .info { background: #34c759; color: #111; }
    .label {
      font-size: 34px;
      font-weight: 800;
      margin: 8px 0;
    }
    .score {
      font-size: 22px;
      opacity: 0.85;
    }
    .meta {
      margin-top: 18px;
      font-size: 15px;
      opacity: 0.75;
      line-height: 1.6;
    }
    .items {
      margin-top: 22px;
      padding: 16px;
      border-radius: 14px;
      background: #171717;
    }
    .item {
      padding: 7px 0;
      border-bottom: 1px solid #333;
    }
    .item:last-child {
      border-bottom: 0;
    }
    .raw {
      margin-top: 20px;
      font-size: 13px;
      opacity: 0.55;
      word-break: break-word;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h2>Jetson Nano Sound Recognition</h2>
    <div class="card">
      <div id="level" class="level info">WAITING</div>
      <div id="label" class="label">Waiting for sound...</div>
      <div id="score" class="score">score: -</div>
      <div class="meta">
        <div id="time">time: -</div>
        <div id="infer">infer: -</div>
        <div id="total">total: -</div>
      </div>
      <div class="items" id="items"></div>
      <div class="raw" id="raw"></div>
    </div>
  </div>

  <script>
    const evt = new EventSource("/events");

    function fmtScore(x) {
      if (x === null || x === undefined) return "-";
      return Number(x).toFixed(3);
    }

    evt.onmessage = function(e) {
      const d = JSON.parse(e.data);

      const level = document.getElementById("level");
      level.className = "level " + (d.level || "info");
      level.textContent = (d.level || "info").toUpperCase();

      document.getElementById("label").textContent = d.label || "Waiting for sound...";
      document.getElementById("score").textContent = "score: " + fmtScore(d.score);
      document.getElementById("time").textContent = "time: " + (d.time || "-");
      document.getElementById("infer").textContent =
        "infer: " + (d.infer_sec === null || d.infer_sec === undefined ? "-" : d.infer_sec.toFixed(3) + " sec");
      document.getElementById("total").textContent =
        "total: " + (d.total_sec === null || d.total_sec === undefined ? "-" : d.total_sec.toFixed(3) + " sec");

      const items = document.getElementById("items");
      items.innerHTML = "";

      if (d.items) {
        d.items.forEach(function(it, idx) {
          const div = document.createElement("div");
          div.className = "item";
          div.textContent = (idx + 1) + ". " + it.label + " / " + fmtScore(it.score);
          items.appendChild(div);
        });
      }

      document.getElementById("raw").textContent = d.raw || "";
    };
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
                        data = json.dumps(latest_result, ensure_ascii=False)

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
                data = json.dumps(latest_result, ensure_ascii=False).encode("utf-8")

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
    server = ThreadedHTTPServer(("0.0.0.0", port), Handler)
    print("Wi-Fi web server started.")
    print("Open this on phone browser: http://<JETSON_IP>:%d" % port)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log", default=None)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    cmd = args.cmd
    if len(cmd) > 0 and cmd[0] == "--":
        cmd = cmd[1:]

    if len(cmd) == 0:
        raise RuntimeError("No command given after --")

    log_f = None

    if args.log is not None:
        log_dir = os.path.dirname(args.log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.log, "a")

    server_thread = threading.Thread(target=run_server, args=(args.port,))
    server_thread.daemon = True
    server_thread.start()

    print("Starting inference command:")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    try:
        for line in proc.stdout:
            print(line, end="")

            if log_f is not None:
                log_f.write(line)
                log_f.flush()

            parsed = parse_line(line)

            if parsed is not None:
                with latest_lock:
                    latest_result.clear()
                    latest_result.update(parsed)

    except KeyboardInterrupt:
        print("\nstopped.")

    finally:
        try:
            proc.terminate()
        except Exception:
            pass

        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    main()
