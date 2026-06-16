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
except ImportError:  # Python 2 fallback kept for compatibility with older Jetson images
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn

DISPLAY_SCORE_THRESHOLD = float(os.environ.get("DISPLAY_SCORE_THRESHOLD", "0.70"))
DISPLAY_MIN_DB = float(os.environ.get("DISPLAY_MIN_DB", "45.0"))

latest_lock = threading.Lock()
latest_result = {
    "status": "starting",
    "display_state": "listening",
    "raw": "",
    "time": "",
    "label": "",
    "display_label": "소리를 듣고 있어요",
    "display_icon": "〰️",
    "score": 0.0,
    "display_score": None,
    "infer_sec": None,
    "total_sec": None,
    "db": None,
    "level": "listening",
    "items": [],
    "strong_items": [],
    "direction": None,
    "angle": None,
    "direction_text": "",
    "thresholds": {
        "score": DISPLAY_SCORE_THRESHOLD,
        "min_db": DISPLAY_MIN_DB,
    },
}

LINE_RE = re.compile(
    r'^\[(?P<time>[0-9:]+)\]\s+infer=(?P<infer>[0-9.]+)s(?P<middle>.*?)\|\s+(?P<items>.*)$'
)

DB_RE = re.compile(r'db=([-+]?[0-9.]+)\s*dB', re.IGNORECASE)

KOREAN_LABELS = {
    # Fine-tuned project labels
    "cat_meow": "고양이 소리",
    "dog_bark": "강아지 짖는 소리",
    "animal_cry": "동물 소리",
    "baby_cry": "아기 울음소리",
    "alarm_siren": "경보/사이렌 소리",
    "glass_shatter": "유리 깨지는 소리",
    "fire_alarm": "화재 경보 소리",
    "door_knock": "노크 소리",
    "car_horn": "차량 경적 소리",

    # Common English labels from AudioSet / ESC-50 / EfficientAT
    "cat": "고양이 소리",
    "meow": "고양이 소리",
    "caterwaul": "고양이 울음소리",
    "dog": "강아지 소리",
    "bark": "강아지 짖는 소리",
    "crying baby": "아기 울음소리",
    "baby cry": "아기 울음소리",
    "infant cry": "아기 울음소리",
    "cry": "울음소리",
    "crying": "울음소리",
    "scream": "비명 소리",
    "screaming": "비명 소리",
    "shout": "고함 소리",
    "yell": "고함 소리",
    "horn": "경적 소리",
    "vehicle horn": "차량 경적 소리",
    "car horn": "차량 경적 소리",
    "honking": "차량 경적 소리",
    "vehicle horn, car horn, honking": "차량 경적 소리",
    "siren": "사이렌 소리",
    "alarm": "경보 소리",
    "alarm clock": "알람 시계 소리",
    "gunshot": "총소리",
    "gunfire": "총소리",
    "explosion": "폭발음",
    "fire": "불소리",
    "crackling fire": "불타는 소리",
    "construction": "공사장 소리",
    "jackhammer": "착암기 소리",
    "drill": "드릴 소리",
    "water": "물소리",
    "rain": "빗소리",
    "raindrop": "빗방울 소리",
    "water tap": "수돗물 소리",
    "pour": "물 따르는 소리",
    "pouring water": "물 따르는 소리",
    "knock": "노크 소리",
    "door knock": "노크 소리",
    "door": "문 소리",
    "door, wood creaks": "문 삐걱이는 소리",
    "appliances": "가전제품 소리",
    "vacuum cleaner": "청소기 소리",
    "washing machine": "세탁기 소리",
    "glass": "유리 소리",
    "shatter": "깨지는 소리",
    "glass breaking": "유리 깨지는 소리",
    "breaking": "깨지는 소리",
    "bicycle": "자전거 소리",
    "bicycle bell": "자전거 벨 소리",
    "engine": "엔진 소리",
    "vehicle": "차량 소리",
    "car": "자동차 소리",
    "truck": "트럭 소리",
    "listening": "소리를 듣고 있어요",
}

# Fallback substring rules for labels not exactly listed above.
# This keeps the UI Korean even when the model prints labels such as
# "Emergency vehicle siren" or "Domestic animals, pets".
KOREAN_LABEL_KEYWORDS = [
    (("vehicle horn", "car horn", "honking", "horn"), "차량 경적 소리"),
    (("siren",), "사이렌 소리"),
    (("alarm",), "경보 소리"),
    (("scream", "screaming"), "비명 소리"),
    (("shout", "yell"), "고함 소리"),
    (("gunshot", "gunfire", "gun"), "총소리"),
    (("explosion", "explosive"), "폭발음"),
    (("glass", "shatter", "breaking"), "유리 깨지는 소리"),
    (("fire", "smoke"), "화재/불소리"),
    (("cat", "meow", "caterwaul"), "고양이 소리"),
    (("dog", "bark"), "강아지 소리"),
    (("baby", "infant"), "아기 울음소리"),
    (("cry", "crying"), "울음소리"),
    (("water", "rain", "raindrop", "tap", "pour"), "물소리"),
    (("knock",), "노크 소리"),
    (("door",), "문 소리"),
    (("vacuum", "washing", "appliance"), "가전제품 소리"),
    (("construction", "jackhammer", "drill"), "공사장 소리"),
    (("bicycle",), "자전거 소리"),
    (("engine", "vehicle", "car", "truck"), "차량 소리"),
]

ICON_MAP = [
    (("cat", "meow"), "🐱"),
    (("dog",), "🐶"),
    (("baby", "cry"), "👶"),
    (("horn", "vehicle", "car"), "🚗"),
    (("siren", "alarm"), "⚠️"),
    (("gun", "gunshot", "explosion"), "🚨"),
    (("glass", "shatter"), "🔔"),
    (("water", "rain", "tap"), "💧"),
    (("knock", "door"), "🚪"),
    (("appliance", "vacuum", "washing"), "🔌"),
    (("construction", "drill", "jackhammer"), "🏗️"),
]


def normalize_label(label):
    """Return a Korean display label for any model label.

    The model can print mixed sources: project labels such as cat_meow,
    AudioSet labels such as "Vehicle horn, car horn, honking", and short
    labels such as scream. The UI should not expose raw English labels.
    """
    if not label:
        return ""

    raw = str(label).strip()
    key = raw.lower().strip().replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)

    if key in KOREAN_LABELS:
        return KOREAN_LABELS[key]

    # For compound labels, translate the first known component.
    for sep in [",", "/", "|"]:
        if sep in key:
            for part in [x.strip() for x in key.split(sep) if x.strip()]:
                if part in KOREAN_LABELS:
                    return KOREAN_LABELS[part]

    for keywords, korean in KOREAN_LABEL_KEYWORDS:
        for word in keywords:
            if word in key:
                return korean

    # Last resort: avoid showing raw English in the central UI.
    # The original label is still kept internally in the `label` field for debugging.
    return "알 수 없는 소리"


def icon_for_label(label):
    text = (label or "").lower()
    for words, icon in ICON_MAP:
        for word in words:
            if word in text:
                return icon
    return "🔊"


def risk_level(label):
    text = (label or "").lower()

    danger_words = [
        "horn", "siren", "alarm", "scream", "screaming",
        "gun", "gunshot", "explosion", "glass", "fire",
        "smoke", "crash", "shout"
    ]

    caution_words = [
        "vehicle", "car", "truck", "dog", "cry", "baby",
        "knock", "door", "water", "engine", "cat", "animal"
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

    # DOA is appended by jetson_wifi_bridge_finetuned.py after parsing. If a log line
    # already contains it, ignore that suffix so it is not treated as a class label.
    items_text = items_text.split("| DOA", 1)[0].strip()
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
            "display_label": normalize_label(label),
            "score": score,
        })

    return items


def format_items_korean(items):
    if not items:
        return ""
    return " / ".join(
        "%s %.3f" % (normalize_label(it.get("label", "")), float(it.get("score", 0.0)))
        for it in items
    )


def make_display_raw(parsed, items=None):
    """Build a Korean-only raw line for the bottom debug area."""
    if parsed is None:
        return ""
    shown_items = items if items is not None else parsed.get("items", [])
    base = "[%s] 추론=%.3f초" % (parsed.get("time") or "--:--:--", float(parsed.get("infer_sec") or 0.0))
    if parsed.get("total_sec") is not None:
        base += " 전체=%.3f초" % float(parsed.get("total_sec"))
    if parsed.get("db") is not None:
        base += " 음량=%.1fdB" % float(parsed.get("db"))
    if shown_items:
        base += " | " + format_items_korean(shown_items)
    else:
        base += " | 소리를 듣고 있어요"
    if parsed.get("direction_text"):
        base += " | 방향 " + str(parsed.get("direction_text"))
    return base


def format_direction(direction, angle):
    if angle is None:
        return ""
    try:
        deg = int(round(float(angle))) % 360
    except Exception:
        return ""
    if direction:
        return "%s쪽 %d°" % (direction, deg)
    return "%d°" % deg


def _db_is_too_low(db_value):
    return db_value is not None and float(db_value) <= DISPLAY_MIN_DB


def _score_is_confident(score):
    try:
        return float(score) > DISPLAY_SCORE_THRESHOLD
    except Exception:
        return False


def apply_display_fields(parsed):
    """Apply UI policy without mutating the raw inference line.

    UI policy:
    - show classification only when top score > DISPLAY_SCORE_THRESHOLD and dB > DISPLAY_MIN_DB
    - show only class candidates whose score > DISPLAY_SCORE_THRESHOLD
    - otherwise show animated listening state
    """
    if parsed is None:
        return None

    items = list(parsed.get("items") or [])
    for it in items:
        it["display_label"] = normalize_label(it.get("label", ""))

    strong_items = [it for it in items if _score_is_confident(it.get("score"))]
    top_item = strong_items[0] if strong_items else (items[0] if items else None)
    top_score = float(top_item.get("score", 0.0)) if top_item else 0.0
    top_label = top_item.get("label", "") if top_item else ""

    db_value = parsed.get("db")
    low_db = _db_is_too_low(db_value)
    low_score = not _score_is_confident(top_score)
    forced_listening = (str(top_label).lower() == "listening")

    parsed["thresholds"] = {
        "score": DISPLAY_SCORE_THRESHOLD,
        "min_db": DISPLAY_MIN_DB,
    }
    parsed["raw_inference"] = parsed.get("raw", "")
    parsed["strong_items"] = strong_items
    parsed["items"] = strong_items  # Requirement: show only confident candidates; <=70% stays listening.

    if low_db or low_score or forced_listening:
        if low_db:
            reason = "low_db"
        elif low_score:
            reason = "low_score"
        else:
            reason = "listening"

        parsed["display_state"] = "listening"
        parsed["display_label"] = "소리를 듣고 있어요"
        parsed["display_icon"] = "〰️"
        parsed["display_score"] = None
        parsed["items"] = []  # Do not leak a class in the visible candidate list while listening.
        parsed["level"] = "listening"
        parsed["reason"] = reason
        safe_db = parsed.get("db")
        if safe_db is None:
            parsed["raw"] = "[%s] listening | reason=%s" % (parsed.get("time") or "--:--:--", reason)
        else:
            parsed["raw"] = "[%s] listening db=%.1fdB | reason=%s" % (parsed.get("time") or "--:--:--", float(safe_db), reason)
        parsed["display_raw"] = make_display_raw(parsed, [])
        return parsed

    parsed["display_state"] = "detected"
    parsed["display_label"] = normalize_label(top_label)
    parsed["display_icon"] = icon_for_label(top_label)
    parsed["display_score"] = top_score
    parsed["level"] = risk_level(top_label)
    parsed["reason"] = "detected"
    parsed["display_raw"] = make_display_raw(parsed, strong_items)
    return parsed


def parse_line(line):
    m = LINE_RE.search(line.strip())

    if not m:
        return None

    mid = m.group("middle")
    total_sec = None
    db_value = None

    total_match = re.search(r'total=([0-9.]+)s', mid)
    if total_match:
        total_sec = float(total_match.group(1))

    db_match = DB_RE.search(mid)
    if db_match:
        db_value = float(db_match.group(1))

    items = parse_items(m.group("items"))
    if len(items) == 0:
        return None

    top = items[0]

    parsed = {
        "status": "ok",
        "time": m.group("time"),
        "label": top["label"],
        "score": top["score"],
        "infer_sec": float(m.group("infer")),
        "total_sec": total_sec,
        "db": db_value,
        "level": risk_level(top["label"]),
        "items": items,
        "raw": line.strip(),
    }
    return apply_display_fields(parsed)


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <title>Jetson Sound Alert</title>
  <style>
    :root {
      --accent: #2fe6ff;
      --accent-soft: rgba(47, 230, 255, 0.24);
      --danger: #ff3b65;
      --caution: #ffd34d;
      --info: #28e6a5;
      --panel: rgba(35, 31, 48, 0.86);
      --panel-border: rgba(255, 255, 255, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      min-height: 100vh;
      color: #f7fbff;
      background:
        radial-gradient(circle at 50% 30%, rgba(44, 232, 255, 0.12), transparent 28%),
        radial-gradient(circle at 50% 55%, rgba(165, 56, 255, 0.16), transparent 35%),
        linear-gradient(120deg, #06090f 0%, #0b1018 56%, #07080e 100%);
      overflow-x: hidden;
    }
    .wrap {
      width: min(100vw, 980px);
      min-height: 100vh;
      margin: 0 auto;
      padding: 22px 24px 26px;
      position: relative;
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 64px 4px 96px;
      color: rgba(255, 255, 255, 0.62);
      font-size: 15px;
      font-weight: 700;
    }
    .pill {
      border: 1px solid rgba(47, 230, 255, 0.8);
      color: var(--accent);
      border-radius: 999px;
      padding: 8px 16px;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.06em;
      background: rgba(47, 230, 255, 0.06);
      box-shadow: 0 0 14px rgba(47, 230, 255, 0.12), inset 0 0 10px rgba(47, 230, 255, 0.06);
      transition: color 180ms ease, border-color 180ms ease, background 180ms ease, box-shadow 180ms ease;
    }
    .pill.listening, .pill.info {
      border-color: rgba(47, 230, 255, 0.85);
      color: var(--accent);
      background: rgba(47, 230, 255, 0.08);
      box-shadow: 0 0 16px rgba(47, 230, 255, 0.16), inset 0 0 12px rgba(47, 230, 255, 0.08);
    }
    .pill.danger {
      border-color: rgba(255, 59, 101, 0.95);
      color: var(--danger);
      background: rgba(255, 59, 101, 0.10);
      box-shadow: 0 0 16px rgba(255, 59, 101, 0.20), inset 0 0 12px rgba(255, 59, 101, 0.09);
    }
    .pill.caution {
      border-color: rgba(255, 211, 77, 0.95);
      color: var(--caution);
      background: rgba(255, 211, 77, 0.10);
      box-shadow: 0 0 16px rgba(255, 211, 77, 0.18), inset 0 0 12px rgba(255, 211, 77, 0.08);
    }
    .stage {
      position: relative;
      width: min(68vw, 520px);
      height: min(68vw, 520px);
      min-width: 340px;
      min-height: 340px;
      margin: 10px auto 18px;
    }
    .halo, .ring, .ring2, .orb, .core {
      position: absolute;
      border-radius: 50%;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }
    .halo {
      width: 100%;
      height: 100%;
      background: radial-gradient(circle, rgba(42, 227, 255, 0.12), transparent 68%);
      filter: blur(4px);
    }
    .ring {
      width: 84%;
      height: 84%;
      border: 3px solid rgba(47, 230, 255, 0.30);
      box-shadow: 0 0 34px rgba(47, 230, 255, 0.12), inset 0 0 28px rgba(47, 230, 255, 0.06);
    }
    .ring2 {
      width: 96%;
      height: 96%;
      border: 1.5px solid rgba(183, 71, 255, 0.60);
      box-shadow: 0 0 34px rgba(183, 71, 255, 0.10);
    }
    .orb {
      width: 66%;
      height: 66%;
      background:
        radial-gradient(circle at 50% 40%, rgba(58, 43, 92, 0.94), rgba(30, 21, 48, 0.96) 62%, rgba(16, 13, 25, 0.96));
      box-shadow:
        inset 0 0 42px rgba(148, 54, 255, 0.26),
        0 20px 72px rgba(0, 0, 0, 0.48);
    }
    .direction-arrow {
      position: absolute;
      left: 50%;
      top: 50%;
      z-index: 5;
      width: 118px;
      height: 118px;
      line-height: 118px;
      text-align: center;
      font-size: 88px;
      font-weight: 900;
      color: var(--accent);
      text-shadow: 0 0 16px rgba(47, 230, 255, 0.95), 0 0 36px rgba(47, 230, 255, 0.48);
      transform: translate(-50%, -50%) rotate(0deg) translateY(-186px);
      transform-origin: center center;
      transition: transform 260ms ease, opacity 180ms ease, color 180ms ease;
      opacity: 0.95;
      pointer-events: none;
    }
    .direction-arrow.hidden { opacity: 0; }
    .direction-arrow.danger { color: var(--danger); text-shadow: 0 0 20px rgba(255, 59, 101, 0.90); }
    .direction-arrow.caution { color: var(--caution); text-shadow: 0 0 18px rgba(255, 211, 77, 0.88); }
    .center {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      width: 66%;
      text-align: center;
      z-index: 6;
    }
    .icon {
      font-size: 52px;
      margin-bottom: 10px;
      filter: drop-shadow(0 0 18px rgba(47, 230, 255, 0.24));
    }
    .title {
      font-size: clamp(42px, 7vw, 68px);
      line-height: 1.02;
      font-weight: 900;
      letter-spacing: -0.05em;
      color: var(--accent);
      text-shadow: 0 0 20px rgba(47, 230, 255, 0.32);
      word-break: keep-all;
    }
    .sub {
      margin-top: 10px;
      font-size: 19px;
      font-weight: 800;
      color: rgba(255, 255, 255, 0.90);
    }
    .score {
      margin-top: 8px;
      font-size: 15px;
      color: rgba(255, 255, 255, 0.58);
      font-weight: 700;
    }
    .listen-bars {
      display: none;
      align-items: flex-end;
      justify-content: center;
      height: 64px;
      gap: 7px;
      margin: 14px 0 4px;
    }
    .listen-bars span {
      display: block;
      width: 9px;
      height: 18px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 18px rgba(47, 230, 255, 0.55);
      animation: barPulse 0.82s ease-in-out infinite;
    }
    .listen-bars span:nth-child(2) { animation-delay: 0.10s; }
    .listen-bars span:nth-child(3) { animation-delay: 0.20s; }
    .listen-bars span:nth-child(4) { animation-delay: 0.30s; }
    .listen-bars span:nth-child(5) { animation-delay: 0.40s; }
    body.listening .listen-bars { display: flex; }
    body.listening .orb::after {
      content: "";
      position: absolute;
      inset: -10px;
      border-radius: 50%;
      border: 1px solid rgba(47, 230, 255, 0.28);
      animation: listenRing 1.8s ease-out infinite;
    }
    body.listening .title {
      font-size: clamp(32px, 5.3vw, 48px);
      letter-spacing: -0.04em;
    }
    @keyframes barPulse {
      0%, 100% { height: 16px; opacity: 0.58; }
      50% { height: 58px; opacity: 1; }
    }
    @keyframes listenRing {
      0% { transform: scale(0.96); opacity: 0.9; }
      100% { transform: scale(1.12); opacity: 0; }
    }
    .panel {
      width: min(92vw, 600px);
      margin: 0 auto;
      padding: 16px;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      box-shadow: 0 18px 58px rgba(0,0,0,0.42), inset 0 0 26px rgba(255,255,255,0.03);
      backdrop-filter: blur(10px);
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-bottom: 12px;
    }
    .meta-card {
      padding: 11px 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .meta-card .k {
      display: block;
      font-size: 11px;
      color: rgba(255, 255, 255, 0.48);
      font-weight: 800;
      letter-spacing: 0.04em;
    }
    .meta-card .v {
      margin-top: 3px;
      display: block;
      font-size: 16px;
      color: #fff;
      font-weight: 900;
    }
    .items {
      padding: 12px;
      border-radius: 16px;
      background: rgba(0, 0, 0, 0.24);
      min-height: 54px;
    }
    .item {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.07);
      font-weight: 800;
    }
    .item:last-child { border-bottom: 0; }
    .empty {
      padding: 7px 2px;
      color: rgba(255,255,255,0.56);
      font-size: 14px;
      font-weight: 700;
    }
    .raw {
      margin-top: 11px;
      font-size: 12px;
      color: rgba(255,255,255,0.42);
      word-break: break-word;
    }
    .dangerText { color: var(--danger); }
    .cautionText { color: var(--caution); }
    .infoText { color: var(--accent); }
    @media (max-width: 640px) {
      .wrap { padding: 18px 14px 24px; }
      .topbar { padding: 0 8px; }
      .stage { min-width: 320px; min-height: 320px; width: 92vw; height: 92vw; max-width: 480px; max-height: 480px; }
      .direction-arrow { font-size: 76px; transform: translate(-50%, -50%) rotate(0deg) translateY(-160px); }
      .meta-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .meta-card { padding: 9px 8px; }
    }
  </style>
</head>
<body class="listening">
  <div class="wrap">
    <div class="topbar">
      <div>Jetson Nano Sound Recognition</div>
      <div class="pill" id="statePill">LISTENING</div>
    </div>

    <div class="stage">
      <div class="halo"></div>
      <div class="ring2"></div>
      <div class="ring"></div>
      <div class="orb"></div>
      <div id="directionArrow" class="direction-arrow hidden">⬆</div>
      <div class="center">
        <div id="icon" class="icon">〰️</div>
        <div class="listen-bars" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>
        <div id="label" class="title">소리를 듣고 있어요</div>
        <div id="direction" class="sub">방향 확인 중</div>
        <div id="score" class="score">70% 이상 확신되는 소리만 표시</div>
      </div>
    </div>

    <div class="panel">
      <div class="meta-grid">
        <div class="meta-card"><span class="k">TIME</span><span class="v" id="time">-</span></div>
        <div class="meta-card"><span class="k">INFER</span><span class="v" id="infer">-</span></div>
        <div class="meta-card"><span class="k">DB</span><span class="v" id="db">-</span></div>
      </div>
      <div class="items" id="items"><div class="empty">70% 초과로 확신되는 소리를 기다리는 중</div></div>
      <div class="raw" id="raw"></div>
    </div>
  </div>

  <script>
    const evt = new EventSource("/events");
    const ARROW_RADIUS_DESKTOP = 186;
    const ARROW_RADIUS_MOBILE = 160;

    function fmtPercent(x) {
      if (x === null || x === undefined || Number.isNaN(Number(x))) return "-";
      return (Number(x) * 100).toFixed(1) + "%";
    }
    function fmtSec(x) {
      if (x === null || x === undefined || Number.isNaN(Number(x))) return "-";
      return Number(x).toFixed(3) + "s";
    }
    function fmtDb(x) {
      if (x === null || x === undefined || Number.isNaN(Number(x))) return "-";
      return Number(x).toFixed(1) + "dB";
    }
    function arrowRadius() {
      return window.matchMedia("(max-width: 640px)").matches ? ARROW_RADIUS_MOBILE : ARROW_RADIUS_DESKTOP;
    }
    function setArrow(d, level) {
      const arrow = document.getElementById("directionArrow");
      if (d.angle === null || d.angle === undefined || Number.isNaN(Number(d.angle))) {
        arrow.className = "direction-arrow hidden";
        return;
      }
      const angle = Number(d.angle);
      arrow.className = "direction-arrow " + (level || "");
      arrow.style.transform = "translate(-50%, -50%) rotate(" + angle + "deg) translateY(-" + arrowRadius() + "px)";
    }
    function setItems(list) {
      const items = document.getElementById("items");
      items.innerHTML = "";
      if (!list || list.length === 0) {
        const div = document.createElement("div");
        div.className = "empty";
        div.textContent = "70% 초과로 확신되는 소리를 기다리는 중";
        items.appendChild(div);
        return;
      }
      list.forEach(function(it, idx) {
        const div = document.createElement("div");
        div.className = "item";
        const label = document.createElement("span");
        const score = document.createElement("span");
        label.textContent = (idx + 1) + ". " + (it.display_label || it.label || "-");
        score.textContent = fmtPercent(it.score);
        div.appendChild(label);
        div.appendChild(score);
        items.appendChild(div);
      });
    }

    evt.onmessage = function(e) {
      const d = JSON.parse(e.data);
      const state = d.display_state || "listening";
      const level = d.level || "listening";
      document.body.className = state === "listening" ? "listening" : "detected";

      const pill = document.getElementById("statePill");
      const pillLevel = state === "listening" ? "listening" : (["danger", "caution", "info"].indexOf(level) >= 0 ? level : "info");
      pill.className = "pill " + pillLevel;
      pill.textContent = state === "listening" ? "LISTENING" : pillLevel.toUpperCase();

      document.getElementById("icon").textContent = d.display_icon || (state === "listening" ? "〰️" : "🔊");
      const label = document.getElementById("label");
      label.textContent = d.display_label || (state === "listening" ? "소리를 듣고 있어요" : (d.label || "-"));
      label.className = "title " + (level === "danger" ? "dangerText" : (level === "caution" ? "cautionText" : "infoText"));

      document.getElementById("direction").textContent = d.direction_text || (d.angle !== null && d.angle !== undefined ? Math.round(Number(d.angle)) + "°" : "방향 확인 중");
      document.getElementById("score").textContent = state === "listening"
        ? "70% 초과 · 45dB 초과일 때만 분류 표시"
        : "score: " + fmtPercent(d.display_score);

      document.getElementById("time").textContent = d.time || "-";
      document.getElementById("infer").textContent = fmtSec(d.infer_sec);
      document.getElementById("db").textContent = fmtDb(d.db);
      setItems(d.items || d.strong_items || []);
      setArrow(d, level);
      document.getElementById("raw").textContent = d.display_raw || d.raw || "";
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
        log_f = open(args.log, "a", encoding="utf-8")

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
