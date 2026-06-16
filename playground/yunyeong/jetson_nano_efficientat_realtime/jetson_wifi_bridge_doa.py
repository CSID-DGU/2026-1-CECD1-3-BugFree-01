#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import io
import json
import os
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

import jetson_wifi_bridge as bridge


DOA_ENABLE = os.environ.get("DOA_ENABLE", "1") != "0"
DOA_NORTH_OFFSET = float(os.environ.get("DOA_NORTH_OFFSET", "0"))

_doa_lock = threading.Lock()
_doa_started = False
_doa_angle = None
_doa_direction = None
_doa_status = "not_started"


def angle_to_cardinal(angle, north_offset=0.0):
    corrected = (float(angle) - float(north_offset)) % 360.0

    if corrected < 45 or corrected >= 315:
        return "북"
    elif corrected < 135:
        return "동"
    elif corrected < 225:
        return "남"
    else:
        return "서"


class InlineTuning(object):
    """
    tuning.py가 없을 때를 위한 최소 DOA reader.
    ReSpeaker USB 4 Mic Array의 DOAANGLE만 읽는다.
    """

    TIMEOUT = 100000

    def __init__(self, dev, usb_util):
        self.dev = dev
        self.usb_util = usb_util

    @property
    def direction(self):
        # tuning.py 기준:
        # DOAANGLE = id 21, offset 0, type int
        # read cmd = 0x80 | offset | 0x40 = 0xC0
        response = self.dev.ctrl_transfer(
            self.usb_util.CTRL_IN |
            self.usb_util.CTRL_TYPE_VENDOR |
            self.usb_util.CTRL_RECIPIENT_DEVICE,
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


def create_tuning(dev):
    try:
        from tuning import Tuning
        return Tuning(dev)
    except Exception:
        import usb.util
        return InlineTuning(dev, usb.util)


def set_doa_status(status):
    global _doa_status
    with _doa_lock:
        _doa_status = status


def doa_loop():
    global _doa_angle, _doa_direction

    try:
        import usb.core

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
                direction = angle_to_cardinal(angle, DOA_NORTH_OFFSET)

                with _doa_lock:
                    _doa_angle = angle
                    _doa_direction = direction

            except Exception as e:
                set_doa_status("read_error")
                print("[DOA] read error:", repr(e), file=sys.stderr)

            time.sleep(0.2)

    except Exception as e:
        set_doa_status("disabled")
        print("[DOA] disabled:", repr(e), file=sys.stderr)


def ensure_doa_thread():
    global _doa_started

    if not DOA_ENABLE:
        set_doa_status("disabled_by_env")
        return

    if _doa_started:
        return

    _doa_started = True

    t = threading.Thread(target=doa_loop)
    t.daemon = True
    t.start()


_orig_parse_line = bridge.parse_line


def parse_line_with_doa(line):
    parsed = _orig_parse_line(line)

    if parsed is None:
        return None

    ensure_doa_thread()

    # 첫 alert에서 방향이 비어 보이는 문제를 줄이기 위해 잠깐 기다림.
    direction = None

    deadline = time.time() + 0.5
    while time.time() < deadline:
        with _doa_lock:
            direction = _doa_direction

        if direction is not None:
            break

        time.sleep(0.05)

    if direction is None:
        with _doa_lock:
            status = _doa_status

        parsed["doa_status"] = status
        parsed["raw"] = "%s | DOA unavailable:%s" % (parsed.get("raw", line.strip()), status)
        return parsed

    # 핵심 변경점:
    # 기존: "동 123°"
    # 변경: "동"
    tag = direction

    parsed["direction"] = direction

    if "angle" in parsed:
        parsed.pop("angle", None)

    parsed["label"] = "%s [%s]" % (parsed["label"], tag)
    parsed["raw"] = "%s | DOA %s" % (parsed.get("raw", line.strip()), tag)

    if parsed.get("items"):
        parsed["items"][0]["label"] = "%s [%s]" % (parsed["items"][0]["label"], tag)

    return parsed


bridge.parse_line = parse_line_with_doa


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

    ensure_doa_thread()

    log_f = None
    if args.log is not None:
        log_dir = os.path.dirname(args.log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.log, "a", encoding="utf-8")

    server_thread = threading.Thread(target=bridge.run_server, args=(args.port,))
    server_thread.daemon = True
    server_thread.start()

    print("Starting inference command:")
    print(" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            parsed = bridge.parse_line(line)

            if parsed is not None:
                out_line = parsed.get("raw", line.strip())
            else:
                out_line = line.rstrip("\n")

            print(out_line)

            if log_f is not None:
                log_f.write(out_line + "\n")
                log_f.flush()

            if parsed is not None:
                with bridge.latest_lock:
                    bridge.latest_result.clear()
                    bridge.latest_result.update(parsed)

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
