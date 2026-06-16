#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import re
import subprocess
import sys
import io

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

LINE_RE = re.compile(r'^(?P<head>\[[0-9:]+\]\s+infer=[0-9.]+s.*?\|\s+)(?P<items>.*)$')


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


class DOAReader(object):
    def __init__(self):
        self.ok = False
        self.tuning = None

        try:
            import usb.core
            from tuning import Tuning

            dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
            if dev is None:
                print("[DOA] ReSpeaker USB control device not found.", file=sys.stderr)
                return

            self.tuning = Tuning(dev)
            self.ok = True
            print("[DOA] ReSpeaker DOA reader enabled.", file=sys.stderr)

        except Exception as e:
            print("[DOA] disabled:", repr(e), file=sys.stderr)

    def read_angle(self):
        if not self.ok or self.tuning is None:
            return None

        try:
            angle = self.tuning.direction
            if angle is None:
                return None
            return int(float(angle)) % 360
        except Exception:
            return None


def add_direction_to_line(line, angle, direction):
    m = LINE_RE.match(line.rstrip("\n"))
    if not m:
        return line

    items_text = m.group("items")
    parts = items_text.split(" / ")
    if not parts:
        return line

    first = parts[0].strip()

    try:
        label, score_text = first.rsplit(" ", 1)
        float(score_text)
    except Exception:
        return line

    parts[0] = "%s [%s %d°] %s" % (label, direction, angle, score_text)

    return m.group("head") + " / ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--north-offset",
        type=float,
        default=0.0,
        help="DOA angle that should be treated as North. Default: 0",
    )

    args, live_args = parser.parse_known_args()

    doa = DOAReader()

    cmd = ["python3", "-u", "jetson_live.py"] + live_args
    print("[DOA] command:", " ".join(cmd), file=sys.stderr)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    try:
        for line in proc.stdout:
            angle = doa.read_angle()

            if angle is not None:
                direction = angle_to_cardinal(angle, args.north_offset)
                line = add_direction_to_line(line, angle, direction)

            print(line, end="", flush=True)

    except KeyboardInterrupt:
        pass

    finally:
        try:
            proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
