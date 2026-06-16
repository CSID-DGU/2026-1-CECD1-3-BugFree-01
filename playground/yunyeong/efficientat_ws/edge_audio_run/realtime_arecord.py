import argparse
import json
import subprocess
import time

import numpy as np

from mel import load_config, waveform_to_mel_tensor
from trt_infer import TRTClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="default", help="ALSA device, e.g. plughw:2,0")
    parser.add_argument("--engine", default="deploy/efficientat_sound_fp16.engine")
    parser.add_argument("--labels", default="deploy/labels.json")
    parser.add_argument("--risk_map", default="deploy/risk_map.json")
    parser.add_argument("--config", default="deploy/preprocess_config.json")
    parser.add_argument("--mel_basis", default="deploy/mel_basis.npy")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--hop_seconds", type=float, default=0.5)
    parser.add_argument("--print_all", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mel_basis = np.load(args.mel_basis)
    clf = TRTClassifier(args.engine, args.labels, args.risk_map)

    sr = int(cfg["sample_rate"])
    clip_samples = int(cfg["clip_samples"])
    hop_samples = int(sr * args.hop_seconds)
    audio_buffer = np.zeros(clip_samples, dtype=np.float32)

    cmd = [
        "arecord",
        "-D", args.device,
        "-f", "S16_LE",
        "-r", str(sr),
        "-c", "1",
        "-t", "raw",
        "-q",
    ]
    print("Starting arecord:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    bytes_per_hop = hop_samples * 2
    last_label = None
    try:
        while True:
            raw = proc.stdout.read(bytes_per_hop)
            if len(raw) < bytes_per_hop:
                print("arecord stream ended or insufficient bytes", flush=True)
                break

            chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            mel = waveform_to_mel_tensor(audio_buffer, cfg, mel_basis)
            t0 = time.time()
            result = clf.predict(mel)
            elapsed_ms = (time.time() - t0) * 1000.0
            label = result["label"]
            conf = result["confidence"]

            event = {
                "timestamp": round(time.time(), 3),
                "label": label,
                "confidence": round(float(conf), 4),
                "risk_level": result["risk_level"],
                "elapsed_ms": round(elapsed_ms, 3),
            }

            if args.print_all:
                print(json.dumps(event, ensure_ascii=False), flush=True)
            else:
                if label != "background" and conf >= args.threshold:
                    print(json.dumps(event, ensure_ascii=False), flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
