#!/usr/bin/env python3
import argparse
import tempfile
import time
from pathlib import Path

from jetson_yamnet_tflite import (
    load_labels,
    read_wav_channel,
    record_chunk,
    create_interpreter,
    run_yamnet,
)


def clean_label(label):
    # 기존 Wi-Fi bridge는 " / " 기준으로 top-k를 나누므로 label 내부 slash를 정리
    return str(label).replace("/", "-").replace("\n", " ").strip()


def build_items(scores, top_idx, labels, threshold):
    items = []

    for idx in top_idx:
        score = float(scores[idx])
        if score < threshold:
            continue

        if labels and idx < len(labels):
            label = labels[idx]
        else:
            label = "class_%d" % int(idx)

        items.append((clean_label(label), score))

    if not items:
        items.append(("No detection", 0.0))

    return items


def print_bridge_line(scores, top_idx, labels, infer_ms, windows, input_len, threshold, total_sec):
    """
    기존 jetson_wifi_bridge.py가 읽을 수 있는 EfficientAT식 한 줄 형식으로 출력.

    예:
    [14:22:01] infer=0.183s total=2.214s windows=3 input_samples=15600 | Speech 0.420 / Vehicle 0.153
    """
    items = build_items(scores, top_idx, labels, threshold)

    item_text = " / ".join(
        "%s %.3f" % (label, score)
        for label, score in items
    )

    ts = time.strftime("%H:%M:%S")
    infer_sec = infer_ms / 1000.0

    print(
        "[%s] infer=%.3fs total=%.3fs windows=%d input_samples=%d | %s"
        % (ts, infer_sec, total_sec, windows, input_len, item_text),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="yamnet.tflite")
    parser.add_argument("--labels", default="yamnet_class_map.csv")

    parser.add_argument("--wav", default=None)
    parser.add_argument("--device", default="plughw:2,0")
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=2)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--window-samples", type=int, default=15600)
    parser.add_argument("--once", action="store_true")

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit("모델 파일이 없습니다: %s" % model_path)

    labels = load_labels(args.labels)
    interpreter = create_interpreter(model_path, args.threads, args.window_samples)

    if args.wav:
        t0 = time.time()

        waveform = read_wav_channel(args.wav, args.channel_index)
        scores, top_idx, infer_ms, windows, input_len = run_yamnet(
            interpreter,
            waveform,
            topk=args.topk,
        )

        total_sec = time.time() - t0

        print_bridge_line(
            scores,
            top_idx,
            labels,
            infer_ms,
            windows,
            input_len,
            args.threshold,
            total_sec,
        )
        return

    while True:
        t0 = time.time()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            record_chunk(args.device, args.channels, args.seconds, tmp.name)
            waveform = read_wav_channel(tmp.name, args.channel_index)

        scores, top_idx, infer_ms, windows, input_len = run_yamnet(
            interpreter,
            waveform,
            topk=args.topk,
        )

        total_sec = time.time() - t0

        print_bridge_line(
            scores,
            top_idx,
            labels,
            infer_ms,
            windows,
            input_len,
            args.threshold,
            total_sec,
        )

        if args.once:
            break


if __name__ == "__main__":
    main()
