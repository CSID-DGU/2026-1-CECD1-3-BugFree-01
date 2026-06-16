#!/usr/bin/env python3
import argparse
import csv
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except Exception as e:
        raise SystemExit(
            "ERROR: tflite_runtime 또는 tensorflow.lite Interpreter를 찾을 수 없습니다.\n"
            "먼저 다음 중 하나를 설치하세요:\n"
            "  python3 -m pip install tflite-runtime\n"
            "  python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime"
        ) from e


def load_labels(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    labels = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "display_name" in reader.fieldnames:
            for row in reader:
                labels.append(row["display_name"])
        else:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line.split(",")[-1])
    return labels


def read_wav_channel(path, channel_index):
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if rate != 16000:
        raise ValueError(f"YAMNet 입력은 16 kHz가 필요합니다. 현재 WAV rate={rate}")

    if channel_index < 0 or channel_index >= channels:
        raise ValueError(f"channel-index={channel_index}가 잘못되었습니다. WAV channels={channels}")

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"지원하지 않는 sample width입니다: {sampwidth}")

    audio = audio.reshape(-1, channels)[:, channel_index]
    audio = np.clip(audio, -1.0, 1.0)
    return np.ascontiguousarray(audio, dtype=np.float32)


def record_chunk(device, channels, seconds, out_wav):
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", str(channels),
        "-d", str(seconds),
        str(out_wav),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode != 0:
        raise RuntimeError("arecord 실패:\n" + p.stderr)


def create_interpreter(model_path, threads, window_samples):
    try:
        interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
    except TypeError:
        interpreter = Interpreter(model_path=str(model_path))

    input_details = interpreter.get_input_details()
    inp = input_details[0]
    shape = list(inp["shape"])
    sig = list(inp.get("shape_signature", shape))

    # 일부 TFLite 모델은 입력 길이가 dynamic일 수 있으므로 필요한 경우 resize.
    if any(int(d) < 0 for d in sig) or any(int(d) == 0 for d in shape):
        if len(shape) == 1:
            new_shape = [window_samples]
        elif len(shape) == 2:
            new_shape = [1, window_samples]
        else:
            raise ValueError(f"지원하지 않는 dynamic input shape입니다: {shape}")
        interpreter.resize_tensor_input(inp["index"], new_shape, strict=False)

    interpreter.allocate_tensors()
    return interpreter


def get_input_sample_len(input_shape):
    shape = [int(x) for x in input_shape if int(x) > 0]

    if len(shape) == 1:
        return shape[0]
    if len(shape) == 2 and shape[0] == 1:
        return shape[1]

    # batch dimension이 1인 일반적인 경우를 최대한 처리
    if shape and shape[0] == 1:
        return int(np.prod(shape[1:]))
    return int(np.prod(shape))


def make_input_tensor(clip, input_detail):
    shape = tuple(int(x) for x in input_detail["shape"])
    x = clip.astype(np.float32).reshape(shape)

    dtype = input_detail["dtype"]
    if dtype != np.float32:
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            x = x / scale + zero_point
        x = x.astype(dtype)

    return x


def run_yamnet(interpreter, waveform, topk=5):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = input_details[0]
    input_len = get_input_sample_len(inp["shape"])

    if waveform.shape[0] < input_len:
        waveform = np.pad(waveform, (0, input_len - waveform.shape[0]))

    hop = max(1, input_len // 2)

    starts = list(range(0, max(1, waveform.shape[0] - input_len + 1), hop))
    if starts[-1] + input_len < waveform.shape[0]:
        starts.append(waveform.shape[0] - input_len)

    all_scores = []
    t0 = time.time()

    for s in starts:
        clip = waveform[s:s + input_len]
        if clip.shape[0] < input_len:
            clip = np.pad(clip, (0, input_len - clip.shape[0]))

        tensor = make_input_tensor(clip, inp)
        interpreter.set_tensor(inp["index"], tensor)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[0]["index"])
        scores = np.asarray(scores).squeeze().astype(np.float32)

        # shape가 (1, 521) 또는 (521,) 모두 처리
        if scores.ndim > 1:
            scores = scores.reshape(-1, scores.shape[-1]).mean(axis=0)

        all_scores.append(scores)

    infer_ms = (time.time() - t0) * 1000.0
    mean_scores = np.mean(np.stack(all_scores, axis=0), axis=0)

    top_idx = np.argsort(mean_scores)[::-1][:topk]
    return mean_scores, top_idx, infer_ms, len(starts), input_len


def print_result(scores, top_idx, labels, infer_ms, windows, input_len, threshold):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] infer_ms={infer_ms:.1f} windows={windows} input_samples={input_len}")

    printed = 0
    for idx in top_idx:
        score = float(scores[idx])
        if score < threshold:
            continue
        label = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
        print(f"  {score:.3f}  {label}")
        printed += 1

    if printed == 0:
        print(f"  no class over threshold={threshold}")

    print("", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yamnet.tflite")
    parser.add_argument("--labels", default="yamnet_class_map.csv")

    parser.add_argument("--wav", default=None, help="이미 녹음된 WAV 파일을 테스트할 때 사용")
    parser.add_argument("--device", default="plughw:2,0")
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=2)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--window-samples", type=int, default=15600)
    parser.add_argument("--once", action="store_true", help="마이크 입력을 1회만 테스트")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"모델 파일이 없습니다: {model_path}")

    labels = load_labels(args.labels)

    interpreter = create_interpreter(model_path, args.threads, args.window_samples)

    if args.wav:
        waveform = read_wav_channel(args.wav, args.channel_index)
        scores, top_idx, infer_ms, windows, input_len = run_yamnet(
            interpreter, waveform, topk=args.topk
        )
        print_result(scores, top_idx, labels, infer_ms, windows, input_len, args.threshold)
        return

    while True:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            record_chunk(args.device, args.channels, args.seconds, tmp.name)
            waveform = read_wav_channel(tmp.name, args.channel_index)

        scores, top_idx, infer_ms, windows, input_len = run_yamnet(
            interpreter, waveform, topk=args.topk
        )
        print_result(scores, top_idx, labels, infer_ms, windows, input_len, args.threshold)

        if args.once:
            break


if __name__ == "__main__":
    main()
