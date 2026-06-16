import argparse
import subprocess
import time
import wave
import numpy as np
import torch

from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels


TARGET_SR = 32000


class nullcontext(object):
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def amp_context(use_cuda):
    if use_cuda and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast()
    return nullcontext()


def read_wav_select_channel(path, channel_index):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth != 2:
        raise RuntimeError("Only S16_LE WAV is supported. Record with -f S16_LE.")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels)

        if channel_index == -1:
            audio = audio.mean(axis=1)
        else:
            if channel_index >= channels:
                raise RuntimeError("channel_index is out of range.")
            audio = audio[:, channel_index]

    return audio.astype(np.float32), sr, channels


def resample_linear(audio, src_sr, dst_sr):
    if src_sr == dst_sr:
        return audio.astype(np.float32)

    if len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / float(src_sr)
    old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_len = int(round(duration * dst_sr))
    new_x = np.linspace(0.0, duration, num=new_len, endpoint=False)

    return np.interp(new_x, old_x, audio).astype(np.float32)


def record_chunk(args, wav_path):
    cmd = [
        "arecord",
        "-q",
        "-D", args.device,
        "-f", "S16_LE",
        "-r", str(args.rate),
        "-c", str(args.channels),
        "-d", str(args.seconds),
        wav_path,
    ]
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, help="Example: plughw:2,0")
    parser.add_argument("--model_name", default="mn04_as")
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--channel-index", type=int, default=0,
                        help="0 = first channel, -1 = average all channels")
    parser.add_argument("--seconds", type=int, default=2)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("device:", device)
    print("model:", args.model_name)
    print("audio device:", args.device)
    print("record:", str(args.rate) + "Hz,", str(args.channels) + "ch,", str(args.seconds) + "s")
    print("channel-index:", args.channel_index)
    print("Ctrl+C to stop")
    print("-" * 60)

    torch.backends.cudnn.benchmark = True

    model = get_mobilenet(
        width_mult=NAME_TO_WIDTH(args.model_name),
        pretrained_name=args.model_name,
        strides=[2, 2, 2, 2],
        head_type="mlp"
    )
    model.to(device)
    model.eval()

    mel = AugmentMelSTFT(
        n_mels=128,
        sr=TARGET_SR,
        win_length=800,
        hopsize=320
    )
    mel.to(device)
    mel.eval()

    # Warm-up
    dummy = torch.zeros(1, TARGET_SR * max(1, args.seconds)).to(device)
    with torch.no_grad(), amp_context(use_cuda):
        spec = mel(dummy)
        _ = model(spec.unsqueeze(0))
    if use_cuda:
        torch.cuda.synchronize()

    wav_path = "/tmp/respeaker_live_chunk.wav"

    while True:
        loop_start = time.time()

        rc = record_chunk(args, wav_path)
        if rc != 0:
            print("arecord failed. Check --device, --rate, --channels.")
            time.sleep(1)
            continue

        audio, src_sr, detected_channels = read_wav_select_channel(
            wav_path,
            args.channel_index
        )

        audio = resample_linear(audio, src_sr, TARGET_SR)

        if len(audio) < TARGET_SR // 4:
            print("audio too short, skipped")
            continue

        waveform = torch.from_numpy(audio[None, :]).to(device)

        if use_cuda:
            torch.cuda.synchronize()

        infer_start = time.time()

        with torch.no_grad(), amp_context(use_cuda):
            spec = mel(waveform)
            preds, _ = model(spec.unsqueeze(0))
            scores = torch.sigmoid(preds.float()).squeeze().detach().cpu().numpy()

        if use_cuda:
            torch.cuda.synchronize()

        infer_time = time.time() - infer_start
        total_time = time.time() - loop_start

        sorted_indexes = np.argsort(scores)[::-1]
        shown = []

        for idx in sorted_indexes[:args.topk]:
            score = float(scores[idx])
            if score >= args.threshold:
                shown.append("%s %.3f" % (labels[idx], score))

        if len(shown) == 0:
            idx = sorted_indexes[0]
            shown.append("%s %.3f" % (labels[idx], float(scores[idx])))

        print(
            "[%s] infer=%.3fs total=%.3fs | %s" %
            (time.strftime("%H:%M:%S"), infer_time, total_time, " / ".join(shown)),
            flush=True
        )


if __name__ == "__main__":
    main()
