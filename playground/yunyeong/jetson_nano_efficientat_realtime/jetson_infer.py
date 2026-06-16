import argparse
import time
import wave
import numpy as np
import torch

from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels


class nullcontext(object):
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def load_pcm_wav_32k_mono(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sr != 32000:
        raise RuntimeError(
            "Input WAV sample rate must be 32000 Hz. "
            "Convert first: ffmpeg -y -i input.wav -ac 1 -ar 32000 -sample_fmt s16 output_32k.wav"
        )

    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("Only 16-bit or 32-bit PCM WAV is supported by this Jetson script.")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--model_name", default="mn10_as")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("device:", device)
    print("model:", args.model_name)

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
        sr=32000,
        win_length=800,
        hopsize=320
    )
    mel.to(device)
    mel.eval()

    audio = load_pcm_wav_32k_mono(args.audio_path)
    waveform = torch.from_numpy(audio[None, :]).to(device)

    amp_context = torch.cuda.amp.autocast() if use_cuda and hasattr(torch.cuda, "amp") else nullcontext()

    times = []
    last_preds = None

    for i in range(args.warmup + args.runs):
        if use_cuda:
            torch.cuda.synchronize()

        t0 = time.time()

        with torch.no_grad(), amp_context:
            spec = mel(waveform)
            preds, _ = model(spec.unsqueeze(0))
            preds = torch.sigmoid(preds.float()).squeeze().detach().cpu().numpy()

        if use_cuda:
            torch.cuda.synchronize()

        t1 = time.time()

        if i >= args.warmup:
            times.append(t1 - t0)
            last_preds = preds

    print("avg inference time: %.4f sec" % (sum(times) / len(times)))

    sorted_indexes = np.argsort(last_preds)[::-1]

    print("************* Acoustic Event Detected *****************")
    for k in range(min(args.topk, len(sorted_indexes))):
        idx = sorted_indexes[k]
        print("%02d. %s: %.3f" % (k + 1, labels[idx], last_preds[idx]))
    print("********************************************************")


if __name__ == "__main__":
    main()
