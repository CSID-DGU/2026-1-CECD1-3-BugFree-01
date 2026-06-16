import argparse
import json
import time
from pathlib import Path

import numpy as np

from mel import load_config, read_wav_mono, waveform_to_mel_tensor
from trt_infer import TRTClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="input wav file")
    parser.add_argument("--engine", default="deploy/efficientat_sound_fp16.engine")
    parser.add_argument("--labels", default="deploy/labels.json")
    parser.add_argument("--risk_map", default="deploy/risk_map.json")
    parser.add_argument("--config", default="deploy/preprocess_config.json")
    parser.add_argument("--mel_basis", default="deploy/mel_basis.npy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mel_basis = np.load(args.mel_basis)
    clf = TRTClassifier(args.engine, args.labels, args.risk_map)

    wav = read_wav_mono(args.wav, int(cfg["sample_rate"]))
    mel = waveform_to_mel_tensor(wav, cfg, mel_basis)

    t0 = time.time()
    result = clf.predict(mel)
    elapsed_ms = (time.time() - t0) * 1000.0
    result.pop("probs", None)
    result["elapsed_ms"] = round(elapsed_ms, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
