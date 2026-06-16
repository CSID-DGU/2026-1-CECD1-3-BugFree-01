from __future__ import annotations

import io
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import torch

from constants import FMAX, HOP_SIZE, MODEL_NAME, MODEL_SAMPLE_RATE, N_FFT, N_MELS, WINDOW_SIZE


class EfficientATLoader:
    def __init__(self, efficientat_dir: Path, device: torch.device) -> None:
        self.efficientat_dir = efficientat_dir
        self.device = device

    def load(self):
        if not self.efficientat_dir.exists():
            raise RuntimeError(
                f"EfficientAT 저장소가 없습니다: {self.efficientat_dir}\n"
                "먼저 `bash install.sh`를 실행하세요."
            )

        repo_dir = self.efficientat_dir.resolve()
        sys.path.insert(0, str(repo_dir))

        old_cwd = os.getcwd()
        try:
            os.chdir(str(repo_dir))
            from helpers.utils import NAME_TO_WIDTH, labels  # type: ignore
            from models.mn.model import get_model as get_mn  # type: ignore
            from models.preprocess import AugmentMelSTFT  # type: ignore

            with redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Don't use ConvNormActivation directly.*",
                    category=UserWarning,
                    module="torchvision\\.ops\\.misc",
                )
                model = get_mn(
                    width_mult=NAME_TO_WIDTH(MODEL_NAME),
                    pretrained_name=MODEL_NAME,
                    strides=(2, 2, 2, 2),
                    head_type="mlp",
                )

            mel = AugmentMelSTFT(
                n_mels=N_MELS,
                sr=MODEL_SAMPLE_RATE,
                win_length=WINDOW_SIZE,
                hopsize=HOP_SIZE,
                n_fft=N_FFT,
                fmax=FMAX,
                freqm=0,
                timem=0,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torchvision":
                raise RuntimeError(
                    "EfficientAT 모델 로딩에 torchvision이 필요합니다. "
                    "`pip install torchvision` 후 다시 실행하세요."
                ) from exc
            raise
        finally:
            os.chdir(old_cwd)

        model.to(self.device).eval()
        mel.to(self.device).eval()
        return model, mel, list(labels)
