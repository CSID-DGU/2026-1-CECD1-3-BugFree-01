from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

from datasets.helpers.audiodatasets import PreprocessDataset, get_roll_func


dataset_dir = Path(__file__).resolve().parents[2] / "datasets" / "soundkey_balanced_v2"
workspace_dir = Path(__file__).resolve().parents[2]


def pad_or_truncate(x, audio_length):
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x), dtype=np.float32)), axis=0)
    return x[:audio_length]


def gain_augment_waveform(waveform, gain_augment=0):
    if gain_augment:
        gain = torch.randint(gain_augment * 2, (1,)).item() - gain_augment
        waveform = waveform * (10 ** (gain / 20))
    return waveform


def add_white_noise(waveform, probability=0.0, snr_min_db=5.0, snr_max_db=25.0):
    if probability <= 0 or np.random.random() >= probability:
        return waveform
    signal_power = float(np.mean(np.square(waveform)) + 1e-12)
    snr_db = np.random.uniform(snr_min_db, snr_max_db)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return waveform + noise


def mix_background_noise(waveform, noise_waveform, snr_min_db=5.0, snr_max_db=20.0):
    signal_power = float(np.mean(np.square(waveform)) + 1e-12)
    noise_power = float(np.mean(np.square(noise_waveform)) + 1e-12)
    snr_db = np.random.uniform(snr_min_db, snr_max_db)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return waveform + noise_waveform * scale


class MixupDataset(TorchDataset):
    def __init__(self, dataset, beta=2, rate=0.5):
        self.dataset = dataset
        self.beta = beta
        self.rate = rate

    def __getitem__(self, index):
        if torch.rand(1) < self.rate:
            x1, f1, y1 = self.dataset[index]
            idx2 = torch.randint(len(self.dataset), (1,)).item()
            x2, _, y2 = self.dataset[idx2]
            lam = np.random.beta(self.beta, self.beta)
            lam = max(lam, 1.0 - lam)
            x = (x1 - x1.mean()) * lam + (x2 - x2.mean()) * (1.0 - lam)
            return x - x.mean(), f1, y1 * lam + y2 * (1.0 - lam)
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SoundKeyBalancedDataset(TorchDataset):
    def __init__(
        self,
        split,
        resample_rate=32000,
        gain_augment=0,
        max_samples=None,
        white_noise_prob=0.0,
        background_noise_prob=0.0,
        noise_snr_min_db=5.0,
        noise_snr_max_db=25.0,
    ):
        self.df = pd.read_csv(dataset_dir / "manifest.csv")
        self.df = self.df[self.df.split == split].reset_index(drop=True)
        if max_samples:
            self.df = self.df.groupby("label", group_keys=False).head(max_samples).reset_index(drop=True)
        self.resample_rate = resample_rate
        self.gain_augment = gain_augment
        self.classes_num = int(pd.read_csv(dataset_dir / "manifest.csv").target.max()) + 1
        self.white_noise_prob = white_noise_prob
        self.background_noise_prob = background_noise_prob
        self.noise_snr_min_db = noise_snr_min_db
        self.noise_snr_max_db = noise_snr_max_db
        self.background_df = self.df[self.df.label == "background_other"].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        offset = float(row.start_sec)
        duration = float(row.duration_sec)
        audio_path = Path(row.file_path)
        if not audio_path.is_absolute():
            audio_path = workspace_dir / audio_path
        waveform, _ = librosa.load(audio_path, sr=self.resample_rate, mono=True, offset=offset, duration=duration)
        waveform = gain_augment_waveform(waveform, self.gain_augment)
        waveform = pad_or_truncate(waveform, int(duration * self.resample_rate))
        waveform = add_white_noise(
            waveform,
            probability=self.white_noise_prob,
            snr_min_db=self.noise_snr_min_db,
            snr_max_db=self.noise_snr_max_db,
        )
        if (
            self.background_noise_prob > 0
            and len(self.background_df) > 0
            and np.random.random() < self.background_noise_prob
        ):
            bg = self.background_df.iloc[np.random.randint(len(self.background_df))]
            bg_path = Path(bg.file_path)
            if not bg_path.is_absolute():
                bg_path = workspace_dir / bg_path
            noise_waveform, _ = librosa.load(
                bg_path,
                sr=self.resample_rate,
                mono=True,
                offset=float(bg.start_sec),
                duration=duration,
            )
            noise_waveform = pad_or_truncate(noise_waveform, int(duration * self.resample_rate))
            waveform = mix_background_noise(
                waveform,
                noise_waveform,
                snr_min_db=self.noise_snr_min_db,
                snr_max_db=self.noise_snr_max_db,
            )
        target = np.zeros(self.classes_num, dtype=np.float32)
        target[int(row.target)] = 1.0
        return waveform.reshape(1, -1), f"{Path(row.file_path).name}@{offset:.1f}", target


def get_num_classes():
    return int(pd.read_csv(dataset_dir / "manifest.csv").target.max()) + 1


def get_labels():
    df = pd.read_csv(dataset_dir / "manifest.csv")
    pairs = df[["label", "target"]].drop_duplicates().values.tolist()
    return [label for label, _ in sorted(pairs, key=lambda x: x[1])]


def get_training_set(
    resample_rate=32000,
    roll=False,
    wavmix=False,
    gain_augment=0,
    max_samples=None,
    white_noise_prob=0.0,
    background_noise_prob=0.0,
    noise_snr_min_db=5.0,
    noise_snr_max_db=25.0,
):
    ds = SoundKeyBalancedDataset(
        "train",
        resample_rate=resample_rate,
        gain_augment=gain_augment,
        max_samples=max_samples,
        white_noise_prob=white_noise_prob,
        background_noise_prob=background_noise_prob,
        noise_snr_min_db=noise_snr_min_db,
        noise_snr_max_db=noise_snr_max_db,
    )
    if roll:
        ds = PreprocessDataset(ds, get_roll_func())
    if wavmix:
        ds = MixupDataset(ds)
    return ds


def get_eval_set(split="val", resample_rate=32000, max_samples=None):
    return SoundKeyBalancedDataset(split, resample_rate=resample_rate, max_samples=max_samples)
