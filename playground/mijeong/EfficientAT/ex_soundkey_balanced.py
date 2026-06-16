import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import soundkey_balanced
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
from models.dymn.model import get_model as get_dymn
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT


dataset_dir = Path(__file__).resolve().parents[1] / "datasets" / "soundkey_balanced_v2"


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _build_model(args, device):
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    num_classes = soundkey_balanced.get_num_classes()

    if model_name.startswith("dymn"):
        model = get_dymn(
            width_mult=width,
            pretrained_name=pretrained_name,
            pretrain_final_temp=args.pretrain_final_temp,
            num_classes=num_classes,
        )
    else:
        model = get_mobilenet(
            width_mult=width,
            pretrained_name=pretrained_name,
            head_type=args.head_type,
            se_dims=args.se_dims,
            num_classes=num_classes,
        )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier")

    return model.to(device)


def _class_weights(args, device):
    if args.class_weighting == "none":
        return None
    df = pd.read_csv(dataset_dir / "manifest.csv")
    counts = df[df.split == "train"].groupby("target").size().sort_index().astype(float)
    weights = counts.max() / counts
    if args.class_weighting == "sqrt_inverse":
        weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights.values, dtype=torch.float32, device=device)


def _evaluate(model, mel, eval_loader, device, labels, class_weights=None):
    model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    with torch.no_grad():
        for x, _, y in tqdm(eval_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
            targets.append(y.detach().cpu().numpy())
            outputs.append(y_hat.float().detach().cpu().numpy())
            losses.append(F.cross_entropy(y_hat, y, weight=class_weights).detach().cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    target_idx = targets.argmax(axis=1)
    output_idx = outputs.argmax(axis=1)
    per_class = metrics.precision_recall_fscore_support(
        target_idx,
        output_idx,
        labels=np.arange(len(labels)),
        zero_division=0,
    )

    return {
        "loss": float(np.stack(losses).mean()),
        "accuracy": float(metrics.accuracy_score(target_idx, output_idx)),
        "macro_f1": float(metrics.f1_score(target_idx, output_idx, average="macro", zero_division=0)),
        "weighted_f1": float(metrics.f1_score(target_idx, output_idx, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(per_class[0][idx]),
                "recall": float(per_class[1][idx]),
                "f1": float(per_class[2][idx]),
                "support": int(per_class[3][idx]),
            }
            for idx, label in enumerate(labels)
        },
    }


def train(args):
    device = torch.device("cuda") if args.cuda and torch.cuda.is_available() else torch.device("cpu")
    labels = soundkey_balanced.get_labels()

    mel = AugmentMelSTFT(
        n_mels=args.n_mels,
        sr=args.resample_rate,
        win_length=args.window_size,
        hopsize=args.hop_size,
        n_fft=args.n_fft,
        freqm=args.freqm,
        timem=args.timem,
        fmin=args.fmin,
        fmax=args.fmax,
        fmin_aug_range=args.fmin_aug_range,
        fmax_aug_range=args.fmax_aug_range,
    ).to(device)

    model = _build_model(args, device)
    weights = _class_weights(args, device)

    eval_dl = DataLoader(
        dataset=soundkey_balanced.get_eval_set(
            split=args.eval_split,
            resample_rate=args.resample_rate,
            max_samples=args.max_eval_samples,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    if args.eval_only:
        result = _evaluate(model, mel, eval_dl, device, labels, weights)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    train_dl = DataLoader(
        dataset=soundkey_balanced.get_training_set(
            resample_rate=args.resample_rate,
            roll=not args.no_roll,
            wavmix=not args.no_wavmix,
            gain_augment=args.gain_augment,
            max_samples=args.max_train_samples,
            white_noise_prob=args.white_noise_prob,
            background_noise_prob=args.background_noise_prob,
            noise_snr_min_db=args.noise_snr_min_db,
            noise_snr_max_db=args.noise_snr_max_db,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value),
    )

    best_macro_f1 = -1.0
    last_metrics = None

    for epoch in range(args.n_epochs):
        model.train()
        mel.train()
        train_losses = []
        for x, _, y in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            x, y = x.to(device), y.to(device)
            x = _mel_forward(x, mel)
            bs = x.size(0)

            if args.mixup_alpha:
                rn_indices, lam = mixup(bs, args.mixup_alpha)
                lam = lam.to(device)
                x = x * lam.reshape(bs, 1, 1, 1) + x[rn_indices] * (1.0 - lam.reshape(bs, 1, 1, 1))
                y_hat, _ = model(x)
                loss_a = F.cross_entropy(y_hat, y, weight=weights, reduction="none")
                loss_b = F.cross_entropy(y_hat, y[rn_indices], weight=weights, reduction="none")
                loss = (loss_a * lam.reshape(bs) + loss_b * (1.0 - lam.reshape(bs))).mean()
            else:
                y_hat, _ = model(x)
                loss = F.cross_entropy(y_hat, y, weight=weights)

            train_losses.append(float(loss.detach().cpu().numpy()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        last_metrics = _evaluate(model, mel, eval_dl, device, labels, weights)
        last_metrics["epoch"] = epoch + 1
        last_metrics["train_loss"] = float(np.mean(train_losses))

        torch.save(model.state_dict(), output_dir / "last.pt")
        (output_dir / "last_metrics.json").write_text(
            json.dumps(last_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if last_metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = last_metrics["macro_f1"]
            torch.save(model.state_dict(), output_dir / "best.pt")
            (output_dir / "best_metrics.json").write_text(
                json.dumps(last_metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        print(json.dumps(last_metrics, ensure_ascii=False, indent=2))

    return last_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="soundkey_balanced")
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="runs/soundkey_balanced_v2")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--model_name", type=str, default="mn10_as")
    parser.add_argument("--pretrain_final_temp", type=float, default=1.0)
    parser.add_argument("--model_width", type=float, default=1.0)
    parser.add_argument("--head_type", type=str, default="mlp")
    parser.add_argument("--se_dims", type=str, default="c")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--mixup_alpha", type=float, default=0.3)
    parser.add_argument("--no_roll", action="store_true", default=False)
    parser.add_argument("--no_wavmix", action="store_true", default=False)
    parser.add_argument("--gain_augment", type=int, default=12)
    parser.add_argument("--white_noise_prob", type=float, default=0.35)
    parser.add_argument("--background_noise_prob", type=float, default=0.35)
    parser.add_argument("--noise_snr_min_db", type=float, default=5.0)
    parser.add_argument("--noise_snr_max_db", type=float, default=25.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--class_weighting", type=str, default="sqrt_inverse", choices=["none", "inverse", "sqrt_inverse"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--warm_up_len", type=int, default=2)
    parser.add_argument("--ramp_down_start", type=int, default=2)
    parser.add_argument("--ramp_down_len", type=int, default=8)
    parser.add_argument("--last_lr_value", type=float, default=0.01)
    parser.add_argument("--resample_rate", type=int, default=32000)
    parser.add_argument("--window_size", type=int, default=800)
    parser.add_argument("--hop_size", type=int, default=320)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--freqm", type=int, default=0)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=None)
    parser.add_argument("--fmin_aug_range", type=int, default=10)
    parser.add_argument("--fmax_aug_range", type=int, default=2000)

    train(parser.parse_args())
