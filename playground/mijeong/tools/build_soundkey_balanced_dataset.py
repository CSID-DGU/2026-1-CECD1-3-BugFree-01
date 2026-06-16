import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


FINAL_LABELS = [
    "baby_cry",
    "bicycle",
    "car_horn",
    "cat_meow",
    "dog_bark",
    "door_knock",
    "fire_alarm",
    "glass_breaking",
    "gunshot",
    "scream",
    "siren",
    "water_sound",
    "background_other",
]

SOUNDKEY_DIRECT_LIMITS = {
    "baby_cry": None,
    "bicycle": None,
    "car_horn": None,
    "cat_meow": 350,
    "dog_bark": 350,
    "door_knock": None,
    "glass_breaking": 60,
    "scream": None,
    "siren": 120,
    "water_sound": 160,
}

SOUNDKEY_MAPPED_LIMITS = {
    "crying": ("baby_cry", 60),
    "appliance_sound": ("background_other", 350),
}


def stable_bucket(key):
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def dbfs(samples):
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    rms = float(np.sqrt(np.mean(arr ** 2)) + 1e-12)
    return 20 * math.log10(rms)


def split_group_key(row):
    return f"{row['source_dataset']}::{row['source_folder']}::{row['source_file']}"


def assign_file_level_splits(rows):
    by_label_group = defaultdict(dict)
    for row in rows:
        by_label_group[row["label"]].setdefault(split_group_key(row), []).append(row)

    for label, grouped_rows in by_label_group.items():
        groups = sorted(
            grouped_rows.items(),
            key=lambda item: stable_bucket(f"{label}/{item[0]}"),
        )
        n = len(groups)
        if n >= 3:
            train_n = max(1, int(n * 0.7))
            val_n = max(1, int(n * 0.15))
            if train_n + val_n >= n:
                train_n = max(1, n - 2)
                val_n = 1
        elif n == 2:
            train_n, val_n = 1, 0
        else:
            train_n, val_n = 1, 0
        for index, (_, group_rows) in enumerate(groups):
            if index < train_n:
                split = "train"
            elif index < train_n + val_n:
                split = "val"
            else:
                split = "test"
            for row in group_rows:
                row["split"] = split


def cap_segments(rows, cap):
    if not cap:
        return rows
    capped = []
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    for label in FINAL_LABELS:
        label_rows = by_label.get(label, [])
        ordered = sorted(label_rows, key=lambda row: stable_bucket(f"{label}/{row['source_dataset']}/{row['source_file']}/{row['start_sec']}"))
        capped.extend(ordered[:cap])
    return capped


def safe_name(text):
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def copy_referenced_files(rows, output_dir):
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for row in rows:
        src = Path(row["original_file_path"])
        if src in mapping:
            row["file_path"] = mapping[src]
            continue
        label_dir = audio_dir / row["label"]
        label_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
        dst = label_dir / f"{safe_name(row['source_dataset'])}_{digest}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        mapping[src] = str(dst)
        row["file_path"] = str(dst)


def edge_rows(edge_manifest):
    df = pd.read_csv(edge_manifest)
    rows = []
    edge_stems = set()
    for item in df.to_dict("records"):
        label = item["label"]
        if label not in FINAL_LABELS:
            continue
        path = Path(item["file_path"])
        edge_stems.add(path.stem)
        rows.append({
            "original_file_path": str(path),
            "file_path": str(path),
            "label": label,
            "target": -1,
            "start_sec": float(item["start_sec"]),
            "duration_sec": float(item["duration_sec"]),
            "rms_dbfs": float(item["rms_dbfs"]),
            "split": "",
            "source_dataset": "edge_audio_dataset",
            "source_folder": str(item.get("source_folder", path.parent.name)),
            "source_file": path.name,
        })
    return rows, edge_stems


def soundkey_audio_files(soundkey_root, label):
    files = []
    for split in ("Training", "Validation"):
        folder = soundkey_root / split / "원본데이터" / label
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
                files.append({
                    "split": split,
                    "project_label": label,
                    "mapped_label": label,
                    "audio_file": path.name,
                    "path": path,
                })
    return files


def select_soundkey_files(soundkey_root, soundkey_manifest, edge_stems, seed):
    rng = random.Random(seed)
    candidates = []
    skipped_overlap = Counter()

    for original_label, limit in SOUNDKEY_DIRECT_LIMITS.items():
        valid = []
        for row in soundkey_audio_files(soundkey_root, original_label):
            path = Path(row["path"])
            if path.stem in edge_stems:
                skipped_overlap[original_label] += 1
                continue
            valid.append(row)
        valid = sorted(valid, key=lambda row: stable_bucket(f"{original_label}/{row['audio_file']}"))
        if limit is not None:
            rng.shuffle(valid)
            valid = valid[:limit]
            valid = sorted(valid, key=lambda row: row["audio_file"])
        candidates.extend(valid)

    for original_label, (mapped_label, limit) in SOUNDKEY_MAPPED_LIMITS.items():
        valid = []
        for row in soundkey_audio_files(soundkey_root, original_label):
            row["mapped_label"] = mapped_label
            path = Path(row["path"])
            if path.stem in edge_stems:
                skipped_overlap[original_label] += 1
                continue
            valid.append(row)
        rng.shuffle(valid)
        candidates.extend(valid[:limit])

    return candidates, skipped_overlap, Counter()


def segment_soundkey_files(candidates, window_sec, threshold_dbfs, resample_rate):
    rows = []
    excluded = Counter()
    for item in candidates:
        path = Path(item["path"])
        try:
            wav, sr = librosa.load(path, sr=resample_rate, mono=True)
        except Exception:
            excluded[f"read_error::{item['project_label']}"] += 1
            continue
        samples_per_window = int(resample_rate * window_sec)
        if len(wav) < samples_per_window:
            excluded[f"shorter_than_window::{item['project_label']}"] += 1
            continue
        segment_count = len(wav) // samples_per_window
        for index in range(segment_count):
            start = index * samples_per_window
            end = start + samples_per_window
            segment = wav[start:end]
            level = dbfs(segment)
            if level < threshold_dbfs:
                excluded[f"below_threshold::{item['project_label']}"] += 1
                continue
            rows.append({
                "original_file_path": str(path),
                "file_path": str(path),
                "label": item["mapped_label"],
                "target": -1,
                "start_sec": round(index * window_sec, 3),
                "duration_sec": window_sec,
                "rms_dbfs": round(level, 3),
                "split": "",
                "source_dataset": f"soundkey_dataset::{item['project_label']}",
                "source_folder": f"{item['split']}/원본데이터/{item['project_label']}",
                "source_file": path.name,
            })
    return rows, excluded


def write_manifest(rows, output_dir):
    label_to_index = {label: index for index, label in enumerate(FINAL_LABELS)}
    for row in rows:
        row["target"] = label_to_index[row["label"]]
    assign_file_level_splits(rows)

    fieldnames = [
        "file_path", "label", "target", "start_sec", "duration_sec", "rms_dbfs",
        "split", "source_dataset", "source_folder", "source_file",
    ]
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    (output_dir / "label_map.json").write_text(
        json.dumps({index: label for index, label in enumerate(FINAL_LABELS)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-manifest", default="EfficientAT/data/edge_audio_dataset/manifest.csv")
    parser.add_argument("--soundkey-root", default="datasets/soundkey_dataset")
    parser.add_argument("--soundkey-manifest", default="datasets/soundkey_dataset/manifest.csv")
    parser.add_argument("--out-dir", default="datasets/soundkey_balanced_v2")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--threshold-dbfs", type=float, default=-45.0)
    parser.add_argument("--resample-rate", type=int, default=32000)
    parser.add_argument("--segment-cap", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, edge_stems = edge_rows(Path(args.edge_manifest))
    soundkey_candidates, skipped_overlap, skipped_missing = select_soundkey_files(
        Path(args.soundkey_root), Path(args.soundkey_manifest), edge_stems, args.seed
    )
    soundkey_rows, soundkey_excluded = segment_soundkey_files(
        soundkey_candidates, args.window_sec, args.threshold_dbfs, args.resample_rate
    )
    all_rows_before_cap = rows + soundkey_rows
    final_rows = cap_segments(all_rows_before_cap, args.segment_cap)
    copy_referenced_files(final_rows, output_dir)
    write_manifest(final_rows, output_dir)

    report = {
        "output_dir": str(output_dir),
        "window_sec": args.window_sec,
        "threshold_dbfs": args.threshold_dbfs,
        "resample_rate": args.resample_rate,
        "segment_cap_per_label": args.segment_cap,
        "labels": FINAL_LABELS,
        "total_segments_before_cap": len(all_rows_before_cap),
        "total_segments": len(final_rows),
        "counts_by_label": dict(Counter(row["label"] for row in final_rows)),
        "counts_by_split": dict(Counter(row["split"] for row in final_rows)),
        "counts_by_source": dict(Counter(row["source_dataset"] for row in final_rows)),
        "soundkey_file_candidates": len(soundkey_candidates),
        "skipped_soundkey_overlap_by_stem": dict(skipped_overlap),
        "skipped_soundkey_missing": dict(skipped_missing),
        "soundkey_excluded_segments": dict(soundkey_excluded),
        "selection_policy": {
            "edge_audio_dataset": "kept as primary source, then segment-capped for balance",
            "soundkey_dataset": "used for recommended classes only, edge stem overlap excluded",
            "split": "file-level split by source_dataset/source_folder/source_file; segments from the same source file stay in one split",
            "crying": "random subset mapped into baby_cry",
            "appliance_sound": "random subset mapped into background_other",
            "gunshot": "not imported from soundkey_dataset; edge gunshot is segment-capped",
            "construction_noise": "not used",
        },
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
