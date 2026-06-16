from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from audio_transforms import AudioConfig, LogMelImageTransform


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ESC50_ROOT = Path("../ESC-50-master")


@dataclass(frozen=True)
class ESC50Metadata:
    dataframe: pd.DataFrame
    class_names: list[str]
    esc50_root: Path
    metadata_path: Path
    audio_dir: Path


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()

    script_relative = (SCRIPT_DIR / path).resolve()
    cwd_relative = path.resolve()
    if script_relative.exists() or not cwd_relative.exists():
        return script_relative
    return cwd_relative


def resolve_esc50_root(path: Path | None) -> Path:
    return resolve_path(path or DEFAULT_ESC50_ROOT)


def build_class_names(metadata: pd.DataFrame) -> list[str]:
    target_to_category: dict[int, str] = {}
    for row in metadata[["target", "category"]].drop_duplicates().itertuples(index=False):
        target = int(row.target)
        category = str(row.category)
        existing = target_to_category.get(target)
        if existing is not None and existing != category:
            raise ValueError(
                f"ESC-50 target {target}에 여러 category가 매핑되어 있습니다: "
                f"{existing}, {category}"
            )
        target_to_category[target] = category

    if not target_to_category:
        raise ValueError("ESC-50 metadata에서 class 정보를 찾지 못했습니다.")

    max_target = max(target_to_category)
    class_names = [target_to_category.get(index, f"class_{index}") for index in range(max_target + 1)]
    return class_names


def load_esc50_metadata(
    esc50_root: Path | None = None,
    metadata_csv: Path | None = None,
    fold: int | None = None,
) -> ESC50Metadata:
    resolved_root = resolve_esc50_root(esc50_root)
    metadata_path = resolve_path(metadata_csv) if metadata_csv is not None else resolved_root / "meta" / "esc50.csv"
    audio_dir = resolved_root / "audio"

    if not resolved_root.exists():
        raise FileNotFoundError(f"ESC-50 directory not found: {resolved_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata CSV not found: {metadata_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 audio directory not found: {audio_dir}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {"filename", "fold", "target", "category"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ESC-50 metadata에 필수 컬럼이 없습니다: {missing}")

    class_names = build_class_names(metadata)

    if fold is not None:
        metadata = metadata[metadata["fold"].astype(int).eq(int(fold))].copy()
        if metadata.empty:
            raise ValueError(f"fold={fold}에 해당하는 ESC-50 샘플이 없습니다.")

    metadata = metadata.sort_values(["fold", "target", "filename"]).reset_index(drop=True)
    return ESC50Metadata(
        dataframe=metadata,
        class_names=class_names,
        esc50_root=resolved_root,
        metadata_path=metadata_path,
        audio_dir=audio_dir,
    )


class ESC50LogMelDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_dir: Path,
        audio_config: AudioConfig,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.transform = LogMelImageTransform(audio_config)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        filename = str(row["filename"])
        audio_path = self.audio_dir / filename
        if not audio_path.exists():
            raise FileNotFoundError(f"Metadata에 있는 오디오 파일이 없습니다: {audio_path}")

        image = self.transform.load_file(audio_path)
        return {
            "image": image,
            "target": torch.tensor(int(row["target"]), dtype=torch.long),
            "filename": filename,
            "fold": int(row["fold"]),
            "category": str(row["category"]),
        }
