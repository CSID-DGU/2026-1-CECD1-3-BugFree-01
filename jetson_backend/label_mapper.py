from __future__ import annotations

from typing import Dict, List, Sequence

from constants import AUDIOSET_CLASS_COUNT, LABEL_MAPPING


class CustomLabelMapper:
    def build_indices(self, labels: Sequence[str]) -> Dict[str, List[int]]:
        if len(labels) != AUDIOSET_CLASS_COUNT:
            raise RuntimeError(
                f"AudioSet 라벨 수가 {AUDIOSET_CLASS_COUNT}개가 아닙니다: {len(labels)}"
            )

        label_to_index = {label: index for index, label in enumerate(labels)}
        indices: Dict[str, List[int]] = {}
        missing: Dict[str, List[str]] = {}

        for custom_label, audioset_labels in LABEL_MAPPING.items():
            matched = [label_to_index[label] for label in audioset_labels if label in label_to_index]
            not_found = [label for label in audioset_labels if label not in label_to_index]
            indices[custom_label] = matched
            if not_found:
                missing[custom_label] = not_found

        if missing:
            details = "; ".join(
                f"{custom_label}: {', '.join(labels)}"
                for custom_label, labels in missing.items()
            )
            raise RuntimeError(f"AudioSet 라벨을 찾지 못했습니다: {details}")

        return indices
