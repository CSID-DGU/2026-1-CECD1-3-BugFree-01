from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"


def create_mobilenetv4_small(
    num_classes: int = 50,
    pretrained: bool = True,
    model_name: str = MODEL_NAME,
) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm이 필요합니다. 설치 예: pip install timm") from exc

    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        raise RuntimeError(
            f"timm 모델 로딩 실패: {model_name}. "
            "timm 버전, pretrained weight 캐시, 네트워크 접근 가능 여부를 확인하세요."
        ) from exc
    return model


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "checkpoint에서 state_dict를 찾지 못했습니다. "
        "지원 키: model_state_dict, state_dict, model"
    )


def _strip_prefix_if_present(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = False,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_prefix_if_present(state_dict, "module.")
    state_dict = _strip_prefix_if_present(state_dict, "model.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    return {
        "checkpoint_path": str(checkpoint_path),
        "strict": strict,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters() if not trainable_only else (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    return int(sum(parameter.numel() for parameter in parameters))
