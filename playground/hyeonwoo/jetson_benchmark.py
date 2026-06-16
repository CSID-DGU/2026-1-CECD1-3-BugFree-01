#!/usr/bin/env python3
"""Jetson/PyTorch synthetic inference benchmark.

This script estimates what size of neural network can run on the current
machine without downloading real models. It benchmarks several representative
model shapes with dummy inputs:

- 2D CNN: image/spectrogram-like workloads
- 1D CNN: raw-audio-like workloads
- Transformer encoder: AST/ViT-like attention workloads

Examples:
    python playground/hyeonwoo/jetson_benchmark.py
    python playground/hyeonwoo/jetson_benchmark.py --device cuda --preset stress
    python playground/hyeonwoo/jetson_benchmark.py --models cnn_tiny,audio_cnn --batch-sizes 1,2,4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for this benchmark.\n"
        "Install a Jetson-compatible PyTorch build first, then run this script again."
    ) from exc

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


class ConvBlock2d(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Cnn2dBenchmark(nn.Module):
    def __init__(self, channels: tuple[int, ...], num_classes: int = 527):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ConvBlock2d(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class AudioCnnBenchmark(nn.Module):
    def __init__(self, channels: tuple[int, ...], num_classes: int = 527):
        super().__init__()
        blocks = []
        in_ch = 1
        for out_ch in channels:
            blocks.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=9, stride=4, padding=4, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class TransformerBenchmark(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, num_classes: int = 527):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_shape: tuple[int, ...]
    factory: Callable[[], nn.Module]
    note: str


@dataclass
class BenchmarkResult:
    model: str
    note: str
    device: str
    dtype: str
    batch_size: int
    params_m: float
    input_shape: str
    mean_ms: float
    median_ms: float
    p90_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    throughput_items_s: float
    realtime_x_10s_audio: float | None
    peak_cuda_mem_mb: float | None
    ram_used_percent: float | None
    cpu_percent: float | None
    status: str


def build_specs() -> dict[str, ModelSpec]:
    return {
        "cnn_tiny": ModelSpec(
            name="cnn_tiny",
            input_shape=(1, 128, 313),
            factory=lambda: Cnn2dBenchmark((16, 32, 64, 128)),
            note="small spectrogram CNN",
        ),
        "cnn_small": ModelSpec(
            name="cnn_small",
            input_shape=(1, 128, 313),
            factory=lambda: Cnn2dBenchmark((32, 64, 128, 256)),
            note="EfficientAT/PANNs-small style CNN load",
        ),
        "cnn_medium": ModelSpec(
            name="cnn_medium",
            input_shape=(1, 224, 224),
            factory=lambda: Cnn2dBenchmark((48, 96, 192, 384, 512)),
            note="heavier image/spectrogram CNN",
        ),
        "audio_cnn": ModelSpec(
            name="audio_cnn",
            input_shape=(1, 160000),
            factory=lambda: AudioCnnBenchmark((32, 64, 128, 256, 384)),
            note="10 second 16 kHz raw-audio CNN",
        ),
        "transformer_tiny": ModelSpec(
            name="transformer_tiny",
            input_shape=(196, 192),
            factory=lambda: TransformerBenchmark(dim=192, depth=4, heads=3),
            note="tiny AST/ViT-like encoder",
        ),
        "transformer_small": ModelSpec(
            name="transformer_small",
            input_shape=(196, 384),
            factory=lambda: TransformerBenchmark(dim=384, depth=6, heads=6),
            note="small AST/ViT-like encoder",
        ),
    }


PRESETS = {
    "quick": ["cnn_tiny", "audio_cnn"],
    "default": ["cnn_tiny", "cnn_small", "audio_cnn", "transformer_tiny"],
    "stress": ["cnn_tiny", "cnn_small", "cnn_medium", "audio_cnn", "transformer_tiny", "transformer_small"],
}


def parse_csv_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("comma separated integers are required") from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return result


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def count_params(model: nn.Module) -> float:
    return sum(param.numel() for param in model.parameters()) / 1_000_000


def maybe_run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def print_system_info(device: torch.device) -> None:
    print("=" * 80)
    print("System")
    print(f"Python      : {platform.python_version()} ({platform.machine()})")
    print(f"OS          : {platform.platform()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"Device      : {device}")
    if psutil is not None:
        vm = psutil.virtual_memory()
        print(f"RAM         : {vm.total / 1024**3:.2f} GB total, {vm.percent:.1f}% used")
    if device.type == "cuda":
        print(f"CUDA        : {torch.version.cuda}")
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory  : {props.total_memory / 1024**3:.2f} GB")
    l4t = maybe_run(["cat", "/etc/nv_tegra_release"])
    if l4t:
        print(f"Jetson L4T  : {l4t.splitlines()[0]}")
    nvpmodel = maybe_run(["nvpmodel", "-q"])
    if nvpmodel:
        first_lines = "\n".join(nvpmodel.splitlines()[:4])
        print(f"nvpmodel    :\n{first_lines}")
    print("=" * 80)


def autocast_context(device: torch.device, use_fp16: bool):
    if device.type == "cuda" and use_fp16:
        return torch.cuda.amp.autocast()
    return torch.inference_mode()


def benchmark_one(
    spec: ModelSpec,
    batch_size: int,
    device: torch.device,
    iterations: int,
    warmup: int,
    use_fp16: bool,
) -> BenchmarkResult:
    model = spec.factory().eval().to(device)
    dtype_name = "fp16-autocast" if device.type == "cuda" and use_fp16 else "fp32"
    params_m = count_params(model)
    x = torch.randn((batch_size, *spec.input_shape), device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                with autocast_context(device, use_fp16):
                    _ = model(x)
            synchronize(device)

            latencies_ms = []
            for _ in range(iterations):
                start = time.perf_counter()
                with autocast_context(device, use_fp16):
                    _ = model(x)
                synchronize(device)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
    except RuntimeError as exc:
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        return BenchmarkResult(
            model=spec.name,
            note=spec.note,
            device=str(device),
            dtype=dtype_name,
            batch_size=batch_size,
            params_m=params_m,
            input_shape=str((batch_size, *spec.input_shape)),
            mean_ms=math.nan,
            median_ms=math.nan,
            p90_ms=math.nan,
            p95_ms=math.nan,
            min_ms=math.nan,
            max_ms=math.nan,
            throughput_items_s=math.nan,
            realtime_x_10s_audio=None,
            peak_cuda_mem_mb=peak_mem,
            ram_used_percent=psutil.virtual_memory().percent if psutil is not None else None,
            cpu_percent=psutil.cpu_percent(interval=None) if psutil is not None else None,
            status=f"failed: {exc}",
        )

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)
    throughput = batch_size / (mean_ms / 1000.0)
    realtime_x = None
    if spec.input_shape == (1, 160000):
        realtime_x = 10_000.0 / mean_ms

    return BenchmarkResult(
        model=spec.name,
        note=spec.note,
        device=str(device),
        dtype=dtype_name,
        batch_size=batch_size,
        params_m=params_m,
        input_shape=str((batch_size, *spec.input_shape)),
        mean_ms=mean_ms,
        median_ms=median_ms,
        p90_ms=percentile(latencies_ms, 0.90),
        p95_ms=percentile(latencies_ms, 0.95),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        throughput_items_s=throughput,
        realtime_x_10s_audio=realtime_x,
        peak_cuda_mem_mb=torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None,
        ram_used_percent=psutil.virtual_memory().percent if psutil is not None else None,
        cpu_percent=psutil.cpu_percent(interval=None) if psutil is not None else None,
        status="ok",
    )


def print_result(result: BenchmarkResult) -> None:
    if result.status != "ok":
        print(
            f"[FAIL] {result.model:18s} bs={result.batch_size:<2d} "
            f"params={result.params_m:6.2f}M status={result.status}"
        )
        return

    realtime = ""
    if result.realtime_x_10s_audio is not None:
        realtime = f" realtime={result.realtime_x_10s_audio:5.1f}x"
    cuda_mem = ""
    if result.peak_cuda_mem_mb is not None:
        cuda_mem = f" cuda_mem={result.peak_cuda_mem_mb:7.1f}MB"
    print(
        f"[ OK ] {result.model:18s} bs={result.batch_size:<2d} "
        f"params={result.params_m:6.2f}M "
        f"mean={result.mean_ms:8.2f}ms p95={result.p95_ms:8.2f}ms "
        f"throughput={result.throughput_items_s:7.2f}/s{realtime}{cuda_mem}"
    )


def save_results(results: list[BenchmarkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"jetson_benchmark_{timestamp}.csv"
    json_path = output_dir / f"jetson_benchmark_{timestamp}.json"

    rows = [asdict(result) for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


def print_interpretation(results: list[BenchmarkResult]) -> None:
    ok_results = [result for result in results if result.status == "ok"]
    if not ok_results:
        return

    print("=" * 80)
    print("How to read")
    print("- mean/p95 under 100 ms: comfortable for frequent near-realtime inference.")
    print("- 100-500 ms: usable for periodic audio event detection depending on window size.")
    print("- over 1000 ms: likely too slow unless inference is infrequent or batched offline.")
    audio_results = [r for r in ok_results if r.realtime_x_10s_audio is not None]
    if audio_results:
        best = max(audio_results, key=lambda r: r.realtime_x_10s_audio or 0.0)
        print(
            f"- Best raw-audio realtime factor here: {best.model} bs={best.batch_size}, "
            f"{best.realtime_x_10s_audio:.1f}x for a 10 second clip."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark approximate model capacity on Jetson or CPU.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--models", type=parse_csv_strings, default=None, help="Comma list. Overrides --preset.")
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=parse_csv_ints("1"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--fp32", action="store_true", help="Disable CUDA fp16 autocast.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for CSV/JSON benchmark output.",
    )
    args = parser.parse_args()

    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("--iterations must be >= 1 and --warmup must be >= 0")

    specs = build_specs()
    model_names = args.models if args.models is not None else PRESETS[args.preset]
    unknown = sorted(set(model_names) - set(specs))
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(sorted(specs))}")

    torch.set_grad_enabled(False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    device = select_device(args.device)
    use_fp16 = not args.fp32

    print_system_info(device)
    print(f"Models      : {', '.join(model_names)}")
    print(f"Batch sizes : {args.batch_sizes}")
    print(f"Iterations  : warmup={args.warmup}, measured={args.iterations}")
    print(f"Precision   : {'fp16 autocast on CUDA' if device.type == 'cuda' and use_fp16 else 'fp32'}")
    print("=" * 80)

    results = []
    for model_name in model_names:
        for batch_size in args.batch_sizes:
            result = benchmark_one(
                spec=specs[model_name],
                batch_size=batch_size,
                device=device,
                iterations=args.iterations,
                warmup=args.warmup,
                use_fp16=use_fp16,
            )
            results.append(result)
            print_result(result)

    save_results(results, args.output_dir)
    print_interpretation(results)


if __name__ == "__main__":
    main()
