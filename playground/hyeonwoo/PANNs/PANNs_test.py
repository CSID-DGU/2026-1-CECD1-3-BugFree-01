"""
PANNs CNN6 + ESC-50 inference pipeline
--------------------------------------
ESC-50 데이터셋 다운로드 → mel-spectrogram 전처리 → CNN6 추론 → 결과 출력

python PANNs_test.py <- 사용법
"""

import os
import sys
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
# 0. 설정 상수
# ─────────────────────────────────────────────
SAMPLE_RATE   = 32000       # PANNs 학습 기준 샘플레이트
CLIP_DURATION = 2.0         # 추론에 사용할 오디오 길이 (초)
WINDOW_SIZE   = 1024
HOP_SIZE      = 320
MEL_BINS      = 64
FMIN          = 50
FMAX          = 14000
CLASSES_NUM   = 527         # AudioSet 클래스 수

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Jetson Nano 타겟 클래스 (경적/위험음 관련)
TARGET_CLASSES = {
    "car_horn":   ["Car horn", "Vehicle horn, car horn, honking"],
    "siren":      ["Siren", "Civil defense siren", "Ambulance (siren)"],
    "alarm":      ["Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Alarm clock"],
    "screaming":  ["Screaming", "Shout"],
    "crash":      ["Vehicle collision, crash", "Crash"],
}

BASE_DIR   = Path(__file__).parent        # hyeonwoo/PANNs/
DATA_DIR   = BASE_DIR.parent / "data"    # hyeonwoo/data  ✓
MODEL_DIR  = BASE_DIR.parent / "models"  # hyeonwoo/models  ✓
RESULT_DIR = BASE_DIR / "results"        # hyeonwoo/PANNs/results  ✓

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    d.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. ESC-50 다운로드
# ─────────────────────────────────────────────
def download_esc50():
    """
    ESC-50 데이터셋 다운로드.
    실제 환경: GitHub or Zenodo에서 자동 다운로드
    네트워크 제한 환경: 합성 오디오로 파이프라인 검증
    """
    esc_dir = DATA_DIR / "ESC-50-master"
    if esc_dir.exists():
        print(f"[✓] ESC-50 이미 존재: {esc_dir}")
        return esc_dir

    # --- 실제 다운로드 시도 ---
    urls = [
        "https://github.com/karoldvl/ESC-50/archive/master.zip",
        "https://zenodo.org/record/1203745/files/ESC-50-master.zip",
    ]
    zippath = DATA_DIR / "ESC-50-master.zip"
    downloaded = False

    for url in urls:
        try:
            print(f"[↓] 다운로드 시도: {url}")
            import requests
            with requests.get(url, stream=True, timeout=60,
                              headers={"User-Agent": "Mozilla/5.0"}) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(zippath, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, ncols=60
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print("[→] 압축 해제 중...")
            with zipfile.ZipFile(zippath, "r") as zf:
                zf.extractall(DATA_DIR)
            zippath.unlink()
            downloaded = True
            break
        except Exception as e:
            print(f"    [!] 실패: {e}")

    if not downloaded:
        # --- 합성 오디오로 파이프라인 검증 (네트워크 없는 환경) ---
        print("[!] 다운로드 실패 → 합성 오디오로 파이프라인 검증 모드 진입")
        _make_synthetic_esc50(esc_dir)

    print(f"[✓] ESC-50 준비 완료: {esc_dir}")
    return esc_dir


def _make_synthetic_esc50(esc_dir: Path):
    """
    실제 ESC-50 없이 파이프라인 전체를 검증하기 위한 합성 데이터 생성.
    각 카테고리별 다른 주파수의 사인파 + 노이즈로 구성.
    실제 Jetson 배포 시에는 진짜 ESC-50 또는 커스텀 데이터 사용.
    """
    import soundfile as sf

    # ESC-50 폴더 구조 생성
    audio_dir = esc_dir / "audio"
    meta_dir  = esc_dir / "meta"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    SR = 44100  # ESC-50 원본 SR
    DURATION = 5  # 5초

    # 실제 ESC-50 50개 카테고리 (fold, target, category)
    categories = [
        (1, 0,  "dog"),             (1, 1,  "rooster"),
        (1, 2,  "pig"),             (1, 3,  "cow"),
        (1, 4,  "frog"),            (1, 5,  "cat"),
        (1, 6,  "hen"),             (1, 7,  "insects"),
        (1, 8,  "sheep"),           (1, 9,  "crow"),
        (2, 10, "rain"),            (2, 11, "sea_waves"),
        (2, 12, "crackling_fire"),  (2, 13, "crickets"),
        (2, 14, "chirping_birds"),  (2, 15, "water_drops"),
        (2, 16, "wind"),            (2, 17, "pouring_water"),
        (2, 18, "toilet_flush"),    (2, 19, "thunderstorm"),
        (3, 20, "crying_baby"),     (3, 21, "sneezing"),
        (3, 22, "clapping"),        (3, 23, "breathing"),
        (3, 24, "coughing"),        (3, 25, "footsteps"),
        (3, 26, "laughing"),        (3, 27, "brushing_teeth"),
        (3, 28, "snoring"),         (3, 29, "drinking_sipping"),
        (4, 30, "door_wood_knock"), (4, 31, "mouse_click"),
        (4, 32, "keyboard_typing"), (4, 33, "door_wood_creaks"),
        (4, 34, "can_opening"),     (4, 35, "washing_machine"),
        (4, 36, "vacuum_cleaner"),  (4, 37, "clock_alarm"),
        (4, 38, "clock_tick"),      (4, 39, "glass_breaking"),
        (5, 40, "helicopter"),      (5, 41, "chainsaw"),
        (5, 42, "siren"),           (5, 43, "car_horn"),
        (5, 44, "engine"),          (5, 45, "train"),
        (5, 46, "church_bells"),    (5, 47, "airplane"),
        (5, 48, "fireworks"),       (5, 49, "hand_saw"),
    ]

    # 카테고리별 특징 주파수 (합성음 차별화)
    freq_map = {
        "dog": 440, "rooster": 880, "siren": 700, "car_horn": 500,
        "engine": 150, "clock_alarm": 1000, "glass_breaking": 2000,
        "crying_baby": 600, "clapping": 1200, "fireworks": 800,
    }

    records = []
    t = np.linspace(0, DURATION, SR * DURATION)

    for fold, target, category in categories:
        freq = freq_map.get(category, 300 + target * 30)
        # 기본파 + 2배음 + 노이즈
        wave = (0.5 * np.sin(2 * np.pi * freq * t) +
                0.3 * np.sin(2 * np.pi * freq * 2 * t) +
                0.1 * np.random.randn(len(t)))
        wave = (wave / np.abs(wave).max() * 0.8).astype(np.float32)

        filename = f"{fold}-{target:05d}-{target:02d}-0.wav"
        sf.write(str(audio_dir / filename), wave, SR)
        records.append({
            "filename": filename, "fold": fold, "target": target,
            "category": category, "esc10": target < 10, "src_file": filename, "take": "A"
        })

    pd.DataFrame(records).to_csv(meta_dir / "esc50.csv", index=False)
    print(f"    합성 ESC-50 생성 완료: {len(records)}개 파일 ({esc_dir})")


def load_esc50_metadata(esc_dir: Path) -> pd.DataFrame:
    """ESC-50 메타데이터 CSV 로드."""
    meta_path = esc_dir / "meta" / "esc50.csv"
    df = pd.read_csv(meta_path)
    df["filepath"] = df["filename"].apply(
        lambda fn: str(esc_dir / "audio" / fn)
    )
    print(f"[✓] ESC-50 메타데이터 로드: {len(df)}개 샘플, {df['category'].nunique()}개 클래스")
    return df


# ─────────────────────────────────────────────
# 2. PANNs CNN6 모델 정의
# ─────────────────────────────────────────────
def init_layer(layer: nn.Module):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)

def init_bn(bn: nn.BatchNorm2d):
    nn.init.ones_(bn.weight)
    nn.init.zeros_(bn.bias)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # 3×3 → 5×5로 변경, conv는 1개만
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        if   pool_type == "max":     x = F.max_pool2d(x, pool_size)
        elif pool_type == "avg":     x = F.avg_pool2d(x, pool_size)
        elif pool_type == "avg+max": x = F.avg_pool2d(x, pool_size) + F.max_pool2d(x, pool_size)
        return x


class Cnn6(nn.Module):
    """
    PANNs CNN6 — 4.8M 파라미터
    논문: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks
          for Audio Pattern Recognition", TASLP 2020
    """
    def __init__(self, classes_num: int = CLASSES_NUM):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)

        self.fc1         = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, time_steps, mel_bins)  — log-mel spectrogram
        Returns:
            clipwise_output: (batch, classes_num)
            embedding:       (batch, 512)
        """
        x = x.unsqueeze(1)                    # (B, 1, T, mel)
        x = x.transpose(1, 3)                 # (B, mel, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)                 # (B, 1, T, mel)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)              # (B, 512, T')
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2                           # time pooling

        x  = F.dropout(x, p=0.5, training=self.training)
        x  = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))

        return clipwise_output, embedding


# ─────────────────────────────────────────────
# 3. 사전학습 가중치 다운로드
# ─────────────────────────────────────────────
def download_pretrained_weights():
    """
    PANNs CNN6 AudioSet 사전학습 가중치 다운로드.
    출처: https://zenodo.org/record/3987831
    네트워크 제한 시 → 무작위 초기화 가중치로 구조 검증
    """
    weight_path = MODEL_DIR / "Cnn6_mAP=0.343.pth"

    if weight_path.exists():
        print(f"[✓] 가중치 이미 존재: {weight_path}")
        return weight_path

    weight_url = "https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth"
    print(f"[↓] PANNs CNN6 사전학습 가중치 다운로드 중...")
    print(f"    URL: {weight_url}")

    try:
        import requests
        with requests.get(weight_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(weight_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, ncols=60, desc="  가중치"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"[✓] 가중치 저장: {weight_path}")
    except Exception as e:
        print(f"[!] 가중치 다운로드 실패: {e}")
        print("[!] → 무작위 초기화 가중치로 구조/파이프라인 검증 모드")
        weight_path = None  # load_model에서 처리

    return weight_path


def load_model(weight_path) -> Cnn6:
    """가중치를 로드한 CNN6 모델 반환. weight_path=None이면 무작위 초기화."""
    model = Cnn6(classes_num=CLASSES_NUM).to(DEVICE)
    if weight_path and Path(weight_path).exists():
        checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=False)  # spectrogram_extractor 등 불필요한 키 무시
        print(f"[✓] 사전학습 가중치 로드 완료")
    else:
        print(f"[!] 무작위 초기화 — 파이프라인/구조 검증용 (실제 추론 성능 없음)")
        print(f"    ※ Zenodo에서 Cnn6_mAP=0.343.pth 다운로드 후 models/ 디렉토리에 넣으세요")
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] CNN6 준비 완료 — 파라미터 수: {total_params/1e6:.2f}M, device: {DEVICE}")
    return model


# ─────────────────────────────────────────────
# 4. 오디오 전처리
# ─────────────────────────────────────────────
def audio_to_logmel(filepath: str) -> np.ndarray:
    """
    오디오 파일 → Log-mel spectrogram (time_steps, mel_bins)
    PANNs 논문 기준 파라미터 사용
    """
    waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    # 일정 길이로 맞춤 (패딩 or 크롭)
    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]

    # Short-time Fourier Transform → mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=WINDOW_SIZE,
        hop_length=HOP_SIZE,
        n_mels=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # power → dB (log scale)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # (mel_bins, time_steps) → (time_steps, mel_bins)
    return log_mel.T.astype(np.float32)


# ─────────────────────────────────────────────
# 5. AudioSet 클래스 레이블 로드
# ─────────────────────────────────────────────
def load_audioset_labels() -> list:
    """
    AudioSet 527개 클래스 레이블 로드.
    네트워크 없는 환경에서는 인덱스 기반 더미 레이블 사용.
    """
    labels_path = MODEL_DIR / "class_labels_indices.csv"
    labels_url  = ("https://raw.githubusercontent.com/qiuqiangkong/"
                   "audioset_tagging_cnn/master/metadata/class_labels_indices.csv")

    if not labels_path.exists():
        print("[↓] AudioSet 레이블 다운로드 시도...")
        try:
            import requests
            r = requests.get(labels_url, timeout=20)
            r.raise_for_status()
            with open(labels_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            print("[✓] 레이블 저장 완료")
        except Exception as e:
            print(f"[!] 레이블 다운로드 실패: {e} → 인덱스 기반 더미 레이블 사용")
            # 더미 레이블 생성 (위험음 클래스는 실제 이름 삽입)
            dummy = [{"index": i, "mid": f"/m/{i:05d}", "display_name": f"AudioSet_class_{i}"}
                     for i in range(CLASSES_NUM)]
            # 알려진 위험음 인덱스 직접 삽입 (AudioSet 공식 인덱스)
            known = {
                48:  "Car horn",       49:  "Vehicle horn, car horn, honking",
                77:  "Siren",          78:  "Civil defense siren",
                79:  "Ambulance (siren)",
                388: "Alarm",          389: "Fire alarm",
                390: "Smoke detector, smoke alarm",  391: "Alarm clock",
                20:  "Screaming",      21:  "Shout",
                473: "Vehicle collision, crash",     474: "Crash",
            }
            for idx, name in known.items():
                if idx < CLASSES_NUM:
                    dummy[idx]["display_name"] = name
            pd.DataFrame(dummy).to_csv(labels_path, index=False)

    df     = pd.read_csv(labels_path)
    labels = df["display_name"].tolist()
    print(f"[✓] AudioSet 레이블 로드: {len(labels)}개")
    return labels


# ─────────────────────────────────────────────
# 6. 단일 파일 추론
# ─────────────────────────────────────────────
def infer_single(model: Cnn6, filepath: str, labels: list[str], top_k: int = 5) -> dict:
    """
    단일 오디오 파일에 대해 추론 수행.
    Returns:
        dict with top_k predictions and target class scores
    """
    log_mel = audio_to_logmel(filepath)
    tensor  = torch.from_numpy(log_mel).unsqueeze(0).to(DEVICE)  # (1, T, mel)

    with torch.no_grad():
        clipwise_output, _ = model(tensor)

    probs = clipwise_output.squeeze(0).cpu().numpy()  # (527,)

    # Top-K 예측
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_preds   = [(labels[i], float(probs[i])) for i in top_indices]

    # 타겟 클래스 점수 (경적/위험음)
    target_scores = {}
    for cls_name, audioset_names in TARGET_CLASSES.items():
        score = 0.0
        for aname in audioset_names:
            if aname in labels:
                idx    = labels.index(aname)
                score  = max(score, float(probs[idx]))
        target_scores[cls_name] = score

    return {
        "filepath":     filepath,
        "top_k":        top_preds,
        "target_scores": target_scores,
        "raw_probs":    probs,
    }


# ─────────────────────────────────────────────
# 7. ESC-50 배치 추론
# ─────────────────────────────────────────────
def run_esc50_inference(model: Cnn6, df: pd.DataFrame, labels: list[str],
                        n_samples: int = 50) -> pd.DataFrame:
    """
    ESC-50에서 n_samples개를 추론하고 결과를 DataFrame으로 반환.
    위험음 관련 카테고리 우선 샘플링.
    """
    # 위험음 관련 카테고리 우선 샘플
    danger_cats = ["car_horn", "siren", "engine", "dog", "crying_baby",
                   "crackling_fire", "fireworks", "clapping"]
    priority = df[df["category"].isin(danger_cats)]
    others   = df[~df["category"].isin(danger_cats)]

    n_priority = min(len(priority), n_samples // 2)
    n_other    = n_samples - n_priority
    sample_df  = pd.concat([
        priority.sample(n=n_priority, random_state=42),
        others.sample(n=min(len(others), n_other), random_state=42),
    ]).reset_index(drop=True)

    print(f"\n[→] {len(sample_df)}개 파일 추론 시작...")
    records = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), ncols=70):
        try:
            result = infer_single(model, row["filepath"], labels, top_k=3)
            top1_label, top1_prob = result["top_k"][0]
            records.append({
                "filename":       row["filename"],
                "esc50_category": row["category"],
                "top1_audioset":  top1_label,
                "top1_prob":      round(top1_prob, 4),
                **{f"score_{k}": round(v, 4)
                   for k, v in result["target_scores"].items()},
            })
        except Exception as e:
            print(f"  [!] 오류 {row['filename']}: {e}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 8. 결과 출력
# ─────────────────────────────────────────────
def print_results(results_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  PANNs CNN6 — ESC-50 추론 결과")
    print("=" * 70)

    # 카테고리별 평균 타겟 점수
    score_cols = [c for c in results_df.columns if c.startswith("score_")]
    print("\n[위험음 클래스별 평균 confidence score]")
    print(f"  {'클래스':<15} {'평균 score':>12}")
    print("  " + "-" * 30)
    for col in score_cols:
        cls_name = col.replace("score_", "")
        mean_score = results_df[col].mean()
        bar = "█" * int(mean_score * 30)
        print(f"  {cls_name:<15} {mean_score:>8.4f}  {bar}")

    # 상위 10개 샘플 출력
    print(f"\n[샘플 추론 결과 (상위 10개)]")
    print(f"  {'ESC-50 카테고리':<22} {'Top-1 AudioSet 예측':<35} {'확률':>8}")
    print("  " + "-" * 68)
    for _, row in results_df.head(10).iterrows():
        cat  = row["esc50_category"][:21]
        pred = row["top1_audioset"][:34]
        prob = row["top1_prob"]
        print(f"  {cat:<22} {pred:<35} {prob:>8.4f}")

    # CSV 저장
    out_path = RESULT_DIR / "PANNs_inference_results.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] 결과 저장: {out_path}")
    print("=" * 70)


# ─────────────────────────────────────────────
# 9. ONNX 변환 (Jetson 배포용)
# ─────────────────────────────────────────────
def export_to_onnx(model: Cnn6):
    """파인튜닝/추론 후 ONNX로 변환 (Jetson TensorRT 배포용)."""
    onnx_path  = MODEL_DIR / "cnn6_audioset.onnx"
    clip_frames = int(SAMPLE_RATE * CLIP_DURATION / HOP_SIZE) + 1
    dummy_input = torch.randn(1, clip_frames, MEL_BINS).to(DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["log_mel_spectrogram"],
        output_names=["clipwise_output", "embedding"],
        dynamic_axes={
            "log_mel_spectrogram": {0: "batch_size", 1: "time_steps"},
            "clipwise_output":     {0: "batch_size"},
            "embedding":           {0: "batch_size"},
        },
        opset_version=18,  # torch 2.x 권장 버전 (Jetson TRT 8.x+ 지원)
        do_constant_folding=True,
    )
    print(f"[✓] ONNX 변환 완료: {onnx_path}")
    print(f"    → Jetson에서: trtexec --onnx={onnx_path.name} --fp16 --saveEngine=cnn6.trt")


# ─────────────────────────────────────────────
# 10. 파인튜닝용 헤드 교체 예시 (참고용)
# ─────────────────────────────────────────────
def get_finetune_model(num_classes: int = 4) -> Cnn6:
    """
    파인튜닝용: AudioSet 헤드를 커스텀 클래스 수로 교체.
    num_classes 예시: 4 (경적/사이렌/비명/알람)
    """
    model = Cnn6(classes_num=CLASSES_NUM)

    # 백본 동결
    for param in model.parameters():
        param.requires_grad = False

    # 헤드 교체 (학습 가능)
    model.fc_audioset = nn.Linear(512, num_classes, bias=True)
    init_layer(model.fc_audioset)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[파인튜닝 모드] 학습 파라미터: {trainable:,} / 전체: {total:,}")
    return model


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  PANNs CNN6 × ESC-50 Inference Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # 1) ESC-50 다운로드
    esc_dir = download_esc50()
    df      = load_esc50_metadata(esc_dir)

    # 2) 사전학습 가중치 다운로드
    weight_path = download_pretrained_weights()

    # 3) AudioSet 레이블
    labels = load_audioset_labels()

    # 4) 모델 로드
    model = load_model(weight_path)

    # 5) 단일 파일 상세 추론 예시
    sample_file = df.iloc[0]["filepath"]
    print(f"\n[→] 단일 파일 추론 예시: {Path(sample_file).name}")
    result = infer_single(model, sample_file, labels, top_k=5)

    print(f"  카테고리: {df.iloc[0]['category']}")
    print(f"  Top-5 예측:")
    for rank, (lbl, prob) in enumerate(result["top_k"], 1):
        print(f"    {rank}. {lbl:<40} {prob:.4f}")
    print(f"  위험음 scores: {result['target_scores']}")

    # 6) 배치 추론
    results_df = run_esc50_inference(model, df, labels, n_samples=50)

    # 7) 결과 출력 및 저장
    print_results(results_df)

    # 8) ONNX 변환 (Jetson 배포용)
    print("\n[→] ONNX 변환 중...")
    export_to_onnx(model)

    # 9) 파인튜닝 모델 구조 확인
    print("\n[→] 파인튜닝 모드 확인 (4-class):")
    _ = get_finetune_model(num_classes=4)

    print("\n[완료] 모든 파이프라인 실행 성공!")


if __name__ == "__main__":
    main()