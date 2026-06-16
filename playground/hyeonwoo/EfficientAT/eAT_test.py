"""
EfficientAT (MobileNetV3) + ESC-50 Inference Pipeline
------------------------------------------------------
EfficientAT 공식 레포(fschmid56/EfficientAT)의 모델을 직접 임포트하여
mn10_as 사전학습 모델을 로드하고 ESC-50으로 추론합니다.
"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
from torch import autocast

# ─────────────────────────────────────────────
# 0. 설정 상수
# ─────────────────────────────────────────────
SAMPLE_RATE   = 32000
CLIP_DURATION = 10.0
WINDOW_SIZE   = 800
HOP_SIZE      = 320
MEL_BINS      = 128
FMIN          = 0
FMAX          = None
CLASSES_NUM   = 527

MODEL_NAME    = "mn10_as"
N_SAMPLES     = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR.parent / "data"
MODEL_DIR  = BASE_DIR.parent / "models"
RESULT_DIR = BASE_DIR / "results"

# EfficientAT 레포 경로 (torch hub 캐시)
REPO_DIR = Path.home() / ".cache" / "torch" / "hub" / "fschmid56_EfficientAT_main"

for d in [DATA_DIR, MODEL_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {
    "car_horn":  ["Car horn", "Vehicle horn, car horn, honking"],
    "siren":     ["Siren", "Civil defense siren", "Ambulance (siren)"],
    "alarm":     ["Alarm", "Fire alarm", "Smoke detector, smoke alarm", "Alarm clock"],
    "screaming": ["Screaming", "Shout"],
    "crash":     ["Vehicle collision, crash", "Crash"],
}

OUTPUT_COLUMNS = [
    "filename", "esc50_category", "top1_audioset", "top1_prob",
    "score_car_horn", "score_siren", "score_alarm", "score_screaming", "score_crash",
]


# ─────────────────────────────────────────────
# 1. ESC-50 다운로드
# ─────────────────────────────────────────────
def download_esc50():
    esc_dir = DATA_DIR / "ESC-50-master"
    if esc_dir.exists():
        print(f"[✓] ESC-50 이미 존재: {esc_dir}")
        return esc_dir

    urls = [
        "https://github.com/karoldvl/ESC-50/archive/master.zip",
        "https://zenodo.org/record/1203745/files/ESC-50-master.zip",
    ]
    zippath = DATA_DIR / "ESC-50-master.zip"

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
            break
        except Exception as e:
            print(f"    [!] 실패: {e}")

    print(f"[✓] ESC-50 준비 완료: {esc_dir}")
    return esc_dir


def load_esc50_metadata(esc_dir: Path) -> pd.DataFrame:
    meta_path = esc_dir / "meta" / "esc50.csv"
    df = pd.read_csv(meta_path)
    df["filepath"] = df["filename"].apply(lambda fn: str(esc_dir / "audio" / fn))
    print(f"[✓] ESC-50 메타데이터 로드: {len(df)}개 샘플, {df['category'].nunique()}개 클래스")
    return df


# ─────────────────────────────────────────────
# 2. EfficientAT 레포 경로를 sys.path에 추가
# ─────────────────────────────────────────────
def setup_efficientat_path():
    """REPO_DIR을 sys.path에 추가하고, 작업 디렉토리를 레포로 변경"""
    repo_str = str(REPO_DIR)
    if not REPO_DIR.exists():
        raise FileNotFoundError(
            f"EfficientAT 레포를 찾을 수 없습니다: {REPO_DIR}\n"
        )
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    
    # ★ 핵심 수정: helpers/utils.py가 'metadata/...' 상대경로를 쓰므로
    #   작업 디렉토리를 레포 루트로 변경해야 함
    os.chdir(REPO_DIR)
    
    print(f"[✓] EfficientAT 레포 경로 등록: {REPO_DIR}")
    print(f"[✓] 작업 디렉토리 변경: {REPO_DIR}")


# ─────────────────────────────────────────────
# 3. EfficientAT 모델 로드 (직접 임포트 방식)
# ─────────────────────────────────────────────
def load_efficientat_model(model_name: str = MODEL_NAME):
    setup_efficientat_path()

    # inference.py와 동일한 방식으로 직접 임포트
    from models.mn.model import get_model as get_mobilenet
    from models.preprocess import AugmentMelSTFT
    from helpers.utils import NAME_TO_WIDTH

    print(f"[↓] EfficientAT 모델 로드 중: {model_name}")

    # 모델 로드
    model = get_mobilenet(
        width_mult=NAME_TO_WIDTH(model_name),
        pretrained_name=model_name,
        strides=[2, 2, 2, 2],
        head_type="mlp",
    )
    model = model.to(DEVICE)
    model.eval()

    # 전처리기 (AugmentMelSTFT) — inference.py와 동일하게 사용
    mel_transform = AugmentMelSTFT(
        n_mels=MEL_BINS,
        sr=SAMPLE_RATE,
        win_length=WINDOW_SIZE,
        hopsize=HOP_SIZE,
    )
    mel_transform = mel_transform.to(DEVICE)
    mel_transform.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] {model_name} 로드 완료 — 파라미터: {total_params/1e6:.2f}M, device: {DEVICE}")
    return model, mel_transform


# ─────────────────────────────────────────────
# 4. AudioSet 레이블 로드
# ─────────────────────────────────────────────
def load_audioset_labels() -> list:
    """helpers.utils.labels를 우선 사용, 없으면 CSV 폴백"""
    # EfficientAT 레포의 내장 레이블 사용 (setup 후 호출)
    try:
        from helpers.utils import labels as _labels
        print(f"[✓] AudioSet 레이블 로드 (EfficientAT 내장): {len(_labels)}개")
        return list(_labels)
    except Exception:
        pass

    labels_path = MODEL_DIR / "class_labels_indices.csv"
    labels_url  = (
        "https://raw.githubusercontent.com/qiuqiangkong/"
        "audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
    )

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
            print(f"[!] 레이블 다운로드 실패: {e} → 더미 레이블 사용")
            dummy = [
                {"index": i, "mid": f"/m/{i:05d}", "display_name": f"AudioSet_class_{i}"}
                for i in range(CLASSES_NUM)
            ]
            pd.DataFrame(dummy).to_csv(labels_path, index=False)

    df = pd.read_csv(labels_path)
    labels = df["display_name"].tolist()
    print(f"[✓] AudioSet 레이블 로드 (CSV): {len(labels)}개")
    return labels


# ─────────────────────────────────────────────
# 5. 단일 파일 추론 (AugmentMelSTFT 사용)
# ─────────────────────────────────────────────
def infer_single(model, mel_transform, filepath: str, labels: list, top_k: int = 5) -> dict:
    # inference.py와 동일한 로드 방식
    waveform, _ = librosa.core.load(filepath, sr=SAMPLE_RATE, mono=True)

    # 길이 맞추기 (10초)
    target_len = int(SAMPLE_RATE * CLIP_DURATION)
    if len(waveform) < target_len:
        repeats = int(np.ceil(target_len / len(waveform)))
        waveform = np.tile(waveform, repeats)[:target_len]
    else:
        waveform = waveform[:target_len]

    waveform_tensor = torch.from_numpy(waveform[None, :]).to(DEVICE)

    # inference.py와 동일: autocast는 cuda일 때만
    ctx = autocast(device_type=DEVICE.type) if DEVICE.type == "cuda" else nullcontext()
    with torch.no_grad(), ctx:
        spec = mel_transform(waveform_tensor)          # (1, n_mels, T)
        preds, features = model(spec.unsqueeze(0))     # (1, 1, n_mels, T)

    probs = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_preds   = [(labels[i], float(probs[i])) for i in top_indices]

    target_scores = {}
    for cls_name, audioset_names in TARGET_CLASSES.items():
        score = 0.0
        for aname in audioset_names:
            if aname in labels:
                idx   = labels.index(aname)
                score = max(score, float(probs[idx]))
        target_scores[cls_name] = score

    return {
        "filepath":      filepath,
        "top_k":         top_preds,
        "target_scores": target_scores,
        "raw_probs":     probs,
    }


# ─────────────────────────────────────────────
# 6. ESC-50 배치 추론
# ─────────────────────────────────────────────
def run_esc50_inference(model, mel_transform, df: pd.DataFrame, labels: list,
                        n_samples: int = N_SAMPLES) -> pd.DataFrame:
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

    print(f"\n[→] {len(sample_df)}개 파일 추론 시작 (모델: {MODEL_NAME})...")
    records = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), ncols=70):
        try:
            result = infer_single(model, mel_transform, row["filepath"], labels, top_k=1)
            top1_label, top1_prob = result["top_k"][0]
            ts = result["target_scores"]

            records.append({
                "filename":        row["filename"],
                "esc50_category":  row["category"],
                "top1_audioset":   top1_label,
                "top1_prob":       round(top1_prob, 4),
                "score_car_horn":  round(ts.get("car_horn",  0.0), 4),
                "score_siren":     round(ts.get("siren",     0.0), 4),
                "score_alarm":     round(ts.get("alarm",     0.0), 4),
                "score_screaming": round(ts.get("screaming", 0.0), 4),
                "score_crash":     round(ts.get("crash",     0.0), 4),
            })
        except Exception as e:
            print(f"  [!] 오류 {row['filename']}: {e}")

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


# ─────────────────────────────────────────────
# 7. 결과 출력
# ─────────────────────────────────────────────
def print_results(results_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print(f"  EfficientAT ({MODEL_NAME}) — ESC-50 추론 결과")
    print("=" * 70)

    score_cols = [c for c in OUTPUT_COLUMNS if c.startswith("score_")]
    print("\n[위험음 클래스별 평균 confidence score]")
    print(f"  {'클래스':<15} {'평균 score':>12}")
    print("  " + "-" * 30)
    for col in score_cols:
        cls_name   = col.replace("score_", "")
        mean_score = results_df[col].mean()
        bar        = "█" * int(mean_score * 30)
        print(f"  {cls_name:<15} {mean_score:>8.4f}  {bar}")

    print(f"\n[샘플 추론 결과 (상위 10개)]")
    print(f"  {'ESC-50 카테고리':<22} {'Top-1 AudioSet 예측':<35} {'확률':>8}")
    print("  " + "-" * 68)
    for _, row in results_df.head(10).iterrows():
        cat  = row["esc50_category"][:21]
        pred = row["top1_audioset"][:34]
        prob = row["top1_prob"]
        print(f"  {cat:<22} {pred:<35} {prob:>8.4f}")

    out_path = RESULT_DIR / f"efficientat_{MODEL_NAME}_results.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[✓] 결과 저장: {out_path}")
    print("=" * 70)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  EfficientAT ({MODEL_NAME}) × ESC-50 Inference Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    esc_dir = download_esc50()
    df      = load_esc50_metadata(esc_dir)

    # 모델 로드 (레포 경로 등록 포함)
    model, mel_transform = load_efficientat_model(MODEL_NAME)

    # 레이블 로드 (레포 등록 후 내장 labels 사용 가능)
    labels = load_audioset_labels()

    # 단일 파일 예시
    sample_file = df.iloc[0]["filepath"]
    print(f"\n[→] 단일 파일 추론 예시: {Path(sample_file).name}")
    result = infer_single(model, mel_transform, sample_file, labels, top_k=5)
    print(f"  카테고리: {df.iloc[0]['category']}")
    print(f"  Top-5 예측:")
    for rank, (lbl, prob) in enumerate(result["top_k"], 1):
        print(f"    {rank}. {lbl:<40} {prob:.4f}")
    print(f"  위험음 scores: {result['target_scores']}")

    # 배치 추론
    results_df = run_esc50_inference(model, mel_transform, df, labels, n_samples=N_SAMPLES)

    print_results(results_df)
    print("\n[완료] EfficientAT inference 파이프라인 실행 성공!")


if __name__ == "__main__":
    main()