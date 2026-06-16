# 청각장애인을 위한 소리 분류 키링 및 어플리케이션

이 저장소는 키링 장치에서 수집한 짧은 소리 구간을 분류하고, 앱으로 위험/생활 소리 알림을 전달하기 위한 실험 프로젝트이다. 현재 모델 실험은 EfficientAT를 기반으로 하며, 최종 학습용 데이터셋은 `datasets/soundkey_balanced_v2`에 정리되어 있다.

## 현재 상태 요약

- 기반 모델: EfficientAT `mn10_as`
- 입력 단위: 2초 audio segment
- 샘플링 레이트: 32 kHz
- 무음/저볼륨 제외 기준: `-45 dBFS`
- 현재 클래스 수: 13개
- 현재 데이터셋 split: file-level split
- 현재 noise augmentation: white noise + `background_other` mix
- 현재 checkpoint: `EfficientAT/runs/soundkey_balanced_v2_head/best.pt`

주의: 현재 checkpoint는 file-level split 적용 전에 학습된 모델이다. `manifest.csv`는 file-level split으로 갱신되었으므로, 이후 성능 비교는 새 split 기준으로 재훈련한 결과를 기준으로 해야 한다.

## 현재 클래스

| index | label |
|---:|---|
| 0 | `baby_cry` |
| 1 | `bicycle` |
| 2 | `car_horn` |
| 3 | `cat_meow` |
| 4 | `dog_bark` |
| 5 | `door_knock` |
| 6 | `fire_alarm` |
| 7 | `glass_breaking` |
| 8 | `gunshot` |
| 9 | `scream` |
| 10 | `siren` |
| 11 | `water_sound` |
| 12 | `background_other` |

## 현재 데이터셋

최종 데이터셋 위치:

```text
datasets/soundkey_balanced_v2/
```

구성 파일:

| 파일/폴더 | 설명 |
|---|---|
| `audio/` | 학습에 사용되는 오디오 파일 복사본 |
| `manifest.csv` | 학습/검증/테스트 split, 라벨, 원본 정보, 2초 segment 위치 |
| `label_map.json` | class index와 label 매핑 |
| `build_report.json` | 데이터셋 생성 조건과 통계 |

생성 기준:

- `edge_audio_dataset`을 기본 데이터로 사용했다.
- `soundkey_dataset`은 보강 데이터로 사용했다.
- 기존 데이터와 제목 stem이 겹치는 soundkey 파일은 제외했다.
- `bicycle`, `scream`은 새 클래스로 추가했다.
- `crying` 일부는 `baby_cry` 보강에 사용했다.
- `appliance_sound`는 `background_other`로만 사용했다.
- `construction_noise`는 사용하지 않았다.
- soundkey의 `gunshot`은 추가하지 않았다. 기존 edge 쪽 `gunshot`이 많아서 cap만 적용했다.
- 각 오디오는 2초 단위로 검사하고, `-45 dBFS`보다 작은 segment는 제외했다.
- split은 `source_dataset + source_folder + source_file` 기준 file-level split이다.

현재 전체 segment 수:

```text
total 6837
train 4788
val   1042
test  1007
```

클래스별 segment 수:

| label | count |
|---|---:|
| `baby_cry` | 650 |
| `bicycle` | 650 |
| `car_horn` | 69 |
| `cat_meow` | 650 |
| `dog_bark` | 650 |
| `door_knock` | 57 |
| `fire_alarm` | 211 |
| `glass_breaking` | 650 |
| `gunshot` | 650 |
| `scream` | 650 |
| `siren` | 650 |
| `water_sound` | 650 |
| `background_other` | 650 |

`car_horn`, `door_knock`, `fire_alarm`은 아직 데이터가 부족하다. 특히 `car_horn`, `door_knock`은 평가 support가 작기 때문에 성능 수치의 신뢰도가 낮다.

## 현재 폴더/파일 구조

```text
.
├─ README.md
├─ .gitignore
├─ datasets/
│  └─ soundkey_balanced_v2/
│     ├─ audio/
│     ├─ manifest.csv
│     ├─ label_map.json
│     └─ build_report.json
├─ EfficientAT/
│  ├─ .venv/
│  ├─ datasets/
│  │  ├─ soundkey_balanced.py
│  │  ├─ audioset.py
│  │  ├─ dcase20.py
│  │  ├─ esc50.py
│  │  ├─ fsd50k.py
│  │  ├─ openmic.py
│  │  └─ helpers/
│  ├─ helpers/
│  ├─ images/
│  ├─ metadata/
│  ├─ models/
│  ├─ resources/
│  ├─ runs/
│  │  └─ soundkey_balanced_v2_head/
│  │     ├─ best.pt
│  │     ├─ best_metrics.json
│  │     ├─ last.pt
│  │     └─ last_metrics.json
│  ├─ ex_soundkey_balanced.py
│  ├─ inference.py
│  ├─ windowed_inference.py
│  └─ EfficientAT 원본 예제 파일들
└─ tools/
   ├─ build_soundkey_balanced_dataset.py
   ├─ build_uiux_final_ppt.py
   ├─ build_uiux_ppt.py
   ├─ build_uiux_report_docx.py
   ├─ create_soundkey_assets.py
   ├─ export_app_screen_captures.py
   └─ 기타 문서/압축 보조 스크립트
```

`EfficientAT/`는 별도 git clone된 EfficientAT 저장소이다. 원본 clone 파일은 보존하고, 프로젝트 전용으로 추가한 핵심 파일은 다음 두 개다.

- `EfficientAT/ex_soundkey_balanced.py`
- `EfficientAT/datasets/soundkey_balanced.py`

## 주요 스크립트

### 데이터셋 생성

```powershell
.\EfficientAT\.venv\Scripts\python.exe .\tools\build_soundkey_balanced_dataset.py --out-dir .\datasets\soundkey_balanced_v2 --segment-cap 650 --seed 42
```

주의: 현재 workspace에는 원본 `edge_audio_dataset`과 원본 `soundkey_dataset` 폴더가 남아 있지 않다. 위 명령으로 데이터셋을 처음부터 재생성하려면 원본 데이터셋을 다시 배치해야 한다.

기본 입력 경로:

```text
EfficientAT/data/edge_audio_dataset/manifest.csv
datasets/soundkey_dataset/
```

### 학습

현재 학습 entrypoint:

```text
EfficientAT/ex_soundkey_balanced.py
```

기존 checkpoint 없이 새로 head fine-tuning을 시작하는 예:

```powershell
cd .\EfficientAT
.\.venv\Scripts\python.exe ex_soundkey_balanced.py --pretrained --model_name mn10_as --freeze_backbone --n_epochs 5 --batch_size 16 --num_workers 0 --lr 0.001 --no_wavmix --no_roll --gain_augment 6 --class_weighting sqrt_inverse --output_dir runs\soundkey_balanced_v2_head
```

noise augmentation이 포함된 재훈련 예:

```powershell
cd .\EfficientAT
.\.venv\Scripts\python.exe ex_soundkey_balanced.py --pretrained --model_name mn10_as --freeze_backbone --n_epochs 10 --batch_size 16 --num_workers 0 --lr 0.001 --gain_augment 6 --class_weighting sqrt_inverse --white_noise_prob 0.35 --background_noise_prob 0.35 --noise_snr_min_db 5 --noise_snr_max_db 25 --output_dir runs\soundkey_balanced_v2_noise
```

마지막 block 또는 전체 backbone fine-tuning은 아직 별도 옵션으로 세분화하지 않았다. 현재는 `--freeze_backbone`을 빼면 전체 모델이 학습된다.

### 평가

validation split 평가:

```powershell
cd .\EfficientAT
.\.venv\Scripts\python.exe ex_soundkey_balanced.py --model_name mn10_as --eval_only --eval_split val --batch_size 16 --num_workers 0 --checkpoint runs\soundkey_balanced_v2_head\best.pt
```

test split 평가:

```powershell
cd .\EfficientAT
.\.venv\Scripts\python.exe ex_soundkey_balanced.py --model_name mn10_as --eval_only --eval_split test --batch_size 16 --num_workers 0 --checkpoint runs\soundkey_balanced_v2_head\best.pt
```

## 현재 checkpoint 성능

현재 checkpoint:

```text
EfficientAT/runs/soundkey_balanced_v2_head/best.pt
```

이 checkpoint는 5 epoch head fine-tuning 결과다.

검증 성능:

```text
accuracy    0.8636
macro_f1    0.8537
weighted_f1 0.8656
```

주의: 이 수치는 file-level split 적용 전 학습/평가 결과다. 현재 manifest는 file-level split으로 갱신되었으므로, 공식 비교에는 새 split 기준 재훈련 결과를 사용해야 한다.

## Noise Augmentation

현재 `EfficientAT/datasets/soundkey_balanced.py`에 학습용 noise augmentation이 추가되어 있다.

적용 방식:

- white noise를 확률적으로 추가한다.
- `background_other` 샘플을 배경음으로 섞는다.
- SNR 범위를 지정할 수 있다.

기본 옵션:

```text
white_noise_prob      0.35
background_noise_prob 0.35
noise_snr_min_db      5.0
noise_snr_max_db      25.0
```

평가 데이터에는 noise augmentation을 적용하지 않는다. 학습 데이터에만 적용된다.

## 훈련 전략 검토

현재 추천 우선순위:

1. file-level split 기준으로 재훈련
2. noise augmentation 포함
3. `car_horn`, `door_knock`에 약한 oversampling 또는 WeightedRandomSampler 적용
4. `--freeze_backbone`을 제거하고 낮은 learning rate로 전체 fine-tuning 실험
5. 마지막 단계에서 LoRA 실험 검토

LoRA는 가능하지만 현재 1순위는 아니다. EfficientAT의 현재 모델은 Transformer가 아니라 MobileNetV3 기반 CNN이다. LoRA는 Linear/Attention layer 중심 모델에서 이점이 크고, CNN에도 적용 가능하지만 구현 복잡도 대비 효과가 불확실하다. 현재 병목은 데이터 부족, file-level 평가, 노이즈 환경 일반화이므로 먼저 이 부분을 해결하는 것이 더 효과적이다.

## 데이터 보강 방향

우선 보강 대상:

- `car_horn`
- `door_knock`
- `fire_alarm`
- `background_other`
- `siren`

권장 방식:

- 실제 키링 마이크 또는 유사 마이크로 2초 단위 수집
- 조용한 실내, 복도, 도로, 카페, 가방/옷 마찰음 등 비경고 소리를 `background_other`에 추가
- `car_horn`, `door_knock`은 단순 복제보다 실제 원본 다양성 확보가 우선
- oversampling은 데이터 폴더를 복제하지 말고 train sampler에서 처리