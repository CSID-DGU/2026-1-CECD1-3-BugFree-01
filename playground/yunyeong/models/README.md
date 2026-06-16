# 소리 모델별 Colab 노트북 실행 가이드 (gpt로 작성)

생성된 노트북 5개:

1. `YAMNet_danger_sound_finetune_colab.ipynb`
2. `PANNs_CNN14_danger_sound_finetune_colab.ipynb`
3. `AST_danger_sound_finetune_colab.ipynb`
4. `BEATs_danger_sound_finetune_colab.ipynb`
5. `EfficientAT_danger_sound_finetune_colab.ipynb`

## 데이터셋 연결 방식

각 노트북은 다음 데이터셋을 모두 지원하도록 작성했습니다.

- ESC-50: 자동 다운로드
- UrbanSound8K: `USE_KAGGLE_URBANSOUND8K=True` + Kaggle API 토큰 필요
- FSD50K: Google Drive에 받은 뒤 `FSD50K_ROOT` 수정
- AudioSet: wav로 준비한 audiofolder를 `AUDIOSET_ROOT`에 연결
- AI Hub 소음 환경 음성인식 데이터: AI Hub에서 직접 받은 뒤 `AIHUB_NOISE_ROOT` 수정

## 위험 소리 타깃 라벨

- `car_horn`
- `siren`
- `glass_breaking`
- `explosion_or_gunshot`
- `construction_or_machine`
- `fire_alarm`
- `baby_cry`
- `doorbell_or_bell`
- `other`

위험 클래스 성능을 높이기 위해 클래스 가중치, weighted sampler, data augmentation, `other` 샘플 수 제한을 넣었습니다.

## 추천 실행 순서

1. `EfficientAT` 또는 `YAMNet`부터 실행해 빠르게 baseline을 잡습니다.
2. `PANNs`를 실행해 CNN 계열 성능을 비교합니다.
3. `AST`, `BEATs`는 GPU 메모리와 checkpoint 준비 상태를 확인한 뒤 실행합니다.
4. 최종 후보는 macro F1, 위험 클래스 recall, latency, model size를 함께 비교하세요.

## 주의

BEATs checkpoint는 Microsoft 공식 README의 OneDrive 링크에서 자동 다운로드를 먼저 시도하도록 했습니다. 실패하면 직접 다운로드해 `/content/BEATs_iter3_plus_AS2M.pt`에 두세요. Hugging Face fallback은 Colab 실행 편의를 위한 옵션이며, 외부 pickle checkpoint를 로드하므로 신뢰 가능한 환경에서만 사용하세요.

## GitHub 업로드용 정리 사항

이 패키지의 `.ipynb` 파일은 GitHub 렌더링을 위해 다음 항목을 제거했습니다.

- Colab 전용 top-level metadata (`colab`, `accelerator` 등)
- 셀별 metadata
- 실행 결과 outputs
- execution count
- widget/runtime/cache 관련 필드

남긴 metadata는 Jupyter/GitHub가 인식하는 최소 kernel 정보(`kernelspec`, `language_info`)뿐입니다.

대용량 데이터셋과 학습 결과물은 GitHub에 올리지 말고 Google Drive 또는 별도 스토리지에 보관하세요. `.gitignore`에 기본 제외 규칙을 넣어두었습니다.

## 모델 실행 결과 비교

각 노트북의 마지막 `classification_report` 기준으로 비교하면 `EfficientAT`이 가장 안정적입니다. `YAMNet`도 성능은 비슷하지만 평가 단위가 window 기준으로 더 많아 직접 비교 시 참고용으로 보는 편이 좋습니다.

| 모델 | Accuracy | Macro recall | Macro F1 | 평가 샘플 | 간단 평가 | 모델 크기 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `EfficientAT` | 0.635 | 0.93 | 0.692 | 156 | 위험 클래스 recall이 전반적으로 높아 최종 후보로 가장 적합 | 약 19.5 MB |
| `YAMNet` | 0.656 | 0.81 | 0.672 | 1,560 | 가볍고 준수하지만 window 기준 결과라 보조 후보 | 14.2 MB |
| `AST` | 0.494 | 0.33 | 0.258 | 156 | 일부 클래스는 맞추지만 전체 균형이 낮음 | 346 MB |
| `BEATs` | 0.045 | 0.10 | 0.030 | 156 | 현재 설정에서는 학습이 거의 잡히지 않음 | 약 360 MB |
| `PANNs_CNN14` | 0.038 | 0.11 | 0.008 | 156 | 한 클래스 쏠림이 커서 재학습/설정 수정 필요 | 327.4 MB |
|`MobileNetV4`| 0.808 | 0.867 | 0.784 | 156 | - | 9.75 MB |

현재 결과만 보면 엣지 탑재 후보는 `EfficientAT`을 우선으로 두고, 빠른 baseline 또는 백업 후보로 `YAMNet`을 유지하는 방향이 좋습니다.
