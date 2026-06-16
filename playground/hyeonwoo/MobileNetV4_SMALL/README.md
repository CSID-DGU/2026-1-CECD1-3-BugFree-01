# MobileNetV4-small ESC-50 평가

## 목적

이 폴더는 ESC-50 환경음 데이터셋을 MobileNetV4-small 기반 이미지 분류 모델로 평가하고, 예측 CSV와 metric/시각화 결과를 저장하기 위한 최소 파이프라인이다. 기존 `playground/hyeonwoo/YAMNET`의 평가 결과 저장 방식과 metric dashboard, per-category metric, confusion matrix, top-k 분석 흐름을 참고했다.

## 방식

ESC-50 wav 파일을 waveform 그대로 모델에 넣지 않는다. 각 5초 clip을 16 kHz mono로 맞춘 뒤 log-mel spectrogram으로 변환하고, dB scale 변환 및 정규화를 거쳐 `224x224` 이미지로 resize한다. MobileNetV4-small은 ImageNet pretrained CNN backbone이므로, 1채널 log-mel 이미지를 3채널로 repeat하고 ImageNet mean/std로 정규화해서 입력한다.

기본 모델은 timm의 `mobilenetv4_conv_small.e2400_r224_in1k`이다.

```python
timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=True, num_classes=50)
```

ImageNet pretrained weight는 backbone 초기값으로 사용되고, classifier head는 ESC-50의 50개 클래스에 맞게 생성된다. 별도 학습 checkpoint 없이 실행하면 classifier head가 학습되지 않은 random baseline이므로 성능이 낮을 수 있다.

## 설치

```bash
pip install -r requirements.txt
```

PyTorch/torchaudio는 CUDA, Jetson, CPU 환경별 설치 방법이 다를 수 있다. 현재 환경에 맞는 wheel을 먼저 설치한 뒤 위 requirements를 설치하는 것을 권장한다.

## 실행

### Fine-tuning

ESC-50 fold 1-4로 학습하고 fold 5로 validation을 수행한다. 가장 좋은 validation accuracy를 낸 모델은 기본적으로 `checkpoints/esc-50_tuning_model.pth`에 저장된다.

```bash
python train.py --esc50-root ../ESC-50-master --epochs 20 --batch-size 16 --device auto
```

validation fold를 바꾸려면 다음처럼 실행한다.

```bash
python train.py --esc50-root ../ESC-50-master --val-fold 1 --epochs 20
```

학습이 끝난 checkpoint를 평가하려면 다음 명령을 사용한다.

```bash
python run_eval.py --esc50-root ../ESC-50-master --checkpoint checkpoints/esc-50_tuning_model.pth
```

이미 저장된 CSV/JSON/PNG 결과만으로 HTML 리포트를 다시 만들려면 다음을 실행한다.

```bash
python build_report.py
```

### Evaluation

이 폴더에서 실행하는 경우:

```bash
python run_eval.py --esc50-root ../ESC-50-master --batch-size 32 --device auto
```

프로젝트 루트에서 실행하는 경우:

```bash
python playground/hyeonwoo/MobileNetV4_SMALL/run_eval.py --esc50-root playground/hyeonwoo/ESC-50-master --batch-size 32 --device auto
```

특정 fold만 평가:

```bash
python run_eval.py --esc50-root ../ESC-50-master --fold 5
```

학습된 checkpoint를 로드해서 평가:

```bash
python run_eval.py --esc50-root ../ESC-50-master --checkpoint checkpoints/best_model.pth
```

지원 옵션:

- `--esc50-root`: ESC-50 루트 폴더. 기본값은 이 스크립트 기준 `../ESC-50-master`
- `--metadata-csv`: 별도 metadata CSV 경로. 기본값은 `<esc50-root>/meta/esc50.csv`
- `--fold`: 특정 fold만 평가. 생략하면 전체 fold 평가
- `--checkpoint`: 학습된 PyTorch checkpoint 경로
- `--batch-size`: 평가 batch size
- `--device`: `auto`, `cpu`, `cuda`, `cuda:0` 등
- `--output-dir`: 결과 저장 폴더. 기본값은 `results`

학습 옵션:

- `--val-fold`: validation으로 사용할 ESC-50 fold. 기본값은 `5`
- `--epochs`: fine-tuning epoch 수. 기본값은 `20`
- `--batch-size`: 학습 batch size. 기본값은 `16`
- `--lr`: learning rate. 기본값은 `1e-4`
- `--weight-decay`: AdamW weight decay. 기본값은 `1e-4`
- `--output-checkpoint`: checkpoint 저장 경로. 기본값은 `checkpoints/esc-50_tuning_model.pth`
- `--no-pretrained`: ImageNet pretrained weight 없이 학습
- `--no-amp`: CUDA mixed precision 비활성화

## 결과 파일

결과는 기본적으로 `playground/hyeonwoo/MobileNetV4_SMALL/results/`에 저장된다.

- `results/predictions.csv`: 샘플별 true label, top-1~top-5 label/index/softmax score, inference time
- `results/overall_metrics.json`: top-1 accuracy, macro precision/recall/F1, weighted F1, MRR, Hit@1~Hit@5
- `results/overall_metrics.csv`: 주요 overall metric 표
- `results/per_category_metrics.csv`: category별 support, precision, recall, F1, MRR, average top-1 score, Hit@1~Hit@5
- `results/confusion_matrix.csv`: row-normalized confusion matrix
- `results/top_confusions.csv`: 가장 많이 틀린 true -> predicted category 조합
- `results/mobilenetv4_analysis_report.html`: 주요 metric, 결과 이미지, 표를 한 번에 보는 HTML 리포트
- `results/plots/mobilenetv4_metrics_dashboard.png`
- `results/plots/mobilenetv4_per_category_metrics_heatmap.png`
- `results/plots/mobilenetv4_per_category_f1.png`
- `results/plots/mobilenetv4_confusion_matrix_normalized.png`
- `results/plots/mobilenetv4_top_confusions.png`
- `results/plots/mobilenetv4_confidence_distribution.png`

## 한계와 다음 단계

checkpoint 없이 실행하면 MobileNetV4-small의 feature extractor는 ImageNet pretrained지만, ESC-50 classifier head는 random initialization이다. 따라서 이 결과는 학습된 sound classifier 성능이 아니라 log-mel image transfer learning 구조의 평가용 baseline으로 해석해야 한다.

실제 성능 평가를 하려면 `train.py`로 ESC-50 train fold를 fine-tuning하고, 저장된 checkpoint를 `--checkpoint`로 넘겨 평가하면 된다.
