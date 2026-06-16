# YAMNet ESC-50 Evaluation

## 목적

이미 학습된 TensorFlow Hub YAMNet 모델로 ESC-50 데이터셋의 wav 파일을 평가하고 결과를 저장한다. 학습, TFLite 변환, 벤치마크 코드는 포함하지 않는다.

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python evaluate_esc50_yamnet.py
```

또는 ESC-50 경로와 top-k 개수를 직접 지정한다.

```bash
python evaluate_esc50_yamnet.py --esc50_dir ../ESC-50-master --topk 5
```

기본 ESC-50 경로는 `../ESC-50-master`이다.

예측 결과를 precision, recall, F1 score, confusion matrix 등으로 시각화하려면 다음을 실행한다.

```bash
python analyze_yamnet_predictions.py
```

## 결과 파일

결과는 `results/` 폴더에 저장된다.

- `results/yamnet_raw_predictions.csv`: 각 wav 파일의 ESC-50 label과 YAMNet top-k 예측 label/score
- `results/yamnet_eval_summary.json`: 전체 파일 수, 매핑된 파일 수, 정답 수, accuracy, 매핑된 category 목록
- `results/yamnet_model_info.json`: 모델 경로, 모델 크기, class 수, parameter 수, 평균/중앙값/min/max inference time
- `results/yamnet_analysis_report.html`: 전체 지표, category별 지표, confusion matrix 차트를 한 번에 보는 HTML 리포트
- `results/yamnet_metrics_dashboard.png`: 주요 지표와 top-k hit rate 요약 차트
- `results/yamnet_per_category_metrics_heatmap.png`: category별 precision, recall, F1 score, hit@k heatmap
- `results/yamnet_per_category_f1.png`: category별 F1 score bar chart
- `results/yamnet_confusion_matrix_counts.png`: confusion matrix count 이미지
- `results/yamnet_confusion_matrix_normalized.png`: 정규화 confusion matrix 이미지
- `results/yamnet_top_confusions.png`: 가장 많이 틀린 true/predicted category 조합
- `results/yamnet_confidence_distribution.png`: top-1 정답/오답별 score 분포

## ESC-50 / YAMNet Label Mapping

ESC-50 category와 YAMNet의 AudioSet label은 완전히 일치하지 않는다. 정확한 성능평가를 하려면 `evaluate_esc50_yamnet.py` 상단의 `ESC50_TO_YAMNET_LABELS` 딕셔너리를 직접 수정해야 한다.

현재 평가는 매핑이 있는 ESC-50 category에 대해서만 top-1 accuracy를 계산한다. 먼저 `results/yamnet_raw_predictions.csv`를 확인한 뒤, ESC-50 category별로 정답으로 인정할 YAMNet label을 추가하는 방식으로 사용한다.
