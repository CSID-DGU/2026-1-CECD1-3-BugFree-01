# Jetson Nano Audio Model Comparison

## Test Environment

- Device: Jetson Nano
- Docker image: nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
- Microphone: reSpeaker 4 Mic Array
- ALSA device example: plughw:2,0
- Sample rate: 16000 Hz
- Channels: 6
- Selected channel: 0

## Result Table

| Model | Runtime | Input seconds | Top-k | Threshold | Avg infer ms | Model size | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| EfficientAT mn04_as | PyTorch | 2 | 5 | 0.10 | TBD | TBD | TBD |
| EfficientAT mn05_as | PyTorch | 2 | 5 | 0.10 | TBD | TBD | TBD |
| EfficientAT mn10_as | PyTorch | 2 | 5 | 0.10 | TBD | TBD | TBD |
| YAMNet TFLite | TFLite | 2 | 5 | 0.10 | TBD | TBD | TBD |

## Evaluation Criteria

- Accuracy / Top-k relevance
- Inference time
- Model size
- False positives in quiet environment
- Jetson CPU/RAM usage
