import os
import sys

# EfficientAT 내부의 상대 경로를 올바르게 인식하도록 작업 디렉토리 변경
if os.path.exists('./EfficientAT'):
    os.chdir('./EfficientAT')
sys.path.append('.')

import torch
import librosa
import numpy as np

# EfficientAT 모듈 불러오기
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT


def test_efficientat():
    print("1. 디바이스 설정 및 모델 로드 중...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 가벼운 모델 로드
    model = get_mobilenet(width_mult=1.0, pretrained_name='mn10_as')
    model.to(device)
    model.eval()

    print("2. 오디오 전처리 모듈(Mel Spectrogram) 로드 중...")
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320)
    mel.to(device)
    mel.eval()

    print("3. 테스트용 가상 오디오 데이터 생성 중...")
    sample_rate = 32000
    waveform = np.zeros(sample_rate * 10)  # 10초짜리 빈 소리 데이터
    waveform = torch.from_numpy(waveform[None, :]).float().to(device)

    print("4. 추론(Inference) 실행!")
    with torch.no_grad():
        spec = mel(waveform)

        # 🔥 [에러 해결 포인트] 3D 텐서를 4D 텐서로 변환 (채널 차원 추가)
        # 예: [1, 128, 1001] -> [1, 1, 128, 1001]
        spec = spec.unsqueeze(1)

        preds, _ = model(spec)

    print("====================================")
    print("🎉 테스트 완료!")
    print("출력 텐서 형태(Shape):", preds.shape)
    print("====================================")
    print("(참고: AudioSet의 클래스 개수가 527개이므로, 형태가 [1, 527]이 나오면 정상입니다.)")


if __name__ == '__main__':
    test_efficientat()